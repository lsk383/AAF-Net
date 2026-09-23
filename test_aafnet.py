



import sys
import os
import random
import datetime
from pathlib import Path

import torch
import numpy as np
import cv2


# Resolve common image-extension mismatches when loading dataset files.

_original_cv2_imread = cv2.imread

RAW_VIS_ENABLE = False
RAW_VIS_DIR = "./test_results/raw_input_vis"
RAW_VIS_MAX_PER_FOLDER = 20
_RAW_VIS_COUNT = {"A": 0, "B": 0, "label": 0}


def _to_uint8_for_vis(img):
    arr = img.copy()

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)

    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val) * 255.0
    else:
        arr = np.zeros_like(arr)

    return np.clip(arr, 0, 255).astype(np.uint8)


def _detect_folder_name(path):
    parts = Path(path).parts
    if "A" in parts:
        return "A"
    if "B" in parts:
        return "B"
    if "label" in parts:
        return "label"
    return None


def _save_raw_visualization(path, img):
    if not RAW_VIS_ENABLE:
        return

    folder_name = _detect_folder_name(path)
    if folder_name is None:
        return

    if _RAW_VIS_COUNT[folder_name] >= RAW_VIS_MAX_PER_FOLDER:
        return

    save_dir = os.path.join(RAW_VIS_DIR, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    img_vis = _to_uint8_for_vis(img)
    save_name = Path(path).stem + ".png"
    save_path = os.path.join(save_dir, save_name)

    cv2.imwrite(save_path, img_vis)
    _RAW_VIS_COUNT[folder_name] += 1

    print(f"[Raw Vis] saved {folder_name}: {save_path}, shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
def _resolve_image_path(path):
    if path is None:
        return path

    if os.path.exists(path):
        return path

    p = Path(path)

    candidates = []

    # 处理 xxx.tif.png -> xxx.tif
    if p.suffix.lower() in [".png", ".jpg", ".jpeg"] and p.stem.lower().endswith((".tif", ".tiff")):
        candidates.append(str(p.with_name(p.stem)))

    # 处理任意后缀，按 stem 搜索常见格式
    stem = p.stem
    if stem.lower().endswith((".tif", ".tiff")):
        stem = Path(stem).stem

    for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
        candidates.append(str(p.with_name(stem + ext)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return path


def _safe_cv2_imread(path, flags=cv2.IMREAD_COLOR):
    fixed_path = _resolve_image_path(path)
    img = _original_cv2_imread(fixed_path, flags)

    if img is None and fixed_path != path:
        img = _original_cv2_imread(path, flags)
        final_path = path
    else:
        final_path = fixed_path

    if img is not None:
        _save_raw_visualization(final_path, img)

    return img


cv2.imread = _safe_cv2_imread

# Load checkpoints on CPU when CUDA is unavailable.

_original_torch_load = torch.load


def _safe_torch_load(*args, **kwargs):
    if not torch.cuda.is_available():
        kwargs["map_location"] = "cpu"
    return _original_torch_load(*args, **kwargs)


torch.load = _safe_torch_load

# ==============================================================================

import torch.backends.cudnn as cudnn
import torch.distributed as dist
from argparse import ArgumentParser

import dataset_GVLM as myDataLoader
import Transforms as myTransforms
from utils import *

from models.model_with_tafm import ModelToVisualize as BaseNet_LWGANet_L2

sys.path.insert(0, "tools")


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--inWidth", type=int, default=256)
    parser.add_argument("--inHeight", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--onGPU",
        default=True,
        type=lambda x: str(x).lower() == "true",
        help="Run on CPU or GPU.",
    )

    parser.add_argument(
    "--weight",
    required=True,
    type=str,
    help="Path to the trained model weight.",
    )

    parser.add_argument(
    "--file_root",
    required=True,
    type=str,
    help="Root directory of the dataset.",
    )

    parser.add_argument(
        "--dataset_name",
        default="GVLM-CD",
        type=str,
        help="Dataset name used by the data loader.",
    )

    parser.add_argument(
        "--save_raw_vis",
        default=False,
        type=lambda x: str(x).lower() == "true",
        help="Save raw input images before Normalize/Scale/ToTensor.",
    )

    parser.add_argument(
        "--raw_vis_num",
        default=20,
        type=int,
        help="Number of raw images to save for each folder: A, B, label.",
    )

    parser.add_argument(
        "--raw_vis_dir",
        default="./test_results/raw_input_vis",
        type=str,
        help="Directory for saving raw input visualizations.",
    )

    return parser.parse_args()


def revert_sync_batchnorm(module):
    res = module

    if isinstance(module, torch.nn.SyncBatchNorm):
        res = torch.nn.BatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )

        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()

        res.running_mean = module.running_mean
        res.running_var = module.running_var
        res.num_batches_tracked = module.num_batches_tracked

    for name, child in module.named_children():
        res.add_module(name, revert_sync_batchnorm(child))

    return res


def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def init_distributed_if_needed():
    if dist.is_available() and not dist.is_initialized():
        port = find_free_port()
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=1,
        )


def load_weights(model, weight_path):
    if not os.path.isfile(weight_path):
        print(f"错误: 找不到权重文件: {weight_path}")
        sys.exit(1)

    print(f"=> Loading weights from: {weight_path}")
    checkpoint = torch.load(weight_path)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 兼容 DataParallel / DistributedDataParallel 的 module. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print("权重加载成功！")
    if len(missing) > 0:
        print(f"Missing keys: {len(missing)}")
    if len(unexpected) > 0:
        print(f"Unexpected keys: {len(unexpected)}")


def testSegmentation(args):
    seed = 2333
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    init_distributed_if_needed()

    if not os.path.exists(args.file_root):
        raise FileNotFoundError(f"数据集路径不存在: {args.file_root}")

    print(f"=> 使用测试数据集路径: {args.file_root}")
    print(f"=> dataset_name: {args.dataset_name}")

    model = BaseNet_LWGANet_L2(pretrained=False)

    if not torch.cuda.is_available():
        model = revert_sync_batchnorm(model)
        args.onGPU = False
        print("已将 SyncBatchNorm 自动降级为 BatchNorm2d，以兼容 CPU")

    load_weights(model, args.weight)

    args.savedir = "./test_results/"
    args.vis_dir = os.path.join(args.savedir, "Vis/")
    os.makedirs(args.vis_dir, exist_ok=True)

    if args.onGPU and torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    total_params = sum(np.prod(p.size()) for p in model.parameters())
    print("Total network parameters:", total_params)

    # 与训练时保持一致
    mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]

    val_transform = myTransforms.Compose(
        [
            myTransforms.Normalize(mean=mean, std=std),
            myTransforms.Scale(args.inWidth, args.inHeight),
            myTransforms.ToTensor(),
        ]
    )

    test_data = myDataLoader.Dataset(
        "test",
        file_root=args.file_root,
        transform=val_transform,
        dataset_name=args.dataset_name,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    print(f"Starting evaluation on Test set... Total batches: {len(test_loader)}")

    model.eval()
    start_time = datetime.datetime.now()

    with torch.no_grad():
        loss_test, score_test = val(args, test_loader, model, 0)

    end_time = datetime.datetime.now()

    print("\n=========================================================================")
    print("测试结果评估完成 (Test Results)")
    print("=========================================================================")
    print(f"IoU:       {score_test['IoU']:.4f}")
    print(f"F1-Score:  {score_test['F1']:.4f}")
    print(f"Recall:    {score_test['recall']:.4f}")
    print(f"Precision: {score_test['precision']:.4f}")
    print("=========================================================================")
    print(f"Total Inference Time: {end_time - start_time}")


if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        args.onGPU = False

    RAW_VIS_ENABLE = args.save_raw_vis
    RAW_VIS_DIR = args.raw_vis_dir
    RAW_VIS_MAX_PER_FOLDER = args.raw_vis_num

    print("Called with args:")
    print(args)

    testSegmentation(args)
#
