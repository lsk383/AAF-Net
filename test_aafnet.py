# #
# import sys
# import torch
# import os
# import numpy as np
# import datetime
# import random
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
#
# _original_torch_load = torch.load
#
# def _safe_torch_load(*args, **kwargs):
#     if not torch.cuda.is_available():
#         kwargs['map_location'] = 'cpu'
#     return _original_torch_load(*args, **kwargs)
#
# torch.load = _safe_torch_load
# # ==============================================================================
#
# import torch.backends.cudnn as cudnn
# # 【关键修复 1】：将 dataset 统一改回与你训练脚本一致的 dataset_GVLM
# import dataset_GVLM as myDataLoader
# import Transforms as myTransforms
# from utils import *
# import os
# import numpy as np
# from argparse import ArgumentParser
# from thop import profile
# import torch.distributed as dist
#
#
# from models.model_with_tafm import ModelToVisualize as BaseNet_LWGANet_L2
# import random
#
# sys.path.insert(0, 'tools')
#
#
# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
#     parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
#     parser.add_argument('--num_workers', type=int, default=0, help='No. of parallel threads')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
#     parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
#                         help='Run on CPU or GPU. If TRUE, then GPU.')
#
#     # 默认指定为模型权重路径
#     parser.add_argument('--weight',
#                         default='/home/LWGANet/results/A2Net_LWGANet/L2/dataset_final_split/pretrain=True/26.02.14-1059_batch32_iter_40000_lr_0.0005/best_model.pth',
#                         type=str, help='Pretrained weight path')
#
#     # 默认指定为数据集路径
#     parser.add_argument('--file_root', default="/home/LWGANet/dataset_final_split",
#                         help='Data directory')
#     args = parser.parse_args()
#     return args
#
#
# def testSegmentation(args):
#     torch.backends.cudnn.benchmark = True
#     SEED = 2333
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed(SEED)
#
#     if not dist.is_initialized():
#         port = random.randint(20000, 30000)
#         dist_backend = 'nccl' if torch.cuda.is_available() else 'gloo'
#         dist.init_process_group(backend=dist_backend, init_method=f'tcp://127.0.0.1:{port}', rank=0, world_size=1)
#
#     # 初始化模型
#     model = BaseNet_LWGANet_L2(pretrained=False)
#
#
#     if not torch.cuda.is_available():
#         def revert_sync_batchnorm(module):
#             res = module
#             if isinstance(module, torch.nn.SyncBatchNorm):
#                 res = torch.nn.BatchNorm2d(
#                     module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats
#                 )
#                 if module.affine:
#                     res.weight.data = module.weight.data.clone().detach()
#                     res.bias.data = module.bias.data.clone().detach()
#                 res.running_mean = module.running_mean
#                 res.running_var = module.running_var
#                 res.num_batches_tracked = module.num_batches_tracked
#             for name, child in module.named_children():
#                 res.add_module(name, revert_sync_batchnorm(child))
#             return res
#
#         model = revert_sync_batchnorm(model)
#         print("已将模型中的 SyncBatchNorm 自动降级为 BatchNorm2d 以兼容 CPU")
#     # ==============================================================================
#
#     # 加载测试权重
#     if not os.path.isfile(args.weight):
#         print(f" 错误: 找不到权重文件 {args.weight}。请检查路径是否正确。")
#         sys.exit(1)
#
#     print(f"=> Loading weights from: {args.weight}")
#     checkpoint = torch.load(args.weight)
#
#
#     if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['state_dict'], strict=False)
#     else:
#         model.load_state_dict(checkpoint, strict=False)
#     print(" 权重加载成功 (已过滤无关的统计参数)！")
#
#     # 【关键修复 2】：强制指定 dataset_name='GVLM'，与你的训练脚本2保持绝对一致
#     dataset_name = 'dataset_B'
#     if os.path.exists(args.file_root):
#         print(f"=> 使用测试数据集路径: {args.file_root}, dataset_name: {dataset_name}")
#     else:
#         raise TypeError(f"数据集路径 {args.file_root} 不存在，请检查！")
#
#
#     args.savedir = './test_results/'
#     args.vis_dir = os.path.join(args.savedir, 'Vis/')
#     if not os.path.exists(args.vis_dir):
#         os.makedirs(args.vis_dir)
#
#     if args.onGPU and torch.cuda.is_available():
#         model = model.cuda()
#
#     total_params = sum([np.prod(p.size()) for p in model.parameters()])
#     print('Total network parameters: ' + str(total_params))
#
#     # 【关键修复 3】：替换为你训练该权重时真实使用的 mean 和 std
#     mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
#
#     # 测试集只需要做归一化和缩放，不需要数据增强
#     valDataset = myTransforms.Compose([
#         myTransforms.Normalize(mean=mean, std=std),
#         myTransforms.Scale(args.inWidth, args.inHeight),
#         myTransforms.ToTensor()
#     ])
#
#     test_data = myDataLoader.Dataset("test", file_root=args.file_root,
#                                      transform=valDataset, dataset_name=dataset_name)
#     testLoader = torch.utils.data.DataLoader(
#         test_data, shuffle=False,
#         batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)
#
#     print(f'Starting evaluation on Test set... Total batches: {len(testLoader)}')
#
#     if args.onGPU and torch.cuda.is_available():
#         cudnn.benchmark = True
#
#     # 锁定 BatchNorm 和 Dropout
#     model.eval()
#
#     start_time = datetime.datetime.now()
#
#     # 进行评估 (调用 utils.py 中的 val 函数)
#     with torch.no_grad():
#         loss_test, score_test = val(args, testLoader, model, 0)
#
#     end_time = datetime.datetime.now()
#
#     print("\n=========================================================================")
#     print(" 测试结果评估完成 (Test Results)")
#     print("=========================================================================")
#     print(f"IoU:      {score_test['IoU']:.4f}")
#     print(f"F1-Score:  {score_test['F1']:.4f}")
#     print(f"Recall:    {score_test['recall']:.4f}")
#     print(f"Precision: {score_test['precision']:.4f}")
#     print("=========================================================================")
#     print(f"Total Inference Time: {end_time - start_time}")
#
#
# if __name__ == '__main__':
#     args = parse_args()
#
#
#     if not torch.cuda.is_available():
#         args.onGPU = False
#
#     print('Called with args:')
#     print(args)
#     testSegmentation(args)





#批量可视化热力图



# import sys
# import torch
# import os
# import numpy as np
# import random
# import matplotlib
#
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
# _original_torch_load = torch.load
#
#
# def _safe_torch_load(*args, **kwargs):
#     if not torch.cuda.is_available():
#         kwargs["map_location"] = "cpu"
#     return _original_torch_load(*args, **kwargs)
#
#
# torch.load = _safe_torch_load
#
# import torch.backends.cudnn as cudnn
# import dataset_GVLM as myDataLoader
# import Transforms as myTransforms
# from argparse import ArgumentParser
# import torch.distributed as dist
#
# try:
#     from tqdm import tqdm
# except ImportError:
#     tqdm = None
#
# from models.model_with_tafm import ModelToVisualize as BaseNet_LWGANet_L2
#
# sys.path.insert(0, "tools")
#
#
# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument("--inWidth", type=int, default=256, help="Width of RGB image")
#     parser.add_argument("--inHeight", type=int, default=256, help="Height of RGB image")
#     parser.add_argument("--num_workers", type=int, default=0, help="No. of parallel threads")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
#     parser.add_argument("--onGPU", default=True, type=lambda x: (str(x).lower() == "true"))
#
#     parser.add_argument(
#         "--weight",
#         default="/home/LWGANet/results/A2Net_LWGANet/L2_DSBN_GVLM_Composite/GVLM-CD-Processed/25.08.14-20.24/best_model.pth",
#         type=str,
#         help="Pretrained weight path",
#     )
#
#     parser.add_argument(
#         "--file_root",
#         default="/home/LWGANet/GVLM-CD-Processed",
#         help="Data directory",
#     )
#
#     parser.add_argument(
#         "--save_root",
#         default="./test_results/HAGD_Heatmaps_Fixed",
#         type=str,
#         help="Directory to save HAGD heatmaps",
#     )
#
#     parser.add_argument(
#         "--threshold",
#         default=0.5,
#         type=float,
#         help="Threshold for binary prediction",
#     )
#
#     return parser.parse_args()
#
#
# def revert_sync_batchnorm(module):
#     res = module
#     if isinstance(module, torch.nn.SyncBatchNorm):
#         res = torch.nn.BatchNorm2d(
#             module.num_features,
#             module.eps,
#             module.momentum,
#             module.affine,
#             module.track_running_stats,
#         )
#         if module.affine:
#             res.weight.data = module.weight.data.clone().detach()
#             res.bias.data = module.bias.data.clone().detach()
#         res.running_mean = module.running_mean
#         res.running_var = module.running_var
#         res.num_batches_tracked = module.num_batches_tracked
#
#     for name, child in module.named_children():
#         res.add_module(name, revert_sync_batchnorm(child))
#
#     return res
#
#
# def denorm_rgb(x):
#     """
#     x: [3, H, W], normalized tensor.
#     """
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#
#     x = x.detach().cpu().float()
#     x = x * std + mean
#     x = torch.clamp(x, 0, 1)
#     return x.permute(1, 2, 0).numpy()
#
#
# def safe_filename(name, idx=None):
#     """
#     确保输出文件名一定有图片后缀。
#     解决 ValueError: unknown file extension.
#     """
#     if name is None:
#         name = ""
#
#     name = str(name).strip()
#     name = os.path.basename(name)
#
#     if name == "" or name.lower() in ["none", "nan"]:
#         if idx is None:
#             name = "unknown.png"
#         else:
#             name = f"{idx:04d}.png"
#
#     base, ext = os.path.splitext(name)
#
#     if base == "":
#         if idx is None:
#             base = "unknown"
#         else:
#             base = f"{idx:04d}"
#
#     if ext == "":
#         ext = ".png"
#
#     # matplotlib/PIL 常见安全后缀
#     allowed_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
#     if ext.lower() not in allowed_exts:
#         ext = ".png"
#
#     return base + ext
#
#
# def save_gray(path, arr):
#     path = ensure_path_has_ext(path)
#     plt.imsave(path, arr, cmap="gray", vmin=0, vmax=1)
#
#
# def save_heatmap(path, arr):
#     path = ensure_path_has_ext(path)
#     plt.imsave(path, arr, cmap="jet", vmin=0, vmax=1)
#
#
# def save_rgb(path, arr):
#     path = ensure_path_has_ext(path)
#     plt.imsave(path, np.clip(arr, 0, 1))
#
#
# def ensure_path_has_ext(path):
#     """
#     双保险：即使 filename 处理漏了，这里也补 .png。
#     """
#     root, ext = os.path.splitext(path)
#     if ext == "":
#         path = path + ".png"
#     return path
#
#
# def make_overlay(post_rgb, pred_bin, alpha=0.45):
#     overlay = post_rgb.copy()
#     red = np.zeros_like(post_rgb)
#     red[..., 0] = 1.0
#     overlay[pred_bin == 1] = (1 - alpha) * overlay[pred_bin == 1] + alpha * red[pred_bin == 1]
#     return np.clip(overlay, 0, 1)
#
#
# def save_panel(path, pre_rgb, post_rgb, gt, m5, m4, m3, m2, pred_bin, overlay):
#     path = ensure_path_has_ext(path)
#
#     titles = [
#         "Pre",
#         "Post",
#         "GT",
#         "m5 coarse",
#         "m4 guided",
#         "m3 refined",
#         "m2 final",
#         "Pred",
#         "Overlay",
#     ]
#
#     images = [
#         pre_rgb,
#         post_rgb,
#         gt,
#         m5,
#         m4,
#         m3,
#         m2,
#         pred_bin,
#         overlay,
#     ]
#
#     plt.figure(figsize=(22, 4))
#
#     for i, (title, img) in enumerate(zip(titles, images)):
#         plt.subplot(1, len(images), i + 1)
#
#         if title in ["m5 coarse", "m4 guided", "m3 refined", "m2 final"]:
#             plt.imshow(img, cmap="jet", vmin=0, vmax=1)
#         elif img.ndim == 2:
#             plt.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             plt.imshow(img)
#
#         plt.title(title, fontsize=10)
#         plt.axis("off")
#
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def build_output_dirs(save_root):
#     dirs = {
#         "m5": os.path.join(save_root, "m5_coarse"),
#         "m4": os.path.join(save_root, "m4_guided"),
#         "m3": os.path.join(save_root, "m3_refined"),
#         "m2": os.path.join(save_root, "m2_final"),
#         "pred": os.path.join(save_root, "pred_binary"),
#         "overlay": os.path.join(save_root, "overlay"),
#         "panel": os.path.join(save_root, "panel"),
#     }
#
#     for d in dirs.values():
#         os.makedirs(d, exist_ok=True)
#
#     return dirs
#
#
# def _basename_from_item(item):
#     """
#     尽量从 dataset 内部记录中提取原始文件名。
#     支持:
#     - str path
#     - tuple/list
#     - dict
#     - int/string id
#     """
#     if isinstance(item, str):
#         return os.path.basename(item)
#
#     if isinstance(item, (int, np.integer)):
#         return f"{int(item)}.png"
#
#     if isinstance(item, (float, np.floating)):
#         return f"{int(item)}.png"
#
#     if isinstance(item, (tuple, list)) and len(item) > 0:
#         # 优先找带图片后缀的路径
#         for x in item:
#             if isinstance(x, str) and x.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
#                 return os.path.basename(x)
#
#         # 找任意字符串
#         for x in item:
#             if isinstance(x, str) and x.strip() != "":
#                 return os.path.basename(x)
#
#         return str(item[0])
#
#     if isinstance(item, dict):
#         keys = [
#             "name",
#             "filename",
#             "file_name",
#             "id",
#             "A",
#             "B",
#             "img",
#             "image",
#             "path",
#             "A_path",
#             "img_path",
#             "image_path",
#         ]
#         for k in keys:
#             if k in item:
#                 return os.path.basename(str(item[k]))
#
#     return str(item)
#
#
# def get_filenames_from_dataset_or_fallback(dataset, file_root):
#     """
#     优先从 dataset 内部读取真实文件名顺序。
#     如果 dataset 内部字段没有后缀，会自动补 .png。
#     """
#
#     print("=> 尝试从 dataset 内部读取真实文件名顺序...")
#
#     # 把最可能是路径的字段放前面；ids 放后面，避免优先拿到纯 id
#     candidate_attrs = [
#         "A_paths",
#         "B_paths",
#         "img_paths",
#         "image_paths",
#         "file_paths",
#         "path_list",
#         "file_list",
#         "files",
#         "imgs",
#         "images",
#         "image_list",
#         "img_list",
#         "data_list",
#         "test_files",
#         "file_names",
#         "filenames",
#         "names",
#         "name_list",
#         "ids",
#     ]
#
#     for attr in candidate_attrs:
#         if not hasattr(dataset, attr):
#             continue
#
#         value = getattr(dataset, attr)
#
#         if isinstance(value, list) and len(value) == len(dataset):
#             raw_names = []
#             ok = True
#
#             for idx, item in enumerate(value):
#                 name = _basename_from_item(item)
#                 if name is None:
#                     ok = False
#                     break
#                 raw_names.append(safe_filename(name, idx))
#
#             if ok:
#                 print(f"=> 文件名已从 dataset.{attr} 读取，数量: {len(raw_names)}")
#                 print(f"=> 前 10 个文件名: {raw_names[:10]}")
#                 return raw_names
#
#     print("=> 未从常见 dataset 字段中找到文件名列表。")
#     print("=> 打印 dataset 内部 list 字段，方便排查：")
#
#     for k, v in dataset.__dict__.items():
#         if isinstance(v, list):
#             print(f"   {k}: list, len={len(v)}")
#             if len(v) > 0:
#                 print(f"      first item: {v[0]}")
#
#     print("=> 退回使用 sorted(test/A) 文件名。")
#
#     img_dir = os.path.join(file_root, "test", "A")
#     if not os.path.isdir(img_dir):
#         raise FileNotFoundError(f"找不到测试影像目录: {img_dir}")
#
#     names = sorted(
#         [
#             f
#             for f in os.listdir(img_dir)
#             if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
#         ]
#     )
#
#     names = [safe_filename(n, idx) for idx, n in enumerate(names)]
#
#     print(f"=> sorted(test/A) 文件数量: {len(names)}")
#     print(f"=> 前 10 个文件名: {names[:10]}")
#
#     return names
#
#
# def visualize_hagd_all_test(args):
#     torch.backends.cudnn.benchmark = True
#
#     SEED = 2333
#     torch.manual_seed(SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(SEED)
#
#     if not dist.is_initialized():
#         port = random.randint(20000, 30000)
#         dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
#         dist.init_process_group(
#             backend=dist_backend,
#             init_method=f"tcp://127.0.0.1:{port}",
#             rank=0,
#             world_size=1,
#         )
#
#     model = BaseNet_LWGANet_L2(pretrained=False)
#
#     if not torch.cuda.is_available():
#         model = revert_sync_batchnorm(model)
#         print("已将模型中的 SyncBatchNorm 自动降级为 BatchNorm2d 以兼容 CPU")
#
#     if not os.path.isfile(args.weight):
#         print(f"错误: 找不到权重文件 {args.weight}。请检查路径是否正确。")
#         sys.exit(1)
#
#     print(f"=> Loading weights from: {args.weight}")
#     checkpoint = torch.load(args.weight)
#
#     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         model.load_state_dict(checkpoint["state_dict"], strict=False)
#     else:
#         model.load_state_dict(checkpoint, strict=False)
#
#     print("权重加载成功！")
#
#     dataset_name = "GVLM"
#     if os.path.exists(args.file_root):
#         print(f"=> 使用测试数据集路径: {args.file_root}, dataset_name: {dataset_name}")
#     else:
#         raise TypeError(f"数据集路径 {args.file_root} 不存在，请检查！")
#
#     if args.onGPU and torch.cuda.is_available():
#         model = model.cuda()
#         cudnn.benchmark = True
#
#     model.eval()
#
#     total_params = sum([np.prod(p.size()) for p in model.parameters()])
#     print("Total network parameters: " + str(total_params))
#
#     # 严格沿用你的测试代码 mean/std
#     mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
#
#     valDataset = myTransforms.Compose(
#         [
#             myTransforms.Normalize(mean=mean, std=std),
#             myTransforms.Scale(args.inWidth, args.inHeight),
#             myTransforms.ToTensor(),
#         ]
#     )
#
#     # 严格沿用你的测试代码读取方式
#     test_data = myDataLoader.Dataset(
#         "test",
#         file_root=args.file_root,
#         transform=valDataset,
#         dataset_name=dataset_name,
#     )
#
#     testLoader = torch.utils.data.DataLoader(
#         test_data,
#         shuffle=False,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=False,
#     )
#
#     names = get_filenames_from_dataset_or_fallback(test_data, args.file_root)
#
#     print(f"=> test_data 数量: {len(test_data)}")
#     print(f"=> 文件名数量: {len(names)}")
#
#     if len(names) != len(test_data):
#         print("警告: 文件名数量与 dataset 长度不一致。")
#         print("将按当前顺序尽量对应，超出部分使用 index 命名。")
#
#     dirs = build_output_dirs(args.save_root)
#
#     print(f"=> 开始输出 HAGD 热力图，可视化结果保存到: {args.save_root}")
#
#     global_idx = 0
#
#     iterator = enumerate(testLoader)
#     if tqdm is not None:
#         iterator = tqdm(iterator, total=len(testLoader), desc="Generating HAGD heatmaps")
#
#     with torch.no_grad():
#         for batch_idx, batched_inputs in iterator:
#             img, target = batched_inputs
#
#             # ==========================================================
#             # 模型推理严格沿用你的原测试代码，不改！
#             # ==========================================================
#             pre_img = img[:, 0:3]
#             post_img = img[:, 3:6]
#
#             if args.onGPU and torch.cuda.is_available():
#                 pre_img = pre_img.cuda()
#                 post_img = post_img.cuda()
#
#             outputs = model(pre_img, post_img)
#
#             # 模型输出顺序: [m2, m3, m4, m5]
#             m2, m3, m4, m5 = outputs
#
#             m2 = m2.detach().cpu().numpy()[:, 0]
#             m3 = m3.detach().cpu().numpy()[:, 0]
#             m4 = m4.detach().cpu().numpy()[:, 0]
#             m5 = m5.detach().cpu().numpy()[:, 0]
#
#             img_cpu = img.detach().cpu()
#             target_cpu = target.detach().cpu()
#
#             batch_size = img_cpu.shape[0]
#
#             for i in range(batch_size):
#                 if global_idx < len(names):
#                     filename = safe_filename(names[global_idx], global_idx)
#                 else:
#                     filename = f"{global_idx:04d}.png"
#
#                 # ==========================================================
#                 # 只修显示，不改推理。
#                 # 你发现可视化里灾前/灾后反了，所以显示时对调：
#                 # 0:3 显示为 Post
#                 # 3:6 显示为 Pre
#                 # ==========================================================
#                 post_rgb = denorm_rgb(img_cpu[i, 0:3])
#                 pre_rgb = denorm_rgb(img_cpu[i, 3:6])
#
#                 if target_cpu[i].ndim == 3:
#                     gt = target_cpu[i].squeeze(0).numpy()
#                 else:
#                     gt = target_cpu[i].numpy()
#                 gt = (gt > 0).astype(np.uint8)
#
#                 m5_i = m5[i]
#                 m4_i = m4[i]
#                 m3_i = m3[i]
#                 m2_i = m2[i]
#
#                 pred_bin = (m2_i > args.threshold).astype(np.uint8)
#                 overlay = make_overlay(post_rgb, pred_bin)
#
#                 save_heatmap(os.path.join(dirs["m5"], filename), m5_i)
#                 save_heatmap(os.path.join(dirs["m4"], filename), m4_i)
#                 save_heatmap(os.path.join(dirs["m3"], filename), m3_i)
#                 save_heatmap(os.path.join(dirs["m2"], filename), m2_i)
#
#                 save_gray(os.path.join(dirs["pred"], filename), pred_bin)
#                 save_rgb(os.path.join(dirs["overlay"], filename), overlay)
#
#                 save_panel(
#                     os.path.join(dirs["panel"], filename),
#                     pre_rgb,
#                     post_rgb,
#                     gt,
#                     m5_i,
#                     m4_i,
#                     m3_i,
#                     m2_i,
#                     pred_bin,
#                     overlay,
#                 )
#
#                 global_idx += 1
#
#             if tqdm is not None:
#                 iterator.set_postfix({"saved": global_idx})
#             else:
#                 print(f"[{batch_idx + 1}/{len(testLoader)}] 已输出 {global_idx} 张")
#
#     print("=========================================================================")
#     print("HAGD 分层掩膜热力图全部输出完成")
#     print("=========================================================================")
#     print(f"m5 粗尺度响应: {dirs['m5']}")
#     print(f"m4 引导响应:   {dirs['m4']}")
#     print(f"m3 精化响应:   {dirs['m3']}")
#     print(f"m2 最终响应:   {dirs['m2']}")
#     print(f"最终二值预测:  {dirs['pred']}")
#     print(f"预测叠加图:    {dirs['overlay']}")
#     print(f"总览拼图:      {dirs['panel']}")
#     print("=========================================================================")
#
#
# if __name__ == "__main__":
#     args = parse_args()
#
#     if not torch.cuda.is_available():
#         args.onGPU = False
#
#     print("Called with args:")
#     print(args)
#
#     visualize_hagd_all_test(args)




# ## TAFM模块debug
# import sys
# import torch
# import os
# import numpy as np
# import random
# import matplotlib
# 
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# 
# _original_torch_load = torch.load
# 
# 
# def _safe_torch_load(*args, **kwargs):
#     if not torch.cuda.is_available():
#         kwargs["map_location"] = "cpu"
#     return _original_torch_load(*args, **kwargs)
# 
# 
# torch.load = _safe_torch_load
# 
# import torch.backends.cudnn as cudnn
# import dataset_GVLM as myDataLoader
# import Transforms as myTransforms
# from argparse import ArgumentParser
# import torch.distributed as dist
# 
# try:
#     from tqdm import tqdm
# except ImportError:
#     tqdm = None
# 
# from models.model_with_tafm_debug import ModelToVisualize as BaseNet_LWGANet_L2
# 
# sys.path.insert(0, "tools")
# 
# 
# def parse_args():
#     parser = ArgumentParser()
# 
#     parser.add_argument("--inWidth", type=int, default=256)
#     parser.add_argument("--inHeight", type=int, default=256)
#     parser.add_argument("--num_workers", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--onGPU", default=True, type=lambda x: (str(x).lower() == "true"))
# 
#     parser.add_argument(
#         "--weight",
#         default="/home/LWGANet/results/A2Net_LWGANet/L2_DSBN_GVLM_Composite/GVLM-CD-Processed/25.08.14-20.24/best_model.pth",
#         type=str,
#         help="Pretrained weight path",
#     )
# 
#     parser.add_argument(
#         "--file_root",
#         default="/home/LWGANet/GVLM-CD-Processed",
#         help="Data directory",
#     )
# 
#     parser.add_argument(
#         "--save_root",
#         default="./test_results/TAFM_Debug_AllStages",
#         type=str,
#         help="Directory to save TAFM debug visualizations",
#     )
# 
#     parser.add_argument(
#         "--max_save",
#         default=0,
#         type=int,
#         help="Max number of test samples to visualize. Set <=0 to save all.",
#     )
# 
#     parser.add_argument(
#         "--topk",
#         default=10,
#         type=int,
#         help="Top-k high/low weight channels",
#     )
# 
#     parser.add_argument(
#         "--threshold",
#         default=0.5,
#         type=float,
#         help="Threshold for binary prediction",
#     )
# 
#     # 你之前发现显示里灾前/灾后反了，所以默认只修显示，不改推理
#     parser.add_argument(
#         "--swap_display",
#         default=True,
#         type=lambda x: (str(x).lower() == "true"),
#         help="Swap T1/T2 for visualization only. Model input order is unchanged.",
#     )
# 
#     return parser.parse_args()
# 
# 
# def revert_sync_batchnorm(module):
#     res = module
#     if isinstance(module, torch.nn.SyncBatchNorm):
#         res = torch.nn.BatchNorm2d(
#             module.num_features,
#             module.eps,
#             module.momentum,
#             module.affine,
#             module.track_running_stats,
#         )
#         if module.affine:
#             res.weight.data = module.weight.data.clone().detach()
#             res.bias.data = module.bias.data.clone().detach()
#         res.running_mean = module.running_mean
#         res.running_var = module.running_var
#         res.num_batches_tracked = module.num_batches_tracked
# 
#     for name, child in module.named_children():
#         res.add_module(name, revert_sync_batchnorm(child))
# 
#     return res
# 
# 
# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)
# 
# 
# def safe_filename(name, idx):
#     name = str(name).strip()
#     name = os.path.basename(name)
# 
#     if name == "" or name.lower() in ["none", "nan"]:
#         name = f"{idx:04d}.png"
# 
#     base, ext = os.path.splitext(name)
# 
#     if base == "":
#         base = f"{idx:04d}"
# 
#     if ext == "":
#         ext = ".png"
# 
#     if ext.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
#         ext = ".png"
# 
#     return base + ext
# 
# 
# def _basename_from_item(item):
#     if isinstance(item, str):
#         return os.path.basename(item)
# 
#     if isinstance(item, (int, np.integer)):
#         return f"{int(item)}.png"
# 
#     if isinstance(item, (tuple, list)) and len(item) > 0:
#         for x in item:
#             if isinstance(x, str) and x.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
#                 return os.path.basename(x)
#         for x in item:
#             if isinstance(x, str) and x.strip() != "":
#                 return os.path.basename(x)
#         return str(item[0])
# 
#     if isinstance(item, dict):
#         keys = [
#             "name",
#             "filename",
#             "file_name",
#             "id",
#             "A",
#             "B",
#             "img",
#             "image",
#             "path",
#             "A_path",
#             "img_path",
#             "image_path",
#         ]
#         for k in keys:
#             if k in item:
#                 return os.path.basename(str(item[k]))
# 
#     return str(item)
# 
# 
# def get_filenames_from_dataset_or_fallback(dataset, file_root):
#     print("=> 尝试从 dataset 内部读取真实文件名顺序...")
# 
#     candidate_attrs = [
#         "A_paths",
#         "B_paths",
#         "img_paths",
#         "image_paths",
#         "file_paths",
#         "path_list",
#         "file_list",
#         "files",
#         "imgs",
#         "images",
#         "image_list",
#         "img_list",
#         "data_list",
#         "test_files",
#         "file_names",
#         "filenames",
#         "names",
#         "name_list",
#         "ids",
#     ]
# 
#     for attr in candidate_attrs:
#         if not hasattr(dataset, attr):
#             continue
# 
#         value = getattr(dataset, attr)
# 
#         if isinstance(value, list) and len(value) == len(dataset):
#             names = []
#             ok = True
# 
#             for idx, item in enumerate(value):
#                 name = _basename_from_item(item)
#                 if name is None:
#                     ok = False
#                     break
#                 names.append(safe_filename(name, idx))
# 
#             if ok:
#                 print(f"=> 文件名已从 dataset.{attr} 读取，数量: {len(names)}")
#                 print(f"=> 前 10 个文件名: {names[:10]}")
#                 return names
# 
#     print("=> 未从 dataset 内部找到可靠文件名，退回 sorted(test/A)。")
# 
#     img_dir = os.path.join(file_root, "test", "A")
# 
#     if not os.path.isdir(img_dir):
#         print(f"警告: 找不到 {img_dir}，将使用 index 命名。")
#         return [f"{i:04d}.png" for i in range(len(dataset))]
# 
#     names = sorted(
#         [
#             f
#             for f in os.listdir(img_dir)
#             if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
#         ]
#     )
# 
#     if len(names) != len(dataset):
#         print("警告: test/A 文件数量与 dataset 长度不一致，使用 index 命名。")
#         return [f"{i:04d}.png" for i in range(len(dataset))]
# 
#     names = [safe_filename(n, idx) for idx, n in enumerate(names)]
# 
#     print(f"=> sorted(test/A) 文件数量: {len(names)}")
#     print(f"=> 前 10 个文件名: {names[:10]}")
# 
#     return names
# 
# 
# def denorm_rgb(x):
#     """
#     x: [3,H,W], normalized tensor.
#     """
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# 
#     x = x.detach().cpu().float()
#     x = x * std + mean
#     x = torch.clamp(x, 0, 1)
#     return x.permute(1, 2, 0).numpy()
# 
# 
# def normalize01(x):
#     x = x.astype(np.float32)
#     return (x - x.min()) / (x.max() - x.min() + 1e-8)
# 
# 
# def feature_to_heatmap(feat, out_size):
#     """
#     feat: torch.Tensor [C,h,w]
#     output: numpy [H,W]
#     """
#     feat_map = torch.mean(torch.abs(feat), dim=0, keepdim=True).unsqueeze(0)
#     feat_map = torch.nn.functional.interpolate(
#         feat_map,
#         size=out_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     feat_map = feat_map[0, 0].detach().cpu().numpy()
#     return normalize01(feat_map)
# 
# 
# def group_response_map(x_fused_i, att_np, out_size, topk=10, high=True):
#     """
#     x_fused_i: torch.Tensor [C,h,w]
#     att_np: numpy [C]
#     """
#     idx_sorted = np.argsort(-att_np)
# 
#     if high:
#         idx = idx_sorted[:topk]
#     else:
#         idx = idx_sorted[-topk:]
# 
#     feat = torch.mean(torch.abs(x_fused_i[idx]), dim=0, keepdim=True).unsqueeze(0)
#     feat = torch.nn.functional.interpolate(
#         feat,
#         size=out_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     feat = feat[0, 0].detach().cpu().numpy()
#     return normalize01(feat)
# 
# 
# def make_overlay(post_rgb, pred_bin, alpha=0.45):
#     overlay = post_rgb.copy()
#     red = np.zeros_like(post_rgb)
#     red[..., 0] = 1.0
#     overlay[pred_bin == 1] = (1 - alpha) * overlay[pred_bin == 1] + alpha * red[pred_bin == 1]
#     return np.clip(overlay, 0, 1)
# 
# 
# def save_weight_curve(ax, att_np, title="Channel weights"):
#     channels = np.arange(1, len(att_np) + 1)
#     ax.plot(channels, att_np, linewidth=1.5)
#     ax.set_xlim(1, len(att_np))
#     ax.set_ylim(0, 1)
#     ax.set_title(title, fontsize=9)
#     ax.set_xlabel("Channel", fontsize=8)
#     ax.set_ylabel("Weight", fontsize=8)
#     ax.tick_params(labelsize=7)
#     ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
# 
# 
# def save_basic_panel(path, pre_rgb, post_rgb, gt, before_map, after_map, pred_bin, overlay, att_np, stage_name):
#     """
#     T1 | T2 | Label | before | after | Pred | Overlay | weight curve
#     """
#     plt.figure(figsize=(22, 4))
# 
#     titles = [
#         "T1",
#         "T2",
#         "Label",
#         f"{stage_name} before",
#         f"{stage_name} after",
#         "Pred",
#         "Overlay",
#     ]
# 
#     images = [
#         pre_rgb,
#         post_rgb,
#         gt,
#         before_map,
#         after_map,
#         pred_bin,
#         overlay,
#     ]
# 
#     for i, (title, img) in enumerate(zip(titles, images)):
#         ax = plt.subplot(1, 8, i + 1)
# 
#         if "before" in title or "after" in title:
#             ax.imshow(img, cmap="jet", vmin=0, vmax=1)
#         elif img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
# 
#         ax.set_title(title, fontsize=9)
#         ax.axis("off")
# 
#     ax_curve = plt.subplot(1, 8, 8)
#     save_weight_curve(ax_curve, att_np, title=f"{stage_name} weights")
# 
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
# 
# 
# def save_high_low_panel(path, pre_rgb, post_rgb, gt, high_map, low_map, pred_bin, stage_name):
#     plt.figure(figsize=(16, 4))
# 
#     titles = [
#         "T1",
#         "T2",
#         "Label",
#         f"{stage_name} high-weight",
#         f"{stage_name} low-weight",
#         "Pred",
#     ]
# 
#     images = [
#         pre_rgb,
#         post_rgb,
#         gt,
#         high_map,
#         low_map,
#         pred_bin,
#     ]
# 
#     for i, (title, img) in enumerate(zip(titles, images)):
#         ax = plt.subplot(1, 6, i + 1)
# 
#         if "high-weight" in title or "low-weight" in title:
#             ax.imshow(img, cmap="jet", vmin=0, vmax=1)
#         elif img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
# 
#         ax.set_title(title, fontsize=9)
#         ax.axis("off")
# 
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
# 
# 
# def save_weight_rank_bar(path, att_np, stage_name, topk=10):
#     idx_sorted = np.argsort(-att_np)
#     top_idx = idx_sorted[:topk]
#     bottom_idx = idx_sorted[-topk:]
# 
#     all_idx = np.concatenate([top_idx, bottom_idx])
#     labels = [f"Ch{c + 1}" for c in all_idx]
#     values = att_np[all_idx]
# 
#     plt.figure(figsize=(10, 4))
#     plt.bar(np.arange(len(values)), values)
#     plt.xticks(np.arange(len(values)), labels, rotation=45, fontsize=8)
#     plt.ylim(0, 1)
#     plt.ylabel("Attention weight")
#     plt.title(f"{stage_name}: Top-{topk} and Bottom-{topk} channel weights")
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
# 
# 
# def save_weights_csv(path, att_np):
#     channel_ids = np.arange(1, len(att_np) + 1)
#     table = np.stack([channel_ids, att_np], axis=1)
#     np.savetxt(
#         path,
#         table,
#         delimiter=",",
#         header="channel,attention_weight",
#         comments="",
#         fmt=["%d", "%.6f"],
#     )
# 
# 
# def calc_region_ratio(x_fused_i, att_np, gt, topk=10):
#     """
#     计算 high/low 通道在滑坡区和背景区的响应比。
#     """
#     idx_sorted = np.argsort(-att_np)
#     high_idx = idx_sorted[:topk]
#     low_idx = idx_sorted[-topk:]
# 
#     gt_t = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
# 
#     Hf, Wf = x_fused_i.shape[1], x_fused_i.shape[2]
#     gt_small = torch.nn.functional.interpolate(gt_t, size=(Hf, Wf), mode="nearest")[0, 0]
# 
#     fg = gt_small > 0.5
#     bg = gt_small <= 0.5
# 
#     feat_abs = torch.abs(x_fused_i)
# 
#     high_resp = torch.mean(feat_abs[high_idx], dim=0)
#     low_resp = torch.mean(feat_abs[low_idx], dim=0)
# 
#     eps = 1e-6
# 
#     high_fg = high_resp[fg].mean().item() if fg.sum() > 0 else 0
#     high_bg = high_resp[bg].mean().item() if bg.sum() > 0 else 0
#     low_fg = low_resp[fg].mean().item() if fg.sum() > 0 else 0
#     low_bg = low_resp[bg].mean().item() if bg.sum() > 0 else 0
# 
#     return {
#         "high_fg": high_fg,
#         "high_bg": high_bg,
#         "high_ratio": high_fg / (high_bg + eps),
#         "low_fg": low_fg,
#         "low_bg": low_bg,
#         "low_ratio": low_fg / (low_bg + eps),
#     }
# 
# 
# def save_region_ratio_csv(path, ratio):
#     with open(path, "w") as f:
#         f.write("type,fg_response,bg_response,fg_bg_ratio\n")
#         f.write(
#             f"high_weight,{ratio['high_fg']:.6f},{ratio['high_bg']:.6f},{ratio['high_ratio']:.6f}\n"
#         )
#         f.write(
#             f"low_weight,{ratio['low_fg']:.6f},{ratio['low_bg']:.6f},{ratio['low_ratio']:.6f}\n"
#         )
# 
# 
# def find_tafm_debug_modules(model):
#     modules = []
# 
#     for name, module in model.named_modules():
#         if (
#             hasattr(module, "debug_att")
#             and hasattr(module, "debug_x_fused")
#             and hasattr(module, "debug_x_refined")
#         ):
#             if module.debug_att is not None:
#                 modules.append((name, module))
# 
#     return modules
# 
# 
# def stage_name_from_module(module_name, idx):
#     name_lower = module_name.lower()
# 
#     if "x2" in name_lower:
#         return "stage2"
#     if "x3" in name_lower:
#         return "stage3"
#     if "x4" in name_lower:
#         return "stage4"
#     if "x5" in name_lower:
#         return "stage5"
# 
#     return f"stage{idx + 2}"
# 
# 
# def build_stage_dirs(save_root, stage_name):
#     root = os.path.join(save_root, stage_name)
# 
#     dirs = {
#         "panel": os.path.join(root, "panel"),
#         "high_low": os.path.join(root, "high_low_response"),
#         "rank": os.path.join(root, "weight_rank"),
#         "csv": os.path.join(root, "weights_csv"),
#         "ratio": os.path.join(root, "region_ratio_csv"),
#     }
# 
#     for d in dirs.values():
#         ensure_dir(d)
# 
#     return dirs
# 
# 
# def load_model(args):
#     model = BaseNet_LWGANet_L2(pretrained=False)
# 
#     if not torch.cuda.is_available():
#         model = revert_sync_batchnorm(model)
#         print("已将模型中的 SyncBatchNorm 自动降级为 BatchNorm2d 以兼容 CPU")
# 
#     if not os.path.isfile(args.weight):
#         print(f"错误: 找不到权重文件 {args.weight}")
#         sys.exit(1)
# 
#     print(f"=> Loading weights from: {args.weight}")
#     checkpoint = torch.load(args.weight)
# 
#     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         state_dict = checkpoint["state_dict"]
#     else:
#         state_dict = checkpoint
# 
#     model.load_state_dict(state_dict, strict=False)
#     print("权重加载成功！")
# 
#     if args.onGPU and torch.cuda.is_available():
#         model = model.cuda()
#         cudnn.benchmark = True
# 
#     model.eval()
# 
#     total_params = sum([np.prod(p.size()) for p in model.parameters()])
#     print("Total network parameters:", total_params)
# 
#     return model
# 
# 
# def build_test_loader(args):
#     dataset_name = "GVLM"
# 
#     if os.path.exists(args.file_root):
#         print(f"=> 使用测试数据集路径: {args.file_root}, dataset_name: {dataset_name}")
#     else:
#         raise TypeError(f"数据集路径 {args.file_root} 不存在，请检查！")
# 
#     mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
# 
#     valDataset = myTransforms.Compose(
#         [
#             myTransforms.Normalize(mean=mean, std=std),
#             myTransforms.Scale(args.inWidth, args.inHeight),
#             myTransforms.ToTensor(),
#         ]
#     )
# 
#     test_data = myDataLoader.Dataset(
#         "test",
#         file_root=args.file_root,
#         transform=valDataset,
#         dataset_name=dataset_name,
#     )
# 
#     testLoader = torch.utils.data.DataLoader(
#         test_data,
#         shuffle=False,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=False,
#     )
# 
#     print(f"=> Test dataset length: {len(test_data)}")
#     print(f"=> Test loader batches: {len(testLoader)}")
# 
#     return test_data, testLoader
# 
# 
# def visualize_tafm_debug_all_stages(args):
#     torch.backends.cudnn.benchmark = True
# 
#     SEED = 2333
#     torch.manual_seed(SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(SEED)
# 
#     if not dist.is_initialized():
#         port = random.randint(20000, 30000)
#         dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
#         dist.init_process_group(
#             backend=dist_backend,
#             init_method=f"tcp://127.0.0.1:{port}",
#             rank=0,
#             world_size=1,
#         )
# 
#     model = load_model(args)
# 
#     test_data, testLoader = build_test_loader(args)
# 
#     names = get_filenames_from_dataset_or_fallback(test_data, args.file_root)
# 
#     ensure_dir(args.save_root)
# 
#     print("=========================================================================")
#     print("开始输出 TAFM debug 可视化结果")
#     print(f"save_root: {args.save_root}")
#     print("=========================================================================")
# 
#     global_idx = 0
#     saved = 0
# 
#     iterator = enumerate(testLoader)
#     if tqdm is not None:
#         iterator = tqdm(iterator, total=len(testLoader), desc="Generating TAFM debug")
# 
#     with torch.no_grad():
#         for batch_idx, batched_inputs in iterator:
#             img, target = batched_inputs
# 
#             # =====================================================
#             # 模型推理严格沿用你的原测试代码，不改！
#             # =====================================================
#             pre_img = img[:, 0:3]
#             post_img = img[:, 3:6]
# 
#             if args.onGPU and torch.cuda.is_available():
#                 pre_img = pre_img.cuda()
#                 post_img = post_img.cuda()
# 
#             outputs = model(pre_img, post_img)
# 
#             tafm_modules = find_tafm_debug_modules(model)
# 
#             if len(tafm_modules) == 0:
#                 raise RuntimeError(
#                     "没有拿到 debug_att！请确认 models/model_with_tafm.py 中 EnhancedTFFModule.forward "
#                     "已经加入 self.debug_att = att.detach()。"
#                 )
# 
#             if batch_idx == 0:
#                 print("=> 已找到 TAFM debug modules:")
#                 for idx, (name, module) in enumerate(tafm_modules):
#                     print(
#                         f"   {idx}: {name}, "
#                         f"debug_att={tuple(module.debug_att.shape)}, "
#                         f"x_fused={tuple(module.debug_x_fused.shape)}"
#                     )
# 
#             # 最终预测使用 m2
#             m2 = outputs[0].detach().cpu().numpy()[:, 0]
# 
#             img_cpu = img.detach().cpu()
#             target_cpu = target.detach().cpu()
# 
#             batch_size = img_cpu.shape[0]
#             H, W = img_cpu.shape[2], img_cpu.shape[3]
# 
#             for i in range(batch_size):
#                 if args.max_save > 0 and saved >= args.max_save:
#                     print(f"=> 已达到 max_save={args.max_save}，提前结束。")
#                     print(f"=> 结果保存到: {args.save_root}")
#                     return
# 
#                 if global_idx < len(names):
#                     filename = safe_filename(names[global_idx], global_idx)
#                 else:
#                     filename = f"{global_idx:04d}.png"
# 
#                 base = os.path.splitext(filename)[0]
# 
#                 # =====================================================
#                 # 只修显示，不改推理
#                 # 你之前发现 T1/T2 显示反了，所以默认 swap_display=True
#                 # =====================================================
#                 if args.swap_display:
#                     post_rgb = denorm_rgb(img_cpu[i, 0:3])
#                     pre_rgb = denorm_rgb(img_cpu[i, 3:6])
#                 else:
#                     pre_rgb = denorm_rgb(img_cpu[i, 0:3])
#                     post_rgb = denorm_rgb(img_cpu[i, 3:6])
# 
#                 if target_cpu[i].ndim == 3:
#                     gt = target_cpu[i].squeeze(0).numpy()
#                 else:
#                     gt = target_cpu[i].numpy()
# 
#                 gt = (gt > 0).astype(np.uint8)
# 
#                 pred_bin = (m2[i] > args.threshold).astype(np.uint8)
#                 overlay = make_overlay(post_rgb, pred_bin)
# 
#                 # =====================================================
#                 # 四个 stage 全部扒出来
#                 # =====================================================
#                 for s_idx, (module_name, module) in enumerate(tafm_modules):
#                     stage_name = stage_name_from_module(module_name, s_idx)
#                     dirs = build_stage_dirs(args.save_root, stage_name)
# 
#                     x_fused = module.debug_x_fused.detach().cpu()
#                     x_refined = module.debug_x_refined.detach().cpu()
#                     att = module.debug_att.detach().cpu()
# 
#                     x_fused_i = x_fused[i]
#                     x_refined_i = x_refined[i]
#                     att_i = att[i, :, 0, 0].numpy()
# 
#                     before_map = feature_to_heatmap(x_fused_i, out_size=(H, W))
#                     after_map = feature_to_heatmap(x_refined_i, out_size=(H, W))
# 
#                     high_map = group_response_map(
#                         x_fused_i,
#                         att_i,
#                         out_size=(H, W),
#                         topk=args.topk,
#                         high=True,
#                     )
# 
#                     low_map = group_response_map(
#                         x_fused_i,
#                         att_i,
#                         out_size=(H, W),
#                         topk=args.topk,
#                         high=False,
#                     )
# 
#                     panel_path = os.path.join(dirs["panel"], filename)
#                     high_low_path = os.path.join(dirs["high_low"], filename)
#                     rank_path = os.path.join(dirs["rank"], filename)
#                     csv_path = os.path.join(dirs["csv"], base + ".csv")
#                     ratio_path = os.path.join(dirs["ratio"], base + "_ratio.csv")
# 
#                     save_basic_panel(
#                         panel_path,
#                         pre_rgb,
#                         post_rgb,
#                         gt,
#                         before_map,
#                         after_map,
#                         pred_bin,
#                         overlay,
#                         att_i,
#                         stage_name,
#                     )
# 
#                     save_high_low_panel(
#                         high_low_path,
#                         pre_rgb,
#                         post_rgb,
#                         gt,
#                         high_map,
#                         low_map,
#                         pred_bin,
#                         stage_name,
#                     )
# 
#                     save_weight_rank_bar(
#                         rank_path,
#                         att_i,
#                         stage_name,
#                         topk=args.topk,
#                     )
# 
#                     save_weights_csv(csv_path, att_i)
# 
#                     ratio = calc_region_ratio(
#                         x_fused_i,
#                         att_i,
#                         gt,
#                         topk=args.topk,
#                     )
# 
#                     save_region_ratio_csv(ratio_path, ratio)
# 
#                 saved += 1
#                 global_idx += 1
# 
#                 if tqdm is None:
#                     print(f"[TAFM] saved {saved}, file={filename}")
# 
#             if tqdm is not None:
#                 iterator.set_postfix({"saved": saved})
# 
#     print("=========================================================================")
#     print("TAFM debug 可视化全部完成")
#     print(f"共保存样本数: {saved}")
#     print(f"输出目录: {args.save_root}")
#     print("=========================================================================")
# 
# 
# if __name__ == "__main__":
#     args = parse_args()
# 
#     if not torch.cuda.is_available():
#         args.onGPU = False
# 
#     print("Called with args:")
#     print(args)
# 
#     visualize_tafm_debug_all_stages(args)





#单张图片TAFM特征图可视化

#单张图片TAFM特征图可视化

# import os
# import sys
# import random
# import warnings
# from argparse import ArgumentParser
#
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F
#
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
# warnings.filterwarnings("ignore", category=UserWarning)
#
# # ==========================================================
# # 更改入口：在这里填写你想要可视化的图片文件名
# # 文件名必须和 /home/LWGANet/GVLM-CD-Processed/test/A 里面的文件名一致
# # 例如：SELECT_IMAGE_NAME = "000123.png"
# # ==========================================================
# SELECT_IMAGE_NAME = "1127.png"
#
# # 是否输出四个 stage。时间紧就只保留 ["stage2"]。
# STAGES_TO_SAVE = ["stage2", "stage3", "stage4", "stage5"]
#
# # ==========================================================
# # CPU 环境下自动 map_location
# # ==========================================================
# _original_torch_load = torch.load
#
#
# def _safe_torch_load(*args, **kwargs):
#     if not torch.cuda.is_available():
#         kwargs["map_location"] = "cpu"
#     return _original_torch_load(*args, **kwargs)
#
#
# torch.load = _safe_torch_load
#
# # ==========================================================
# # 项目内部导入
# # ==========================================================
# import dataset_GVLM as myDataLoader
# import Transforms as myTransforms
#
# # 注意：模型文件放在 /home/LWGANet/models/model_with_tafm_debug_single.py 时，用这一行
# # 如果你仍然用旧文件名 model_with_tafm_debug.py，把下面 import 文件名改回去即可
# from models.model_with_tafm_debug_single import ModelToVisualize as BaseNet_LWGANet_L2
#
#
# def parse_args():
#     parser = ArgumentParser()
#
#     parser.add_argument("--inWidth", type=int, default=256)
#     parser.add_argument("--inHeight", type=int, default=256)
#     parser.add_argument("--num_workers", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--onGPU", default=True, type=lambda x: (str(x).lower() == "true"))
#
#     parser.add_argument(
#         "--weight",
#         default="/home/LWGANet/results/A2Net_LWGANet/L2_DSBN_GVLM_Composite/GVLM-CD-Processed/25.08.14-20.24/best_model.pth",
#         type=str,
#         help="Pretrained weight path",
#     )
#
#     parser.add_argument(
#         "--file_root",
#         default="/home/LWGANet/GVLM-CD-Processed",
#         help="Data directory",
#     )
#
#     parser.add_argument(
#         "--save_root",
#         default="./test_results/TAFM_SingleImage_TopBottom_SharedScale",
#         type=str,
#         help="Directory to save single image TAFM visualizations",
#     )
#
#     parser.add_argument(
#         "--topk",
#         default=10,
#         type=int,
#         help="Top-k high/low weight channels",
#     )
#
#     parser.add_argument(
#         "--shared_percentile",
#         default=99.0,
#         type=float,
#         help="Percentile used as shared vmax for Top/Bottom single-channel maps. "
#              "Use 99.0 by default to reduce the influence of extreme hot pixels.",
#     )
#
#     parser.add_argument(
#         "--threshold",
#         default=0.5,
#         type=float,
#         help="Threshold for binary prediction",
#     )
#
#     # 只修显示，不改推理
#     parser.add_argument(
#         "--swap_display",
#         default=True,
#         type=lambda x: (str(x).lower() == "true"),
#         help="Swap T1/T2 for visualization only. Model input order is unchanged.",
#     )
#
#     return parser.parse_args()
#
#
# def revert_sync_batchnorm(module):
#     res = module
#     if isinstance(module, torch.nn.SyncBatchNorm):
#         res = torch.nn.BatchNorm2d(
#             module.num_features,
#             module.eps,
#             module.momentum,
#             module.affine,
#             module.track_running_stats,
#         )
#         if module.affine:
#             res.weight.data = module.weight.data.clone().detach()
#             res.bias.data = module.bias.data.clone().detach()
#         res.running_mean = module.running_mean
#         res.running_var = module.running_var
#         res.num_batches_tracked = module.num_batches_tracked
#
#     for name, child in module.named_children():
#         res.add_module(name, revert_sync_batchnorm(child))
#
#     return res
#
#
# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)
#
#
# def safe_filename(name, idx):
#     name = str(name).strip()
#     name = os.path.basename(name)
#
#     if name == "" or name.lower() in ["none", "nan"]:
#         name = f"{idx:04d}.png"
#
#     base, ext = os.path.splitext(name)
#
#     if base == "":
#         base = f"{idx:04d}"
#
#     if ext == "":
#         ext = ".png"
#
#     if ext.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
#         ext = ".png"
#
#     return base + ext
#
#
# def _basename_from_item(item):
#     if isinstance(item, str):
#         return os.path.basename(item)
#
#     if isinstance(item, (int, np.integer)):
#         return f"{int(item)}.png"
#
#     if isinstance(item, (tuple, list)) and len(item) > 0:
#         for x in item:
#             if isinstance(x, str) and x.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
#                 return os.path.basename(x)
#         for x in item:
#             if isinstance(x, str) and x.strip() != "":
#                 return os.path.basename(x)
#         return str(item[0])
#
#     if isinstance(item, dict):
#         keys = [
#             "name", "filename", "file_name", "id",
#             "A", "B", "img", "image", "path",
#             "A_path", "img_path", "image_path",
#         ]
#         for k in keys:
#             if k in item:
#                 return os.path.basename(str(item[k]))
#
#     return str(item)
#
#
# def get_filenames_from_dataset_or_fallback(dataset, file_root):
#     print("=> 尝试从 dataset 内部读取真实文件名顺序...")
#
#     candidate_attrs = [
#         "A_paths", "B_paths", "img_paths", "image_paths",
#         "file_paths", "path_list", "file_list", "files",
#         "imgs", "images", "image_list", "img_list",
#         "data_list", "test_files", "file_names", "filenames",
#         "names", "name_list", "ids",
#     ]
#
#     for attr in candidate_attrs:
#         if not hasattr(dataset, attr):
#             continue
#
#         value = getattr(dataset, attr)
#
#         if isinstance(value, list) and len(value) == len(dataset):
#             names = []
#             ok = True
#
#             for idx, item in enumerate(value):
#                 name = _basename_from_item(item)
#                 if name is None:
#                     ok = False
#                     break
#                 names.append(safe_filename(name, idx))
#
#             if ok:
#                 print(f"=> 文件名已从 dataset.{attr} 读取，数量: {len(names)}")
#                 print(f"=> 前 10 个文件名: {names[:10]}")
#                 return names
#
#     print("=> 未从 dataset 内部找到可靠文件名，退回 sorted(test/A)。")
#
#     img_dir = os.path.join(file_root, "val", "A")
#
#     if not os.path.isdir(img_dir):
#         print(f"警告: 找不到 {img_dir}，将使用 index 命名。")
#         return [f"{i:04d}.png" for i in range(len(dataset))]
#
#     names = sorted(
#         [
#             f for f in os.listdir(img_dir)
#             if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
#         ]
#     )
#
#     if len(names) != len(dataset):
#         print("警告: test/A 文件数量与 dataset 长度不一致，使用 index 命名。")
#         return [f"{i:04d}.png" for i in range(len(dataset))]
#
#     names = [safe_filename(n, idx) for idx, n in enumerate(names)]
#
#     print(f"=> sorted(test/A) 文件数量: {len(names)}")
#     print(f"=> 前 10 个文件名: {names[:10]}")
#
#     return names
#
#
#
# def resolve_selected_index_by_name(names, image_name):
#     """
#     根据文件名在测试集文件名列表中找到对应 index。
#
#     names: list[str]，测试集文件名顺序
#     image_name: str，想要选择的图片文件名，例如 "000123.png"
#
#     返回:
#     - selected_idx: int
#     - filename: str
#     """
#     wanted = os.path.basename(str(image_name).strip())
#
#     if wanted == "":
#         raise ValueError("SELECT_IMAGE_NAME 为空，请在代码顶部“更改入口”处填写图片文件名。")
#
#     # 先按完整文件名精确匹配
#     if wanted in names:
#         return names.index(wanted), safe_filename(wanted, names.index(wanted))
#
#     # 再按 safe_filename 后的名字匹配
#     safe_wanted = safe_filename(wanted, 0)
#     if safe_wanted in names:
#         return names.index(safe_wanted), safe_wanted
#
#     # 如果用户只写了不带后缀的名字，也允许按 stem 匹配
#     wanted_stem, wanted_ext = os.path.splitext(wanted)
#     if wanted_ext == "":
#         matched = [n for n in names if os.path.splitext(n)[0] == wanted_stem]
#         if len(matched) == 1:
#             filename = matched[0]
#             return names.index(filename), filename
#         elif len(matched) > 1:
#             raise ValueError(
#                 f"文件名 stem={wanted_stem} 匹配到多个文件，请填写完整文件名：{matched[:10]}"
#             )
#
#     preview = "\n".join([f"  {i}: {n}" for i, n in enumerate(names[:30])])
#     raise ValueError(
#         f"找不到指定图片文件名: {wanted}\n"
#         f"请确认它位于 test/A 中，并且文件名完全一致。\n"
#         f"当前测试集前 30 个文件名如下：\n{preview}"
#     )
#
#
# def denorm_rgb(x):
#     """
#     x: [3,H,W], normalized tensor.
#     """
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#
#     x = x.detach().cpu().float()
#     x = x * std + mean
#     x = torch.clamp(x, 0, 1)
#     return x.permute(1, 2, 0).numpy()
#
#
# def normalize01(x):
#     x = x.astype(np.float32)
#     return (x - x.min()) / (x.max() - x.min() + 1e-8)
#
#
# def feature_to_heatmap(feat, out_size):
#     """
#     feat: torch.Tensor [C,h,w]
#     output: numpy [H,W]
#     """
#     feat_map = torch.mean(torch.abs(feat), dim=0, keepdim=True).unsqueeze(0)
#     feat_map = F.interpolate(
#         feat_map,
#         size=out_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     feat_map = feat_map[0, 0].detach().cpu().numpy()
#     return normalize01(feat_map)
#
#
# def single_channel_heatmap(feat_ch, out_size):
#     """
#     feat_ch: torch.Tensor [h,w]
#     output: numpy [H,W]
#     """
#     feat = torch.abs(feat_ch).unsqueeze(0).unsqueeze(0)
#     feat = F.interpolate(
#         feat,
#         size=out_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     heat = feat[0, 0].detach().cpu().numpy()
#     return normalize01(heat)
#
#
# def single_channel_raw_map(feat_ch, out_size, weight=None):
#     """
#     返回未单独归一化的单通道响应图。
#
#     feat_ch: torch.Tensor [h,w]
#     weight: None 或 float。如果给定，则返回 abs(feat_ch) * weight，
#             用于显示通道注意力加权后的实际贡献强度。
#     output: numpy [H,W], raw response map
#     """
#     feat = torch.abs(feat_ch).unsqueeze(0).unsqueeze(0)
#     feat = F.interpolate(
#         feat,
#         size=out_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     heat = feat[0, 0].detach().cpu().numpy().astype(np.float32)
#
#     if weight is not None:
#         heat = heat * float(weight)
#
#     return heat
#
#
# def normalize_with_shared_vmax(x, vmax):
#     """
#     使用统一 vmax 归一化，而不是每张图单独 normalize01。
#     这样 Top/Bottom 通道之间的真实幅值差异不会被洗掉。
#     """
#     x = x.astype(np.float32)
#     vmax = float(vmax)
#     if vmax <= 1e-8:
#         return np.zeros_like(x, dtype=np.float32)
#     return np.clip(x / vmax, 0.0, 1.0)
#
#
# def collect_top_bottom_raw_maps(
#     x_fused_i,
#     att_np,
#     out_size,
#     topk=10,
#     weighted=False,
# ):
#     """
#     收集 Top-k 和 Bottom-k 单通道原始响应图，不做单图归一化。
#
#     weighted=False: 显示原始通道响应 abs(x_fused[ch])
#     weighted=True : 显示注意力加权响应 abs(x_fused[ch]) * att[ch]
#     """
#     idx_sorted = np.argsort(-att_np)
#     top_idx = idx_sorted[:topk]
#     bottom_idx = idx_sorted[-topk:]
#
#     entries = []
#
#     for rank, ch in enumerate(top_idx, start=1):
#         weight = float(att_np[ch]) if weighted else None
#         heat_raw = single_channel_raw_map(
#             x_fused_i[int(ch)],
#             out_size=out_size,
#             weight=weight,
#         )
#         entries.append(
#             {
#                 "type": "Top",
#                 "rank": rank,
#                 "channel": int(ch),
#                 "weight": float(att_np[ch]),
#                 "raw_map": heat_raw,
#             }
#         )
#
#     for rank, ch in enumerate(bottom_idx, start=1):
#         weight = float(att_np[ch]) if weighted else None
#         heat_raw = single_channel_raw_map(
#             x_fused_i[int(ch)],
#             out_size=out_size,
#             weight=weight,
#         )
#         entries.append(
#             {
#                 "type": "Bottom",
#                 "rank": rank,
#                 "channel": int(ch),
#                 "weight": float(att_np[ch]),
#                 "raw_map": heat_raw,
#             }
#         )
#
#     return entries
#
#
# def compute_shared_vmax(entries, percentile=99.0):
#     """
#     对 Top-k + Bottom-k 所有单通道图一起计算统一色阶上限。
#     默认使用 99% 分位数，避免少数极端亮点把整体压得太暗。
#     """
#     values = []
#     for item in entries:
#         m = item["raw_map"]
#         values.append(m.reshape(-1))
#
#     values = np.concatenate(values, axis=0)
#
#     if values.size == 0:
#         return 1.0
#
#     vmax = np.percentile(values, percentile)
#     if vmax <= 1e-8:
#         vmax = values.max() if values.max() > 1e-8 else 1.0
#
#     return float(vmax)
#
#
# def group_response_map(x_fused_i, att_np, out_size, topk=10, high=True):
#     """
#     x_fused_i: torch.Tensor [C,h,w]
#     att_np: numpy [C]
#     """
#     idx_sorted = np.argsort(-att_np)
#
#     if high:
#         idx = idx_sorted[:topk]
#     else:
#         idx = idx_sorted[-topk:]
#
#     feat = torch.mean(torch.abs(x_fused_i[idx]), dim=0, keepdim=True).unsqueeze(0)
#     feat = F.interpolate(
#         feat,
#         size=out_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     feat = feat[0, 0].detach().cpu().numpy()
#     return normalize01(feat)
#
#
# def make_overlay(post_rgb, pred_bin, alpha=0.45):
#     overlay = post_rgb.copy()
#     red = np.zeros_like(post_rgb)
#     red[..., 0] = 1.0
#     overlay[pred_bin == 1] = (1 - alpha) * overlay[pred_bin == 1] + alpha * red[pred_bin == 1]
#     return np.clip(overlay, 0, 1)
#
#
# def save_weight_curve(ax, att_np, title="Channel weights"):
#     channels = np.arange(1, len(att_np) + 1)
#     ax.plot(channels, att_np, linewidth=1.5)
#     ax.set_xlim(1, len(att_np))
#     ax.set_ylim(0, 1)
#     ax.set_title(title, fontsize=9)
#     ax.set_xlabel("Channel", fontsize=8)
#     ax.set_ylabel("Weight", fontsize=8)
#     ax.tick_params(labelsize=7)
#     ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
#
#
# def save_basic_panel(path, pre_rgb, post_rgb, gt, before_map, after_map, pred_bin, overlay, att_np, stage_name):
#     """
#     T1 | T2 | Label | before | after | Pred | Overlay | weight curve
#     """
#     plt.figure(figsize=(22, 4))
#
#     titles = [
#         "T1", "T2", "Label",
#         f"{stage_name} before",
#         f"{stage_name} after",
#         "Pred", "Overlay",
#     ]
#
#     images = [
#         pre_rgb, post_rgb, gt,
#         before_map, after_map,
#         pred_bin, overlay,
#     ]
#
#     for i, (title, img) in enumerate(zip(titles, images)):
#         ax = plt.subplot(1, 8, i + 1)
#
#         if "before" in title or "after" in title:
#             ax.imshow(img, cmap="jet", vmin=0, vmax=1)
#         elif img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#
#         ax.set_title(title, fontsize=9)
#         ax.axis("off")
#
#     ax_curve = plt.subplot(1, 8, 8)
#     save_weight_curve(ax_curve, att_np, title=f"{stage_name} weights")
#
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_high_low_panel(path, pre_rgb, post_rgb, gt, high_map, low_map, pred_bin, stage_name):
#     plt.figure(figsize=(16, 4))
#
#     titles = [
#         "T1", "T2", "Label",
#         f"{stage_name} high-weight",
#         f"{stage_name} low-weight",
#         "Pred",
#     ]
#
#     images = [
#         pre_rgb, post_rgb, gt,
#         high_map, low_map,
#         pred_bin,
#     ]
#
#     for i, (title, img) in enumerate(zip(titles, images)):
#         ax = plt.subplot(1, 6, i + 1)
#
#         if "high-weight" in title or "low-weight" in title:
#             ax.imshow(img, cmap="jet", vmin=0, vmax=1)
#         elif img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#
#         ax.set_title(title, fontsize=9)
#         ax.axis("off")
#
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_weight_rank_bar(path, att_np, stage_name, topk=10):
#     idx_sorted = np.argsort(-att_np)
#     top_idx = idx_sorted[:topk]
#     bottom_idx = idx_sorted[-topk:]
#
#     all_idx = np.concatenate([top_idx, bottom_idx])
#     labels = [f"Ch{c + 1}" for c in all_idx]
#     values = att_np[all_idx]
#
#     plt.figure(figsize=(10, 4))
#     plt.bar(np.arange(len(values)), values)
#     plt.xticks(np.arange(len(values)), labels, rotation=45, fontsize=8)
#     plt.ylim(0, 1)
#     plt.ylabel("Attention weight")
#     plt.title(f"{stage_name}: Top-{topk} and Bottom-{topk} channel weights")
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_weights_csv(path, att_np):
#     channel_ids = np.arange(1, len(att_np) + 1)
#     table = np.stack([channel_ids, att_np], axis=1)
#     np.savetxt(
#         path,
#         table,
#         delimiter=",",
#         header="channel,attention_weight",
#         comments="",
#         fmt=["%d", "%.6f"],
#     )
#
#
# def calc_one_channel_stats(x_fused_i, ch_idx, gt):
#     """
#     返回某个单通道在滑坡区/背景区的响应统计。
#     x_fused_i: [C,h,w]
#     ch_idx: int, 0-based
#     gt: numpy [H,W], 0/1
#     """
#     feat_abs = torch.abs(x_fused_i[ch_idx])
#
#     gt_t = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
#     gt_small = F.interpolate(
#         gt_t,
#         size=(feat_abs.shape[0], feat_abs.shape[1]),
#         mode="nearest",
#     )[0, 0]
#
#     fg = gt_small > 0.5
#     bg = gt_small <= 0.5
#
#     eps = 1e-6
#
#     fg_mean = feat_abs[fg].mean().item() if fg.sum() > 0 else 0.0
#     bg_mean = feat_abs[bg].mean().item() if bg.sum() > 0 else 0.0
#     ratio = fg_mean / (bg_mean + eps)
#
#     return {
#         "fg_mean": fg_mean,
#         "bg_mean": bg_mean,
#         "fg_bg_ratio": ratio,
#         "global_mean": feat_abs.mean().item(),
#         "global_max": feat_abs.max().item(),
#     }
#
#
# def save_top_bottom_stats_csv(path, x_fused_i, att_np, gt, topk=10):
#     idx_sorted = np.argsort(-att_np)
#     top_idx = idx_sorted[:topk]
#     bottom_idx = idx_sorted[-topk:]
#
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(
#             "rank,type,channel,attention_weight,"
#             "fg_mean_response,bg_mean_response,fg_bg_ratio,"
#             "global_mean_response,global_max_response,"
#             "response_observation,interpretation_name\n"
#         )
#
#         for rank, ch in enumerate(top_idx, start=1):
#             stat = calc_one_channel_stats(x_fused_i, int(ch), gt)
#             f.write(
#                 f"{rank},top,Ch{ch + 1},{att_np[ch]:.6f},"
#                 f"{stat['fg_mean']:.6f},{stat['bg_mean']:.6f},{stat['fg_bg_ratio']:.6f},"
#                 f"{stat['global_mean']:.6f},{stat['global_max']:.6f},,\n"
#             )
#
#         for rank, ch in enumerate(bottom_idx, start=1):
#             stat = calc_one_channel_stats(x_fused_i, int(ch), gt)
#             f.write(
#                 f"{rank},bottom,Ch{ch + 1},{att_np[ch]:.6f},"
#                 f"{stat['fg_mean']:.6f},{stat['bg_mean']:.6f},{stat['fg_bg_ratio']:.6f},"
#                 f"{stat['global_mean']:.6f},{stat['global_max']:.6f},,\n"
#             )
#
#
# def save_single_channel_top_bottom_panel(
#     path,
#     pre_rgb,
#     post_rgb,
#     gt,
#     pred_bin,
#     x_fused_i,
#     att_np,
#     out_size,
#     stage_name,
#     topk=10,
#     shared_percentile=99.0,
#     weighted=False,
# ):
#     """
#     输出一张 Top/Bottom 单通道综合图，使用统一色阶。
#
#     weighted=False:
#         显示原始通道响应 abs(x_fused[ch])，但 Top/Bottom 20 张图共用同一个 vmax。
#     weighted=True:
#         显示注意力加权后的响应 abs(x_fused[ch]) * att[ch]，更接近该通道进入后续解码器的贡献强度。
#
#     注意：
#     不再对每个通道单独 normalize01，而是对 Top-k + Bottom-k 所有图一起计算 shared_vmax。
#     """
#     entries = collect_top_bottom_raw_maps(
#         x_fused_i,
#         att_np,
#         out_size=out_size,
#         topk=topk,
#         weighted=weighted,
#     )
#     shared_vmax = compute_shared_vmax(entries, percentile=shared_percentile)
#
#     top_entries = [e for e in entries if e["type"] == "Top"]
#     bottom_entries = [e for e in entries if e["type"] == "Bottom"]
#
#     ncols = 4 + topk
#     plt.figure(figsize=(2.2 * ncols, 6.0))
#
#     base_titles = ["T1", "T2", "Label", "Pred"]
#     base_imgs = [pre_rgb, post_rgb, gt, pred_bin]
#
#     for j, (title, img) in enumerate(zip(base_titles, base_imgs)):
#         ax = plt.subplot(2, ncols, j + 1)
#         if img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#         ax.set_title(title, fontsize=8)
#         ax.axis("off")
#
#     # Top-k 单通道：统一色阶
#     for k, item in enumerate(top_entries):
#         heat = normalize_with_shared_vmax(item["raw_map"], shared_vmax)
#         ch = item["channel"]
#         ax = plt.subplot(2, ncols, 4 + k + 1)
#         ax.imshow(heat, cmap="jet", vmin=0, vmax=1)
#         ax.set_title(
#             f"Top{item['rank']}\nCh{ch + 1}\nw={item['weight']:.3f}",
#             fontsize=7,
#         )
#         ax.axis("off")
#
#     # 第二行前四格留空
#     for j in range(4):
#         ax = plt.subplot(2, ncols, ncols + j + 1)
#         ax.axis("off")
#
#     # Bottom-k 单通道：统一色阶
#     for k, item in enumerate(bottom_entries):
#         heat = normalize_with_shared_vmax(item["raw_map"], shared_vmax)
#         ch = item["channel"]
#         ax = plt.subplot(2, ncols, ncols + 4 + k + 1)
#         ax.imshow(heat, cmap="jet", vmin=0, vmax=1)
#         ax.set_title(
#             f"Bottom{item['rank']}\nCh{ch + 1}\nw={item['weight']:.3f}",
#             fontsize=7,
#         )
#         ax.axis("off")
#
#     mode = "weighted" if weighted else "raw"
#     plt.suptitle(
#         f"{stage_name}: Top-{topk} / Bottom-{topk} single-channel responses "
#         f"({mode}, shared vmax=P{shared_percentile:.1f}={shared_vmax:.4f})",
#         fontsize=11,
#     )
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_individual_single_channel_maps(
#     save_dir,
#     x_fused_i,
#     att_np,
#     out_size,
#     topk=10,
#     shared_percentile=99.0,
#     weighted=False,
# ):
#     """
#     每个 Top/Bottom 通道单独保存一张热力图，使用统一色阶。
#     """
#     ensure_dir(save_dir)
#
#     entries = collect_top_bottom_raw_maps(
#         x_fused_i,
#         att_np,
#         out_size=out_size,
#         topk=topk,
#         weighted=weighted,
#     )
#     shared_vmax = compute_shared_vmax(entries, percentile=shared_percentile)
#
#     mode = "weighted" if weighted else "raw"
#
#     # 保存 shared scale 信息
#     with open(os.path.join(save_dir, "_shared_scale_info.txt"), "w", encoding="utf-8") as f:
#         f.write(f"mode={mode}\n")
#         f.write(f"shared_percentile={shared_percentile}\n")
#         f.write(f"shared_vmax={shared_vmax:.8f}\n")
#         f.write("note=All Top/Bottom single-channel maps in this folder use the same vmax.\n")
#
#     for item in entries:
#         heat = normalize_with_shared_vmax(item["raw_map"], shared_vmax)
#         ch = item["channel"]
#         rank = item["rank"]
#         typ = item["type"].lower()
#
#         plt.figure(figsize=(4, 4))
#         plt.imshow(heat, cmap="jet", vmin=0, vmax=1)
#         plt.title(
#             f"{item['type']}{rank}: Ch{ch + 1}, w={item['weight']:.3f}\n"
#             f"{mode}, shared vmax={shared_vmax:.4f}"
#         )
#         plt.axis("off")
#         plt.tight_layout()
#         plt.savefig(
#             os.path.join(
#                 save_dir,
#                 f"{typ}{rank:02d}_Ch{ch + 1}_w{item['weight']:.3f}_{mode}_shared.png",
#             ),
#             dpi=300,
#             bbox_inches="tight",
#         )
#         plt.close()
#
#
# def load_model(args):
#     model = BaseNet_LWGANet_L2(pretrained=False)
#
#     if not torch.cuda.is_available():
#         model = revert_sync_batchnorm(model)
#         print("已将模型中的 SyncBatchNorm 自动降级为 BatchNorm2d 以兼容 CPU")
#
#     if not os.path.isfile(args.weight):
#         print(f"错误: 找不到权重文件 {args.weight}")
#         sys.exit(1)
#
#     print(f"=> Loading weights from: {args.weight}")
#     checkpoint = torch.load(args.weight)
#
#     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         state_dict = checkpoint["state_dict"]
#     else:
#         state_dict = checkpoint
#
#     model.load_state_dict(state_dict, strict=False)
#     print("权重加载成功！")
#
#     if args.onGPU and torch.cuda.is_available():
#         model = model.cuda()
#         cudnn.benchmark = True
#
#     model.eval()
#
#     total_params = sum([np.prod(p.size()) for p in model.parameters()])
#     print("Total network parameters:", total_params)
#
#     return model
#
#
# def build_test_dataset(args):
#     dataset_name = "GVLM"
#
#     if os.path.exists(args.file_root):
#         print(f"=> 使用测试数据集路径: {args.file_root}, dataset_name: {dataset_name}")
#     else:
#         raise TypeError(f"数据集路径 {args.file_root} 不存在，请检查！")
#
#     mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
#
#     valDataset = myTransforms.Compose(
#         [
#             myTransforms.Normalize(mean=mean, std=std),
#             myTransforms.Scale(args.inWidth, args.inHeight),
#             myTransforms.ToTensor(),
#         ]
#     )
#
#     test_data = myDataLoader.Dataset(
#         "val",
#         file_root=args.file_root,
#         transform=valDataset,
#         dataset_name=dataset_name,
#     )
#
#     print(f"=> Test dataset length: {len(test_data)}")
#     return test_data
#
#
# def get_selected_sample(test_data, selected_idx):
#     if selected_idx < 0 or selected_idx >= len(test_data):
#         raise IndexError(
#             f"selected_idx={selected_idx} 超出范围，当前 test 集长度为 {len(test_data)}。"
#         )
#
#     item = test_data[selected_idx]
#
#     if not isinstance(item, (tuple, list)) or len(item) < 2:
#         raise RuntimeError("Dataset 返回格式不是 (img, target)，请检查 dataset_GVLM。")
#
#     img, target = item[0], item[1]
#
#     # img: [6,H,W] -> [1,6,H,W]
#     if img.ndim == 3:
#         img = img.unsqueeze(0)
#
#     # target: [H,W] or [1,H,W] -> [1,H,W] or [1,1,H,W]
#     if target.ndim == 2:
#         target = target.unsqueeze(0)
#     elif target.ndim == 3:
#         target = target.unsqueeze(0)
#
#     return img, target
#
#
# def visualize_selected_single_image(args):
#     SEED = 2333
#     torch.manual_seed(SEED)
#     random.seed(SEED)
#     np.random.seed(SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(SEED)
#
#     model = load_model(args)
#     test_data = build_test_dataset(args)
#     names = get_filenames_from_dataset_or_fallback(test_data, args.file_root)
#
#     selected_idx, filename = resolve_selected_index_by_name(names, SELECT_IMAGE_NAME)
#     base = os.path.splitext(filename)[0]
#
#     image_save_dir = os.path.join(args.save_root, base)
#     ensure_dir(image_save_dir)
#
#     print("=========================================================================")
#     print("开始输出单张图片 TAFM Top/Bottom 通道可视化")
#     print(f"SELECT_IMAGE_NAME: {SELECT_IMAGE_NAME}")
#     print(f"resolved_index: {selected_idx}")
#     print(f"filename: {filename}")
#     print(f"image_save_dir: {image_save_dir}")
#     print("=========================================================================")
#
#     img, target = get_selected_sample(test_data, selected_idx)
#
#     img_cpu = img.detach().cpu()
#     target_cpu = target.detach().cpu()
#
#     pre_img = img[:, 0:3]
#     post_img = img[:, 3:6]
#
#     if args.onGPU and torch.cuda.is_available():
#         pre_img = pre_img.cuda()
#         post_img = post_img.cuda()
#
#     with torch.no_grad():
#         outputs = model(pre_img, post_img)
#
#     debug = model.get_tafm_debug()
#
#     # 最终预测使用 m2
#     m2 = outputs[0].detach().cpu().numpy()[0, 0]
#
#     H, W = img_cpu.shape[2], img_cpu.shape[3]
#
#     # 只修显示，不改推理
#     if args.swap_display:
#         post_rgb = denorm_rgb(img_cpu[0, 0:3])
#         pre_rgb = denorm_rgb(img_cpu[0, 3:6])
#     else:
#         pre_rgb = denorm_rgb(img_cpu[0, 0:3])
#         post_rgb = denorm_rgb(img_cpu[0, 3:6])
#
#     # target 兼容 [1,H,W] / [1,1,H,W]
#     gt_arr = target_cpu[0]
#     if gt_arr.ndim == 3:
#         gt = gt_arr.squeeze(0).numpy()
#     else:
#         gt = gt_arr.numpy()
#
#     gt = (gt > 0).astype(np.uint8)
#
#     pred_bin = (m2 > args.threshold).astype(np.uint8)
#     overlay = make_overlay(post_rgb, pred_bin)
#
#     # 保存说明文件
#     with open(os.path.join(image_save_dir, "selected_info.txt"), "w", encoding="utf-8") as f:
#         f.write(f"SELECT_IMAGE_NAME={SELECT_IMAGE_NAME}\n")
#         f.write(f"resolved_index={selected_idx}\n")
#         f.write(f"filename={filename}\n")
#         f.write(f"topk={args.topk}\n")
#         f.write(f"shared_percentile={args.shared_percentile}\n")
#         f.write(f"threshold={args.threshold}\n")
#         f.write(f"swap_display={args.swap_display}\n")
#         f.write(f"stages={','.join(STAGES_TO_SAVE)}\n")
#         f.write("single_channel_visualization=shared scale, not per-channel normalize01\n")
#
#     for stage_name, stage_debug in debug.items():
#         if stage_name not in STAGES_TO_SAVE:
#             continue
#
#         if stage_debug["x_fused"] is None or stage_debug["att"] is None:
#             print(f"警告: {stage_name} 没有拿到 x_fused 或 att，跳过。")
#             continue
#
#         stage_dir = os.path.join(image_save_dir, stage_name)
#         ensure_dir(stage_dir)
#
#         x_fused_i = stage_debug["x_fused"][0].detach().cpu()
#         x_refined_i = stage_debug["x_refined"][0].detach().cpu()
#         att_i = stage_debug["att"][0, :, 0, 0].detach().cpu().numpy()
#
#         before_map = feature_to_heatmap(x_fused_i, out_size=(H, W))
#         after_map = feature_to_heatmap(x_refined_i, out_size=(H, W))
#
#         high_map = group_response_map(
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             topk=args.topk,
#             high=True,
#         )
#
#         low_map = group_response_map(
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             topk=args.topk,
#             high=False,
#         )
#
#         # 1. 原有 panel
#         save_basic_panel(
#             os.path.join(stage_dir, f"01_panel_{stage_name}.png"),
#             pre_rgb,
#             post_rgb,
#             gt,
#             before_map,
#             after_map,
#             pred_bin,
#             overlay,
#             att_i,
#             stage_name,
#         )
#
#         # 2. 原有 high/low 平均响应 panel
#         save_high_low_panel(
#             os.path.join(stage_dir, f"02_high_low_response_{stage_name}.png"),
#             pre_rgb,
#             post_rgb,
#             gt,
#             high_map,
#             low_map,
#             pred_bin,
#             stage_name,
#         )
#
#         # 3. Top10/Bottom10 权重柱状图
#         save_weight_rank_bar(
#             os.path.join(stage_dir, f"03_weight_rank_{stage_name}.png"),
#             att_i,
#             stage_name,
#             topk=args.topk,
#         )
#
#         # 4. Top10/Bottom10 单通道响应综合图：统一色阶，原始响应
#         save_single_channel_top_bottom_panel(
#             os.path.join(
#                 stage_dir,
#                 f"04_top{args.topk}_bottom{args.topk}_single_channels_SHARED_RAW_{stage_name}.png",
#             ),
#             pre_rgb,
#             post_rgb,
#             gt,
#             pred_bin,
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             stage_name=stage_name,
#             topk=args.topk,
#             shared_percentile=args.shared_percentile,
#             weighted=False,
#         )
#
#         # 5. Top10/Bottom10 单通道响应综合图：统一色阶，注意力加权后贡献
#         save_single_channel_top_bottom_panel(
#             os.path.join(
#                 stage_dir,
#                 f"05_top{args.topk}_bottom{args.topk}_single_channels_SHARED_WEIGHTED_{stage_name}.png",
#             ),
#             pre_rgb,
#             post_rgb,
#             gt,
#             pred_bin,
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             stage_name=stage_name,
#             topk=args.topk,
#             shared_percentile=args.shared_percentile,
#             weighted=True,
#         )
#
#         # 6. 每个单通道单独保存：统一色阶，原始响应
#         save_individual_single_channel_maps(
#             os.path.join(stage_dir, "single_channels_shared_raw"),
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             topk=args.topk,
#             shared_percentile=args.shared_percentile,
#             weighted=False,
#         )
#
#         # 7. 每个单通道单独保存：统一色阶，注意力加权后贡献
#         save_individual_single_channel_maps(
#             os.path.join(stage_dir, "single_channels_shared_weighted"),
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             topk=args.topk,
#             shared_percentile=args.shared_percentile,
#             weighted=True,
#         )
#
#         # 8. 所有 64 通道权重表
#         save_weights_csv(
#             os.path.join(stage_dir, f"weights_all64_{stage_name}.csv"),
#             att_i,
#         )
#
#         # 9. Top10/Bottom10 统计表
#         save_top_bottom_stats_csv(
#             os.path.join(stage_dir, f"top{args.topk}_bottom{args.topk}_stats_{stage_name}.csv"),
#             x_fused_i,
#             att_i,
#             gt,
#             topk=args.topk,
#         )
#
#         print(f"=> {stage_name} 保存完成: {stage_dir}")
#
#     print("=========================================================================")
#     print("单张图片 TAFM Top/Bottom 通道可视化完成")
#     print(f"输出目录: {image_save_dir}")
#     print("=========================================================================")
#
#
# if __name__ == "__main__":
#     args = parse_args()
#
#     if not torch.cuda.is_available():
#         args.onGPU = False
#
#     print("Called with args:")
#     print(args)
#     print(f"SELECT_IMAGE_NAME = {SELECT_IMAGE_NAME}")
#
#     visualize_selected_single_image(args)






#权重都变成0.5
# 单张图片：原始 TAFM vs 固定 attention=0.5 前向推理对照
# 输出：T1/T2/Label/原模型Pred/固定0.5 Pred/差异图/指标表

# import os
# import sys
# import random
# import warnings
# from argparse import ArgumentParser
#
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
#
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
# warnings.filterwarnings("ignore", category=UserWarning)
#
# # ==========================================================
# # 更改入口：在这里填写你想要可视化的图片文件名
# # 文件名必须和 /home/LWGANet/GVLM-CD-Processed/test/A 里面的文件名一致
# # 例如：SELECT_IMAGE_NAME = "000123.png"
# # ==========================================================
# SELECT_IMAGE_NAME = "6966.png"
#
# # 固定注意力值。0.5 表示取消通道间自适应差异，只保留统一中性缩放。
# FIXED_ATT_VALUE = 0.5
#
# # ==========================================================
# # CPU 环境下自动 map_location
# # ==========================================================
# _original_torch_load = torch.load
#
#
# def _safe_torch_load(*args, **kwargs):
#     if not torch.cuda.is_available():
#         kwargs["map_location"] = "cpu"
#     return _original_torch_load(*args, **kwargs)
#
#
# torch.load = _safe_torch_load
#
# # ==========================================================
# # 项目内部导入
# # ==========================================================
# import dataset_GVLM as myDataLoader
# import Transforms as myTransforms
#
# # 需要把模型文件放到：
# # /home/LWGANet/models/model_with_tafm_debug_fixedatt.py
# from models.model_with_tafm_debug_fixedatt import ModelToVisualize as BaseNet_LWGANet_L2
#
#
# def parse_args():
#     parser = ArgumentParser()
#
#     parser.add_argument("--inWidth", type=int, default=256)
#     parser.add_argument("--inHeight", type=int, default=256)
#     parser.add_argument("--num_workers", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--onGPU", default=True, type=lambda x: (str(x).lower() == "true"))
#
#     parser.add_argument(
#         "--weight",
#         default="/home/LWGANet/results/A2Net_LWGANet/L2_DSBN_GVLM_Composite/GVLM-CD-Processed/25.08.14-20.24/best_model.pth",
#         type=str,
#         help="Pretrained weight path",
#     )
#
#     parser.add_argument(
#         "--file_root",
#         default="/home/LWGANet/GVLM-CD-Processed",
#         help="Data directory",
#     )
#
#     parser.add_argument(
#         "--save_root",
#         default="./test_results/TAFM_FixedAtt05_Compare",
#         type=str,
#         help="Directory to save fixed attention comparison results",
#     )
#
#     parser.add_argument(
#         "--threshold",
#         default=0.5,
#         type=float,
#         help="Threshold for binary prediction",
#     )
#
#     # 只修显示，不改推理
#     parser.add_argument(
#         "--swap_display",
#         default=True,
#         type=lambda x: (str(x).lower() == "true"),
#         help="Swap T1/T2 for visualization only. Model input order is unchanged.",
#     )
#
#     return parser.parse_args()
#
#
# def revert_sync_batchnorm(module):
#     res = module
#     if isinstance(module, torch.nn.SyncBatchNorm):
#         res = torch.nn.BatchNorm2d(
#             module.num_features,
#             module.eps,
#             module.momentum,
#             module.affine,
#             module.track_running_stats,
#         )
#         if module.affine:
#             res.weight.data = module.weight.data.clone().detach()
#             res.bias.data = module.bias.data.clone().detach()
#         res.running_mean = module.running_mean
#         res.running_var = module.running_var
#         res.num_batches_tracked = module.num_batches_tracked
#
#     for name, child in module.named_children():
#         res.add_module(name, revert_sync_batchnorm(child))
#
#     return res
#
#
# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)
#
#
# def safe_filename(name, idx):
#     name = str(name).strip()
#     name = os.path.basename(name)
#
#     if name == "" or name.lower() in ["none", "nan"]:
#         name = f"{idx:04d}.png"
#
#     base, ext = os.path.splitext(name)
#
#     if base == "":
#         base = f"{idx:04d}"
#
#     if ext == "":
#         ext = ".png"
#
#     if ext.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
#         ext = ".png"
#
#     return base + ext
#
#
# def _basename_from_item(item):
#     if isinstance(item, str):
#         return os.path.basename(item)
#
#     if isinstance(item, (int, np.integer)):
#         return f"{int(item)}.png"
#
#     if isinstance(item, (tuple, list)) and len(item) > 0:
#         for x in item:
#             if isinstance(x, str) and x.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
#                 return os.path.basename(x)
#         for x in item:
#             if isinstance(x, str) and x.strip() != "":
#                 return os.path.basename(x)
#         return str(item[0])
#
#     if isinstance(item, dict):
#         keys = [
#             "name", "filename", "file_name", "id",
#             "A", "B", "img", "image", "path",
#             "A_path", "img_path", "image_path",
#         ]
#         for k in keys:
#             if k in item:
#                 return os.path.basename(str(item[k]))
#
#     return str(item)
#
#
# def get_filenames_from_dataset_or_fallback(dataset, file_root):
#     print("=> 尝试从 dataset 内部读取真实文件名顺序...")
#
#     candidate_attrs = [
#         "A_paths", "B_paths", "img_paths", "image_paths",
#         "file_paths", "path_list", "file_list", "files",
#         "imgs", "images", "image_list", "img_list",
#         "data_list", "test_files", "file_names", "filenames",
#         "names", "name_list", "ids",
#     ]
#
#     for attr in candidate_attrs:
#         if not hasattr(dataset, attr):
#             continue
#
#         value = getattr(dataset, attr)
#
#         if isinstance(value, list) and len(value) == len(dataset):
#             names = []
#             ok = True
#
#             for idx, item in enumerate(value):
#                 name = _basename_from_item(item)
#                 if name is None:
#                     ok = False
#                     break
#                 names.append(safe_filename(name, idx))
#
#             if ok:
#                 print(f"=> 文件名已从 dataset.{attr} 读取，数量: {len(names)}")
#                 print(f"=> 前 10 个文件名: {names[:10]}")
#                 return names
#
#     print("=> 未从 dataset 内部找到可靠文件名，退回 sorted(test/A)。")
#
#     img_dir = os.path.join(file_root, "test", "A")
#
#     if not os.path.isdir(img_dir):
#         print(f"警告: 找不到 {img_dir}，将使用 index 命名。")
#         return [f"{i:04d}.png" for i in range(len(dataset))]
#
#     names = sorted(
#         [
#             f for f in os.listdir(img_dir)
#             if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
#         ]
#     )
#
#     if len(names) != len(dataset):
#         print("警告: test/A 文件数量与 dataset 长度不一致，使用 index 命名。")
#         return [f"{i:04d}.png" for i in range(len(dataset))]
#
#     names = [safe_filename(n, idx) for idx, n in enumerate(names)]
#
#     print(f"=> sorted(test/A) 文件数量: {len(names)}")
#     print(f"=> 前 10 个文件名: {names[:10]}")
#
#     return names
#
#
# def resolve_selected_index_by_name(names, image_name):
#     wanted = os.path.basename(str(image_name).strip())
#
#     if wanted == "":
#         raise ValueError("SELECT_IMAGE_NAME 为空，请在代码顶部“更改入口”处填写图片文件名。")
#
#     if wanted in names:
#         return names.index(wanted), safe_filename(wanted, names.index(wanted))
#
#     safe_wanted = safe_filename(wanted, 0)
#     if safe_wanted in names:
#         return names.index(safe_wanted), safe_wanted
#
#     wanted_stem, wanted_ext = os.path.splitext(wanted)
#     if wanted_ext == "":
#         matched = [n for n in names if os.path.splitext(n)[0] == wanted_stem]
#         if len(matched) == 1:
#             filename = matched[0]
#             return names.index(filename), filename
#         elif len(matched) > 1:
#             raise ValueError(
#                 f"文件名 stem={wanted_stem} 匹配到多个文件，请填写完整文件名：{matched[:10]}"
#             )
#
#     preview = "\n".join([f"  {i}: {n}" for i, n in enumerate(names[:30])])
#     raise ValueError(
#         f"找不到指定图片文件名: {wanted}\n"
#         f"请确认它位于 test/A 中，并且文件名完全一致。\n"
#         f"当前测试集前 30 个文件名如下：\n{preview}"
#     )
#
#
# def denorm_rgb(x):
#     """
#     x: [3,H,W], normalized tensor.
#     """
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#
#     x = x.detach().cpu().float()
#     x = x * std + mean
#     x = torch.clamp(x, 0, 1)
#     return x.permute(1, 2, 0).numpy()
#
#
# def make_overlay(rgb, pred_bin, alpha=0.45):
#     overlay = rgb.copy()
#     red = np.zeros_like(rgb)
#     red[..., 0] = 1.0
#     overlay[pred_bin == 1] = (1 - alpha) * overlay[pred_bin == 1] + alpha * red[pred_bin == 1]
#     return np.clip(overlay, 0, 1)
#
#
# def make_error_map(gt, pred):
#     """
#     输出 RGB 错误图：
#     TP: white
#     FP: red
#     FN: blue
#     TN: black
#     """
#     h, w = gt.shape
#     out = np.zeros((h, w, 3), dtype=np.float32)
#
#     tp = (gt == 1) & (pred == 1)
#     fp = (gt == 0) & (pred == 1)
#     fn = (gt == 1) & (pred == 0)
#
#     out[tp] = [1, 1, 1]
#     out[fp] = [1, 0, 0]
#     out[fn] = [0, 0.2, 1]
#
#     return out
#
#
# def calc_metrics(gt, pred):
#     gt = gt.astype(np.uint8)
#     pred = pred.astype(np.uint8)
#
#     tp = int(((gt == 1) & (pred == 1)).sum())
#     tn = int(((gt == 0) & (pred == 0)).sum())
#     fp = int(((gt == 0) & (pred == 1)).sum())
#     fn = int(((gt == 1) & (pred == 0)).sum())
#
#     eps = 1e-8
#     precision = tp / (tp + fp + eps)
#     recall = tp / (tp + fn + eps)
#     f1 = 2 * precision * recall / (precision + recall + eps)
#     iou = tp / (tp + fp + fn + eps)
#
#     pred_area = int(pred.sum())
#     gt_area = int(gt.sum())
#
#     return {
#         "tp": tp,
#         "tn": tn,
#         "fp": fp,
#         "fn": fn,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "iou": iou,
#         "pred_area": pred_area,
#         "gt_area": gt_area,
#     }
#
#
# def save_metrics_csv(path, metrics_ori, metrics_fix, fixed_att_value):
#     keys = [
#         "tp", "tn", "fp", "fn",
#         "precision", "recall", "f1", "iou",
#         "pred_area", "gt_area",
#     ]
#
#     with open(path, "w", encoding="utf-8") as f:
#         f.write("mode,fixed_att_value," + ",".join(keys) + "\n")
#
#         f.write("original,adaptive,")
#         f.write(",".join([f"{metrics_ori[k]:.6f}" if isinstance(metrics_ori[k], float) else str(metrics_ori[k]) for k in keys]))
#         f.write("\n")
#
#         f.write(f"fixed_att,{fixed_att_value},")
#         f.write(",".join([f"{metrics_fix[k]:.6f}" if isinstance(metrics_fix[k], float) else str(metrics_fix[k]) for k in keys]))
#         f.write("\n")
#
#
# def save_compare_panel(
#     path,
#     pre_rgb,
#     post_rgb,
#     gt,
#     pred_ori,
#     pred_fix,
#     prob_ori,
#     prob_fix,
#     overlay_ori,
#     overlay_fix,
#     err_ori,
#     err_fix,
#     filename,
#     fixed_att_value,
# ):
#     diff_prob = np.abs(prob_ori - prob_fix)
#     diff_pred = (pred_ori != pred_fix).astype(np.uint8)
#
#     plt.figure(figsize=(24, 8))
#
#     titles = [
#         "T1",
#         "T2",
#         "Label",
#         "Pred: original adaptive att",
#         f"Pred: fixed att={fixed_att_value}",
#         "Prediction difference",
#         "Overlay original",
#         f"Overlay fixed {fixed_att_value}",
#         "Error original\nwhite TP / red FP / blue FN",
#         f"Error fixed {fixed_att_value}\nwhite TP / red FP / blue FN",
#         "Prob original",
#         f"Prob fixed {fixed_att_value}",
#     ]
#
#     images = [
#         pre_rgb,
#         post_rgb,
#         gt,
#         pred_ori,
#         pred_fix,
#         diff_pred,
#         overlay_ori,
#         overlay_fix,
#         err_ori,
#         err_fix,
#         prob_ori,
#         prob_fix,
#     ]
#
#     for i, (title, img) in enumerate(zip(titles, images)):
#         ax = plt.subplot(2, 6, i + 1)
#
#         if i in [0, 1, 6, 7, 8, 9]:
#             ax.imshow(img)
#         elif i in [10, 11]:
#             ax.imshow(img, cmap="jet", vmin=0, vmax=1)
#         else:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#
#         ax.set_title(title, fontsize=9)
#         ax.axis("off")
#
#     plt.suptitle(
#         f"TAFM fixed attention comparison: {filename}",
#         fontsize=13,
#     )
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def load_model(args):
#     model = BaseNet_LWGANet_L2(pretrained=False)
#
#     if not torch.cuda.is_available():
#         model = revert_sync_batchnorm(model)
#         print("已将模型中的 SyncBatchNorm 自动降级为 BatchNorm2d 以兼容 CPU")
#
#     if not os.path.isfile(args.weight):
#         print(f"错误: 找不到权重文件 {args.weight}")
#         sys.exit(1)
#
#     print(f"=> Loading weights from: {args.weight}")
#     checkpoint = torch.load(args.weight)
#
#     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         state_dict = checkpoint["state_dict"]
#     else:
#         state_dict = checkpoint
#
#     model.load_state_dict(state_dict, strict=False)
#     print("权重加载成功！")
#
#     if args.onGPU and torch.cuda.is_available():
#         model = model.cuda()
#         cudnn.benchmark = True
#
#     model.eval()
#
#     total_params = sum([np.prod(p.size()) for p in model.parameters()])
#     print("Total network parameters:", total_params)
#
#     return model
#
#
# def build_test_dataset(args):
#     dataset_name = "GVLM"
#
#     if os.path.exists(args.file_root):
#         print(f"=> 使用测试数据集路径: {args.file_root}, dataset_name: {dataset_name}")
#     else:
#         raise TypeError(f"数据集路径 {args.file_root} 不存在，请检查！")
#
#     mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
#
#     valDataset = myTransforms.Compose(
#         [
#             myTransforms.Normalize(mean=mean, std=std),
#             myTransforms.Scale(args.inWidth, args.inHeight),
#             myTransforms.ToTensor(),
#         ]
#     )
#
#     test_data = myDataLoader.Dataset(
#         "test",
#         file_root=args.file_root,
#         transform=valDataset,
#         dataset_name=dataset_name,
#     )
#
#     print(f"=> Test dataset length: {len(test_data)}")
#     return test_data
#
#
# def get_selected_sample(test_data, selected_idx):
#     if selected_idx < 0 or selected_idx >= len(test_data):
#         raise IndexError(
#             f"selected_idx={selected_idx} 超出范围，当前 test 集长度为 {len(test_data)}。"
#         )
#
#     item = test_data[selected_idx]
#
#     if not isinstance(item, (tuple, list)) or len(item) < 2:
#         raise RuntimeError("Dataset 返回格式不是 (img, target)，请检查 dataset_GVLM。")
#
#     img, target = item[0], item[1]
#
#     if img.ndim == 3:
#         img = img.unsqueeze(0)
#
#     if target.ndim == 2:
#         target = target.unsqueeze(0)
#     elif target.ndim == 3:
#         target = target.unsqueeze(0)
#
#     return img, target
#
#
# def visualize_fixed_attention_comparison(args):
#     SEED = 2333
#     torch.manual_seed(SEED)
#     random.seed(SEED)
#     np.random.seed(SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(SEED)
#
#     model = load_model(args)
#     test_data = build_test_dataset(args)
#     names = get_filenames_from_dataset_or_fallback(test_data, args.file_root)
#
#     selected_idx, filename = resolve_selected_index_by_name(names, SELECT_IMAGE_NAME)
#     base = os.path.splitext(filename)[0]
#
#     image_save_dir = os.path.join(args.save_root, base)
#     ensure_dir(image_save_dir)
#
#     print("=========================================================================")
#     print("开始输出 TAFM 固定注意力 0.5 对照实验")
#     print(f"SELECT_IMAGE_NAME: {SELECT_IMAGE_NAME}")
#     print(f"resolved_index: {selected_idx}")
#     print(f"filename: {filename}")
#     print(f"FIXED_ATT_VALUE: {FIXED_ATT_VALUE}")
#     print(f"image_save_dir: {image_save_dir}")
#     print("=========================================================================")
#
#     img, target = get_selected_sample(test_data, selected_idx)
#
#     img_cpu = img.detach().cpu()
#     target_cpu = target.detach().cpu()
#
#     pre_img = img[:, 0:3]
#     post_img = img[:, 3:6]
#
#     if args.onGPU and torch.cuda.is_available():
#         pre_img = pre_img.cuda()
#         post_img = post_img.cuda()
#
#     # 1. 原始自适应注意力模型
#     model.set_tafm_force_att(None)
#     with torch.no_grad():
#         outputs_ori = model(pre_img, post_img)
#
#     # 2. 固定 attention = 0.5 对照
#     model.set_tafm_force_att(FIXED_ATT_VALUE)
#     with torch.no_grad():
#         outputs_fix = model(pre_img, post_img)
#
#     # ==========================================================
#     # 验证入口：检查固定 attention 是否真的生效
#     # 如果成功，四个 stage 的 att min/max/mean 都应该等于 0.5
#     # ==========================================================
#     debug = model.get_tafm_debug()
#     for stage_name, d in debug.items():
#         att = d["att"]
#         print(
#             stage_name,
#             "att min =", att.min().item(),
#             "att max =", att.max().item(),
#             "att mean =", att.mean().item()
#         )
#     # 退出时恢复原模型模式，防止后续误用
#     model.set_tafm_force_att(None)
#
#     prob_ori = outputs_ori[0].detach().cpu().numpy()[0, 0]
#     prob_fix = outputs_fix[0].detach().cpu().numpy()[0, 0]
#
#     H, W = img_cpu.shape[2], img_cpu.shape[3]
#
#     if args.swap_display:
#         post_rgb = denorm_rgb(img_cpu[0, 0:3])
#         pre_rgb = denorm_rgb(img_cpu[0, 3:6])
#     else:
#         pre_rgb = denorm_rgb(img_cpu[0, 0:3])
#         post_rgb = denorm_rgb(img_cpu[0, 3:6])
#
#     gt_arr = target_cpu[0]
#     if gt_arr.ndim == 3:
#         gt = gt_arr.squeeze(0).numpy()
#     else:
#         gt = gt_arr.numpy()
#     gt = (gt > 0).astype(np.uint8)
#
#     pred_ori = (prob_ori > args.threshold).astype(np.uint8)
#     pred_fix = (prob_fix > args.threshold).astype(np.uint8)
#
#     overlay_ori = make_overlay(post_rgb, pred_ori)
#     overlay_fix = make_overlay(post_rgb, pred_fix)
#
#     err_ori = make_error_map(gt, pred_ori)
#     err_fix = make_error_map(gt, pred_fix)
#
#     metrics_ori = calc_metrics(gt, pred_ori)
#     metrics_fix = calc_metrics(gt, pred_fix)
#
#     save_compare_panel(
#         os.path.join(image_save_dir, "01_original_vs_fixed_att05_comparison.png"),
#         pre_rgb,
#         post_rgb,
#         gt,
#         pred_ori,
#         pred_fix,
#         prob_ori,
#         prob_fix,
#         overlay_ori,
#         overlay_fix,
#         err_ori,
#         err_fix,
#         filename,
#         FIXED_ATT_VALUE,
#     )
#
#     save_metrics_csv(
#         os.path.join(image_save_dir, "metrics_original_vs_fixed_att05.csv"),
#         metrics_ori,
#         metrics_fix,
#         FIXED_ATT_VALUE,
#     )
#
#     with open(os.path.join(image_save_dir, "selected_info.txt"), "w", encoding="utf-8") as f:
#         f.write(f"SELECT_IMAGE_NAME={SELECT_IMAGE_NAME}\n")
#         f.write(f"resolved_index={selected_idx}\n")
#         f.write(f"filename={filename}\n")
#         f.write(f"FIXED_ATT_VALUE={FIXED_ATT_VALUE}\n")
#         f.write(f"threshold={args.threshold}\n")
#         f.write(f"swap_display={args.swap_display}\n")
#         f.write("original_mode=adaptive channel attention\n")
#         f.write("fixed_mode=all TAFM channel attention weights are fixed to 0.5\n")
#
#     print("=========================================================================")
#     print("TAFM 固定注意力 0.5 对照实验完成")
#     print(f"输出目录: {image_save_dir}")
#     print("原始模型指标:", metrics_ori)
#     print("固定0.5指标:", metrics_fix)
#     print("=========================================================================")
#
#
# if __name__ == "__main__":
#     args = parse_args()
#
#     if not torch.cuda.is_available():
#         args.onGPU = False
#
#     print("Called with args:")
#     print(args)
#     print(f"SELECT_IMAGE_NAME = {SELECT_IMAGE_NAME}")
#     print(f"FIXED_ATT_VALUE = {FIXED_ATT_VALUE}")
#
#     visualize_fixed_attention_comparison(args)






#固定阈值为所有通道的均值

# 单张图片：原始 TAFM vs 统一均值权重前向推理对照
# 输出：T1/T2/Label/原模型Pred/UnifiedMean Pred/差异图/指标表/attention统计表

# import os
# import sys
# import random
# import warnings
# from argparse import ArgumentParser
#
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
#
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
# warnings.filterwarnings("ignore", category=UserWarning)
#
# # ==========================================================
# # 更改入口：在这里填写你想要可视化的图片文件名
# # 文件名必须和 /home/LWGANet/GVLM-CD-Processed/test/A 里面的文件名一致
# # 例如：SELECT_IMAGE_NAME = "000123.png"
# # ==========================================================
# SELECT_IMAGE_NAME = "6966.png"
#
# # 对照模式：
# # "unified_mean" = 将每个 stage 的所有通道统一为该 stage 原始自适应 attention 的均值
# # "constant"     = 将所有通道统一为常数 CONSTANT_ATT_VALUE，仅用于敏感性分析
# COMPARE_ATT_MODE = "unified_mean"
# CONSTANT_ATT_VALUE = 0.5
#
# # ==========================================================
# # CPU 环境下自动 map_location
# # ==========================================================
# _original_torch_load = torch.load
#
#
# def _safe_torch_load(*args, **kwargs):
#     if not torch.cuda.is_available():
#         kwargs["map_location"] = "cpu"
#     return _original_torch_load(*args, **kwargs)
#
#
# torch.load = _safe_torch_load
#
# # ==========================================================
# # 项目内部导入
# # ==========================================================
# import dataset_GVLM as myDataLoader
# import Transforms as myTransforms
#
# # 需要把模型文件放到：
# # /home/LWGANet/models/model_with_tafm_debug_unifiedmean.py
# from models.model_with_tafm_debug_unifiedmean import ModelToVisualize as BaseNet_LWGANet_L2
#
#
# def parse_args():
#     parser = ArgumentParser()
#
#     parser.add_argument("--inWidth", type=int, default=256)
#     parser.add_argument("--inHeight", type=int, default=256)
#     parser.add_argument("--num_workers", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--onGPU", default=True, type=lambda x: (str(x).lower() == "true"))
#
#     parser.add_argument(
#         "--weight",
#         default="/home/LWGANet/results/A2Net_LWGANet/L2_DSBN_GVLM_Composite/GVLM-CD-Processed/25.08.14-20.24/best_model.pth",
#         type=str,
#         help="Pretrained weight path",
#     )
#
#     parser.add_argument(
#         "--file_root",
#         default="/home/LWGANet/GVLM-CD-Processed",
#         help="Data directory",
#     )
#
#     parser.add_argument(
#         "--save_root",
#         default="./test_results/TAFM_UnifiedMeanAtt_Compare",
#         type=str,
#         help="Directory to save unified-mean attention comparison results",
#     )
#
#     parser.add_argument(
#         "--threshold",
#         default=0.5,
#         type=float,
#         help="Threshold for binary prediction",
#     )
#
#     # 只修显示，不改推理
#     parser.add_argument(
#         "--swap_display",
#         default=True,
#         type=lambda x: (str(x).lower() == "true"),
#         help="Swap T1/T2 for visualization only. Model input order is unchanged.",
#     )
#
#     return parser.parse_args()
#
#
# def revert_sync_batchnorm(module):
#     res = module
#     if isinstance(module, torch.nn.SyncBatchNorm):
#         res = torch.nn.BatchNorm2d(
#             module.num_features,
#             module.eps,
#             module.momentum,
#             module.affine,
#             module.track_running_stats,
#         )
#         if module.affine:
#             res.weight.data = module.weight.data.clone().detach()
#             res.bias.data = module.bias.data.clone().detach()
#         res.running_mean = module.running_mean
#         res.running_var = module.running_var
#         res.num_batches_tracked = module.num_batches_tracked
#
#     for name, child in module.named_children():
#         res.add_module(name, revert_sync_batchnorm(child))
#
#     return res
#
#
# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)
#
#
# def safe_filename(name, idx):
#     name = str(name).strip()
#     name = os.path.basename(name)
#
#     if name == "" or name.lower() in ["none", "nan"]:
#         name = f"{idx:04d}.png"
#
#     base, ext = os.path.splitext(name)
#
#     if base == "":
#         base = f"{idx:04d}"
#
#     if ext == "":
#         ext = ".png"
#
#     if ext.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
#         ext = ".png"
#
#     return base + ext
#
#
# def _basename_from_item(item):
#     if isinstance(item, str):
#         return os.path.basename(item)
#
#     if isinstance(item, (int, np.integer)):
#         return f"{int(item)}.png"
#
#     if isinstance(item, (tuple, list)) and len(item) > 0:
#         for x in item:
#             if isinstance(x, str) and x.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
#                 return os.path.basename(x)
#         for x in item:
#             if isinstance(x, str) and x.strip() != "":
#                 return os.path.basename(x)
#         return str(item[0])
#
#     if isinstance(item, dict):
#         keys = [
#             "name", "filename", "file_name", "id",
#             "A", "B", "img", "image", "path",
#             "A_path", "img_path", "image_path",
#         ]
#         for k in keys:
#             if k in item:
#                 return os.path.basename(str(item[k]))
#
#     return str(item)
#
#
# def get_filenames_from_dataset_or_fallback(dataset, file_root):
#     print("=> 尝试从 dataset 内部读取真实文件名顺序...")
#
#     candidate_attrs = [
#         "A_paths", "B_paths", "img_paths", "image_paths",
#         "file_paths", "path_list", "file_list", "files",
#         "imgs", "images", "image_list", "img_list",
#         "data_list", "test_files", "file_names", "filenames",
#         "names", "name_list", "ids",
#     ]
#
#     for attr in candidate_attrs:
#         if not hasattr(dataset, attr):
#             continue
#
#         value = getattr(dataset, attr)
#
#         if isinstance(value, list) and len(value) == len(dataset):
#             names = []
#             ok = True
#
#             for idx, item in enumerate(value):
#                 name = _basename_from_item(item)
#                 if name is None:
#                     ok = False
#                     break
#                 names.append(safe_filename(name, idx))
#
#             if ok:
#                 print(f"=> 文件名已从 dataset.{attr} 读取，数量: {len(names)}")
#                 print(f"=> 前 10 个文件名: {names[:10]}")
#                 return names
#
#     print("=> 未从 dataset 内部找到可靠文件名，退回 sorted(test/A)。")
#
#     img_dir = os.path.join(file_root, "test", "A")
#
#     if not os.path.isdir(img_dir):
#         print(f"警告: 找不到 {img_dir}，将使用 index 命名。")
#         return [f"{i:04d}.png" for i in range(len(dataset))]
#
#     names = sorted(
#         [
#             f for f in os.listdir(img_dir)
#             if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
#         ]
#     )
#
#     if len(names) != len(dataset):
#         print("警告: test/A 文件数量与 dataset 长度不一致，使用 index 命名。")
#         return [f"{i:04d}.png" for i in range(len(dataset))]
#
#     names = [safe_filename(n, idx) for idx, n in enumerate(names)]
#
#     print(f"=> sorted(test/A) 文件数量: {len(names)}")
#     print(f"=> 前 10 个文件名: {names[:10]}")
#
#     return names
#
#
# def resolve_selected_index_by_name(names, image_name):
#     wanted = os.path.basename(str(image_name).strip())
#
#     if wanted == "":
#         raise ValueError("SELECT_IMAGE_NAME 为空，请在代码顶部“更改入口”处填写图片文件名。")
#
#     if wanted in names:
#         return names.index(wanted), safe_filename(wanted, names.index(wanted))
#
#     safe_wanted = safe_filename(wanted, 0)
#     if safe_wanted in names:
#         return names.index(safe_wanted), safe_wanted
#
#     wanted_stem, wanted_ext = os.path.splitext(wanted)
#     if wanted_ext == "":
#         matched = [n for n in names if os.path.splitext(n)[0] == wanted_stem]
#         if len(matched) == 1:
#             filename = matched[0]
#             return names.index(filename), filename
#         elif len(matched) > 1:
#             raise ValueError(
#                 f"文件名 stem={wanted_stem} 匹配到多个文件，请填写完整文件名：{matched[:10]}"
#             )
#
#     preview = "\n".join([f"  {i}: {n}" for i, n in enumerate(names[:30])])
#     raise ValueError(
#         f"找不到指定图片文件名: {wanted}\n"
#         f"请确认它位于 test/A 中，并且文件名完全一致。\n"
#         f"当前测试集前 30 个文件名如下：\n{preview}"
#     )
#
#
# def denorm_rgb(x):
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#
#     x = x.detach().cpu().float()
#     x = x * std + mean
#     x = torch.clamp(x, 0, 1)
#     return x.permute(1, 2, 0).numpy()
#
#
# def make_overlay(rgb, pred_bin, alpha=0.45):
#     overlay = rgb.copy()
#     red = np.zeros_like(rgb)
#     red[..., 0] = 1.0
#     overlay[pred_bin == 1] = (1 - alpha) * overlay[pred_bin == 1] + alpha * red[pred_bin == 1]
#     return np.clip(overlay, 0, 1)
#
#
# def make_error_map(gt, pred):
#     h, w = gt.shape
#     out = np.zeros((h, w, 3), dtype=np.float32)
#
#     tp = (gt == 1) & (pred == 1)
#     fp = (gt == 0) & (pred == 1)
#     fn = (gt == 1) & (pred == 0)
#
#     out[tp] = [1, 1, 1]
#     out[fp] = [1, 0, 0]
#     out[fn] = [0, 0.2, 1]
#
#     return out
#
#
# def calc_metrics(gt, pred):
#     gt = gt.astype(np.uint8)
#     pred = pred.astype(np.uint8)
#
#     tp = int(((gt == 1) & (pred == 1)).sum())
#     tn = int(((gt == 0) & (pred == 0)).sum())
#     fp = int(((gt == 0) & (pred == 1)).sum())
#     fn = int(((gt == 1) & (pred == 0)).sum())
#
#     eps = 1e-8
#     precision = tp / (tp + fp + eps)
#     recall = tp / (tp + fn + eps)
#     f1 = 2 * precision * recall / (precision + recall + eps)
#     iou = tp / (tp + fp + fn + eps)
#
#     pred_area = int(pred.sum())
#     gt_area = int(gt.sum())
#
#     return {
#         "tp": tp,
#         "tn": tn,
#         "fp": fp,
#         "fn": fn,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "iou": iou,
#         "pred_area": pred_area,
#         "gt_area": gt_area,
#     }
#
#
# def save_metrics_csv(path, metrics_ori, metrics_cmp, compare_mode):
#     keys = ["tp", "tn", "fp", "fn", "precision", "recall", "f1", "iou", "pred_area", "gt_area"]
#
#     with open(path, "w", encoding="utf-8") as f:
#         f.write("mode,compare_mode," + ",".join(keys) + "\n")
#         f.write("adaptive,original,")
#         f.write(",".join([f"{metrics_ori[k]:.6f}" if isinstance(metrics_ori[k], float) else str(metrics_ori[k]) for k in keys]))
#         f.write("\n")
#
#         f.write(f"{compare_mode},{compare_mode},")
#         f.write(",".join([f"{metrics_cmp[k]:.6f}" if isinstance(metrics_cmp[k], float) else str(metrics_cmp[k]) for k in keys]))
#         f.write("\n")
#
#
# def get_att_stats(debug):
#     rows = []
#
#     for stage_name, d in debug.items():
#         att_raw = d["att_raw"]
#         att = d["att"]
#         att_mean = d.get("att_mean", None)
#
#         rows.append({
#             "stage": stage_name,
#             "raw_min": float(att_raw.min().item()),
#             "raw_max": float(att_raw.max().item()),
#             "raw_mean": float(att_raw.mean().item()),
#             "used_min": float(att.min().item()),
#             "used_max": float(att.max().item()),
#             "used_mean": float(att.mean().item()),
#             "stored_att_mean": float(att_mean.mean().item()) if att_mean is not None else np.nan,
#         })
#
#     return rows
#
#
# def print_att_stats(title, rows):
#     print("==========", title, "==========")
#     for r in rows:
#         print(
#             r["stage"],
#             "raw min/max/mean =",
#             f"{r['raw_min']:.6f}",
#             f"{r['raw_max']:.6f}",
#             f"{r['raw_mean']:.6f}",
#             "| used min/max/mean =",
#             f"{r['used_min']:.6f}",
#             f"{r['used_max']:.6f}",
#             f"{r['used_mean']:.6f}",
#         )
#     print("=====================================")
#
#
# def save_att_stats_csv(path, rows_ori, rows_cmp, compare_mode):
#     with open(path, "w", encoding="utf-8") as f:
#         f.write("mode,stage,raw_min,raw_max,raw_mean,used_min,used_max,used_mean,stored_att_mean\n")
#
#         for r in rows_ori:
#             f.write(
#                 f"adaptive,{r['stage']},"
#                 f"{r['raw_min']:.8f},{r['raw_max']:.8f},{r['raw_mean']:.8f},"
#                 f"{r['used_min']:.8f},{r['used_max']:.8f},{r['used_mean']:.8f},"
#                 f"{r['stored_att_mean']:.8f}\n"
#             )
#
#         for r in rows_cmp:
#             f.write(
#                 f"{compare_mode},{r['stage']},"
#                 f"{r['raw_min']:.8f},{r['raw_max']:.8f},{r['raw_mean']:.8f},"
#                 f"{r['used_min']:.8f},{r['used_max']:.8f},{r['used_mean']:.8f},"
#                 f"{r['stored_att_mean']:.8f}\n"
#             )
#
#
# def save_attention_mean_txt(path, rows_ori, rows_cmp, compare_title):
#     """
#     保存 attention 均值说明，便于直接截图或写进报告。
#     """
#     with open(path, "w", encoding="utf-8") as f:
#         f.write("TAFM attention mean summary\n")
#         f.write("========================================\n")
#         f.write("说明：\n")
#         f.write("raw_mean 表示原始自适应 attention 的通道均值。\n")
#         f.write("used_mean 表示实际用于前向推理的 attention 均值。\n")
#         f.write("在 Unified-Mean 模式下，所有通道统一为 raw_mean，因此 used_min=used_max=used_mean=raw_mean。\n")
#         f.write("========================================\n\n")
#
#         f.write("[Adaptive]\n")
#         for r in rows_ori:
#             f.write(
#                 f"{r['stage']}: "
#                 f"raw_mean={r['raw_mean']:.6f}, "
#                 f"used_min={r['used_min']:.6f}, "
#                 f"used_max={r['used_max']:.6f}, "
#                 f"used_mean={r['used_mean']:.6f}\n"
#             )
#
#         f.write(f"\n[{compare_title}]\n")
#         for r in rows_cmp:
#             f.write(
#                 f"{r['stage']}: "
#                 f"raw_mean={r['raw_mean']:.6f}, "
#                 f"used_min={r['used_min']:.6f}, "
#                 f"used_max={r['used_max']:.6f}, "
#                 f"used_mean={r['used_mean']:.6f}\n"
#             )
#
#
# def make_attention_summary_string(rows_cmp, compare_title):
#     """
#     生成用于放到综合图标题/角标里的 attention 均值字符串。
#     """
#     parts = []
#     for r in rows_cmp:
#         parts.append(
#             f"{r['stage']}: used={r['used_mean']:.3f}"
#         )
#     return f"{compare_title} att mean: " + " | ".join(parts)
#
#
# def save_attention_mean_table_png(path, rows_ori, rows_cmp, compare_title):
#     """
#     保存一张 attention 统计表图片，直观看到 unified_mean 当前设定的均值。
#     """
#     fig, ax = plt.subplots(figsize=(12, 3.8))
#     ax.axis("off")
#
#     col_labels = [
#         "Mode",
#         "Stage",
#         "raw min",
#         "raw max",
#         "raw mean",
#         "used min",
#         "used max",
#         "used mean",
#     ]
#
#     table_data = []
#     for r in rows_ori:
#         table_data.append([
#             "Adaptive",
#             r["stage"],
#             f"{r['raw_min']:.4f}",
#             f"{r['raw_max']:.4f}",
#             f"{r['raw_mean']:.4f}",
#             f"{r['used_min']:.4f}",
#             f"{r['used_max']:.4f}",
#             f"{r['used_mean']:.4f}",
#         ])
#
#     for r in rows_cmp:
#         table_data.append([
#             compare_title,
#             r["stage"],
#             f"{r['raw_min']:.4f}",
#             f"{r['raw_max']:.4f}",
#             f"{r['raw_mean']:.4f}",
#             f"{r['used_min']:.4f}",
#             f"{r['used_max']:.4f}",
#             f"{r['used_mean']:.4f}",
#         ])
#
#     table = ax.table(
#         cellText=table_data,
#         colLabels=col_labels,
#         loc="center",
#         cellLoc="center",
#     )
#
#     table.auto_set_font_size(False)
#     table.set_fontsize(8)
#     table.scale(1.0, 1.25)
#
#     ax.set_title(
#         "TAFM attention statistics: raw adaptive weights vs used weights",
#         fontsize=12,
#         pad=16,
#     )
#
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
#
# def save_compare_panel(
#     path,
#     pre_rgb,
#     post_rgb,
#     gt,
#     pred_ori,
#     pred_cmp,
#     prob_ori,
#     prob_cmp,
#     overlay_ori,
#     overlay_cmp,
#     err_ori,
#     err_cmp,
#     filename,
#     compare_title,
#     attention_summary_text=None,
# ):
#     diff_pred = (pred_ori != pred_cmp).astype(np.uint8)
#
#     plt.figure(figsize=(24, 8))
#
#     titles = [
#         "T1",
#         "T2",
#         "Label",
#         "Pred: adaptive att",
#         f"Pred: {compare_title}",
#         "Prediction difference",
#         "Overlay adaptive",
#         f"Overlay {compare_title}",
#         "Error adaptive\nwhite TP / red FP / blue FN",
#         f"Error {compare_title}\nwhite TP / red FP / blue FN",
#         "Prob adaptive",
#         f"Prob {compare_title}",
#     ]
#
#     images = [
#         pre_rgb,
#         post_rgb,
#         gt,
#         pred_ori,
#         pred_cmp,
#         diff_pred,
#         overlay_ori,
#         overlay_cmp,
#         err_ori,
#         err_cmp,
#         prob_ori,
#         prob_cmp,
#     ]
#
#     for i, (title, img) in enumerate(zip(titles, images)):
#         ax = plt.subplot(2, 6, i + 1)
#
#         if i in [0, 1, 6, 7, 8, 9]:
#             ax.imshow(img)
#         elif i in [10, 11]:
#             ax.imshow(img, cmap="jet", vmin=0, vmax=1)
#         else:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#
#         ax.set_title(title, fontsize=9)
#         ax.axis("off")
#
#     plt.suptitle(f"TAFM adaptive attention vs {compare_title}: {filename}", fontsize=13)
#
#     if attention_summary_text is not None:
#         plt.figtext(
#             0.5,
#             0.015,
#             attention_summary_text,
#             ha="center",
#             fontsize=9,
#         )
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.96])
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def load_model(args):
#     model = BaseNet_LWGANet_L2(pretrained=False)
#
#     if not torch.cuda.is_available():
#         model = revert_sync_batchnorm(model)
#         print("已将模型中的 SyncBatchNorm 自动降级为 BatchNorm2d 以兼容 CPU")
#
#     if not os.path.isfile(args.weight):
#         print(f"错误: 找不到权重文件 {args.weight}")
#         sys.exit(1)
#
#     print(f"=> Loading weights from: {args.weight}")
#     checkpoint = torch.load(args.weight)
#
#     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         state_dict = checkpoint["state_dict"]
#     else:
#         state_dict = checkpoint
#
#     model.load_state_dict(state_dict, strict=False)
#     print("权重加载成功！")
#
#     if args.onGPU and torch.cuda.is_available():
#         model = model.cuda()
#         cudnn.benchmark = True
#
#     model.eval()
#
#     total_params = sum([np.prod(p.size()) for p in model.parameters()])
#     print("Total network parameters:", total_params)
#
#     return model
#
#
# def build_test_dataset(args):
#     dataset_name = "GVLM"
#
#     if os.path.exists(args.file_root):
#         print(f"=> 使用测试数据集路径: {args.file_root}, dataset_name: {dataset_name}")
#     else:
#         raise TypeError(f"数据集路径 {args.file_root} 不存在，请检查！")
#
#     mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
#
#     valDataset = myTransforms.Compose(
#         [
#             myTransforms.Normalize(mean=mean, std=std),
#             myTransforms.Scale(args.inWidth, args.inHeight),
#             myTransforms.ToTensor(),
#         ]
#     )
#
#     test_data = myDataLoader.Dataset(
#         "test",
#         file_root=args.file_root,
#         transform=valDataset,
#         dataset_name=dataset_name,
#     )
#
#     print(f"=> Test dataset length: {len(test_data)}")
#     return test_data
#
#
# def get_selected_sample(test_data, selected_idx):
#     if selected_idx < 0 or selected_idx >= len(test_data):
#         raise IndexError(f"selected_idx={selected_idx} 超出范围，当前 test 集长度为 {len(test_data)}。")
#
#     item = test_data[selected_idx]
#
#     if not isinstance(item, (tuple, list)) or len(item) < 2:
#         raise RuntimeError("Dataset 返回格式不是 (img, target)，请检查 dataset_GVLM。")
#
#     img, target = item[0], item[1]
#
#     if img.ndim == 3:
#         img = img.unsqueeze(0)
#
#     if target.ndim == 2:
#         target = target.unsqueeze(0)
#     elif target.ndim == 3:
#         target = target.unsqueeze(0)
#
#     return img, target
#
#
# def visualize_unified_mean_attention_comparison(args):
#     SEED = 2333
#     torch.manual_seed(SEED)
#     random.seed(SEED)
#     np.random.seed(SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(SEED)
#
#     model = load_model(args)
#     test_data = build_test_dataset(args)
#     names = get_filenames_from_dataset_or_fallback(test_data, args.file_root)
#
#     selected_idx, filename = resolve_selected_index_by_name(names, SELECT_IMAGE_NAME)
#     base = os.path.splitext(filename)[0]
#
#     compare_mode_name = COMPARE_ATT_MODE
#     if COMPARE_ATT_MODE == "constant":
#         compare_mode_name = f"constant_{CONSTANT_ATT_VALUE}"
#
#     image_save_dir = os.path.join(args.save_root, base)
#     ensure_dir(image_save_dir)
#
#     print("=========================================================================")
#     print("开始输出 TAFM 自适应权重 vs 统一均值权重 对照实验")
#     print(f"SELECT_IMAGE_NAME: {SELECT_IMAGE_NAME}")
#     print(f"resolved_index: {selected_idx}")
#     print(f"filename: {filename}")
#     print(f"COMPARE_ATT_MODE: {COMPARE_ATT_MODE}")
#     print(f"CONSTANT_ATT_VALUE: {CONSTANT_ATT_VALUE}")
#     print(f"image_save_dir: {image_save_dir}")
#     print("=========================================================================")
#
#     img, target = get_selected_sample(test_data, selected_idx)
#
#     img_cpu = img.detach().cpu()
#     target_cpu = target.detach().cpu()
#
#     pre_img = img[:, 0:3]
#     post_img = img[:, 3:6]
#
#     if args.onGPU and torch.cuda.is_available():
#         pre_img = pre_img.cuda()
#         post_img = post_img.cuda()
#
#     # 1. 原始自适应注意力模型
#     model.set_tafm_att_mode("adaptive")
#     with torch.no_grad():
#         outputs_ori = model(pre_img, post_img)
#     debug_ori = model.get_tafm_debug()
#     rows_ori = get_att_stats(debug_ori)
#     print_att_stats("Original adaptive attention", rows_ori)
#
#     # 2. 统一均值权重对照，或常数权重敏感性分析
#     if COMPARE_ATT_MODE == "unified_mean":
#         model.set_tafm_att_mode("unified_mean")
#         compare_title = "Unified-Mean"
#     elif COMPARE_ATT_MODE == "constant":
#         model.set_tafm_att_mode("constant", constant_value=CONSTANT_ATT_VALUE)
#         compare_title = f"Unified-{CONSTANT_ATT_VALUE}"
#     else:
#         raise ValueError("COMPARE_ATT_MODE 只能是 unified_mean 或 constant")
#
#     with torch.no_grad():
#         outputs_cmp = model(pre_img, post_img)
#     debug_cmp = model.get_tafm_debug()
#     rows_cmp = get_att_stats(debug_cmp)
#     print_att_stats(compare_title, rows_cmp)
#
#     # 退出时恢复原模型模式，防止后续误用
#     model.set_tafm_att_mode("adaptive")
#
#     prob_ori = outputs_ori[0].detach().cpu().numpy()[0, 0]
#     prob_cmp = outputs_cmp[0].detach().cpu().numpy()[0, 0]
#
#     if args.swap_display:
#         post_rgb = denorm_rgb(img_cpu[0, 0:3])
#         pre_rgb = denorm_rgb(img_cpu[0, 3:6])
#     else:
#         pre_rgb = denorm_rgb(img_cpu[0, 0:3])
#         post_rgb = denorm_rgb(img_cpu[0, 3:6])
#
#     gt_arr = target_cpu[0]
#     if gt_arr.ndim == 3:
#         gt = gt_arr.squeeze(0).numpy()
#     else:
#         gt = gt_arr.numpy()
#     gt = (gt > 0).astype(np.uint8)
#
#     pred_ori = (prob_ori > args.threshold).astype(np.uint8)
#     pred_cmp = (prob_cmp > args.threshold).astype(np.uint8)
#
#     overlay_ori = make_overlay(post_rgb, pred_ori)
#     overlay_cmp = make_overlay(post_rgb, pred_cmp)
#
#     err_ori = make_error_map(gt, pred_ori)
#     err_cmp = make_error_map(gt, pred_cmp)
#
#     metrics_ori = calc_metrics(gt, pred_ori)
#     metrics_cmp = calc_metrics(gt, pred_cmp)
#
#     attention_summary_text = make_attention_summary_string(rows_cmp, compare_title)
#
#     save_compare_panel(
#         os.path.join(image_save_dir, f"01_adaptive_vs_{compare_mode_name}_comparison.png"),
#         pre_rgb,
#         post_rgb,
#         gt,
#         pred_ori,
#         pred_cmp,
#         prob_ori,
#         prob_cmp,
#         overlay_ori,
#         overlay_cmp,
#         err_ori,
#         err_cmp,
#         filename,
#         compare_title,
#         attention_summary_text=attention_summary_text,
#     )
#
#     save_metrics_csv(
#         os.path.join(image_save_dir, f"metrics_adaptive_vs_{compare_mode_name}.csv"),
#         metrics_ori,
#         metrics_cmp,
#         compare_mode_name,
#     )
#
#     save_att_stats_csv(
#         os.path.join(image_save_dir, f"attention_stats_adaptive_vs_{compare_mode_name}.csv"),
#         rows_ori,
#         rows_cmp,
#         compare_mode_name,
#     )
#
#     save_attention_mean_txt(
#         os.path.join(image_save_dir, f"attention_mean_summary_adaptive_vs_{compare_mode_name}.txt"),
#         rows_ori,
#         rows_cmp,
#         compare_title,
#     )
#
#     save_attention_mean_table_png(
#         os.path.join(image_save_dir, f"02_attention_mean_table_adaptive_vs_{compare_mode_name}.png"),
#         rows_ori,
#         rows_cmp,
#         compare_title,
#     )
#
#     with open(os.path.join(image_save_dir, "selected_info.txt"), "w", encoding="utf-8") as f:
#         f.write(f"SELECT_IMAGE_NAME={SELECT_IMAGE_NAME}\n")
#         f.write(f"resolved_index={selected_idx}\n")
#         f.write(f"filename={filename}\n")
#         f.write(f"COMPARE_ATT_MODE={COMPARE_ATT_MODE}\n")
#         f.write(f"CONSTANT_ATT_VALUE={CONSTANT_ATT_VALUE}\n")
#         f.write(f"threshold={args.threshold}\n")
#         f.write(f"swap_display={args.swap_display}\n")
#         f.write("adaptive_mode=original per-channel adaptive attention\n")
#         f.write("unified_mean_mode=all channels are replaced by the original attention mean of that sample/stage\n")
#         f.write("attention_display=terminal print + attention_stats csv + attention_mean_summary txt + attention_mean_table png + figure footer\n")
#
#     print("=========================================================================")
#     print("TAFM 统一均值权重对照实验完成")
#     print(f"输出目录: {image_save_dir}")
#     print("原始模型指标:", metrics_ori)
#     print(f"{compare_title} 指标:", metrics_cmp)
#     print("=========================================================================")
#
#
# if __name__ == "__main__":
#     args = parse_args()
#
#     if not torch.cuda.is_available():
#         args.onGPU = False
#
#     print("Called with args:")
#     print(args)
#     print(f"SELECT_IMAGE_NAME = {SELECT_IMAGE_NAME}")
#     print(f"COMPARE_ATT_MODE = {COMPARE_ATT_MODE}")
#     print(f"CONSTANT_ATT_VALUE = {CONSTANT_ATT_VALUE}")
#
#     visualize_unified_mean_attention_comparison(args)




# 单张图片 TAFM 特征图 + MLP 输入输出统计可视化
# 版本用途：
#   1. 只跑 stage2，节省时间；
#   2. 固定输出 C64/C62/C57/C56/C10/C28；
#   3. 额外输出 ChannelAttention 的 MLP 输入/输出：
#        avg_pool_input / max_pool_input / avg_out / max_out / pre_sigmoid / attention_weight
#   4. 输出 fg/bg ratio 作为后验验证指标，不参与模型推理。



#生成stage 2空间通道权重
# import os
# import sys
# import random
# import warnings
# from argparse import ArgumentParser
#
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F
#
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
# warnings.filterwarnings("ignore", category=UserWarning)
#
# # ==========================================================
# # 更改入口：在这里填写你想要可视化的图片文件名
# # 文件名必须和 /home/LWGANet/GVLM-CD-Processed/val/A 里面的文件名一致
# # 例如：SELECT_IMAGE_NAME = "1127.png"
# # ==========================================================
# SELECT_IMAGE_NAME = "1127.png"
#
# # 今天只输出 stage2：分辨率高，更适合看纹理、边缘、结构响应
# STAGES_TO_SAVE = ["stage2"]
#
# # 固定代表性通道，1-based：
# # C64/C62/C57/C56/C10/C28
# FIXED_CHANNELS_1BASED = [64, 62, 57, 56, 10, 28]
#
# # ==========================================================
# # CPU 环境下自动 map_location
# # ==========================================================
# _original_torch_load = torch.load
#
#
# def _safe_torch_load(*args, **kwargs):
#     if not torch.cuda.is_available():
#         kwargs["map_location"] = "cpu"
#     return _original_torch_load(*args, **kwargs)
#
#
# torch.load = _safe_torch_load
#
# # ==========================================================
# # 项目内部导入
# # ==========================================================
# import dataset_GVLM as myDataLoader
# import Transforms as myTransforms
#
# # 保持与你之前上传代码一致的模型导入路径：
# # 请把新版模型文件保存为：
# # /home/LWGANet/models/model_with_tafm_debug_single.py
# from models.model_with_tafm_debug_single import ModelToVisualize as BaseNet_LWGANet_L2
#
#
# def parse_args():
#     parser = ArgumentParser()
#
#     parser.add_argument("--inWidth", type=int, default=256)
#     parser.add_argument("--inHeight", type=int, default=256)
#     parser.add_argument("--num_workers", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--onGPU", default=True, type=lambda x: (str(x).lower() == "true"))
#
#     parser.add_argument(
#         "--weight",
#         default="/home/LWGANet/results/A2Net_LWGANet/L2_DSBN_GVLM_Composite/GVLM-CD-Processed/25.08.14-20.24/best_model.pth",
#         type=str,
#         help="Pretrained weight path",
#     )
#
#     parser.add_argument(
#         "--file_root",
#         default="/home/LWGANet/GVLM-CD-Processed",
#         help="Data directory",
#     )
#
#     parser.add_argument(
#         "--save_root",
#         default="./test_results/TAFM_SingleImage_MLP_IO_FixedChannels",
#         type=str,
#         help="Directory to save single image TAFM visualizations",
#     )
#
#     parser.add_argument(
#         "--topk",
#         default=10,
#         type=int,
#         help="Top-k high/low weight channels",
#     )
#
#     parser.add_argument(
#         "--shared_percentile",
#         default=99.0,
#         type=float,
#         help="Percentile used as shared vmax for Top/Bottom/fixed single-channel maps. "
#              "Use 99.0 by default to reduce the influence of extreme hot pixels.",
#     )
#
#     parser.add_argument(
#         "--threshold",
#         default=0.5,
#         type=float,
#         help="Threshold for binary prediction",
#     )
#
#     # 只修显示，不改推理
#     parser.add_argument(
#         "--swap_display",
#         default=True,
#         type=lambda x: (str(x).lower() == "true"),
#         help="Swap T1/T2 for visualization only. Model input order is unchanged.",
#     )
#
#     return parser.parse_args()
#
#
# def revert_sync_batchnorm(module):
#     res = module
#     if isinstance(module, torch.nn.SyncBatchNorm):
#         res = torch.nn.BatchNorm2d(
#             module.num_features,
#             module.eps,
#             module.momentum,
#             module.affine,
#             module.track_running_stats,
#         )
#         if module.affine:
#             res.weight.data = module.weight.data.clone().detach()
#             res.bias.data = module.bias.data.clone().detach()
#         res.running_mean = module.running_mean
#         res.running_var = module.running_var
#         res.num_batches_tracked = module.num_batches_tracked
#
#     for name, child in module.named_children():
#         res.add_module(name, revert_sync_batchnorm(child))
#
#     return res
#
#
# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)
#
#
# def safe_filename(name, idx):
#     name = str(name).strip()
#     name = os.path.basename(name)
#
#     if name == "" or name.lower() in ["none", "nan"]:
#         name = f"{idx:04d}.png"
#
#     base, ext = os.path.splitext(name)
#
#     if base == "":
#         base = f"{idx:04d}"
#
#     if ext == "":
#         ext = ".png"
#
#     if ext.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
#         ext = ".png"
#
#     return base + ext
#
#
# def _basename_from_item(item):
#     if isinstance(item, str):
#         return os.path.basename(item)
#
#     if isinstance(item, (int, np.integer)):
#         return f"{int(item)}.png"
#
#     if isinstance(item, (tuple, list)) and len(item) > 0:
#         for x in item:
#             if isinstance(x, str) and x.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
#                 return os.path.basename(x)
#         for x in item:
#             if isinstance(x, str) and x.strip() != "":
#                 return os.path.basename(x)
#         return str(item[0])
#
#     if isinstance(item, dict):
#         keys = [
#             "name", "filename", "file_name", "id",
#             "A", "B", "img", "image", "path",
#             "A_path", "img_path", "image_path",
#         ]
#         for k in keys:
#             if k in item:
#                 return os.path.basename(str(item[k]))
#
#     return str(item)
#
#
# def get_filenames_from_dataset_or_fallback(dataset, file_root):
#     print("=> 尝试从 dataset 内部读取真实文件名顺序...")
#
#     candidate_attrs = [
#         "A_paths", "B_paths", "img_paths", "image_paths",
#         "file_paths", "path_list", "file_list", "files",
#         "imgs", "images", "image_list", "img_list",
#         "data_list", "test_files", "file_names", "filenames",
#         "names", "name_list", "ids",
#     ]
#
#     for attr in candidate_attrs:
#         if not hasattr(dataset, attr):
#             continue
#
#         value = getattr(dataset, attr)
#
#         if isinstance(value, list) and len(value) == len(dataset):
#             names = []
#             ok = True
#
#             for idx, item in enumerate(value):
#                 name = _basename_from_item(item)
#                 if name is None:
#                     ok = False
#                     break
#                 names.append(safe_filename(name, idx))
#
#             if ok:
#                 print(f"=> 文件名已从 dataset.{attr} 读取，数量: {len(names)}")
#                 print(f"=> 前 10 个文件名: {names[:10]}")
#                 return names
#
#     print("=> 未从 dataset 内部找到可靠文件名，退回 sorted(val/A)。")
#
#     img_dir = os.path.join(file_root, "val", "A")
#
#     if not os.path.isdir(img_dir):
#         print(f"警告: 找不到 {img_dir}，将使用 index 命名。")
#         return [f"{i:04d}.png" for i in range(len(dataset))]
#
#     names = sorted(
#         [
#             f for f in os.listdir(img_dir)
#             if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
#         ]
#     )
#
#     if len(names) != len(dataset):
#         print("警告: val/A 文件数量与 dataset 长度不一致，使用 index 命名。")
#         return [f"{i:04d}.png" for i in range(len(dataset))]
#
#     names = [safe_filename(n, idx) for idx, n in enumerate(names)]
#
#     print(f"=> sorted(val/A) 文件数量: {len(names)}")
#     print(f"=> 前 10 个文件名: {names[:10]}")
#
#     return names
#
#
# def resolve_selected_index_by_name(names, image_name):
#     """
#     根据文件名在测试集文件名列表中找到对应 index。
#
#     names: list[str]，测试集文件名顺序
#     image_name: str，想要选择的图片文件名，例如 "1127.png"
#
#     返回:
#     - selected_idx: int
#     - filename: str
#     """
#     wanted = os.path.basename(str(image_name).strip())
#
#     if wanted == "":
#         raise ValueError("SELECT_IMAGE_NAME 为空，请在代码顶部“更改入口”处填写图片文件名。")
#
#     if wanted in names:
#         return names.index(wanted), safe_filename(wanted, names.index(wanted))
#
#     safe_wanted = safe_filename(wanted, 0)
#     if safe_wanted in names:
#         return names.index(safe_wanted), safe_wanted
#
#     wanted_stem, wanted_ext = os.path.splitext(wanted)
#     if wanted_ext == "":
#         matched = [n for n in names if os.path.splitext(n)[0] == wanted_stem]
#         if len(matched) == 1:
#             filename = matched[0]
#             return names.index(filename), filename
#         elif len(matched) > 1:
#             raise ValueError(
#                 f"文件名 stem={wanted_stem} 匹配到多个文件，请填写完整文件名：{matched[:10]}"
#             )
#
#     preview = "\n".join([f"  {i}: {n}" for i, n in enumerate(names[:30])])
#     raise ValueError(
#         f"找不到指定图片文件名: {wanted}\n"
#         f"请确认它位于 val/A 中，并且文件名完全一致。\n"
#         f"当前测试集前 30 个文件名如下：\n{preview}"
#     )
#
#
# def denorm_rgb(x):
#     """
#     x: [3,H,W], normalized tensor.
#     """
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#
#     x = x.detach().cpu().float()
#     x = x * std + mean
#     x = torch.clamp(x, 0, 1)
#     return x.permute(1, 2, 0).numpy()
#
#
# def normalize01(x):
#     x = x.astype(np.float32)
#     return (x - x.min()) / (x.max() - x.min() + 1e-8)
#
#
# def feature_to_heatmap(feat, out_size):
#     """
#     feat: torch.Tensor [C,h,w]
#     output: numpy [H,W]
#     """
#     feat_map = torch.mean(torch.abs(feat), dim=0, keepdim=True).unsqueeze(0)
#     feat_map = F.interpolate(
#         feat_map,
#         size=out_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     feat_map = feat_map[0, 0].detach().cpu().numpy()
#     return normalize01(feat_map)
#
#
# def single_channel_raw_map(feat_ch, out_size, weight=None):
#     """
#     返回未单独归一化的单通道响应图。
#
#     feat_ch: torch.Tensor [h,w]
#     weight: None 或 float。如果给定，则返回 abs(feat_ch) * weight，
#             用于显示通道注意力加权后的实际贡献强度。
#     output: numpy [H,W], raw response map
#     """
#     feat = torch.abs(feat_ch).unsqueeze(0).unsqueeze(0)
#     feat = F.interpolate(
#         feat,
#         size=out_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     heat = feat[0, 0].detach().cpu().numpy().astype(np.float32)
#
#     if weight is not None:
#         heat = heat * float(weight)
#
#     return heat
#
#
# def normalize_with_shared_vmax(x, vmax):
#     """
#     使用统一 vmax 归一化，而不是每张图单独 normalize01。
#     这样不同通道之间的真实幅值差异不会被完全洗掉。
#     """
#     x = x.astype(np.float32)
#     vmax = float(vmax)
#     if vmax <= 1e-8:
#         return np.zeros_like(x, dtype=np.float32)
#     return np.clip(x / vmax, 0.0, 1.0)
#
#
# def compute_shared_vmax(entries, percentile=99.0):
#     """
#     entries: list[dict]，每个 item 包含 raw_map
#     对所有 raw_map 一起计算统一色阶上限。
#     默认使用 99% 分位数，避免少数极端亮点把整体压得太暗。
#     """
#     values = []
#     for item in entries:
#         m = item["raw_map"]
#         values.append(m.reshape(-1))
#
#     if len(values) == 0:
#         return 1.0
#
#     values = np.concatenate(values, axis=0)
#
#     if values.size == 0:
#         return 1.0
#
#     vmax = np.percentile(values, percentile)
#     if vmax <= 1e-8:
#         vmax = values.max() if values.max() > 1e-8 else 1.0
#
#     return float(vmax)
#
#
# def collect_top_bottom_raw_maps(
#     x_fused_i,
#     att_np,
#     out_size,
#     topk=10,
#     weighted=False,
# ):
#     """
#     收集 Top-k 和 Bottom-k 单通道原始响应图，不做单图归一化。
#     """
#     idx_sorted = np.argsort(-att_np)
#     top_idx = idx_sorted[:topk]
#     bottom_idx = idx_sorted[-topk:]
#
#     entries = []
#
#     for rank, ch in enumerate(top_idx, start=1):
#         weight = float(att_np[ch]) if weighted else None
#         heat_raw = single_channel_raw_map(
#             x_fused_i[int(ch)],
#             out_size=out_size,
#             weight=weight,
#         )
#         entries.append(
#             {
#                 "type": "Top",
#                 "rank": rank,
#                 "channel": int(ch),
#                 "weight": float(att_np[ch]),
#                 "raw_map": heat_raw,
#             }
#         )
#
#     for rank, ch in enumerate(bottom_idx, start=1):
#         weight = float(att_np[ch]) if weighted else None
#         heat_raw = single_channel_raw_map(
#             x_fused_i[int(ch)],
#             out_size=out_size,
#             weight=weight,
#         )
#         entries.append(
#             {
#                 "type": "Bottom",
#                 "rank": rank,
#                 "channel": int(ch),
#                 "weight": float(att_np[ch]),
#                 "raw_map": heat_raw,
#             }
#         )
#
#     return entries
#
#
# def group_response_map(x_fused_i, att_np, out_size, topk=10, high=True):
#     """
#     x_fused_i: torch.Tensor [C,h,w]
#     att_np: numpy [C]
#     """
#     idx_sorted = np.argsort(-att_np)
#
#     if high:
#         idx = idx_sorted[:topk]
#     else:
#         idx = idx_sorted[-topk:]
#
#     feat = torch.mean(torch.abs(x_fused_i[idx]), dim=0, keepdim=True).unsqueeze(0)
#     feat = F.interpolate(
#         feat,
#         size=out_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     feat = feat[0, 0].detach().cpu().numpy()
#     return normalize01(feat)
#
#
# def make_overlay(post_rgb, pred_bin, alpha=0.45):
#     overlay = post_rgb.copy()
#     red = np.zeros_like(post_rgb)
#     red[..., 0] = 1.0
#     overlay[pred_bin == 1] = (1 - alpha) * overlay[pred_bin == 1] + alpha * red[pred_bin == 1]
#     return np.clip(overlay, 0, 1)
#
#
# def save_weight_curve(ax, att_np, title="Channel weights"):
#     channels = np.arange(1, len(att_np) + 1)
#     ax.plot(channels, att_np, linewidth=1.5)
#     ax.set_xlim(1, len(att_np))
#     ax.set_ylim(0, 1)
#     ax.set_title(title, fontsize=9)
#     ax.set_xlabel("Channel", fontsize=8)
#     ax.set_ylabel("Weight", fontsize=8)
#     ax.tick_params(labelsize=7)
#     ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
#
#
# def save_basic_panel(path, pre_rgb, post_rgb, gt, before_map, after_map, pred_bin, overlay, att_np, stage_name):
#     """
#     T1 | T2 | Label | before | after | Pred | Overlay | weight curve
#     """
#     plt.figure(figsize=(22, 4))
#
#     titles = [
#         "T1", "T2", "Label",
#         f"{stage_name} before",
#         f"{stage_name} after",
#         "Pred", "Overlay",
#     ]
#
#     images = [
#         pre_rgb, post_rgb, gt,
#         before_map, after_map,
#         pred_bin, overlay,
#     ]
#
#     for i, (title, img) in enumerate(zip(titles, images)):
#         ax = plt.subplot(1, 8, i + 1)
#
#         if "before" in title or "after" in title:
#             ax.imshow(img, cmap="jet", vmin=0, vmax=1)
#         elif img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#
#         ax.set_title(title, fontsize=9)
#         ax.axis("off")
#
#     ax_curve = plt.subplot(1, 8, 8)
#     save_weight_curve(ax_curve, att_np, title=f"{stage_name} weights")
#
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_high_low_panel(path, pre_rgb, post_rgb, gt, high_map, low_map, pred_bin, stage_name):
#     plt.figure(figsize=(16, 4))
#
#     titles = [
#         "T1", "T2", "Label",
#         f"{stage_name} high-weight",
#         f"{stage_name} low-weight",
#         "Pred",
#     ]
#
#     images = [
#         pre_rgb, post_rgb, gt,
#         high_map, low_map,
#         pred_bin,
#     ]
#
#     for i, (title, img) in enumerate(zip(titles, images)):
#         ax = plt.subplot(1, 6, i + 1)
#
#         if "high-weight" in title or "low-weight" in title:
#             ax.imshow(img, cmap="jet", vmin=0, vmax=1)
#         elif img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#
#         ax.set_title(title, fontsize=9)
#         ax.axis("off")
#
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_weight_rank_bar(path, att_np, stage_name, topk=10):
#     idx_sorted = np.argsort(-att_np)
#     top_idx = idx_sorted[:topk]
#     bottom_idx = idx_sorted[-topk:]
#
#     all_idx = np.concatenate([top_idx, bottom_idx])
#     labels = [f"Ch{c + 1}" for c in all_idx]
#     values = att_np[all_idx]
#
#     plt.figure(figsize=(10, 4))
#     plt.bar(np.arange(len(values)), values)
#     plt.xticks(np.arange(len(values)), labels, rotation=45, fontsize=8)
#     plt.ylim(0, 1)
#     plt.ylabel("Attention weight")
#     plt.title(f"{stage_name}: Top-{topk} and Bottom-{topk} channel weights")
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_weights_csv(path, att_np):
#     channel_ids = np.arange(1, len(att_np) + 1)
#     table = np.stack([channel_ids, att_np], axis=1)
#     np.savetxt(
#         path,
#         table,
#         delimiter=",",
#         header="channel,attention_weight",
#         comments="",
#         fmt=["%d", "%.6f"],
#     )
#
#
# def calc_one_channel_stats(x_fused_i, ch_idx, gt):
#     """
#     返回某个单通道在滑坡区/背景区的响应统计。
#     x_fused_i: [C,h,w]
#     ch_idx: int, 0-based
#     gt: numpy [H,W], 0/1
#
#     注意：
#     fg/bg ratio 是后验验证指标，不参与模型推理。
#     """
#     feat_abs = torch.abs(x_fused_i[ch_idx])
#
#     gt_t = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
#     gt_small = F.interpolate(
#         gt_t,
#         size=(feat_abs.shape[0], feat_abs.shape[1]),
#         mode="nearest",
#     )[0, 0]
#
#     fg = gt_small > 0.5
#     bg = gt_small <= 0.5
#
#     eps = 1e-6
#
#     fg_mean = feat_abs[fg].mean().item() if fg.sum() > 0 else 0.0
#     bg_mean = feat_abs[bg].mean().item() if bg.sum() > 0 else 0.0
#     ratio = fg_mean / (bg_mean + eps)
#
#     return {
#         "fg_mean": fg_mean,
#         "bg_mean": bg_mean,
#         "fg_bg_ratio": ratio,
#         "global_mean": feat_abs.mean().item(),
#         "global_max": feat_abs.max().item(),
#     }
#
#
# def calc_pred_metrics(gt, pred_bin):
#     gt_bin = (gt > 0).astype(np.uint8)
#     pred_bin = (pred_bin > 0).astype(np.uint8)
#
#     tp = int(((pred_bin == 1) & (gt_bin == 1)).sum())
#     fp = int(((pred_bin == 1) & (gt_bin == 0)).sum())
#     fn = int(((pred_bin == 0) & (gt_bin == 1)).sum())
#     tn = int(((pred_bin == 0) & (gt_bin == 0)).sum())
#
#     precision = tp / (tp + fp + 1e-6)
#     recall = tp / (tp + fn + 1e-6)
#     f1 = 2 * precision * recall / (precision + recall + 1e-6)
#     iou = tp / (tp + fp + fn + 1e-6)
#
#     return {
#         "tp": tp,
#         "fp": fp,
#         "fn": fn,
#         "tn": tn,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "iou": iou,
#         "pred_area": int(pred_bin.sum()),
#         "gt_area": int(gt_bin.sum()),
#     }
#
#
# def save_top_bottom_stats_csv(path, x_fused_i, att_np, gt, topk=10):
#     idx_sorted = np.argsort(-att_np)
#     top_idx = idx_sorted[:topk]
#     bottom_idx = idx_sorted[-topk:]
#
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(
#             "rank,type,channel,attention_weight,"
#             "fg_mean_response,bg_mean_response,fg_bg_ratio,"
#             "global_mean_response,global_max_response,"
#             "response_observation,interpretation_name\n"
#         )
#
#         for rank, ch in enumerate(top_idx, start=1):
#             stat = calc_one_channel_stats(x_fused_i, int(ch), gt)
#             f.write(
#                 f"{rank},top,Ch{ch + 1},{att_np[ch]:.6f},"
#                 f"{stat['fg_mean']:.6f},{stat['bg_mean']:.6f},{stat['fg_bg_ratio']:.6f},"
#                 f"{stat['global_mean']:.6f},{stat['global_max']:.6f},,\n"
#             )
#
#         for rank, ch in enumerate(bottom_idx, start=1):
#             stat = calc_one_channel_stats(x_fused_i, int(ch), gt)
#             f.write(
#                 f"{rank},bottom,Ch{ch + 1},{att_np[ch]:.6f},"
#                 f"{stat['fg_mean']:.6f},{stat['bg_mean']:.6f},{stat['fg_bg_ratio']:.6f},"
#                 f"{stat['global_mean']:.6f},{stat['global_max']:.6f},,\n"
#             )
#
#
# def save_mlp_io_fixed_channels_csv(
#     path,
#     stage_debug,
#     x_fused_i,
#     att_np,
#     gt,
#     pred_bin,
#     filename="unknown",
#     stage_name="stage2",
# ):
#     """
#     输出固定代表性通道的 MLP 输入/输出与后验验证指标。
#
#     注意：
#     - avg_pool_input / max_pool_input 是 MLP 的真实输入统计摘要；
#     - peak_mean_ratio = max_pool_input / abs(avg_pool_input)，可理解为局部峰值相对整体响应的显著性；
#     - attention_weight 是 MLP + sigmoid 后的输出；
#     - fg/bg ratio 不参与模型推理，只是利用 GT 做后验验证。
#     """
#
#     avg_pool = stage_debug["avg_pool"][0, :, 0, 0].detach().cpu().numpy()
#     max_pool = stage_debug["max_pool"][0, :, 0, 0].detach().cpu().numpy()
#     avg_out = stage_debug["avg_out"][0, :, 0, 0].detach().cpu().numpy()
#     max_out = stage_debug["max_out"][0, :, 0, 0].detach().cpu().numpy()
#     pre_sigmoid = stage_debug["pre_sigmoid"][0, :, 0, 0].detach().cpu().numpy()
#
#     metrics = calc_pred_metrics(gt, pred_bin)
#
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(
#             "filename,stage,channel,"
#             "avg_pool_input,max_pool_input,peak_mean_ratio,"
#             "mlp_avg_out,mlp_max_out,pre_sigmoid,attention_weight,"
#             "fg_mean,bg_mean,fg_bg_ratio,global_mean,global_max,"
#             "tp,fp,fn,tn,precision,recall,f1,iou,pred_area,gt_area,"
#             "manual_observation,interpretation\n"
#         )
#
#         for ch1 in FIXED_CHANNELS_1BASED:
#             ch = ch1 - 1
#
#             if ch < 0 or ch >= x_fused_i.shape[0]:
#                 continue
#
#             stat = calc_one_channel_stats(x_fused_i, ch, gt)
#
#             mean_val = float(avg_pool[ch])
#             max_val = float(max_pool[ch])
#             peak_mean_ratio = max_val / (abs(mean_val) + 1e-6)
#
#             f.write(
#                 f"{filename},{stage_name},Ch{ch1},"
#                 f"{mean_val:.6f},{max_val:.6f},{peak_mean_ratio:.6f},"
#                 f"{avg_out[ch]:.6f},{max_out[ch]:.6f},"
#                 f"{pre_sigmoid[ch]:.6f},{att_np[ch]:.6f},"
#                 f"{stat['fg_mean']:.6f},{stat['bg_mean']:.6f},"
#                 f"{stat['fg_bg_ratio']:.6f},"
#                 f"{stat['global_mean']:.6f},{stat['global_max']:.6f},"
#                 f"{metrics['tp']},{metrics['fp']},{metrics['fn']},{metrics['tn']},"
#                 f"{metrics['precision']:.6f},{metrics['recall']:.6f},"
#                 f"{metrics['f1']:.6f},{metrics['iou']:.6f},"
#                 f"{metrics['pred_area']},{metrics['gt_area']},"
#                 f",\n"
#             )
#
#
# def save_single_channel_top_bottom_panel(
#     path,
#     pre_rgb,
#     post_rgb,
#     gt,
#     pred_bin,
#     x_fused_i,
#     att_np,
#     out_size,
#     stage_name,
#     topk=10,
#     shared_percentile=99.0,
#     weighted=False,
# ):
#     """
#     输出一张 Top/Bottom 单通道综合图，使用统一色阶。
#     """
#     entries = collect_top_bottom_raw_maps(
#         x_fused_i,
#         att_np,
#         out_size=out_size,
#         topk=topk,
#         weighted=weighted,
#     )
#     shared_vmax = compute_shared_vmax(entries, percentile=shared_percentile)
#
#     top_entries = [e for e in entries if e["type"] == "Top"]
#     bottom_entries = [e for e in entries if e["type"] == "Bottom"]
#
#     ncols = 4 + topk
#     plt.figure(figsize=(2.2 * ncols, 6.0))
#
#     base_titles = ["T1", "T2", "Label", "Pred"]
#     base_imgs = [pre_rgb, post_rgb, gt, pred_bin]
#
#     for j, (title, img) in enumerate(zip(base_titles, base_imgs)):
#         ax = plt.subplot(2, ncols, j + 1)
#         if img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#         ax.set_title(title, fontsize=8)
#         ax.axis("off")
#
#     for k, item in enumerate(top_entries):
#         heat = normalize_with_shared_vmax(item["raw_map"], shared_vmax)
#         ch = item["channel"]
#         ax = plt.subplot(2, ncols, 4 + k + 1)
#         ax.imshow(heat, cmap="jet", vmin=0, vmax=1)
#         ax.set_title(
#             f"Top{item['rank']}\nCh{ch + 1}\nw={item['weight']:.3f}",
#             fontsize=7,
#         )
#         ax.axis("off")
#
#     for j in range(4):
#         ax = plt.subplot(2, ncols, ncols + j + 1)
#         ax.axis("off")
#
#     for k, item in enumerate(bottom_entries):
#         heat = normalize_with_shared_vmax(item["raw_map"], shared_vmax)
#         ch = item["channel"]
#         ax = plt.subplot(2, ncols, ncols + 4 + k + 1)
#         ax.imshow(heat, cmap="jet", vmin=0, vmax=1)
#         ax.set_title(
#             f"Bottom{item['rank']}\nCh{ch + 1}\nw={item['weight']:.3f}",
#             fontsize=7,
#         )
#         ax.axis("off")
#
#     mode = "weighted" if weighted else "raw"
#     plt.suptitle(
#         f"{stage_name}: Top-{topk} / Bottom-{topk} single-channel responses "
#         f"({mode}, shared vmax=P{shared_percentile:.1f}={shared_vmax:.4f})",
#         fontsize=11,
#     )
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_fixed_channels_panel(
#     path,
#     pre_rgb,
#     post_rgb,
#     gt,
#     pred_bin,
#     x_fused_i,
#     att_np,
#     out_size,
#     stage_name="stage2",
#     weighted=False,
#     shared_percentile=99.0,
# ):
#     """
#     保存固定通道 C64/C62/C57/C56/C10/C28 的响应图。
#     weighted=False: abs(x_fused)
#     weighted=True : abs(x_fused) * attention_weight
#     """
#
#     entries = []
#
#     for ch1 in FIXED_CHANNELS_1BASED:
#         ch = ch1 - 1
#         if ch < 0 or ch >= x_fused_i.shape[0]:
#             continue
#
#         weight = float(att_np[ch]) if weighted else None
#
#         raw_map = single_channel_raw_map(
#             x_fused_i[ch],
#             out_size=out_size,
#             weight=weight,
#         )
#
#         entries.append(
#             {
#                 "channel_1based": ch1,
#                 "channel_0based": ch,
#                 "weight": float(att_np[ch]),
#                 "raw_map": raw_map,
#             }
#         )
#
#     shared_vmax = compute_shared_vmax(entries, percentile=shared_percentile)
#
#     ncols = 4 + len(entries)
#     plt.figure(figsize=(2.2 * ncols, 3.2))
#
#     base_titles = ["T1", "T2", "Label", "Pred"]
#     base_imgs = [pre_rgb, post_rgb, gt, pred_bin]
#
#     for j, (title, img) in enumerate(zip(base_titles, base_imgs)):
#         ax = plt.subplot(1, ncols, j + 1)
#         if img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#         ax.set_title(title, fontsize=8)
#         ax.axis("off")
#
#     for k, item in enumerate(entries):
#         heat = normalize_with_shared_vmax(item["raw_map"], shared_vmax)
#
#         ax = plt.subplot(1, ncols, 4 + k + 1)
#         ax.imshow(heat, cmap="jet", vmin=0, vmax=1)
#         ax.set_title(
#             f"Ch{item['channel_1based']}\nw={item['weight']:.3f}",
#             fontsize=7,
#         )
#         ax.axis("off")
#
#     mode = "weighted" if weighted else "raw"
#     plt.suptitle(
#         f"{stage_name}: fixed channels {FIXED_CHANNELS_1BASED} "
#         f"({mode}, shared vmax=P{shared_percentile:.1f}={shared_vmax:.4f})",
#         fontsize=10,
#     )
#
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_individual_single_channel_maps(
#     save_dir,
#     x_fused_i,
#     att_np,
#     out_size,
#     topk=10,
#     shared_percentile=99.0,
#     weighted=False,
# ):
#     """
#     每个 Top/Bottom 通道单独保存一张热力图，使用统一色阶。
#     """
#     ensure_dir(save_dir)
#
#     entries = collect_top_bottom_raw_maps(
#         x_fused_i,
#         att_np,
#         out_size=out_size,
#         topk=topk,
#         weighted=weighted,
#     )
#     shared_vmax = compute_shared_vmax(entries, percentile=shared_percentile)
#
#     mode = "weighted" if weighted else "raw"
#
#     with open(os.path.join(save_dir, "_shared_scale_info.txt"), "w", encoding="utf-8") as f:
#         f.write(f"mode={mode}\n")
#         f.write(f"shared_percentile={shared_percentile}\n")
#         f.write(f"shared_vmax={shared_vmax:.8f}\n")
#         f.write("note=All Top/Bottom single-channel maps in this folder use the same vmax.\n")
#
#     for item in entries:
#         heat = normalize_with_shared_vmax(item["raw_map"], shared_vmax)
#         ch = item["channel"]
#         rank = item["rank"]
#         typ = item["type"].lower()
#
#         plt.figure(figsize=(4, 4))
#         plt.imshow(heat, cmap="jet", vmin=0, vmax=1)
#         plt.title(
#             f"{item['type']}{rank}: Ch{ch + 1}, w={item['weight']:.3f}\n"
#             f"{mode}, shared vmax={shared_vmax:.4f}"
#         )
#         plt.axis("off")
#         plt.tight_layout()
#         plt.savefig(
#             os.path.join(
#                 save_dir,
#                 f"{typ}{rank:02d}_Ch{ch + 1}_w{item['weight']:.3f}_{mode}_shared.png",
#             ),
#             dpi=300,
#             bbox_inches="tight",
#         )
#         plt.close()
#
#
# def load_model(args):
#     model = BaseNet_LWGANet_L2(pretrained=False)
#
#     if not torch.cuda.is_available():
#         model = revert_sync_batchnorm(model)
#         print("已将模型中的 SyncBatchNorm 自动降级为 BatchNorm2d 以兼容 CPU")
#
#     if not os.path.isfile(args.weight):
#         print(f"错误: 找不到权重文件 {args.weight}")
#         sys.exit(1)
#
#     print(f"=> Loading weights from: {args.weight}")
#     checkpoint = torch.load(args.weight)
#
#     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         state_dict = checkpoint["state_dict"]
#     else:
#         state_dict = checkpoint
#
#     model.load_state_dict(state_dict, strict=False)
#     print("权重加载成功！")
#
#     if args.onGPU and torch.cuda.is_available():
#         model = model.cuda()
#         cudnn.benchmark = True
#
#     model.eval()
#
#     total_params = sum([np.prod(p.size()) for p in model.parameters()])
#     print("Total network parameters:", total_params)
#
#     return model
#
#
# def build_test_dataset(args):
#     dataset_name = "GVLM"
#
#     if os.path.exists(args.file_root):
#         print(f"=> 使用测试数据集路径: {args.file_root}, dataset_name: {dataset_name}")
#     else:
#         raise TypeError(f"数据集路径 {args.file_root} 不存在，请检查！")
#
#     mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
#
#     valDataset = myTransforms.Compose(
#         [
#             myTransforms.Normalize(mean=mean, std=std),
#             myTransforms.Scale(args.inWidth, args.inHeight),
#             myTransforms.ToTensor(),
#         ]
#     )
#
#     test_data = myDataLoader.Dataset(
#         "val",
#         file_root=args.file_root,
#         transform=valDataset,
#         dataset_name=dataset_name,
#     )
#
#     print(f"=> Test dataset length: {len(test_data)}")
#     return test_data
#
#
# def get_selected_sample(test_data, selected_idx):
#     if selected_idx < 0 or selected_idx >= len(test_data):
#         raise IndexError(
#             f"selected_idx={selected_idx} 超出范围，当前 test 集长度为 {len(test_data)}。"
#         )
#
#     item = test_data[selected_idx]
#
#     if not isinstance(item, (tuple, list)) or len(item) < 2:
#         raise RuntimeError("Dataset 返回格式不是 (img, target)，请检查 dataset_GVLM。")
#
#     img, target = item[0], item[1]
#
#     # img: [6,H,W] -> [1,6,H,W]
#     if img.ndim == 3:
#         img = img.unsqueeze(0)
#
#     # target: [H,W] or [1,H,W] -> [1,H,W] or [1,1,H,W]
#     if target.ndim == 2:
#         target = target.unsqueeze(0)
#     elif target.ndim == 3:
#         target = target.unsqueeze(0)
#
#     return img, target
#
#
# def visualize_selected_single_image(args):
#     SEED = 2333
#     torch.manual_seed(SEED)
#     random.seed(SEED)
#     np.random.seed(SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(SEED)
#
#     model = load_model(args)
#     test_data = build_test_dataset(args)
#     names = get_filenames_from_dataset_or_fallback(test_data, args.file_root)
#
#     selected_idx, filename = resolve_selected_index_by_name(names, SELECT_IMAGE_NAME)
#     base = os.path.splitext(filename)[0]
#
#     image_save_dir = os.path.join(args.save_root, base)
#     ensure_dir(image_save_dir)
#
#     print("=========================================================================")
#     print("开始输出单张图片 TAFM MLP 输入输出 + 固定通道可视化")
#     print(f"SELECT_IMAGE_NAME: {SELECT_IMAGE_NAME}")
#     print(f"resolved_index: {selected_idx}")
#     print(f"filename: {filename}")
#     print(f"image_save_dir: {image_save_dir}")
#     print("=========================================================================")
#
#     img, target = get_selected_sample(test_data, selected_idx)
#
#     img_cpu = img.detach().cpu()
#     target_cpu = target.detach().cpu()
#
#     pre_img = img[:, 0:3]
#     post_img = img[:, 3:6]
#
#     if args.onGPU and torch.cuda.is_available():
#         pre_img = pre_img.cuda()
#         post_img = post_img.cuda()
#
#     with torch.no_grad():
#         outputs = model(pre_img, post_img)
#
#     debug = model.get_tafm_debug()
#
#     # 最终预测使用 m2
#     m2 = outputs[0].detach().cpu().numpy()[0, 0]
#
#     H, W = img_cpu.shape[2], img_cpu.shape[3]
#
#     # 只修显示，不改推理
#     if args.swap_display:
#         post_rgb = denorm_rgb(img_cpu[0, 0:3])
#         pre_rgb = denorm_rgb(img_cpu[0, 3:6])
#     else:
#         pre_rgb = denorm_rgb(img_cpu[0, 0:3])
#         post_rgb = denorm_rgb(img_cpu[0, 3:6])
#
#     # target 兼容 [1,H,W] / [1,1,H,W]
#     gt_arr = target_cpu[0]
#     if gt_arr.ndim == 3:
#         gt = gt_arr.squeeze(0).numpy()
#     else:
#         gt = gt_arr.numpy()
#
#     gt = (gt > 0).astype(np.uint8)
#
#     pred_bin = (m2 > args.threshold).astype(np.uint8)
#     overlay = make_overlay(post_rgb, pred_bin)
#
#     with open(os.path.join(image_save_dir, "selected_info.txt"), "w", encoding="utf-8") as f:
#         f.write(f"SELECT_IMAGE_NAME={SELECT_IMAGE_NAME}\n")
#         f.write(f"resolved_index={selected_idx}\n")
#         f.write(f"filename={filename}\n")
#         f.write(f"topk={args.topk}\n")
#         f.write(f"shared_percentile={args.shared_percentile}\n")
#         f.write(f"threshold={args.threshold}\n")
#         f.write(f"swap_display={args.swap_display}\n")
#         f.write(f"stages={','.join(STAGES_TO_SAVE)}\n")
#         f.write(f"fixed_channels_1based={','.join([str(x) for x in FIXED_CHANNELS_1BASED])}\n")
#         f.write("note=fg_bg_ratio is post-hoc analysis using GT, not model input.\n")
#
#     for stage_name, stage_debug in debug.items():
#         if stage_name not in STAGES_TO_SAVE:
#             continue
#
#         required_keys = ["x_fused", "x_refined", "att", "avg_pool", "max_pool", "avg_out", "max_out", "pre_sigmoid"]
#         missing = [k for k in required_keys if stage_debug.get(k, None) is None]
#         if len(missing) > 0:
#             print(f"警告: {stage_name} 缺少 {missing}，跳过。")
#             continue
#
#         stage_dir = os.path.join(image_save_dir, stage_name)
#         ensure_dir(stage_dir)
#
#         x_fused_i = stage_debug["x_fused"][0].detach().cpu()
#         x_refined_i = stage_debug["x_refined"][0].detach().cpu()
#         att_i = stage_debug["att"][0, :, 0, 0].detach().cpu().numpy()
#
#         before_map = feature_to_heatmap(x_fused_i, out_size=(H, W))
#         after_map = feature_to_heatmap(x_refined_i, out_size=(H, W))
#
#         high_map = group_response_map(
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             topk=args.topk,
#             high=True,
#         )
#
#         low_map = group_response_map(
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             topk=args.topk,
#             high=False,
#         )
#
#         # 1. 基础 panel
#         save_basic_panel(
#             os.path.join(stage_dir, f"01_panel_{stage_name}.png"),
#             pre_rgb,
#             post_rgb,
#             gt,
#             before_map,
#             after_map,
#             pred_bin,
#             overlay,
#             att_i,
#             stage_name,
#         )
#
#         # 2. high/low 平均响应 panel
#         save_high_low_panel(
#             os.path.join(stage_dir, f"02_high_low_response_{stage_name}.png"),
#             pre_rgb,
#             post_rgb,
#             gt,
#             high_map,
#             low_map,
#             pred_bin,
#             stage_name,
#         )
#
#         # 3. Top10/Bottom10 权重柱状图
#         save_weight_rank_bar(
#             os.path.join(stage_dir, f"03_weight_rank_{stage_name}.png"),
#             att_i,
#             stage_name,
#             topk=args.topk,
#         )
#
#         # 4. Top10/Bottom10 单通道响应综合图：统一色阶，原始响应
#         save_single_channel_top_bottom_panel(
#             os.path.join(
#                 stage_dir,
#                 f"04_top{args.topk}_bottom{args.topk}_single_channels_SHARED_RAW_{stage_name}.png",
#             ),
#             pre_rgb,
#             post_rgb,
#             gt,
#             pred_bin,
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             stage_name=stage_name,
#             topk=args.topk,
#             shared_percentile=args.shared_percentile,
#             weighted=False,
#         )
#
#         # 5. Top10/Bottom10 单通道响应综合图：统一色阶，注意力加权后贡献
#         save_single_channel_top_bottom_panel(
#             os.path.join(
#                 stage_dir,
#                 f"05_top{args.topk}_bottom{args.topk}_single_channels_SHARED_WEIGHTED_{stage_name}.png",
#             ),
#             pre_rgb,
#             post_rgb,
#             gt,
#             pred_bin,
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             stage_name=stage_name,
#             topk=args.topk,
#             shared_percentile=args.shared_percentile,
#             weighted=True,
#         )
#
#         # 6. 固定通道 MLP 输入/输出 + 后验验证统计表
#         save_mlp_io_fixed_channels_csv(
#             os.path.join(stage_dir, f"06_mlp_io_fixed_channels_{stage_name}.csv"),
#             stage_debug,
#             x_fused_i,
#             att_i,
#             gt,
#             pred_bin,
#             filename=filename,
#             stage_name=stage_name,
#         )
#
#         # 7. 固定通道原始响应图
#         save_fixed_channels_panel(
#             os.path.join(stage_dir, f"07_fixed_channels_RAW_{stage_name}.png"),
#             pre_rgb,
#             post_rgb,
#             gt,
#             pred_bin,
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             stage_name=stage_name,
#             weighted=False,
#             shared_percentile=args.shared_percentile,
#         )
#
#         # 8. 固定通道注意力加权响应图
#         save_fixed_channels_panel(
#             os.path.join(stage_dir, f"08_fixed_channels_WEIGHTED_{stage_name}.png"),
#             pre_rgb,
#             post_rgb,
#             gt,
#             pred_bin,
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             stage_name=stage_name,
#             weighted=True,
#             shared_percentile=args.shared_percentile,
#         )
#
#         # 9. 每个 Top/Bottom 单通道单独保存：统一色阶，原始响应
#         save_individual_single_channel_maps(
#             os.path.join(stage_dir, "single_channels_shared_raw"),
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             topk=args.topk,
#             shared_percentile=args.shared_percentile,
#             weighted=False,
#         )
#
#         # 10. 每个 Top/Bottom 单通道单独保存：统一色阶，注意力加权后贡献
#         save_individual_single_channel_maps(
#             os.path.join(stage_dir, "single_channels_shared_weighted"),
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             topk=args.topk,
#             shared_percentile=args.shared_percentile,
#             weighted=True,
#         )
#
#         # 11. 所有 64 通道权重表
#         save_weights_csv(
#             os.path.join(stage_dir, f"weights_all64_{stage_name}.csv"),
#             att_i,
#         )
#
#         # 12. Top10/Bottom10 统计表
#         save_top_bottom_stats_csv(
#             os.path.join(stage_dir, f"top{args.topk}_bottom{args.topk}_stats_{stage_name}.csv"),
#             x_fused_i,
#             att_i,
#             gt,
#             topk=args.topk,
#         )
#
#         print(f"=> {stage_name} 保存完成: {stage_dir}")
#
#     print("=========================================================================")
#     print("单张图片 TAFM MLP 输入输出 + 固定通道可视化完成")
#     print(f"输出目录: {image_save_dir}")
#     print("重点查看：stage2/06_mlp_io_fixed_channels_stage2.csv")
#     print("重点查看：stage2/07_fixed_channels_RAW_stage2.png")
#     print("重点查看：stage2/08_fixed_channels_WEIGHTED_stage2.png")
#     print("=========================================================================")
#
#
# if __name__ == "__main__":
#     args = parse_args()
#
#     if not torch.cuda.is_available():
#         args.onGPU = False
#
#     print("Called with args:")
#     print(args)
#     print(f"SELECT_IMAGE_NAME = {SELECT_IMAGE_NAME}")
#     print(f"FIXED_CHANNELS_1BASED = {FIXED_CHANNELS_1BASED}")
#
#     visualize_selected_single_image(args)





# 单张图片 TAFM 特征图 + MLP 输入输出统计可视化
# 版本用途：
#   1. 只跑 stage2，节省时间；
#   2. 固定输出 C64/C62/C57/C56/C10/C28；
#   3. 额外输出 ChannelAttention 的 MLP 输入/输出：
#        avg_pool_input / max_pool_input / avg_out / max_out / pre_sigmoid / attention_weight
#   4. 输出 fg/bg ratio 作为后验验证指标，不参与模型推理。

# import os
# import sys
# import random
# import warnings
# from argparse import ArgumentParser
#
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F
#
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
# warnings.filterwarnings("ignore", category=UserWarning)
#
# # ==========================================================
# # 更改入口：在这里填写你想要可视化的图片文件名
# # 文件名必须和 /home/LWGANet/GVLM-CD-Processed/val/A 里面的文件名一致
# # 例如：SELECT_IMAGE_NAME = "1127.png"
# # ==========================================================
# SELECT_IMAGE_NAME = "712.png"
#
# # 今天只输出 stage2：分辨率高，更适合看纹理、边缘、结构响应
# STAGES_TO_SAVE = ["stage2"]
#
# # 固定代表性通道，1-based：
# # C64/C62/C57/C56/C10/C28
# FIXED_CHANNELS_1BASED = [64, 62, 57, 56, 10, 28]
#
# # ==========================================================
# # CPU 环境下自动 map_location
# # ==========================================================
# _original_torch_load = torch.load
#
#
# def _safe_torch_load(*args, **kwargs):
#     if not torch.cuda.is_available():
#         kwargs["map_location"] = "cpu"
#     return _original_torch_load(*args, **kwargs)
#
#
# torch.load = _safe_torch_load
#
# # ==========================================================
# # 项目内部导入
# # ==========================================================
# import dataset_GVLM as myDataLoader
# import Transforms as myTransforms
#
# # 保持与你之前上传代码一致的模型导入路径：
# # 请把新版模型文件保存为：
# # /home/LWGANet/models/model_with_tafm_debug_single.py
# from models.model_with_tafm_debug_single import ModelToVisualize as BaseNet_LWGANet_L2
#
#
# def parse_args():
#     parser = ArgumentParser()
#
#     parser.add_argument("--inWidth", type=int, default=256)
#     parser.add_argument("--inHeight", type=int, default=256)
#     parser.add_argument("--num_workers", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--onGPU", default=True, type=lambda x: (str(x).lower() == "true"))
#
#     parser.add_argument(
#         "--weight",
#         default="/home/LWGANet/results/A2Net_LWGANet/L2_DSBN_GVLM_Composite/GVLM-CD-Processed/25.08.14-20.24/best_model.pth",
#         type=str,
#         help="Pretrained weight path",
#     )
#
#     parser.add_argument(
#         "--file_root",
#         default="/home/LWGANet/GVLM-CD-Processed",
#         help="Data directory",
#     )
#
#     parser.add_argument(
#         "--save_root",
#         default="./test_results/TAFM_SingleImage_MLP_IO_FixedChannels",
#         type=str,
#         help="Directory to save single image TAFM visualizations",
#     )
#
#     parser.add_argument(
#         "--topk",
#         default=10,
#         type=int,
#         help="Top-k high/low weight channels",
#     )
#
#     parser.add_argument(
#         "--shared_percentile",
#         default=99.0,
#         type=float,
#         help="Percentile used as shared vmax for Top/Bottom/fixed single-channel maps. "
#              "Use 99.0 by default to reduce the influence of extreme hot pixels.",
#     )
#
#     parser.add_argument(
#         "--threshold",
#         default=0.5,
#         type=float,
#         help="Threshold for binary prediction",
#     )
#
#     # 只修显示，不改推理
#     parser.add_argument(
#         "--swap_display",
#         default=True,
#         type=lambda x: (str(x).lower() == "true"),
#         help="Swap T1/T2 for visualization only. Model input order is unchanged.",
#     )
#
#     return parser.parse_args()
#
#
# def revert_sync_batchnorm(module):
#     res = module
#     if isinstance(module, torch.nn.SyncBatchNorm):
#         res = torch.nn.BatchNorm2d(
#             module.num_features,
#             module.eps,
#             module.momentum,
#             module.affine,
#             module.track_running_stats,
#         )
#         if module.affine:
#             res.weight.data = module.weight.data.clone().detach()
#             res.bias.data = module.bias.data.clone().detach()
#         res.running_mean = module.running_mean
#         res.running_var = module.running_var
#         res.num_batches_tracked = module.num_batches_tracked
#
#     for name, child in module.named_children():
#         res.add_module(name, revert_sync_batchnorm(child))
#
#     return res
#
#
# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)
#
#
# def safe_filename(name, idx):
#     name = str(name).strip()
#     name = os.path.basename(name)
#
#     if name == "" or name.lower() in ["none", "nan"]:
#         name = f"{idx:04d}.png"
#
#     base, ext = os.path.splitext(name)
#
#     if base == "":
#         base = f"{idx:04d}"
#
#     if ext == "":
#         ext = ".png"
#
#     if ext.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
#         ext = ".png"
#
#     return base + ext
#
#
# def _basename_from_item(item):
#     if isinstance(item, str):
#         return os.path.basename(item)
#
#     if isinstance(item, (int, np.integer)):
#         return f"{int(item)}.png"
#
#     if isinstance(item, (tuple, list)) and len(item) > 0:
#         for x in item:
#             if isinstance(x, str) and x.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
#                 return os.path.basename(x)
#         for x in item:
#             if isinstance(x, str) and x.strip() != "":
#                 return os.path.basename(x)
#         return str(item[0])
#
#     if isinstance(item, dict):
#         keys = [
#             "name", "filename", "file_name", "id",
#             "A", "B", "img", "image", "path",
#             "A_path", "img_path", "image_path",
#         ]
#         for k in keys:
#             if k in item:
#                 return os.path.basename(str(item[k]))
#
#     return str(item)
#
#
# def get_filenames_from_dataset_or_fallback(dataset, file_root):
#     print("=> 尝试从 dataset 内部读取真实文件名顺序...")
#
#     candidate_attrs = [
#         "A_paths", "B_paths", "img_paths", "image_paths",
#         "file_paths", "path_list", "file_list", "files",
#         "imgs", "images", "image_list", "img_list",
#         "data_list", "test_files", "file_names", "filenames",
#         "names", "name_list", "ids",
#     ]
#
#     for attr in candidate_attrs:
#         if not hasattr(dataset, attr):
#             continue
#
#         value = getattr(dataset, attr)
#
#         if isinstance(value, list) and len(value) == len(dataset):
#             names = []
#             ok = True
#
#             for idx, item in enumerate(value):
#                 name = _basename_from_item(item)
#                 if name is None:
#                     ok = False
#                     break
#                 names.append(safe_filename(name, idx))
#
#             if ok:
#                 print(f"=> 文件名已从 dataset.{attr} 读取，数量: {len(names)}")
#                 print(f"=> 前 10 个文件名: {names[:10]}")
#                 return names
#
#     print("=> 未从 dataset 内部找到可靠文件名，退回 sorted(test/A)。")
#
#     img_dir = os.path.join(file_root, "test", "A")
#
#     if not os.path.isdir(img_dir):
#         print(f"警告: 找不到 {img_dir}，将使用 index 命名。")
#         return [f"{i:04d}.png" for i in range(len(dataset))]
#
#     names = sorted(
#         [
#             f for f in os.listdir(img_dir)
#             if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
#         ]
#     )
#
#     if len(names) != len(dataset):
#         print("警告: val/A 文件数量与 dataset 长度不一致，使用 index 命名。")
#         return [f"{i:04d}.png" for i in range(len(dataset))]
#
#     names = [safe_filename(n, idx) for idx, n in enumerate(names)]
#
#     print(f"=> sorted(test/A) 文件数量: {len(names)}")
#     print(f"=> 前 10 个文件名: {names[:10]}")
#
#     return names
#
#
# def resolve_selected_index_by_name(names, image_name):
#     """
#     根据文件名在测试集文件名列表中找到对应 index。
#
#     names: list[str]，测试集文件名顺序
#     image_name: str，想要选择的图片文件名，例如 "1127.png"
#
#     返回:
#     - selected_idx: int
#     - filename: str
#     """
#     wanted = os.path.basename(str(image_name).strip())
#
#     if wanted == "":
#         raise ValueError("SELECT_IMAGE_NAME 为空，请在代码顶部“更改入口”处填写图片文件名。")
#
#     if wanted in names:
#         return names.index(wanted), safe_filename(wanted, names.index(wanted))
#
#     safe_wanted = safe_filename(wanted, 0)
#     if safe_wanted in names:
#         return names.index(safe_wanted), safe_wanted
#
#     wanted_stem, wanted_ext = os.path.splitext(wanted)
#     if wanted_ext == "":
#         matched = [n for n in names if os.path.splitext(n)[0] == wanted_stem]
#         if len(matched) == 1:
#             filename = matched[0]
#             return names.index(filename), filename
#         elif len(matched) > 1:
#             raise ValueError(
#                 f"文件名 stem={wanted_stem} 匹配到多个文件，请填写完整文件名：{matched[:10]}"
#             )
#
#     preview = "\n".join([f"  {i}: {n}" for i, n in enumerate(names[:30])])
#     raise ValueError(
#         f"找不到指定图片文件名: {wanted}\n"
#         f"请确认它位于 val/A 中，并且文件名完全一致。\n"
#         f"当前测试集前 30 个文件名如下：\n{preview}"
#     )
#
#
# def denorm_rgb(x):
#     """
#     x: [3,H,W], normalized tensor.
#     """
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#
#     x = x.detach().cpu().float()
#     x = x * std + mean
#     x = torch.clamp(x, 0, 1)
#     return x.permute(1, 2, 0).numpy()
#
#
# def normalize01(x):
#     x = x.astype(np.float32)
#     return (x - x.min()) / (x.max() - x.min() + 1e-8)
#
#
# def feature_to_heatmap(feat, out_size):
#     """
#     feat: torch.Tensor [C,h,w]
#     output: numpy [H,W]
#     """
#     feat_map = torch.mean(torch.abs(feat), dim=0, keepdim=True).unsqueeze(0)
#     feat_map = F.interpolate(
#         feat_map,
#         size=out_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     feat_map = feat_map[0, 0].detach().cpu().numpy()
#     return normalize01(feat_map)
#
#
# def single_channel_raw_map(feat_ch, out_size, weight=None):
#     """
#     返回未单独归一化的单通道响应图。
#
#     feat_ch: torch.Tensor [h,w]
#     weight: None 或 float。如果给定，则返回 abs(feat_ch) * weight，
#             用于显示通道注意力加权后的实际贡献强度。
#     output: numpy [H,W], raw response map
#     """
#     feat = torch.abs(feat_ch).unsqueeze(0).unsqueeze(0)
#     feat = F.interpolate(
#         feat,
#         size=out_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     heat = feat[0, 0].detach().cpu().numpy().astype(np.float32)
#
#     if weight is not None:
#         heat = heat * float(weight)
#
#     return heat
#
#
# def normalize_with_shared_vmax(x, vmax):
#     """
#     使用统一 vmax 归一化，而不是每张图单独 normalize01。
#     这样不同通道之间的真实幅值差异不会被完全洗掉。
#     """
#     x = x.astype(np.float32)
#     vmax = float(vmax)
#     if vmax <= 1e-8:
#         return np.zeros_like(x, dtype=np.float32)
#     return np.clip(x / vmax, 0.0, 1.0)
#
#
# def compute_shared_vmax(entries, percentile=99.0):
#     """
#     entries: list[dict]，每个 item 包含 raw_map
#     对所有 raw_map 一起计算统一色阶上限。
#     默认使用 99% 分位数，避免少数极端亮点把整体压得太暗。
#     """
#     values = []
#     for item in entries:
#         m = item["raw_map"]
#         values.append(m.reshape(-1))
#
#     if len(values) == 0:
#         return 1.0
#
#     values = np.concatenate(values, axis=0)
#
#     if values.size == 0:
#         return 1.0
#
#     vmax = np.percentile(values, percentile)
#     if vmax <= 1e-8:
#         vmax = values.max() if values.max() > 1e-8 else 1.0
#
#     return float(vmax)
#
#
# def collect_top_bottom_raw_maps(
#     x_fused_i,
#     att_np,
#     out_size,
#     topk=10,
#     weighted=False,
# ):
#     """
#     收集 Top-k 和 Bottom-k 单通道原始响应图，不做单图归一化。
#     """
#     idx_sorted = np.argsort(-att_np)
#     top_idx = idx_sorted[:topk]
#     bottom_idx = idx_sorted[-topk:]
#
#     entries = []
#
#     for rank, ch in enumerate(top_idx, start=1):
#         weight = float(att_np[ch]) if weighted else None
#         heat_raw = single_channel_raw_map(
#             x_fused_i[int(ch)],
#             out_size=out_size,
#             weight=weight,
#         )
#         entries.append(
#             {
#                 "type": "Top",
#                 "rank": rank,
#                 "channel": int(ch),
#                 "weight": float(att_np[ch]),
#                 "raw_map": heat_raw,
#             }
#         )
#
#     for rank, ch in enumerate(bottom_idx, start=1):
#         weight = float(att_np[ch]) if weighted else None
#         heat_raw = single_channel_raw_map(
#             x_fused_i[int(ch)],
#             out_size=out_size,
#             weight=weight,
#         )
#         entries.append(
#             {
#                 "type": "Bottom",
#                 "rank": rank,
#                 "channel": int(ch),
#                 "weight": float(att_np[ch]),
#                 "raw_map": heat_raw,
#             }
#         )
#
#     return entries
#
#
# def group_response_map(x_fused_i, att_np, out_size, topk=10, high=True):
#     """
#     x_fused_i: torch.Tensor [C,h,w]
#     att_np: numpy [C]
#     """
#     idx_sorted = np.argsort(-att_np)
#
#     if high:
#         idx = idx_sorted[:topk]
#     else:
#         idx = idx_sorted[-topk:]
#
#     feat = torch.mean(torch.abs(x_fused_i[idx]), dim=0, keepdim=True).unsqueeze(0)
#     feat = F.interpolate(
#         feat,
#         size=out_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     feat = feat[0, 0].detach().cpu().numpy()
#     return normalize01(feat)
#
#
# def make_overlay(post_rgb, pred_bin, alpha=0.45):
#     overlay = post_rgb.copy()
#     red = np.zeros_like(post_rgb)
#     red[..., 0] = 1.0
#     overlay[pred_bin == 1] = (1 - alpha) * overlay[pred_bin == 1] + alpha * red[pred_bin == 1]
#     return np.clip(overlay, 0, 1)
#
#
# def save_weight_curve(ax, att_np, title="Channel weights"):
#     channels = np.arange(1, len(att_np) + 1)
#     ax.plot(channels, att_np, linewidth=1.5)
#     ax.set_xlim(1, len(att_np))
#     ax.set_ylim(0, 1)
#     ax.set_title(title, fontsize=9)
#     ax.set_xlabel("Channel", fontsize=8)
#     ax.set_ylabel("Weight", fontsize=8)
#     ax.tick_params(labelsize=7)
#     ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
#
#
# def save_basic_panel(path, pre_rgb, post_rgb, gt, before_map, after_map, pred_bin, overlay, att_np, stage_name):
#     """
#     T1 | T2 | Label | before | after | Pred | Overlay | weight curve
#     """
#     plt.figure(figsize=(22, 4))
#
#     titles = [
#         "T1", "T2", "Label",
#         f"{stage_name} before",
#         f"{stage_name} after",
#         "Pred", "Overlay",
#     ]
#
#     images = [
#         pre_rgb, post_rgb, gt,
#         before_map, after_map,
#         pred_bin, overlay,
#     ]
#
#     for i, (title, img) in enumerate(zip(titles, images)):
#         ax = plt.subplot(1, 8, i + 1)
#
#         if "before" in title or "after" in title:
#             ax.imshow(img, cmap="jet", vmin=0, vmax=1)
#         elif img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#
#         ax.set_title(title, fontsize=9)
#         ax.axis("off")
#
#     ax_curve = plt.subplot(1, 8, 8)
#     save_weight_curve(ax_curve, att_np, title=f"{stage_name} weights")
#
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_high_low_panel(path, pre_rgb, post_rgb, gt, high_map, low_map, pred_bin, stage_name):
#     plt.figure(figsize=(16, 4))
#
#     titles = [
#         "T1", "T2", "Label",
#         f"{stage_name} high-weight",
#         f"{stage_name} low-weight",
#         "Pred",
#     ]
#
#     images = [
#         pre_rgb, post_rgb, gt,
#         high_map, low_map,
#         pred_bin,
#     ]
#
#     for i, (title, img) in enumerate(zip(titles, images)):
#         ax = plt.subplot(1, 6, i + 1)
#
#         if "high-weight" in title or "low-weight" in title:
#             ax.imshow(img, cmap="jet", vmin=0, vmax=1)
#         elif img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#
#         ax.set_title(title, fontsize=9)
#         ax.axis("off")
#
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_weight_rank_bar(path, att_np, stage_name, topk=10):
#     idx_sorted = np.argsort(-att_np)
#     top_idx = idx_sorted[:topk]
#     bottom_idx = idx_sorted[-topk:]
#
#     all_idx = np.concatenate([top_idx, bottom_idx])
#     labels = [f"Ch{c + 1}" for c in all_idx]
#     values = att_np[all_idx]
#
#     plt.figure(figsize=(10, 4))
#     plt.bar(np.arange(len(values)), values)
#     plt.xticks(np.arange(len(values)), labels, rotation=45, fontsize=8)
#     plt.ylim(0, 1)
#     plt.ylabel("Attention weight")
#     plt.title(f"{stage_name}: Top-{topk} and Bottom-{topk} channel weights")
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_weights_csv(path, att_np):
#     channel_ids = np.arange(1, len(att_np) + 1)
#     table = np.stack([channel_ids, att_np], axis=1)
#     np.savetxt(
#         path,
#         table,
#         delimiter=",",
#         header="channel,attention_weight",
#         comments="",
#         fmt=["%d", "%.6f"],
#     )
#
#
# def calc_one_channel_stats(x_fused_i, ch_idx, gt):
#     """
#     返回某个单通道在滑坡区/背景区的响应统计。
#     x_fused_i: [C,h,w]
#     ch_idx: int, 0-based
#     gt: numpy [H,W], 0/1
#
#     注意：
#     fg/bg ratio 是后验验证指标，不参与模型推理。
#     """
#     feat_abs = torch.abs(x_fused_i[ch_idx])
#
#     gt_t = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
#     gt_small = F.interpolate(
#         gt_t,
#         size=(feat_abs.shape[0], feat_abs.shape[1]),
#         mode="nearest",
#     )[0, 0]
#
#     fg = gt_small > 0.5
#     bg = gt_small <= 0.5
#
#     eps = 1e-6
#
#     fg_mean = feat_abs[fg].mean().item() if fg.sum() > 0 else 0.0
#     bg_mean = feat_abs[bg].mean().item() if bg.sum() > 0 else 0.0
#     ratio = fg_mean / (bg_mean + eps)
#
#     return {
#         "fg_mean": fg_mean,
#         "bg_mean": bg_mean,
#         "fg_bg_ratio": ratio,
#         "global_mean": feat_abs.mean().item(),
#         "global_max": feat_abs.max().item(),
#     }
#
#
# def calc_pred_metrics(gt, pred_bin):
#     gt_bin = (gt > 0).astype(np.uint8)
#     pred_bin = (pred_bin > 0).astype(np.uint8)
#
#     tp = int(((pred_bin == 1) & (gt_bin == 1)).sum())
#     fp = int(((pred_bin == 1) & (gt_bin == 0)).sum())
#     fn = int(((pred_bin == 0) & (gt_bin == 1)).sum())
#     tn = int(((pred_bin == 0) & (gt_bin == 0)).sum())
#
#     precision = tp / (tp + fp + 1e-6)
#     recall = tp / (tp + fn + 1e-6)
#     f1 = 2 * precision * recall / (precision + recall + 1e-6)
#     iou = tp / (tp + fp + fn + 1e-6)
#
#     return {
#         "tp": tp,
#         "fp": fp,
#         "fn": fn,
#         "tn": tn,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "iou": iou,
#         "pred_area": int(pred_bin.sum()),
#         "gt_area": int(gt_bin.sum()),
#     }
#
#
# def save_top_bottom_stats_csv(path, x_fused_i, att_np, gt, topk=10):
#     idx_sorted = np.argsort(-att_np)
#     top_idx = idx_sorted[:topk]
#     bottom_idx = idx_sorted[-topk:]
#
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(
#             "rank,type,channel,attention_weight,"
#             "fg_mean_response,bg_mean_response,fg_bg_ratio,"
#             "global_mean_response,global_max_response,"
#             "response_observation,interpretation_name\n"
#         )
#
#         for rank, ch in enumerate(top_idx, start=1):
#             stat = calc_one_channel_stats(x_fused_i, int(ch), gt)
#             f.write(
#                 f"{rank},top,Ch{ch + 1},{att_np[ch]:.6f},"
#                 f"{stat['fg_mean']:.6f},{stat['bg_mean']:.6f},{stat['fg_bg_ratio']:.6f},"
#                 f"{stat['global_mean']:.6f},{stat['global_max']:.6f},,\n"
#             )
#
#         for rank, ch in enumerate(bottom_idx, start=1):
#             stat = calc_one_channel_stats(x_fused_i, int(ch), gt)
#             f.write(
#                 f"{rank},bottom,Ch{ch + 1},{att_np[ch]:.6f},"
#                 f"{stat['fg_mean']:.6f},{stat['bg_mean']:.6f},{stat['fg_bg_ratio']:.6f},"
#                 f"{stat['global_mean']:.6f},{stat['global_max']:.6f},,\n"
#             )
#
#
# def save_mlp_io_fixed_channels_csv(
#     path,
#     stage_debug,
#     x_fused_i,
#     att_np,
#     gt,
#     pred_bin,
#     filename="unknown",
#     stage_name="stage2",
# ):
#     """
#     输出固定代表性通道的 MLP 输入/输出与后验验证指标。
#
#     注意：
#     - avg_pool_input / max_pool_input 是 MLP 的真实输入统计摘要；
#     - peak_mean_ratio = max_pool_input / abs(avg_pool_input)，可理解为局部峰值相对整体响应的显著性；
#     - attention_weight 是 MLP + sigmoid 后的输出；
#     - fg/bg ratio 不参与模型推理，只是利用 GT 做后验验证。
#     """
#
#     avg_pool = stage_debug["avg_pool"][0, :, 0, 0].detach().cpu().numpy()
#     max_pool = stage_debug["max_pool"][0, :, 0, 0].detach().cpu().numpy()
#     avg_out = stage_debug["avg_out"][0, :, 0, 0].detach().cpu().numpy()
#     max_out = stage_debug["max_out"][0, :, 0, 0].detach().cpu().numpy()
#     pre_sigmoid = stage_debug["pre_sigmoid"][0, :, 0, 0].detach().cpu().numpy()
#
#     metrics = calc_pred_metrics(gt, pred_bin)
#
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(
#             "filename,stage,channel,"
#             "avg_pool_input,max_pool_input,peak_mean_ratio,"
#             "mlp_avg_out,mlp_max_out,pre_sigmoid,attention_weight,"
#             "fg_mean,bg_mean,fg_bg_ratio,global_mean,global_max,"
#             "tp,fp,fn,tn,precision,recall,f1,iou,pred_area,gt_area,"
#             "manual_observation,interpretation\n"
#         )
#
#         for ch1 in FIXED_CHANNELS_1BASED:
#             ch = ch1 - 1
#
#             if ch < 0 or ch >= x_fused_i.shape[0]:
#                 continue
#
#             stat = calc_one_channel_stats(x_fused_i, ch, gt)
#
#             mean_val = float(avg_pool[ch])
#             max_val = float(max_pool[ch])
#             peak_mean_ratio = max_val / (abs(mean_val) + 1e-6)
#
#             f.write(
#                 f"{filename},{stage_name},Ch{ch1},"
#                 f"{mean_val:.6f},{max_val:.6f},{peak_mean_ratio:.6f},"
#                 f"{avg_out[ch]:.6f},{max_out[ch]:.6f},"
#                 f"{pre_sigmoid[ch]:.6f},{att_np[ch]:.6f},"
#                 f"{stat['fg_mean']:.6f},{stat['bg_mean']:.6f},"
#                 f"{stat['fg_bg_ratio']:.6f},"
#                 f"{stat['global_mean']:.6f},{stat['global_max']:.6f},"
#                 f"{metrics['tp']},{metrics['fp']},{metrics['fn']},{metrics['tn']},"
#                 f"{metrics['precision']:.6f},{metrics['recall']:.6f},"
#                 f"{metrics['f1']:.6f},{metrics['iou']:.6f},"
#                 f"{metrics['pred_area']},{metrics['gt_area']},"
#                 f",\n"
#             )
#
#
#
# def save_mlp_io_all64_channels_csv(
#     path,
#     stage_debug,
#     x_fused_i,
#     att_np,
#     gt,
#     pred_bin,
#     filename="unknown",
#     stage_name="stage2",
# ):
#     """
#     输出全部 64 个通道的 MLP 输入/输出与后验验证指标。
#
#     - avg_pool_input / max_pool_input: MLP 真实输入统计摘要；
#     - peak_mean_ratio: max / abs(mean)，用于描述局部峰值显著性；
#     - mlp_avg_out / mlp_max_out / pre_sigmoid / attention_weight: MLP 映射过程和输出；
#     - fg_mean / bg_mean / fg_bg_ratio: 用 GT 后验验证响应是否集中于滑坡区，不参与模型推理；
#     - precision / recall / f1 / iou: 当前样本预测指标。
#     """
#
#     avg_pool = stage_debug["avg_pool"][0, :, 0, 0].detach().cpu().numpy()
#     max_pool = stage_debug["max_pool"][0, :, 0, 0].detach().cpu().numpy()
#     avg_out = stage_debug["avg_out"][0, :, 0, 0].detach().cpu().numpy()
#     max_out = stage_debug["max_out"][0, :, 0, 0].detach().cpu().numpy()
#     pre_sigmoid = stage_debug["pre_sigmoid"][0, :, 0, 0].detach().cpu().numpy()
#
#     metrics = calc_pred_metrics(gt, pred_bin)
#     num_channels = int(x_fused_i.shape[0])
#
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(
#             "filename,stage,channel,"
#             "avg_pool_input,max_pool_input,peak_mean_ratio,"
#             "mlp_avg_out,mlp_max_out,pre_sigmoid,attention_weight,"
#             "fg_mean,bg_mean,fg_bg_ratio,global_mean,global_max,"
#             "tp,fp,fn,tn,precision,recall,f1,iou,pred_area,gt_area,"
#             "suggested_role,manual_observation,interpretation\n"
#         )
#
#         for ch in range(num_channels):
#             stat = calc_one_channel_stats(x_fused_i, ch, gt)
#
#             mean_val = float(avg_pool[ch])
#             max_val = float(max_pool[ch])
#             peak_mean_ratio = max_val / (abs(mean_val) + 1e-6)
#
#             f.write(
#                 f"{filename},{stage_name},Ch{ch + 1},"
#                 f"{mean_val:.6f},{max_val:.6f},{peak_mean_ratio:.6f},"
#                 f"{avg_out[ch]:.6f},{max_out[ch]:.6f},"
#                 f"{pre_sigmoid[ch]:.6f},{att_np[ch]:.6f},"
#                 f"{stat['fg_mean']:.6f},{stat['bg_mean']:.6f},"
#                 f"{stat['fg_bg_ratio']:.6f},"
#                 f"{stat['global_mean']:.6f},{stat['global_max']:.6f},"
#                 f"{metrics['tp']},{metrics['fp']},{metrics['fn']},{metrics['tn']},"
#                 f"{metrics['precision']:.6f},{metrics['recall']:.6f},"
#                 f"{metrics['f1']:.6f},{metrics['iou']:.6f},"
#                 f"{metrics['pred_area']},{metrics['gt_area']},"
#                 f",,\n"
#             )
#
#
# def save_all64_channel_pages(
#     save_dir,
#     x_fused_i,
#     att_np,
#     out_size,
#     stage_name="stage2",
#     weighted=False,
#     shared_percentile=99.0,
#     channels_per_page=16,
# ):
#     """
#     把全部 64 个通道都打印成热力图，并使用统一色阶。
#
#     输出：
#     - all64_RAW_page01_stage2.png ... page04
#     - all64_WEIGHTED_page01_stage2.png ... page04
#     - _shared_scale_info.txt
#
#     weighted=False: abs(x_fused)
#     weighted=True : abs(x_fused) * attention_weight
#     """
#
#     ensure_dir(save_dir)
#
#     entries = []
#     num_channels = int(x_fused_i.shape[0])
#
#     for ch in range(num_channels):
#         weight = float(att_np[ch]) if weighted else None
#         raw_map = single_channel_raw_map(
#             x_fused_i[ch],
#             out_size=out_size,
#             weight=weight,
#         )
#         entries.append(
#             {
#                 "channel": ch,
#                 "channel_1based": ch + 1,
#                 "weight": float(att_np[ch]),
#                 "raw_map": raw_map,
#             }
#         )
#
#     shared_vmax = compute_shared_vmax(entries, percentile=shared_percentile)
#     mode = "WEIGHTED" if weighted else "RAW"
#
#     with open(os.path.join(save_dir, "_shared_scale_info.txt"), "w", encoding="utf-8") as f:
#         f.write(f"stage_name={stage_name}\n")
#         f.write(f"mode={mode}\n")
#         f.write(f"shared_percentile={shared_percentile}\n")
#         f.write(f"shared_vmax={shared_vmax:.8f}\n")
#         f.write("note=All 64 channel maps in this folder use the same color scale.\n")
#         f.write("note2=WEIGHTED maps are abs(channel_feature) multiplied by attention_weight.\n")
#
#     num_pages = int(np.ceil(num_channels / channels_per_page))
#
#     for page_idx in range(num_pages):
#         page_entries = entries[
#             page_idx * channels_per_page : (page_idx + 1) * channels_per_page
#         ]
#
#         n = len(page_entries)
#         ncols = 4
#         nrows = int(np.ceil(n / ncols))
#
#         plt.figure(figsize=(3.0 * ncols, 3.0 * nrows))
#
#         for i, item in enumerate(page_entries):
#             heat = normalize_with_shared_vmax(item["raw_map"], shared_vmax)
#             ax = plt.subplot(nrows, ncols, i + 1)
#             ax.imshow(heat, cmap="jet", vmin=0, vmax=1)
#             ax.set_title(
#                 f"Ch{item['channel_1based']}  w={item['weight']:.3f}",
#                 fontsize=8,
#             )
#             ax.axis("off")
#
#         plt.suptitle(
#             f"{stage_name}: all 64 channels, {mode}, "
#             f"page {page_idx + 1}/{num_pages}, shared vmax=P{shared_percentile:.1f}={shared_vmax:.4f}",
#             fontsize=11,
#         )
#         plt.tight_layout()
#         plt.savefig(
#             os.path.join(
#                 save_dir,
#                 f"all64_{mode}_page{page_idx + 1:02d}_{stage_name}.png",
#             ),
#             dpi=300,
#             bbox_inches="tight",
#         )
#         plt.close()
#
#
#
# def save_single_channel_top_bottom_panel(
#     path,
#     pre_rgb,
#     post_rgb,
#     gt,
#     pred_bin,
#     x_fused_i,
#     att_np,
#     out_size,
#     stage_name,
#     topk=10,
#     shared_percentile=99.0,
#     weighted=False,
# ):
#     """
#     输出一张 Top/Bottom 单通道综合图，使用统一色阶。
#     """
#     entries = collect_top_bottom_raw_maps(
#         x_fused_i,
#         att_np,
#         out_size=out_size,
#         topk=topk,
#         weighted=weighted,
#     )
#     shared_vmax = compute_shared_vmax(entries, percentile=shared_percentile)
#
#     top_entries = [e for e in entries if e["type"] == "Top"]
#     bottom_entries = [e for e in entries if e["type"] == "Bottom"]
#
#     ncols = 4 + topk
#     plt.figure(figsize=(2.2 * ncols, 6.0))
#
#     base_titles = ["T1", "T2", "Label", "Pred"]
#     base_imgs = [pre_rgb, post_rgb, gt, pred_bin]
#
#     for j, (title, img) in enumerate(zip(base_titles, base_imgs)):
#         ax = plt.subplot(2, ncols, j + 1)
#         if img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#         ax.set_title(title, fontsize=8)
#         ax.axis("off")
#
#     for k, item in enumerate(top_entries):
#         heat = normalize_with_shared_vmax(item["raw_map"], shared_vmax)
#         ch = item["channel"]
#         ax = plt.subplot(2, ncols, 4 + k + 1)
#         ax.imshow(heat, cmap="jet", vmin=0, vmax=1)
#         ax.set_title(
#             f"Top{item['rank']}\nCh{ch + 1}\nw={item['weight']:.3f}",
#             fontsize=7,
#         )
#         ax.axis("off")
#
#     for j in range(4):
#         ax = plt.subplot(2, ncols, ncols + j + 1)
#         ax.axis("off")
#
#     for k, item in enumerate(bottom_entries):
#         heat = normalize_with_shared_vmax(item["raw_map"], shared_vmax)
#         ch = item["channel"]
#         ax = plt.subplot(2, ncols, ncols + 4 + k + 1)
#         ax.imshow(heat, cmap="jet", vmin=0, vmax=1)
#         ax.set_title(
#             f"Bottom{item['rank']}\nCh{ch + 1}\nw={item['weight']:.3f}",
#             fontsize=7,
#         )
#         ax.axis("off")
#
#     mode = "weighted" if weighted else "raw"
#     plt.suptitle(
#         f"{stage_name}: Top-{topk} / Bottom-{topk} single-channel responses "
#         f"({mode}, shared vmax=P{shared_percentile:.1f}={shared_vmax:.4f})",
#         fontsize=11,
#     )
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_fixed_channels_panel(
#     path,
#     pre_rgb,
#     post_rgb,
#     gt,
#     pred_bin,
#     x_fused_i,
#     att_np,
#     out_size,
#     stage_name="stage2",
#     weighted=False,
#     shared_percentile=99.0,
# ):
#     """
#     保存固定通道 C64/C62/C57/C56/C10/C28 的响应图。
#     weighted=False: abs(x_fused)
#     weighted=True : abs(x_fused) * attention_weight
#     """
#
#     entries = []
#
#     for ch1 in FIXED_CHANNELS_1BASED:
#         ch = ch1 - 1
#         if ch < 0 or ch >= x_fused_i.shape[0]:
#             continue
#
#         weight = float(att_np[ch]) if weighted else None
#
#         raw_map = single_channel_raw_map(
#             x_fused_i[ch],
#             out_size=out_size,
#             weight=weight,
#         )
#
#         entries.append(
#             {
#                 "channel_1based": ch1,
#                 "channel_0based": ch,
#                 "weight": float(att_np[ch]),
#                 "raw_map": raw_map,
#             }
#         )
#
#     shared_vmax = compute_shared_vmax(entries, percentile=shared_percentile)
#
#     ncols = 4 + len(entries)
#     plt.figure(figsize=(2.2 * ncols, 3.2))
#
#     base_titles = ["T1", "T2", "Label", "Pred"]
#     base_imgs = [pre_rgb, post_rgb, gt, pred_bin]
#
#     for j, (title, img) in enumerate(zip(base_titles, base_imgs)):
#         ax = plt.subplot(1, ncols, j + 1)
#         if img.ndim == 2:
#             ax.imshow(img, cmap="gray", vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#         ax.set_title(title, fontsize=8)
#         ax.axis("off")
#
#     for k, item in enumerate(entries):
#         heat = normalize_with_shared_vmax(item["raw_map"], shared_vmax)
#
#         ax = plt.subplot(1, ncols, 4 + k + 1)
#         ax.imshow(heat, cmap="jet", vmin=0, vmax=1)
#         ax.set_title(
#             f"Ch{item['channel_1based']}\nw={item['weight']:.3f}",
#             fontsize=7,
#         )
#         ax.axis("off")
#
#     mode = "weighted" if weighted else "raw"
#     plt.suptitle(
#         f"{stage_name}: fixed channels {FIXED_CHANNELS_1BASED} "
#         f"({mode}, shared vmax=P{shared_percentile:.1f}={shared_vmax:.4f})",
#         fontsize=10,
#     )
#
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#
#
# def save_individual_single_channel_maps(
#     save_dir,
#     x_fused_i,
#     att_np,
#     out_size,
#     topk=10,
#     shared_percentile=99.0,
#     weighted=False,
# ):
#     """
#     每个 Top/Bottom 通道单独保存一张热力图，使用统一色阶。
#     """
#     ensure_dir(save_dir)
#
#     entries = collect_top_bottom_raw_maps(
#         x_fused_i,
#         att_np,
#         out_size=out_size,
#         topk=topk,
#         weighted=weighted,
#     )
#     shared_vmax = compute_shared_vmax(entries, percentile=shared_percentile)
#
#     mode = "weighted" if weighted else "raw"
#
#     with open(os.path.join(save_dir, "_shared_scale_info.txt"), "w", encoding="utf-8") as f:
#         f.write(f"mode={mode}\n")
#         f.write(f"shared_percentile={shared_percentile}\n")
#         f.write(f"shared_vmax={shared_vmax:.8f}\n")
#         f.write("note=All Top/Bottom single-channel maps in this folder use the same vmax.\n")
#
#     for item in entries:
#         heat = normalize_with_shared_vmax(item["raw_map"], shared_vmax)
#         ch = item["channel"]
#         rank = item["rank"]
#         typ = item["type"].lower()
#
#         plt.figure(figsize=(4, 4))
#         plt.imshow(heat, cmap="jet", vmin=0, vmax=1)
#         plt.title(
#             f"{item['type']}{rank}: Ch{ch + 1}, w={item['weight']:.3f}\n"
#             f"{mode}, shared vmax={shared_vmax:.4f}"
#         )
#         plt.axis("off")
#         plt.tight_layout()
#         plt.savefig(
#             os.path.join(
#                 save_dir,
#                 f"{typ}{rank:02d}_Ch{ch + 1}_w{item['weight']:.3f}_{mode}_shared.png",
#             ),
#             dpi=300,
#             bbox_inches="tight",
#         )
#         plt.close()
#
#
# def load_model(args):
#     model = BaseNet_LWGANet_L2(pretrained=False)
#
#     if not torch.cuda.is_available():
#         model = revert_sync_batchnorm(model)
#         print("已将模型中的 SyncBatchNorm 自动降级为 BatchNorm2d 以兼容 CPU")
#
#     if not os.path.isfile(args.weight):
#         print(f"错误: 找不到权重文件 {args.weight}")
#         sys.exit(1)
#
#     print(f"=> Loading weights from: {args.weight}")
#     checkpoint = torch.load(args.weight)
#
#     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         state_dict = checkpoint["state_dict"]
#     else:
#         state_dict = checkpoint
#
#     model.load_state_dict(state_dict, strict=False)
#     print("权重加载成功！")
#
#     if args.onGPU and torch.cuda.is_available():
#         model = model.cuda()
#         cudnn.benchmark = True
#
#     model.eval()
#
#     total_params = sum([np.prod(p.size()) for p in model.parameters()])
#     print("Total network parameters:", total_params)
#
#     return model
#
#
# def build_test_dataset(args):
#     dataset_name = "GVLM"
#
#     if os.path.exists(args.file_root):
#         print(f"=> 使用测试数据集路径: {args.file_root}, dataset_name: {dataset_name}")
#     else:
#         raise TypeError(f"数据集路径 {args.file_root} 不存在，请检查！")
#
#     mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
#
#     valDataset = myTransforms.Compose(
#         [
#             myTransforms.Normalize(mean=mean, std=std),
#             myTransforms.Scale(args.inWidth, args.inHeight),
#             myTransforms.ToTensor(),
#         ]
#     )
#
#     test_data = myDataLoader.Dataset(
#         "test",
#         file_root=args.file_root,
#         transform=valDataset,
#         dataset_name=dataset_name,
#     )
#
#     print(f"=> Test dataset length: {len(test_data)}")
#     return test_data
#
#
# def get_selected_sample(test_data, selected_idx):
#     if selected_idx < 0 or selected_idx >= len(test_data):
#         raise IndexError(
#             f"selected_idx={selected_idx} 超出范围，当前 test 集长度为 {len(test_data)}。"
#         )
#
#     item = test_data[selected_idx]
#
#     if not isinstance(item, (tuple, list)) or len(item) < 2:
#         raise RuntimeError("Dataset 返回格式不是 (img, target)，请检查 dataset_GVLM。")
#
#     img, target = item[0], item[1]
#
#     # img: [6,H,W] -> [1,6,H,W]
#     if img.ndim == 3:
#         img = img.unsqueeze(0)
#
#     # target: [H,W] or [1,H,W] -> [1,H,W] or [1,1,H,W]
#     if target.ndim == 2:
#         target = target.unsqueeze(0)
#     elif target.ndim == 3:
#         target = target.unsqueeze(0)
#
#     return img, target
#
#
# def visualize_selected_single_image(args):
#     SEED = 2333
#     torch.manual_seed(SEED)
#     random.seed(SEED)
#     np.random.seed(SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(SEED)
#
#     model = load_model(args)
#     test_data = build_test_dataset(args)
#     names = get_filenames_from_dataset_or_fallback(test_data, args.file_root)
#
#     selected_idx, filename = resolve_selected_index_by_name(names, SELECT_IMAGE_NAME)
#     base = os.path.splitext(filename)[0]
#
#     image_save_dir = os.path.join(args.save_root, base)
#     ensure_dir(image_save_dir)
#
#     print("=========================================================================")
#     print("开始输出单张图片 TAFM MLP 输入输出 + 固定通道可视化")
#     print(f"SELECT_IMAGE_NAME: {SELECT_IMAGE_NAME}")
#     print(f"resolved_index: {selected_idx}")
#     print(f"filename: {filename}")
#     print(f"image_save_dir: {image_save_dir}")
#     print("=========================================================================")
#
#     img, target = get_selected_sample(test_data, selected_idx)
#
#     img_cpu = img.detach().cpu()
#     target_cpu = target.detach().cpu()
#
#     pre_img = img[:, 0:3]
#     post_img = img[:, 3:6]
#
#     if args.onGPU and torch.cuda.is_available():
#         pre_img = pre_img.cuda()
#         post_img = post_img.cuda()
#
#     with torch.no_grad():
#         outputs = model(pre_img, post_img)
#
#     debug = model.get_tafm_debug()
#
#     # 最终预测使用 m2
#     m2 = outputs[0].detach().cpu().numpy()[0, 0]
#
#     H, W = img_cpu.shape[2], img_cpu.shape[3]
#
#     # 只修显示，不改推理
#     if args.swap_display:
#         post_rgb = denorm_rgb(img_cpu[0, 0:3])
#         pre_rgb = denorm_rgb(img_cpu[0, 3:6])
#     else:
#         pre_rgb = denorm_rgb(img_cpu[0, 0:3])
#         post_rgb = denorm_rgb(img_cpu[0, 3:6])
#
#     # target 兼容 [1,H,W] / [1,1,H,W]
#     gt_arr = target_cpu[0]
#     if gt_arr.ndim == 3:
#         gt = gt_arr.squeeze(0).numpy()
#     else:
#         gt = gt_arr.numpy()
#
#     gt = (gt > 0).astype(np.uint8)
#
#     pred_bin = (m2 > args.threshold).astype(np.uint8)
#     overlay = make_overlay(post_rgb, pred_bin)
#
#     with open(os.path.join(image_save_dir, "selected_info.txt"), "w", encoding="utf-8") as f:
#         f.write(f"SELECT_IMAGE_NAME={SELECT_IMAGE_NAME}\n")
#         f.write(f"resolved_index={selected_idx}\n")
#         f.write(f"filename={filename}\n")
#         f.write(f"topk={args.topk}\n")
#         f.write(f"shared_percentile={args.shared_percentile}\n")
#         f.write(f"threshold={args.threshold}\n")
#         f.write(f"swap_display={args.swap_display}\n")
#         f.write(f"stages={','.join(STAGES_TO_SAVE)}\n")
#         f.write(f"fixed_channels_1based={','.join([str(x) for x in FIXED_CHANNELS_1BASED])}\n")
#         f.write("note=fg_bg_ratio is post-hoc analysis using GT, not model input.\n")
#
#     for stage_name, stage_debug in debug.items():
#         if stage_name not in STAGES_TO_SAVE:
#             continue
#
#         required_keys = ["x_fused", "x_refined", "att", "avg_pool", "max_pool", "avg_out", "max_out", "pre_sigmoid"]
#         missing = [k for k in required_keys if stage_debug.get(k, None) is None]
#         if len(missing) > 0:
#             print(f"警告: {stage_name} 缺少 {missing}，跳过。")
#             continue
#
#         stage_dir = os.path.join(image_save_dir, stage_name)
#         ensure_dir(stage_dir)
#
#         x_fused_i = stage_debug["x_fused"][0].detach().cpu()
#         x_refined_i = stage_debug["x_refined"][0].detach().cpu()
#         att_i = stage_debug["att"][0, :, 0, 0].detach().cpu().numpy()
#
#         before_map = feature_to_heatmap(x_fused_i, out_size=(H, W))
#         after_map = feature_to_heatmap(x_refined_i, out_size=(H, W))
#
#         high_map = group_response_map(
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             topk=args.topk,
#             high=True,
#         )
#
#         low_map = group_response_map(
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             topk=args.topk,
#             high=False,
#         )
#
#         # 1. 基础 panel
#         save_basic_panel(
#             os.path.join(stage_dir, f"01_panel_{stage_name}.png"),
#             pre_rgb,
#             post_rgb,
#             gt,
#             before_map,
#             after_map,
#             pred_bin,
#             overlay,
#             att_i,
#             stage_name,
#         )
#
#         # 2. high/low 平均响应 panel
#         save_high_low_panel(
#             os.path.join(stage_dir, f"02_high_low_response_{stage_name}.png"),
#             pre_rgb,
#             post_rgb,
#             gt,
#             high_map,
#             low_map,
#             pred_bin,
#             stage_name,
#         )
#
#         # 3. Top10/Bottom10 权重柱状图
#         save_weight_rank_bar(
#             os.path.join(stage_dir, f"03_weight_rank_{stage_name}.png"),
#             att_i,
#             stage_name,
#             topk=args.topk,
#         )
#
#         # 4. Top10/Bottom10 单通道响应综合图：统一色阶，原始响应
#         save_single_channel_top_bottom_panel(
#             os.path.join(
#                 stage_dir,
#                 f"04_top{args.topk}_bottom{args.topk}_single_channels_SHARED_RAW_{stage_name}.png",
#             ),
#             pre_rgb,
#             post_rgb,
#             gt,
#             pred_bin,
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             stage_name=stage_name,
#             topk=args.topk,
#             shared_percentile=args.shared_percentile,
#             weighted=False,
#         )
#
#         # 5. Top10/Bottom10 单通道响应综合图：统一色阶，注意力加权后贡献
#         save_single_channel_top_bottom_panel(
#             os.path.join(
#                 stage_dir,
#                 f"05_top{args.topk}_bottom{args.topk}_single_channels_SHARED_WEIGHTED_{stage_name}.png",
#             ),
#             pre_rgb,
#             post_rgb,
#             gt,
#             pred_bin,
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             stage_name=stage_name,
#             topk=args.topk,
#             shared_percentile=args.shared_percentile,
#             weighted=True,
#         )
#
#         # 6. 固定通道 MLP 输入/输出 + 后验验证统计表
#         save_mlp_io_fixed_channels_csv(
#             os.path.join(stage_dir, f"06_mlp_io_fixed_channels_{stage_name}.csv"),
#             stage_debug,
#             x_fused_i,
#             att_i,
#             gt,
#             pred_bin,
#             filename=filename,
#             stage_name=stage_name,
#         )
#
#         # 7. 固定通道原始响应图
#         save_fixed_channels_panel(
#             os.path.join(stage_dir, f"07_fixed_channels_RAW_{stage_name}.png"),
#             pre_rgb,
#             post_rgb,
#             gt,
#             pred_bin,
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             stage_name=stage_name,
#             weighted=False,
#             shared_percentile=args.shared_percentile,
#         )
#
#         # 8. 固定通道注意力加权响应图
#         save_fixed_channels_panel(
#             os.path.join(stage_dir, f"08_fixed_channels_WEIGHTED_{stage_name}.png"),
#             pre_rgb,
#             post_rgb,
#             gt,
#             pred_bin,
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             stage_name=stage_name,
#             weighted=True,
#             shared_percentile=args.shared_percentile,
#         )
#
#         # 8.5 全部 64 通道 MLP 输入/输出 + 后验验证统计总表
#         save_mlp_io_all64_channels_csv(
#             os.path.join(stage_dir, f"09_mlp_io_all64_channels_{stage_name}.csv"),
#             stage_debug,
#             x_fused_i,
#             att_i,
#             gt,
#             pred_bin,
#             filename=filename,
#             stage_name=stage_name,
#         )
#
#         # 8.6 全部 64 通道原始响应图，统一色阶，分页输出
#         save_all64_channel_pages(
#             os.path.join(stage_dir, f"all64_channels_RAW_{stage_name}"),
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             stage_name=stage_name,
#             weighted=False,
#             shared_percentile=args.shared_percentile,
#             channels_per_page=16,
#         )
#
#         # 8.7 全部 64 通道注意力加权响应图，统一色阶，分页输出
#         save_all64_channel_pages(
#             os.path.join(stage_dir, f"all64_channels_WEIGHTED_{stage_name}"),
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             stage_name=stage_name,
#             weighted=True,
#             shared_percentile=args.shared_percentile,
#             channels_per_page=16,
#         )
#
#         # 9. 每个 Top/Bottom 单通道单独保存：统一色阶，原始响应
#         save_individual_single_channel_maps(
#             os.path.join(stage_dir, "single_channels_shared_raw"),
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             topk=args.topk,
#             shared_percentile=args.shared_percentile,
#             weighted=False,
#         )
#
#         # 10. 每个 Top/Bottom 单通道单独保存：统一色阶，注意力加权后贡献
#         save_individual_single_channel_maps(
#             os.path.join(stage_dir, "single_channels_shared_weighted"),
#             x_fused_i,
#             att_i,
#             out_size=(H, W),
#             topk=args.topk,
#             shared_percentile=args.shared_percentile,
#             weighted=True,
#         )
#
#         # 11. 所有 64 通道权重表
#         save_weights_csv(
#             os.path.join(stage_dir, f"weights_all64_{stage_name}.csv"),
#             att_i,
#         )
#
#         # 12. Top10/Bottom10 统计表
#         save_top_bottom_stats_csv(
#             os.path.join(stage_dir, f"top{args.topk}_bottom{args.topk}_stats_{stage_name}.csv"),
#             x_fused_i,
#             att_i,
#             gt,
#             topk=args.topk,
#         )
#
#         print(f"=> {stage_name} 保存完成: {stage_dir}")
#
#     print("=========================================================================")
#     print("单张图片 TAFM MLP 输入输出 + 固定通道可视化完成")
#     print(f"输出目录: {image_save_dir}")
#     print("重点查看：stage2/06_mlp_io_fixed_channels_stage2.csv")
#     print("重点查看：stage2/07_fixed_channels_RAW_stage2.png")
#     print("重点查看：stage2/08_fixed_channels_WEIGHTED_stage2.png")
#     print("重点查看：stage2/09_mlp_io_all64_channels_stage2.csv")
#     print("重点查看：stage2/all64_channels_WEIGHTED_stage2/all64_WEIGHTED_page01_stage2.png 等分页图")
#     print("=========================================================================")
#
#
# if __name__ == "__main__":
#     args = parse_args()
#
#     if not torch.cuda.is_available():
#         args.onGPU = False
#
#     print("Called with args:")
#     print(args)
#     print(f"SELECT_IMAGE_NAME = {SELECT_IMAGE_NAME}")
#     print(f"FIXED_CHANNELS_1BASED = {FIXED_CHANNELS_1BASED}")
#
#     visualize_selected_single_image(args)









# import os
# import sys
# import random
# import warnings
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
#
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
#
# warnings.filterwarnings('ignore', category=UserWarning)
#
# # ==========================================================
# # 只改这里：样本名、权重、失活强度
# # ==========================================================
# SAMPLE_NAME = '4274.png'
#
# WEIGHT_PATH = '/home/LWGANet/results/A2Net_LWGANet/L2_DSBN_GVLM_Composite/GVLM-CD-Processed/25.08.14-20.24/best_model.pth'
# FILE_ROOT = '/home/LWGANet/GVLM-CD-Processed'
# SAVE_ROOT = './test_results/HAGD_HardGuidanceCut_SingleImage'
#
# IN_WIDTH = 256
# IN_HEIGHT = 256
# THRESHOLD = 0.5
# ON_GPU = True
# SWAP_DISPLAY = True  # 沿用你旧脚本：只修显示，不改推理
#
# # 0.5 = 中性空间引导，保留浅层主体；0.0 = 更强干预但可能过度退化
# GUIDE_VALUE = 0.5
#
# # 0.0 = 完全切断被指定级别的深层上采样残差 p_up；0.25/0.5 = 弱化而不是完全切断
# RESIDUAL_ALPHA = 0.0
#
# # 默认 False：先只切断跨层概率 guide + 深层上采样残差。
# # 若仍不明显，可改 True：同时把 GRM 内部 x*sigmoid(m) 自门控失活为 x*1。
# DISABLE_SELF_GATE = False
#
# # 累积切断模式：从完整 HAGD 到逐级切断全部跨层引导
# CUT_MODES = [
#     ('Full HAGD', 'full'),
#     ('Only cut prob guides', 'cut_g_all'),
#     ('Cut G+R 5→4', 'cut_gr_5'),
#     ('Cut G+R 5→4,4→3', 'cut_gr_54'),
#     ('Cut all G+R', 'cut_gr_all'),
#     ('P2 shallow only', 'p2_only'),
# ]
#
# # ==========================================================
# # CPU 环境下自动 map_location
# # ==========================================================
# _original_torch_load = torch.load
#
# def _safe_torch_load(*args, **kwargs):
#     if not torch.cuda.is_available():
#         kwargs['map_location'] = 'cpu'
#     return _original_torch_load(*args, **kwargs)
#
# torch.load = _safe_torch_load
#
# # ==========================================================
# # 项目内部导入
# # 请先把 model_hagd_guidance_hardcut.py 放到 /home/LWGANet/models/ 下
# # ==========================================================
# sys.path.insert(0, '/home/LWGANet')
# import dataset_GVLM as myDataLoader
# import Transforms as myTransforms
# from models.model_hagd_guidance_hardcut import ModelToVisualize as BaseNet_LWGANet_L2
#
#
# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)
#
#
# def revert_sync_batchnorm(module):
#     res = module
#     if isinstance(module, torch.nn.SyncBatchNorm):
#         res = torch.nn.BatchNorm2d(
#             module.num_features,
#             module.eps,
#             module.momentum,
#             module.affine,
#             module.track_running_stats,
#         )
#         if module.affine:
#             res.weight.data = module.weight.data.clone().detach()
#             res.bias.data = module.bias.data.clone().detach()
#         res.running_mean = module.running_mean
#         res.running_var = module.running_var
#         res.num_batches_tracked = module.num_batches_tracked
#
#     for name, child in module.named_children():
#         res.add_module(name, revert_sync_batchnorm(child))
#     return res
#
#
# def safe_filename(name, idx):
#     name = os.path.basename(str(name).strip())
#     if name == '' or name.lower() in ['none', 'nan']:
#         name = f'{idx:04d}.png'
#     base, ext = os.path.splitext(name)
#     if base == '':
#         base = f'{idx:04d}'
#     if ext == '':
#         ext = '.png'
#     if ext.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
#         ext = '.png'
#     return base + ext
#
#
# def _basename_from_item(item):
#     if isinstance(item, str):
#         return os.path.basename(item)
#     if isinstance(item, (int, np.integer)):
#         return f'{int(item)}.png'
#     if isinstance(item, (tuple, list)) and len(item) > 0:
#         for x in item:
#             if isinstance(x, str) and x.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
#                 return os.path.basename(x)
#         for x in item:
#             if isinstance(x, str) and x.strip() != '':
#                 return os.path.basename(x)
#         return str(item[0])
#     if isinstance(item, dict):
#         for k in ['name', 'filename', 'file_name', 'id', 'A', 'B', 'img', 'image', 'path', 'A_path', 'img_path', 'image_path']:
#             if k in item:
#                 return os.path.basename(str(item[k]))
#     return str(item)
#
#
# def get_filenames_from_dataset_or_fallback(dataset, file_root):
#     candidate_attrs = [
#         'A_paths', 'B_paths', 'img_paths', 'image_paths', 'file_paths', 'path_list',
#         'file_list', 'files', 'imgs', 'images', 'image_list', 'img_list', 'data_list',
#         'test_files', 'file_names', 'filenames', 'names', 'name_list', 'ids'
#     ]
#     for attr in candidate_attrs:
#         if not hasattr(dataset, attr):
#             continue
#         value = getattr(dataset, attr)
#         if isinstance(value, list) and len(value) == len(dataset):
#             names = []
#             ok = True
#             for idx, item in enumerate(value):
#                 name = _basename_from_item(item)
#                 if name is None:
#                     ok = False
#                     break
#                 names.append(safe_filename(name, idx))
#             if ok:
#                 print(f'=> 文件名已从 dataset.{attr} 读取，数量: {len(names)}')
#                 return names
#
#     img_dir = os.path.join(file_root, 'test', 'A')
#     if not os.path.isdir(img_dir):
#         print(f'警告: 找不到 {img_dir}，将使用 index 命名。')
#         return [f'{i:04d}.png' for i in range(len(dataset))]
#
#     names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
#     if len(names) != len(dataset):
#         print('警告: test/A 文件数量与 dataset 长度不一致，使用 index 命名。')
#         return [f'{i:04d}.png' for i in range(len(dataset))]
#     return [safe_filename(n, idx) for idx, n in enumerate(names)]
#
#
# def resolve_selected_index_by_name(names, image_name):
#     wanted = os.path.basename(str(image_name).strip())
#     if wanted in names:
#         return names.index(wanted), wanted
#     safe_wanted = safe_filename(wanted, 0)
#     if safe_wanted in names:
#         return names.index(safe_wanted), safe_wanted
#     wanted_stem, wanted_ext = os.path.splitext(wanted)
#     if wanted_ext == '':
#         matched = [n for n in names if os.path.splitext(n)[0] == wanted_stem]
#         if len(matched) == 1:
#             return names.index(matched[0]), matched[0]
#     preview = '\n'.join([f'  {i}: {n}' for i, n in enumerate(names[:30])])
#     raise ValueError(f'找不到指定图片文件名: {wanted}\n当前测试集前 30 个文件名如下：\n{preview}')
#
#
# def denorm_rgb(x):
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#     x = x.detach().cpu().float()
#     x = x * std + mean
#     x = torch.clamp(x, 0, 1)
#     return x.permute(1, 2, 0).numpy()
#
#
# def make_overlay(rgb, pred_bin, alpha=0.45):
#     overlay = rgb.copy()
#     red = np.zeros_like(rgb)
#     red[..., 0] = 1.0
#     mask = pred_bin.astype(bool)
#     overlay[mask] = (1 - alpha) * overlay[mask] + alpha * red[mask]
#     return np.clip(overlay, 0, 1)
#
#
# def calc_metrics(gt, pred_bin):
#     gt_bin = (gt > 0).astype(np.uint8)
#     pred_bin = (pred_bin > 0).astype(np.uint8)
#     tp = int(((pred_bin == 1) & (gt_bin == 1)).sum())
#     fp = int(((pred_bin == 1) & (gt_bin == 0)).sum())
#     fn = int(((pred_bin == 0) & (gt_bin == 1)).sum())
#     tn = int(((pred_bin == 0) & (gt_bin == 0)).sum())
#     precision = tp / (tp + fp + 1e-6)
#     recall = tp / (tp + fn + 1e-6)
#     f1 = 2 * precision * recall / (precision + recall + 1e-6)
#     iou = tp / (tp + fp + fn + 1e-6)
#     return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, iou=iou,
#                 pred_area=int(pred_bin.sum()), gt_area=int(gt_bin.sum()))
#
#
# def load_model():
#     model = BaseNet_LWGANet_L2(pretrained=False)
#     if not torch.cuda.is_available():
#         model = revert_sync_batchnorm(model)
#         print('已将模型中的 SyncBatchNorm 自动降级为 BatchNorm2d 以兼容 CPU')
#
#     if not os.path.isfile(WEIGHT_PATH):
#         raise FileNotFoundError(f'找不到权重文件: {WEIGHT_PATH}')
#
#     print(f'=> Loading weights from: {WEIGHT_PATH}')
#     checkpoint = torch.load(WEIGHT_PATH)
#     state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
#     missing, unexpected = model.load_state_dict(state_dict, strict=False)
#     print('权重加载完成 strict=False')
#     print(f'missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')
#     if len(missing) > 0:
#         print('前 10 个 missing:', missing[:10])
#     if len(unexpected) > 0:
#         print('前 10 个 unexpected:', unexpected[:10])
#
#     if ON_GPU and torch.cuda.is_available():
#         model = model.cuda()
#         cudnn.benchmark = True
#     model.eval()
#     return model
#
#
# def build_test_dataset():
#     if not os.path.exists(FILE_ROOT):
#         raise TypeError(f'数据集路径不存在: {FILE_ROOT}')
#     mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
#     val_transform = myTransforms.Compose([
#         myTransforms.Normalize(mean=mean, std=std),
#         myTransforms.Scale(IN_WIDTH, IN_HEIGHT),
#         myTransforms.ToTensor(),
#     ])
#     test_data = myDataLoader.Dataset('test', file_root=FILE_ROOT, transform=val_transform, dataset_name='GVLM')
#     print(f'=> Test dataset length: {len(test_data)}')
#     return test_data
#
#
# def get_selected_sample(test_data, selected_idx):
#     item = test_data[selected_idx]
#     if not isinstance(item, (tuple, list)) or len(item) < 2:
#         raise RuntimeError('Dataset 返回格式不是 (img, target)，请检查 dataset_GVLM。')
#     img, target = item[0], item[1]
#     if img.ndim == 3:
#         img = img.unsqueeze(0)
#     if target.ndim == 2:
#         target = target.unsqueeze(0)
#     elif target.ndim == 3:
#         target = target.unsqueeze(0)
#     return img, target
#
#
# def save_binary_panel(path, pre_rgb, post_rgb, gt, pred_maps):
#     n = 3 + len(pred_maps)
#     plt.figure(figsize=(3.0 * n, 3.2))
#     base_titles = ['T1', 'T2', 'Label']
#     base_imgs = [pre_rgb, post_rgb, gt]
#     for i, (title, img) in enumerate(zip(base_titles, base_imgs)):
#         ax = plt.subplot(1, n, i + 1)
#         if img.ndim == 2:
#             ax.imshow(img, cmap='gray', vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#         ax.set_title(title, fontsize=9)
#         ax.axis('off')
#     for j, item in enumerate(pred_maps):
#         ax = plt.subplot(1, n, 3 + j + 1)
#         ax.imshow(item['bin'], cmap='gray', vmin=0, vmax=1)
#         ax.set_title(item['name'], fontsize=8)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches='tight')
#     plt.close()
#
#
# def save_prob_panel(path, pre_rgb, post_rgb, gt, pred_maps):
#     n = 3 + len(pred_maps)
#     plt.figure(figsize=(3.0 * n, 3.2))
#     base_titles = ['T1', 'T2', 'Label']
#     base_imgs = [pre_rgb, post_rgb, gt]
#     for i, (title, img) in enumerate(zip(base_titles, base_imgs)):
#         ax = plt.subplot(1, n, i + 1)
#         if img.ndim == 2:
#             ax.imshow(img, cmap='gray', vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#         ax.set_title(title, fontsize=9)
#         ax.axis('off')
#     for j, item in enumerate(pred_maps):
#         ax = plt.subplot(1, n, 3 + j + 1)
#         ax.imshow(item['prob'], cmap='jet', vmin=0, vmax=1)
#         ax.set_title(item['name'], fontsize=8)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches='tight')
#     plt.close()
#
#
# def save_overlay_panel(path, pre_rgb, post_rgb, gt, pred_maps):
#     n = 3 + len(pred_maps)
#     plt.figure(figsize=(3.0 * n, 3.2))
#     base_titles = ['T1', 'T2', 'Label']
#     base_imgs = [pre_rgb, post_rgb, gt]
#     for i, (title, img) in enumerate(zip(base_titles, base_imgs)):
#         ax = plt.subplot(1, n, i + 1)
#         if img.ndim == 2:
#             ax.imshow(img, cmap='gray', vmin=0, vmax=1)
#         else:
#             ax.imshow(img)
#         ax.set_title(title, fontsize=9)
#         ax.axis('off')
#     for j, item in enumerate(pred_maps):
#         ax = plt.subplot(1, n, 3 + j + 1)
#         ax.imshow(make_overlay(post_rgb, item['bin']))
#         ax.set_title(item['name'], fontsize=8)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches='tight')
#     plt.close()
#
#
#
# def normalize01(x):
#     x = x.astype(np.float32)
#     return (x - x.min()) / (x.max() - x.min() + 1e-8)
#
#
# def save_diff_panel(path, pred_maps):
#     """相对于 Full HAGD 的概率差异图：越亮表示 hard cut 后该位置概率变化越大。"""
#     if len(pred_maps) <= 1:
#         return
#     full_prob = pred_maps[0]['prob']
#     n = len(pred_maps) - 1
#     plt.figure(figsize=(3.0 * n, 3.2))
#     for j, item in enumerate(pred_maps[1:]):
#         ax = plt.subplot(1, n, j + 1)
#         diff = np.abs(full_prob - item['prob'])
#         ax.imshow(normalize01(diff), cmap='jet', vmin=0, vmax=1)
#         ax.set_title('Diff: Full vs ' + item['name'], fontsize=8)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches='tight')
#     plt.close()
#
# def main():
#     seed = 2333
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#
#     model = load_model()
#     test_data = build_test_dataset()
#     names = get_filenames_from_dataset_or_fallback(test_data, FILE_ROOT)
#     selected_idx, filename = resolve_selected_index_by_name(names, SAMPLE_NAME)
#     base = os.path.splitext(filename)[0]
#     save_dir = os.path.join(SAVE_ROOT, base)
#     ensure_dir(save_dir)
#
#     print('========================================================================')
#     print('HAGD 分层引导链路强失活单图推理')
#     print(f'SAMPLE_NAME={SAMPLE_NAME}')
#     print(f'resolved_index={selected_idx}')
#     print(f'filename={filename}')
#     print(f'save_dir={save_dir}')
#     print(f'GUIDE_VALUE={GUIDE_VALUE}, RESIDUAL_ALPHA={RESIDUAL_ALPHA}, DISABLE_SELF_GATE={DISABLE_SELF_GATE}')
#     print('========================================================================')
#
#     img, target = get_selected_sample(test_data, selected_idx)
#     img_cpu = img.detach().cpu()
#     target_cpu = target.detach().cpu()
#
#     pre_img = img[:, 0:3]
#     post_img = img[:, 3:6]
#     if ON_GPU and torch.cuda.is_available():
#         pre_img = pre_img.cuda()
#         post_img = post_img.cuda()
#
#     if SWAP_DISPLAY:
#         post_rgb = denorm_rgb(img_cpu[0, 0:3])
#         pre_rgb = denorm_rgb(img_cpu[0, 3:6])
#     else:
#         pre_rgb = denorm_rgb(img_cpu[0, 0:3])
#         post_rgb = denorm_rgb(img_cpu[0, 3:6])
#
#     gt_arr = target_cpu[0]
#     if gt_arr.ndim == 3:
#         gt = gt_arr.squeeze(0).numpy()
#     else:
#         gt = gt_arr.numpy()
#     gt = (gt > 0).astype(np.uint8)
#
#     pred_maps = []
#     with torch.no_grad():
#         for name, mode in CUT_MODES:
#             outputs = model(
#                 pre_img,
#                 post_img,
#                 cut_mode=mode,
#                 guide_value=GUIDE_VALUE,
#                 residual_alpha=RESIDUAL_ALPHA,
#                 disable_self_gate=DISABLE_SELF_GATE,
#             )
#             prob = outputs[0].detach().cpu().numpy()[0, 0]
#             pred_bin = (prob > THRESHOLD).astype(np.uint8)
#             pred_maps.append({
#                 'name': name,
#                 'mode': mode,
#                 'prob': prob,
#                 'bin': pred_bin,
#                 'metrics': calc_metrics(gt, pred_bin),
#             })
#
#     save_binary_panel(os.path.join(save_dir, '01_hard_guidance_cut_binary_panel.png'), pre_rgb, post_rgb, gt, pred_maps)
#     save_prob_panel(os.path.join(save_dir, '02_hard_guidance_cut_probability_panel.png'), pre_rgb, post_rgb, gt, pred_maps)
#     save_overlay_panel(os.path.join(save_dir, '03_hard_guidance_cut_overlay_panel.png'), pre_rgb, post_rgb, gt, pred_maps)
#     save_diff_panel(os.path.join(save_dir, '05_hard_guidance_cut_diff_from_full.png'), pred_maps)
#
#     with open(os.path.join(save_dir, '04_hard_guidance_cut_metrics.csv'), 'w', encoding='utf-8') as f:
#         f.write('name,mode,tp,fp,fn,tn,precision,recall,f1,iou,pred_area,gt_area\n')
#         for item in pred_maps:
#             m = item['metrics']
#             f.write(
#                 f"{item['name']},{item['mode']},{m['tp']},{m['fp']},{m['fn']},{m['tn']},"
#                 f"{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},{m['iou']:.6f},"
#                 f"{m['pred_area']},{m['gt_area']}\n"
#             )
#
#     with open(os.path.join(save_dir, '00_selected_info.txt'), 'w', encoding='utf-8') as f:
#         f.write(f'SAMPLE_NAME={SAMPLE_NAME}\n')
#         f.write(f'resolved_index={selected_idx}\n')
#         f.write(f'filename={filename}\n')
#         f.write(f'WEIGHT_PATH={WEIGHT_PATH}\n')
#         f.write(f'FILE_ROOT={FILE_ROOT}\n')
#         f.write(f'GUIDE_VALUE={GUIDE_VALUE}\n')
#         f.write(f'RESIDUAL_ALPHA={RESIDUAL_ALPHA}\n')
#         f.write(f'DISABLE_SELF_GATE={DISABLE_SELF_GATE}\n')
#         f.write(f'THRESHOLD={THRESHOLD}\n')
#         f.write('note=This is inference-time hard deactivation of HAGD cross-scale probability guides and deep residual paths. No retraining is used.\n')
#         f.write('note2=GUIDE_VALUE=0.5 means spatially uniform neutral guide; RESIDUAL_ALPHA=0 cuts selected deep residual paths, while preserving shallow TAFM feature pathways.\n')
#
#     print('========================================================================')
#     print('完成。重点查看：')
#     print(os.path.join(save_dir, '01_hard_guidance_cut_binary_panel.png'))
#     print(os.path.join(save_dir, '02_hard_guidance_cut_probability_panel.png'))
#     print(os.path.join(save_dir, '03_hard_guidance_cut_overlay_panel.png'))
#     print(os.path.join(save_dir, '04_hard_guidance_cut_metrics.csv'))
#     print(os.path.join(save_dir, '05_hard_guidance_cut_diff_from_full.png'))
#     print('========================================================================')
#
#
# if __name__ == '__main__':
#     if not torch.cuda.is_available():
#         ON_GPU = False
#     main()




import sys
import os
import random
import datetime
from pathlib import Path

import torch
import numpy as np
import cv2

# ==============================================================================
# 关键修复 0：兼容 tif 数据集
# 你的 dataset_GVLM.py 会把 test_xxx.tif 拼成 test_xxx.tif.png
# 这里拦截 cv2.imread，如果原路径读不到，就自动尝试 .tif/.tiff/.png/.jpg
# ==============================================================================

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

# ==============================================================================
# 关键修复 1：CPU 环境自动 map_location
# ==============================================================================

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
        default="/home/LWGANet/results/A2Net_LWGANet/L2_DSBN_GVLM_Composite/GVLM-CD-Processed/25.08.14-20.24/best_model.pth",
        type=str,
        help="Pretrained weight path",
    )

    parser.add_argument(
        "--file_root",
        default="/home/LWGANet/GVLM-CD-Processed",
        type=str,
        help="Data directory",
    )

    parser.add_argument(
        "--dataset_name",
        default="dataset_B",
        type=str,
        help="Dataset name used by dataset_GVLM.",
    )

    parser.add_argument(
        "--save_raw_vis",
        default=True,
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
