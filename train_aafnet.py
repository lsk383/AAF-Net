import datetime
import socket
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim.lr_scheduler
# 使用别名以保持清晰
import dataset_GVLM as myDataLoader
import Transforms as myTransforms
# 确保从正确的 utils 文件导入
from utils import train, val
import os
import numpy as np
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from thop import profile
# 确保导入了正确的模型文件
from models.model_with_tafm import ModelToVisualize as AAFNet

sys.path.insert(0, 'tools')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=20000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=0, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--logFile', default='trainValLog.txt', help='File that stores logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'), help='Run on GPU')
    parser.add_argument('--pretrained', default=True, help='Use ImageNet pre-trained weights for backbone')
    parser.add_argument('--file_root', default="/home/LWGANet/GVLM-CD-Processed", help='Data directory root')
    args = parser.parse_args()
    print('Called with args:');
    print(args)
    return args


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def init_distributed_if_needed():
    if dist.is_available() and not dist.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(
            backend=backend,
            init_method='tcp://127.0.0.1:{}'.format(find_free_port()),
            rank=0,
            world_size=1,
        )


def trainValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


    model = AAFNet(pretrained=args.pretrained)

    model_name = 'AAF-Net'

    if args.onGPU:
        if not torch.cuda.is_available():
            raise RuntimeError('onGPU=True, but CUDA is not available.')
        init_distributed_if_needed()
        model = model.cuda()

    device = next(model.parameters()).device
    input_data = torch.randn(1, 3, 256, 256, device=device)
    flops, params = profile(model, inputs=(input_data, input_data), verbose=False)
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M.")

    args.savedir = f'results/{model_name}/{os.path.basename(args.file_root)}/{datetime.datetime.now().strftime("%y.%m.%d-%H:%M")}/'
    args.vis_dir = os.path.join(args.savedir, 'Vis')
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    tensorboard_dir = os.path.join(args.savedir, 'runs')
    tb_writer = SummaryWriter(tensorboard_dir)

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total network parameters: {total_params / 1e6:.2f} M')

    mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]

    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        myTransforms.ToTensor()
    ])
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    train_data = myDataLoader.Dataset("train", file_root=args.file_root, transform=trainDataset_main,
                                      dataset_name='GVLM')
    trainLoader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_data = myDataLoader.Dataset("val", file_root=args.file_root, transform=valDataset, dataset_name='GVLM')
    valLoader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset, dataset_name='GVLM')
    testLoader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    max_batches = len(trainLoader)
    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    print(f'Training will run for {args.max_epochs} epochs to reach {args.max_steps} steps.')

    start_epoch = 0
    cur_iter = 0
    max_F1_val = 0

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        cur_iter = start_epoch * len(trainLoader)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        max_F1_val = checkpoint.get('max_F1_val', 0)
        print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    else:
        print(f"=> no checkpoint found at '{args.resume}'")

    logFileLoc = os.path.join(args.savedir, args.logFile)
    logger = open(logFileLoc, 'a')
    logger.write(f"Arguments: {args}\n")
    logger.write(f"FLOPs(G): {flops / 1e9:.2f}, Params(M): {params / 1e6:.2f}\n\n")
    logger.flush()

    start_time = datetime.datetime.now()
    print("Training started at:", start_time)

    for epoch in range(start_epoch, args.max_epochs):
        lossTr, score_tr, lr = train(args, trainLoader, model, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader)
        torch.cuda.empty_cache()

        lossVal, score_val = val(args, valLoader, model, epoch)
        torch.cuda.empty_cache()

        log_message = (f"\nEpoch(val): {epoch}\tKappa: {score_val['Kappa']:.4f}\tIoU: {score_val['IoU']:.4f}\t"
                       f"F1: {score_val['F1']:.4f}\tR: {score_val['recall']:.4f}\tP: {score_val['precision']:.4f}")
        print(log_message)
        logger.write(log_message)
        logger.flush()

        tb_writer.add_scalar("Loss/train", lossTr, epoch)
        tb_writer.add_scalar("Loss/val", lossVal, epoch)
        tb_writer.add_scalar("Metrics/F1_val", score_val['F1'], epoch)
        tb_writer.add_scalar("Metrics/IoU_val", score_val['IoU'], epoch)
        tb_writer.add_scalar("Metrics/Kappa_val", score_val['Kappa'], epoch)
        tb_writer.add_scalar("LR", lr, epoch)

        checkpoint_path = os.path.join(args.savedir, 'checkpoint.pth.tar')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'max_F1_val': max_F1_val,
        }, checkpoint_path)

        if max_F1_val < score_val['F1']:
            max_F1_val = score_val['F1']
            best_model_path = os.path.join(args.savedir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"*** New best model saved at epoch {epoch} with F1-score: {max_F1_val:.4f} ***")

    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("Training finished at:", end_time)
    print("Total training time:", total_time)

    # Final evaluation on the test set with the best model
    print("\n--- Evaluating on test set with BEST model ---")
    best_model_path = os.path.join(args.savedir, 'best_model.pth')
    if os.path.exists(best_model_path):
        best_state_dict = torch.load(best_model_path)
        model.load_state_dict(best_state_dict)
        loss_test, score_test = val(args, testLoader, model, args.max_epochs)
        test_log = (f"\nTest(best_epoch):\tKappa: {score_test['Kappa']:.4f}\tIoU: {score_test['IoU']:.4f}\t"
                    f"F1: {score_test['F1']:.4f}\tR: {score_test['recall']:.4f}\tP: {score_test['precision']:.4f}")
        print(test_log)
        logger.write(test_log)
    else:
        print("Best model file not found. Skipping test evaluation.")

    # <<< 新增代码: 在最后一个epoch的模型上进行测试 >>>
    print("\n--- Evaluating on test set with LAST model ---")
    last_model_path = os.path.join(args.savedir, 'checkpoint.pth.tar')
    if os.path.exists(last_model_path):
        last_checkpoint = torch.load(last_model_path)
        model.load_state_dict(last_checkpoint['state_dict'])
        loss_test_last, score_test_last = val(args, testLoader, model, args.max_epochs)
        test_log_last = (
            f"\nTest(last_epoch):\tKappa: {score_test_last['Kappa']:.4f}\tIoU: {score_test_last['IoU']:.4f}\t"
            f"F1: {score_test_last['F1']:.4f}\tR: {score_test_last['recall']:.4f}\tP: {score_test_last['precision']:.4f}\n"
            f"Total time: {total_time}")
        print(test_log_last)
        logger.write(test_log_last)
    else:
        print("Last checkpoint file not found. Skipping last epoch test evaluation.")
    # <<< 新增代码结束 >>>

    logger.flush()
    logger.close()
    tb_writer.close()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    trainValidateSegmentation(args)
