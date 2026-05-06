import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from unet.student_unet_model import StudentUNet,StudentUNet2,StudentUNet3
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from pytorch_msssim import ssim

dir_img = Path('./traindata/data15/imgs/')
dir_mask = Path('./traindata/data15/masks/')
dir_checkpoint = Path('./savecheckpoints/checkpoints20260504_32/')

import torch.nn.functional as F

def gradient(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy
def edge_loss(pred, target):
    dx_p, dy_p = gradient(pred)
    dx_t, dy_t = gradient(target)

    loss_x = F.l1_loss(dx_p, dx_t)
    loss_y = F.l1_loss(dy_p, dy_t)

    return loss_x + loss_y

def normalize_wafer_torch(img):
    # Đảm bảo đầu vào là tensor
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32)

    # Tính percentile 1% và 99% (quantile nhận giá trị từ 0 đến 1)
    # Nếu img có nhiều chiều, quantile sẽ làm phẳng nó trừ khi ta chỉ định dim
    p = torch.quantile(img, torch.tensor([0.001, 0.999]).to(img.device))
    p1, p99 = p[0], p[1]
    if p1 == p99:
        p1 = img.min()
        p99 = img.max()
    # Chuẩn hóa (Min-Max scaling dựa trên percentile)
    img = (img - p1) / (p99 - p1)

    # Giới hạn giá trị trong khoảng [0, 1]
    img = torch.clamp(img, 0, 1)

    return img

def normalize_wafer_batch_torch(batch):
    """
    Chuẩn hóa Min-Max dựa trên percentile 0.1% và 99.9% cho từng ảnh trong batch.
    Input shape: (B, C, H, W) hoặc (B, H, W)
    """
    if not isinstance(batch, torch.Tensor):
        batch = torch.as_tensor(batch, dtype=torch.float32)

    # 1. Giữ lại shape gốc để phục vụ việc view/reshape
    original_shape = batch.shape
    B = original_shape[0]
    
    # 2. Flatten các chiều còn lại sau chiều Batch để tính quantile cho từng ảnh
    # View batch thành (B, -1) -> mỗi hàng là một ảnh
    flat_batch = batch.reshape(B, -1)

    # 3. Tính quantile dọc theo chiều dim=1 (từng ảnh)
    # q shape: (2, B)
    q = torch.quantile(flat_batch, torch.tensor([0.001, 0.999], device=batch.device), dim=1)
    p1 = q[0].view(B, 1)   # Shape (B, 1) để broadcast
    p99 = q[1].view(B, 1)

    # 4. Xử lý trường hợp p1 == p99 để tránh chia cho 0
    # Nếu p1 == p99, ta dùng min/max thực tế của ảnh đó
    mask = (p1 == p99).squeeze()
    if mask.any():
        actual_min = flat_batch[mask].min(dim=1, keepdim=True)[0]
        actual_max = flat_batch[mask].max(dim=1, keepdim=True)[0]
        p1[mask] = actual_min
        p99[mask] = actual_max

    # 5. Chuẩn hóa trên flat_batch (Broadcast tự động từ (B, 1) sang (B, N))
    # Công thức: (x - p1) / (p99 - p1)
    denom = p99 - p1
    # Tránh chia cho 0 tuyệt đối nếu cả ảnh là một màu duy nhất
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    
    normalized_flat = (flat_batch - p1) / denom

    # 6. Clamp và Reshape về shape ban đầu
    normalized_batch = torch.clamp(normalized_flat, 0, 1)
    return normalized_batch.view(original_shape)

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # split dataset
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )

    loader_args = dict(batch_size=batch_size,
                       num_workers=os.cpu_count(),
                       pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''
        Starting training:
        Epochs: {epochs}
        Batch size: {batch_size}
        LR: {learning_rate}
        Train size: {n_train}
        Val size: {n_val}
        Device: {device}
    ''')

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        patience=10
    )

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Loss functions
    l1_loss = nn.L1Loss()

    global_step = 0

    for epoch in range(epochs):

        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{epochs}', unit='img') as pbar:

            for batch in train_loader:

                images = batch['image']
                targets = batch['mask']
                
                # images = images.to(device=device, dtype=torch.float32)[0][0].unsqueeze(0).unsqueeze(0)
                # targets = targets.to(device=device, dtype=torch.float32).unsqueeze(0)/65535.0
                # images = normalize_wafer_torch(images.to(device=device, dtype=torch.float32)[0][0].unsqueeze(0).unsqueeze(0))
                # targets = normalize_wafer_torch(targets.to(device=device, dtype=torch.float32).unsqueeze(0)/65535.0)
                # images = images.to(device=device, dtype=torch.float32)[:,0,:,:].unsqueeze(1)
                # targets = targets.to(device=device, dtype=torch.float32).unsqueeze(1)/65535.0
                images = normalize_wafer_batch_torch(images.to(device=device, dtype=torch.float32)[:,0,:,:].unsqueeze(1))
                targets = normalize_wafer_batch_torch(targets.to(device=device, dtype=torch.float32).unsqueeze(1)/65535.0)
                # print(images.size())
                # print(targets.size())
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):

                    # -----------------------------
                    # Residual learning
                    # model output = enhancement
                    # -----------------------------
                    residual_pred = model(images)

                    preds = images*(1 + residual_pred)

                    # L1 loss
                    loss_l1 = l1_loss(preds, targets)

                    # SSIM loss
                    loss_ssim = 1 - ssim(preds, targets, data_range=1)

                    # Edge loss (Gradient)
                    # loss_edge = edge_loss(preds, targets)
                    
                    # loss = 0.6 * loss_l1 + 0.2 * loss_ssim + 0.2 * loss_edge
                    loss = 0.8 * loss_l1 + 0.2 * loss_ssim 

                optimizer.zero_grad(set_to_none=True)

                grad_scaler.scale(loss).backward()

                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])

                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(loss=loss.item())

        scheduler.step(epoch_loss)

        logging.info(f'Epoch {epoch+1} Loss: {epoch_loss / len(train_loader)}')

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            if (epoch+1)%5==0:
                torch.save(
                    model.state_dict(),
                    str(dir_checkpoint / f'checkpoint_epoch{epoch+1}.pth')
                )
    
                logging.info(f'Checkpoint {epoch+1} saved!')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    args.load = ""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = UNet(n_channels=1, n_classes=args.classes, base_c=64, bilinear=args.bilinear, separable=True)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
