import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from utils.data_loading import BasicDataset, CarvanaDataset
from tqdm import tqdm
from pathlib import Path
from pytorch_msssim import ssim

from torch import optim

# import từ repo milesial
from unet import UNet  # teacher
from unet.student_unet_model import StudentUNet,StudentUNet2,StudentUNet3


# =========================
# CONFIG
# =========================
class Config:
    epochs = 200
    batch_size = 8
    lr = 1e-4
    img_scale = 1
    bilinear=False
    classes=1
    val_percent: float = 0.1
    weight_decay: float = 1e-8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = False
    gradient_clipping: float = 1.0
    # distillation weights
    lambda_seg = 1.0
    lambda_kd = 0.5
    lambda_feat = 0.2

    # confidence threshold (quan trọng cho wafer)
    kd_threshold = 0.1


cfg = Config()

class FeatureAlignModule(nn.Module):
    # def __init__(self):
    #     super().__init__()

    #     self.align = nn.ModuleDict({
    #         "enc1": nn.Conv2d(16, 64, 1),
    #         "enc2": nn.Conv2d(32, 128, 1),
    #         "enc3": nn.Conv2d(64, 256, 1),
    #         "bottleneck": nn.Conv2d(128, 1024, 1),
    #     })

    def __init__(self):
        super().__init__()

        self.align = nn.ModuleDict({
            "enc1": nn.Conv2d(16, 64, 1),
            "enc2": nn.Conv2d(32, 128, 1),
            "enc3": nn.Conv2d(64, 256, 1),
            "enc4": nn.Conv2d(128, 512, 1),
            "bottleneck": nn.Conv2d(256, 1024, 1),
        })

    def forward(self, s_feats, t_feats, l1_loss = None):
        loss = 0

        for k in self.align:
            s = s_feats[k]
            t = t_feats[k]

            # align spatial
            s = F.interpolate(s, size=t.shape[-2:], mode='bilinear', align_corners=False)

            # align channel
            s = self.align[k](s)
            if l1_loss == None:
                loss += F.mse_loss(s, t.detach())
            else:
                loss += l1_loss(s, t.detach())


        return loss / len(self.align)
# =========================
# LOSS
# =========================
# def segmentation_loss(pred, target):
#     return F.binary_cross_entropy_with_logits(pred, target)
def segmentation_loss(images, residual_pred, targets,l1_loss):
    preds = images*(1 + residual_pred)

    # L1 loss
    loss_l1 = l1_loss(preds, targets)

    # SSIM loss
    loss_ssim = 1 - ssim(preds, targets, data_range=1)

    # Edge loss (Gradient)
    # loss_edge = edge_loss(preds, targets)
    
    # loss = 0.6 * loss_l1 + 0.2 * loss_ssim + 0.2 * loss_edge
    loss = 0.8 * loss_l1 + 0.2 * loss_ssim
    return loss


def kd_loss(student, teacher, mask=None):
    if mask is not None:
        return ((student - teacher) ** 2 * mask).mean()
    return F.mse_loss(student, teacher)

def kd_loss2(student, teacher, l1_loss):
    return l1_loss(student, teacher)


def kd_loss3(image, student, teacher, l1_loss):
    s = image*(1+student)
    t = image*(1+teacher)
    def norm(x):
        return (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-6)
    
    return l1_loss(norm(s), norm(t))

def feature_loss(s_feat, t_feat):
    loss = 0
    for k in s_feat.keys():
        loss += F.mse_loss(s_feat[k], t_feat[k].detach())
    return loss / len(s_feat)
    
def feature_loss2(s_feat, t_feat, l1_loss):
    loss = 0
    for k in s_feat.keys():
        loss += l1_loss(s_feat[k], t_feat[k].detach())
    return loss / len(s_feat)

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


# =========================
# TRAIN FUNCTION
# =========================
def train_one_epoch(student, teacher, loader, optimizer,grad_scaler,l1_loss,align_module):
    student.train()
    teacher.eval()

    total_loss = 0

    # pbar = tqdm(loader, desc="Training")
    with tqdm(total=n_train, desc="Training", unit='img') as pbar:

        # for images, masks in pbar:
        #     images = images.to(cfg.device)
        #     masks = masks.to(cfg.device)
        for batch in loader:

            images = batch['image']
            targets = batch['mask']
            # images = images.to(device=device, dtype=torch.float32)[0][0].unsqueeze(0).unsqueeze(0)
            # targets = targets.to(device=device, dtype=torch.float32).unsqueeze(0)/65535.0
            # images = normalize_wafer_torch(images.to(device=device, dtype=torch.float32)[0][0].unsqueeze(0).unsqueeze(0))
            # targets = normalize_wafer_torch(targets.to(device=device, dtype=torch.float32).unsqueeze(0)/65535.0)
            # images = images.to(device=device, dtype=torch.float32)[:,0,:,:].unsqueeze(1)
            # targets = targets.to(device=device, dtype=torch.float32).unsqueeze(1)/65535.0
            images = normalize_wafer_batch_torch(images.to(device=cfg.device, dtype=torch.float32)[:,0,:,:].unsqueeze(1))
            targets = normalize_wafer_batch_torch(targets.to(device=cfg.device, dtype=torch.float32).unsqueeze(1)/65535.0)
            # print(images.size())
            # print(targets.size())

            # =========================
            # Teacher forward (no grad)
            # =========================
            with torch.no_grad():
                t_out, t_feat = teacher(images)

            # =========================
            # Student forward
            # =========================
            s_out, s_feat = student(images)

            # =========================
            # Segmentation loss
            # =========================
            loss_seg = segmentation_loss(images, s_out, targets,l1_loss)

            # =========================
            # Confidence mask (wafer trick)
            # =========================
            # with torch.no_grad():
            #     conf = torch.sigmoid(t_out)
            #     mask = (conf > cfg.kd_threshold).float()

            # =========================
            # KD loss
            # =========================
            # loss_kd = kd_loss(s_out, t_out, mask)
            # loss_kd = kd_loss(s_out, t_out)
            # loss_kd = kd_loss2(s_out, t_out, l1_loss)
            loss_kd = kd_loss3(images, s_out, t_out, l1_loss)

            # =========================
            # Feature distillation
            # =========================
            # loss_feat = feature_loss(s_feat, t_feat)
            # loss_feat = align_module(s_feat, t_feat)
            # loss_feat = align_module(s_feat, t_feat, l1_loss)

            # =========================
            # Total loss
            # =========================
            loss = (
                cfg.lambda_seg * loss_seg 
                + cfg.lambda_kd * loss_kd 
                # + cfg.lambda_feat * loss_feat
            )

            optimizer.zero_grad()
            # loss.backward()
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.gradient_clipping)

            # optimizer.step()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({
                "loss": loss.item(),
                "seg": loss_seg.item(),
                "kd": loss_kd.item(),
                # "feat": loss_feat.item()
            })

    return total_loss / len(loader)


# =========================
# VALIDATION
# =========================
def evaluate(model, loader,l1_loss):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        # for images, masks in loader:
        #     images = images.to(cfg.device)
        #     masks = masks.to(cfg.device)
        for batch in loader:

            images = batch['image']
            targets = batch['mask']
            # images = images.to(device=device, dtype=torch.float32)[0][0].unsqueeze(0).unsqueeze(0)
            # targets = targets.to(device=device, dtype=torch.float32).unsqueeze(0)/65535.0
            # images = normalize_wafer_torch(images.to(device=device, dtype=torch.float32)[0][0].unsqueeze(0).unsqueeze(0))
            # targets = normalize_wafer_torch(targets.to(device=device, dtype=torch.float32).unsqueeze(0)/65535.0)
            # images = images.to(device=device, dtype=torch.float32)[:,0,:,:].unsqueeze(1)
            # targets = targets.to(device=device, dtype=torch.float32).unsqueeze(1)/65535.0
            images = normalize_wafer_batch_torch(images.to(device=cfg.device, dtype=torch.float32, non_blocking=True)[:,0,:,:].unsqueeze(1))
            targets = normalize_wafer_batch_torch(targets.to(device=cfg.device, dtype=torch.float32, non_blocking=True).unsqueeze(1)/65535.0)

            preds, _ = model(images)
            loss = segmentation_loss(images,preds, targets,l1_loss)
            total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# MAIN TRAIN LOOP
# =========================
def train(train_loader, val_loader, teacher_ckpt=None):
    # =========================
    # Teacher
    # =========================
    teacher = UNet(n_channels=1, n_classes=cfg.classes, bilinear=cfg.bilinear)
    teacher = teacher.to(cfg.device)

    # load pretrained teacher
    if teacher_ckpt is not None:
        teacher.load_state_dict(torch.load(teacher_ckpt))

    # freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # IMPORTANT: sửa UNet để return features
    # bạn cần chỉnh forward teacher:
    # return logits, features

    # =========================
    # Student
    # =========================
    student = StudentUNet3(
        n_channels=1,
        n_classes=cfg.classes,
        base_c=16,
        bilinear=cfg.bilinear,
        return_features=True
    ).to(cfg.device)

    # optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        patience=10
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    l1_loss = nn.L1Loss()
    align_module = FeatureAlignModule().to(cfg.device)
    # =========================
    # Training loop
    # =========================
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")

        train_loss = train_one_epoch(student, teacher, train_loader, optimizer,grad_scaler,l1_loss,align_module)
        val_loss = evaluate(student, val_loader,l1_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(student.state_dict(), "student_best.pth")
            print("Saved best model")
        
        scheduler.step(train_loss)


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    # TODO: thay bằng dataset của bạn
    dir_img = Path('./traindata/data15/imgs/')
    dir_mask = Path('./traindata/data15/masks/')
    # dir_checkpoint = Path('./savecheckpoints/checkpoints20260421/')
    teacher_model_path = Path('./savecheckpoints/checkpoints20260409/checkpoint_epoch20.pth')
    
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, cfg.img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, cfg.img_scale)

    # split dataset
    n_val = int(len(dataset) * cfg.val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )

    loader_args = dict(batch_size=cfg.batch_size,
                       num_workers=min(8, os.cpu_count()),
                       pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''
        Starting training:
        Epochs: {cfg.epochs}
        Batch size: {cfg.batch_size}
        LR: {cfg.lr}
        Train size: {n_train}
        Val size: {n_val}
        Device: {cfg.device}
    ''')

    # train_dataset = ...
    # val_dataset = ...

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=cfg.batch_size,
    #     shuffle=True,
    #     num_workers=4
    # )

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=cfg.batch_size,
    #     shuffle=False,
    #     num_workers=4
    # )
    try:
        train(
            train_loader,
            val_loader,
            teacher_ckpt=teacher_model_path
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        train(
            train_loader,
            val_loader,
            teacher_ckpt=teacher_model_path
        )