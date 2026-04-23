import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Depthwise Separable Conv
# =========================
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


# =========================
# Double DSConv (thay DoubleConv)
# =========================
class DoubleDSConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = DSConv(in_ch, out_ch)
        self.conv2 = DSConv(out_ch, out_ch)

    def forward(self, x):
        return self.conv2(self.conv1(x))


# =========================
# Down block
# =========================
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleDSConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


# =========================
# Up block
# =========================
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleDSConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleDSConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # padding nếu lệch size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# =========================
# Output layer
# =========================
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# =========================
# STUDENT UNET
# =========================
class StudentUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_c=16, bilinear=True, return_features=False):
        super().__init__()

        self.return_features = return_features

        # Encoder (giảm depth còn 3 level)
        self.inc = DoubleDSConv(n_channels, base_c)        # 16
        self.down1 = Down(base_c, base_c * 2)              # 32
        self.down2 = Down(base_c * 2, base_c * 4)          # 64

        # Bottleneck
        self.bottleneck = Down(base_c * 4, base_c * 8)     # 128

        # Decoder
        self.up1 = Up(base_c * 8 + base_c * 4, base_c * 4, bilinear)
        self.up2 = Up(base_c * 4 + base_c * 2, base_c * 2, bilinear)
        self.up3 = Up(base_c * 2 + base_c, base_c, bilinear)

        self.outc = OutConv(base_c, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Bottleneck
        x4 = self.bottleneck(x3)

        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)

        if self.return_features:
            features = {
                "enc1": x1,
                "enc2": x2,
                "enc3": x3,
                "bottleneck": x4
            }
            return logits, features

        return logits