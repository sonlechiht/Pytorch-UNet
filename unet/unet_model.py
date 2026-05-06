""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_c = 64, bilinear=False,separable=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, base_c))
        self.down1 = (Down(base_c, base_c*2,separable=separable))
        self.down2 = (Down(base_c*2, base_c*4,separable=separable))
        self.down3 = (Down(base_c*4, base_c*8,separable=separable))
        factor = 2 if bilinear else 1
        self.down4 = (Down(base_c*8, base_c*16 // factor))
        self.up1 = (Up(base_c*16, base_c*8 // factor, bilinear))
        self.up2 = (Up(base_c*8, base_c*4 // factor, bilinear,separable=separable))
        self.up3 = (Up(base_c*4, base_c*2 // factor, bilinear,separable=separable))
        self.up4 = (Up(base_c*2, base_c, bilinear,separable=separable))
        self.outc = (OutConv(base_c, n_classes))

    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)
    #     x = self.up1(x5, x4)
    #     x = self.up2(x, x3)
    #     x = self.up3(x, x2)
    #     x = self.up4(x, x1)
    #     logits = self.outc(x)
    #     return logits

    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)

    #     features = {
    #         "enc1": x1,
    #         "enc2": x2,
    #         "enc3": x3,
    #         "bottleneck": x5
    #     }

    #     x = self.up1(x5, x4)
    #     x = self.up2(x, x3)
    #     x = self.up3(x, x2)
    #     x = self.up4(x, x1)

    #     logits = self.outc(x)

    #     return logits, features
    
    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)

    #     features = {
    #         "enc1": x1,
    #         "enc2": x2,
    #         "enc3": x3,
    #         "enc4": x4,
    #         "bottleneck": x5
    #     }

    #     x = self.up1(x5, x4)
    #     x = self.up2(x, x3)
    #     x = self.up3(x, x2)
    #     x = self.up4(x, x1)

    #     logits = self.outc(x)

    #     return logits, features
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)