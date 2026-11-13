import torch
import torch.nn as nn

from SE_UNet_MK_KAN_RPA import (
    DoubleConv,
    SELayer,
    MultiKernelRefineBlock,
    KANSelectiveKernel,
    ResidualParallelAttention,
)


class OptionalEnhance(nn.Module):
    def __init__(self, channels, use_mk=False, use_kan=False, use_rpa=False):
        super().__init__()
        layers = []
        if use_mk:
            layers.append(MultiKernelRefineBlock(channels))
        if use_kan:
            layers.append(KANSelectiveKernel(channels))
        if use_rpa:
            layers.append(ResidualParallelAttention(channels))
        self.block = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x):
        return self.block(x)


class SE_UNet_Ablation(nn.Module):
    def __init__(self, in_ch, out_ch, use_mk=False, use_kan=False, use_rpa=False):
        super().__init__()
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.enhance5 = OptionalEnhance(512, use_mk, use_kan, use_rpa)

        self.se1 = SELayer(256)
        self.se2 = SELayer(128)
        self.se3 = SELayer(64)
        self.se4 = SELayer(32)
        self.enhance6 = OptionalEnhance(256, use_mk, use_kan, use_rpa)
        self.enhance7 = OptionalEnhance(128, use_mk, use_kan, use_rpa)
        self.enhance8 = OptionalEnhance(64, use_mk, use_kan, use_rpa)
        self.enhance9 = OptionalEnhance(32, use_mk, use_kan, use_rpa)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)

        self.conv10 = nn.Conv2d(32, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))

        c6 = self.conv6(torch.cat([self.up6(c5), c4], dim=1))
        c6 = self.enhance6(self.se1(c6))
        c7 = self.conv7(torch.cat([self.up7(c6), c3], dim=1))
        c7 = self.enhance7(self.se2(c7))
        c8 = self.conv8(torch.cat([self.up8(c7), c2], dim=1))
        c8 = self.enhance8(self.se3(c8))
        c9 = self.conv9(torch.cat([self.up9(c8), c1], dim=1))
        c9 = self.enhance9(self.se4(c9))
        return self.last_activation(self.conv10(c9))


class SE_UNet_MK(SE_UNet_Ablation):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, use_mk=True)


class SE_UNet_KAN(SE_UNet_Ablation):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, use_kan=True)


class SE_UNet_RPA(SE_UNet_Ablation):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, use_rpa=True)


class SE_UNet_MK_KAN(SE_UNet_Ablation):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, use_mk=True, use_kan=True)


class SE_UNet_MK_RPA(SE_UNet_Ablation):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, use_mk=True, use_rpa=True)


class SE_UNet_KAN_RPA(SE_UNet_Ablation):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, use_kan=True, use_rpa=True)


if __name__ == "__main__":
    model = SE_UNet_MK_KAN(3, 1)
    x = torch.randn(2, 3, 224, 224)
    print(model(x).shape)
