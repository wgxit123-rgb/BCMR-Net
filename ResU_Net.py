from typing import Dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块：两个3x3卷积 + 跳跃连接（必要时使用1x1卷积调整通道）"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接：若分辨率或通道数变化，则用1x1卷积
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class Down(nn.Module):
    """下采样：MaxPool2d + 残差块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.res_block = ResidualBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_block(self.pool(x))


class Up(nn.Module):
    """上采样模块，显式指定各通道数"""
    def __init__(self, x1_channels: int, x2_channels: int, out_channels: int, bilinear: bool = True):
        """
        Args:
            x1_channels: 上采样前的输入通道数（来自解码器下层）
            x2_channels: 跳跃连接的特征通道数（来自编码器对应层）
            out_channels: 输出通道数
            bilinear: 是否使用双线性插值（否则用转置卷积）
        """
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            up_channels = x1_channels   # 双线性插值不改变通道数
        else:
            self.up = nn.ConvTranspose2d(x1_channels, x1_channels // 2, kernel_size=2, stride=2)
            up_channels = x1_channels // 2

        # 拼接后的总通道数 = 上采样后的通道数 + 跳跃连接通道数
        concat_channels = up_channels + x2_channels
        self.res_block = ResidualBlock(concat_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # 处理尺寸不一致（当输入尺寸为奇数时可能发生）
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.res_block(x)


class OutConv(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class ResUnet(nn.Module):
    """使用残差块的UNet"""
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(ResUnet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 编码器（下采样）
        self.in_conv = ResidualBlock(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        # 解码器（上采样），显式指定每个Up的通道参数
        # up1: 输入来自 down4 (x1_channels = base_c*16//factor)，跳跃连接来自 down3 (x2_channels = base_c*8)
        self.up1 = Up(base_c * 16 // factor, base_c * 8, base_c * 8 // factor, bilinear)
        # up2: 输入来自 up1 输出 (x1_channels = base_c*8//factor)，跳跃连接来自 down2 (x2_channels = base_c*4)
        self.up2 = Up(base_c * 8 // factor, base_c * 4, base_c * 4 // factor, bilinear)
        # up3: 输入来自 up2 输出 (x1_channels = base_c*4//factor)，跳跃连接来自 down1 (x2_channels = base_c*2)
        self.up3 = Up(base_c * 4 // factor, base_c * 2, base_c * 2 // factor, bilinear)
        # up4: 输入来自 up3 输出 (x1_channels = base_c*2//factor)，跳跃连接来自 in_conv (x2_channels = base_c)
        self.up4 = Up(base_c * 2 // factor, base_c, base_c, bilinear)

        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码
        x1 = self.in_conv(x)      # base_c
        x2 = self.down1(x1)       # base_c*2
        x3 = self.down2(x2)       # base_c*4
        x4 = self.down3(x3)       # base_c*8
        x5 = self.down4(x4)       # base_c*16//factor

        # 解码
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return x
        return torch.sigmoid(x)


if __name__ == "__main__":
    model = ResUnet(in_channels=3, num_classes=1, base_c=32, bilinear=True)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)   # 应为 torch.Size([1, 1, 256, 256])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
