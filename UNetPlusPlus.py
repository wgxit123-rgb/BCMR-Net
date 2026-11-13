from typing import Dict, List
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
    """上采样模块（用于嵌套卷积块内部的尺寸恢复）"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """上采样 x1 并裁剪/填充到与 x2 相同尺寸后拼接"""
        x1 = self.up(x1)
        # 处理尺寸不一致
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        return torch.cat([x2, x1], dim=1)


class NestedConvBlock(nn.Module):
    """UNet++ 核心：嵌套卷积块（修正版）"""
    def __init__(self, in_channels_list: List[int], out_channels: int, bilinear: bool = True):
        super().__init__()
        total_in_channels = sum(in_channels_list)
        self.res_block = ResidualBlock(total_in_channels, out_channels)

        # 为每个下层输入（索引≥1）准备一个上采样器
        self.upsamplers = nn.ModuleList()
        for i in range(1, len(in_channels_list)):
            in_ch = in_channels_list[i]
            if bilinear:
                self.upsamplers.append(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
            else:
                # 转置卷积，保持通道数不变
                self.upsamplers.append(
                    nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
                )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # 水平方向的输入（inputs[0]）不经过上采样
        features = [inputs[0]]

        # 对其余输入进行上采样并对齐尺寸
        for i in range(1, len(inputs)):
            upsampled = self.upsamplers[i - 1](inputs[i])
            # 确保与 inputs[0] 的空间尺寸一致
            if upsampled.shape[2:] != inputs[0].shape[2:]:
                upsampled = F.interpolate(upsampled, size=inputs[0].shape[2:],
                                          mode='bilinear', align_corners=True)
            features.append(upsampled)

        # 通道维度拼接所有特征，然后通过残差块
        x = torch.cat(features, dim=1)
        return self.res_block(x)


class NestedUNet(nn.Module):
    """
    UNet++ (Nested U-Net)
    编码器通道数: [32, 64, 128, 256, 512]
    支持深度监督（deep supervision）
    """
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 deep_supervision: bool = False,
                 base_channels: List[int] = None):
        """
        Args:
            in_channels: 输入图像通道数
            num_classes: 分割类别数（二分类设为1）
            bilinear: 是否使用双线性插值上采样（否则使用转置卷积）
            deep_supervision: 是否启用深度监督
            base_channels: 编码器各层通道数，默认 [32, 64, 128, 256, 512]
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision

        if base_channels is None:
            base_channels = [32, 64, 128, 256, 512]

        c0, c1, c2, c3, c4 = base_channels

        # ==================== 编码器 ====================
        self.conv0_0 = ResidualBlock(in_channels, c0)
        self.down1   = Down(c0, c1)
        self.down2   = Down(c1, c2)
        self.down3   = Down(c2, c3)
        self.down4   = Down(c3, c4)

        # ==================== 嵌套解码器（跳跃路径） ====================
        # --- 第一列（X_{0, j}）: 编码器特征直连 ---
        self.conv0_1 = NestedConvBlock([c0, c1], c0, bilinear)  # 输入: X^{0,0}, X^{1,0}
        self.conv0_2 = NestedConvBlock([c0, c1, c2], c0, bilinear)  # 输入: X^{0,1}, X^{1,1}, X^{2,0}
        self.conv0_3 = NestedConvBlock([c0, c1, c2, c3], c0, bilinear)
        self.conv0_4 = NestedConvBlock([c0, c1, c2, c3, c4], c0, bilinear)

        # --- 第二列（X_{1, j}） ---
        self.conv1_1 = NestedConvBlock([c1, c2], c1, bilinear)
        self.conv1_2 = NestedConvBlock([c1, c2, c3], c1, bilinear)
        self.conv1_3 = NestedConvBlock([c1, c2, c3, c4], c1, bilinear)

        # --- 第三列（X_{2, j}） ---
        self.conv2_1 = NestedConvBlock([c2, c3], c2, bilinear)
        self.conv2_2 = NestedConvBlock([c2, c3, c4], c2, bilinear)

        # --- 第四列（X_{3, j}） ---
        self.conv3_1 = NestedConvBlock([c3, c4], c3, bilinear)

        # ==================== 深度监督输出层 ====================
        if deep_supervision:
            # 在 X^{0,1}, X^{0,2}, X^{0,3}, X^{0,4} 后各加一个 1x1 卷积
            self.out_0_1 = nn.Conv2d(c0, num_classes, kernel_size=1)
            self.out_0_2 = nn.Conv2d(c0, num_classes, kernel_size=1)
            self.out_0_3 = nn.Conv2d(c0, num_classes, kernel_size=1)
            self.out_0_4 = nn.Conv2d(c0, num_classes, kernel_size=1)
        else:
            # 仅使用最终输出层
            self.out_conv = nn.Conv2d(c0, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # ==================== 编码阶段 ====================
        x0_0 = self.conv0_0(x)                          # 32 通道
        x1_0 = self.down1(x0_0)                         # 64 通道
        x2_0 = self.down2(x1_0)                         # 128 通道
        x3_0 = self.down3(x2_0)                         # 256 通道
        x4_0 = self.down4(x3_0)                         # 512 通道

        # ==================== 嵌套解码阶段 ====================
        # 第一列
        x0_1 = self.conv0_1([x0_0, x1_0])
        x1_1 = self.conv1_1([x1_0, x2_0])
        x2_1 = self.conv2_1([x2_0, x3_0])
        x3_1 = self.conv3_1([x3_0, x4_0])

        # 第二层 (j=2) —— 依赖 j=1 节点和编码器特征
        x0_2 = self.conv0_2([x0_1, x1_1, x2_0])
        x1_2 = self.conv1_2([x1_1, x2_1, x3_0])
        x2_2 = self.conv2_2([x2_1, x3_1, x4_0])

        # 第三层 (j=3) —— 依赖 j=2 节点和更深层特征
        x0_3 = self.conv0_3([x0_2, x1_2, x2_1, x3_0])
        x1_3 = self.conv1_3([x1_2, x2_2, x3_1, x4_0])

        # 第四层 (j=4) —— 最终输出
        x0_4 = self.conv0_4([x0_3, x1_3, x2_2, x3_1, x4_0])

        # ==================== 输出阶段 ====================

            # 训练时返回多尺度输出，用于深度监督
        logits = self.out_0_4(x0_4)
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return logits
        return torch.sigmoid(logits)        # 最终输出


if __name__ == "__main__":
    # 测试模型
    model = NestedUNet(in_channels=3, num_classes=1, deep_supervision=True, bilinear=True)
    x = torch.randn(1, 3, 224, 224)
    model.train()
    out = model(x)
    print(out.shape)  # 应为 torch.Size([1, 1, 224, 224])
