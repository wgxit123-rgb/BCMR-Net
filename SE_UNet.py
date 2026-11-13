import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SELayer(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channel, in_channel // ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channel // ratio, in_channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs).view(b, c)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x).view(b, c, 1, 1)
        return x * inputs


class SE_UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SE_UNet, self).__init__()

        # 编码器 (通道: in_ch → 32 → 64 → 128 → 256 → 512)
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(256, 512)

        # SE注意力模块，通道数与解码器各阶段输出一致
        self.se1 = SELayer(256)   # c6 输出 256
        self.se2 = SELayer(128)   # c7 输出 128
        self.se3 = SELayer(64)    # c8 输出 64
        self.se4 = SELayer(32)    # c9 输出 32

        # 解码器
        # up6: 瓶颈 512 -> 256（拼接后 256+256=512 -> 卷积输出 256）
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)

        # up7: 256 -> 128（拼接后 128+128=256 -> 卷积输出 128）
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)

        # up8: 128 -> 64（拼接后 64+64=128 -> 卷积输出 64）
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)

        # up9: 64 -> 32（拼接后 32+32=64 -> 卷积输出 32）
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)

        self.conv10 = nn.Conv2d(32, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        # 编码器
        c1 = self.conv1(x)          # 32
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)         # 64
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)         # 128
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)         # 256
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)         # 512

        # 解码器
        up_6 = self.up6(c5)                     # 256
        merge6 = torch.cat([up_6, c4], dim=1)   # 512
        c6 = self.conv6(merge6)                 # 256
        c6 = self.se1(c6)                       # SE 通道重标定

        up_7 = self.up7(c6)                     # 128
        merge7 = torch.cat([up_7, c3], dim=1)   # 256
        c7 = self.conv7(merge7)                 # 128
        c7 = self.se2(c7)

        up_8 = self.up8(c7)                     # 64
        merge8 = torch.cat([up_8, c2], dim=1)   # 128
        c8 = self.conv8(merge8)                 # 64
        c8 = self.se3(c8)

        up_9 = self.up9(c8)                     # 32
        merge9 = torch.cat([up_9, c1], dim=1)   # 64
        c9 = self.conv9(merge9)                 # 32
        c9 = self.se4(c9)

        out = self.conv10(c9)
        out = self.last_activation(out)
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SE_UNet(in_ch=3, out_ch=1).to(device)
    x = torch.rand((2, 3, 224, 224)).to(device)
    print(model(x).shape)   # 预期: torch.Size([2, 1, 224, 224])