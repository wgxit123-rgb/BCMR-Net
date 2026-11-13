import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.fc1 = nn.Linear(in_channel, max(in_channel // ratio, 1), bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(max(in_channel // ratio, 1), in_channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.avg_pool(inputs).view(b, c)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x).view(b, c, 1, 1)
        return x * inputs


class ECA(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weights = self.pool(x).squeeze(-1).transpose(1, 2)
        weights = self.sigmoid(self.conv(weights)).transpose(1, 2).unsqueeze(-1)
        return x * weights


class MultiKernelRefineBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 16)
        self.expand = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.k3 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.k5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        self.k7 = nn.Conv2d(hidden, hidden, 7, padding=3, groups=hidden, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, max(hidden // 4, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(hidden // 4, 8), hidden * 3, 1),
        )
        self.project = nn.Sequential(
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.project[-1].weight)

    def forward(self, x):
        z = self.expand(x)
        branches = torch.stack([self.k3(z), self.k5(z), self.k7(z)], dim=1)
        weights = self.gate(z).view(z.size(0), 3, z.size(1), 1, 1)
        weights = torch.softmax(weights, dim=1)
        mixed = (branches * weights).sum(dim=1)
        return self.act(x + self.project(mixed))


class KANSelectiveKernel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        hidden = max(channels // 8, 16)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden * 4, len(self.branches) * channels, 1)
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        nn.init.zeros_(self.project[1].weight)

    def forward(self, x):
        stacked = torch.stack([branch(x) for branch in self.branches], dim=1)
        pooled = F.adaptive_avg_pool2d(stacked.sum(dim=1), 1)
        z = self.reduce(pooled)
        basis = torch.cat([z, z * z, torch.sin(z), torch.cos(z)], dim=1)
        weights = self.expand(basis).view(x.size(0), len(self.branches), x.size(1), 1, 1)
        weights = torch.softmax(weights, dim=1)
        mixed = (stacked * weights).sum(dim=1)
        return x + self.project(mixed)


class ResidualParallelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.channel_branch = ECA(channels)
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.strip_h = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.strip_w = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels + hidden * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.branch_weights = nn.Parameter(torch.zeros(3))
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.project[-1].weight)

    def forward(self, x):
        channel = self.channel_branch(x)
        spatial = self.spatial_branch(x)
        h_context = F.interpolate(self.strip_h(x.mean(dim=3, keepdim=True)), size=x.shape[2:], mode="bilinear", align_corners=False)
        w_context = F.interpolate(self.strip_w(x.mean(dim=2, keepdim=True)), size=x.shape[2:], mode="bilinear", align_corners=False)
        strip = self.mix(torch.cat([x, h_context, w_context], dim=1))
        weights = torch.softmax(self.branch_weights, dim=0)
        fused = weights[0] * channel + weights[1] * spatial + weights[2] * strip
        return x + self.project(fused)


class MKKANRPA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            MultiKernelRefineBlock(channels),
            KANSelectiveKernel(channels),
            ResidualParallelAttention(channels),
        )

    def forward(self, x):
        return self.block(x)


class SE_UNet_MK_KAN_RPA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SE_UNet_MK_KAN_RPA, self).__init__()
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.enhance5 = MKKANRPA(512)

        self.se1 = SELayer(256)
        self.se2 = SELayer(128)
        self.se3 = SELayer(64)
        self.se4 = SELayer(32)
        self.enhance6 = MKKANRPA(256)
        self.enhance7 = MKKANRPA(128)
        self.enhance8 = MKKANRPA(64)
        self.enhance9 = MKKANRPA(32)

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

        up_6 = self.up6(c5)
        c6 = self.conv6(torch.cat([up_6, c4], dim=1))
        c6 = self.enhance6(self.se1(c6))

        up_7 = self.up7(c6)
        c7 = self.conv7(torch.cat([up_7, c3], dim=1))
        c7 = self.enhance7(self.se2(c7))

        up_8 = self.up8(c7)
        c8 = self.conv8(torch.cat([up_8, c2], dim=1))
        c8 = self.enhance8(self.se3(c8))

        up_9 = self.up9(c8)
        c9 = self.conv9(torch.cat([up_9, c1], dim=1))
        c9 = self.enhance9(self.se4(c9))

        return self.last_activation(self.conv10(c9))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SE_UNet_MK_KAN_RPA(3, 1).to(device)
    x = torch.rand((2, 3, 224, 224), device=device)
    print(model(x).shape)
