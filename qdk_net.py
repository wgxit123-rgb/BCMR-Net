"""Quantile-distribution kernel network for lightweight BUS segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eprc_net import (
    ConvBNAct,
    CrossScaleHighResolutionFusion,
    PlainImageStem,
    PyramidContextBlock,
    ResidualDWBlock,
)


class QuantileEchoBasis(nn.Module):
    """Describe channel responses with a differentiable local CDF."""

    def __init__(self, channels, bins=(-1.25, -0.5, 0.0, 0.5, 1.25)):
        super().__init__()
        self.register_buffer("bins", torch.tensor(bins))
        descriptor_size = len(bins) + 2
        self.gate = nn.Sequential(
            nn.Linear(descriptor_size, 8),
            nn.SiLU(inplace=True),
            nn.Linear(8, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, feature):
        pooled = F.adaptive_avg_pool2d(feature, (16, 16))
        mean = pooled.mean(dim=(2, 3), keepdim=True)
        std = pooled.var(dim=(2, 3), keepdim=True, unbiased=False).add(1e-5).sqrt()
        normalized = ((pooled - mean) / std).clamp(-3.0, 3.0)
        bins = self.bins.view(1, 1, -1, 1, 1)
        occupancy = torch.sigmoid(
            (normalized.unsqueeze(2) - bins) / 0.25
        ).mean(dim=(-1, -2))
        descriptor = torch.cat(
            [mean.squeeze(-1), std.squeeze(-1), occupancy], dim=2
        )
        gate = torch.tanh(self.gate(descriptor)).squeeze(-1).unsqueeze(-1).unsqueeze(-1)
        return feature * (1.0 + 0.25 * gate)


class AnnularContrastDescriptor(nn.Module):
    """Condition channels using pooled center-versus-ring response statistics."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(4, 8),
            nn.SiLU(inplace=True),
            nn.Linear(8, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, feature):
        pooled = F.adaptive_avg_pool2d(feature, (32, 32))
        center = F.avg_pool2d(pooled, 3, stride=1, padding=1)
        neighborhood = F.avg_pool2d(pooled, 9, stride=1, padding=4)
        ring = (81.0 * neighborhood - 9.0 * center) / 72.0
        descriptor = torch.stack(
            [
                center.mean(dim=(2, 3)),
                ring.mean(dim=(2, 3)),
                (center - ring).abs().mean(dim=(2, 3)),
                pooled.var(dim=(2, 3), unbiased=False).add(1e-5).sqrt(),
            ],
            dim=2,
        )
        gate = torch.tanh(self.gate(descriptor)).squeeze(-1).unsqueeze(-1).unsqueeze(-1)
        return feature * (1.0 + 0.25 * gate)


class DistributionKernelRouter(nn.Module):
    """Route local depthwise kernels with one sample-level distribution prior."""

    def __init__(self, channels):
        super().__init__()
        self.local = ConvBNAct(channels, channels, 3, groups=channels)
        self.broad = ConvBNAct(channels, channels, 5, groups=channels)
        self.dilated = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        hidden = max(channels // 4, 12)
        self.router = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.normal_(self.project.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.project.bias)
        self.gain = nn.Parameter(torch.tensor(0.10))

    def forward(self, feature):
        mean = feature.mean(dim=(2, 3))
        std = feature.var(dim=(2, 3), unbiased=False).add(1e-5).sqrt()
        weights = torch.softmax(self.router(torch.cat([mean, std], dim=1)), dim=1)
        branches = [self.local(feature), self.broad(feature), self.dilated(feature)]
        mixed = sum(
            weights[:, index : index + 1, None, None] * branch
            for index, branch in enumerate(branches)
        )
        residual = torch.tanh(self.project(mixed - feature))
        return feature + 0.5 * torch.tanh(self.gain) * residual


class QDKNet(nn.Module):
    """CAMP lightweight baseline with distribution-only local kernel adaptation."""

    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        channels=(32, 64, 128, 256, 512),
        use_qeb=True,
        use_acd=True,
        use_dkr=True,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.stem = PlainImageStem(in_ch, c1)
        self.stage1 = nn.Sequential(ResidualDWBlock(c1), ResidualDWBlock(c1))
        self.down2 = ConvBNAct(c1, c2, 3, stride=2)
        self.stage2 = nn.Sequential(ResidualDWBlock(c2), ResidualDWBlock(c2))
        self.down3 = ConvBNAct(c2, c3, 3, stride=2)
        self.stage3 = nn.Sequential(ResidualDWBlock(c3), ResidualDWBlock(c3))
        self.down4 = ConvBNAct(c3, c4, 3, stride=2)
        self.stage4 = nn.Sequential(ResidualDWBlock(c4), ResidualDWBlock(c4))
        self.down5 = ConvBNAct(c4, c5, 3, stride=2)
        self.stage5 = nn.Sequential(
            ResidualDWBlock(c5),
            PyramidContextBlock(c5),
            ResidualDWBlock(c5),
        )
        self.fusion = CrossScaleHighResolutionFusion(channels, 48)
        self.qeb = QuantileEchoBasis(48) if use_qeb else None
        self.acd = AnnularContrastDescriptor(48) if use_acd else None
        self.dkr = DistributionKernelRouter(48) if use_dkr else None
        self.head = nn.Sequential(
            ConvBNAct(48, 32, 3),
            nn.Conv2d(32, out_ch, 1),
        )

    def forward(self, image):
        x1, _ = self.stem(image)
        x1 = self.stage1(x1)
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        x4 = self.stage4(self.down4(x3))
        x5 = self.stage5(self.down5(x4))
        feature = self.fusion([x1, x2, x3, x4, x5])
        if self.qeb is not None:
            feature = self.qeb(feature)
        if self.acd is not None:
            feature = self.acd(feature)
        if self.dkr is not None:
            feature = self.dkr(feature)
        return torch.sigmoid(self.head(feature))


def build_qdk_net(
    in_ch=3,
    out_ch=1,
    channels=(32, 64, 128, 256, 512),
    use_qeb=True,
    use_acd=True,
    use_dkr=True,
):
    return QDKNet(
        in_ch=in_ch,
        out_ch=out_ch,
        channels=channels,
        use_qeb=use_qeb,
        use_acd=use_acd,
        use_dkr=use_dkr,
    )
