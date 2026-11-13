"""Lightweight axial-lateral coherent echo transport network for BUS segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        dilation=1,
    ):
        kernel = _pair(kernel_size)
        dilation_pair = _pair(dilation)
        padding = tuple(
            ((size - 1) * rate) // 2
            for size, rate in zip(kernel, dilation_pair)
        )
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation_pair,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class AcousticResidualBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, 3, groups=channels),
            ConvBNAct(channels, hidden, 1),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, feature):
        return self.activation(feature + self.block(feature))


class MultiHorizonAcousticContext(nn.Module):
    """Aggregate local and elongated context without self-attention."""

    def __init__(self, channels):
        super().__init__()
        branch_channels = channels // 4
        self.local = ConvBNAct(channels, branch_channels, 3, groups=1)
        self.axial = nn.Sequential(
            ConvBNAct(channels, channels, (7, 1), groups=channels),
            ConvBNAct(channels, branch_channels, 1),
        )
        self.lateral = nn.Sequential(
            ConvBNAct(channels, channels, (1, 7), groups=channels),
            ConvBNAct(channels, branch_channels, 1),
        )
        self.global_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(branch_channels * 4, channels, 1),
            AcousticResidualBlock(channels),
        )

    def forward(self, feature):
        global_context = F.interpolate(
            self.global_projection(feature),
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fuse(
            torch.cat(
                [
                    self.local(feature),
                    self.axial(feature),
                    self.lateral(feature),
                    global_context,
                ],
                dim=1,
            )
        )


class BidirectionalAxialTransport(nn.Module):
    """Transport cumulative echo evidence in both acoustic-depth directions."""

    def __init__(self, channels):
        super().__init__()
        self.value = ConvBNAct(channels, channels, 1)
        self.fuse = nn.Sequential(
            ConvBNAct(channels * 4, channels, 1),
            ConvBNAct(channels, channels, (7, 1), groups=channels),
        )

    @staticmethod
    def _cumulative_means(value):
        height = value.shape[2]
        denominator = torch.arange(
            1, height + 1, device=value.device, dtype=value.dtype
        ).view(1, 1, height, 1)
        shallow_to_deep = torch.cumsum(value, dim=2) / denominator
        reversed_value = torch.flip(value, dims=(2,))
        deep_to_shallow = torch.flip(
            torch.cumsum(reversed_value, dim=2) / denominator, dims=(2,)
        )
        return shallow_to_deep, deep_to_shallow

    def forward(self, feature):
        value = self.value(feature)
        shallow, deep = self._cumulative_means(value)
        return self.fuse(
            torch.cat(
                [value - shallow, value - deep, shallow, deep], dim=1
            )
        )


class LateralScanlineCoherence(nn.Module):
    """Preserve responses coherent across neighboring ultrasound scan lines."""

    def __init__(self, channels):
        super().__init__()
        self.narrow = ConvBNAct(channels, channels, (1, 5), groups=channels)
        self.wide = ConvBNAct(channels, channels, (1, 11), groups=channels)
        self.fuse = ConvBNAct(channels * 3, channels, 1)

    def forward(self, feature):
        narrow = self.narrow(feature)
        wide = self.wide(feature)
        disagreement = (narrow - wide).abs()
        return self.fuse(torch.cat([narrow, wide, disagreement], dim=1))


class AcousticReliabilityField(nn.Module):
    """Estimate reliability from attenuation and cross-scanline consistency."""

    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNAct(6, 16, 3),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feature, gray):
        gray = F.interpolate(
            gray, size=feature.shape[-2:], mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        height = gray.shape[2]
        denominator = torch.arange(
            1, height + 1, device=gray.device, dtype=gray.dtype
        ).view(1, 1, height, 1)
        shallow = torch.cumsum(gray, dim=2) / denominator
        reversed_gray = torch.flip(gray, dims=(2,))
        deep = torch.flip(
            torch.cumsum(reversed_gray, dim=2) / denominator, dims=(2,)
        )
        axial_asymmetry = (shallow - deep).abs()
        posterior_drop = (shallow - deep).clamp_min(0.0)
        lateral_mean = F.avg_pool2d(gray, (1, 9), stride=1, padding=(0, 4))
        lateral_disagreement = (gray - lateral_mean).abs()
        local_mean = F.avg_pool2d(gray, 5, stride=1, padding=2)
        local_second = F.avg_pool2d(gray.square(), 5, stride=1, padding=2)
        local_std = (local_second - local_mean.square()).clamp_min(1e-6).sqrt()
        energy = feature.square().mean(dim=1, keepdim=True).sqrt()
        energy = energy / (energy.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return self.refine(
            torch.cat(
                [
                    gray,
                    posterior_drop,
                    axial_asymmetry,
                    lateral_disagreement,
                    local_std,
                    energy,
                ],
                dim=1,
            )
        )


class AxialLateralCoherentEchoTransport(nn.Module):
    """Compete directional echo evidence under an acoustic reliability field."""

    def __init__(
        self,
        channels,
        use_axial=True,
        use_lateral=True,
        use_reliability=True,
        independent_routing=False,
    ):
        super().__init__()
        self.use_axial = use_axial
        self.use_lateral = use_lateral
        self.use_reliability = use_reliability
        self.independent_routing = independent_routing
        self.axial = BidirectionalAxialTransport(channels) if use_axial else None
        self.lateral = LateralScanlineCoherence(channels) if use_lateral else None
        self.reliability = AcousticReliabilityField() if use_reliability else None
        self.path_count = 1 + int(use_axial) + int(use_lateral)
        router_channels = channels + int(use_reliability)
        self.router = nn.Sequential(
            nn.Conv2d(router_channels, max(channels // 2, 16), 1, bias=False),
            nn.GroupNorm(4, max(channels // 2, 16)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(channels // 2, 16), self.path_count, 1),
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.normal_(self.project.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.project.bias)
        self.gain = nn.Parameter(torch.tensor(0.10))

    def forward(self, feature, gray):
        paths = [feature]
        if self.use_axial:
            paths.append(self.axial(feature))
        if self.use_lateral:
            paths.append(self.lateral(feature))

        if self.use_reliability:
            reliability = self.reliability(feature, gray)
            router_input = torch.cat([feature, reliability], dim=1)
        else:
            reliability = None
            router_input = feature
        routing_logits = self.router(router_input)
        if reliability is not None and self.path_count == 3:
            directional_prior = torch.cat(
                [1.0 - reliability, reliability, reliability], dim=1
            )
            routing_logits = routing_logits + 0.5 * (directional_prior - 0.5)

        if self.independent_routing:
            weights = torch.sigmoid(routing_logits) / float(self.path_count)
        else:
            weights = torch.softmax(routing_logits, dim=1)
        fused = sum(
            weights[:, index : index + 1] * path
            for index, path in enumerate(paths)
        )
        residual = torch.tanh(self.project(fused - feature))
        return feature + 0.5 * torch.tanh(self.gain) * residual


class ACETNet(nn.Module):
    """Lightweight acoustic pyramid with scale-shared directional transport."""

    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        channels=(32, 64, 128, 256, 512),
        mode="full",
        decoder_channels=64,
    ):
        super().__init__()
        if mode not in {
            "base",
            "axial",
            "lateral",
            "no_reliability",
            "independent",
            "full",
        }:
            raise ValueError(f"Unknown ACET mode: {mode}")
        c1, c2, c3, c4, c5 = channels
        self.mode = mode
        self.stem = ConvBNAct(in_ch, c1, 3)
        self.stage1 = nn.Sequential(
            AcousticResidualBlock(c1), AcousticResidualBlock(c1)
        )
        self.down2 = ConvBNAct(c1, c2, 3, stride=2)
        self.stage2 = nn.Sequential(
            AcousticResidualBlock(c2), AcousticResidualBlock(c2)
        )
        self.down3 = ConvBNAct(c2, c3, 3, stride=2)
        self.stage3 = nn.Sequential(
            AcousticResidualBlock(c3), AcousticResidualBlock(c3)
        )
        self.down4 = ConvBNAct(c3, c4, 3, stride=2)
        self.stage4 = nn.Sequential(
            AcousticResidualBlock(c4), AcousticResidualBlock(c4)
        )
        self.down5 = ConvBNAct(c4, c5, 3, stride=2)
        self.stage5 = nn.Sequential(
            AcousticResidualBlock(c5),
            MultiHorizonAcousticContext(c5),
            AcousticResidualBlock(c5),
        )

        self.projections = nn.ModuleList(
            [ConvBNAct(channel, decoder_channels, 1) for channel in channels]
        )
        self.decoder_refine = nn.ModuleList(
            [AcousticResidualBlock(decoder_channels) for _ in range(4)]
        )
        if mode == "base":
            self.echo_transport = None
        else:
            self.echo_transport = AxialLateralCoherentEchoTransport(
                decoder_channels,
                use_axial=mode != "lateral",
                use_lateral=mode != "axial",
                use_reliability=mode not in {"no_reliability"},
                independent_routing=mode == "independent",
            )
        self.head = nn.Sequential(
            AcousticResidualBlock(decoder_channels),
            ConvBNAct(decoder_channels, 32, 3),
            nn.Conv2d(32, out_ch, 1),
        )

    @staticmethod
    def _upsample(feature, reference):
        return F.interpolate(
            feature,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def _transport(self, feature, gray):
        if self.echo_transport is None:
            return feature
        return self.echo_transport(feature, gray)

    def forward(self, image):
        gray = image.mean(dim=1, keepdim=True)
        x1 = self.stage1(self.stem(image))
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        x4 = self.stage4(self.down4(x3))
        x5 = self.stage5(self.down5(x4))

        p1, p2, p3, p4, p5 = [
            projection(feature)
            for projection, feature in zip(
                self.projections, [x1, x2, x3, x4, x5]
            )
        ]
        p4 = self.decoder_refine[0](p4 + self._upsample(p5, p4))
        p4 = self._transport(p4, gray)
        p3 = self.decoder_refine[1](p3 + self._upsample(p4, p3))
        p3 = self._transport(p3, gray)
        p2 = self.decoder_refine[2](p2 + self._upsample(p3, p2))
        p2 = self._transport(p2, gray)
        p1 = self.decoder_refine[3](p1 + self._upsample(p2, p1))
        return torch.sigmoid(self.head(p1))


def build_acet_net(
    in_ch=3,
    out_ch=1,
    channels=(32, 64, 128, 256, 512),
    mode="full",
):
    return ACETNet(in_ch=in_ch, out_ch=out_ch, channels=channels, mode=mode)
