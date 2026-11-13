import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        groups=1,
        activate=True,
    ):
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if activate:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class LiteResidualBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.expand = ConvNormAct(channels, hidden)
        self.depthwise = ConvNormAct(
            hidden, hidden, kernel_size=3, groups=hidden
        )
        self.project = ConvNormAct(
            hidden, channels, activate=False
        )
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-2))

    def forward(self, features):
        residual = self.project(self.depthwise(self.expand(features)))
        scale = self.layer_scale.view(1, -1, 1, 1)
        return features + scale * residual


class LiteDownsample(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvNormAct(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
            ),
            ConvNormAct(in_channels, out_channels),
        )


class UltrasoundGroupDescriptor(nn.Module):
    """Group-wise intensity and directional variation descriptors."""

    def __init__(self, channels, groups=8, eps=1e-6):
        super().__init__()
        if channels % groups != 0:
            raise ValueError(
                "channels={} must be divisible by groups={}".format(
                    channels, groups
                )
            )
        self.groups = groups
        self.group_channels = channels // groups
        self.eps = float(eps)
        self.norm = nn.LayerNorm(4)

    def forward(self, features):
        batch, _, height, width = features.shape
        grouped = features.reshape(
            batch,
            self.groups,
            self.group_channels,
            height,
            width,
        )
        mean = grouped.mean(dim=(2, 3, 4))
        variance = (grouped - mean[:, :, None, None, None]).square().mean(
            dim=(2, 3, 4)
        )
        std = torch.sqrt(variance + self.eps)
        if height > 1:
            axial = (grouped[..., 1:, :] - grouped[..., :-1, :]).abs().mean(
                dim=(2, 3, 4)
            )
        else:
            axial = torch.zeros_like(mean)
        if width > 1:
            lateral = (
                grouped[..., :, 1:] - grouped[..., :, :-1]
            ).abs().mean(dim=(2, 3, 4))
        else:
            lateral = torch.zeros_like(mean)
        descriptor = torch.stack([mean, std, axial, lateral], dim=-1)
        return torch.tanh(self.norm(descriptor))


class CubicBSplineBasis(nn.Module):
    def __init__(self, basis_count=6, degree=3):
        super().__init__()
        if basis_count <= degree:
            raise ValueError("basis_count must be greater than degree")
        self.basis_count = int(basis_count)
        self.degree = int(degree)
        interior_count = basis_count - degree - 1
        interior = torch.linspace(-1.0, 1.0, interior_count + 2)[1:-1]
        knots = torch.cat(
            [
                torch.full((degree + 1,), -1.0),
                interior,
                torch.full((degree + 1,), 1.0),
            ]
        )
        self.register_buffer("knots", knots)

    def forward(self, values):
        values = values.clamp(-1.0, 1.0 - 1e-6)
        knots = self.knots.to(device=values.device, dtype=values.dtype)
        expanded = values.unsqueeze(-1)
        basis = (
            (expanded >= knots[:-1]) & (expanded < knots[1:])
        ).to(values.dtype)
        knot_count = knots.numel()
        for degree in range(1, self.degree + 1):
            count = knot_count - degree - 1
            left_den = knots[degree:degree + count] - knots[:count]
            right_den = (
                knots[degree + 1:degree + 1 + count]
                - knots[1:1 + count]
            )
            left_den_safe = torch.where(
                left_den.abs() > 1e-7,
                left_den,
                torch.ones_like(left_den),
            )
            right_den_safe = torch.where(
                right_den.abs() > 1e-7,
                right_den,
                torch.ones_like(right_den),
            )
            left = (expanded - knots[:count]) / left_den_safe
            right = (
                knots[degree + 1:degree + 1 + count] - expanded
            ) / right_den_safe
            left = torch.where(
                left_den.abs() > 1e-7,
                left,
                torch.zeros_like(left),
            )
            right = torch.where(
                right_den.abs() > 1e-7,
                right,
                torch.zeros_like(right),
            )
            basis = (
                left * basis[..., :count]
                + right * basis[..., 1:count + 1]
            )
        return basis


class ContextResidualBase(nn.Module):
    def __init__(self, channels, groups=8, max_gain=0.25):
        super().__init__()
        self.channels = int(channels)
        self.groups = int(groups)
        self.group_channels = channels // groups
        self.max_gain = float(max_gain)
        self.descriptor = UltrasoundGroupDescriptor(channels, groups)
        self.mix = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def transform(self, descriptor):
        raise NotImplementedError

    def forward(self, features):
        descriptor = self.descriptor(features)
        channel_delta = self.transform(descriptor).reshape(
            features.size(0), self.channels, 1, 1
        )
        modulated = features * torch.tanh(channel_delta)
        correction = torch.tanh(self.mix(modulated))
        return features + self.max_gain * correction


class GroupedSplineKANContext(ContextResidualBase):
    """Cubic B-spline KAN over ultrasound group descriptors."""

    def __init__(
        self,
        channels,
        groups=8,
        basis_count=6,
        degree=3,
        max_gain=0.25,
    ):
        super().__init__(channels, groups, max_gain)
        self.basis = CubicBSplineBasis(basis_count, degree)
        self.base_weight = nn.Parameter(
            torch.empty(groups, self.group_channels, 4)
        )
        self.spline_weight = nn.Parameter(
            torch.empty(
                groups,
                self.group_channels,
                4,
                basis_count,
            )
        )
        self.bias = nn.Parameter(
            torch.zeros(groups, self.group_channels)
        )
        nn.init.normal_(self.base_weight, std=0.02)
        nn.init.normal_(self.spline_weight, std=0.01)

    def transform(self, descriptor):
        spline = self.basis(descriptor)
        base_value = torch.einsum(
            "bgd,gcd->bgc", F.silu(descriptor), self.base_weight
        )
        spline_value = torch.einsum(
            "bgdm,gcdm->bgc", spline, self.spline_weight
        )
        return base_value + spline_value + self.bias.unsqueeze(0)


class GroupedMLPContext(ContextResidualBase):
    """Parameter-matched nonlinear control for the spline context."""

    def __init__(
        self,
        channels,
        groups=8,
        hidden=24,
        max_gain=0.25,
    ):
        super().__init__(channels, groups, max_gain)
        self.weight1 = nn.Parameter(torch.empty(groups, 4, hidden))
        self.bias1 = nn.Parameter(torch.zeros(groups, hidden))
        self.weight2 = nn.Parameter(
            torch.empty(groups, hidden, self.group_channels)
        )
        self.bias2 = nn.Parameter(
            torch.zeros(groups, self.group_channels)
        )
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def transform(self, descriptor):
        hidden = torch.einsum(
            "bgd,gdh->bgh", descriptor, self.weight1
        )
        hidden = F.silu(hidden + self.bias1.unsqueeze(0))
        output = torch.einsum(
            "bgh,ghc->bgc", hidden, self.weight2
        )
        return output + self.bias2.unsqueeze(0)


class LiteEncoder(nn.Module):
    def __init__(self, widths, context_mode="none"):
        super().__init__()
        c1, c2, c3, c4, c5 = widths
        self.stem = ConvNormAct(3, c1, kernel_size=3, stride=2)
        self.stage1 = nn.Sequential(
            LiteResidualBlock(c1),
            LiteResidualBlock(c1),
        )
        self.down2 = LiteDownsample(c1, c2)
        self.stage2 = nn.Sequential(
            LiteResidualBlock(c2),
            LiteResidualBlock(c2),
        )
        self.down3 = LiteDownsample(c2, c3)
        self.stage3 = nn.Sequential(
            LiteResidualBlock(c3),
            LiteResidualBlock(c3),
        )
        self.down4 = LiteDownsample(c3, c4)
        self.stage4 = nn.Sequential(
            LiteResidualBlock(c4),
            LiteResidualBlock(c4),
        )
        self.down5 = LiteDownsample(c4, c5)
        self.stage5 = nn.Sequential(
            LiteResidualBlock(c5),
            LiteResidualBlock(c5),
        )

        if context_mode == "none":
            self.context4 = nn.Identity()
            self.context5 = nn.Identity()
        elif context_mode == "mlp":
            self.context4 = GroupedMLPContext(c4)
            self.context5 = GroupedMLPContext(c5)
        elif context_mode == "deep":
            self.context4 = nn.Identity()
            self.context5 = GroupedSplineKANContext(c5)
        elif context_mode == "full":
            self.context4 = GroupedSplineKANContext(c4)
            self.context5 = GroupedSplineKANContext(c5)
        else:
            raise ValueError(
                "Unknown LF-KAN context mode: {}".format(context_mode)
            )

    def forward(self, inputs):
        c1 = self.stage1(self.stem(inputs))
        c2 = self.stage2(self.down2(c1))
        c3 = self.stage3(self.down3(c2))
        c4 = self.context4(self.stage4(self.down4(c3)))
        c5 = self.context5(self.stage5(self.down5(c4)))
        return c1, c2, c3, c4, c5


class PyramidFuse(nn.Module):
    def __init__(self, in_channels, lateral_channels, out_channels):
        super().__init__()
        self.up_project = ConvNormAct(in_channels, out_channels)
        self.lateral_project = ConvNormAct(
            lateral_channels, out_channels
        )
        self.refine = nn.Sequential(
            ConvNormAct(
                out_channels,
                out_channels,
                kernel_size=3,
                groups=out_channels,
            ),
            ConvNormAct(out_channels, out_channels),
        )

    def forward(self, coarse, lateral):
        coarse = F.interpolate(
            self.up_project(coarse),
            size=lateral.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(coarse + self.lateral_project(lateral))


class LFKANNet(nn.Module):
    def __init__(
        self,
        out_channels=1,
        widths=(24, 48, 96, 192, 320),
        decoder_channels=96,
        context_mode="full",
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = widths
        self.encoder = LiteEncoder(widths, context_mode=context_mode)
        self.top = ConvNormAct(c5, decoder_channels)
        self.fuse4 = PyramidFuse(
            decoder_channels, c4, decoder_channels
        )
        self.fuse3 = PyramidFuse(
            decoder_channels, c3, decoder_channels
        )
        self.fuse2 = PyramidFuse(
            decoder_channels, c2, decoder_channels
        )
        self.fuse1 = PyramidFuse(
            decoder_channels, c1, decoder_channels
        )
        self.high_resolution = nn.Sequential(
            ConvNormAct(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                groups=decoder_channels,
            ),
            ConvNormAct(decoder_channels, 48),
        )
        self.classifier = nn.Conv2d(48, out_channels, 1)

    def forward(self, inputs):
        c1, c2, c3, c4, c5 = self.encoder(inputs)
        decoded = self.top(c5)
        decoded = self.fuse4(decoded, c4)
        decoded = self.fuse3(decoded, c3)
        decoded = self.fuse2(decoded, c2)
        decoded = self.fuse1(decoded, c1)
        decoded = F.interpolate(
            decoded,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        logits = self.classifier(self.high_resolution(decoded))
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return logits
        return torch.sigmoid(logits)


def build_lfkan_net(model_name, in_ch=3, out_ch=1):
    if in_ch != 3:
        raise ValueError("LF-KAN currently expects three input channels")
    variants = {
        "LFKAN_Base": "none",
        "LFKAN_MLP": "mlp",
        "LFKAN_Deep": "deep",
        "LFKAN_Net": "full",
    }
    if model_name not in variants:
        raise ValueError(
            "Unknown LF-KAN variant: {}".format(model_name)
        )
    return LFKANNet(
        out_channels=out_ch,
        context_mode=variants[model_name],
    )
