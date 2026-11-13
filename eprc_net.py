"""Contrast-adaptive morphology-prototype network for BUS segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualDWBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, 3, groups=channels),
            ConvBNAct(channels, hidden, 1),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class ContrastAdaptiveFieldStem(nn.Module):
    """Encode raw echoes together with multi-scale local contrast fields."""

    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        # RGB/BGR image plus three local means, two standardized residuals,
        # and two local contrast maps.
        self.project = nn.Sequential(
            ConvBNAct(in_channels + 7, out_channels, 3),
            ResidualDWBlock(out_channels),
        )
        self.ambiguity_refine = nn.Sequential(
            ConvBNAct(3, max(out_channels // 2, 8), 3),
            nn.Conv2d(max(out_channels // 2, 8), 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        gray = image.mean(dim=1, keepdim=True).clamp_min(0.0)
        mean_3 = F.avg_pool2d(gray, 3, stride=1, padding=1)
        mean_7 = F.avg_pool2d(gray, 7, stride=1, padding=3)
        mean_15 = F.avg_pool2d(gray, 15, stride=1, padding=7)
        second_3 = F.avg_pool2d(gray.square(), 3, stride=1, padding=1)
        second_7 = F.avg_pool2d(gray.square(), 7, stride=1, padding=3)
        contrast_3 = (second_3 - mean_3.square()).clamp_min(1e-6).sqrt()
        contrast_7 = (second_7 - mean_7.square()).clamp_min(1e-6).sqrt()
        standardized_3 = ((gray - mean_3) / (contrast_3 + 0.03)).clamp(-3.0, 3.0) / 3.0
        standardized_7 = ((gray - mean_7) / (contrast_7 + 0.03)).clamp(-3.0, 3.0) / 3.0
        fields = torch.cat(
            [
                image,
                mean_3,
                mean_7,
                mean_15,
                standardized_3,
                standardized_7,
                contrast_3,
                contrast_7,
            ],
            dim=1,
        )
        feature = self.project(fields)
        contrast = 0.5 * (contrast_3 + contrast_7)
        normalized_contrast = contrast / (
            contrast.amax(dim=(2, 3), keepdim=True) + 1e-6
        )
        deterministic_ambiguity = 1.0 - normalized_contrast.clamp(0.0, 1.0)
        learned_ambiguity = self.ambiguity_refine(
            torch.cat(
                [contrast_3, contrast_7, (standardized_3 - standardized_7).abs()],
                dim=1,
            )
        )
        ambiguity = 0.5 * (deterministic_ambiguity + learned_ambiguity)
        return feature, ambiguity


class PlainImageStem(nn.Module):
    """Parameter-matched conventional stem used by EPRC ablations."""

    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.project = nn.Sequential(
            ConvBNAct(in_channels, out_channels, 3),
            ResidualDWBlock(out_channels),
        )

    def forward(self, image):
        feature = self.project(image)
        ambiguity = torch.zeros(
            image.size(0),
            1,
            image.size(2),
            image.size(3),
            dtype=image.dtype,
            device=image.device,
        )
        return feature, ambiguity


class PyramidContextBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        branch_channels = channels // 4
        self.branches = nn.ModuleList(
            [
                ConvBNAct(channels, branch_channels, 3, groups=1),
                nn.Sequential(
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
                    nn.Conv2d(channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.SiLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        3,
                        padding=4,
                        dilation=4,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.SiLU(inplace=True),
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channels, branch_channels, 1, bias=False),
                    nn.SiLU(inplace=True),
                ),
            ]
        )
        self.fuse = nn.Sequential(
            ConvBNAct(branch_channels * 4, channels, 1),
            ResidualDWBlock(channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        features = []
        for index, branch in enumerate(self.branches):
            value = branch(x)
            if index == 3:
                value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
            features.append(value)
        return self.fuse(torch.cat(features, dim=1))


class CrossScaleHighResolutionFusion(nn.Module):
    """Fuse a semantic pyramid into one persistent high-resolution contour stream."""

    def __init__(self, channels=(32, 64, 128, 256, 512), output_channels=48):
        super().__init__()
        self.projections = nn.ModuleList(
            [ConvBNAct(channel, output_channels, 1) for channel in channels]
        )
        hidden = max(output_channels // 2, 16)
        self.router = nn.Sequential(
            nn.Linear(sum(channels), hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, len(channels)),
        )
        self.refine = nn.Sequential(
            ResidualDWBlock(output_channels),
            ResidualDWBlock(output_channels),
        )

    def forward(self, features):
        target_size = features[0].shape[-2:]
        descriptor = torch.cat(
            [feature.mean(dim=(2, 3)) for feature in features], dim=1
        )
        weights = torch.softmax(self.router(descriptor), dim=1)
        fused = 0.0
        for index, (feature, projection) in enumerate(zip(features, self.projections)):
            value = projection(feature)
            if value.shape[-2:] != target_size:
                value = F.interpolate(
                    value, size=target_size, mode="bilinear", align_corners=False
                )
            fused = fused + weights[:, index : index + 1, None, None] * value
        return self.refine(fused)


class MorphologyPrototypeRouter(nn.Module):
    """Route pixels using lesion-core, boundary and background prototypes."""

    def __init__(self, channels):
        super().__init__()
        self.embedding = ConvBNAct(channels, channels, 1)
        self.morphology_router = nn.Sequential(
            nn.Linear(4, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 3),
        )
        self.residual = nn.Sequential(
            ConvBNAct(channels + 6, channels, 3),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        # The residual projection is zero initialized, while a small nonzero gain
        # keeps gradients flowing through the branch from the first update.
        self.gain = nn.Parameter(torch.tensor(0.10))

    @staticmethod
    def _prototype(embedding, weight, eps=1e-6):
        numerator = (embedding * weight).sum(dim=(2, 3), keepdim=True)
        denominator = weight.sum(dim=(2, 3), keepdim=True) + eps
        return F.normalize(numerator / denominator, dim=1, eps=eps)

    def forward(self, feature, coarse_logits, ambiguity):
        probability = torch.sigmoid(coarse_logits)
        uncertainty = 4.0 * probability * (1.0 - probability)
        reliability = 1.0 - ambiguity
        embedding = F.normalize(self.embedding(feature), dim=1, eps=1e-6)
        core_weight = torch.sigmoid((probability - 0.65) / 0.10) * reliability
        boundary_weight = uncertainty * (0.5 + 0.5 * ambiguity)
        background_weight = (1.0 - probability) * reliability * (1.0 - uncertainty)
        core = self._prototype(embedding, core_weight)
        boundary = self._prototype(embedding, boundary_weight)
        background = self._prototype(embedding, background_weight)
        affinities = torch.cat(
            [
                (embedding * core).sum(dim=1, keepdim=True),
                (embedding * boundary).sum(dim=1, keepdim=True),
                (embedding * background).sum(dim=1, keepdim=True),
            ],
            dim=1,
        )
        dilation = F.max_pool2d(probability, 3, stride=1, padding=1)
        erosion = -F.max_pool2d(-probability, 3, stride=1, padding=1)
        soft_perimeter = (dilation - erosion).mean(dim=(2, 3))
        descriptor = torch.cat(
            [
                probability.mean(dim=(2, 3)),
                uncertainty.mean(dim=(2, 3)),
                ambiguity.mean(dim=(2, 3)),
                soft_perimeter,
            ],
            dim=1,
        )
        morphology_bias = self.morphology_router(descriptor).unsqueeze(-1).unsqueeze(-1)
        prototype_weights = torch.softmax(4.0 * affinities + morphology_bias, dim=1)
        evidence = torch.cat(
            [feature, probability, uncertainty, prototype_weights, reliability], dim=1
        )
        residual = torch.tanh(self.residual(evidence))
        return feature + 0.25 * torch.tanh(self.gain) * residual


class TopologyConstrainedContourEvolution(nn.Module):
    """Evolve uncertain contours while suppressing disconnected support."""

    def __init__(self, channels, steps=3):
        super().__init__()
        self.steps = steps
        self.update = nn.Sequential(
            ConvBNAct(channels + 8, channels, 3),
            ConvBNAct(channels, channels, 3, groups=channels),
            nn.Conv2d(channels, 1, 1, bias=True),
        )
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        self.step_gain = nn.Parameter(torch.tensor(0.10))
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian", laplacian)

    def forward(self, feature, initial_logits, ambiguity):
        logits = initial_logits
        reliability = 1.0 - ambiguity
        for _ in range(self.steps):
            probability = torch.sigmoid(logits)
            uncertainty = 4.0 * probability * (1.0 - probability)
            dilation = F.max_pool2d(probability, 3, stride=1, padding=1)
            erosion = -F.max_pool2d(-probability, 3, stride=1, padding=1)
            outer_band = (dilation - probability).clamp_min(0.0)
            inner_band = (probability - erosion).clamp_min(0.0)
            curvature = F.conv2d(probability, self.laplacian, padding=1)
            core = torch.sigmoid((probability - 0.70) / 0.08)
            traversable = (probability + uncertainty).clamp(0.0, 1.0)
            connected_support = core
            for _ in range(3):
                expanded = F.max_pool2d(
                    connected_support, 3, stride=1, padding=1
                )
                connected_support = torch.maximum(
                    core, expanded * traversable
                )
            orphan_response = (probability - connected_support).clamp_min(0.0)
            update = self.update(
                torch.cat(
                    [
                        feature,
                        probability,
                        uncertainty,
                        outer_band,
                        inner_band,
                        curvature,
                        reliability,
                        connected_support,
                        orphan_response,
                    ],
                    dim=1,
                )
            )
            logits = logits + 0.5 * torch.tanh(self.step_gain) * uncertainty * torch.tanh(update)
        return logits


class CAMPNet(nn.Module):
    """Contrast-adaptive morphology-prototype network with a persistent HR stream."""

    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        channels=(32, 64, 128, 256, 512),
        use_contrast_fields=True,
        use_prototype_router=True,
        use_contour_evolution=True,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.use_prototype_router = use_prototype_router
        self.use_contour_evolution = use_contour_evolution
        self.contrast_stem = (
            ContrastAdaptiveFieldStem(in_ch, c1)
            if use_contrast_fields
            else PlainImageStem(in_ch, c1)
        )
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
        self.high_resolution_fusion = CrossScaleHighResolutionFusion(channels, 48)
        self.coarse_head = nn.Sequential(
            ConvBNAct(48, 32, 3),
            nn.Conv2d(32, out_ch, 1),
        )
        self.morphology_router = (
            MorphologyPrototypeRouter(48) if use_prototype_router else None
        )
        self.contour_evolution = (
            TopologyConstrainedContourEvolution(48)
            if use_contour_evolution
            else None
        )

    def forward(self, image):
        x1, ambiguity = self.contrast_stem(image)
        x1 = self.stage1(x1)
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        x4 = self.stage4(self.down4(x3))
        x5 = self.stage5(self.down5(x4))
        high_resolution = self.high_resolution_fusion([x1, x2, x3, x4, x5])
        coarse_logits = self.coarse_head(high_resolution)
        routed_feature = high_resolution
        logits = coarse_logits
        if self.use_prototype_router:
            routed_feature = self.morphology_router(
                routed_feature, logits, ambiguity
            )
            logits = self.coarse_head(routed_feature)
        if self.use_contour_evolution:
            logits = self.contour_evolution(
                routed_feature, logits, ambiguity
            )
        return torch.sigmoid(logits)


def build_camp_net(
    in_ch=3,
    out_ch=1,
    channels=(32, 64, 128, 256, 512),
    use_contrast_fields=True,
    use_prototype_router=True,
    use_contour_evolution=True,
):
    return CAMPNet(
        in_ch=in_ch,
        out_ch=out_ch,
        channels=channels,
        use_contrast_fields=use_contrast_fields,
        use_prototype_router=use_prototype_router,
        use_contour_evolution=use_contour_evolution,
    )
