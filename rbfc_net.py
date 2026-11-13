"""Region-boundary fallback consensus network for BUS segmentation."""

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


class SemanticOccupancyPyramid(nn.Module):
    """Predict lesion occupancy from the three deepest semantic scales."""

    def __init__(self, channels, hidden=48):
        super().__init__()
        self.projections = nn.ModuleList(
            [ConvBNAct(channel, 32, 1) for channel in channels]
        )
        self.refine = nn.Sequential(
            ConvBNAct(96, hidden, 1),
            ResidualDWBlock(hidden),
            ResidualDWBlock(hidden),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, features, target_size):
        target = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            value = projection(feature)
            if value.shape[-2:] != target:
                value = F.interpolate(
                    value, size=target, mode="bilinear", align_corners=False
                )
            projected.append(value)
        logits = self.refine(torch.cat(projected, dim=1))
        return F.interpolate(
            logits, size=target_size, mode="bilinear", align_corners=False
        )


class LocalBoundaryContinuityStream(nn.Module):
    """Recover local contour evidence from shallow adjacent-resolution features."""

    def __init__(self, channels):
        super().__init__()
        c1, c2 = channels
        self.shallow = ConvBNAct(c1, 24, 1)
        self.context = ConvBNAct(c2, 24, 1)
        self.fuse = ConvBNAct(48, 32, 1)
        self.local = ConvBNAct(32, 32, 3, groups=32)
        self.broad = ConvBNAct(32, 32, 5, groups=32)
        self.refine = nn.Sequential(
            ConvBNAct(128, 32, 1),
            ResidualDWBlock(32),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, shallow, context):
        shallow = self.shallow(shallow)
        context = F.interpolate(
            self.context(context),
            size=shallow.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature = self.fuse(torch.cat([shallow, context], dim=1))
        local = self.local(feature)
        broad = self.broad(feature)
        disagreement = (local - broad).abs()
        return self.refine(torch.cat([feature, local, broad, disagreement], dim=1))


class AgreementReliabilityFallback(nn.Module):
    """Fuse two predictions and fall back wherever their evidence is unreliable."""

    def __init__(self, mode="full"):
        super().__init__()
        if mode not in {"region", "boundary", "no_agreement", "no_fallback", "full"}:
            raise ValueError(f"Unknown RBFC fusion mode: {mode}")
        self.mode = mode
        self.temperature_logit = (
            nn.Parameter(torch.tensor(0.0))
            if mode in {"no_fallback", "full"}
            else None
        )
        self.fallback_logit = (
            nn.Parameter(torch.tensor(0.0)) if mode == "full" else None
        )

    @staticmethod
    def _confidence(logits):
        probability = torch.sigmoid(logits)
        return probability, (2.0 * probability - 1.0).abs()

    def forward(self, baseline, region=None, boundary=None):
        if self.mode == "region":
            _, baseline_confidence = self._confidence(baseline)
            _, region_confidence = self._confidence(region)
            blend = 0.5 * region_confidence / (0.5 + 0.5 * baseline_confidence)
            return baseline + blend * (region - baseline)
        if self.mode == "boundary":
            _, baseline_confidence = self._confidence(baseline)
            _, boundary_confidence = self._confidence(boundary)
            blend = 0.5 * boundary_confidence / (0.5 + 0.5 * baseline_confidence)
            return baseline + blend * (boundary - baseline)

        region_probability, region_confidence = self._confidence(region)
        boundary_probability, boundary_confidence = self._confidence(boundary)
        if self.mode == "no_agreement":
            return 0.5 * (region + boundary)

        temperature = 0.35 + F.softplus(self.temperature_logit)
        weights = torch.softmax(
            torch.cat([region_confidence, boundary_confidence], dim=1) / temperature,
            dim=1,
        )
        candidate = weights[:, :1] * region + weights[:, 1:] * boundary
        if self.mode == "no_fallback":
            return candidate

        _, baseline_confidence = self._confidence(baseline)
        agreement = 1.0 - (region_probability - boundary_probability).abs()
        evidence = torch.maximum(region_confidence, boundary_confidence)
        reliability = agreement * (0.25 + 0.75 * evidence)
        baseline_guard = 0.5 + 0.5 * baseline_confidence
        strength = torch.sigmoid(self.fallback_logit)
        blend = strength * reliability / baseline_guard.clamp_min(0.25)
        return baseline + blend.clamp(0.0, 1.0) * (candidate - baseline)


class RBFCNet(nn.Module):
    """Lightweight region-boundary dual stream with reliability fallback."""

    MODES = {"base", "region", "boundary", "no_agreement", "no_fallback", "full"}

    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        channels=(32, 64, 128, 256, 512),
        mode="full",
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown RBFC mode: {mode}")
        self.mode = mode
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
        uses_baseline = mode in {"base", "region", "boundary", "full"}
        self.fusion = (
            CrossScaleHighResolutionFusion(channels, 48) if uses_baseline else None
        )
        self.base_head = (
            nn.Sequential(ConvBNAct(48, 32, 3), nn.Conv2d(32, out_ch, 1))
            if uses_baseline
            else None
        )

        uses_region = mode in {"region", "no_agreement", "no_fallback", "full"}
        uses_boundary = mode in {"boundary", "no_agreement", "no_fallback", "full"}
        self.region_stream = (
            SemanticOccupancyPyramid((c3, c4, c5)) if uses_region else None
        )
        self.boundary_stream = (
            LocalBoundaryContinuityStream((c1, c2)) if uses_boundary else None
        )
        self.consensus = None if mode == "base" else AgreementReliabilityFallback(mode)

    def forward(self, image):
        x1, _ = self.stem(image)
        x1 = self.stage1(x1)
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        x4 = self.stage4(self.down4(x3))
        x5 = self.stage5(self.down5(x4))
        baseline = None
        if self.fusion is not None:
            feature = self.fusion([x1, x2, x3, x4, x5])
            baseline = self.base_head(feature)
        if self.mode == "base":
            return torch.sigmoid(baseline)

        region = None
        boundary = None
        if self.region_stream is not None:
            region = self.region_stream([x3, x4, x5], image.shape[-2:])
        if self.boundary_stream is not None:
            boundary = self.boundary_stream(x1, x2)
        logits = self.consensus(baseline, region, boundary)
        return torch.sigmoid(logits)


def build_rbfc_net(
    in_ch=3,
    out_ch=1,
    channels=(32, 64, 128, 256, 512),
    mode="full",
):
    return RBFCNet(in_ch=in_ch, out_ch=out_ch, channels=channels, mode=mode)
