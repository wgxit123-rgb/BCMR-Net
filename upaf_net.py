import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from SE_UNet_MK_KAN_RPA import DoubleConv


def _group_count(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class PhasePreservingSpectralDecomposition(nn.Module):
    """Split features with smooth real masks and optional phase retention."""

    def __init__(
        self,
        mode="anisotropic",
        cutoff=0.28,
        temperature=0.08,
        preserve_phase=True,
    ):
        super().__init__()
        if mode not in {"isotropic", "anisotropic"}:
            raise ValueError("mode must be 'isotropic' or 'anisotropic'")
        self.mode = mode
        self.preserve_phase = bool(preserve_phase)
        normalized_cutoff = (cutoff - 0.08) / (0.58 - 0.08)
        normalized_cutoff = min(max(normalized_cutoff, 1e-4), 1.0 - 1e-4)
        self.cutoff_logit = nn.Parameter(
            torch.tensor(math.log(normalized_cutoff / (1.0 - normalized_cutoff)))
        )
        self.register_buffer(
            "temperature", torch.tensor(float(temperature)), persistent=False
        )

    @property
    def band_count(self):
        return 2 if self.mode == "isotropic" else 4

    def cutoff(self):
        return 0.08 + 0.50 * torch.sigmoid(self.cutoff_logit)

    def _frequency_masks(self, height, width, device, dtype):
        axial = torch.fft.fftfreq(height, device=device, dtype=dtype).abs() / 0.5
        lateral = torch.fft.rfftfreq(width, device=device, dtype=dtype) / 0.5
        axial = axial[:, None]
        lateral = lateral[None, :]
        radial = torch.sqrt(axial.square() + lateral.square()) / math.sqrt(2.0)
        cutoff = self.cutoff().to(device=device, dtype=dtype)
        temperature = self.temperature.to(device=device, dtype=dtype)
        low_score = (cutoff - radial) / temperature
        high_score = (radial - cutoff) / temperature

        if self.mode == "isotropic":
            scores = torch.stack([low_score, high_score], dim=0)
        else:
            direction_temperature = temperature * 1.5
            axial_score = high_score + (axial - lateral) / direction_temperature
            lateral_score = high_score + (lateral - axial) / direction_temperature
            nondirectional_score = (
                high_score - (axial - lateral).abs() / direction_temperature
            )
            scores = torch.stack(
                [low_score, axial_score, lateral_score, nondirectional_score],
                dim=0,
            )
        return torch.softmax(scores, dim=0)

    def forward(self, features):
        original_dtype = features.dtype
        work = features.float()
        height, width = work.shape[-2:]
        spectrum = torch.fft.rfft2(work, norm="ortho")
        masks = self._frequency_masks(
            height, width, work.device, work.dtype
        ).unsqueeze(1).unsqueeze(1)
        bands = []
        for index in range(self.band_count):
            band_spectrum = spectrum * masks[index]
            if not self.preserve_phase:
                magnitude = torch.sqrt(
                    band_spectrum.real.square()
                    + band_spectrum.imag.square()
                    + 1e-12
                )
                band_spectrum = torch.complex(
                    magnitude, torch.zeros_like(magnitude)
                )
            band = torch.fft.irfft2(
                band_spectrum, s=(height, width), norm="ortho"
            )
            bands.append(band.to(dtype=original_dtype))
        return tuple(bands)


class UltrasoundPhaseAnisotropicFrequencyResidual(nn.Module):
    """Decoder-conditioned spectral residual for ultrasound skip features."""

    def __init__(
        self,
        up_channels,
        skip_channels,
        mode="anisotropic",
        semantic_router=True,
        max_gain=0.25,
        preserve_phase=True,
        zero_initialize=True,
    ):
        super().__init__()
        hidden = max(8, min(32, skip_channels // 4))
        groups = _group_count(hidden)
        self.max_gain = float(max_gain)
        self.semantic_router = bool(semantic_router)
        self.skip_project = nn.Sequential(
            nn.Conv2d(skip_channels, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        self.up_project = nn.Sequential(
            nn.Conv2d(up_channels, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        self.decompose = PhasePreservingSpectralDecomposition(
            mode=mode,
            preserve_phase=preserve_phase,
        )
        band_count = self.decompose.band_count

        if self.semantic_router:
            router_hidden = max(hidden // 2, 8)
            self.channel_router = nn.Sequential(
                nn.Conv2d(hidden * (band_count + 1), router_hidden, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(router_hidden, hidden * band_count, 1),
            )
            self.spatial_router = nn.Sequential(
                nn.Conv2d(band_count + 2, 16, 3, padding=1, bias=False),
                nn.GroupNorm(4, 16),
                nn.SiLU(inplace=True),
                nn.Conv2d(16, band_count, 1),
            )
        else:
            self.band_logits = nn.Parameter(torch.zeros(band_count, hidden))

        self.interaction = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden, hidden, 3, padding=1, groups=hidden, bias=False
            ),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        self.delta = nn.Conv2d(hidden, skip_channels, 1)
        if zero_initialize:
            nn.init.zeros_(self.delta.weight)
        else:
            nn.init.kaiming_normal_(
                self.delta.weight, mode="fan_in", nonlinearity="linear"
            )
        nn.init.zeros_(self.delta.bias)

    def _route(self, bands, up_context):
        stacked = torch.stack(bands, dim=1)
        batch, band_count, channels, height, width = stacked.shape
        if not self.semantic_router:
            weights = torch.softmax(self.band_logits, dim=0)
            weights = weights.view(1, band_count, channels, 1, 1)
            return (stacked * weights).sum(dim=1)

        band_energy = [
            F.adaptive_avg_pool2d(band.abs(), 1) for band in bands
        ]
        decoder_descriptor = F.adaptive_avg_pool2d(up_context, 1)
        channel_logits = self.channel_router(
            torch.cat(band_energy + [decoder_descriptor], dim=1)
        )
        channel_logits = channel_logits.view(
            batch, band_count, channels, 1, 1
        )

        spatial_energy = [band.abs().mean(dim=1, keepdim=True) for band in bands]
        decoder_mean = up_context.mean(dim=1, keepdim=True)
        decoder_std = torch.sqrt(
            (up_context - decoder_mean).square().mean(dim=1, keepdim=True)
            + 1e-6
        )
        spatial_logits = self.spatial_router(
            torch.cat(spatial_energy + [decoder_mean, decoder_std], dim=1)
        ).unsqueeze(2)
        weights = torch.softmax(channel_logits + spatial_logits, dim=1)
        return (stacked * weights).sum(dim=1)

    def forward(self, up_feature, skip_feature):
        if up_feature.shape[-2:] != skip_feature.shape[-2:]:
            up_feature = F.interpolate(
                up_feature,
                size=skip_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        skip_context = self.skip_project(skip_feature)
        up_context = self.up_project(up_feature)
        bands = self.decompose(skip_context)
        routed = self._route(bands, up_context)
        disagreement = (routed - skip_context).abs()
        correction = torch.tanh(
            self.delta(
                self.interaction(
                    torch.cat(
                        [skip_context, routed, up_context, disagreement], dim=1
                    )
                )
            )
        )
        return skip_feature + self.max_gain * correction


class UPAFDecoderFuse(nn.Module):
    def __init__(
        self,
        channels,
        use_frequency,
        mode="anisotropic",
        semantic_router=True,
    ):
        super().__init__()
        self.adapter = (
            UltrasoundPhaseAnisotropicFrequencyResidual(
                channels,
                channels,
                mode=mode,
                semantic_router=semantic_router,
            )
            if use_frequency
            else None
        )
        self.fuse = DoubleConv(channels * 2, channels)

    def forward(self, up_feature, skip_feature):
        if self.adapter is not None:
            skip_feature = self.adapter(up_feature, skip_feature)
        return self.fuse(torch.cat([up_feature, skip_feature], dim=1))


class ChannelUPAFNet(nn.Module):
    """U-Net with phase-preserving axial/lateral spectral skip fusion."""

    def __init__(
        self,
        in_ch,
        out_ch,
        channels,
        use_frequency=True,
        mode="anisotropic",
        semantic_router=True,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.conv1 = DoubleConv(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(c4, c5)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = UPAFDecoderFuse(
            c4, use_frequency, mode, semantic_router
        )
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = UPAFDecoderFuse(
            c3, use_frequency, mode, semantic_router
        )
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = UPAFDecoderFuse(
            c2, use_frequency, mode, semantic_router
        )
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = UPAFDecoderFuse(
            c1, use_frequency, mode, semantic_router
        )
        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, inputs):
        c1 = self.conv1(inputs)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.conv5(self.pool4(c4))
        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        c9 = self.refine9(self.up9(c8), c1)
        logits = self.conv10(c9)
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return logits
        return self.last_activation(logits)


def build_upaf_net(model_name, in_ch=3, out_ch=1, channels=None):
    channels = channels or (32, 64, 128, 256, 512)
    variants = {
        "UPAF_Base": dict(use_frequency=False),
        "UPAF_Isotropic": dict(
            use_frequency=True, mode="isotropic", semantic_router=True
        ),
        "UPAF_Anisotropic": dict(
            use_frequency=True, mode="anisotropic", semantic_router=False
        ),
        "UPAF_Net": dict(
            use_frequency=True, mode="anisotropic", semantic_router=True
        ),
        "UNet_FREQ": dict(
            use_frequency=True, mode="anisotropic", semantic_router=False
        ),
    }
    if model_name not in variants:
        raise ValueError("Unknown UPAF variant: {}".format(model_name))
    return ChannelUPAFNet(
        in_ch, out_ch, channels, **variants[model_name]
    )
