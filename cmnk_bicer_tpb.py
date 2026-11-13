from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from SE_UNet_MK_KAN_RPA import (
    DoubleConv,
    KANSelectiveKernel,
    MultiKernelRefineBlock,
)


class CoupledMultiScaleNonlinearKernel(nn.Module):
    """Condition nonlinear-kernel routing on the preceding scale state."""

    def __init__(self, channels: int):
        super().__init__()
        self.scale_refine = MultiKernelRefineBlock(channels)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        3,
                        padding=1,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ),
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
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        5,
                        padding=2,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        hidden = max(channels // 8, 16)
        self.state_reduce = nn.Conv2d(channels * 2, hidden, 1)
        self.route_expand = nn.Conv2d(
            hidden * 4,
            len(self.branches) * channels,
            1,
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        nn.init.zeros_(self.project[1].weight)

    def forward(self, x):
        scale_state = self.scale_refine(x)
        scale_residual = scale_state - x
        stacked = torch.stack(
            [branch(scale_state) for branch in self.branches],
            dim=1,
        )
        descriptor = F.adaptive_avg_pool2d(
            torch.cat([scale_state, scale_residual], dim=1),
            1,
        )
        z = self.state_reduce(descriptor)
        basis = torch.cat([z, z.square(), torch.sin(z), torch.cos(z)], dim=1)
        weights = self.route_expand(basis).view(
            x.size(0),
            len(self.branches),
            x.size(1),
            1,
            1,
        )
        weights = torch.softmax(weights, dim=1)
        mixed = (stacked * weights).sum(dim=1)
        return scale_state + self.project(mixed)


class ParallelMatchedMultiScaleNonlinearKernel(nn.Module):
    """Parameter-matched parallel control for the coupled CMNK operator."""

    def __init__(self, channels: int):
        super().__init__()
        self.multi_kernel = MultiKernelRefineBlock(channels)
        self.nonlinear_kernel = KANSelectiveKernel(channels)
        rank = max(int(round(channels / 24.0)), 4)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, rank, 1, bias=False),
            nn.BatchNorm2d(rank),
            nn.ReLU(inplace=True),
            nn.Conv2d(rank, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.fusion[-1].weight)

    def forward(self, x):
        multi_kernel_delta = self.multi_kernel(x) - x
        nonlinear_delta = self.nonlinear_kernel(x) - x
        correction = self.fusion(
            torch.cat([multi_kernel_delta, nonlinear_delta], dim=1)
        )
        return self.act(x + correction)


class SerialMultiScaleNonlinearKernel(nn.Module):
    """Simple serial control matching the historical MKR then KAN path."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            MultiKernelRefineBlock(channels),
            KANSelectiveKernel(channels),
        )

    def forward(self, x):
        return self.block(x)


def create_representation(mode: str, channels: int) -> nn.Module:
    if mode == "identity":
        return nn.Identity()
    if mode == "coupled":
        return CoupledMultiScaleNonlinearKernel(channels)
    if mode == "mk_only":
        return MultiKernelRefineBlock(channels)
    if mode == "nonlinear_only":
        return KANSelectiveKernel(channels)
    if mode == "serial":
        return SerialMultiScaleNonlinearKernel(channels)
    if mode == "parallel":
        return ParallelMatchedMultiScaleNonlinearKernel(channels)
    raise ValueError(f"Unknown representation mode: {mode}")


class LesionScaleConditionedCompetitionPrior(nn.Module):
    """Predict sample-specific competition temperature and prior residuals."""

    def __init__(
        self,
        hidden: int = 8,
        max_log_temperature_delta: float = 0.25,
        max_prior_delta: float = 0.25,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.max_log_temperature_delta = max_log_temperature_delta
        self.max_prior_delta = max_prior_delta
        self.eps = eps
        self.controller = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def forward(self, semantic, uncertainty, coherence, scale_noise):
        lesion_scale = torch.sqrt(
            semantic.mean(dim=(1, 2, 3)).clamp_min(self.eps)
        )
        uncertainty_level = uncertainty.mean(dim=(1, 2, 3))
        uncertainty_mass = uncertainty.sum(dim=(1, 2, 3)).clamp_min(self.eps)
        coherent_boundary = (coherence * uncertainty).sum(dim=(1, 2, 3)) / uncertainty_mass
        inconsistent_noise = (scale_noise * uncertainty).sum(dim=(1, 2, 3)) / uncertainty_mass
        descriptor = torch.stack(
            [
                lesion_scale,
                uncertainty_level,
                coherent_boundary,
                inconsistent_noise,
            ],
            dim=1,
        )
        control = self.controller(descriptor)
        log_temperature_delta = self.max_log_temperature_delta * torch.tanh(
            control[:, :1]
        )
        prior_delta = self.max_prior_delta * torch.tanh(control[:, 1:])
        return log_temperature_delta, prior_delta


class TemperaturePriorBiCER(nn.Module):
    """BiCER with learnable competition temperature and signed prior logits."""

    MODES = {
        "full",
        "recovery_only",
        "suppression_only",
        "independent_sigmoid",
    }

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        max_gain: float = 0.10,
        eps: float = 1e-6,
        mode: str = "full",
        use_decoder_semantics: bool = True,
        use_coherence: bool = True,
        use_scale_noise: bool = True,
        zero_init_heads: bool = True,
        learnable_global_prior: bool = True,
        use_dynamic_prior: bool = False,
        initial_recovery_bias: float = -0.10,
        initial_suppression_bias: float = 0.10,
        initial_temperature: float = 1.0,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown BiCER mode: {mode}")
        self.eps = eps
        self.max_gain = max_gain
        self.mode = mode
        self.use_decoder_semantics = use_decoder_semantics
        self.use_coherence = use_coherence
        self.use_scale_noise = use_scale_noise
        self.use_dynamic_prior = use_dynamic_prior
        hidden = max(channels // reduction, 8)

        self.skip_project = nn.Conv2d(channels, hidden, 1, bias=False)
        self.decoder_project = nn.Conv2d(channels, hidden, 1, bias=False)
        self.semantic_local = nn.Conv2d(hidden, 1, 1, bias=True)
        self.semantic_global = nn.Conv2d(hidden, 1, 1, bias=True)

        descriptor_channels = channels * 4 + hidden * 2
        self.channel_router = nn.Sequential(
            nn.Conv2d(descriptor_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 1, bias=True),
        )
        self.spatial_router = nn.Sequential(
            nn.Conv2d(6, hidden, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, bias=True),
        )

        self.interaction = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.recovery_head = nn.Conv2d(hidden, channels, 1, bias=True)
        self.suppression_head = nn.Conv2d(hidden, channels, 1, bias=True)

        log_temperature = torch.tensor(
            math.log(initial_temperature),
            dtype=torch.float32,
        )
        prior_logits = torch.tensor(
            [initial_recovery_bias, initial_suppression_bias],
            dtype=torch.float32,
        )
        if learnable_global_prior:
            self.log_temperature = nn.Parameter(log_temperature)
            self.prior_logits = nn.Parameter(prior_logits)
        else:
            self.register_buffer("log_temperature", log_temperature)
            self.register_buffer("prior_logits", prior_logits)
        self.dynamic_prior = (
            LesionScaleConditionedCompetitionPrior(hidden=max(hidden // 2, 8))
            if use_dynamic_prior
            else None
        )

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        if zero_init_heads:
            nn.init.zeros_(self.recovery_head.weight)
            nn.init.zeros_(self.recovery_head.bias)
            nn.init.zeros_(self.suppression_head.weight)
            nn.init.zeros_(self.suppression_head.bias)

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(0.5, 2.0)

    def _edge_magnitude(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

    def _normalize(self, x):
        maximum = x.amax(dim=(2, 3), keepdim=True)
        return x / (maximum + self.eps)

    def forward(self, decoder_feature, skip_feature):
        skip_context = self.skip_project(skip_feature)
        projected_decoder = self.decoder_project(decoder_feature)
        if self.use_decoder_semantics:
            decoder_context = projected_decoder
            semantic_source = decoder_context
        else:
            decoder_context = torch.zeros_like(projected_decoder)
            semantic_source = skip_context

        global_context = F.adaptive_avg_pool2d(semantic_source, 1)
        semantic = torch.sigmoid(
            self.semantic_local(semantic_source)
            + self.semantic_global(global_context)
        )
        uncertainty = 4.0 * semantic * (1.0 - semantic)

        gray = skip_context.mean(dim=1, keepdim=True)
        smooth3 = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        smooth5 = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        edge3 = self._edge_magnitude(smooth3)
        edge5 = self._edge_magnitude(smooth5)
        norm_edge3 = self._normalize(edge3)
        norm_edge5 = self._normalize(edge5)
        coherence = 1.0 - (norm_edge3 - norm_edge5).abs() / (
            norm_edge3 + norm_edge5 + self.eps
        )
        coherence = coherence.clamp(0.0, 1.0)
        scale_noise = self._normalize(
            (gray - smooth3).abs() + (smooth3 - smooth5).abs()
        )

        coherence_factor = coherence if self.use_coherence else 1.0
        incoherence_factor = (1.0 - coherence) if self.use_coherence else 1.0
        clean_factor = (1.0 - scale_noise) if self.use_scale_noise else 1.0
        noise_factor = scale_noise if self.use_scale_noise else 1.0
        recovery_evidence = (
            uncertainty * semantic * coherence_factor * clean_factor
        )
        suppression_evidence = (
            uncertainty
            * (1.0 - semantic)
            * noise_factor
            * incoherence_factor
        )

        skip_mean = F.adaptive_avg_pool2d(skip_feature, 1)
        skip_max = F.adaptive_max_pool2d(skip_feature, 1)
        skip_min = -F.adaptive_max_pool2d(-skip_feature, 1)
        skip_std = torch.sqrt(
            torch.var(
                skip_feature,
                dim=(2, 3),
                keepdim=True,
                unbiased=False,
            )
            + self.eps
        )
        if self.use_decoder_semantics:
            decoder_mean = F.adaptive_avg_pool2d(decoder_context, 1)
            decoder_std = torch.sqrt(
                torch.var(
                    decoder_context,
                    dim=(2, 3),
                    keepdim=True,
                    unbiased=False,
                )
                + self.eps
            )
        else:
            decoder_mean = torch.zeros_like(
                F.adaptive_avg_pool2d(skip_context, 1)
            )
            decoder_std = torch.zeros_like(decoder_mean)

        channel_logits = self.channel_router(
            torch.cat(
                [
                    skip_mean,
                    skip_max,
                    skip_min,
                    skip_std,
                    decoder_mean,
                    decoder_std,
                ],
                dim=1,
            )
        )
        channel_logits = channel_logits.view(
            skip_feature.size(0),
            2,
            skip_feature.size(1),
            1,
            1,
        )

        coherence_input = coherence if self.use_coherence else torch.zeros_like(coherence)
        scale_noise_input = (
            scale_noise if self.use_scale_noise else torch.zeros_like(scale_noise)
        )
        spatial_logits = self.spatial_router(
            torch.cat(
                [
                    recovery_evidence,
                    suppression_evidence,
                    semantic,
                    uncertainty,
                    coherence_input,
                    scale_noise_input,
                ],
                dim=1,
            )
        ).unsqueeze(2)

        prior = self.prior_logits.unsqueeze(0)
        log_temperature = self.log_temperature.reshape(1, 1)
        if self.dynamic_prior is not None:
            log_delta, prior_delta = self.dynamic_prior(
                semantic,
                uncertainty,
                coherence_input,
                scale_noise_input,
            )
            log_temperature = log_temperature + log_delta
            prior = prior + prior_delta
        prior = prior.expand(skip_feature.size(0), -1)
        log_temperature = log_temperature.expand(skip_feature.size(0), -1)
        prior = prior - prior.mean(dim=1, keepdim=True)
        prior = prior.view(skip_feature.size(0), 2, 1, 1, 1)
        temperature = log_temperature.exp().clamp(0.5, 2.0)
        temperature = temperature.view(skip_feature.size(0), 1, 1, 1, 1)
        logits = (channel_logits + spatial_logits + prior) / temperature

        interaction = self.interaction(
            torch.cat(
                [
                    skip_context,
                    decoder_context,
                    (skip_context - decoder_context).abs(),
                ],
                dim=1,
            )
        )
        recovery = torch.tanh(self.recovery_head(interaction))
        suppression = torch.tanh(self.suppression_head(interaction))

        if self.mode == "recovery_only":
            correction = torch.sigmoid(logits[:, 0]) * recovery
        elif self.mode == "suppression_only":
            correction = -torch.sigmoid(logits[:, 1]) * suppression
        elif self.mode == "independent_sigmoid":
            weights = torch.sigmoid(logits)
            correction = weights[:, 0] * recovery - weights[:, 1] * suppression
        else:
            weights = torch.softmax(logits, dim=1)
            correction = weights[:, 0] * recovery - weights[:, 1] * suppression

        return skip_feature + self.max_gain * uncertainty * correction


class IdentityBiCER(nn.Module):
    def forward(self, decoder_feature, skip_feature):
        return skip_feature


@dataclass(frozen=True)
class TPBVariant:
    representation: str = "coupled"
    bicer_mode: str = "full"
    use_decoder_semantics: bool = True
    use_coherence: bool = True
    use_scale_noise: bool = True
    zero_init_heads: bool = True
    learnable_global_prior: bool = True
    use_dynamic_prior: bool = True


TPB_VARIANTS = {
    "CMNK_BiCER_LSCP_UNet": TPBVariant(),
    "CMNK_BiCER_LSCP_RecoveryOnly": TPBVariant(bicer_mode="recovery_only"),
    "CMNK_BiCER_LSCP_SuppressionOnly": TPBVariant(bicer_mode="suppression_only"),
    "CMNK_BiCER_LSCP_IndependentSigmoid": TPBVariant(
        bicer_mode="independent_sigmoid"
    ),
    "CMNK_BiCER_LSCP_NoDecoder": TPBVariant(use_decoder_semantics=False),
    "CMNK_BiCER_LSCP_NoCoherence": TPBVariant(use_coherence=False),
    "CMNK_BiCER_LSCP_NoScaleNoise": TPBVariant(use_scale_noise=False),
    "CMNK_BiCER_LSCP_NonZeroInit": TPBVariant(zero_init_heads=False),
    "CMNK_MKOnly_BiCER_LSCP_UNet": TPBVariant(representation="mk_only"),
    "CMNK_NonlinearOnly_BiCER_LSCP_UNet": TPBVariant(
        representation="nonlinear_only"
    ),
    "CMNK_Serial_BiCER_LSCP_UNet": TPBVariant(representation="serial"),
    "CMNK_Parallel_BiCER_LSCP_UNet": TPBVariant(representation="parallel"),
    "CMNK_BiCER_FixedCompetition_UNet": TPBVariant(
        learnable_global_prior=False,
        use_dynamic_prior=False,
    ),
    "CMNK_BiCER_TPB_UNet": TPBVariant(use_dynamic_prior=False),
    "CMNK_Only_UNet": TPBVariant(
        bicer_mode="none",
        learnable_global_prior=False,
        use_dynamic_prior=False,
    ),
    "BiCER_TPB_Only_UNet": TPBVariant(
        representation="identity",
        use_dynamic_prior=False,
    ),
}
TPB_MODEL_BASE_NAMES = tuple(TPB_VARIANTS)


class TPBDecoderRefineFuse(nn.Module):
    def __init__(self, channels: int, variant: TPBVariant):
        super().__init__()
        self.up_refine = create_representation(variant.representation, channels)
        self.skip_refine = create_representation(variant.representation, channels)
        self.bicer = (
            IdentityBiCER()
            if variant.bicer_mode == "none"
            else TemperaturePriorBiCER(
                channels,
                mode=variant.bicer_mode,
                use_decoder_semantics=variant.use_decoder_semantics,
                use_coherence=variant.use_coherence,
                use_scale_noise=variant.use_scale_noise,
                zero_init_heads=variant.zero_init_heads,
                learnable_global_prior=variant.learnable_global_prior,
                use_dynamic_prior=variant.use_dynamic_prior,
                initial_recovery_bias=(
                    -0.10 if variant.learnable_global_prior else 0.0
                ),
                initial_suppression_bias=(
                    0.10 if variant.learnable_global_prior else 0.0
                ),
            )
        )
        self.fuse = DoubleConv(channels * 2, channels)
        self.out_refine = create_representation(variant.representation, channels)

    def forward(self, up_feature, skip_feature):
        up_feature = self.up_refine(up_feature)
        skip_feature = self.skip_refine(skip_feature)
        skip_feature = self.bicer(up_feature, skip_feature)
        fused = self.fuse(torch.cat([up_feature, skip_feature], dim=1))
        return self.out_refine(fused)


class CMNKBiCERTPBUNet(nn.Module):
    def __init__(self, in_ch, out_ch, channels, variant: TPBVariant):
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
        self.enhance5 = create_representation(variant.representation, c5)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.fuse6 = TPBDecoderRefineFuse(c4, variant)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.fuse7 = TPBDecoderRefineFuse(c3, variant)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.fuse8 = TPBDecoderRefineFuse(c2, variant)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.fuse9 = TPBDecoderRefineFuse(c1, variant)

        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))
        c6 = self.fuse6(self.up6(c5), c4)
        c7 = self.fuse7(self.up7(c6), c3)
        c8 = self.fuse8(self.up8(c7), c2)
        c9 = self.fuse9(self.up9(c8), c1)
        return self.last_activation(self.conv10(c9))


def create_tpb_model(base_name: str, in_ch: int, out_ch: int, channels):
    try:
        variant = TPB_VARIANTS[base_name]
    except KeyError as exc:
        raise ValueError(f"Unknown TPB model: {base_name}") from exc
    return CMNKBiCERTPBUNet(in_ch, out_ch, channels, variant)
