import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from SE_UNet_MK_KAN_RPA import (
    DoubleConv,
    KANSelectiveKernel,
    MultiKernelRefineBlock,
)
from upaf_net import (
    PhasePreservingSpectralDecomposition,
    UltrasoundPhaseAnisotropicFrequencyResidual,
)


def _group_count(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class SpatialContourEvidence(nn.Module):
    """Four-band spatial proxy used when DPSD is deliberately removed."""

    def forward(self, features):
        low = F.avg_pool2d(
            features,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        axial_high = features - F.avg_pool2d(
            features,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0),
        )
        lateral_high = features - F.avg_pool2d(
            features,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2),
        )
        nondirectional_high = (
            features - low - axial_high - lateral_high
        )
        return low, axial_high, lateral_high, nondirectional_high


class ControlledSpatialDPSDEvidence(nn.Module):
    """Keep spatial evidence primary and add a gated DPSD residual."""

    def __init__(self, channels, max_residual=1.0):
        super().__init__()
        hidden = max(8, min(16, channels // 8))
        groups = _group_count(hidden)
        self.max_residual = float(max_residual)
        self.spatial = SpatialContourEvidence()
        self.spectral = PhasePreservingSpectralDecomposition(
            mode="anisotropic",
            preserve_phase=True,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(8, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4, 1, bias=True),
        )
        self.band_scales = nn.Parameter(torch.zeros(4))

        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, features, return_diagnostics=False):
        spatial_bands = self.spatial(features)
        spectral_bands = self.spectral(features)
        disagreement = [
            (spectral - spatial).abs().mean(dim=1, keepdim=True)
            for spatial, spectral in zip(
                spatial_bands,
                spectral_bands,
            )
        ]
        spectral_energy = [
            spectral.abs().mean(dim=1, keepdim=True)
            for spectral in spectral_bands
        ]
        gates = torch.sigmoid(
            self.gate(
                torch.cat(disagreement + spectral_energy, dim=1)
            )
        )
        scales = (
            self.max_residual
            * torch.tanh(self.band_scales)
        ).view(1, 4, 1, 1)
        fused_bands = tuple(
            spatial
            + scales[:, index : index + 1]
            * gates[:, index : index + 1]
            * (spectral - spatial)
            for index, (spatial, spectral) in enumerate(
                zip(spatial_bands, spectral_bands)
            )
        )
        if return_diagnostics:
            return fused_bands, gates, scales
        return fused_bands


class UncertaintyGatedDPSDLogitCorrection(nn.Module):
    """Calibrate uncertain boundary logits without changing PSCA features."""

    def __init__(self, channels, max_logit_shift=1.5):
        super().__init__()
        hidden = max(8, min(24, channels // 2))
        groups = _group_count(hidden)
        self.max_logit_shift = float(max_logit_shift)
        self.decompose = PhasePreservingSpectralDecomposition(
            mode="anisotropic",
            preserve_phase=True,
        )
        self.band_project = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        cue_channels = hidden + 7
        self.reliability_gate = nn.Sequential(
            nn.Conv2d(cue_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        self.delta_head = nn.Sequential(
            nn.Conv2d(cue_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.zeros_(self.reliability_gate[-1].bias)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    @staticmethod
    def _boundary_cue(probability):
        horizontal = F.pad(
            (probability[:, :, :, 1:] - probability[:, :, :, :-1]).abs(),
            (0, 1, 0, 0),
        )
        vertical = F.pad(
            (probability[:, :, 1:, :] - probability[:, :, :-1, :]).abs(),
            (0, 0, 0, 1),
        )
        return torch.clamp(horizontal + vertical, 0.0, 1.0)

    def forward(self, features, base_logits, return_diagnostics=False):
        bands = self.decompose(features)
        spectral_context = self.band_project(torch.cat(bands, dim=1))
        band_energy = torch.cat(
            [
                band.abs().mean(dim=1, keepdim=True)
                for band in bands
            ],
            dim=1,
        )
        probability = torch.sigmoid(base_logits)
        uncertainty = 4.0 * probability * (1.0 - probability)
        boundary = self._boundary_cue(probability)
        cues = torch.cat(
            [
                spectral_context,
                band_energy,
                probability,
                uncertainty,
                boundary,
            ],
            dim=1,
        )
        reliability = torch.sigmoid(self.reliability_gate(cues))
        raw_delta = self.delta_head(cues) + self.global_bias.view(1, 1, 1, 1)
        correction = (
            self.max_logit_shift
            * uncertainty
            * reliability
            * torch.tanh(raw_delta)
        )
        calibrated_logits = base_logits + correction
        if return_diagnostics:
            return calibrated_logits, correction, reliability, uncertainty
        return calibrated_logits


class PhaseSemanticContourArbitration(nn.Module):
    """Arbitrate skip recovery and suppression from phase and semantics."""

    def __init__(
        self,
        channels,
        max_gain=0.10,
        evidence_mode="dpsd",
        use_prior=True,
        router_mode="ternary",
        use_suppression=True,
        use_conflict=True,
        use_hierarchical_consensus=False,
        use_signed_hierarchical_calibration=False,
        use_disagreement_reconciliation=False,
        max_consensus_gain=0.50,
        initial_consensus_gain=0.05,
        disagreement_scale=0.05,
        confidence_margin=0.03,
        confidence_temperature=0.05,
    ):
        super().__init__()
        if evidence_mode not in {"dpsd", "spatial", "dual"}:
            raise ValueError(
                "evidence_mode must be 'dpsd', 'spatial', or 'dual'"
            )
        if router_mode not in {"ternary", "binary"}:
            raise ValueError("router_mode must be 'ternary' or 'binary'")
        hidden = max(8, min(32, channels // 4))
        groups = _group_count(hidden)
        self.max_gain = float(max_gain)
        self.evidence_mode = evidence_mode
        self.use_prior = bool(use_prior)
        self.router_mode = router_mode
        self.use_suppression = bool(use_suppression)
        self.use_conflict = bool(use_conflict)
        self.use_hierarchical_consensus = bool(
            use_hierarchical_consensus
        )
        self.use_signed_hierarchical_calibration = bool(
            use_signed_hierarchical_calibration
        )
        self.use_disagreement_reconciliation = bool(
            use_disagreement_reconciliation
        )
        if (
            sum(
                (
                    self.use_hierarchical_consensus,
                    self.use_signed_hierarchical_calibration,
                    self.use_disagreement_reconciliation,
                )
            )
            > 1
        ):
            raise ValueError(
                "hierarchical reconciliation modes are mutually exclusive"
            )
        self.max_consensus_gain = float(max_consensus_gain)
        self.disagreement_scale = float(disagreement_scale)
        self.confidence_margin = float(confidence_margin)
        self.confidence_temperature = float(confidence_temperature)
        if self.disagreement_scale <= 0.0:
            raise ValueError("disagreement_scale must be positive")
        if self.confidence_temperature <= 0.0:
            raise ValueError("confidence_temperature must be positive")
        if self.use_signed_hierarchical_calibration:
            self.consensus_logit = nn.Parameter(torch.zeros(()))
        elif self.use_hierarchical_consensus:
            if not (
                0.0
                < float(initial_consensus_gain)
                < self.max_consensus_gain
            ):
                raise ValueError(
                    "initial_consensus_gain must be inside "
                    "(0, max_consensus_gain)"
                )
            initial_ratio = (
                float(initial_consensus_gain)
                / self.max_consensus_gain
            )
            self.consensus_logit = nn.Parameter(
                torch.logit(torch.tensor(initial_ratio))
            )
        else:
            self.register_parameter("consensus_logit", None)
        if self.use_disagreement_reconciliation:
            reconcile_hidden = 12
            self.reconciliation_router = nn.Sequential(
                nn.Conv2d(8, reconcile_hidden, 1, bias=False),
                nn.GroupNorm(4, reconcile_hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(reconcile_hidden, 1, 1, bias=True),
            )
            nn.init.zeros_(self.reconciliation_router[-1].weight)
            nn.init.constant_(self.reconciliation_router[-1].bias, -1.3862944)
        else:
            self.reconciliation_router = None

        self.skip_project = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        self.decoder_project = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        if evidence_mode == "dpsd":
            self.decompose = PhasePreservingSpectralDecomposition(
                mode="anisotropic",
                preserve_phase=True,
            )
        elif evidence_mode == "dual":
            self.decompose = ControlledSpatialDPSDEvidence(
                hidden,
                max_residual=1.0,
            )
        else:
            self.decompose = SpatialContourEvidence()
        self.phase_fuse = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        self.interaction = nn.Sequential(
            nn.Conv2d(hidden * 5, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        self.state_router = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, 1, bias=True),
        )
        self.recovery_head = nn.Conv2d(hidden, channels, 1, bias=True)
        self.suppression_head = nn.Conv2d(hidden, channels, 1, bias=True)

        nn.init.zeros_(self.state_router[-1].weight)
        with torch.no_grad():
            self.state_router[-1].bias.copy_(
                torch.tensor([1.0, 0.0, 0.0])
            )
        nn.init.zeros_(self.recovery_head.weight)
        nn.init.zeros_(self.recovery_head.bias)
        nn.init.zeros_(self.suppression_head.weight)
        nn.init.zeros_(self.suppression_head.bias)

    @staticmethod
    def _similarity(left, right):
        similarity = F.cosine_similarity(left, right, dim=1).unsqueeze(1)
        return 0.5 * (similarity + 1.0)

    @staticmethod
    def _normalized_state_entropy(states):
        probabilities = states.clamp_min(1e-6)
        return -(
            probabilities * probabilities.log()
        ).sum(dim=1, keepdim=True) / torch.log(
            probabilities.new_tensor(3.0)
        )

    @staticmethod
    def _normalized_state_disagreement(local, prior):
        midpoint = 0.5 * (local + prior)
        local_kl = (
            local.clamp_min(1e-6)
            * (
                local.clamp_min(1e-6).log()
                - midpoint.clamp_min(1e-6).log()
            )
        ).sum(dim=1, keepdim=True)
        prior_kl = (
            prior.clamp_min(1e-6)
            * (
                prior.clamp_min(1e-6).log()
                - midpoint.clamp_min(1e-6).log()
            )
        ).sum(dim=1, keepdim=True)
        return (0.5 * (local_kl + prior_kl) / torch.log(
            local.new_tensor(2.0)
        )).clamp(0.0, 1.0)

    def forward(
        self,
        decoder_feature,
        skip_feature,
        return_states=False,
        state_prior=None,
    ):
        if decoder_feature.shape[-2:] != skip_feature.shape[-2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=skip_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        skip_context = self.skip_project(skip_feature)
        decoder_context = self.decoder_project(decoder_feature)
        bands = self.decompose(skip_context)
        phase_context = self.phase_fuse(torch.cat(bands, dim=1))
        high_context = bands[1] + bands[2] + bands[3]

        semantic_agreement = self._similarity(
            skip_context,
            decoder_context,
        )
        phase_agreement = self._similarity(
            skip_context - bands[0],
            high_context,
        )
        semantic_uncertainty = 1.0 - semantic_agreement
        phase_uncertainty = 1.0 - phase_agreement
        cross_domain_conflict = (
            semantic_agreement - phase_agreement
        ).abs()
        if not self.use_conflict:
            cross_domain_conflict = torch.zeros_like(
                cross_domain_conflict
            )

        router_features = torch.cat(
            [
                semantic_agreement,
                phase_agreement,
                semantic_uncertainty,
                phase_uncertainty,
                cross_domain_conflict,
            ],
            dim=1,
        )
        learned_logits = self.state_router(router_features)
        prior_logits = torch.cat(
            [
                semantic_agreement * phase_agreement,
                semantic_uncertainty * phase_agreement,
                semantic_uncertainty * phase_uncertainty,
            ],
            dim=1,
        )
        if not self.use_prior:
            prior_logits = torch.zeros_like(prior_logits)
        state_logits = learned_logits + prior_logits
        if self.router_mode == "binary":
            routed = torch.softmax(state_logits[:, 1:3], dim=1)
            states = torch.cat(
                [torch.zeros_like(routed[:, 0:1]), routed],
                dim=1,
            )
        else:
            states = torch.softmax(state_logits, dim=1)

        if (
            (
                self.use_hierarchical_consensus
                or self.use_signed_hierarchical_calibration
                or self.use_disagreement_reconciliation
            )
            and state_prior is not None
        ):
            consensus = F.interpolate(
                state_prior.detach(),
                size=states.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp_min(1e-6)
            consensus = consensus / consensus.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-6)
            local_entropy = self._normalized_state_entropy(states)
            prior_confidence = (
                1.0
                - self._normalized_state_entropy(consensus)
            )
            disagreement = self._normalized_state_disagreement(
                states,
                consensus,
            )
            if self.use_disagreement_reconciliation:
                local_confidence = 1.0 - local_entropy
                confidence_advantage = (
                    prior_confidence - local_confidence
                )
                disagreement_signal = 1.0 - torch.exp(
                    -disagreement / self.disagreement_scale
                )
                eligibility = torch.sigmoid(
                    (
                        confidence_advantage - self.confidence_margin
                    )
                    / self.confidence_temperature
                )
                router_features = torch.cat(
                    [
                        local_entropy,
                        1.0 - prior_confidence,
                        disagreement,
                        confidence_advantage,
                        semantic_uncertainty,
                        phase_uncertainty,
                        cross_domain_conflict,
                        states.amax(dim=1, keepdim=True),
                    ],
                    dim=1,
                )
                learned_gate = torch.sigmoid(
                    self.reconciliation_router(router_features)
                )
                consensus_gain = (
                    self.max_consensus_gain
                    * learned_gate
                    * disagreement_signal
                    * eligibility
                ).clamp(0.0, self.max_consensus_gain)
                states = (
                    (1.0 - consensus_gain) * states
                    + consensus_gain * consensus
                )
            else:
                reliability = prior_confidence * (
                    0.5 * local_entropy
                    + 0.5 * disagreement
                )
            if self.use_signed_hierarchical_calibration:
                signed_gain = (
                    self.max_consensus_gain
                    * torch.tanh(self.consensus_logit)
                    * (0.5 + reliability)
                )
                states = torch.softmax(
                    states.clamp_min(1e-6).log()
                    + signed_gain
                    * consensus.clamp_min(1e-6).log(),
                    dim=1,
                )
            elif self.use_hierarchical_consensus:
                base_consensus_gain = (
                    self.max_consensus_gain
                    * torch.sigmoid(self.consensus_logit)
                )
                consensus_gain = (
                    base_consensus_gain
                    * (0.5 + reliability)
                ).clamp(max=self.max_consensus_gain)
                states = (
                    (1.0 - consensus_gain) * states
                    + consensus_gain * consensus
                )

        interaction = self.interaction(
            torch.cat(
                [
                    skip_context,
                    decoder_context,
                    phase_context,
                    (skip_context - decoder_context).abs(),
                    (high_context - decoder_context).abs(),
                ],
                dim=1,
            )
        )
        recovery = torch.tanh(self.recovery_head(interaction))
        suppression = torch.tanh(self.suppression_head(interaction))
        correction = states[:, 1:2] * recovery
        if self.use_suppression:
            correction = (
                correction - states[:, 2:3] * suppression
            )
        output = skip_feature + self.max_gain * correction
        if return_states:
            return output, states
        return output


class CrossDomainAdaptiveConsensus(nn.Module):
    """Reconcile independent spatial and spectral arbitration decisions."""

    def __init__(
        self,
        channels,
        fusion_mode="adaptive",
        disagreement_scale=0.05,
        confidence_temperature=0.10,
    ):
        super().__init__()
        if fusion_mode not in {
            "adaptive",
            "equal",
            "no_disagreement",
            "no_confidence",
        }:
            raise ValueError("Unknown cross-domain fusion mode")
        if disagreement_scale <= 0.0:
            raise ValueError("disagreement_scale must be positive")
        if confidence_temperature <= 0.0:
            raise ValueError("confidence_temperature must be positive")
        self.fusion_mode = fusion_mode
        self.disagreement_scale = float(disagreement_scale)
        self.confidence_temperature = float(confidence_temperature)
        self.spatial = PhaseSemanticContourArbitration(
            channels,
            evidence_mode="spatial",
        )
        # Keep the shared PSCA path on the exact matched-seed initialization.
        with torch.random.fork_rng(devices=[]):
            self.spectral = PhaseSemanticContourArbitration(
                channels,
                evidence_mode="dpsd",
            )
            hidden = 16
            self.domain_router = nn.Sequential(
                nn.Conv2d(8, hidden, 3, padding=1, bias=False),
                nn.GroupNorm(4, hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 1, 1, bias=True),
            )
            nn.init.zeros_(self.domain_router[-1].weight)
            nn.init.zeros_(self.domain_router[-1].bias)

    @staticmethod
    def _state_disagreement(left, right):
        midpoint = 0.5 * (left + right)
        left_kl = (
            left.clamp_min(1e-6)
            * (
                left.clamp_min(1e-6).log()
                - midpoint.clamp_min(1e-6).log()
            )
        ).sum(dim=1, keepdim=True)
        right_kl = (
            right.clamp_min(1e-6)
            * (
                right.clamp_min(1e-6).log()
                - midpoint.clamp_min(1e-6).log()
            )
        ).sum(dim=1, keepdim=True)
        return (0.5 * (left_kl + right_kl) / torch.log(
            left.new_tensor(2.0)
        )).clamp(0.0, 1.0)

    def forward(
        self,
        decoder_feature,
        skip_feature,
        return_states=False,
        state_prior=None,
    ):
        spatial_output, spatial_states = self.spatial(
            decoder_feature,
            skip_feature,
            return_states=True,
        )
        spectral_output, spectral_states = self.spectral(
            decoder_feature,
            skip_feature,
            return_states=True,
        )
        disagreement = self._state_disagreement(
            spatial_states,
            spectral_states,
        )
        feature_delta = (
            spatial_output - spectral_output
        ).abs().mean(dim=1, keepdim=True)
        learned_alpha = torch.sigmoid(
            self.domain_router(
                torch.cat(
                    [
                        spatial_states,
                        spectral_states,
                        disagreement,
                        feature_delta,
                    ],
                    dim=1,
                )
            )
        )
        spatial_confidence = spatial_states.amax(dim=1, keepdim=True)
        spectral_confidence = spectral_states.amax(dim=1, keepdim=True)
        confidence_alpha = torch.sigmoid(
            (
                spatial_confidence - spectral_confidence
            )
            / self.confidence_temperature
        )
        if self.fusion_mode == "equal":
            alpha = torch.full_like(learned_alpha, 0.5)
        elif self.fusion_mode == "no_disagreement":
            alpha = 0.5 * (learned_alpha + confidence_alpha)
        else:
            disagreement_signal = 1.0 - torch.exp(
                -disagreement / self.disagreement_scale
            )
            specialist_alpha = learned_alpha
            if self.fusion_mode != "no_confidence":
                specialist_alpha = 0.5 * (
                    learned_alpha + confidence_alpha
                )
            alpha = (
                (1.0 - disagreement_signal) * 0.5
                + disagreement_signal * specialist_alpha
            )
        output = (
            alpha * spatial_output
            + (1.0 - alpha) * spectral_output
        )
        if return_states:
            states = (
                alpha * spatial_states
                + (1.0 - alpha) * spectral_states
            )
            return output, states
        return output


class PSCADecoderRefineFuse(nn.Module):
    """Refine both paths, arbitrate the skip once, and then fuse."""

    def __init__(
        self,
        channels,
        adapter="psca",
        arbitration_options=None,
    ):
        super().__init__()
        if adapter not in {"none", "psca", "dpsd", "cdac"}:
            raise ValueError("adapter must be none, psca, dpsd, or cdac")
        arbitration_options = arbitration_options or {}
        self.adapter = adapter
        self.up_refine = nn.Sequential(
            MultiKernelRefineBlock(channels),
            KANSelectiveKernel(channels),
        )
        self.skip_refine = nn.Sequential(
            MultiKernelRefineBlock(channels),
            KANSelectiveKernel(channels),
        )
        if adapter == "psca":
            self.arbitration = PhaseSemanticContourArbitration(
                channels,
                **arbitration_options,
            )
        elif adapter == "cdac":
            self.arbitration = CrossDomainAdaptiveConsensus(
                channels,
                **arbitration_options,
            )
        elif adapter == "dpsd":
            self.arbitration = (
                UltrasoundPhaseAnisotropicFrequencyResidual(
                    channels,
                    channels,
                    mode="anisotropic",
                    semantic_router=False,
                    max_gain=0.10,
                    preserve_phase=True,
                    zero_initialize=True,
                )
            )
        else:
            self.arbitration = nn.Identity()
        self.fuse = DoubleConv(channels * 2, channels)
        self.out_refine = nn.Sequential(
            MultiKernelRefineBlock(channels),
            KANSelectiveKernel(channels),
        )

    def forward(
        self,
        decoder_feature,
        skip_feature,
        return_states=False,
        state_prior=None,
    ):
        decoder_feature = self.up_refine(decoder_feature)
        skip_feature = self.skip_refine(skip_feature)
        states = None
        if self.adapter != "none":
            if return_states and self.adapter in {"psca", "cdac"}:
                skip_feature, states = self.arbitration(
                    decoder_feature,
                    skip_feature,
                    return_states=True,
                    state_prior=state_prior,
                )
            elif self.adapter in {"psca", "cdac"}:
                skip_feature = self.arbitration(
                    decoder_feature,
                    skip_feature,
                    state_prior=state_prior,
                )
            else:
                skip_feature = self.arbitration(
                    decoder_feature,
                    skip_feature,
                )
        fused = self.fuse(
            torch.cat([decoder_feature, skip_feature], dim=1)
        )
        output = self.out_refine(fused)
        if return_states:
            return output, states
        return output


class ChannelPSCADPSDNet(nn.Module):
    """Matched decoder backbone with phase-semantic contour arbitration."""

    def __init__(self, in_ch, out_ch, channels, variant="full"):
        super().__init__()
        variants = {
            "base": {
                "adapter": "none",
                "arbitration_options": {},
            },
            "psca_only": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "spatial",
                },
            },
            "dpsd_only": {
                "adapter": "dpsd",
                "arbitration_options": {},
            },
            "full": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "dpsd",
                },
            },
            "dual_evidence": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "dual",
                },
            },
            "no_prior": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "dpsd",
                    "use_prior": False,
                },
            },
            "binary_router": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "dpsd",
                    "router_mode": "binary",
                },
            },
            "no_suppression": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "dpsd",
                    "use_suppression": False,
                },
            },
            "no_conflict": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "dpsd",
                    "use_conflict": False,
                },
            },
            "policy_error": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "spatial",
                },
                "policy_mode": "error",
            },
            "policy_boundary": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "spatial",
                },
                "policy_mode": "boundary",
            },
            "policy_counterfactual": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "spatial",
                },
                "policy_mode": "counterfactual",
            },
            "policy_risk_projected": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "spatial",
                },
                "policy_mode": "risk_projected",
            },
            "policy_region_projected": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "spatial",
                },
                "policy_mode": "region_projected",
            },
            "policy_overlap_guard": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "spatial",
                },
                "policy_mode": "overlap_guard",
            },
            "policy_axial_equivariant": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "spatial",
                },
                "policy_mode": "axial_equivariant",
            },
            "hierarchical_consensus": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "spatial",
                },
                "propagate_psca_states": True,
            },
            "state_shrinkage": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "spatial",
                },
                "propagate_psca_states": True,
                "signed_hierarchical_calibration": True,
            },
            "disagreement_reconciliation": {
                "adapter": "psca",
                "arbitration_options": {
                    "evidence_mode": "spatial",
                },
                "propagate_psca_states": True,
                "disagreement_reconciliation": True,
            },
            "cross_domain_consensus": {
                "adapter": "cdac",
                "arbitration_options": {
                    "fusion_mode": "adaptive",
                },
            },
            "cross_domain_equal": {
                "adapter": "cdac",
                "arbitration_options": {
                    "fusion_mode": "equal",
                },
            },
            "cross_domain_no_disagreement": {
                "adapter": "cdac",
                "arbitration_options": {
                    "fusion_mode": "no_disagreement",
                },
            },
            "cross_domain_no_confidence": {
                "adapter": "cdac",
                "arbitration_options": {
                    "fusion_mode": "no_confidence",
                },
            },
        }
        if variant not in variants:
            raise ValueError("Unknown PSCA variant: {}".format(variant))
        variant_options = dict(variants[variant])
        self.policy_mode = variant_options.pop("policy_mode", None)
        self.propagate_psca_states = variant_options.pop(
            "propagate_psca_states",
            False,
        )
        self.signed_hierarchical_calibration = variant_options.pop(
            "signed_hierarchical_calibration",
            False,
        )
        self.disagreement_reconciliation = variant_options.pop(
            "disagreement_reconciliation",
            False,
        )
        refine_options = variant_options
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
        self.enhance5 = nn.Sequential(
            MultiKernelRefineBlock(c5),
            KANSelectiveKernel(c5),
        )

        def build_refine(channels, receives_state_prior):
            options = dict(refine_options)
            arbitration_options = dict(
                options.get("arbitration_options", {})
            )
            if (
                self.propagate_psca_states
                and receives_state_prior
            ):
                if self.signed_hierarchical_calibration:
                    arbitration_options[
                        "use_signed_hierarchical_calibration"
                    ] = True
                elif self.disagreement_reconciliation:
                    arbitration_options[
                        "use_disagreement_reconciliation"
                    ] = True
                else:
                    arbitration_options[
                        "use_hierarchical_consensus"
                    ] = True
            options["arbitration_options"] = arbitration_options
            return PSCADecoderRefineFuse(channels, **options)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = build_refine(c4, False)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = build_refine(c3, True)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = build_refine(c2, True)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = build_refine(c1, True)

        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward_decoder_features(self, inputs, return_states=False):
        c1 = self.conv1(inputs)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))

        if return_states or self.propagate_psca_states:
            c6, states6 = self.refine6(
                self.up6(c5),
                c4,
                return_states=True,
            )
            c7, states7 = self.refine7(
                self.up7(c6),
                c3,
                return_states=True,
                state_prior=(
                    states6
                    if self.propagate_psca_states
                    else None
                ),
            )
            c8, states8 = self.refine8(
                self.up8(c7),
                c2,
                return_states=True,
                state_prior=(
                    states7
                    if self.propagate_psca_states
                    else None
                ),
            )
            c9, states9 = self.refine9(
                self.up9(c8),
                c1,
                return_states=True,
                state_prior=(
                    states8
                    if self.propagate_psca_states
                    else None
                ),
            )
            if return_states:
                return c9, (
                    states6,
                    states7,
                    states8,
                    states9,
                )
            return c9

        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        c9 = self.refine9(self.up9(c8), c1)
        return c9

    def forward(self, inputs):
        if self.training and self.policy_mode is not None:
            c9, states = self.forward_decoder_features(
                inputs,
                return_states=True,
            )
            logits = self.conv10(c9)
            main = (
                logits
                if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1"
                else self.last_activation(logits)
            )
            return {
                "main": main,
                "psca_states": states,
            }

        c9 = self.forward_decoder_features(inputs)
        logits = self.conv10(c9)
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return logits
        return self.last_activation(logits)


class ChannelPSCASpectralCalibratorNet(ChannelPSCADPSDNet):
    """Frozen PSCA-only backbone with a conservative spectral logit adapter."""

    def __init__(self, in_ch, out_ch, channels):
        super().__init__(
            in_ch,
            out_ch,
            channels,
            variant="psca_only",
        )
        self.spectral_calibrator = UncertaintyGatedDPSDLogitCorrection(
            channels[0],
            max_logit_shift=1.5,
        )

    def forward(self, inputs):
        decoder_features = self.forward_decoder_features(inputs)
        base_logits = self.conv10(decoder_features)
        logits = self.spectral_calibrator(
            decoder_features,
            base_logits,
        )
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return logits
        return self.last_activation(logits)


def build_psca_model(in_ch, out_ch, channels, variant="full"):
    return ChannelPSCADPSDNet(
        in_ch,
        out_ch,
        channels,
        variant=variant,
    )


def build_psca_spectral_calibrator(in_ch, out_ch, channels):
    return ChannelPSCASpectralCalibratorNet(
        in_ch,
        out_ch,
        channels,
    )
