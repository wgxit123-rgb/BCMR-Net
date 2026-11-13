import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from UNet import UNet
from UNetPlusPlus import NestedUNet
from DeepLabv3plus import DeepLabV3Plus
from ResU_Net import ResUnet
from SE_UNet import SE_UNet
from SwinUNet import SwinUNet
from TransUNet import TransUNet
try:
    from official_transunet import OfficialTransUNetR50ViTB16
except ModuleNotFoundError as exc:
    if exc.name not in {"third_party", "third_party.TransUNet_official"}:
        raise
    OfficialTransUNetR50ViTB16 = None
from SE_UNet_MK_KAN_RPA import (
    DoubleConv,
    SELayer,
    MultiKernelRefineBlock,
    KANSelectiveKernel,
    ResidualParallelAttention,
    SE_UNet_MK_KAN_RPA,
)
from SE_UNet_Ablation import (
    SE_UNet_MK,
    SE_UNet_KAN,
    SE_UNet_RPA,
    SE_UNet_MK_KAN,
    SE_UNet_MK_RPA,
    SE_UNet_KAN_RPA,
)
from cmnk_bicer_tpb import TPB_MODEL_BASE_NAMES, create_tpb_model
from acet_net import build_acet_net
from eprc_net import build_camp_net
from qdk_net import build_qdk_net
from sct_net import build_sct_net
from rbfc_net import build_rbfc_net
from upaf_net import (
    UltrasoundPhaseAnisotropicFrequencyResidual,
    build_upaf_net,
)
from lfkan_net import build_lfkan_net
from bcar_feature_variants import (
    build_bcar_diffusion_model,
    build_bcar_dinov3_model,
    build_bcar_lite_model,
)
from external_model_wrappers import MFMSNetOfficialFS256
from psca_net import build_psca_model, build_psca_spectral_calibrator


class ChannelUNet(nn.Module):
    def __init__(self, in_ch, out_ch, channels):
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
        self.conv6 = DoubleConv(c4 * 2, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.conv7 = DoubleConv(c3 * 2, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.conv8 = DoubleConv(c2 * 2, c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.conv9 = DoubleConv(c1 * 2, c1)
        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.conv5(self.pool4(c4))
        c6 = self.conv6(torch.cat([self.up6(c5), c4], dim=1))
        c7 = self.conv7(torch.cat([self.up7(c6), c3], dim=1))
        c8 = self.conv8(torch.cat([self.up8(c7), c2], dim=1))
        c9 = self.conv9(torch.cat([self.up9(c8), c1], dim=1))
        logits = self.conv10(c9)
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return logits
        return self.last_activation(logits)


class AttentionGate(nn.Module):
    """Gate an encoder skip with decoder semantics before concatenation."""

    def __init__(self, skip_channels, gate_channels, inter_channels):
        super().__init__()
        self.skip_project = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.gate_project = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, skip_feature, gate_feature):
        score = self.skip_project(skip_feature) + self.gate_project(gate_feature)
        return skip_feature * self.attention(score)


class ChannelAttentionUNet(nn.Module):
    """Attention U-Net with an explicit five-stage channel schedule."""

    def __init__(self, in_ch, out_ch, channels):
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
        self.att6 = AttentionGate(c4, c4, max(c4 // 2, 8))
        self.conv6 = DoubleConv(c4 * 2, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.att7 = AttentionGate(c3, c3, max(c3 // 2, 8))
        self.conv7 = DoubleConv(c3 * 2, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.att8 = AttentionGate(c2, c2, max(c2 // 2, 8))
        self.conv8 = DoubleConv(c2 * 2, c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.att9 = AttentionGate(c1, c1, max(c1 // 2, 8))
        self.conv9 = DoubleConv(c1 * 2, c1)
        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.conv5(self.pool4(c4))
        u6 = self.up6(c5)
        c6 = self.conv6(torch.cat([u6, self.att6(c4, u6)], dim=1))
        u7 = self.up7(c6)
        c7 = self.conv7(torch.cat([u7, self.att7(c3, u7)], dim=1))
        u8 = self.up8(c7)
        c8 = self.conv8(torch.cat([u8, self.att8(c2, u8)], dim=1))
        u9 = self.up9(c8)
        c9 = self.conv9(torch.cat([u9, self.att9(c1, u9)], dim=1))
        logits = self.conv10(c9)
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return logits
        return self.last_activation(logits)


class OptionalEnhance(nn.Module):
    def __init__(self, channels, use_mk=False, use_kan=False, use_rpa=False):
        super().__init__()
        layers = []
        if use_mk:
            layers.append(MultiKernelRefineBlock(channels))
        if use_kan:
            layers.append(KANSelectiveKernel(channels))
        if use_rpa:
            layers.append(ResidualParallelAttention(channels))
        self.block = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x):
        return self.block(x)


class DecoderRefineFuse(nn.Module):
    """Refine upsampled and skip features before decoder fusion."""

    def __init__(self, up_channels, skip_channels, out_channels):
        super().__init__()
        self.up_recalibrate = nn.Sequential(
            SELayer(up_channels),
            MultiKernelRefineBlock(up_channels),
            KANSelectiveKernel(up_channels),
        )
        self.skip_recalibrate = nn.Sequential(
            SELayer(skip_channels),
            MultiKernelRefineBlock(skip_channels),
            KANSelectiveKernel(skip_channels),
        )
        self.fuse = DoubleConv(up_channels + skip_channels, out_channels)
        self.out_recalibrate = nn.Sequential(
            SELayer(out_channels),
            MultiKernelRefineBlock(out_channels),
            KANSelectiveKernel(out_channels),
        )

    def forward(self, up_feature, skip_feature):
        up_feature = self.up_recalibrate(up_feature)
        skip_feature = self.skip_recalibrate(skip_feature)
        fused = self.fuse(torch.cat([up_feature, skip_feature], dim=1))
        return self.out_recalibrate(fused)


class BoundarySpeckleAdaptiveSE(nn.Module):
    """SE-style attention with boundary and speckle-aware descriptors."""

    def __init__(self, channels, ratio=16, spatial_kernel=7, eps=1e-6):
        super().__init__()
        self.eps = eps
        hidden = max(channels // ratio, 4)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.repeat(channels, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(channels, 1, 1, 1))
        self.register_buffer("gray_sobel_x", sobel_x)
        self.register_buffer("gray_sobel_y", sobel_y)

        self.channel_reduce = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.channel_expand = nn.Conv2d(hidden, channels, 1)
        padding = spatial_kernel // 2
        self.spatial_project = nn.Conv2d(4, 1, spatial_kernel, padding=padding)

        nn.init.zeros_(self.channel_expand.weight)
        nn.init.zeros_(self.channel_expand.bias)
        nn.init.zeros_(self.spatial_project.weight)
        nn.init.zeros_(self.spatial_project.bias)

    def forward(self, x):
        mean = F.adaptive_avg_pool2d(x, 1)
        max_value = F.adaptive_max_pool2d(x, 1)
        std = torch.sqrt((x - mean).pow(2).mean(dim=(2, 3), keepdim=True) + self.eps)
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.size(1))
        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)
        edge_desc = F.adaptive_avg_pool2d(edge, 1)

        descriptor = torch.cat([mean, max_value, std, edge_desc], dim=1)
        channel_logits = self.channel_expand(self.channel_reduce(descriptor))
        channel_gate = 0.5 + torch.sigmoid(channel_logits)

        spatial_mean = x.mean(dim=1, keepdim=True)
        spatial_max = x.amax(dim=1, keepdim=True)
        spatial_std = torch.sqrt((x - spatial_mean).pow(2).mean(dim=1, keepdim=True) + self.eps)
        gray = spatial_mean
        gray_grad_x = F.conv2d(gray, self.gray_sobel_x, padding=1)
        gray_grad_y = F.conv2d(gray, self.gray_sobel_y, padding=1)
        spatial_edge = torch.sqrt(gray_grad_x * gray_grad_x + gray_grad_y * gray_grad_y + self.eps)
        spatial_edge = spatial_edge / (spatial_edge.amax(dim=(2, 3), keepdim=True) + self.eps)
        spatial_logits = self.spatial_project(torch.cat([spatial_mean, spatial_max, spatial_std, spatial_edge], dim=1))
        spatial_gate = 0.5 + torch.sigmoid(spatial_logits)

        return x * channel_gate * spatial_gate


class BoundaryConsistencyAdaptiveResidualSE(nn.Module):
    """Boundary-consistent, speckle-aware residual channel-spatial attention."""

    def __init__(self, channels, ratio=16, spatial_kernel=7, max_gain=0.25, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.max_gain = max_gain
        hidden = max(channels // ratio, 4)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        self.channel_reduce = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.channel_expand = nn.Conv2d(hidden, channels, 1)
        self.spatial_project = nn.Conv2d(
            4, 1, spatial_kernel, padding=spatial_kernel // 2
        )

        # The module starts as an identity mapping and learns only bounded corrections.
        nn.init.zeros_(self.channel_expand.weight)
        nn.init.zeros_(self.channel_expand.bias)
        nn.init.zeros_(self.spatial_project.weight)
        nn.init.zeros_(self.spatial_project.bias)

    def _edge_magnitude(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

    def _spatial_normalize(self, x):
        return x / (x.amax(dim=(2, 3), keepdim=True) + self.eps)

    def forward(self, x):
        # Attention statistics do not need a gradient path back through every
        # multi-scale filtering operation. The residual output still provides
        # a direct gradient to x, while attention parameters remain trainable.
        source = x.detach()
        mean = F.adaptive_avg_pool2d(source, 1)
        maximum = F.adaptive_max_pool2d(source, 1)
        minimum = -F.adaptive_max_pool2d(-source, 1)
        std = torch.sqrt(
            torch.var(source, dim=(2, 3), keepdim=True, unbiased=False) + self.eps
        )
        channel_desc = torch.cat([mean, maximum, minimum, std], dim=1)
        channel_logits = self.channel_expand(self.channel_reduce(channel_desc))

        gray = source.mean(dim=1, keepdim=True)
        smooth3 = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        smooth5 = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        edge3 = self._edge_magnitude(smooth3)
        edge5 = self._edge_magnitude(smooth5)
        norm_edge3 = self._spatial_normalize(edge3)
        norm_edge5 = self._spatial_normalize(edge5)
        agreement = 1.0 - (norm_edge3 - norm_edge5).abs() / (
            norm_edge3 + norm_edge5 + self.eps
        )
        consistent_edge = 0.5 * (edge3 + edge5) * agreement.clamp(0.0, 1.0)
        scale_noise = (gray - smooth3).abs() + (smooth3 - smooth5).abs()

        channel_range = source.amax(dim=1, keepdim=True) - source.amin(
            dim=1, keepdim=True
        )
        spatial_edge = self._spatial_normalize(consistent_edge.mean(dim=1, keepdim=True))
        spatial_noise = self._spatial_normalize(scale_noise.mean(dim=1, keepdim=True))
        spatial_logits = self.spatial_project(
            torch.cat([gray, channel_range, spatial_edge, spatial_noise], dim=1)
        )

        modulation = torch.tanh(channel_logits + spatial_logits)
        return x * (1.0 + self.max_gain * modulation)


class MCRCMechanismVariant(nn.Module):
    """Configurable MCRC used only for mechanism-level ablations."""

    def __init__(
        self,
        channels,
        ratio=16,
        spatial_kernel=7,
        max_gain=0.25,
        eps=1e-6,
        use_channel=True,
        use_spatial=True,
        use_agreement=True,
        use_scale_noise=True,
        channel_mode="moments",
        zero_initialize=True,
    ):
        super().__init__()
        if not use_channel and not use_spatial:
            raise ValueError("At least one MCRC branch must remain active")
        if channel_mode not in ("moments", "mean_only"):
            raise ValueError("Unknown MCRC channel mode: {}".format(channel_mode))

        self.eps = eps
        self.max_gain = max_gain
        self.use_channel = use_channel
        self.use_spatial = use_spatial
        self.use_agreement = use_agreement
        self.use_scale_noise = use_scale_noise
        self.channel_mode = channel_mode

        if use_channel:
            hidden = max(channels // ratio, 4)
            self.channel_reduce = nn.Sequential(
                nn.Conv2d(channels * 4, hidden, 1, bias=False),
                nn.ReLU(inplace=True),
            )
            self.channel_expand = nn.Conv2d(hidden, channels, 1)
            if zero_initialize:
                nn.init.zeros_(self.channel_expand.weight)
                nn.init.zeros_(self.channel_expand.bias)
        else:
            self.channel_reduce = None
            self.channel_expand = None

        if use_spatial:
            sobel_x = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
            ).view(1, 1, 3, 3)
            self.register_buffer("sobel_x", sobel_x)
            self.register_buffer("sobel_y", sobel_y)
            self.spatial_project = nn.Conv2d(
                4, 1, spatial_kernel, padding=spatial_kernel // 2
            )
            if zero_initialize:
                nn.init.zeros_(self.spatial_project.weight)
                nn.init.zeros_(self.spatial_project.bias)
        else:
            self.spatial_project = None

    def _edge_magnitude(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

    def _spatial_normalize(self, x):
        return x / (x.amax(dim=(2, 3), keepdim=True) + self.eps)

    def _channel_logits(self, source):
        mean = F.adaptive_avg_pool2d(source, 1)
        if self.channel_mode == "mean_only":
            zeros = torch.zeros_like(mean)
            descriptor = torch.cat([mean, zeros, zeros, zeros], dim=1)
        else:
            maximum = F.adaptive_max_pool2d(source, 1)
            minimum = -F.adaptive_max_pool2d(-source, 1)
            std = torch.sqrt(
                torch.var(
                    source,
                    dim=(2, 3),
                    keepdim=True,
                    unbiased=False,
                )
                + self.eps
            )
            descriptor = torch.cat([mean, maximum, minimum, std], dim=1)
        return self.channel_expand(self.channel_reduce(descriptor))

    def _spatial_logits(self, source):
        gray = source.mean(dim=1, keepdim=True)
        smooth3 = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        smooth5 = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        edge3 = self._edge_magnitude(smooth3)
        edge5 = self._edge_magnitude(smooth5)

        if self.use_agreement:
            norm_edge3 = self._spatial_normalize(edge3)
            norm_edge5 = self._spatial_normalize(edge5)
            agreement = 1.0 - (norm_edge3 - norm_edge5).abs() / (
                norm_edge3 + norm_edge5 + self.eps
            )
            edge = 0.5 * (edge3 + edge5) * agreement.clamp(0.0, 1.0)
        else:
            edge = 0.5 * (edge3 + edge5)

        scale_noise = (gray - smooth3).abs() + (smooth3 - smooth5).abs()
        if not self.use_scale_noise:
            scale_noise = torch.zeros_like(scale_noise)

        channel_range = source.amax(dim=1, keepdim=True) - source.amin(
            dim=1, keepdim=True
        )
        spatial_edge = self._spatial_normalize(edge)
        spatial_noise = self._spatial_normalize(scale_noise)
        descriptor = torch.cat(
            [gray, channel_range, spatial_edge, spatial_noise],
            dim=1,
        )
        return self.spatial_project(descriptor)

    def forward(self, x):
        source = x.detach()
        logits = []
        if self.use_channel:
            logits.append(self._channel_logits(source))
        if self.use_spatial:
            logits.append(self._spatial_logits(source))
        modulation = torch.tanh(sum(logits))
        return x * (1.0 + self.max_gain * modulation)


def build_mcrc_mechanism_variant(channels, variant):
    if variant == "identity":
        return nn.Identity()
    options = {
        "channel_only": dict(use_spatial=False),
        "spatial_only": dict(use_channel=False),
        "no_agreement": dict(use_agreement=False),
        "no_scale_noise": dict(use_scale_noise=False),
        "mean_only": dict(channel_mode="mean_only"),
        "nonzero_init": dict(zero_initialize=False),
    }
    if variant not in options:
        raise ValueError("Unknown MCRC mechanism variant: {}".format(variant))
    return MCRCMechanismVariant(channels, **options[variant])


class BidirectionalCoherenceEvidenceRecalibration(nn.Module):
    """Compete coherent-boundary recovery against speckle suppression."""

    def __init__(self, channels, reduction=8, max_gain=0.10, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.max_gain = max_gain
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
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.recovery_head = nn.Conv2d(hidden, channels, 1, bias=True)
        self.suppression_head = nn.Conv2d(hidden, channels, 1, bias=True)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # The full network is trained from scratch, but BiCER starts as identity.
        nn.init.zeros_(self.recovery_head.weight)
        nn.init.zeros_(self.recovery_head.bias)
        nn.init.zeros_(self.suppression_head.weight)
        nn.init.zeros_(self.suppression_head.bias)

    def _edge_magnitude(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

    def _normalize(self, x):
        maximum = x.amax(dim=(2, 3), keepdim=True)
        return x / (maximum + self.eps)

    def forward(self, decoder_feature, skip_feature):
        skip_context = self.skip_project(skip_feature)
        decoder_context = self.decoder_project(decoder_feature)

        global_context = F.adaptive_avg_pool2d(decoder_context, 1)
        semantic = torch.sigmoid(
            self.semantic_local(decoder_context)
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

        recovery_evidence = (
            uncertainty * semantic * coherence * (1.0 - scale_noise)
        )
        suppression_evidence = (
            uncertainty * (1.0 - semantic) * scale_noise * (1.0 - coherence)
        )

        skip_mean = F.adaptive_avg_pool2d(skip_feature, 1)
        skip_max = F.adaptive_max_pool2d(skip_feature, 1)
        skip_min = -F.adaptive_max_pool2d(-skip_feature, 1)
        skip_std = torch.sqrt(
            torch.var(skip_feature, dim=(2, 3), keepdim=True, unbiased=False)
            + self.eps
        )
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
            skip_feature.size(0), 2, skip_feature.size(1), 1, 1
        )

        spatial_logits = self.spatial_router(
            torch.cat(
                [
                    recovery_evidence,
                    suppression_evidence,
                    semantic,
                    uncertainty,
                    coherence,
                    scale_noise,
                ],
                dim=1,
            )
        ).unsqueeze(2)
        competition = torch.softmax(channel_logits + spatial_logits, dim=1)

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
        correction = uncertainty * (
            competition[:, 0] * recovery - competition[:, 1] * suppression
        )
        return skip_feature + self.max_gain * correction


class BiCERDecoderRefineFuse(nn.Module):
    """CMNK decoder fusion with one joint bidirectional skip recalibration."""

    def __init__(self, channels):
        super().__init__()
        self.up_refine = nn.Sequential(
            MultiKernelRefineBlock(channels),
            KANSelectiveKernel(channels),
        )
        self.skip_refine = nn.Sequential(
            MultiKernelRefineBlock(channels),
            KANSelectiveKernel(channels),
        )
        self.bicer = BidirectionalCoherenceEvidenceRecalibration(channels)
        self.fuse = DoubleConv(channels * 2, channels)
        self.out_refine = nn.Sequential(
            MultiKernelRefineBlock(channels),
            KANSelectiveKernel(channels),
        )

    def forward(self, up_feature, skip_feature):
        up_feature = self.up_refine(up_feature)
        skip_feature = self.skip_refine(skip_feature)
        skip_feature = self.bicer(up_feature, skip_feature)
        fused = self.fuse(torch.cat([up_feature, skip_feature], dim=1))
        return self.out_refine(fused)


class LearnableLogitCalibration(nn.Module):
    """Identity-initialized global temperature and decision-bias control."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, logits):
        temperature = self.log_temperature.exp().clamp(0.5, 2.0)
        return (logits - self.bias) / temperature


class ScaleConditionedPrototypeConnectivityRecalibration(nn.Module):
    """Calibrate logits using lesion scale, prototypes and connected support."""

    def __init__(
        self,
        channels,
        hidden=16,
        max_gain=0.50,
        connectivity_steps=4,
        eps=1e-6,
    ):
        super().__init__()
        self.max_gain = max_gain
        self.connectivity_steps = connectivity_steps
        self.eps = eps
        hidden = max(hidden, 8)
        groups = 4 if hidden % 4 == 0 else 1

        self.feature_projection = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        self.scale_router = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(hidden + 7, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )

        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))
        self.recovery_gain = nn.Parameter(torch.zeros(()))
        self.suppression_gain = nn.Parameter(torch.zeros(()))

        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def _weighted_prototype(self, embedding, weight):
        numerator = (embedding * weight).sum(dim=(2, 3), keepdim=True)
        denominator = weight.sum(dim=(2, 3), keepdim=True) + self.eps
        return F.normalize(numerator / denominator, dim=1, eps=self.eps)

    def _connected_support(self, core, uncertainty):
        traversable = (core + uncertainty).clamp(0.0, 1.0)
        support = core
        for _ in range(self.connectivity_steps):
            expanded = F.max_pool2d(support, 3, stride=1, padding=1)
            support = torch.maximum(core, expanded * traversable)
        return support.clamp(0.0, 1.0)

    def forward(self, feature, logits):
        probability = torch.sigmoid(logits)
        uncertainty = 4.0 * probability * (1.0 - probability)
        positive_anchor = torch.sigmoid((probability - 0.70) / 0.08)
        negative_anchor = torch.sigmoid((0.30 - probability) / 0.08)
        support = self._connected_support(positive_anchor, uncertainty)

        embedding = F.normalize(
            self.feature_projection(feature), dim=1, eps=self.eps
        )
        positive_prototype = self._weighted_prototype(
            embedding, positive_anchor
        )
        negative_prototype = self._weighted_prototype(
            embedding, negative_anchor
        )
        similarity = (
            (embedding * positive_prototype).sum(dim=1, keepdim=True)
            - (embedding * negative_prototype).sum(dim=1, keepdim=True)
        )
        lesion_evidence = torch.sigmoid(4.0 * similarity)

        edge = (
            probability
            - F.avg_pool2d(probability, 3, stride=1, padding=1)
        ).abs()
        recovery = uncertainty * support * lesion_evidence
        suppression = (
            probability
            * (1.0 - support)
            * (1.0 - lesion_evidence)
        )

        scale_descriptor = torch.cat(
            [
                positive_anchor.mean(dim=(2, 3)),
                probability.mean(dim=(2, 3)),
                uncertainty.mean(dim=(2, 3)),
                edge.mean(dim=(2, 3)),
            ],
            dim=1,
        )
        route = torch.softmax(self.scale_router(scale_descriptor), dim=1)
        route = route.unsqueeze(-1).unsqueeze(-1)

        residual_features = torch.cat(
            [
                embedding,
                probability,
                uncertainty,
                support,
                lesion_evidence,
                recovery,
                suppression,
                edge,
            ],
            dim=1,
        )
        learned_residual = torch.tanh(self.residual_head(residual_features))
        signed_prior = (
            torch.tanh(self.recovery_gain) * route[:, 0:1] * recovery
            - torch.tanh(self.suppression_gain) * route[:, 1:2] * suppression
        )
        spatial_gate = (recovery + suppression).clamp(0.0, 1.0)
        correction = self.max_gain * (
            signed_prior + spatial_gate * learned_residual
        )

        temperature = self.log_temperature.exp().clamp(0.5, 2.0)
        calibrated_logits = (logits - self.bias) / temperature
        return calibrated_logits + correction


class BCARDecoderRefineFuse(nn.Module):
    """Decoder refinement using boundary-consistency adaptive residual SE."""

    def __init__(self, up_channels, skip_channels, out_channels):
        super().__init__()
        self.up_recalibrate = nn.Sequential(
            BoundaryConsistencyAdaptiveResidualSE(up_channels),
            MultiKernelRefineBlock(up_channels),
            KANSelectiveKernel(up_channels),
        )
        self.skip_recalibrate = nn.Sequential(
            BoundaryConsistencyAdaptiveResidualSE(skip_channels),
            MultiKernelRefineBlock(skip_channels),
            KANSelectiveKernel(skip_channels),
        )
        self.fuse = DoubleConv(up_channels + skip_channels, out_channels)
        self.out_recalibrate = nn.Sequential(
            BoundaryConsistencyAdaptiveResidualSE(out_channels),
            MultiKernelRefineBlock(out_channels),
            KANSelectiveKernel(out_channels),
        )

    def forward(self, up_feature, skip_feature):
        up_feature = self.up_recalibrate(up_feature)
        skip_feature = self.skip_recalibrate(skip_feature)
        fused = self.fuse(torch.cat([up_feature, skip_feature], dim=1))
        return self.out_recalibrate(fused)


class BCARFrequencyDecoderRefineFuse(BCARDecoderRefineFuse):
    """Inject phase-preserving directional frequency evidence before BCAR."""

    def __init__(
        self,
        up_channels,
        skip_channels,
        out_channels,
        mode="anisotropic",
        preserve_phase=True,
        zero_initialize=True,
    ):
        super().__init__(up_channels, skip_channels, out_channels)
        self.frequency = UltrasoundPhaseAnisotropicFrequencyResidual(
            up_channels,
            skip_channels,
            mode=mode,
            semantic_router=False,
            preserve_phase=preserve_phase,
            zero_initialize=zero_initialize,
        )

    def forward(self, up_feature, skip_feature):
        skip_feature = self.frequency(up_feature, skip_feature)
        return super().forward(up_feature, skip_feature)


class ZeroInitDecoderGuidanceResidual(nn.Module):
    """Minimal decoder-guided skip correction with an exact identity start."""

    def __init__(self, up_channels, skip_channels, reduction=8, max_gain=0.10):
        super().__init__()
        hidden = max(min(up_channels, skip_channels) // reduction, 8)
        self.max_gain = max_gain
        self.up_project = nn.Conv2d(up_channels, hidden, 1, bias=True)
        self.skip_project = nn.Conv2d(skip_channels, hidden, 1, bias=True)
        self.interaction = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True),
            nn.ReLU(inplace=True),
        )
        self.delta = nn.Conv2d(hidden, skip_channels, 1, bias=True)

        # The transferred BCAR model is reproduced exactly before fine-tuning.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, up_feature, skip_feature):
        up_context = self.up_project(up_feature)
        skip_context = self.skip_project(skip_feature)
        interaction = torch.cat(
            [up_context, skip_context, (up_context - skip_context).abs()], dim=1
        )
        correction = torch.tanh(self.delta(self.interaction(interaction)))
        return skip_feature + self.max_gain * correction


class BCARDGResidualDecoderRefineFuse(nn.Module):
    """Original BCAR fusion plus one zero-initialized skip residual."""

    def __init__(self, up_channels, skip_channels, out_channels):
        super().__init__()
        # Keep these names and definitions identical to BCARDecoderRefineFuse
        # so every original BCAR tensor transfers without remapping.
        self.up_recalibrate = nn.Sequential(
            BoundaryConsistencyAdaptiveResidualSE(up_channels),
            MultiKernelRefineBlock(up_channels),
            KANSelectiveKernel(up_channels),
        )
        self.skip_recalibrate = nn.Sequential(
            BoundaryConsistencyAdaptiveResidualSE(skip_channels),
            MultiKernelRefineBlock(skip_channels),
            KANSelectiveKernel(skip_channels),
        )
        self.fuse = DoubleConv(up_channels + skip_channels, out_channels)
        self.out_recalibrate = nn.Sequential(
            BoundaryConsistencyAdaptiveResidualSE(out_channels),
            MultiKernelRefineBlock(out_channels),
            KANSelectiveKernel(out_channels),
        )
        self.skip_guidance = ZeroInitDecoderGuidanceResidual(
            up_channels, skip_channels
        )

    def forward(self, up_feature, skip_feature):
        up_feature = self.up_recalibrate(up_feature)
        skip_feature = self.skip_recalibrate(skip_feature)
        skip_feature = self.skip_guidance(up_feature, skip_feature)
        fused = self.fuse(torch.cat([up_feature, skip_feature], dim=1))
        return self.out_recalibrate(fused)


class SpeckleCoherenceFrequencyAdapter(nn.Module):
    """Decoder-conditioned residual that favors scale-coherent ultrasound edges."""

    def __init__(self, up_channels, skip_channels, reduction=8, max_gain=0.10, eps=1e-6):
        super().__init__()
        hidden = max(min(up_channels, skip_channels) // reduction, 8)
        self.max_gain = max_gain
        self.eps = eps
        self.up_project = nn.Conv2d(up_channels, hidden, 1, bias=True)
        self.skip_project = nn.Conv2d(skip_channels, hidden, 1, bias=True)
        self.semantic_gate = nn.Conv2d(hidden, 1, 1, bias=True)
        self.interaction = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True),
            nn.ReLU(inplace=True),
        )
        self.delta = nn.Conv2d(hidden, skip_channels, 1, bias=True)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def _normalize(self, value):
        scale = value.amax(dim=(2, 3), keepdim=True)
        return value / (scale + self.eps)

    def forward(self, up_feature, skip_feature):
        up_context = self.up_project(up_feature)
        skip_context = self.skip_project(skip_feature)
        low3 = F.avg_pool2d(skip_context, 3, stride=1, padding=1)
        low5 = F.avg_pool2d(skip_context, 5, stride=1, padding=2)
        high3 = skip_context - low3
        high5 = skip_context - low5

        edge3 = self._normalize((low3 - F.avg_pool2d(low3, 3, 1, 1)).abs())
        edge5 = self._normalize((low5 - F.avg_pool2d(low5, 3, 1, 1)).abs())
        coherence = torch.exp(-(edge3 - edge5).abs()).mean(dim=1, keepdim=True)
        speckle = self._normalize((high3 - high5).abs()).mean(dim=1, keepdim=True)
        semantic = torch.sigmoid(self.semantic_gate(up_context))
        gate = coherence * (1.0 - speckle).clamp(0.0, 1.0) * semantic

        features = torch.cat([up_context, skip_context, low3, high3 - high5], dim=1)
        correction = torch.tanh(self.delta(self.interaction(features)))
        return skip_feature + self.max_gain * gate * correction


class BCARSCFADecoderRefineFuse(nn.Module):
    """Original BCAR fusion with one identity-initialized SCFA skip adapter."""

    def __init__(self, up_channels, skip_channels, out_channels):
        super().__init__()
        self.up_recalibrate = nn.Sequential(
            BoundaryConsistencyAdaptiveResidualSE(up_channels),
            MultiKernelRefineBlock(up_channels),
            KANSelectiveKernel(up_channels),
        )
        self.skip_recalibrate = nn.Sequential(
            BoundaryConsistencyAdaptiveResidualSE(skip_channels),
            MultiKernelRefineBlock(skip_channels),
            KANSelectiveKernel(skip_channels),
        )
        self.fuse = DoubleConv(up_channels + skip_channels, out_channels)
        self.out_recalibrate = nn.Sequential(
            BoundaryConsistencyAdaptiveResidualSE(out_channels),
            MultiKernelRefineBlock(out_channels),
            KANSelectiveKernel(out_channels),
        )
        self.skip_scfa = SpeckleCoherenceFrequencyAdapter(
            up_channels, skip_channels
        )

    def forward(self, up_feature, skip_feature):
        up_feature = self.up_recalibrate(up_feature)
        skip_feature = self.skip_recalibrate(skip_feature)
        skip_feature = self.skip_scfa(up_feature, skip_feature)
        fused = self.fuse(torch.cat([up_feature, skip_feature], dim=1))
        return self.out_recalibrate(fused)


class WaveletSpeckleCoherenceAdaptiveResidual(nn.Module):
    """Haar-band coherence, speckle competition and spline-routed skip residual."""

    def __init__(self, up_channels, skip_channels, reduction=8, max_gain=0.10, eps=1e-6):
        super().__init__()
        hidden = max(min(up_channels, skip_channels) // reduction, 8)
        if hidden % 8 != 0:
            raise ValueError("WSCAR hidden channels must be divisible by 8")
        self.hidden = hidden
        self.max_gain = max_gain
        self.eps = eps
        haar = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ).unsqueeze(1) * 0.5
        self.register_buffer("haar_filters", haar)
        self.up_project = nn.Conv2d(up_channels, hidden, 1, bias=True)
        self.skip_project = nn.Conv2d(skip_channels, hidden, 1, bias=True)
        self.semantic_gate = nn.Conv2d(hidden, 1, 1, bias=True)
        self.band_router = GroupedSplineScaleRouter(
            groups=8, descriptor_dim=4, scales=3, basis_count=5, degree=3
        )
        self.interaction = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True),
            nn.ReLU(inplace=True),
        )
        self.delta = nn.Conv2d(hidden, skip_channels, 1, bias=True)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def _filters(self, channels, dtype):
        return self.haar_filters.to(dtype=dtype).repeat(channels, 1, 1, 1)

    def _dwt(self, x):
        batch, channels, _, _ = x.shape
        bands = F.conv2d(
            x, self._filters(channels, x.dtype), stride=2, groups=channels
        )
        bands = bands.view(batch, channels, 4, bands.shape[-2], bands.shape[-1])
        return tuple(bands[:, :, index] for index in range(4))

    def _idwt(self, ll, horizontal, vertical, diagonal):
        batch, channels, height, width = ll.shape
        bands = torch.stack([ll, horizontal, vertical, diagonal], dim=2)
        bands = bands.reshape(batch, channels * 4, height, width)
        return F.conv_transpose2d(
            bands, self._filters(channels, bands.dtype), stride=2, groups=channels
        )

    def _normalize(self, value):
        return value / (value.amax(dim=(2, 3), keepdim=True) + self.eps)

    def forward(self, up_feature, skip_feature):
        up_context = self.up_project(up_feature)
        skip_context = self.skip_project(skip_feature)
        ll1, horizontal1, vertical1, diagonal1 = self._dwt(skip_context)
        _, horizontal2, vertical2, _ = self._dwt(ll1)

        edge1 = torch.sqrt(horizontal1.square() + vertical1.square() + self.eps)
        edge2 = torch.sqrt(horizontal2.square() + vertical2.square() + self.eps)
        edge2 = F.interpolate(edge2, size=edge1.shape[-2:], mode="bilinear", align_corners=False)
        coherence = torch.exp(-(self._normalize(edge1) - self._normalize(edge2)).abs())

        horizontal_energy = horizontal1.abs()
        vertical_energy = vertical1.abs()
        directionality = (horizontal_energy - vertical_energy).abs() / (
            horizontal_energy + vertical_energy + diagonal1.abs() + self.eps
        )
        isotropic_noise = (
            1.0 - directionality.clamp(0.0, 1.0)
        ) * self._normalize(diagonal1.abs())
        semantic = torch.sigmoid(self.semantic_gate(up_context))
        semantic = F.avg_pool2d(semantic, 2, stride=2)

        batch = skip_context.shape[0]
        grouped_edge = edge1.view(batch, 8, self.hidden // 8, *edge1.shape[-2:])
        grouped_noise = isotropic_noise.view(
            batch, 8, self.hidden // 8, *isotropic_noise.shape[-2:]
        )
        descriptor = torch.stack(
            [
                grouped_edge.mean(dim=(2, 3, 4)),
                grouped_edge.std(dim=(2, 3, 4), unbiased=False),
                grouped_noise.mean(dim=(2, 3, 4)),
                semantic.mean(dim=(2, 3)).expand(-1, 8),
            ],
            dim=-1,
        )
        weights = self.band_router(descriptor)
        weights = weights.repeat_interleave(self.hidden // 8, dim=1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        preserve = coherence * semantic * (1.0 - isotropic_noise).clamp(0.0, 1.0)
        suppress = (1.0 - semantic) * isotropic_noise
        competition = preserve - 0.5 * suppress
        routed_horizontal = weights[:, :, 0] * horizontal1 * competition
        routed_vertical = weights[:, :, 1] * vertical1 * competition
        routed_diagonal = weights[:, :, 2] * diagonal1 * competition
        wavelet_residual = self._idwt(
            torch.zeros_like(ll1), routed_horizontal, routed_vertical, routed_diagonal
        )

        features = torch.cat([up_context, skip_context, wavelet_residual], dim=1)
        correction = torch.tanh(self.delta(self.interaction(features)))
        return skip_feature + self.max_gain * correction


class BCARWSCARDecoderRefineFuse(nn.Module):
    """Original BCAR decoder fusion with an identity-initialized WSCAR path."""

    def __init__(self, up_channels, skip_channels, out_channels):
        super().__init__()
        self.up_recalibrate = nn.Sequential(
            BoundaryConsistencyAdaptiveResidualSE(up_channels),
            MultiKernelRefineBlock(up_channels),
            KANSelectiveKernel(up_channels),
        )
        self.skip_recalibrate = nn.Sequential(
            BoundaryConsistencyAdaptiveResidualSE(skip_channels),
            MultiKernelRefineBlock(skip_channels),
            KANSelectiveKernel(skip_channels),
        )
        self.fuse = DoubleConv(up_channels + skip_channels, out_channels)
        self.out_recalibrate = nn.Sequential(
            BoundaryConsistencyAdaptiveResidualSE(out_channels),
            MultiKernelRefineBlock(out_channels),
            KANSelectiveKernel(out_channels),
        )
        self.skip_wscar = WaveletSpeckleCoherenceAdaptiveResidual(
            up_channels, skip_channels
        )

    def forward(self, up_feature, skip_feature):
        up_feature = self.up_recalibrate(up_feature)
        skip_feature = self.skip_recalibrate(skip_feature)
        skip_feature = self.skip_wscar(up_feature, skip_feature)
        fused = self.fuse(torch.cat([up_feature, skip_feature], dim=1))
        return self.out_recalibrate(fused)


class MorphologicalInnerOuterContourResidual(nn.Module):
    """Uncertainty-gated asymmetric logit correction for inner/outer contours."""

    def __init__(self, channels, hidden_ratio=4, max_gain=0.10):
        super().__init__()
        hidden = max(channels // hidden_ratio, 8)
        self.max_gain = max_gain
        self.coarse = nn.Conv2d(channels, 1, 1, bias=True)
        self.feature_project = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Conv2d(hidden + 3, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True),
            nn.ReLU(inplace=True),
        )
        self.inner_delta = nn.Conv2d(hidden, 1, 1, bias=True)
        self.outer_delta = nn.Conv2d(hidden, 1, 1, bias=True)
        nn.init.zeros_(self.inner_delta.weight)
        nn.init.zeros_(self.inner_delta.bias)
        nn.init.zeros_(self.outer_delta.weight)
        nn.init.zeros_(self.outer_delta.bias)

    def forward(self, feature):
        probability = torch.sigmoid(self.coarse(feature))
        dilated = F.max_pool2d(probability, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-probability, 3, stride=1, padding=1)
        outer_band = (dilated - probability).clamp_min(0.0)
        inner_band = (probability - eroded).clamp_min(0.0)
        uncertainty = 4.0 * probability * (1.0 - probability)
        context = self.context(
            torch.cat(
                [self.feature_project(feature), inner_band, outer_band, uncertainty],
                dim=1,
            )
        )
        inner = inner_band * torch.tanh(self.inner_delta(context))
        outer = outer_band * torch.tanh(self.outer_delta(context))
        return self.max_gain * uncertainty * (inner - outer)


class DecoderGuidedBCAR(nn.Module):
    """Decoder-guided preservation and suppression for ultrasound skip features."""

    def __init__(self, channels, ratio=8, spatial_kernel=7, max_gain=0.25, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.max_gain = max_gain
        hidden = max(channels // ratio, 8)

        self.up_embed = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.skip_embed = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.preserve_channel = nn.Conv2d(hidden, channels, 1)
        self.suppress_channel = nn.Conv2d(hidden, channels, 1)
        padding = spatial_kernel // 2
        self.preserve_spatial = nn.Conv2d(6, 1, spatial_kernel, padding=padding)
        self.suppress_spatial = nn.Conv2d(6, 1, spatial_kernel, padding=padding)

        self.preserve_delta = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.noise_delta = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.preserve_gain = nn.Parameter(torch.tensor(0.10))
        self.suppress_gain = nn.Parameter(torch.tensor(0.10))

        for layer in (
            self.preserve_channel, self.suppress_channel,
            self.preserve_spatial, self.suppress_spatial,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _normalize_map(self, value):
        minimum = value.amin(dim=(2, 3), keepdim=True)
        maximum = value.amax(dim=(2, 3), keepdim=True)
        return (value - minimum) / (maximum - minimum + self.eps)

    def _edge_magnitude(self, value):
        grad_x = F.pad(value[:, :, :, 2:] - value[:, :, :, :-2], (1, 1, 0, 0)) * 0.5
        grad_y = F.pad(value[:, :, 2:, :] - value[:, :, :-2, :], (0, 0, 1, 1)) * 0.5
        return torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

    def forward(self, up_feature, skip_feature):
        up_embed = self.up_embed(up_feature)
        skip_embed = self.skip_embed(skip_feature)
        semantic_agreement = torch.exp(
            -(up_embed - skip_embed).abs().mean(dim=1, keepdim=True)
        )

        gray = skip_feature.mean(dim=1, keepdim=True)
        smooth3 = F.avg_pool2d(gray, 3, stride=1, padding=1)
        smooth5 = F.avg_pool2d(gray, 5, stride=1, padding=2)
        speckle = self._normalize_map((gray - smooth3).abs() + (smooth3 - smooth5).abs())
        edge3 = self._normalize_map(self._edge_magnitude(smooth3))
        edge5 = self._normalize_map(self._edge_magnitude(smooth5))
        edge_agreement = 1.0 - (edge3 - edge5).abs() / (edge3 + edge5 + self.eps)
        consistent_edge = (0.5 * (edge3 + edge5) * edge_agreement.clamp(0.0, 1.0))
        local_contrast = self._normalize_map((gray - smooth5).abs())

        spatial_context = torch.cat(
            [
                up_embed.mean(dim=1, keepdim=True),
                skip_embed.mean(dim=1, keepdim=True),
                semantic_agreement,
                local_contrast,
                consistent_edge,
                speckle,
            ],
            dim=1,
        )
        up_mean = F.adaptive_avg_pool2d(up_feature, 1)
        skip_mean = F.adaptive_avg_pool2d(skip_feature, 1)
        skip_std = torch.sqrt(
            torch.var(skip_feature, dim=(2, 3), keepdim=True, unbiased=False) + self.eps
        )
        cross_difference = F.adaptive_avg_pool2d((up_feature - skip_feature).abs(), 1)
        channel_context = self.channel_reduce(
            torch.cat([up_mean, skip_mean, skip_std, cross_difference], dim=1)
        )

        preserve = torch.sigmoid(
            self.preserve_channel(channel_context) + self.preserve_spatial(spatial_context)
        )
        suppress = torch.sigmoid(
            self.suppress_channel(channel_context) + self.suppress_spatial(spatial_context)
        )
        suppress = suppress * (1.0 - preserve)

        high_frequency = skip_feature - F.avg_pool2d(skip_feature, 3, stride=1, padding=1)
        preserve_delta = self.preserve_delta(skip_feature)
        noise_delta = self.noise_delta(high_frequency)
        preserve_gain = self.max_gain * torch.tanh(self.preserve_gain)
        suppress_gain = self.max_gain * torch.tanh(self.suppress_gain)
        return (
            skip_feature
            + preserve_gain * preserve * preserve_delta
            - suppress_gain * suppress * noise_delta
        )


class DGBCARDecoderRefineFuse(nn.Module):
    """DG-BCAR ablation while preserving the proven MK-KAN decoder refinement."""

    def __init__(self, channels):
        super().__init__()
        self.skip_attention = DecoderGuidedBCAR(channels)
        self.up_refine = nn.Sequential(
            MultiKernelRefineBlock(channels),
            KANSelectiveKernel(channels),
        )
        self.skip_refine = nn.Sequential(
            MultiKernelRefineBlock(channels),
            KANSelectiveKernel(channels),
        )
        self.fuse = DoubleConv(channels * 2, channels)
        self.out_refine = nn.Sequential(
            MultiKernelRefineBlock(channels),
            KANSelectiveKernel(channels),
        )

    def forward(self, up_feature, skip_feature):
        guided_skip = self.skip_attention(up_feature, skip_feature)
        fused = self.fuse(
            torch.cat([self.up_refine(up_feature), self.skip_refine(guided_skip)], dim=1)
        )
        return self.out_refine(fused)


class USNCRRefinement(nn.Module):
    """Uncertainty-guided signed-distance normal contrast refinement."""

    def __init__(self, channels, sample_channels=8, radius=2.0, tau=0.20, max_gain=0.25):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.radius = radius
        self.tau = tau
        self.max_gain = max_gain
        self.feature_project = nn.Sequential(
            nn.Conv2d(channels, sample_channels, 1, bias=False),
            nn.BatchNorm2d(sample_channels),
            nn.ReLU(inplace=True),
        )
        self.sdf_head = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Tanh(),
        )
        self.coarse_head = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())
        profile_channels = sample_channels * 3 + 1
        self.correction = nn.Sequential(
            nn.Conv2d(profile_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    @staticmethod
    def _base_grid(batch, height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    @staticmethod
    def _normal_field(sdf):
        grad_x = F.pad(sdf[:, :, :, 2:] - sdf[:, :, :, :-2], (1, 1, 0, 0)) * 0.5
        grad_y = F.pad(sdf[:, :, 2:, :] - sdf[:, :, :-2, :], (0, 0, 1, 1)) * 0.5
        magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return grad_x / magnitude, grad_y / magnitude

    def forward(self, feature):
        sdf = self.sdf_head(feature)
        coarse = self.coarse_head(feature)
        projected = self.feature_project(feature)

        geometry = sdf.detach()
        normal_x, normal_y = self._normal_field(geometry)
        batch, _, height, width = projected.shape
        base_grid = self._base_grid(batch, height, width, projected.device, projected.dtype)
        offset_x = normal_x[:, 0] * (2.0 * self.radius / max(width - 1, 1))
        offset_y = normal_y[:, 0] * (2.0 * self.radius / max(height - 1, 1))
        offset = torch.stack([offset_x, offset_y], dim=-1)
        plus = F.grid_sample(
            projected, base_grid + offset, mode='bilinear', padding_mode='border', align_corners=True
        )
        minus = F.grid_sample(
            projected, base_grid - offset, mode='bilinear', padding_mode='border', align_corners=True
        )
        profile = torch.cat(
            [projected, (plus - minus).abs(), (projected - 0.5 * (plus + minus)).abs()],
            dim=1,
        )
        uncertainty = 4.0 * coarse * (1.0 - coarse) * torch.exp(-geometry.abs() / self.tau)
        correction = torch.tanh(self.correction(torch.cat([profile, uncertainty], dim=1)))
        return feature + self.max_gain * uncertainty * correction, sdf, coarse


class GroupedSplineScaleRouter(nn.Module):
    """Grouped cubic B-spline KAN used only for morphology-conditioned scale routing."""

    def __init__(self, groups=8, descriptor_dim=4, scales=3, basis_count=5, degree=3):
        super().__init__()
        if basis_count <= degree:
            raise ValueError("basis_count must be greater than spline degree")
        self.groups = groups
        self.descriptor_dim = descriptor_dim
        self.scales = scales
        self.basis_count = basis_count
        self.degree = degree

        interior_count = basis_count - degree - 1
        interior = torch.linspace(-1.0, 1.0, interior_count + 2)[1:-1]
        knots = torch.cat(
            [torch.full((degree + 1,), -1.0), interior, torch.full((degree + 1,), 1.0)]
        )
        self.register_buffer("knots", knots)
        self.descriptor_norm = nn.LayerNorm(descriptor_dim)
        self.base_weight = nn.Parameter(torch.zeros(groups, scales, descriptor_dim))
        self.spline_weight = nn.Parameter(
            torch.zeros(groups, scales, descriptor_dim, basis_count)
        )
        self.scale_bias = nn.Parameter(torch.zeros(groups, scales))

    def _basis(self, x):
        x = x.clamp(-1.0, 1.0 - 1e-6)
        knots = self.knots.to(device=x.device, dtype=x.dtype)
        basis = ((x.unsqueeze(-1) >= knots[:-1]) & (x.unsqueeze(-1) < knots[1:])).to(x.dtype)
        knot_count = knots.numel()
        for degree in range(1, self.degree + 1):
            count = knot_count - degree - 1
            left_den = knots[degree:degree + count] - knots[:count]
            right_den = knots[degree + 1:degree + 1 + count] - knots[1:1 + count]
            left = torch.where(
                left_den.abs() > 1e-7,
                (x.unsqueeze(-1) - knots[:count]) / left_den.clamp_min(1e-7),
                torch.zeros_like(x.unsqueeze(-1)),
            )
            right = torch.where(
                right_den.abs() > 1e-7,
                (knots[degree + 1:degree + 1 + count] - x.unsqueeze(-1))
                / right_den.clamp_min(1e-7),
                torch.zeros_like(x.unsqueeze(-1)),
            )
            basis = left * basis[..., :count] + right * basis[..., 1:count + 1]
        return basis

    def forward(self, descriptor):
        normalized = torch.tanh(self.descriptor_norm(descriptor))
        basis = self._basis(normalized)
        base_logits = torch.einsum(
            "bgj,gsj->bgs", F.silu(normalized), self.base_weight
        )
        spline_logits = torch.einsum(
            "bgjm,gsjm->bgs", basis, self.spline_weight
        )
        logits = base_logits + spline_logits + self.scale_bias.unsqueeze(0)
        return torch.softmax(logits, dim=-1)


class MorphologyAwareSplineAdaptiveKernel(nn.Module):
    """MSAK: morphology statistics drive factorized channel-spatial scale routing."""

    def __init__(self, channels, groups=8, basis_count=5, degree=3, eps=1e-6):
        super().__init__()
        if channels % groups != 0:
            raise ValueError(f"channels={channels} must be divisible by groups={groups}")
        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups
        self.eps = eps
        self.input_project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        channels, channels, 3, padding=3, dilation=3,
                        groups=channels, bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.channel_router = GroupedSplineScaleRouter(
            groups=groups, descriptor_dim=4, scales=3,
            basis_count=basis_count, degree=degree,
        )
        self.spatial_router = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=False),
            nn.BatchNorm2d(channels * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 3, 3, 1),
        )
        self.output_project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.residual_gain = nn.Parameter(torch.zeros(1))

    def _morphology_descriptor(self, source, expert3, expert5):
        batch, _, height, width = source.shape
        grouped = source.reshape(
            batch, self.groups, self.group_channels, height, width
        )
        mean = grouped.mean(dim=(2, 3, 4))
        std = torch.sqrt(
            grouped.var(dim=(2, 3, 4), unbiased=False) + self.eps
        )

        group_map = grouped.mean(dim=2)
        grad_x = F.pad(group_map[:, :, :, 2:] - group_map[:, :, :, :-2], (1, 1, 0, 0)) * 0.5
        grad_y = F.pad(group_map[:, :, 2:, :] - group_map[:, :, :-2, :], (0, 0, 1, 1)) * 0.5
        edge_energy = torch.sqrt(grad_x.square() + grad_y.square() + self.eps).mean(dim=(2, 3))

        disagreement = (expert3 - expert5).abs().reshape(
            batch, self.groups, self.group_channels, height, width
        ).mean(dim=(2, 3, 4))
        return torch.stack([mean, std, edge_energy, disagreement], dim=-1)

    def forward(self, feature):
        source = self.input_project(feature)
        expert_features = [expert(source) for expert in self.experts]
        descriptor = self._morphology_descriptor(
            source, expert_features[0], expert_features[1]
        )
        channel_weights = self.channel_router(descriptor)
        spatial_weights = torch.softmax(
            self.spatial_router(torch.cat(expert_features, dim=1)), dim=1
        )

        batch, _, height, width = feature.shape
        channel_weights = channel_weights.repeat_interleave(
            self.group_channels, dim=1
        ).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        spatial_weights = spatial_weights.unsqueeze(2)
        joint_weights = channel_weights * spatial_weights
        joint_weights = joint_weights / (joint_weights.sum(dim=1, keepdim=True) + self.eps)
        stacked = torch.stack(expert_features, dim=1)
        fused = (joint_weights * stacked).sum(dim=1)
        output = feature + self.residual_gain * self.output_project(fused)

        average_route = channel_weights[:, :, :, 0, 0].mean(dim=2)
        route_balance = (
            average_route - 1.0 / len(self.experts)
        ).square().mean(dim=1)
        return output, route_balance


class MorphologySplineKernelRouter(nn.Module):
    """MSKR: decoder-skip morphology descriptors drive spline scale routing."""

    def __init__(self, channels, groups=8, basis_count=5, degree=3, max_gain=0.25, eps=1e-6):
        super().__init__()
        if channels % groups != 0:
            raise ValueError(f"channels={channels} must be divisible by groups={groups}")
        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups
        self.max_gain = max_gain
        self.eps = eps
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 1, bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 1, bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        channels, channels, 3, padding=3, dilation=3,
                        groups=channels, bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 1, bias=False),
                ),
            ]
        )
        self.router = GroupedSplineScaleRouter(
            groups=groups,
            descriptor_dim=4,
            scales=len(self.experts),
            basis_count=basis_count,
            degree=degree,
        )
        self.output_project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.residual_gain = nn.Parameter(torch.tensor(0.10))

    def _descriptor(self, feature, up_feature, skip_feature):
        batch, _, height, width = feature.shape
        grouped = feature.reshape(batch, self.groups, self.group_channels, height, width)
        mean = grouped.mean(dim=(2, 3, 4))
        std = torch.sqrt(grouped.var(dim=(2, 3, 4), unbiased=False) + self.eps)

        group_map = grouped.mean(dim=2)
        grad_x = F.pad(group_map[:, :, :, 2:] - group_map[:, :, :, :-2], (1, 1, 0, 0)) * 0.5
        grad_y = F.pad(group_map[:, :, 2:, :] - group_map[:, :, :-2, :], (0, 0, 1, 1)) * 0.5
        edge_density = torch.sqrt(grad_x.square() + grad_y.square() + self.eps).mean(dim=(2, 3))

        up_group = up_feature.reshape(batch, self.groups, self.group_channels, height, width)
        skip_group = skip_feature.reshape(batch, self.groups, self.group_channels, height, width)
        numerator = (up_group * skip_group).mean(dim=(2, 3, 4))
        denominator = torch.sqrt(
            up_group.square().mean(dim=(2, 3, 4))
            * skip_group.square().mean(dim=(2, 3, 4))
            + self.eps
        )
        correlation = numerator / denominator
        return torch.stack([mean, std, edge_density, correlation], dim=-1)

    def forward(self, feature, up_feature, skip_feature):
        expert_features = [expert(feature) for expert in self.experts]
        weights = self.router(self._descriptor(feature, up_feature, skip_feature))
        channel_weights = weights.repeat_interleave(self.group_channels, dim=1)
        channel_weights = channel_weights.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        stacked = torch.stack(expert_features, dim=1)
        fused = (channel_weights * stacked).sum(dim=1)
        gain = self.max_gain * torch.tanh(self.residual_gain)
        output = feature + gain * self.output_project(fused)
        route_balance = (weights.mean(dim=1) - 1.0 / len(self.experts)).square().mean(dim=1)
        return output, route_balance


class DGBCARMSKRFuse(nn.Module):
    """DG-BCAR skip purification followed by optional middle-scale MSKR."""

    def __init__(self, channels, use_mskr=False):
        super().__init__()
        self.skip_attention = DecoderGuidedBCAR(channels)
        self.up_refine = MultiKernelRefineBlock(channels)
        self.skip_refine = MultiKernelRefineBlock(channels)
        self.fuse = DoubleConv(channels * 2, channels)
        self.mskr = MorphologySplineKernelRouter(channels) if use_mskr else None

    def forward(self, up_feature, skip_feature):
        guided_skip = self.skip_attention(up_feature, skip_feature)
        up_refined = self.up_refine(up_feature)
        skip_refined = self.skip_refine(guided_skip)
        fused = self.fuse(torch.cat([up_refined, skip_refined], dim=1))
        if self.mskr is None:
            return fused, None
        return self.mskr(fused, up_refined, skip_refined)


class MBSRDecoderStage(nn.Module):
    """BCAR on skip, MSAK after fusion, and optional high-resolution USNCR."""

    def __init__(self, channels, use_usncr=False, usncr_radius=2.0):
        super().__init__()
        self.skip_attention = BoundaryConsistencyAdaptiveResidualSE(channels)
        self.fuse = DoubleConv(channels * 2, channels)
        self.msak = MorphologyAwareSplineAdaptiveKernel(channels)
        self.usncr = (
            USNCRRefinement(channels, radius=usncr_radius)
            if use_usncr else None
        )

    def forward(self, up_feature, skip_feature):
        skip_feature = self.skip_attention(skip_feature)
        fused = self.fuse(torch.cat([up_feature, skip_feature], dim=1))
        enhanced, route_balance = self.msak(fused)
        if self.usncr is None:
            return enhanced, route_balance, None, None
        refined, sdf, coarse = self.usncr(enhanced)
        return refined, route_balance, sdf, coarse


class MBSRUNet(nn.Module):
    def __init__(self, in_ch, out_ch, channels):
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
        self.stage6 = MBSRDecoderStage(c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.stage7 = MBSRDecoderStage(c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.stage8 = MBSRDecoderStage(c2, use_usncr=True, usncr_radius=2.0)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.stage9 = MBSRDecoderStage(c1, use_usncr=True, usncr_radius=3.0)

        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.conv5(self.pool4(c4))

        c6, route6, _, _ = self.stage6(self.up6(c5), c4)
        c7, route7, _, _ = self.stage7(self.up7(c6), c3)
        c8, route8, sdf8, coarse8 = self.stage8(self.up8(c7), c2)
        c9, route9, sdf9, coarse9 = self.stage9(self.up9(c8), c1)
        output = self.last_activation(self.conv10(c9))
        if self.training:
            return (
                output,
                (sdf8, sdf9),
                (coarse8, coarse9),
                (route6, route7, route8, route9),
            )
        return output


class BSADecoderRefineFuse(nn.Module):
    """Decoder refinement using Boundary-Speckle Adaptive SE."""

    def __init__(self, up_channels, skip_channels, out_channels):
        super().__init__()
        self.up_recalibrate = nn.Sequential(
            BoundarySpeckleAdaptiveSE(up_channels),
            MultiKernelRefineBlock(up_channels),
            KANSelectiveKernel(up_channels),
        )
        self.skip_recalibrate = nn.Sequential(
            BoundarySpeckleAdaptiveSE(skip_channels),
            MultiKernelRefineBlock(skip_channels),
            KANSelectiveKernel(skip_channels),
        )
        self.fuse = DoubleConv(up_channels + skip_channels, out_channels)
        self.out_recalibrate = nn.Sequential(
            BoundarySpeckleAdaptiveSE(out_channels),
            MultiKernelRefineBlock(out_channels),
            KANSelectiveKernel(out_channels),
        )

    def forward(self, up_feature, skip_feature):
        up_feature = self.up_recalibrate(up_feature)
        skip_feature = self.skip_recalibrate(skip_feature)
        fused = self.fuse(torch.cat([up_feature, skip_feature], dim=1))
        return self.out_recalibrate(fused)


class DecoderGuidedSkipFuse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up_context = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.skip_context = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 8), channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.fuse = DoubleConv(channels * 2, channels)
        self.refine = OptionalEnhance(channels, use_mk=True, use_kan=True)

    def forward(self, up_feature, skip_feature):
        up_context = self.up_context(up_feature)
        skip_context = self.skip_context(skip_feature)
        channel_gate = self.channel_gate(up_context)
        spatial_gate = self.spatial_gate(torch.cat([up_context, skip_context], dim=1))
        skip_feature = skip_feature * (1.0 + channel_gate * spatial_gate)
        fused = self.fuse(torch.cat([up_feature, skip_feature], dim=1))
        return self.refine(fused)


class SobelBoundaryAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.edge_project = nn.Sequential(
            nn.Conv2d(1, max(channels // 8, 8), 3, padding=1, bias=False),
            nn.BatchNorm2d(max(channels // 8, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 8), 1, 1),
            nn.Sigmoid(),
        )
        self.feature_project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.feature_project[-1].weight)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        edge_gate = self.edge_project(edge)
        return x + self.feature_project(x * edge_gate)


class BoundaryDrivenFuse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse = DoubleConv(channels * 2, channels)
        self.se = SELayer(channels)
        self.boundary = SobelBoundaryAttention(channels)
        self.refine = OptionalEnhance(channels, use_mk=True, use_kan=True)

    def forward(self, up_feature, skip_feature):
        fused = self.fuse(torch.cat([up_feature, skip_feature], dim=1))
        fused = self.se(fused)
        fused = self.boundary(fused)
        return self.refine(fused)


class ScaleMoEKANFuse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse = DoubleConv(channels * 2, channels)
        self.small_expert = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.mid_expert = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.large_expert = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        hidden = max(channels // 8, 16)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden * 4, channels * 3, 1)
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.post_refine = OptionalEnhance(channels, use_mk=True, use_kan=True)
        nn.init.zeros_(self.project[1].weight)

    def forward(self, up_feature, skip_feature):
        fused = self.fuse(torch.cat([up_feature, skip_feature], dim=1))
        experts = torch.stack(
            [self.small_expert(fused), self.mid_expert(fused), self.large_expert(fused)],
            dim=1,
        )
        z = self.reduce(F.adaptive_avg_pool2d(fused, 1))
        basis = torch.cat([z, z * z, torch.sin(z), torch.cos(z)], dim=1)
        weights = self.expand(basis).view(fused.size(0), 3, fused.size(1), 1, 1)
        weights = torch.softmax(weights, dim=1)
        mixed = (experts * weights).sum(dim=1)
        return self.post_refine(fused + self.project(mixed))


class ChannelSEUNet(nn.Module):
    def __init__(self, in_ch, out_ch, channels, use_mk=False, use_kan=False, use_rpa=False):
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
        self.enhance5 = OptionalEnhance(c5, use_mk, use_kan, use_rpa)

        self.se1 = SELayer(c4)
        self.se2 = SELayer(c3)
        self.se3 = SELayer(c2)
        self.se4 = SELayer(c1)
        self.enhance6 = OptionalEnhance(c4, use_mk, use_kan, use_rpa)
        self.enhance7 = OptionalEnhance(c3, use_mk, use_kan, use_rpa)
        self.enhance8 = OptionalEnhance(c2, use_mk, use_kan, use_rpa)
        self.enhance9 = OptionalEnhance(c1, use_mk, use_kan, use_rpa)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.conv6 = DoubleConv(c4 * 2, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.conv7 = DoubleConv(c3 * 2, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.conv8 = DoubleConv(c2 * 2, c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.conv9 = DoubleConv(c1 * 2, c1)
        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))
        c6 = self.conv6(torch.cat([self.up6(c5), c4], dim=1))
        c6 = self.enhance6(self.se1(c6))
        c7 = self.conv7(torch.cat([self.up7(c6), c3], dim=1))
        c7 = self.enhance7(self.se2(c7))
        c8 = self.conv8(torch.cat([self.up8(c7), c2], dim=1))
        c8 = self.enhance8(self.se3(c8))
        c9 = self.conv9(torch.cat([self.up9(c8), c1], dim=1))
        c9 = self.enhance9(self.se4(c9))
        logits = self.conv10(c9)
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return logits
        return self.last_activation(logits)


class ChannelBCARUNet(nn.Module):
    """BCAR-only ablation on the plain B512 U-Net decoder.

    This model intentionally excludes MKR, NBSK and DecoderRefine so the
    isolated contribution of BCAR can be compared with SE-U-Net.
    """

    def __init__(self, in_ch, out_ch, channels):
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
        self.conv6 = DoubleConv(c4 * 2, c4)
        self.bcar6 = BoundaryConsistencyAdaptiveResidualSE(c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.conv7 = DoubleConv(c3 * 2, c3)
        self.bcar7 = BoundaryConsistencyAdaptiveResidualSE(c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.conv8 = DoubleConv(c2 * 2, c2)
        self.bcar8 = BoundaryConsistencyAdaptiveResidualSE(c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.conv9 = DoubleConv(c1 * 2, c1)
        self.bcar9 = BoundaryConsistencyAdaptiveResidualSE(c1)

        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.conv5(self.pool4(c4))

        c6 = self.bcar6(self.conv6(torch.cat([self.up6(c5), c4], dim=1)))
        c7 = self.bcar7(self.conv7(torch.cat([self.up7(c6), c3], dim=1)))
        c8 = self.bcar8(self.conv8(torch.cat([self.up8(c7), c2], dim=1)))
        c9 = self.bcar9(self.conv9(torch.cat([self.up9(c8), c1], dim=1)))
        return self.last_activation(self.conv10(c9))


class ChannelSEUNetMKKANDecoderRefine(nn.Module):
    def __init__(self, in_ch, out_ch, channels):
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
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = DecoderRefineFuse(c4, c4, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = DecoderRefineFuse(c3, c3, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = DecoderRefineFuse(c2, c2, c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = DecoderRefineFuse(c1, c1, c1)

        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))

        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        c9 = self.refine9(self.up9(c8), c1)
        return self.last_activation(self.conv10(c9))


class ChannelBSAUNetMKKANDecoderRefine(nn.Module):
    def __init__(self, in_ch, out_ch, channels):
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
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = BSADecoderRefineFuse(c4, c4, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = BSADecoderRefineFuse(c3, c3, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = BSADecoderRefineFuse(c2, c2, c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = BSADecoderRefineFuse(c1, c1, c1)

        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))

        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        c9 = self.refine9(self.up9(c8), c1)
        return self.last_activation(self.conv10(c9))


class ChannelBCARUNetMKKANDecoderRefine(nn.Module):
    def __init__(self, in_ch, out_ch, channels):
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
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = BCARDecoderRefineFuse(c4, c4, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = BCARDecoderRefineFuse(c3, c3, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = BCARDecoderRefineFuse(c2, c2, c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = BCARDecoderRefineFuse(c1, c1, c1)

        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))

        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        c9 = self.refine9(self.up9(c8), c1)
        logits = self.conv10(c9)
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return logits
        return self.last_activation(logits)


class ChannelMCRCMechanismAblationUNetMKKANDecoderRefine(
    ChannelBCARUNetMKKANDecoderRefine
):
    """Keep the full spatial backbone fixed and replace only MCRC."""

    def __init__(self, in_ch, out_ch, channels, variant):
        super().__init__(in_ch, out_ch, channels)
        scale_channels = {
            "refine6": channels[3],
            "refine7": channels[2],
            "refine8": channels[1],
            "refine9": channels[0],
        }
        for name, current_channels in scale_channels.items():
            refine = getattr(self, name)
            refine.up_recalibrate[0] = build_mcrc_mechanism_variant(
                current_channels, variant
            )
            refine.skip_recalibrate[0] = build_mcrc_mechanism_variant(
                current_channels, variant
            )
            refine.out_recalibrate[0] = build_mcrc_mechanism_variant(
                current_channels, variant
            )


class ChannelBCARFrequencyVariantUNetMKKANDecoderRefine(
    ChannelBCARUNetMKKANDecoderRefine
):
    """Strongest BCAR with configurable frequency residual evidence."""

    def __init__(
        self,
        in_ch,
        out_ch,
        channels,
        mode="anisotropic",
        preserve_phase=True,
        zero_initialize=True,
        active_scales=("refine6", "refine7", "refine8", "refine9"),
    ):
        super().__init__(in_ch, out_ch, channels)
        c1, c2, c3, c4, _ = channels
        scale_channels = {
            "refine6": c4,
            "refine7": c3,
            "refine8": c2,
            "refine9": c1,
        }
        unknown = set(active_scales).difference(scale_channels)
        if unknown:
            raise ValueError(
                "Unknown BCAR frequency scales: {}".format(sorted(unknown))
            )
        for name in active_scales:
            channels_at_scale = scale_channels[name]
            setattr(
                self,
                name,
                BCARFrequencyDecoderRefineFuse(
                    channels_at_scale,
                    channels_at_scale,
                    channels_at_scale,
                    mode=mode,
                    preserve_phase=preserve_phase,
                    zero_initialize=zero_initialize,
                ),
            )


class ChannelBCARUPAFUNetMKKANDecoderRefine(
    ChannelBCARFrequencyVariantUNetMKKANDecoderRefine
):
    """Full four-scale anisotropic phase-preserving BCAR frequency model."""

    def __init__(self, in_ch, out_ch, channels):
        super().__init__(in_ch, out_ch, channels)


class ChannelMCRCIdentityFrequencyUNetMKKANDecoderRefine(
    ChannelBCARFrequencyVariantUNetMKKANDecoderRefine
):
    """Matched frequency branch with every MCRC operator removed."""

    def __init__(self, in_ch, out_ch, channels):
        super().__init__(in_ch, out_ch, channels)
        for name in ("refine6", "refine7", "refine8", "refine9"):
            refine = getattr(self, name)
            refine.up_recalibrate[0] = nn.Identity()
            refine.skip_recalibrate[0] = nn.Identity()
            refine.out_recalibrate[0] = nn.Identity()


class ChannelEIBCNet(nn.Module):
    """BCMR backbone with ordered boundary-interval auxiliary predictions.

    The core and envelope logits are parameterized around the nominal lesion
    logit, so their probabilities are ordered by construction. Auxiliary heads
    are used only during training; evaluation returns the nominal prediction.
    """

    def __init__(self, in_ch, out_ch, channels, use_interval, use_echo):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.use_interval = use_interval
        self.use_echo_randomization = use_echo

        self.conv1 = DoubleConv(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(c4, c5)
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = BCARDecoderRefineFuse(c4, c4, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = BCARDecoderRefineFuse(c3, c3, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = BCARDecoderRefineFuse(c2, c2, c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = BCARDecoderRefineFuse(c1, c1, c1)

        self.main_head = nn.Conv2d(c1, out_ch, 1)
        if use_interval:
            hidden = max(c1 // 2, 16)
            self.interval_head = nn.Sequential(
                nn.Conv2d(c1, hidden, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 2 * out_ch, 1, bias=True),
            )
            nn.init.zeros_(self.interval_head[-1].weight)
            nn.init.constant_(self.interval_head[-1].bias, -2.0)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))

        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        c9 = self.refine9(self.up9(c8), c1)
        main_logits = self.main_head(c9)

        if self.training:
            outputs = {"main": main_logits}
            if self.use_interval:
                inward_raw, outward_raw = self.interval_head(c9).chunk(2, dim=1)
                inward = F.softplus(inward_raw)
                outward = F.softplus(outward_raw)
                outputs["core"] = main_logits - inward
                outputs["envelope"] = main_logits + outward
            return outputs

        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return main_logits
        return torch.sigmoid(main_logits)


class DistanceConfidenceReconciliation(nn.Module):
    """Boundary-local signed-distance evidence with bounded logit fusion."""

    def __init__(self, channels, out_channels=1, use_fusion=True, max_gain=0.25):
        super().__init__()
        hidden = max(channels // 2, 16)
        projected = max(channels // 4, 8)
        self.use_fusion = use_fusion
        self.max_gain = max_gain
        self.distance_head = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=True),
            nn.Tanh(),
        )
        self.distance_log_scale = nn.Parameter(torch.tensor(1.3862944))

        if use_fusion:
            self.feature_project = nn.Sequential(
                nn.Conv2d(channels, projected, 1, bias=False),
                nn.BatchNorm2d(projected),
                nn.ReLU(inplace=True),
            )
            self.reliability = nn.Sequential(
                nn.Conv2d(projected + 3 * out_channels, hidden, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, 1, bias=True),
            )
            self.fusion_gain = nn.Parameter(torch.zeros(1))

    def forward(self, feature, base_logits):
        signed_distance = self.distance_head(feature)
        distance_scale = self.distance_log_scale.exp().clamp(2.0, 8.0)
        distance_logits = distance_scale * signed_distance

        if not self.use_fusion:
            return base_logits, signed_distance, distance_logits

        base_probability = torch.sigmoid(base_logits)
        distance_probability = torch.sigmoid(distance_logits)
        disagreement = (base_probability - distance_probability).abs()
        uncertainty = 4.0 * base_probability * (1.0 - base_probability)
        gate_input = torch.cat(
            [
                self.feature_project(feature),
                base_probability,
                distance_probability,
                disagreement,
            ],
            dim=1,
        )
        reliability = torch.sigmoid(self.reliability(gate_input))
        correction = (
            self.max_gain
            * torch.tanh(self.fusion_gain)
            * uncertainty
            * reliability
            * (distance_logits - base_logits).clamp(-4.0, 4.0)
        )
        return base_logits + correction, signed_distance, distance_logits


class ChannelDCRNet(nn.Module):
    """BCAR-DRNet with training-supervised distance-confidence reconciliation."""

    def __init__(self, in_ch, out_ch, channels, use_fusion):
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
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = BCARDecoderRefineFuse(c4, c4, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = BCARDecoderRefineFuse(c3, c3, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = BCARDecoderRefineFuse(c2, c2, c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = BCARDecoderRefineFuse(c1, c1, c1)

        self.main_head = nn.Conv2d(c1, out_ch, 1)
        self.dcr = DistanceConfidenceReconciliation(
            c1, out_channels=out_ch, use_fusion=use_fusion
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))

        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        c9 = self.refine9(self.up9(c8), c1)
        base_logits = self.main_head(c9)
        final_logits, signed_distance, distance_logits = self.dcr(c9, base_logits)

        use_logits = os.environ.get("UC_OUTPUT_LOGITS", "0") == "1"
        main_output = final_logits if use_logits else torch.sigmoid(final_logits)
        if self.training:
            return {
                "main": main_output,
                "sdf": signed_distance,
                "distance_logits": distance_logits,
            }
        return main_output


class ChannelBCARDGResidualUNetMKKANDecoderRefine(nn.Module):
    """Weight-compatible BCAR model with only decoder-guided skip residuals."""

    def __init__(self, in_ch, out_ch, channels):
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
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = BCARDGResidualDecoderRefineFuse(c4, c4, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = BCARDGResidualDecoderRefineFuse(c3, c3, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = BCARDGResidualDecoderRefineFuse(c2, c2, c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = BCARDGResidualDecoderRefineFuse(c1, c1, c1)

        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))

        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        c9 = self.refine9(self.up9(c8), c1)
        return self.last_activation(self.conv10(c9))


class ChannelBCARSCFAUNetMKKANDecoderRefine(nn.Module):
    """Weight-compatible BCAR model with ultrasound-specific SCFA adapters."""

    def __init__(self, in_ch, out_ch, channels):
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
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = BCARSCFADecoderRefineFuse(c4, c4, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = BCARSCFADecoderRefineFuse(c3, c3, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = BCARSCFADecoderRefineFuse(c2, c2, c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = BCARSCFADecoderRefineFuse(c1, c1, c1)

        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))
        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        c9 = self.refine9(self.up9(c8), c1)
        return self.last_activation(self.conv10(c9))


class ChannelBCARMICRAUNetMKKANDecoderRefine(nn.Module):
    """Original BCAR with identity-initialized inner/outer contour residuals."""

    def __init__(self, in_ch, out_ch, channels):
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
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = BCARDecoderRefineFuse(c4, c4, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = BCARDecoderRefineFuse(c3, c3, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = BCARDecoderRefineFuse(c2, c2, c2)
        self.micra8 = MorphologicalInnerOuterContourResidual(c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = BCARDecoderRefineFuse(c1, c1, c1)
        self.micra9 = MorphologicalInnerOuterContourResidual(c1)

        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))
        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        residual8 = self.micra8(c8)
        c9 = self.refine9(self.up9(c8), c1)
        residual9 = self.micra9(c9)
        logits = self.conv10(c9)
        residual8 = F.interpolate(
            residual8, size=logits.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.last_activation(logits + residual8 + residual9)


class ChannelBCARWSCARUNetMKKANDecoderRefine(nn.Module):
    """BCAR backbone with wavelet speckle-coherence adaptive skip residuals."""

    def __init__(self, in_ch, out_ch, channels):
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
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = BCARWSCARDecoderRefineFuse(c4, c4, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = BCARWSCARDecoderRefineFuse(c3, c3, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = BCARWSCARDecoderRefineFuse(c2, c2, c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = BCARWSCARDecoderRefineFuse(c1, c1, c1)
        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))
        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        c9 = self.refine9(self.up9(c8), c1)
        return self.last_activation(self.conv10(c9))


class DGBCARUNet(nn.Module):
    """DG-BCAR ablation on the B512 MK-KAN decoder-refine backbone."""

    def __init__(self, in_ch, out_ch, channels):
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
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = DGBCARDecoderRefineFuse(c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = DGBCARDecoderRefineFuse(c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = DGBCARDecoderRefineFuse(c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = DGBCARDecoderRefineFuse(c1)
        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))
        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        c9 = self.refine9(self.up9(c8), c1)
        return self.last_activation(self.conv10(c9))


class DGBCARMSKRUNet(nn.Module):
    """DG-BCAR with middle-resolution morphology spline kernel routing."""

    def __init__(self, in_ch, out_ch, channels, boundary_supervision=False):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.boundary_supervision = boundary_supervision
        self.conv1 = DoubleConv(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(c4, c5)
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = DGBCARMSKRFuse(c4, use_mskr=True)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = DGBCARMSKRFuse(c3, use_mskr=True)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = DGBCARMSKRFuse(c2, use_mskr=False)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = DGBCARMSKRFuse(c1, use_mskr=False)

        if boundary_supervision:
            self.boundary8 = self._boundary_head(c2)
            self.boundary9 = self._boundary_head(c1)
        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    @staticmethod
    def _boundary_head(channels):
        hidden = max(channels // 4, 8)
        return nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))
        c6, route6 = self.refine6(self.up6(c5), c4)
        c7, route7 = self.refine7(self.up7(c6), c3)
        c8, _ = self.refine8(self.up8(c7), c2)
        c9, _ = self.refine9(self.up9(c8), c1)
        output = self.last_activation(self.conv10(c9))
        if self.training:
            boundary_preds = (
                (self.boundary8(c8), self.boundary9(c9))
                if self.boundary_supervision else ()
            )
            return output, (), (), (route6, route7), boundary_preds
        return output


class ChannelSEUNetMKKANUSNCR(nn.Module):
    def __init__(self, in_ch, out_ch, channels):
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
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = DecoderRefineFuse(c4, c4, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = DecoderRefineFuse(c3, c3, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = DecoderRefineFuse(c2, c2, c2)
        self.usncr8 = USNCRRefinement(c2, radius=2.0)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = DecoderRefineFuse(c1, c1, c1)
        self.usncr9 = USNCRRefinement(c1, radius=3.0)

        self.conv10 = nn.Conv2d(c1, out_ch, 1)
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.enhance5(self.conv5(self.pool4(c4)))
        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8, sdf8, coarse8 = self.usncr8(self.refine8(self.up8(c7), c2))
        c9, sdf9, coarse9 = self.usncr9(self.refine9(self.up9(c8), c1))
        output = self.last_activation(self.conv10(c9))
        if self.training:
            return output, (sdf8, sdf9), (coarse8, coarse9)
        return output


class ChannelSEUNetMKKANDecoderVariant(nn.Module):
    def __init__(self, in_ch, out_ch, channels, fuse_block):
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
        self.enhance5 = OptionalEnhance(c5, use_mk=True, use_kan=True)

        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.fuse6 = fuse_block(c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.fuse7 = fuse_block(c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.fuse8 = fuse_block(c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.fuse9 = fuse_block(c1)

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


class ChannelCMNKBiCERCalibratedUNet(ChannelSEUNetMKKANDecoderVariant):
    """BiCER model with a two-parameter calibration control."""

    def __init__(self, in_ch, out_ch, channels):
        super().__init__(in_ch, out_ch, channels, BiCERDecoderRefineFuse)
        self.calibrator = LearnableLogitCalibration()

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
        logits = self.calibrator(self.conv10(c9))
        return self.last_activation(logits)


class ChannelCMNKBiCERSCPRUNet(ChannelSEUNetMKKANDecoderVariant):
    """BiCER with scale-conditioned prototype-connectivity recalibration."""

    def __init__(self, in_ch, out_ch, channels):
        super().__init__(in_ch, out_ch, channels, BiCERDecoderRefineFuse)
        self.scpr = ScaleConditionedPrototypeConnectivityRecalibration(
            channels[0]
        )

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
        logits = self.scpr(c9, self.conv10(c9))
        return self.last_activation(logits)


CHANNELS = {
    "B512": [32, 64, 128, 256, 512],
    "B1024": [64, 128, 256, 512, 1024],
}


FACTORY_MODEL_CHOICES = [
    "UNet",
    "UNet_B512",
    "UNet_B1024",
    "UNetPlusPlus",
    "UNetPlusPlus_B512",
    "UNetPlusPlus_B1024",
    "AttentionUNet",
    "AttentionUNet_B512",
    "DeepLabV3Plus",
    "ResUnet",
    "ResUnet_B512",
    "ResUnet_B1024",
    "SE_UNet",
    "SE_UNet_B512",
    "SE_UNet_B1024",
    "BCAR_UNet_B512",
    "SwinUNet",
    "SwinUNet_B512",
    "SwinUNet_B1024",
    "TransUNet",
    "TransUNet_B512",
    "TransUNet_B1024",
    "OfficialTransUNet_R50_ViT_B16",
    "SE_UNet_MK",
    "SE_UNet_MK_B512",
    "SE_UNet_MK_B1024",
    "SE_UNet_KAN",
    "SE_UNet_KAN_B512",
    "SE_UNet_KAN_B1024",
    "SE_UNet_RPA",
    "SE_UNet_RPA_B512",
    "SE_UNet_RPA_B1024",
    "SE_UNet_MK_KAN",
    "SE_UNet_MK_KAN_B512",
    "SE_UNet_MK_KAN_B1024",
    "SE_UNet_MK_KAN_DecoderRefine",
    "SE_UNet_MK_KAN_DecoderRefine_B512",
    "SE_UNet_MK_KAN_DecoderRefine_B1024",
    "BSA_SE_UNet_MK_KAN_DecoderRefine",
    "BSA_SE_UNet_MK_KAN_DecoderRefine_B512",
    "BSA_SE_UNet_MK_KAN_DecoderRefine_B1024",
    "BCAR_SE_UNet_MK_KAN_DecoderRefine",
    "BCAR_SE_UNet_MK_KAN_DecoderRefine_B512",
    "BCAR_SE_UNet_MK_KAN_DecoderRefine_B1024",
    "BCAR_UPAF_UNet_B512",
    "BCAR_FREQ_Isotropic_UNet_B512",
    "BCAR_FREQ_NoPhase_UNet_B512",
    "BCAR_FREQ_NonZeroInit_UNet_B512",
    "BCAR_FREQ_DeepOnly_UNet_B512",
    "BCAR_FREQ_ShallowOnly_UNet_B512",
    "UNet_FREQ_B512",
    "MCRC_Identity_UNet_B512",
    "MCRC_ChannelOnly_UNet_B512",
    "MCRC_SpatialOnly_UNet_B512",
    "MCRC_NoAgreement_UNet_B512",
    "MCRC_NoScaleNoise_UNet_B512",
    "MCRC_MeanOnly_UNet_B512",
    "MCRC_NonZeroInit_UNet_B512",
    "MCRC_Identity_FREQ_UNet_B512",
    "PSCA_BASE_UNet_B512",
    "PSCA_ONLY_UNet_B512",
    "DPSD_ONLY_UNet_B512",
    "PSCA_DPSD_UNet_B512",
    "PSCA_DUAL_EVIDENCE_UNet_B512",
    "PSCA_SPECTRAL_CALIBRATOR_UNet_B512",
    "PSCA_ERROR_POLICY_UNet_B512",
    "PSCA_BOUNDARY_POLICY_UNet_B512",
    "PSCA_COUNTERFACTUAL_POLICY_UNet_B512",
    "PSCA_RISK_PROJECTED_UNet_B512",
    "PSCA_REGION_PROJECTED_UNet_B512",
    "PSCA_OVERLAP_GUARD_UNet_B512",
    "PSCA_AXIAL_EQUIVARIANT_UNet_B512",
    "PSCA_HIERARCHICAL_UNet_B512",
    "PSCA_STATE_SHRINKAGE_UNet_B512",
    "PSCA_DCRA_UNet_B512",
    "PSCA_CDAC_UNet_B512",
    "PSCA_CDAC_Equal_UNet_B512",
    "PSCA_CDAC_NoDisagreement_UNet_B512",
    "PSCA_CDAC_NoConfidence_UNet_B512",
    "PSCA_DPSD_NoPrior_UNet_B512",
    "PSCA_DPSD_BinaryRouter_UNet_B512",
    "PSCA_DPSD_NoSuppression_UNet_B512",
    "PSCA_DPSD_NoConflict_UNet_B512",
    "MFMSNet_Official_FS256",
    "BCAR_LiteBase_UNet_B512",
    "BCAR_LiteKAN_UNet_B512",
    "BCAR_DiffusionRandomFrozen_UNet_B512",
    "BCAR_DiffusionFrozen_UNet_B512",
    "BCAR_DINOv3_Frozen_UNet_B512",
    "EIBC_Base_B512",
    "EIBC_Echo_B512",
    "EIBC_Interval_B512",
    "EIBC_Net_B512",
    "DCR_Base_B512",
    "DCR_Distance_B512",
    "DCR_Net_B512",
    "BCAR_DGResidual_UNet_B512",
    "BCAR_SCFA_UNet_B512",
    "BCAR_MICRA_UNet_B512",
    "BCAR_WSCAR_UNet_B512",
    "CAMP_Base_B512",
    "CAMP_CAF_B512",
    "CAMP_MPR_B512",
    "CAMP_TCE_B512",
    "CAMP_CAF_MPR_B512",
    "CAMP_Net_B512",
    "ACET_Base_B512",
    "ACET_Axial_B512",
    "ACET_Lateral_B512",
    "ACET_NoReliability_B512",
    "ACET_Independent_B512",
    "ACET_Net_B512",
    "QDK_Base_B512",
    "QDK_QEB_B512",
    "QDK_ACD_B512",
    "QDK_DKR_B512",
    "QDK_QEB_ACD_B512",
    "QDK_Net_B512",
    "SCT_Base_B512",
    "SCT_CTE_B512",
    "SCT_LRG_B512",
    "SCT_LWR_B512",
    "SCT_NoLocal_B512",
    "SCT_Net_B512",
    "RBFC_Base_B512",
    "RBFC_Region_B512",
    "RBFC_Boundary_B512",
    "RBFC_NoAgreement_B512",
    "RBFC_NoFallback_B512",
    "RBFC_Net_B512",
    "RBOF_Base_B512",
    "RBOF_Net_B512",
    "UPAF_Base_B512",
    "UPAF_Isotropic_B512",
    "UPAF_Anisotropic_B512",
    "UPAF_Net_B512",
    "LFKAN_Base",
    "LFKAN_MLP",
    "LFKAN_Deep",
    "LFKAN_Net",
    "CMNK_BiCER_UNet_B512",
    "CMNK_BiCER_Calibrated_B512",
    "CMNK_BiCER_SCPR_UNet_B512",
    "DGBCAR_UNet",
    "DGBCAR_UNet_B512",
    "DGBCAR_MSKR_UNet",
    "DGBCAR_MSKR_UNet_B512",
    "DGBCAR_MSKR_Boundary_UNet",
    "DGBCAR_MSKR_Boundary_UNet_B512",
    "SE_UNet_MK_KAN_USNCR",
    "SE_UNet_MK_KAN_USNCR_B512",
    "SE_UNet_MK_KAN_USNCR_B1024",
    "MBSR_UNet",
    "MBSR_UNet_B512",
    "SE_UNet_MK_KAN_DGSkip",
    "SE_UNet_MK_KAN_DGSkip_B512",
    "SE_UNet_MK_KAN_DGSkip_B1024",
    "SE_UNet_MK_KAN_BDR",
    "SE_UNet_MK_KAN_BDR_B512",
    "SE_UNet_MK_KAN_BDR_B1024",
    "SE_UNet_MK_KAN_ScaleMoE",
    "SE_UNet_MK_KAN_ScaleMoE_B512",
    "SE_UNet_MK_KAN_ScaleMoE_B1024",
    "SE_UNet_MK_RPA",
    "SE_UNet_MK_RPA_B512",
    "SE_UNet_MK_RPA_B1024",
    "SE_UNet_KAN_RPA",
    "SE_UNet_KAN_RPA_B512",
    "SE_UNet_KAN_RPA_B1024",
    "SE_UNet_MK_KAN_RPA",
    "SE_UNet_MK_KAN_RPA_B512",
    "SE_UNet_MK_KAN_RPA_B1024",
]
FACTORY_MODEL_CHOICES += [
    f"{base_name}_B512" for base_name in TPB_MODEL_BASE_NAMES
]


def _baseline_suffix(model_name):
    if model_name.endswith("_B1024"):
        return "B1024", model_name[:-6]
    if model_name.endswith("_B512"):
        return "B512", model_name[:-5]
    return "B512", model_name


def create_model(model_name, in_ch=3, out_ch=1):
    suffix, base_name = _baseline_suffix(model_name)
    channels = CHANNELS[suffix]

    if model_name == "UNet":
        return UNet(in_ch, out_ch)
    if base_name == "UNet":
        if suffix == "B1024":
            return UNet(in_ch, out_ch)
        return ChannelUNet(in_ch, out_ch, channels)
    if base_name in ("UNetPlusPlus", "UNet++"):
        return NestedUNet(in_channels=in_ch, num_classes=out_ch, deep_supervision=True,
                          bilinear=True, base_channels=channels)
    if base_name in ("AttentionUNet", "Attention_UNet", "AttUNet"):
        return ChannelAttentionUNet(in_ch, out_ch, channels)
    if base_name in ("DeepLabV3Plus", "DeepLabv3+"):
        return DeepLabV3Plus(
            n_classes=out_ch,
            n_blocks=[3, 4, 6, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,
        )
    if base_name in ("ResUnet", "ResUNet"):
        if suffix == "B1024":
            return ResUnet(in_channels=in_ch, num_classes=out_ch, base_c=64, bilinear=False)
        return ResUnet(in_channels=in_ch, num_classes=out_ch, base_c=32, bilinear=False)
    if base_name in ("SE_UNet", "SEUNet"):
        if model_name == "SE_UNet":
            return SE_UNet(in_ch=in_ch, out_ch=out_ch)
        return ChannelSEUNet(in_ch, out_ch, channels)
    if base_name == "BCAR_UNet":
        return ChannelBCARUNet(in_ch, out_ch, channels)
    if base_name in ("SwinUNet", "Swin-Unet"):
        return SwinUNet(in_ch=in_ch, out_ch=out_ch, base_ch=64 if suffix == "B1024" else 32)
    if base_name in ("TransUNet", "TransUnet"):
        return TransUNet(in_ch=in_ch, out_ch=out_ch, base_ch=64 if suffix == "B1024" else 32)
    if model_name == "OfficialTransUNet_R50_ViT_B16":
        if OfficialTransUNetR50ViTB16 is None:
            raise ImportError(
                "Official TransUNet requires the separately obtained "
                "third_party/TransUNet_official repository."
            )
        return OfficialTransUNetR50ViTB16(in_ch=in_ch, out_ch=out_ch, img_size=224)
    if base_name == "SE_UNet_MK_KAN_DecoderRefine":
        return ChannelSEUNetMKKANDecoderRefine(in_ch, out_ch, channels)
    if base_name == "BSA_SE_UNet_MK_KAN_DecoderRefine":
        return ChannelBSAUNetMKKANDecoderRefine(in_ch, out_ch, channels)
    if base_name == "BCAR_SE_UNet_MK_KAN_DecoderRefine":
        return ChannelBCARUNetMKKANDecoderRefine(in_ch, out_ch, channels)
    if base_name == "BCAR_UPAF_UNet":
        return ChannelBCARUPAFUNetMKKANDecoderRefine(
            in_ch, out_ch, channels
        )
    bcar_frequency_variants = {
        "BCAR_FREQ_Isotropic_UNet": dict(mode="isotropic"),
        "BCAR_FREQ_NoPhase_UNet": dict(preserve_phase=False),
        "BCAR_FREQ_NonZeroInit_UNet": dict(zero_initialize=False),
        "BCAR_FREQ_DeepOnly_UNet": dict(active_scales=("refine6",)),
        "BCAR_FREQ_ShallowOnly_UNet": dict(active_scales=("refine9",)),
    }
    if base_name in bcar_frequency_variants:
        return ChannelBCARFrequencyVariantUNetMKKANDecoderRefine(
            in_ch,
            out_ch,
            channels,
            **bcar_frequency_variants[base_name],
        )
    if base_name == "UNet_FREQ":
        return build_upaf_net(
            "UNet_FREQ",
            in_ch=in_ch,
            out_ch=out_ch,
            channels=channels,
        )
    mcrc_mechanism_variants = {
        "MCRC_Identity_UNet": "identity",
        "MCRC_ChannelOnly_UNet": "channel_only",
        "MCRC_SpatialOnly_UNet": "spatial_only",
        "MCRC_NoAgreement_UNet": "no_agreement",
        "MCRC_NoScaleNoise_UNet": "no_scale_noise",
        "MCRC_MeanOnly_UNet": "mean_only",
        "MCRC_NonZeroInit_UNet": "nonzero_init",
    }
    if base_name in mcrc_mechanism_variants:
        return ChannelMCRCMechanismAblationUNetMKKANDecoderRefine(
            in_ch,
            out_ch,
            channels,
            variant=mcrc_mechanism_variants[base_name],
        )
    if base_name == "MCRC_Identity_FREQ_UNet":
        return ChannelMCRCIdentityFrequencyUNetMKKANDecoderRefine(
            in_ch,
            out_ch,
            channels,
        )
    psca_variants = {
        "PSCA_BASE_UNet": "base",
        "PSCA_ONLY_UNet": "psca_only",
        "DPSD_ONLY_UNet": "dpsd_only",
        "PSCA_DPSD_UNet": "full",
        "PSCA_DUAL_EVIDENCE_UNet": "dual_evidence",
        "PSCA_ERROR_POLICY_UNet": "policy_error",
        "PSCA_BOUNDARY_POLICY_UNet": "policy_boundary",
        "PSCA_COUNTERFACTUAL_POLICY_UNet": "policy_counterfactual",
        "PSCA_RISK_PROJECTED_UNet": "policy_risk_projected",
        "PSCA_REGION_PROJECTED_UNet": "policy_region_projected",
        "PSCA_OVERLAP_GUARD_UNet": "policy_overlap_guard",
        "PSCA_AXIAL_EQUIVARIANT_UNet": "policy_axial_equivariant",
        "PSCA_HIERARCHICAL_UNet": "hierarchical_consensus",
        "PSCA_STATE_SHRINKAGE_UNet": "state_shrinkage",
        "PSCA_DCRA_UNet": "disagreement_reconciliation",
        "PSCA_CDAC_UNet": "cross_domain_consensus",
        "PSCA_CDAC_Equal_UNet": "cross_domain_equal",
        "PSCA_CDAC_NoDisagreement_UNet": (
            "cross_domain_no_disagreement"
        ),
        "PSCA_CDAC_NoConfidence_UNet": "cross_domain_no_confidence",
        "PSCA_DPSD_NoPrior_UNet": "no_prior",
        "PSCA_DPSD_BinaryRouter_UNet": "binary_router",
        "PSCA_DPSD_NoSuppression_UNet": "no_suppression",
        "PSCA_DPSD_NoConflict_UNet": "no_conflict",
    }
    if base_name in psca_variants:
        return build_psca_model(
            in_ch,
            out_ch,
            channels,
            variant=psca_variants[base_name],
        )
    if base_name == "PSCA_SPECTRAL_CALIBRATOR_UNet":
        return build_psca_spectral_calibrator(
            in_ch,
            out_ch,
            channels,
        )
    if base_name == "MFMSNet_Official_FS256":
        return MFMSNetOfficialFS256(in_ch=in_ch, out_ch=out_ch)
    if base_name == "BCAR_LiteBase_UNet":
        return build_bcar_lite_model(
            False,
            in_ch,
            out_ch,
            channels,
            BCARDecoderRefineFuse,
            OptionalEnhance,
        )
    if base_name == "BCAR_LiteKAN_UNet":
        return build_bcar_lite_model(
            True,
            in_ch,
            out_ch,
            channels,
            BCARDecoderRefineFuse,
            OptionalEnhance,
        )
    if base_name == "BCAR_DiffusionRandomFrozen_UNet":
        return build_bcar_diffusion_model(
            False,
            in_ch,
            out_ch,
            channels,
            BCARDecoderRefineFuse,
            OptionalEnhance,
        )
    if base_name == "BCAR_DiffusionFrozen_UNet":
        return build_bcar_diffusion_model(
            True,
            in_ch,
            out_ch,
            channels,
            BCARDecoderRefineFuse,
            OptionalEnhance,
        )
    if base_name == "BCAR_DINOv3_Frozen_UNet":
        return build_bcar_dinov3_model(
            in_ch,
            out_ch,
            channels,
            BCARDecoderRefineFuse,
            OptionalEnhance,
        )
    eibc_variants = {
        "EIBC_Base": (False, False),
        "EIBC_Echo": (False, True),
        "EIBC_Interval": (True, False),
        "EIBC_Net": (True, True),
    }
    if base_name in eibc_variants:
        use_interval, use_echo = eibc_variants[base_name]
        return ChannelEIBCNet(
            in_ch, out_ch, channels,
            use_interval=use_interval,
            use_echo=use_echo,
        )
    if base_name == "DCR_Base":
        return ChannelBCARUNetMKKANDecoderRefine(in_ch, out_ch, channels)
    if base_name == "DCR_Distance":
        return ChannelDCRNet(in_ch, out_ch, channels, use_fusion=False)
    if base_name == "DCR_Net":
        return ChannelDCRNet(in_ch, out_ch, channels, use_fusion=True)
    if base_name.startswith("UPAF_"):
        return build_upaf_net(base_name, in_ch, out_ch, channels)
    if base_name.startswith("LFKAN_"):
        return build_lfkan_net(base_name, in_ch, out_ch)
    if base_name == "BCAR_DGResidual_UNet":
        return ChannelBCARDGResidualUNetMKKANDecoderRefine(
            in_ch, out_ch, channels
        )
    if base_name == "BCAR_SCFA_UNet":
        return ChannelBCARSCFAUNetMKKANDecoderRefine(
            in_ch, out_ch, channels
        )
    if base_name == "BCAR_MICRA_UNet":
        return ChannelBCARMICRAUNetMKKANDecoderRefine(
            in_ch, out_ch, channels
        )
    if base_name == "BCAR_WSCAR_UNet":
        return ChannelBCARWSCARUNetMKKANDecoderRefine(
            in_ch, out_ch, channels
        )
    camp_variants = {
        "CAMP_Base": (False, False, False),
        "CAMP_CAF": (True, False, False),
        "CAMP_MPR": (False, True, False),
        "CAMP_TCE": (False, False, True),
        "CAMP_CAF_MPR": (True, True, False),
        "CAMP_Net": (True, True, True),
    }
    if base_name in camp_variants:
        use_caf, use_mpr, use_tce = camp_variants[base_name]
        return build_camp_net(
            in_ch=in_ch,
            out_ch=out_ch,
            channels=channels,
            use_contrast_fields=use_caf,
            use_prototype_router=use_mpr,
            use_contour_evolution=use_tce,
        )
    acet_variants = {
        "ACET_Base": "base",
        "ACET_Axial": "axial",
        "ACET_Lateral": "lateral",
        "ACET_NoReliability": "no_reliability",
        "ACET_Independent": "independent",
        "ACET_Net": "full",
    }
    if base_name in acet_variants:
        return build_acet_net(
            in_ch=in_ch,
            out_ch=out_ch,
            channels=channels,
            mode=acet_variants[base_name],
        )
    qdk_variants = {
        "QDK_Base": (False, False, False),
        "QDK_QEB": (True, False, False),
        "QDK_ACD": (False, True, False),
        "QDK_DKR": (False, False, True),
        "QDK_QEB_ACD": (True, True, False),
        "QDK_Net": (True, True, True),
    }
    if base_name in qdk_variants:
        use_qeb, use_acd, use_dkr = qdk_variants[base_name]
        return build_qdk_net(
            in_ch=in_ch,
            out_ch=out_ch,
            channels=channels,
            use_qeb=use_qeb,
            use_acd=use_acd,
            use_dkr=use_dkr,
        )
    sct_variants = {
        "SCT_Base": "base",
        "SCT_CTE": "cte",
        "SCT_LRG": "lrg",
        "SCT_LWR": "lwr",
        "SCT_NoLocal": "no_local",
        "SCT_Net": "full",
    }
    if base_name in sct_variants:
        return build_sct_net(
            in_ch=in_ch,
            out_ch=out_ch,
            channels=channels,
            mode=sct_variants[base_name],
        )
    rbfc_variants = {
        "RBFC_Base": "base",
        "RBFC_Region": "region",
        "RBFC_Boundary": "boundary",
        "RBFC_NoAgreement": "no_agreement",
        "RBFC_NoFallback": "no_fallback",
        "RBFC_Net": "full",
    }
    if base_name in rbfc_variants:
        return build_rbfc_net(
            in_ch=in_ch,
            out_ch=out_ch,
            channels=channels,
            mode=rbfc_variants[base_name],
        )
    if base_name == "RBOF_Base":
        return build_rbfc_net(
            in_ch=in_ch, out_ch=out_ch, channels=channels, mode="base"
        )
    if base_name == "RBOF_Net":
        return build_rbfc_net(
            in_ch=in_ch, out_ch=out_ch, channels=channels, mode="no_agreement"
        )
    if base_name == "CMNK_BiCER_UNet":
        return ChannelSEUNetMKKANDecoderVariant(
            in_ch, out_ch, channels, BiCERDecoderRefineFuse
        )
    if base_name == "CMNK_BiCER_Calibrated":
        return ChannelCMNKBiCERCalibratedUNet(
            in_ch, out_ch, channels
        )
    if base_name == "CMNK_BiCER_SCPR_UNet":
        return ChannelCMNKBiCERSCPRUNet(in_ch, out_ch, channels)
    if base_name in TPB_MODEL_BASE_NAMES:
        return create_tpb_model(base_name, in_ch, out_ch, channels)
    if base_name == "DGBCAR_UNet":
        return DGBCARUNet(in_ch, out_ch, channels)
    if base_name == "DGBCAR_MSKR_UNet":
        return DGBCARMSKRUNet(in_ch, out_ch, channels, boundary_supervision=False)
    if base_name == "DGBCAR_MSKR_Boundary_UNet":
        return DGBCARMSKRUNet(in_ch, out_ch, channels, boundary_supervision=True)
    if base_name == "SE_UNet_MK_KAN_USNCR":
        return ChannelSEUNetMKKANUSNCR(in_ch, out_ch, channels)
    if base_name == "MBSR_UNet":
        return MBSRUNet(in_ch, out_ch, channels)
    decoder_variants = {
        "SE_UNet_MK_KAN_DGSkip": DecoderGuidedSkipFuse,
        "SE_UNet_MK_KAN_BDR": BoundaryDrivenFuse,
        "SE_UNet_MK_KAN_ScaleMoE": ScaleMoEKANFuse,
    }
    if base_name in decoder_variants:
        return ChannelSEUNetMKKANDecoderVariant(in_ch, out_ch, channels, decoder_variants[base_name])

    enhance_flags = {
        "SE_UNet_MK": (True, False, False),
        "SE_UNet_KAN": (False, True, False),
        "SE_UNet_RPA": (False, False, True),
        "SE_UNet_MK_KAN": (True, True, False),
        "SE_UNet_MK_RPA": (True, False, True),
        "SE_UNet_KAN_RPA": (False, True, True),
        "SE_UNet_MK_KAN_RPA": (True, True, True),
        "SEUNet_MK_KAN_RPA": (True, True, True),
        "SE_MK_KAN_RPA": (True, True, True),
    }
    if base_name in enhance_flags:
        if suffix == "B512" and model_name == base_name:
            legacy = {
                "SE_UNet_MK": SE_UNet_MK,
                "SE_UNet_KAN": SE_UNet_KAN,
                "SE_UNet_RPA": SE_UNet_RPA,
                "SE_UNet_MK_KAN": SE_UNet_MK_KAN,
                "SE_UNet_MK_RPA": SE_UNet_MK_RPA,
                "SE_UNet_KAN_RPA": SE_UNet_KAN_RPA,
                "SE_UNet_MK_KAN_RPA": SE_UNet_MK_KAN_RPA,
                "SEUNet_MK_KAN_RPA": SE_UNet_MK_KAN_RPA,
                "SE_MK_KAN_RPA": SE_UNet_MK_KAN_RPA,
            }
            return legacy[base_name](in_ch=in_ch, out_ch=out_ch)
        return ChannelSEUNet(in_ch, out_ch, channels, *enhance_flags[base_name])

    raise ValueError(f"Unsupported model: {model_name}")
