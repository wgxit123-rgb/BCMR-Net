import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from SE_UNet_MK_KAN_RPA import DoubleConv
from lfkan_net import (
    ConvNormAct,
    GroupedSplineKANContext,
    LiteDownsample,
    LiteResidualBlock,
)


def _segmentation_output(logits):
    if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
        return logits
    return torch.sigmoid(logits)


def _strip_parallel_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[7:]: value for key, value in state_dict.items()}
    return state_dict


class BCARDecoderHead(nn.Module):
    """The strongest BCAR bottleneck and decoder over five-scale features."""

    def __init__(
        self,
        out_channels,
        channels,
        decoder_fuse_cls,
        enhance_cls,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.enhance5 = enhance_cls(c5, use_mk=True, use_kan=True)
        self.up6 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.refine6 = decoder_fuse_cls(c4, c4, c4)
        self.up7 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.refine7 = decoder_fuse_cls(c3, c3, c3)
        self.up8 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.refine8 = decoder_fuse_cls(c2, c2, c2)
        self.up9 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.refine9 = decoder_fuse_cls(c1, c1, c1)
        self.classifier = nn.Conv2d(c1, out_channels, 1)

    def forward(self, features):
        c1, c2, c3, c4, c5 = features
        c5 = self.enhance5(c5)
        c6 = self.refine6(self.up6(c5), c4)
        c7 = self.refine7(self.up7(c6), c3)
        c8 = self.refine8(self.up8(c7), c2)
        c9 = self.refine9(self.up9(c8), c1)
        return self.classifier(c9)


class BCARLitePyramidEncoder(nn.Module):
    """Full-resolution lightweight encoder with optional spline-KAN context."""

    def __init__(
        self,
        in_channels=3,
        widths=(24, 48, 96, 160, 256),
        output_channels=(32, 64, 128, 256, 512),
        use_kan=False,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = widths
        self.stem = ConvNormAct(
            in_channels, c1, kernel_size=3, stride=1
        )
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
        self.context4 = (
            GroupedSplineKANContext(c4) if use_kan else nn.Identity()
        )
        self.context5 = (
            GroupedSplineKANContext(c5) if use_kan else nn.Identity()
        )
        self.projections = nn.ModuleList(
            [
                ConvNormAct(source, target)
                for source, target in zip(widths, output_channels)
            ]
        )

    def forward(self, inputs):
        c1 = self.stage1(self.stem(inputs))
        c2 = self.stage2(self.down2(c1))
        c3 = self.stage3(self.down3(c2))
        c4 = self.context4(self.stage4(self.down4(c3)))
        c5 = self.context5(self.stage5(self.down5(c4)))
        features = (c1, c2, c3, c4, c5)
        return tuple(
            projection(feature)
            for projection, feature in zip(self.projections, features)
        )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        if dimension % 2:
            raise ValueError("time embedding dimension must be even")
        self.dimension = int(dimension)

    def forward(self, timesteps):
        half = self.dimension // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale
            * torch.arange(
                half,
                device=timesteps.device,
                dtype=torch.float32,
            )
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class DiffusionTimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels=128):
        super().__init__()
        self.input = ConvNormAct(
            in_channels, out_channels, kernel_size=3
        )
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.output = nn.Sequential(
            ConvNormAct(
                out_channels,
                out_channels,
                kernel_size=3,
            ),
            ConvNormAct(
                out_channels,
                out_channels,
                kernel_size=3,
                activate=False,
            ),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else ConvNormAct(
                in_channels, out_channels, activate=False
            )
        )

    def forward(self, features, time_embedding):
        hidden = self.input(features)
        hidden = hidden + self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        return F.silu(self.shortcut(features) + self.output(F.silu(hidden)))


class DiffusionPyramidEncoder(nn.Module):
    """Time-conditioned pyramid shared by denoising and segmentation."""

    def __init__(
        self,
        in_channels=3,
        channels=(32, 64, 128, 256, 512),
        time_channels=128,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_channels),
            nn.Linear(time_channels, time_channels),
            nn.SiLU(inplace=True),
            nn.Linear(time_channels, time_channels),
        )
        self.block1 = DiffusionTimeBlock(
            in_channels, c1, time_channels
        )
        self.down2 = ConvNormAct(
            c1, c2, kernel_size=3, stride=2
        )
        self.block2 = DiffusionTimeBlock(c2, c2, time_channels)
        self.down3 = ConvNormAct(
            c2, c3, kernel_size=3, stride=2
        )
        self.block3 = DiffusionTimeBlock(c3, c3, time_channels)
        self.down4 = ConvNormAct(
            c3, c4, kernel_size=3, stride=2
        )
        self.block4 = DiffusionTimeBlock(c4, c4, time_channels)
        self.down5 = ConvNormAct(
            c4, c5, kernel_size=3, stride=2
        )
        self.block5 = DiffusionTimeBlock(c5, c5, time_channels)

    def forward(self, inputs, timesteps=None):
        if timesteps is None:
            timesteps = torch.zeros(
                inputs.size(0),
                device=inputs.device,
                dtype=torch.long,
            )
        time_embedding = self.time_embedding(timesteps)
        c1 = self.block1(inputs, time_embedding)
        c2 = self.block2(self.down2(c1), time_embedding)
        c3 = self.block3(self.down3(c2), time_embedding)
        c4 = self.block4(self.down4(c3), time_embedding)
        c5 = self.block5(self.down5(c4), time_embedding)
        return c1, c2, c3, c4, c5


class DiffusionDenoiser(nn.Module):
    """Compact noise predictor used only for image-only pretraining."""

    def __init__(
        self,
        in_channels=3,
        channels=(32, 64, 128, 256, 512),
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.encoder = DiffusionPyramidEncoder(
            in_channels=in_channels,
            channels=channels,
        )
        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.decode4 = DoubleConv(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.decode3 = DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.decode2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.decode1 = DoubleConv(c1 * 2, c1)
        self.noise_head = nn.Conv2d(c1, in_channels, 1)

    def forward(self, noisy_images, timesteps):
        c1, c2, c3, c4, c5 = self.encoder(
            noisy_images, timesteps
        )
        d4 = self.decode4(torch.cat([self.up4(c5), c4], dim=1))
        d3 = self.decode3(torch.cat([self.up3(d4), c3], dim=1))
        d2 = self.decode2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.decode1(torch.cat([self.up1(d2), c1], dim=1))
        return self.noise_head(d1)


class DINOv3PyramidAdapter(nn.Module):
    """Frozen DINOv3 ViT features with trainable dense pyramid recovery."""

    def __init__(
        self,
        backbone,
        embedding_dim,
        channels=(32, 64, 128, 256, 512),
    ):
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        c1, c2, c3, c4, c5 = channels
        self.stem1 = ConvNormAct(3, c1, kernel_size=3)
        self.stem2 = ConvNormAct(
            c1, c2, kernel_size=3, stride=2
        )
        self.stem3 = ConvNormAct(
            c2, c3, kernel_size=3, stride=2
        )
        self.stem4 = ConvNormAct(
            c3, c4, kernel_size=3, stride=2
        )
        self.stem5 = ConvNormAct(
            c4, c5, kernel_size=3, stride=2
        )
        self.projections = nn.ModuleList(
            [
                ConvNormAct(embedding_dim, output_channels)
                for output_channels in (c2, c3, c4, c5)
            ]
        )
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _extract_maps(self, normalized):
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                normalized,
                n=4,
                reshape=True,
                norm=True,
            )
        maps = []
        for feature in features:
            if isinstance(feature, (tuple, list)):
                feature = feature[0]
            if feature.ndim != 4:
                raise RuntimeError(
                    "DINOv3 intermediate features must be BCHW maps"
                )
            maps.append(feature)
        if len(maps) != 4:
            raise RuntimeError(
                "DINOv3 backbone must return four intermediate maps"
            )
        return maps

    def forward(self, inputs):
        # Repository images are loaded by OpenCV (BGR); DINOv3 expects RGB.
        rgb = inputs[:, [2, 1, 0]]
        normalized = (rgb - self.image_mean) / self.image_std
        transformer_maps = self._extract_maps(normalized)

        c1 = self.stem1(inputs)
        c2 = self.stem2(c1)
        c2 = c2 + F.interpolate(
            self.projections[0](transformer_maps[0]),
            size=c2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        c3 = self.stem3(c2)
        c3 = c3 + F.interpolate(
            self.projections[1](transformer_maps[1]),
            size=c3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        c4 = self.stem4(c3)
        c4 = c4 + F.interpolate(
            self.projections[2](transformer_maps[2]),
            size=c4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        c5 = self.stem5(c4)
        c5 = c5 + F.interpolate(
            self.projections[3](transformer_maps[3]),
            size=c5.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return c1, c2, c3, c4, c5


class BCARExtractorUNet(nn.Module):
    def __init__(
        self,
        encoder,
        out_channels,
        channels,
        decoder_fuse_cls,
        enhance_cls,
        freeze_encoder=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
        self.decoder = BCARDecoderHead(
            out_channels,
            channels,
            decoder_fuse_cls,
            enhance_cls,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, inputs):
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(inputs)
        else:
            features = self.encoder(inputs)
        return _segmentation_output(self.decoder(features))


def _load_diffusion_encoder(encoder, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Diffusion encoder checkpoint not found: {}".format(
                checkpoint_path
            )
        )
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu"
    )
    state_dict = checkpoint.get(
        "encoder_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )
    state_dict = _strip_parallel_prefix(state_dict)
    if any(key.startswith("encoder.") for key in state_dict):
        state_dict = {
            key[len("encoder."):]: value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
    encoder.load_state_dict(state_dict, strict=True)


def build_bcar_lite_model(
    use_kan,
    in_channels,
    out_channels,
    channels,
    decoder_fuse_cls,
    enhance_cls,
):
    encoder = BCARLitePyramidEncoder(
        in_channels=in_channels,
        output_channels=channels,
        use_kan=use_kan,
    )
    return BCARExtractorUNet(
        encoder,
        out_channels,
        channels,
        decoder_fuse_cls,
        enhance_cls,
    )


def build_bcar_diffusion_model(
    pretrained,
    in_channels,
    out_channels,
    channels,
    decoder_fuse_cls,
    enhance_cls,
):
    encoder = DiffusionPyramidEncoder(
        in_channels=in_channels,
        channels=channels,
    )
    if pretrained:
        checkpoint = os.environ.get(
            "BCAR_DIFFUSION_ENCODER_CHECKPOINT"
        )
        if not checkpoint:
            count_target = os.environ.get("BCAR_COUNT_MODEL", "")
            if count_target != "BCAR_DiffusionFrozen_UNet_B512":
                raise RuntimeError(
                    "Set BCAR_DIFFUSION_ENCODER_CHECKPOINT to the image-only "
                    "diffusion pretraining checkpoint."
                )
        else:
            _load_diffusion_encoder(encoder, checkpoint)
    return BCARExtractorUNet(
        encoder,
        out_channels,
        channels,
        decoder_fuse_cls,
        enhance_cls,
        freeze_encoder=True,
    )


def load_official_dinov3_vits16():
    repo_path = os.environ.get("BCAR_DINOV3_REPO")
    weights_path = os.environ.get("BCAR_DINOV3_WEIGHTS")
    if not repo_path or not Path(repo_path).is_dir():
        raise RuntimeError(
            "Set BCAR_DINOV3_REPO to a local clone of the official "
            "facebookresearch/dinov3 repository."
        )
    if not weights_path or not Path(weights_path).is_file():
        raise RuntimeError(
            "Set BCAR_DINOV3_WEIGHTS to the access-approved official "
            "DINOv3 ViT-S/16 checkpoint."
        )
    return torch.hub.load(
        repo_path,
        "dinov3_vits16",
        source="local",
        weights=weights_path,
    )


def build_bcar_dinov3_model(
    in_channels,
    out_channels,
    channels,
    decoder_fuse_cls,
    enhance_cls,
):
    if in_channels != 3:
        raise ValueError("DINOv3 adapter requires three-channel input")
    backbone = load_official_dinov3_vits16()
    encoder = DINOv3PyramidAdapter(
        backbone,
        embedding_dim=384,
        channels=channels,
    )
    return BCARExtractorUNet(
        encoder,
        out_channels,
        channels,
        decoder_fuse_cls,
        enhance_cls,
    )
