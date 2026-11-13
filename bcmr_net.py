"""Public entry point for the archived BCMR-Net implementation.

The IJIST manuscript names the evaluated model BCMR-Net. Its historical
training identifier is ``BCAR_SE_UNet_MK_KAN_DecoderRefine_B512`` and its
implementation class is ``ChannelBCARUNetMKKANDecoderRefine``. This module
exposes that exact class under a manuscript-facing name without changing the
forward graph or state-dict keys.
"""

from model_factory import ChannelBCARUNetMKKANDecoderRefine


ARCHIVED_MODEL_NAME = "BCAR_SE_UNet_MK_KAN_DecoderRefine_B512"
DEFAULT_CHANNELS = (32, 64, 128, 256, 512)

# An alias preserves the exact archived implementation and checkpoint keys.
BCMRNet = ChannelBCARUNetMKKANDecoderRefine


def build_bcmr_net(
    in_channels=3,
    out_channels=1,
    channels=DEFAULT_CHANNELS,
):
    """Build the BCMR-Net architecture evaluated in the IJIST manuscript."""

    return BCMRNet(in_channels, out_channels, list(channels))


__all__ = [
    "ARCHIVED_MODEL_NAME",
    "BCMRNet",
    "DEFAULT_CHANNELS",
    "build_bcmr_net",
]
