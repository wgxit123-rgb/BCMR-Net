import copy
import os

import numpy as np
import torch
import torch.nn as nn

from third_party.TransUNet_official.networks.vit_seg_modeling import CONFIGS, VisionTransformer


class OfficialTransUNetR50ViTB16(nn.Module):
    """Thin binary-segmentation adapter around the official TransUNet model."""

    def __init__(self, in_ch=3, out_ch=1, img_size=224):
        super().__init__()
        if in_ch not in (1, 3):
            raise ValueError("Official TransUNet supports one- or three-channel input")

        config = copy.deepcopy(CONFIGS["R50-ViT-B_16"])
        config.n_classes = out_ch
        config.n_skip = 3
        config.patches.grid = (img_size // 16, img_size // 16)
        self.network = VisionTransformer(config, img_size=img_size, num_classes=out_ch)

        if os.environ.get("TRANSUNET_LOAD_PRETRAINED", "0") == "1":
            weight_path = os.environ.get("TRANSUNET_PRETRAINED_PATH", "")
            if not weight_path or not os.path.isfile(weight_path):
                raise FileNotFoundError(
                    "TRANSUNET_PRETRAINED_PATH must point to R50+ViT-B_16.npz"
                )
            self.network.load_from(weights=np.load(weight_path))
            print("Loaded official TransUNet weights: {}".format(weight_path))

    def forward(self, x):
        return torch.sigmoid(self.network(x))
