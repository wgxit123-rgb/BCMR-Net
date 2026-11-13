import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFMSNetOfficialFS256(nn.Module):
    """Run the public MFMSNet architecture at its required 256-pixel size."""

    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        if in_ch != 3:
            raise ValueError("MFMSNet official implementation requires three input channels")

        source_dir = (
            Path(__file__).resolve().parent
            / "external_repos"
            / "MFMSNet"
            / "Oct_ResNet"
        )
        source_file = source_dir / "oct.py"
        if not source_file.is_file():
            raise FileNotFoundError(
                "MFMSNet source is missing; expected external_repos/MFMSNet/Oct_ResNet/oct.py"
            )

        source_dir_text = str(source_dir)
        if source_dir_text not in sys.path:
            sys.path.insert(0, source_dir_text)

        module_name = "_mfmsnet_official_oct"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, source_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load MFMSNet from {source_file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        # The public repository does not distribute its referenced pretrained file.
        self.network = module.oct_resnet50_fusion(
            pretrained=False,
            num_classes=out_ch,
        )
        self.native_size = (256, 256)

    def forward(self, x):
        output_size = x.shape[-2:]
        if output_size != self.native_size:
            x = F.interpolate(
                x,
                size=self.native_size,
                mode="bilinear",
                align_corners=False,
            )
        logits = self.network(x)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        if os.environ.get("UC_OUTPUT_LOGITS", "0") == "1":
            return logits
        return torch.sigmoid(logits)
