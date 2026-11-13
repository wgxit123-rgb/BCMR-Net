"""Data-free smoke test for the public BCMR-Net entry point."""

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcmr_net import ARCHIVED_MODEL_NAME, build_bcmr_net


def main():
    torch.manual_seed(0)
    model = build_bcmr_net().cpu().eval()
    sample = torch.randn(1, 3, 64, 64)

    with torch.inference_mode():
        prediction = model(sample)

    expected_shape = (1, 1, 64, 64)
    if tuple(prediction.shape) != expected_shape:
        raise AssertionError(
            "Unexpected output shape: {} != {}".format(
                tuple(prediction.shape), expected_shape
            )
        )
    if not torch.isfinite(prediction).all():
        raise AssertionError("BCMR-Net produced non-finite values")
    if prediction.min().item() < 0.0 or prediction.max().item() > 1.0:
        raise AssertionError("BCMR-Net probability output is outside [0, 1]")

    report = {
        "model": "BCMR-Net",
        "archived_model_name": ARCHIVED_MODEL_NAME,
        "implementation_class": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": list(sample.shape),
        "output_shape": list(prediction.shape),
        "finite": True,
        "probability_range_ok": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
