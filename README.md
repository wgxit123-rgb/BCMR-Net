# BCMR-Net code companion

BCMR-Net is a breast ultrasound lesion segmentation architecture. This code-only
companion contains the archived model, trainer, evaluator, boundary metrics, and a
synthetic smoke test. It does not include the study manuscript, datasets, case-level
scores or split keys, model weights, review files, or unpublished analysis scripts.

## Setup and synthetic check

The dependency list documents the local compatibility environment, not an immutable
snapshot of the historical training run. Use Python 3.8 (checked locally with
Python 3.8.20). Use matching PyTorch 2.3.0 and torchvision 0.18.0 builds appropriate
for your platform and CPU or CUDA setup. The pins below include both packages; when
choosing platform-specific builds, use the PyTorch distribution instructions.

```sh
python -m pip install -r requirements.txt
python experiment_scripts/smoke_test_bcmr.py
```

The smoke test uses a generated CPU tensor and random weights. It checks the output
shape, numerical finiteness, and the default probability range. It does not estimate
segmentation performance. Run `python Train_Unet.py --help` and
`python Test.py --help` to inspect the archived command-line interfaces.

## Historical training and evaluation

See [BCMR_NET.md](BCMR_NET.md) for the model identifier and retained BUS-BRA
training recipe. Obtain [BUS-BRA](https://doi.org/10.5281/zenodo.8231412) directly
from its provider. For each split, the prepared dataset root has `images/` and
`masks/` subfolders; preserve the original image-to-mask correspondence. Respect
the official patient groups and the documented split: folds 3-5 for training,
fold 2 for validation, and fold 1 for test. Check patient isolation before
training, and keep the test split out of training and checkpoint selection.

`Test.py` requires a locally supplied compatible checkpoint and prepared evaluation
data. Load only trusted checkpoints. The historical threshold was 0.5 on
probabilities; set `UC_OUTPUT_LOGITS=1` for evaluation to apply sigmoid to the
raw-logit model output, as the historical recipe did. The original checkpoint
files are not redistributed. The external official TransUNet/MFMSNet backends are
not bundled or needed for BCMR-Net. Their optional adapters and local factory
dependencies are kept to preserve the original model entry point. The factory
import guard also handles environments where a third-party namespace exists
without the optional TransUNet backend. This package
does not reproduce the manuscript's saved-score statistics or a complete
historical training environment.

## Citation and permissions

Cite the [archived upstream CET-PSCA software release, v1.0.2](https://doi.org/10.5281/zenodo.22536370).
The DOI and [CITATION.cff](CITATION.cff) describe that upstream release, not a DOI
for this code-only selection. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)
identifies the local source commit and records the shipped file hashes.
Software credit is separate from paper authorship.

The upstream [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md)
are retained. The MIT license does not cover third-party datasets, model weights,
or externally licensed code. This repository is research software, not a clinical tool.
