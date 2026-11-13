# BCMR-Net implementation

This repository also contains the code evaluated in the manuscript:

**Boundary-consistency calibration and multiscale decoder refinement for breast ultrasound lesion segmentation**

## Exact implementation mapping

The manuscript name **BCMR-Net** maps to the following archived code without
reparameterization or pruning:

| Manuscript term | Repository implementation |
|---|---|
| BCMR-Net | `ChannelBCARUNetMKKANDecoderRefine` |
| Historical run/model identifier | `BCAR_SE_UNet_MK_KAN_DecoderRefine_B512` |
| Public constructor | `bcmr_net.build_bcmr_net()` |
| Core implementation | `model_factory.py` |
| Training entry point | `Train_Unet.py` |
| Evaluation entry point | `Test.py` |

The default channel widths are `[32, 64, 128, 256, 512]`. The public
constructor is an alias around the archived implementation, so existing
checkpoint parameter names remain unchanged.

## Audited interpretation

The evaluated system contains boundary-consistency adaptive residual
calibration (BCAR), multi-kernel refinement/routing, and multistage decoder
refinement. Historical NBSK-labelled residual branches were present in the
archived model definition but produced identity mappings in the audited
checkpoints. They are retained here for checkpoint fidelity and are not
presented as an independently effective contribution. This release does not
claim component-wise causality, an optimal operation order, zero-shot external
generalization, or clinical readiness.

## Installation and smoke test

Create the environment described in the repository README, then run:

```bash
python experiment_scripts/smoke_test_bcmr.py
```

The test instantiates BCMR-Net, performs a deterministic CPU forward pass, and
checks the output shape and probability range. It does not access a dataset or
checkpoint and does not reproduce performance values.

The model can also be instantiated directly:

```python
import torch

from bcmr_net import build_bcmr_net

model = build_bcmr_net().eval()
image = torch.randn(1, 3, 224, 224)
with torch.inference_mode():
    probability = model(image)
```

## Training interface

The following recipe documents the retained historical BUS-BRA BCMR-Net
configuration (`BCAR_BASE`), not the BUSI runs or local comparator recipes.
Use the original patient-isolated split (folds 3-5 for training, fold 2 for
validation and fold 1 reserved for test). Set the two dataset paths in a
fresh PowerShell session. Run once per seed (666, 3407, 2027) with a distinct
session name. The generic trainer defaults to a different probability-based
loss; both the raw-logit output and loss settings below are required.

```powershell
$seed = 666
$trainPath = 'PATH_TO_PREPARED_TRAIN'
$valPath = 'PATH_TO_PREPARED_VALIDATION'
$env:CUDA_VISIBLE_DEVICES = '0,1'
$env:UC_AMP = '0'
$env:UC_CHANNELS_LAST = '0'
$env:UC_OUTPUT_LOGITS = '1'
$env:UC_LOSS_MODE = 'logits_dice_bce'
$env:UC_DISABLE_LR_SCHEDULER = '1'
$env:UC_PRINT_FREQUENCY = '9999'
$env:UC_TRAIN_METRICS_INTERVAL = '0'
$env:UC_PIN_MEMORY = '0'
$env:UC_NUM_WORKERS = '8'
$env:UC_LR = '1e-4'
$env:UC_SEED = [string]$seed
$env:UC_DETERMINISTIC = '1'
$env:PYTHONHASHSEED = [string]$seed
$env:PYTHONUNBUFFERED = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python Train_Unet.py --model BCAR_SE_UNet_MK_KAN_DecoderRefine_B512 `
  --train-dataset $trainPath --val-dataset $valPath `
  --batch-size 32 --epochs 100 --gpu 0,1 --data-parallel `
  --early-stopping 100000 --min-epochs 100 --train-min-delta 0 `
  --val-interval 1 --vis-frequency 0 --num-workers 8 `
  --session-name "BCMR_BUSBRA_S${seed}"
```

The loss gives equal weight to per-image soft Dice loss after sigmoid
(smoothing constant 1e-5) and BCEWithLogits. Validation uses a 0.5
probability threshold and saves a new best checkpoint only when Dice
strictly improves; a later tie does not replace an existing best. Test was
not a selection input. The pinned [Zenodo software release](https://doi.org/10.5281/zenodo.22536370)
contains the implementation, not a complete immutable record of the
historical training environment or the manuscript-specific S2/S3 bundles.
This recipe was reconstructed from retained run records; it is not a
claim of bitwise reproducibility. Dataset images, clinical records and
trained checkpoints are not redistributed by this guide.
