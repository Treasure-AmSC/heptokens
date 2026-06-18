# heptokens

[![pytorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/-Lightning-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra-89b8cd?logo=hydra&logoColor=white)](https://hydra.cc/)
[![wandb](https://img.shields.io/badge/-WandB-orange?logo=weightsandbiases&logoColor=white)](https://wandb.ai)
[![pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)

Tokenize HEP physics objects (jets, tracks, calorimeter clusters, …) with residual VQ-VAEs.

`heptokens` is an installable Python library. It ships:
- **Data classes** — generic HDF5 datasets and Lightning DataModules for structured (object + constituent) and flat (dense, maskless) data.
- **VQ-VAE model** — `LitVqVae` wrapping `ResidualVQ` with pluggable encoder/decoder.
- **CLI entry points** — `heptokens-train` and `heptokens-export` for Hydra-configured training and tokenization runs.

## Quick Start

### 1. Install

```bash
pip install heptokens
```

Or, for development with a [Pixi](https://pixi.sh/latest/installation/)-managed environment:

```bash
git clone <repo>
cd heptokens
pixi shell          # activates the environment
```

### 2. Write your datamodule config

Create a YAML file describing your data. This is the only config you must supply — the library bundles everything else (model, callbacks, trainer).

```yaml
# my_configs/datamodule/my_jets.yaml
_target_: heptokens.data.structured_array.StructuredArrayModule
data_path: /path/to/data.h5
obj_group: jets
set_group: tracks
obj_features: [pt, eta, phi]
set_features: [pt, deta, dphi, d0, z0]
label_key: HadronConeExclTruthLabelID
mask_key: valid
train_frac: 0.7
val_frac: 0.15
test_frac: 0.15
batch_size: 1024
num_workers: 4
```

Your HDF5 must have this layout (see `StructuredArrayHDF5Dataset` for full docs):

```
<obj_group>/
    <feature>: float [N]
    <label_key>: int [N]
<set_group>/
    <feature>: float [N, max_elements]
    <mask_key>: bool [N, max_elements]
```

### 3. Fit preprocessors

The VQ-VAE expects normalised inputs. Fit sklearn scalers on your training data and save them:

```python
from heptokens.data.transforms import create_preprocessing_transformer
from joblib import dump

cst_scaler = create_preprocessing_transformer(
    mode="log_quantile",
    log_feature_indices=[0],   # index of pT in set_features
    n_quantiles=500,
)
cst_scaler.fit(training_constituents)  # [n_valid_csts, n_features]
dump(cst_scaler, "resources/cst_quantiles.joblib")

jet_scaler = create_preprocessing_transformer(mode="quantile", n_quantiles=500)
jet_scaler.fit(training_jets)  # [n_jets, n_jet_features]
dump(jet_scaler, "resources/jet_quantiles.joblib")
```

Then add them to your datamodule config:

```yaml
# append to my_configs/datamodule/my_jets.yaml
transforms:
  preprocess:
    _target_: heptokens.data.collation.preprocess_batch
    _partial_: true
    cst_fn:
      _target_: joblib.load
      filename: resources/cst_quantiles.joblib
    jet_fn:
      _target_: joblib.load
      filename: resources/jet_quantiles.joblib
```

See [Preprocessing](#preprocessing) for available modes and details.

### 4. Train

```bash
heptokens-train -cd /path/to/my_configs \
  datamodule=my_jets \
  project_name=my_project \
  network_name=run_01 \
  output_dir=results
```

### 5. Export tokens

```bash
heptokens-export -cd /path/to/my_configs \
  datamodule=my_jets \
  ckpt_path=results/my_project/run_01/checkpoints/last.ckpt \
  output_dir=results/tokens
```

Output is an `.npz` with `indices [N, n_csts, num_quantizers]`, `labels [N]`, and `codebooks [Q, K, D]`.

### Python API

For scripting or notebooks without Hydra:

```python
from heptokens.data.structured_array import StructuredArrayHDF5Dataset
from heptokens.models.vq_vae import LitVqVae
from heptokens.models.coders import Encoder, Decoder
import lightning as L
from torch.utils.data import DataLoader

ds = StructuredArrayHDF5Dataset(
    "data.h5",
    obj_group="jets",      set_group="tracks",
    obj_features=["pt", "eta", "phi"],
    set_features=["pt", "eta", "phi"],
    mask_key="valid",      label_key=None,
)
loader = DataLoader(ds, batch_size=256, shuffle=True)

sample = next(iter(loader))["csts"]          # [B, N, 3]
model = LitVqVae(
    encoder=Encoder, decoder=Decoder,
    codebook_size=512, codebook_dim=64, num_quantizers=4,
    data_sample=sample,
)

trainer = L.Trainer(max_epochs=10, logger=False)
trainer.fit(model, loader)

batch = next(iter(loader))
z_q, indices, _ = model.encode(batch)   # indices: [B, N, num_quantizers]
```

## CLI Reference

`heptokens-train` and `heptokens-export` are Hydra-based CLI tools. Use `-cd` to point at your experiment's config directory containing the `datamodule/` folder. All other configs (model, callbacks, trainer) are bundled.

```bash
# Without installing (pixi environment):
pixi run python -m heptokens.train -cd /path/to/my/configs datamodule=my_jets ...
pixi run python -m heptokens.export_tokens -cd /path/to/my/configs datamodule=my_jets ...
```

### Bundled configs

| Group | Location | Description |
|---|---|---|
| `model` | `src/heptokens/conf/model/` | `vqvae`, `transformer_vqvae`, `classifier`, … |
| `callbacks` | `src/heptokens/conf/callbacks/` | `pretrain`, `encode`, `classify`, … |

`datamodule` is not bundled — it is experiment-specific. See `StructuredArrayModule` in `src/heptokens/data/structured_array.py` for the constructor args.

### Output layout

```
<output_dir>/<project_name>/<network_name>/
    checkpoints/
    full_config.yaml      # saved for reproducibility / resume
```

### Resuming a run

```bash
heptokens-train -cd /path/to/my/configs \
  full_resume=true ckpt_flag=last.ckpt \
  output_dir=... project_name=... network_name=...
```

Or from an explicit checkpoint path:

```bash
heptokens-train -cd /path/to/my/configs \
  full_resume=false ckpt_path=/abs/path/to/last.ckpt \
  datamodule=my_atlas output_dir=... project_name=... network_name=...
```

## Repository Layout

```
src/heptokens/
    data/
        structured_array.py     # StructuredArrayHDF5Dataset / StructuredArrayModule
        flat_array.py           # FlatArrayHDF5Dataset / FlatArrayModule (dense, no mask)
        base.py                 # BaseMapModule (shared DataLoader logic)
        collation.py, transforms.py
    models/
        vq_vae.py               # LitVqVae
        coders.py               # Encoder, Decoder (base classes + default MLP impl)
        new_modality_coders.py  # template for custom encoders/decoders
        classifier.py, transformer.py, …
    conf/                       # bundled Hydra configs (model, callbacks)
    callbacks/                  # Lightning callbacks
    utils/
    train.py                    # heptokens-train entry point
    export_tokens.py            # heptokens-export entry point
tests/
profiles/                       # Slurm executor profiles for pixi tasks
```

## Adding a New Modality

A modality is any input representation with its own data format, token definition, and reconstruction metric. The VQ-VAE core (`LitVqVae`, `ResidualVQ`) is modality-agnostic and never changes.

### Token strategy

| Data type | Token strategy |
|---|---|
| Sparse set (tracks, clusters, PFOs) | One token per object. Boolean validity mask for variable-length padding. |
| Dense grid (η-φ image, calorimeter towers) | Fixed-size patching in `__getitem__`. Each patch is one token. No mask needed if all patches are always valid. |

### Checklist

**1. Dataset / DataModule** — `src/heptokens/data/<modality>.py`

See `StructuredArrayModule` or `FlatArrayModule` for reference implementations.
- `MyDataset.__getitem__` returns a dict: constituent tensor + optional bool mask.
- `MyDataModule` extends `BaseMapModule`; implement `setup()` and `get_data_sample()`.

**2. Encoder / Decoder** — `src/heptokens/models/<modality>_coders.py`

Start from `src/heptokens/models/new_modality_coders.py`.
- Extend `Encoder` and `Decoder` from `heptokens.models.coders`.
- `Encoder.forward(batch)` → `[B, N, codebook_dim]`, masked positions zeroed.
- `Decoder.compute_loss(z_q, batch)` → scalar loss.
- Set `self.input_key` and `self.mask_key` to match your dataset's output keys.

**3. Model config** — `src/heptokens/conf/model/<modality>_vqvae.yaml`

Start from `src/heptokens/conf/model/new_modality_vqvae.yaml`.
Point `encoder._target_` and `decoder._target_` at your new classes.

**4. Datamodule config** — in your experiment repo's config directory

```yaml
# my_configs/datamodule/my_modality.yaml
_target_: heptokens.data.<modality>.MyDataModule
data_path: /path/to/data.h5
batch_size: 512
num_workers: 4
# ... your fields
```

**5. Callbacks config** *(optional)* — `src/heptokens/conf/callbacks/<modality>_encode.yaml`

For physical-space reconstruction metrics, subclass `BaseReconstructionMonitor`:

```python
from heptokens.callbacks.recon import BaseReconstructionMonitor

class MyMonitor(BaseReconstructionMonitor):
    def __init__(self, inverse_fn, **kwargs):
        super().__init__(input_key="my_key", mask_key=None, **kwargs)
        self.inverse_fn = inverse_fn

    def compute_and_log_metrics(self, trainer, pl_module, batch, batch_idx):
        import torch as T
        with T.no_grad():
            z_q = pl_module.encode(batch)[0]
            recon = pl_module.decode(z_q, batch)
        orig = self.inverse_fn(batch["my_key"].cpu())
        reco = self.inverse_fn(recon.cpu())
        pl_module.log("val/my_mae_unscaled", T.mean(T.abs(orig - reco)))
```

**Train:**

```bash
heptokens-train -cd /path/to/my/configs \
  model=<modality>_vqvae callbacks=<modality>_encode \
  datamodule=<modality> trainer.max_epochs=30
```

### Common pitfalls

- **`input_key` mismatch**: The encoder reads `batch[self.input_key]`. Set it consistently in the dataset and encoder.
- **`n_classes` not needed for VQ-VAE**: Omit or set to `null` in your datamodule config.
- **Mask dtype**: The mask must be `bool`, shape `[B, N]`. Float/int masks will silently produce wrong results.
- **Dense grids**: Make sure `N` (number of patches) is fixed across samples so batches stack without a mask.

---

## Preprocessing

The VQ-VAE expects normalised inputs. The library provides sklearn-based scalers in `heptokens.data.transforms` and a batch-level application function in `heptokens.data.collation`. The workflow is: **fit offline → save as joblib → load in your Hydra config** (shown in [Quick Start step 3](#3-fit-preprocessors)).

### How it connects

`BaseMapModule` accepts a `transforms` kwarg. When set, it wraps the DataLoader's collate function so transforms run on every batch automatically — no manual intervention at train time.

### Available modes (`create_preprocessing_transformer`)

| Mode | Description |
|------|-------------|
| `"quantile"` | `QuantileTransformer` on all features |
| `"standard"` | `StandardScaler` on all features |
| `"log_quantile"` | Log-scale selected features, then quantile-transform all |
| `"log_standard"` | Log-scale selected features, then standard-scale all |

Parameters: `mode`, `log_feature_indices` (which features to log-scale), `n_quantiles`, `log_offset`.

### Inverse transforms

For evaluating reconstruction quality in physical units:

```python
from heptokens.data.collation import inverse_preprocess_batch

original = inverse_preprocess_batch(batch, cst_fn=cst_scaler, jet_fn=jet_scaler)
```

---

## Troubleshooting

### DataLoader workers exit unexpectedly
- Set `num_workers=0` to confirm worker issue.
- Set `pin_memory=False` if worker shutdown appears unstable.

### Training becomes slow
- Tune `num_workers`, `batch_size`, `pin_memory`.
- Avoid `reload_dataloaders_every_n_epochs=1` on iterable datasets unless necessary.

### W&B run lands under wrong account
- Override `logger.entity=<username_or_team>` when calling `heptokens-train`.

### Debugger uses wrong Python interpreter
- Use `.vscode/launch.json` with `"python": "${workspaceFolder}/.pixi/envs/default/bin/python"`.

---

## Resources
- [Pixi documentation](https://pixi.sh/)
- [Reproducible Machine Learning Workflows for Scientists, Matthew Feickert, 2025](https://carpentries-incubator.github.io/reproducible-ml-workflows/)
- [Good ML project structure](https://github.com/mattcleigh/JetSSL-Lite/tree/master)

## Contributors
Jeffrey Krupa, Samuel Klein, Michael Kagan
