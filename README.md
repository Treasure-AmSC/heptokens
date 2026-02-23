# gdig (Gold Diggers)
Tokenization and downstream classification workflows for ATLAS-style jet data.

This repository is built around Hydra configs, Lightning training loops, and Pixi-managed environments.

## Quick Start
1. Install [Pixi](https://pixi.sh/latest/installation/).
2. Clone the repo and `cd` to the top-level directory.
3. Create/use the environment:

```bash
pixi shell
```

or run commands without entering a shell:

```bash
pixi run python scripts/train.py ...
```

## Repository Layout
- `scripts/`: entry points for training, preprocessing, plotting, and profiling.
- `configs/`: Hydra config groups for models, callbacks, datamodules, and training defaults.
- `src/gdig/`: library code (models, dataloaders, callbacks, utilities).
- `workflow/`: Snakemake workflows for larger experiment grids.
- `profiles/`: Snakemake executor profiles (`local`, `s3df`/Slurm).

## Data and Preprocessing
Preprocessing files are required before normal training runs.

The training config expects:
- an input HDF5 jet dataset via `datamodule.data_path`
- preprocessing transformers (`cst_fn`, `jet_fn`) for `preprocess_batch`

Generate transformers with:

```bash
pixi run python scripts/get_preprocessing.py \
  --file_path /path/to/dataset.h5 \
  --num_jets 1000000 \
  --cst_mode log_quantile \
  --jet_mode log_quantile \
  --output_dir /path/to/resources
```

Then pass transformer files into training:

```bash
datamodule.transforms.preprocess.cst_fn.filename=/path/to/cst_quantiles.joblib
datamodule.transforms.preprocess.jet_fn.filename=/path/to/jet_quantiles.joblib
```

Important path note:
- `configs/hydra/default.yaml` sets `hydra.job.chdir=true`, so relative paths can be resolved from the run directory (`<output_dir>/<project>/<network>`), not the repo root.
- Prefer absolute transformer paths in CLI overrides.

## First Run Sequence
1. Build preprocessing transformers with `scripts/get_preprocessing.py`.
2. Launch training with transformer paths passed via Hydra overrides.

## Core Training Pattern
All training is launched via `scripts/train.py` + Hydra overrides:

```bash
pixi run python scripts/train.py \
  model=vqvae \
  callbacks=encode \
  datamodule=atlas_mappable \
  project_name=my_project \
  network_name=my_run \
  output_dir=/path/to/results
```

Output structure is:

`<output_dir>/<project_name>/<network_name>/`

and includes checkpoints and `full_config.yaml` for reproducibility/resume.

## Common Runs
Train tokenizer (VQ-VAE):

```bash
pixi run python scripts/train.py \
  model=vqvae callbacks=encode \
  datamodule.num_jets=100000 \
  trainer.max_epochs=30
```

Train classifier on raw features:

```bash
pixi run python scripts/train.py \
  model=feature_classifier callbacks=classify \
  trainer.max_epochs=30
```

Train classifier on learned tokens:

```bash
pixi run python scripts/train.py \
  model=token_classifier callbacks=classify \
  model.tokenizer_ckpt=/path/to/tokenizer/checkpoints/last.ckpt \
  trainer.max_epochs=30
```

## W&B Logging
The default logger is `WandbLogger` in `configs/train.yaml`.

Useful overrides:

```bash
logger.entity=<your_username_or_team>
logger.project=<project_name>
logger.offline=true
```

## Resume Runs
Use `full_resume=true` only if the previous run saved `full_config.yaml` in the expected run directory.
This works best when the original run used an absolute `output_dir` and you reuse the same
`output_dir`, `project_name`, and `network_name`.
Also, ensure that `trainer.max_epochs` exceeds the current epoch.

```bash
pixi run python scripts/train.py \
  full_resume=true \
  ckpt_flag=last.ckpt \
  output_dir=/path/to/results \
  project_name=my_project \
  network_name=my_run
```

If `full_resume` cannot find the prior config, resume directly from a checkpoint path:

```bash
pixi run python scripts/train.py \
  full_resume=false \
  ckpt_path=/abs/path/to/checkpoints/last.ckpt \
  model=vqvae \
  callbacks=encode \
  output_dir=/path/to/results \
  project_name=my_project \
  network_name=my_run
```

## Snakemake Workflows
End-to-end experiment workflows are in:
- `workflow/Snakefile`
- `workflow/preprocessing.smk`

Run locally:

```bash
pixi run snakemake -s workflow/Snakefile --profile profiles/local
```

Run on Slurm:

```bash
pixi run snakemake -s workflow/Snakefile --profile profiles/s3df
```


## Hydra Reference
Main training entry point:
- `scripts/train.py`

Top-level config:
- `configs/train.yaml`

Config groups:
- `configs/model/*.yaml`
- `configs/callbacks/*.yaml`
- `configs/datamodule/*.yaml`
- `configs/hydra/default.yaml`

Hydra override pattern:
- choose config groups: `model=vqvae callbacks=encode datamodule=atlas_mappable`
- override scalar values: `trainer.max_epochs=30`
- use `+key=value` only when adding a new key that does not already exist

Common high-value overrides:

```bash
output_dir=/path/to/results project_name=my_project network_name=my_run
logger.entity=<username_or_team>
trainer.max_epochs=30 trainer.check_val_every_n_epoch=1
datamodule.transforms.preprocess.cst_fn.filename=/path/to/cst_quantiles.joblib
datamodule.transforms.preprocess.jet_fn.filename=/path/to/jet_quantiles.joblib
```

## Reconstruction Callback Settings
For `callbacks=encode`, reconstruction monitoring is configured in `configs/callbacks/encode.yaml`. For `callbacks=classify`, monitoring is configured in `configs/callbacks/classify.yaml`.


## Troubleshooting
### DataLoader workers exit unexpectedly
- start with `datamodule.num_workers=0` to confirm worker issue
- set `datamodule.pin_memory=false` if worker shutdown/restart appears unstable
- set `trainer.reload_dataloaders_every_n_epochs=0` to avoid costly per-epoch dataloader rebuild

### Training becomes slow
- set `trainer.reload_dataloaders_every_n_epochs=0`
- tune `datamodule.num_workers`, `datamodule.batch_size`, `datamodule.pin_memory`
- use `scripts/profile_dataloader.py` to profile dataloading throughput

### W&B run lands under wrong account
- set `logger.entity=<username_or_team>`

### Debugger uses wrong Python interpreter
- use `.vscode/launch.json`
- set `"python": "${workspaceFolder}/.pixi/envs/default/bin/python"`

## Resources
- [Pixi documentation](https://pixi.sh/)
- [Snakemake documentation](https://snakemake.readthedocs.io/en/stable/)
- [Reproducible Machine Learning Workflows for Scientists, Matthew Feickert, 2025](https://carpentries-incubator.github.io/reproducible-ml-workflows/)
- [Good ML project structure](https://github.com/mattcleigh/JetSSL-Lite/tree/master)
