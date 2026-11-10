# Changelog

## 0.2.0

### Breaking

- **The VQ-VAE bottleneck is now an injected quantizer.** `LitVqVae` takes a required
  `quantizer` (a partial `heptokens.models.quantizers.Quantizer`, called with
  `dim=codebook_dim`) in place of `codebook_size`, `num_quantizers` and `commitment_weight`.
  In Hydra configs this is a `quantizer:` block, with bundled options in the new
  `model/quantizer` config group (`residual_vq`, `latent_bins`), so the bottleneck can be
  swapped on the command line with e.g. `model/quantizer=latent_bins`.
  - Migrate configs by moving `codebook_size` and `num_quantizers` into
    `quantizer: {codebook_size: ..., num_quantizers: ...}` (with `defaults: [quantizer: residual_vq]`)
    and `commitment_weight` to the quantizer's `commitment`.
  - Checkpoints saved by heptokens 0.1 still load if you pass the matching quantizer:
    `LitVqVae.load_from_checkpoint(path, quantizer=partial(ResidualVQQuantizer, codebook_size=..., num_quantizers=...))`.
    The saved weights are unchanged.
- Removed `DiffusionDecoder` and the `diffusion_vqvae` model config. They had not
  worked since `LitVqVae` began passing `data_sample` to its decoder.

### Added

- **Calorimeter-cell tokenizer** (#19, @jeffkrupa): model-side support for the COCOA
  calorimeter-cell RVQ tokenizer.
  - `CellDecoder` energy loss with optional per-pixel weighting (`energy_loss_alpha`), an
    `energy_power` preprocessing exponent, and an L1 option (`energy_loss_type="l1"`).
  - Optional per-token positions: `positions_key` on the coders and a learned positional
    encoding in `Transformer` (`pos_dim`).
  - `"sum"` pooling in `SetToVectorTransformer`, for targets that scale with the number of
    constituents.
  - `scheduler_interval` hyperparameter to step the LR scheduler per `"step"` instead of
    per epoch.
- **Dead-code revival** (#19, @jeffkrupa): `threshold_ema_dead_code > 0` on
  `ResidualVQQuantizer` reinitializes collapsed codebook entries to encoder outputs from
  the current batch, never from padding. The default (`0`) disables it.
- **Latent-binning bottleneck** (#23, @Alonl14): the `LatentBins` quantizer replaces the
  RVQ with per-dimension uniform bins (`model/quantizer=latent_bins`, or the
  `transformer_latent_bins` model config).
  Exports write one quantizer per latent dimension, and the exported codebooks reconstruct
  the binned latent.
- **TorchScript inference modules** (#22): `heptokens-export` also writes scripted
  `encoder.pt`, `decoder.pt`, `codebook_lookup.pt` and `index_encoder.pt` to
  `<output_dir>/modules/`. They can be loaded with plain `torch.jit.load`, or through
  `heptokens.traced_modules.VqVaeKit`. This works with any quantizer, including `LatentBins`
  (#25). Models with custom coders skip this step with a warning.
- **Positions-out building blocks** (#21, @jeffkrupa): keep per-element positions out of
  the VQ-VAE input (`csts`) and tokenize them separately.
  - `position_features` on `StructuredArrayHDF5Dataset` and `StructuredArrayIterableDataset`
    returns the listed fields as a separate tensor (key set by `output_positions_key`,
    default `"positions"`).
  - `PositionTokenizer`: a fixed-bin (non-learned) uniform quantizer with one token per
    position feature, and `decode` back to bin centres. It plugs into `heptokens-export`
    through the `pos_tokenizer` config option.
  - `phi_to_cossin` replaces an angle column with its cosine and sine.
  - `load_transformer(..., features=..., fitted_features=...)` and `select_features` reuse a
    fitted log+standard scaler on a subset of its columns, selected by name.
- Continuous integration: lint and the test suite run on every pull request and on `main`.
- `pixi run test` task.

### Changed

- `inverse_preprocess_batch` accepts `jet_fn=None` to invert constituents only (#21).
- Token export raises `OverflowError` instead of silently wrapping when an index does not
  fit the int16 export format (#23, @Alonl14).
- The `export` output file is named after `datamodule.dataset.file_path` when the datamodule
  uses the dataset-factory config (`data_path` and `data_paths` still work).

### Fixed

- Dead-code revival only samples replacement codes from valid (unmasked) positions, so
  padding never becomes a codebook entry. It also keeps revived codes stable for
  thresholds other than 1 and works with mixed-precision (fp16/bf16) inputs (#19).
- Sum pooling applies its LayerNorm per token before summing, so the pooled vector keeps
  its scaling with token multiplicity (#19).
- The README quickstart datamodule config and the `heptokens-export --help` text now use
  the `dataset:` factory config that `StructuredArrayModule` requires.
- The train-then-export integration test, which had been failing since the switch to the
  dataset-factory config, passes again.

### Contributors

Thanks to @jeffkrupa and @Alonl14 for their contributions to this release.
