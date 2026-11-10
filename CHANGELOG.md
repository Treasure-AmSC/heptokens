# Changelog

## 0.2.0

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
- **Dead-code revival** (#19, @jeffkrupa): `threshold_ema_dead_code > 0` on `LitVqVae` swaps
  in `RevivingResidualVQ`, which reinitializes collapsed codebook entries to encoder outputs
  from the current batch. The default (`0`) keeps the plain `ResidualVQ`.
- **Latent-binning bottleneck** (#23, @Alonl14): `LitVqVae(latent_bins=...)` replaces the
  RVQ with per-dimension uniform bins, with a `transformer_latent_bins` model config.
  Exports write one quantizer per latent dimension, and the exported codebooks reconstruct
  the binned latent.
- **TorchScript inference modules** (#22): `heptokens-export` also writes scripted
  `encoder.pt`, `decoder.pt`, `codebook_lookup.pt` and `index_encoder.pt` to
  `<output_dir>/modules/`. They can be loaded with plain `torch.jit.load`, or through
  `heptokens.traced_modules.VqVaeKit`. Models with custom coders or the `LatentBins`
  bottleneck skip this step with a warning.
- Continuous integration: lint and the test suite run on every pull request and on `main`.
- `pixi run test` task.

### Changed

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
