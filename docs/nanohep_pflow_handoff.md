# Handoff: nanoHEP particle flow with heptokens tokenizers (positions-in vs positions-out)

Goal: train **nanoHEP** (decoder-only GPT from `FM-for-HEP/hep-any2any`) for particle flow,
`(track, topo) token IDs -> truthpart token IDs`, on **~89M events**, once per input tokenizer,
and compare **positions-in** vs **positions-out** track/topo tokenizers. This runs on a different
cluster than S3DF; everything needed is listed below.

## 1. Code (all on GitHub)

| repo | branch / commit | what you need from it |
|---|---|---|
| `treasure-slac/heptokens` (remote `origin`; also `Treasure-AmSC/heptokens`) | `feature/cocoa-dataloader` | tokenizers (`src/heptokens`), COCOA datasets incl. `position_features`, `PositionTokenizer`, token export `scripts/build_hep4m_preprocessed.py`, truthpart decode `scripts/pflow_decode_metrics.py`, scalers in `resources/` |
| `FM-for-HEP/HEP4M` | `heptokens-tokenizer-adapter` | `hep4m/models/heptokens_adapter.py`: wraps a heptokens `LitVqVae` as a HEP4M tokenizer (`vae_type: heptokens`); port it into hep-any2any for decoding |
| `FM-for-HEP/hep-any2any` | `main` (checked at `d5aac9b`) | nanoHEP: `hep4m/train_nano_hep.py`, `configs/train/nanohep_pflow.yml`, `hep4m/datasets/tokenized_memmap.py`, `hep4m/models/nano_hep/vocab.py`, inference `configs/infer/nanoHEP-pflow.yml` |

## 2. Artifacts to copy from S3DF (not in git)

Tokenizer checkpoints (~20 MB each) + their `full_config.yaml` (defines features, position
features, scaler, model size). Use `checkpoints/best.ckpt` **after** `SUCCESS.txt` exists.

```
/sdf/data/atlas/u/jkrupa/heptokens/results/cocoa_scan/tracks_e100/<run>/{checkpoints/best.ckpt,full_config.yaml}
/sdf/data/atlas/u/jkrupa/heptokens/results/cocoa_scan/topos_e100/<run>/{checkpoints/best.ckpt,full_config.yaml}
/sdf/data/atlas/u/jkrupa/heptokens/results/cocoa_scan/truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001/{checkpoints/best.ckpt,full_config.yaml}
```

The track/topo runs (all d=8, 100 epochs, cosine LR, Snakemake target `all_design_e100`) were
still training on 2026-10-01 (about epoch 50-70 of 100):

| group | runs (`<run>`) |
|---|---|
| positions-in | `cocoa_{tracks,topos}_cb4096_cd8_nq{2,3,4,6,8}_lr0.001_e100`, `cocoa_{tracks,topos}_cb{64,256,1024}_cd8_nq4_lr0.001_e100` |
| positions-out | `cocoa_{tracks,topos}_posout_cb{256,1024,4096}_cd8_nq4_lr0.001_e100` |

**Comparison to run:** positions-in vs positions-out at the same (K, d=8, Q=4) for
K in {256, 1024, 4096}; start with K=4096. The truthpart tokenizer is the same for every run
(K=128, d=8, Q=3, positions-in).

Scalers (in git, `resources/`): tracks `cocoa_cst_log_standard.joblib` (posout:
`..._posout.joblib`), topos `cocoa_topo_log_standard.joblib` (posout: `..._posout.joblib`),
truthpart `cocoa_truthpart_log_standard.joblib`. Each run's `full_config.yaml` names the one it used.

### Raw data: the 89M events are NOT on S3DF as ROOT

- Our tokenizers need raw COCOA ROOT files (tree `EventTree`; branches `track_*`, `topo_*`,
  `particle_*`, `eventNumber`). On S3DF there are only:
  `.../treasure/4m/data_new/COCOA_samples_singlejet/{train_topup2_20M_cells256 (20.0M), val_100K_cells256 (99,955), test_1M_cells256_isInfFalse}`
  and `.../treasure/4m_sam/Jeff_ROOT_data` (24.1M; 1 unreadable file; its segments overlap our val (1) and test (10) segments).
- `.../treasure/4m_sam/Nilotpal_89Mtokens_numpy_topo_0p5gev` is an 89M **token** store made with the
  original HEP4M tokenizers (track/topo K=256 Q=3 + 3 position tokens); it cannot be re-encoded.
  Its topo has 88,741,573 events vs 88,929,883 for the others.
- **Action:** locate the raw ~89M COCOA single-jet sample (the one behind the Nilotpal store / the
  hep-any2any release, "not in the record" per its README), and make sure it has no events from the
  val/test files you evaluate on (check `eventNumber` / file segments).

## 3. What to build

### 3a. Token store export (heptokens env)

`scripts/build_hep4m_preprocessed.py` already writes the exact layout nanoHEP reads
(`{mod}_data.npy` int16 `[rows, n_codebooks + n_pos_codebooks]`, `{mod}_offsets.npy` int64,
`{mod}_meta.npz`, `{mod}_is_empty.npy`, `event_numbers.npy`), but needs these changes:

1. **Build the dataset from the checkpoint's `full_config.yaml`**, not the hard-coded
   `MODALITY_SPECS`: `datamodule.features`, `datamodule.position_features`, `max_csts`, and the
   scaler. Positions-out tracks use features `[track_pt, track_d0, track_z0]` + positions
   `[track_eta, track_phi]`; positions-out topos `[topo_e, topo_rho, topo_sigma_eta, topo_sigma_phi, topo_ecal_e, topo_hcal_e]` + `[topo_eta, topo_phi]`.
2. **Positions-out: write position tokens.** The dataset returns `batch["positions"]` =
   (eta, cos phi, sin phi) (`heptokens.data.cocoa_base.eta_phi_to_eta_cossin`). Bin them with
   `heptokens.models.pos_tokenizer.PositionTokenizer(ranges=[(-3,3),(-1,1),(-1,1)], n_bins=1024)`
   (exactly the config in `full_config.yaml` under `callbacks.reconstruction_monitor.pos_tokenizer`)
   and append the 3 indices after the content codes; set `n_pos_codebooks=3` in `{mod}_meta.npz`.
   This matches nanoHEP's default position vocabulary (`pos_codebook_size=1024`, `num_q_pos=3`).
   Positions-in: `n_pos_codebooks=0`.
3. **Stream per ROOT file.** The map-style COCOA datasets load everything into RAM (topos at 89M
   would be ~140 GB of float32). Loop over files, append to the output files, then write offsets.
4. **Sort order:** nanoHEP interleaves track/topo elements in on-disk order and generates truth
   particles in on-disk order. Sort tracks by `track_pt`, **topos by `topo_e`**, truthpart by
   `truthpart_pt`, all descending. The current script sorts by raw column 0 (`--pt_idx 0`),
   which is `topo_eta` for positions-in topos (wrong) and `topo_e` for positions-out topos.
5. Same event set and order for all three modalities (the script already asserts identical
   `eventNumber` order); export `train`, `val`, `test` splits.

### 3b. nanoHEP config (hep-any2any)

Copy `configs/train/nanohep_pflow.yml` and change:

- `tokenized_root:` your store; `all_modalities: [topo, track, truthpart]`.
- `vocab_args:` must give **all** four dicts (`Vocab.build` defaults are the HEP4M tokenizers):
  ```yaml
  vocab_args:
      modalities: [topo, track, truthpart]
      codebook_sizes:     {topo: 4096, track: 4096, truthpart: 128}
      num_quantizers:     {topo: 4,    track: 4,    truthpart: 3}
      pos_codebook_sizes: {topo: 1024, track: 1024, truthpart: 1024}
      num_q_pos:          {topo: 3,    track: 3,    truthpart: 0}   # positions-in: topo 0, track 0
  ```
  With `num_q_pos=0` the loader emits no position tokens; leaving the default (3) would insert 3
  dummy zero tokens per element.
- `block_size` (data) and `gpt_config.block_size`: sequence = 2 + 1 + sum_elements(width) + 1 +
  N_truth*3 + 1, width = Q (+3 positions-out). Caps are 15 tracks, 50 topos, 16 truth
  particles: positions-in Q=4 max 313, positions-out Q=4 max 508. Sequences longer than
  `block_size` are silently truncated, so use 512 for both (`gpt_config.block_size` >= that).
- `pflow_eval.modality_dict_path` / Detokenizer: see 3c, or set `pflow_eval.enabled: false`.

### 3c. Decoding truthpart tokens

hep-any2any's `Detokenizer` loads HEP4M VQ-VAEs from a modality dict. Either port
`heptokens_adapter.py` (HEP4M branch above) so a modality dict entry with `vae_type: heptokens`
decodes through heptokens, or decode offline with heptokens'
`scripts/pflow_decode_metrics.py` (token IDs -> pt/eta/phi/E/class via the truthpart `LitVqVae`
+ inverse scaler) followed by `hep4m.performance.pflow_report.run_report` (IQR/median).

### 3d. Inference + metric

`python -m hep4m.eval_hep4m -i configs/infer/nanoHEP-pflow.yml` (model_type `nano_hep`, argmax).
Metric (same as the paper / 4M runs): jet pT = four-vector sum of decoded truth-particle
predictions; resolution = (P75 - P25)/median of pT_reco/pT_truth; bootstrap errors.
**Cardinality:** nanoHEP chooses how many particles to emit (EOS); our HEP4M numbers used the true
multiplicity (`use_truth_cardinality: true`). Report nanoHEP as-is, and if possible also with the
multiplicity forced to the truth, so it can be compared with the HEP4M results below.

## 4. Smoke tests before the full runs

1. Export ~10k events per tokenizer; check `meta.npz` (n_codebooks, n_pos_codebooks), code
   ranges (`< K`, positions `< 1024`), identical `event_numbers` across modalities.
2. `Vocab.build(**vocab_args)`; build 5 samples with `TokenizedMemmapDataset`; check
   `decode_element_with_pos` round-trips the stored codes and no sequence hits `block_size`.
3. Decode the stored truthpart tokens of the val split (no model) and compute IQR/median: this
   is the output-tokenizer floor that every run includes.
4. `configs/train/nanohep_pflow_tiny.yml` with your vocab for ~50 steps, then inference + metric
   on a few hundred events end-to-end.

## 5. Compute

The paper recipe (`nanohep_pflow.yml`): 4 GPUs, bf16, batch 128 x accum 2 x 4 GPUs = 1024
sequences/step, 360k steps (~4 passes over 89M), cosine LR 1.2e-3 -> 1.2e-4. One run per
tokenizer; positions-in/out at K=4096 first (2 runs), then K=256/1024.

## 6. Reference results (HEP4M 4M model, S3DF, for context)

Val 99,840 events, true multiplicity, 20M training events, truthpart cb128/Q=3 output:
- Positions-in d=8/8 Q=3 (K=64..4096): IQR/median 0.148, 0.138, 0.135, 0.135.
- Track d=4 / topo d=8, Q=4: 0.144, 0.137, 0.138, 0.133.
- Continuous inputs (MLP input head, same trunk): 0.1285.
Code: heptokens `workflow/Snakefile` rules `*_pflow_*`; HEP4M branch above.
