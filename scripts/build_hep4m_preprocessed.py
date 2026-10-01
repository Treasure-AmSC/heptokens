"""Build a HEP4M preprocessed-token directory from heptokens tokenizers.

For a given split (e.g. val or train), tokenizes each modality with its heptokens
checkpoint (using the heptokens datamodule, i.e. the SAME preprocessing the
tokenizer was trained with) and writes the on-disk format HEP4M's
``TokenDatasetNumpy`` consumes -- directly, without the intermediate ROOT or
HEP4M's (non-import-safe) converter:

    {out}/{split}/{mod}_data.npy      raw int16  [sum_valid_tokens, nq]   (C-order)
    {out}/{split}/{mod}_offsets.npy   raw int64  [n_events + 1]
    {out}/{split}/{mod}_meta.npz      n_events, n_codebooks=nq, n_pos_codebooks=0
    {out}/{split}/{mod}_is_empty.npy  .npy int64 (indices of 0-token events)
    {out}/{split}/event_numbers.npy   raw int64  [n_events]   (shared)

Tokens are sorted per event by pt (descending) to match HEP4M's ``sort_by_var``.
All modalities are checked to share an identical event ordering (HEP4M's
preprocessed loader trusts row-alignment across modalities).

Example:
    pixi run python scripts/build_hep4m_preprocessed.py \
        --split_dir val_100K_cells256 --split_name val \
        --output_dir /sdf/scratch/.../hep4m_tokens --num_events 5000
"""

import argparse
import logging
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import torch

from heptokens.data.cocoa_base import root_files_from_dir
from heptokens.data.cocoa_topos import COCOATopoDataset
from heptokens.data.cocoa_tracks import COCOATrackDataset
from heptokens.data.cocoa_truthpart import COCOATruthPartDataset
from heptokens.data.collation import collate_and_transform, preprocess_batch
from heptokens.models.vq_vae import LitVqVae

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

WS = Path("/sdf/data/atlas/u/jkrupa/heptokens")

# HEP4M modality -> tokenizer checkpoint, scaler, dataset class, pad length.
# Checkpoints match configs/heptokens_pflow/modality_dict.yml.
MODALITY_SPECS = {
    "track": dict(
        ds=COCOATrackDataset, max_csts=15,
        ckpt=WS / "results/cocoa_scan/tracks/cocoa_tracks_cb256_cd8_nq3_lr0.001/checkpoints/best.ckpt",
        scaler=WS / "resources/cocoa_cst_log_standard.joblib",
    ),
    "topo": dict(
        ds=COCOATopoDataset, max_csts=50,
        ckpt=WS / "results/cocoa_scan/topos/cocoa_topos_cb256_cd8_nq3_lr0.001/checkpoints/best.ckpt",
        scaler=WS / "resources/cocoa_topo_log_standard.joblib",
    ),
    "truthpart": dict(
        ds=COCOATruthPartDataset, max_csts=16,
        ckpt=WS / "results/cocoa_scan/truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001/checkpoints/best.ckpt",
        scaler=WS / "resources/cocoa_truthpart_log_standard.joblib",
    ),
}


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data_dir", type=str,
        default="/fs/ddn/sdf/group/atlas/d/jkrupa/treasure/4m/data_new/COCOA_samples_singlejet",
    )
    p.add_argument("--split_dir", required=True, help="e.g. val_100K_cells256")
    p.add_argument("--split_name", required=True, choices=["train", "val", "test"])
    p.add_argument("--output_dir", required=True, type=Path, help="preprocessed_dir root")
    p.add_argument("--modalities", nargs="+", default=list(MODALITY_SPECS))
    p.add_argument("--num_events", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pt_idx", type=int, default=0)
    # Optional per-modality checkpoint overrides (for input-tokenizer sweeps).
    p.add_argument("--track_ckpt", type=str, default=None)
    p.add_argument("--topo_ckpt", type=str, default=None)
    p.add_argument("--truthpart_ckpt", type=str, default=None)
    # Add {mod}_cont.npy (scaled float32 features, same rows as {mod}_data.npy) to an existing export.
    p.add_argument("--cont_only", action="store_true")
    args = p.parse_args()
    for mod in ("track", "topo", "truthpart"):
        override = getattr(args, f"{mod}_ckpt")
        if override:
            MODALITY_SPECS[mod]["ckpt"] = Path(override)
    return args


@torch.no_grad()
def tokenize_modality(spec, files, args, device, with_tokens=True, cont_path=None):
    """Return (data int16 [Ntok, nq] | None, offsets int64 [E+1], event_numbers int64 [E], nq, n_feat).

    If cont_path is given, the scaled csts (float32 [Ntok, n_feat]) are streamed there in
    the same row order as the tokens.
    """
    ds = spec["ds"](files, max_csts=spec["max_csts"], num_events=args.num_events)
    scaler = joblib.load(spec["scaler"])
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
        collate_fn=partial(collate_and_transform, transforms={"preprocess": partial(preprocess_batch, cst_fn=scaler)}),
    )
    model, nq = None, 0
    if with_tokens:
        model = LitVqVae.load_from_checkpoint(spec["ckpt"], map_location=device).to(device).eval()
        nq = int(model.hparams.num_quantizers)
    cont_f = open(cont_path, "wb") if cont_path is not None else None
    n_feat = None

    rows, counts, evnums = [], [], []
    for batch in loader:
        gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        # Batch-level transfer to CPU/numpy ONCE (per-event .item()/.cpu() would
        # force a GPU sync per event and is ~1000x slower).
        if model is not None:
            idx_np = model.encode_indices(gpu).to(torch.int16).cpu().numpy()  # [B, n, nq], -1 on padding
        mask_np = gpu["mask"].cpu().numpy()
        pt_np = gpu["csts"][:, :, args.pt_idx].float().cpu().numpy()
        if cont_f is not None:
            csts_np = gpu["csts"].float().cpu().numpy()
            n_feat = csts_np.shape[-1]
            cont_rows = []
        ev = batch["eventNumber"]
        ev_np = ev.numpy() if isinstance(ev, torch.Tensor) else np.asarray(ev)
        for i in range(mask_np.shape[0]):
            m = mask_np[i]
            k = int(m.sum())
            if k:
                order = np.argsort(-pt_np[i][m], kind="stable")
                if model is not None:
                    rows.append(idx_np[i][m][order])  # [k, nq]
                if cont_f is not None:
                    cont_rows.append(csts_np[i][m][order])  # [k, n_feat]
            counts.append(k)
            evnums.append(int(ev_np[i]))
        if cont_f is not None and cont_rows:
            cont_f.write(np.concatenate(cont_rows, axis=0).astype(np.float32).tobytes())

    if cont_f is not None:
        cont_f.close()
    data = None
    if with_tokens:
        data = np.concatenate(rows, axis=0) if rows else np.zeros((0, nq), dtype=np.int16)
    counts = np.asarray(counts, dtype=np.int64)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return data, offsets, np.asarray(evnums, dtype=np.int64), nq, n_feat


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = root_files_from_dir(Path(args.data_dir) / args.split_dir)
    out = args.output_dir / args.split_name

    if args.cont_only:
        # Row alignment against the existing token export is what the 4M dataset relies on.
        ref_ev = np.fromfile(out / "event_numbers.npy", dtype=np.int64)
        for mod in args.modalities:
            cont_path = out / f"{mod}_cont.npy"
            log.info(f"[{args.split_name}] exporting continuous {mod} -> {cont_path}")
            _, offsets, ev, _, n_feat = tokenize_modality(
                MODALITY_SPECS[mod], files, args, device, with_tokens=False, cont_path=cont_path)
            ref_off = np.fromfile(out / f"{mod}_offsets.npy", dtype=np.int64)
            if not (np.array_equal(ev, ref_ev) and np.array_equal(offsets, ref_off)):
                cont_path.unlink()
                raise RuntimeError(f"{mod}: continuous rows do not align with the token export in {out}")
            np.savez(out / f"{mod}_cont_meta.npz", n_rows=int(offsets[-1]), n_feat=n_feat)
            log.info(f"  {mod}: {len(ev):,} events, {int(offsets[-1]):,} rows x {n_feat} feats (aligned)")
        return

    out.mkdir(parents=True, exist_ok=True)
    ref_ev = None
    for mod in args.modalities:
        spec = MODALITY_SPECS[mod]
        log.info(f"[{args.split_name}] tokenizing {mod} ...")
        data, offsets, ev, nq, _ = tokenize_modality(spec, files, args, device)

        data.tofile(out / f"{mod}_data.npy")
        offsets.tofile(out / f"{mod}_offsets.npy")
        np.savez(out / f"{mod}_meta.npz", n_events=len(ev), n_codebooks=nq, n_pos_codebooks=0)
        lengths = offsets[1:] - offsets[:-1]
        np.save(out / f"{mod}_is_empty.npy", np.where(lengths == 0)[0])
        log.info(f"  {mod}: {len(ev):,} events, {len(data):,} tokens, {int((lengths==0).sum())} empty")

        if ref_ev is None:
            ref_ev = ev
            ev.tofile(out / "event_numbers.npy")
        elif not np.array_equal(ev, ref_ev):
            raise RuntimeError(f"Event-number misalignment: {mod} differs from reference ordering!")

    log.info(f"Wrote preprocessed tokens -> {out} (alignment verified across {len(args.modalities)} modalities)")


if __name__ == "__main__":
    main()
