"""Export COCOA tokens (from a heptokens VQ-VAE) into HEP4M's intermediate ROOT.

Writes a ROOT file with an ``event_tree`` whose schema matches what
``hep4m.pre_processing.convert_to_npy_tokenized`` expects, so the standard HEP4M
converter (+ ``find_empty_event``) can turn it into the preprocessed token
memmaps consumed by 4M training:

    {out_modality}_token_0 .. _token_{nq-1}   jagged int  (content codebooks)
    n{out_modality}                            int         (constituents per event)
    {out_modality}_pt                          jagged float(sort key; scaled pt)
    event_number                               int

Tokenization uses the heptokens datamodule (i.e. the SAME preprocessing the
tokenizer was trained with), so tokens are consistent with the checkpoint.

Example:
    pixi run python scripts/export_cocoa_tokens_hep4m.py \
        --modality tracks \
        --ckpt results/cocoa_scan/tracks/cocoa_tracks_cb256_cd8_nq3_lr0.001/checkpoints/best.ckpt \
        --cst_fn resources/cocoa_cst_log_standard.joblib \
        --split_dir val_100K_cells256 \
        --output /tmp/hep4m_tokens/track_tokens_val.root
"""

import argparse
import logging
from functools import partial
from pathlib import Path

import awkward as ak
import joblib
import numpy as np
import torch
import uproot

from heptokens.data.cocoa_base import root_files_from_dir
from heptokens.data.cocoa_topos import COCOATopoDataset
from heptokens.data.cocoa_tracks import COCOATrackDataset
from heptokens.data.cocoa_truthpart import COCOATruthPartDataset
from heptokens.data.collation import collate_and_transform, preprocess_batch
from heptokens.models.vq_vae import LitVqVae

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# heptokens datamodule name -> (Dataset class, HEP4M modality name, default max_csts)
MODALITIES = {
    "tracks": (COCOATrackDataset, "track", 15),
    "topos": (COCOATopoDataset, "topo", 50),
    "truthpart": (COCOATruthPartDataset, "truthpart", 16),
}


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--modality", required=True, choices=list(MODALITIES))
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--cst_fn", required=True, type=Path, help="Preprocessing scaler joblib.")
    p.add_argument(
        "--data_dir",
        type=str,
        default="/fs/ddn/sdf/group/atlas/d/jkrupa/treasure/4m/data_new/COCOA_samples_singlejet",
    )
    p.add_argument("--split_dir", required=True, help="e.g. val_100K_cells256 or train_topup2_20M_cells256")
    p.add_argument("--output", required=True, type=Path, help="Output ROOT path.")
    p.add_argument("--max_csts", type=int, default=None, help="Override pad/truncate length.")
    p.add_argument("--num_events", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pt_idx", type=int, default=0, help="Feature index used as the sort key.")
    return p.parse_args()


def build_dataset(args):
    ds_cls, out_modality, default_max = MODALITIES[args.modality]
    max_csts = args.max_csts if args.max_csts is not None else default_max
    files = root_files_from_dir(Path(args.data_dir) / args.split_dir)
    ds = ds_cls(files, max_csts=max_csts, num_events=args.num_events)
    return ds, out_modality


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds, out_modality = build_dataset(args)
    scaler = joblib.load(args.cst_fn)
    transforms = {"preprocess": partial(preprocess_batch, cst_fn=scaler)}
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=partial(collate_and_transform, transforms=transforms),
    )

    model = LitVqVae.load_from_checkpoint(args.ckpt, map_location=device)
    model.to(device).eval()
    nq = int(model.hparams.num_quantizers)
    log.info(f"Loaded tokenizer ({out_modality}): num_quantizers={nq}, "
             f"codebook_size={model.codebook_size}, num_classes={getattr(model, 'num_classes', 0)}")

    # Per-event accumulators
    token_cols = [[] for _ in range(nq)]  # each: list of per-event int arrays
    pt_col = []
    n_col = []
    ev_col = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            indices = model.encode_indices(batch)  # [B, n, nq], -1 on padding
            mask = batch["mask"]
            csts = batch["csts"]
            evnums = batch["eventNumber"]
            B = mask.shape[0]
            for i in range(B):
                m = mask[i]
                k = int(m.sum().item())
                idx_i = indices[i][m].to(torch.int64).cpu().numpy()  # [k, nq]
                pt_i = csts[i][m][:, args.pt_idx].float().cpu().numpy()  # [k]
                # Sort by pt descending to match HEP4M's sort_by_var convention.
                order = np.argsort(-pt_i, kind="stable")
                idx_i = idx_i[order]
                pt_i = pt_i[order]
                for q in range(nq):
                    token_cols[q].append(idx_i[:, q].astype(np.int32))
                pt_col.append(pt_i.astype(np.float32))
                n_col.append(k)
                ev_col.append(int(evnums[i].item()))

    n_events = len(n_col)
    log.info(f"Tokenized {n_events:,} events; writing ROOT -> {args.output}")

    tree_dict = {f"{out_modality}_token_{q}": ak.Array(token_cols[q]) for q in range(nq)}
    tree_dict[f"n{out_modality}"] = np.asarray(n_col, dtype=np.int32)
    tree_dict[f"{out_modality}_pt"] = ak.Array(pt_col)
    tree_dict["event_number"] = np.asarray(ev_col, dtype=np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(args.output) as f:
        f["event_tree"] = tree_dict

    log.info("Done.")


if __name__ == "__main__":
    main()
