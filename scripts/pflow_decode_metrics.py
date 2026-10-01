"""Stage 2 of the heptokens->HEP4M PF eval: decode predicted/truth truthpart
tokens into physics and write a `pflow_report`-compatible ROOT.

Input: the token ROOT from HEP4M's fast-forward inference
(`hep4m.eval_hep4m` with store_tokens=true), which contains, per event:
    truthpart_truth_tokens        flat int   [card*nq]
    truthpart_reco_token_logits   flat float [card*nq*vocab]
    truthpart_truth_card          int
    event_number

This script decodes both the truth tokens and the argmax'd predicted tokens with
the heptokens truthpart tokenizer (its OWN decoder + fitted log_standard scaler --
the correct inverse for these tokens), then writes per-particle pt/eta/phi/class
for `{mod}_truth` and `{mod}_reco`, which `hep4m.performance.pflow_report` reads.

Runs in the heptokens env (vq 0.2.2, omegaconf present).

Example:
    pixi run python scripts/pflow_decode_metrics.py \
        --pred_root results/hep4m_pflow_infer/inference_ff/smoke/prediction_tokens.root \
        --ckpt results/cocoa_scan/truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001/checkpoints/best.ckpt \
        --scaler resources/cocoa_truthpart_log_standard.joblib \
        --out_root results/hep4m_pflow_infer/inference_ff/smoke/prediction_physics.root
"""

import argparse
from pathlib import Path

import awkward as ak
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import uproot

from heptokens.models.vq_vae import LitVqVae

# truthpart continuous feature order (matches heptokens.data.cocoa_truthpart)
PT, ETA, PHI, E = 0, 1, 2, 3


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_root", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--scaler", required=True, type=Path)
    p.add_argument("--out_root", required=True, type=Path)
    p.add_argument("--modality", default="truthpart")
    p.add_argument("--vocab", type=int, default=128)
    p.add_argument("--token_metrics_json", type=Path, default=None,
                   help="If set, write per-quantizer/overall token accuracy here.")
    return p.parse_args()


def indices_to_zq(model, idx, mask):
    """Sum per-quantizer codebook vectors (heptokens' old RVQ has no helper)."""
    idx = idx.clamp(min=0)
    z = None
    for q, layer in enumerate(model.vector_quantization.layers):
        code = F.embedding(idx[..., q], layer.embed.t())
        z = code if z is None else z + code
    return z * mask.unsqueeze(-1).to(z.dtype if z is not None else torch.float32)


@torch.no_grad()
def decode_events(model, scaler, tokens_per_event, nq, n_cont, num_classes):
    """tokens_per_event: list of int arrays [card_e, nq]. Returns per-event
    lists of pt, eta, phi, class (physics units)."""
    E = len(tokens_per_event)
    max_card = max((t.shape[0] for t in tokens_per_event), default=1)
    max_card = max(max_card, 1)
    tokens = np.zeros((E, max_card, nq), dtype=np.int64)
    mask = np.zeros((E, max_card), dtype=bool)
    for i, t in enumerate(tokens_per_event):
        k = t.shape[0]
        if k:
            tokens[i, :k] = t
            mask[i, :k] = True

    tok = torch.from_numpy(tokens)
    msk = torch.from_numpy(mask)
    z_q = indices_to_zq(model, tok, msk)
    full = model.decoder(z_q, {"mask": msk})  # [E, max_card, n_cont+num_classes]
    cont = full[..., :n_cont]
    cls = full[..., n_cont:].argmax(dim=-1) if num_classes > 0 else None

    # inverse the log_standard scaler on valid continuous rows
    cont_valid = cont[msk].cpu().numpy()  # [n_valid, n_cont]
    phys = scaler.inverse_transform(cont_valid)  # pt, eta, phi, e

    out = {"pt": [], "eta": [], "phi": [], "class": []}
    cursor = 0
    cls_np = cls.cpu().numpy() if cls is not None else None
    for i, t in enumerate(tokens_per_event):
        k = t.shape[0]
        rows = phys[cursor:cursor + k]
        out["pt"].append(rows[:, PT].astype(np.float32))
        out["eta"].append(rows[:, ETA].astype(np.float32))
        out["phi"].append(rows[:, PHI].astype(np.float32))
        if cls_np is not None:
            out["class"].append(cls_np[i, :k].astype(np.int32))
        else:
            out["class"].append(np.zeros(k, dtype=np.int32))
        cursor += k
    return out


def main():
    args = get_args()
    model = LitVqVae.load_from_checkpoint(args.ckpt, map_location="cpu").eval()
    scaler = joblib.load(args.scaler)
    nq = int(model.hparams.num_quantizers)
    n_cont = int(model.cont_dim)
    num_classes = int(getattr(model, "num_classes", 0))
    mod = args.modality

    tree = uproot.open(f"{args.pred_root}:event_tree")
    d = tree.arrays(
        [f"{mod}_truth_card", f"{mod}_truth_tokens", f"{mod}_reco_token_logits", "event_number"],
        library="ak",
    )
    card = ak.to_numpy(d[f"{mod}_truth_card"]).astype(int)
    evnum = ak.to_numpy(d["event_number"]).astype(np.int64)
    n_events = len(card)

    truth_tokens, reco_tokens = [], []
    for e in range(n_events):
        c = int(card[e])
        tt = ak.to_numpy(d[f"{mod}_truth_tokens"][e]).astype(np.int64).reshape(c, nq)
        rl = ak.to_numpy(d[f"{mod}_reco_token_logits"][e]).astype(np.float32).reshape(c, nq, args.vocab)
        truth_tokens.append(tt)
        reco_tokens.append(rl.argmax(-1).astype(np.int64))

    if args.token_metrics_json is not None:
        # token-index accuracy: reco argmax vs truth token, per quantizer and overall.
        tt_all = np.concatenate(truth_tokens, axis=0) if truth_tokens else np.zeros((0, nq), np.int64)
        rt_all = np.concatenate(reco_tokens, axis=0) if reco_tokens else np.zeros((0, nq), np.int64)
        match = (tt_all == rt_all)
        # event-level bootstrap for the overall accuracy uncertainty (accounts for
        # intra-event particle correlations).
        ev_sums = np.array([(t == r).sum() for t, r in zip(truth_tokens, reco_tokens)], dtype=np.float64)
        ev_counts = np.array([t.size for t in truth_tokens], dtype=np.float64)

        def _boot_acc_err(n_boot=1000, seed=0):
            E = len(ev_counts)
            if E == 0:
                return float("nan")
            rng = np.random.default_rng(seed)
            vals = []
            for _ in range(n_boot):
                idx = rng.integers(0, E, E)
                denom = ev_counts[idx].sum()
                vals.append(ev_sums[idx].sum() / denom if denom else np.nan)
            return float(np.nanstd(vals))

        tok = {
            "n_particles": int(tt_all.shape[0]),
            "token_acc_overall": float(match.mean()) if match.size else float("nan"),
            "token_acc_overall_err": _boot_acc_err(),
            "token_acc_exact": float(match.all(axis=1).mean()) if match.size else float("nan"),
            "token_acc_per_q": [float(match[:, q].mean()) if match.size else float("nan") for q in range(nq)],
        }
        args.token_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        args.token_metrics_json.write_text(_json.dumps(tok, indent=2))
        print(f"Wrote token metrics -> {args.token_metrics_json}: {tok}")


    truth = decode_events(model, scaler, truth_tokens, nq, n_cont, num_classes)
    reco = decode_events(model, scaler, reco_tokens, nq, n_cont, num_classes)

    out = {"event_number": evnum}
    for grp, data in [(f"{mod}_truth", truth), (f"{mod}_reco", reco)]:
        for field in ("pt", "eta", "phi", "class"):
            out[f"{grp}_{field}"] = ak.Array(data[field])

    args.out_root.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(args.out_root) as f:
        f["event_tree"] = out
    print(f"Wrote decoded physics ROOT -> {args.out_root} ({n_events} events)")


if __name__ == "__main__":
    main()
