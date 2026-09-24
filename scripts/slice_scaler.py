"""Column-slice a fitted CompositeTransformer (log + StandardScaler) to a subset of features.

Each column is transformed independently (per-column log, per-column mean/scale), so the
sliced scaler applies exactly the same transform to the kept features as the full one.
Used for positions-out runs, where the position columns are removed from csts.

usage: python scripts/slice_scaler.py INPUT.joblib OUTPUT.joblib --drop 1 2
"""

import argparse
import copy

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from heptokens.data.transforms import CompositeTransformer, FeatureWiseTransformer


def slice_scaler(scaler: CompositeTransformer, keep: list[int]) -> CompositeTransformer:
    if not isinstance(scaler, CompositeTransformer) or not isinstance(
        scaler.final_transformer, StandardScaler
    ):
        raise TypeError(f"only CompositeTransformer(..., StandardScaler) supported, got {scaler}")
    new_index = {old: new for new, old in enumerate(keep)}

    configs = []
    for cfg in scaler.log_transformer.feature_configs:
        idx = [cfg["indices"]] if isinstance(cfg["indices"], int) else list(cfg["indices"])
        kept = [new_index[i] for i in idx if i in new_index]
        if kept:
            configs.append({"indices": kept, "transformers": copy.deepcopy(cfg["transformers"])})
    log_t = FeatureWiseTransformer(configs)
    log_t.n_features_in_ = len(keep)

    final = copy.deepcopy(scaler.final_transformer)
    for attr in ("mean_", "scale_", "var_"):
        if getattr(final, attr, None) is not None:
            setattr(final, attr, np.asarray(getattr(final, attr))[keep])
    if np.ndim(final.n_samples_seen_) > 0:
        final.n_samples_seen_ = np.asarray(final.n_samples_seen_)[keep]
    final.n_features_in_ = len(keep)

    out = CompositeTransformer(log_t, final)
    out.n_features_in_ = len(keep)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--drop", type=int, nargs="+", required=True, help="column indices to remove")
    args = p.parse_args()

    scaler = joblib.load(args.input)
    keep = [i for i in range(scaler.n_features_in_) if i not in set(args.drop)]
    joblib.dump(slice_scaler(scaler, keep), args.output)
    print(f"{args.input} ({scaler.n_features_in_} features) -> {args.output} (kept {keep})")


if __name__ == "__main__":
    main()
