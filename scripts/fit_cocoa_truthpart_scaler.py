"""Fit the COCOA truth-particle preprocessing scaler.

Builds the tokenized feature vector (continuous features + one-hot class) from a
few COCOA train ROOT files and fits a log_standard transformer: log-scale pt and
e, then standard-scale all features. Saves a joblib compatible with
``heptokens.data.collation.preprocess_batch``.

Usage:
    pixi run python scripts/fit_cocoa_truthpart_scaler.py \
        --data_dir /fs/ddn/sdf/group/atlas/d/jkrupa/treasure/4m/data_new/COCOA_samples_singlejet \
        --output resources/cocoa_truthpart_log_standard.joblib
"""

import argparse
import logging
from pathlib import Path

from joblib import dump

from heptokens.data.cocoa_base import root_files_from_dir
from heptokens.data.cocoa_truthpart import (
    TRUTHPART_FEATURES,
    COCOATruthPartDataset,
)
from heptokens.data.transforms import create_preprocessing_transformer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_args():
    p = argparse.ArgumentParser(
        description="Fit COCOA truth-particle log_standard scaler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="/fs/ddn/sdf/group/atlas/d/jkrupa/treasure/4m/data_new/COCOA_samples_singlejet",
    )
    p.add_argument("--train_dir", type=str, default="train_topup2_20M_cells256")
    p.add_argument("--max_csts", type=int, default=16)
    p.add_argument(
        "--num_events",
        type=int,
        default=1_000_000,
        help="Number of events used to fit the transformer.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("resources/cocoa_truthpart_log_standard.joblib"),
    )
    return p.parse_args()


def main():
    args = get_args()

    files = root_files_from_dir(Path(args.data_dir) / args.train_dir)
    ds = COCOATruthPartDataset(files, max_csts=args.max_csts, num_events=args.num_events)

    # Valid constituents only: [n_valid, n_features]
    csts = ds.csts[ds.mask]
    log.info(f"Fitting on {len(csts):,} valid truth particles, {csts.shape[-1]} features")

    # Log-scale pt (index 0) and e (index 3); standard-scale everything.
    pt_idx = TRUTHPART_FEATURES.index("particle_pt")
    e_idx = TRUTHPART_FEATURES.index("particle_e")
    transformer = create_preprocessing_transformer(
        mode="log_standard",
        log_feature_indices=[pt_idx, e_idx],
        n_features=csts.shape[-1],
    )
    transformer.fit(csts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(transformer, args.output)
    log.info(f"Saved transformer to {args.output}")


if __name__ == "__main__":
    main()
