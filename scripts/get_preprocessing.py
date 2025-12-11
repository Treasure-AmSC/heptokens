"""Create preprocessors for the constituents and jets using JetClass."""

import argparse
import logging
from pathlib import Path

from joblib import dump
from sklearn.preprocessing import QuantileTransformer

from gdig.data.atlas_mappable import SingleFileMapModule

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Preprocess constituents and jets.")
    parser.add_argument(
        "--n_quantiles",
        type=int,
        default=500,
        help="Number of quantiles for QuantileTransformer.",
    )
    parser.add_argument(
        "--num_jets",
        type=int,
        default=1000_000,
        help="Number of jets to load.",
    )
    parser.add_argument(
        "--num_csts",
        type=int,
        default=64,
        help="Number of constituents.",
    )
    parser.add_argument(
        "--file_path",
        type=Path,
        default="/sdf/scratch/users/s/samklein/data/atlas/mc-flavtag-ttbar-small.h5",
        help="Path to the dataset file.",
    )
    return parser.parse_args()


# Create the datasets
def main():
    args = get_args()

    log.info("Loading the dataset")
    # TODO: this should probably be hydra instantiated?
    # TODO: or preprocessing should be part of the datamodule?
    data = SingleFileMapModule(
        data_path=args.file_path,
        num_jets=args.num_jets,
        num_csts=args.num_csts,
        n_classes=3,
        cst_features=["pt", "deta", "dphi", "d0"],
        jet_features=["pt", "mass"],
    )
    # Extract the training dataset
    data_dict = data.train_set[:]
    # Extract the parent directory for saving resources
    # TODO: should probably be more careful with naming conventions
    sv_dir = args.file_path.parent / "resources"
    sv_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading pure arrays of the constituents")
    csts = data_dict["csts"][data_dict["mask"]]
    jets = data_dict["jets"]
    log.info(f"Loaded {len(csts)} constituents and {len(jets)} jets")

    # log.info("Ignoring neutral impact parameters for the neutral constituents")
    # jc_id = data_dict["csts_id"][data_dict["mask"]]
    # is_neut = (jc_id == 0) | (jc_id == 2)
    # csts[is_neut, 3:] = np.nan

    log.info("Fitting the quantile transformer for the constituents")
    cst_qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=args.n_quantiles,
        subsample=None,
    )
    cst_qt.fit(csts)
    dump(cst_qt, sv_dir / "cst_quantiles.joblib")

    log.info("Fitting the quantile transformer for the jets")
    jet_qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=args.n_quantiles,
        subsample=None,
    )
    jet_qt.fit(jets)
    dump(jet_qt, sv_dir / "jet_quantiles.joblib")


if __name__ == "__main__":
    main()
