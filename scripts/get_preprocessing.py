"""Create preprocessors for the constituents and jets using JetClass."""

import argparse
import logging
from pathlib import Path

from joblib import dump

from heptokens.data.atlas_mappable import SingleFileMapModule
from heptokens.data.transforms import create_preprocessing_transformer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description="Create preprocessing transformers for constituents and jets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data loading args
    parser.add_argument(
        "--file_path",
        type=Path,
        default="/sdf/scratch/users/s/samklein/data/atlas/mc-flavtag-ttbar-small.h5",
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--num_jets",
        type=int,
        default=1_000_000,
        help="Number of jets to load for fitting transformers.",
    )
    parser.add_argument(
        "--num_csts",
        type=int,
        default=64,
        help="Maximum number of constituents per jet.",
    )

    # Constituent preprocessing args
    parser.add_argument(
        "--cst_mode",
        type=str,
        choices=["quantile", "log_quantile", "log_standard", "standard"],
        default="quantile",
        help="Preprocessing mode for constituents. "
        "'quantile': quantile transform only. "
        "'log_quantile': log scale selected features then quantile transform. "
        "'log_standard': log scale selected features then standard scale. "
        "'standard': standard scale only.",
    )
    parser.add_argument(
        "--cst_log_features",
        type=str,
        default="pt",
        help="Comma-separated list of constituent features to log scale (e.g., 'pt' or 'pt,d0'). "
        "Only used with log_* modes.",
    )
    parser.add_argument(
        "--cst_features",
        type=str,
        default="pt,deta,dphi,d0,d0RelativeToBeamspot,d0Uncertainty,d0RelativeToBeamspotUncertainty,z0RelativeToBeamspot,z0RelativeToBeamspotUncertainty,z0SinTheta,z0SinThetaUncertainty,lifetimeSignedD0,lifetimeSignedD0Significance,lifetimeSignedZ0SinTheta,lifetimeSignedZ0SinThetaSignificance,theta,thetaUncertainty,qOverP,qOverPUncertainty,ptfrac",
        help="Comma-separated list of constituent features to use.",
    )

    # Jet preprocessing args
    parser.add_argument(
        "--jet_mode",
        type=str,
        choices=["quantile", "log_quantile", "log_standard", "standard"],
        default="quantile",
        help="Preprocessing mode for jet-level features.",
    )
    parser.add_argument(
        "--jet_log_features",
        type=str,
        default="pt",
        help="Comma-separated list of jet features to log scale. Only used with log_* modes.",
    )
    parser.add_argument(
        "--jet_features",
        type=str,
        default="pt,mass,eta,phi",
        help="Comma-separated list of jet-level features to use.",
    )

    # Transformer hyperparameters
    parser.add_argument(
        "--n_quantiles",
        type=int,
        default=500,
        help="Number of quantiles for QuantileTransformer.",
    )
    parser.add_argument(
        "--log_offset",
        type=float,
        default=1.0,
        help="Offset for log transform: log(x + offset). Default 1.0 gives log(1+x).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save transformers.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Parse feature lists
    cst_features = [f.strip() for f in args.cst_features.split(",")]
    jet_features = [f.strip() for f in args.jet_features.split(",")]

    log.info(f"Constituent features: {cst_features}")
    log.info(f"Jet features: {jet_features}")

    # Load the dataset
    log.info("Loading the dataset")
    data = SingleFileMapModule(
        data_path=args.file_path,
        num_jets=args.num_jets,
        num_csts=args.num_csts,
        n_classes=3,
        cst_features=cst_features,
        jet_features=jet_features,
    )

    # Extract training data
    data_dict = data.train_set[:]
    csts = data_dict["csts"][data_dict["mask"]]
    jets = data_dict["jets"]
    log.info(f"Loaded {len(csts)} constituents and {len(jets)} jets for fitting")

    # Setup save directory
    if args.output_dir is None:
        sv_dir = args.file_path.parent / "resources"
    else:
        sv_dir = Path(args.output_dir)
    sv_dir.mkdir(parents=True, exist_ok=True)

    # Create and fit constituent transformer
    log.info(f"Creating constituent transformer with mode: {args.cst_mode}")
    cst_log_indices = []
    if args.cst_mode.startswith("log"):
        cst_log_features = [f.strip() for f in args.cst_log_features.split(",")]
        cst_log_indices = [cst_features.index(f) for f in cst_log_features if f in cst_features]
        log.info(f"Log-scaling constituent features: {[cst_features[i] for i in cst_log_indices]}")

    cst_transformer = create_preprocessing_transformer(
        mode=args.cst_mode,
        log_feature_indices=cst_log_indices if cst_log_indices else None,
        n_quantiles=args.n_quantiles,
        log_offset=args.log_offset,
        n_features=len(cst_features),
    )
    log.info("Fitting constituent transformer")
    cst_transformer.fit(csts)
    dump(cst_transformer, sv_dir / "cst_quantiles.joblib")

    # Create and fit jet transformer
    log.info(f"Creating jet transformer with mode: {args.jet_mode}")
    jet_log_indices = []
    if args.jet_mode.startswith("log"):
        jet_log_features = [f.strip() for f in args.jet_log_features.split(",")]
        jet_log_indices = [jet_features.index(f) for f in jet_log_features if f in jet_features]
        log.info(f"Log-scaling jet features: {[jet_features[i] for i in jet_log_indices]}")

    jet_transformer = create_preprocessing_transformer(
        mode=args.jet_mode,
        log_feature_indices=jet_log_indices if jet_log_indices else None,
        n_quantiles=args.n_quantiles,
        log_offset=args.log_offset,
        n_features=len(jet_features),
    )
    log.info("Fitting jet transformer")
    jet_transformer.fit(jets)
    dump(jet_transformer, sv_dir / "jet_quantiles.joblib")

    log.info(f"Saved transformers to {sv_dir}")
    log.info("Constituent transformer: cst_quantiles.joblib")
    log.info("Jet transformer: jet_quantiles.joblib")


if __name__ == "__main__":
    main()
