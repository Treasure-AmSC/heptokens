"""Snakemake workflow for exploring preprocessing strategies."""
from pathlib import Path

# Configuration
DATA_PATH = Path("/sdf/scratch/users/s/samklein/data/atlas/mc-flavtag-ttbar-small.h5")
RESOURCES_DIR = DATA_PATH.parent / "resources"
OUTPUT_DIR = Path("/sdf/group/mli/samklein/code/gold_diggers/results/preprocessing_study")
NUM_JETS = 5_000_000

# Define preprocessing configurations
PREPROCESS_CONFIGS = {
    "quantile_500": {
        "cst_mode": "quantile",
        "jet_mode": "quantile",
        "n_quantiles": 500,
    },
    "quantile_100": {
        "cst_mode": "quantile",
        "jet_mode": "quantile",
        "n_quantiles": 100,
    },
    "log_standard": {
        "cst_mode": "log_standard",
        "jet_mode": "log_standard",
        "cst_log_features": "pt",
        "jet_log_features": "pt",
    },
    "log_quantile": {
        "cst_mode": "log_quantile",
        "jet_mode": "log_quantile",
        "cst_log_features": "pt",
        "jet_log_features": "pt",
        "n_quantiles": 500,
    },
}


rule all:
    input:
        # All VQ-VAE models
        expand(
            OUTPUT_DIR / "vqvae/{preprocess}/SUCCESS.txt",
            preprocess=PREPROCESS_CONFIGS.keys(),
        ),
        # All classifiers
        expand(
            OUTPUT_DIR / "classifier/{preprocess}/SUCCESS.txt",
            preprocess=PREPROCESS_CONFIGS.keys(),
        ),


rule create_preprocessor:
    """Create and save preprocessing transformers."""
    output:
        cst_transformer=RESOURCES_DIR / "preprocessing/{preprocess}/cst_quantiles.joblib",
        jet_transformer=RESOURCES_DIR / "preprocessing/{preprocess}/jet_quantiles.joblib",
    params:
        cst_mode=lambda wildcards: PREPROCESS_CONFIGS[wildcards.preprocess]["cst_mode"],
        jet_mode=lambda wildcards: PREPROCESS_CONFIGS[wildcards.preprocess]["jet_mode"],
        n_quantiles=lambda wildcards: PREPROCESS_CONFIGS[wildcards.preprocess].get("n_quantiles", 500),
        cst_log_features=lambda wildcards: (
            f'--cst_log_features {PREPROCESS_CONFIGS[wildcards.preprocess].get("cst_log_features")}'
            if PREPROCESS_CONFIGS[wildcards.preprocess].get("cst_log_features")
            else ''
        ),
        jet_log_features=lambda wildcards: (
            f'--jet_log_features {PREPROCESS_CONFIGS[wildcards.preprocess].get("jet_log_features")}'
            if PREPROCESS_CONFIGS[wildcards.preprocess].get("jet_log_features")
            else ''
        ),
        data_path=DATA_PATH,
        num_jets=NUM_JETS,
        output_dir=lambda wildcards: RESOURCES_DIR / "preprocessing" / wildcards.preprocess,
        
    group:
        "single_preprocessors",
    shell:
        """
        pixi run python scripts/get_preprocessing.py \
            --file_path {params.data_path} \
            --num_jets {params.num_jets} \
            --cst_mode {params.cst_mode} \
            --jet_mode {params.jet_mode} \
            --n_quantiles {params.n_quantiles} \
            --output_dir {params.output_dir} \
            {params.cst_log_features} \
            {params.jet_log_features} \
        """


rule train_vqvae:
    """Train VQ-VAE with specific preprocessing."""
    input:
        cst_transformer=RESOURCES_DIR / "preprocessing/{preprocess}/cst_quantiles.joblib",
        jet_transformer=RESOURCES_DIR / "preprocessing/{preprocess}/jet_quantiles.joblib",
    output:
        success=OUTPUT_DIR / "vqvae/{preprocess}/SUCCESS.txt",
        ckpt=OUTPUT_DIR / "vqvae/{preprocess}/checkpoints/last.ckpt",
    params:
        output_dir=OUTPUT_DIR,
    group:
        "single_preprocessors",
    shell:
        """
        pixi run python scripts/train.py \
            output_dir={params.output_dir} \
            project_name=vqvae \
            network_name={wildcards.preprocess} \
            model=vqvae \
            datamodule.num_jets={NUM_JETS} \
            datamodule.transforms.preprocess.cst_fn.filename={input.cst_transformer} \
            datamodule.transforms.preprocess.jet_fn.filename={input.jet_transformer} \
            trainer.max_epochs=30 \
            trainer.gradient_clip_val=1.0 \
            +trainer.num_sanity_val_steps=0 \
            callbacks=encode
        """


rule train_classifier:
    """Train classifier on VQ-VAE tokens."""
    input:
        vqvae_ckpt=OUTPUT_DIR / "vqvae/{preprocess}/checkpoints/last.ckpt",
        cst_transformer=RESOURCES_DIR / "preprocessing/{preprocess}/cst_quantiles.joblib",
        jet_transformer=RESOURCES_DIR / "preprocessing/{preprocess}/jet_quantiles.joblib",
    output:
        success=OUTPUT_DIR / "classifier/{preprocess}/SUCCESS.txt",
    params:
        output_dir=OUTPUT_DIR,
    group:
        "single_preprocessors",
    shell:
        """
        pixi run python scripts/train.py \
            output_dir={params.output_dir} \
            project_name=classifier \
            network_name={wildcards.preprocess} \
            model=token_classifier \
            model.tokenizer_ckpt={input.vqvae_ckpt} \
            datamodule.num_jets={NUM_JETS} \
            datamodule.transforms.preprocess.cst_fn.filename={input.cst_transformer} \
            datamodule.transforms.preprocess.jet_fn.filename={input.jet_transformer} \
            trainer.max_epochs=30 \
            +trainer.num_sanity_val_steps=0 \
            callbacks=classify
        """
