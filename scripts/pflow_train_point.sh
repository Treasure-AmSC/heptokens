#!/usr/bin/env bash
# TRAIN stage for ONE input-tokenizer sweep point (fixed cs): generate the per-point
# configs, then train the 4M pflow. Extracted verbatim from scripts/pflow_input_sweep.sh's
# config generation + TRAIN_WRAP so the Snakemake rule (train_pflow_point) runs identical
# commands. Requires the export stage (scripts/pflow_export_point.sh) to have populated
# the per-point preprocessed_dir first.
#
# Runs in the HEP4M env (cd $HEP4M; pixi run python -m hep4m.train_hep4m).
# Usage: scripts/pflow_train_point.sh <cs> <tag> <nq> <track_cd> <topo_cd> [num_epochs] [early_stop_patience]
set -euo pipefail

CS="${1:?usage: pflow_train_point.sh <cs> <tag> <nq> <track_cd> <topo_cd> [num_epochs] [early_stop_patience]}"
TAG="${2:?tag required}"
NQ="${3:?nq required}"
TRACK_CD="${4:?track_cd required}"
TOPO_CD="${5:?topo_cd required}"
EPOCHS="${6:-300}"
PATIENCE="${7:-}"

WS=/sdf/data/atlas/u/jkrupa/heptokens
HEP4M="$WS/HEP4M"
CFGDIR="$HEP4M/configs/heptokens_pflow/sweep/${TAG}/cs${CS}"
BASE_ROOT="/fs/ddn/sdf/group/atlas/d/jkrupa/hep4m_pflow_sweep/${TAG}/cs${CS}"

# Generate per-point configs (byte-identical to the original driver).
bash "$WS/scripts/pflow_gen_configs.sh" "$CS" "$TAG" "$NQ" "$TRACK_CD" "$TOPO_CD" "$EPOCHS" "$PATIENCE"

cd "$HEP4M"
export HEP4M_LOGGER=wandb
export WANDB_MODE=${SWEEP_WANDB_MODE:-online}
pixi run python -m hep4m.train_hep4m \
  -md "${CFGDIR}/modality_dict.yml" -cm configs/fourm/pflow/model.yml -ct "${CFGDIR}/train.yml"

touch "$BASE_ROOT/TRAIN_SUCCESS.txt"
echo "[cs$CS] train done -> $BASE_ROOT/TRAIN_SUCCESS.txt"
