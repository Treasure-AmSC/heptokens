#!/usr/bin/env bash
# EXPORT stage for ONE input-tokenizer sweep point (fixed cs): re-tokenize the
# COCOA track+topo (with the cs tokenizer) + truthpart (fixed) into a per-point
# preprocessed_dir. Extracted verbatim from scripts/pflow_input_sweep.sh's
# EXPORT_WRAP so the Snakemake rule (export_pflow_point) runs identical commands.
#
# Runs in the heptokens env (pixi --manifest-path $WS/pyproject.toml).
# Usage: scripts/pflow_export_point.sh <cs> <tag> <nq> <track_cd> <topo_cd> [train_num_events|all]
set -euo pipefail

CS="${1:?usage: pflow_export_point.sh <cs> <tag> <nq> <track_cd> <topo_cd> [train_num_events|all]}"
TAG="${2:?tag required}"
NQ="${3:?nq required}"
TRACK_CD="${4:?track_cd required}"
TOPO_CD="${5:?topo_cd required}"
TRAIN_N="${6:-all}"

WS=/sdf/data/atlas/u/jkrupa/heptokens
MANIFEST="$WS/pyproject.toml"
DATA_DIR=/fs/ddn/sdf/group/atlas/d/jkrupa/treasure/4m/data_new/COCOA_samples_singlejet

TRACK_CKPT="$WS/results/cocoa_scan/tracks/cocoa_tracks_cb${CS}_cd${TRACK_CD}_nq${NQ}_lr0.001/checkpoints/best.ckpt"
TOPO_CKPT="$WS/results/cocoa_scan/topos/cocoa_topos_cb${CS}_cd${TOPO_CD}_nq${NQ}_lr0.001/checkpoints/best.ckpt"
for c in "$TRACK_CKPT" "$TOPO_CKPT"; do
  [ -f "$c" ] || { echo "ERROR: missing checkpoint $c" >&2; exit 1; }
done

PP=/fs/ddn/sdf/group/atlas/d/jkrupa/hep4m_sweep/${TAG}/cs${CS}
mkdir -p "$PP"

pixi run --manifest-path "$MANIFEST" python "$WS/scripts/build_hep4m_preprocessed.py" \
  --data_dir "$DATA_DIR" --split_dir val_100K_cells256 --split_name val \
  --output_dir "$PP" --num_workers 8 --track_ckpt "$TRACK_CKPT" --topo_ckpt "$TOPO_CKPT"

# "all" = full train split (~20M events), matching the original nq3 sweep export.
NEV_ARGS=()
[ "$TRAIN_N" != "all" ] && NEV_ARGS=(--num_events "$TRAIN_N")

pixi run --manifest-path "$MANIFEST" python "$WS/scripts/build_hep4m_preprocessed.py" \
  --data_dir "$DATA_DIR" --split_dir train_topup2_20M_cells256 --split_name train \
  --output_dir "$PP" --num_workers 8 ${NEV_ARGS[@]+"${NEV_ARGS[@]}"} --track_ckpt "$TRACK_CKPT" --topo_ckpt "$TOPO_CKPT"

touch "$PP/EXPORT_SUCCESS.txt"
echo "[cs$CS] export done -> $PP"
