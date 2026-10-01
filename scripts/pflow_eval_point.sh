#!/usr/bin/env bash
# 3-stage particle-flow eval for ONE input-tokenizer sweep point (fixed cs).
#
# Stage 1 (HEP4M env): fast-forward inference of the trained 4M -> truthpart
#   truth tokens + predicted token logits (prediction_tokens.root).
# Stage 2 (heptokens env): decode truth & argmax-predicted tokens into physics
#   (pt/eta/phi/class) via the truthpart tokenizer + scaler; also emit token
#   accuracy (prediction_physics.root + token_metrics.json).
# Stage 3 (HEP4M env): pflow_report.run_report -> metrics.json + plots.
#
# The best-val 4M checkpoint (lowest val_total_loss) is auto-discovered.
# Usage: scripts/pflow_eval_point.sh <cs> <tag> [reduce_ds]   (reduce_ds=-1 => all val)
set -euo pipefail

CS="${1:?usage: pflow_eval_point.sh <cs> <tag> [reduce_ds]}"
TAG="${2:?tag required}"
REDUCE="${3:--1}"

WS=/sdf/data/atlas/u/jkrupa/heptokens
HEP4M="$WS/HEP4M"
MANIFEST="$WS/pyproject.toml"
SWEEP_TOK="${PFLOW_PP_DIR:-/fs/ddn/sdf/group/atlas/d/jkrupa/hep4m_sweep/${TAG}/cs${CS}}"
BASE_ROOT="/fs/ddn/sdf/group/atlas/d/jkrupa/hep4m_pflow_sweep/${TAG}/cs${CS}"
CFGDIR="$HEP4M/configs/heptokens_pflow/sweep/${TAG}/cs${CS}"     # per-point modality_dict.yml
EVALDIR="$BASE_ROOT/eval"
MODEL_CFG="$HEP4M/configs/fourm/pflow/model.yml"
TP_CKPT="$WS/results/cocoa_scan/truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001/checkpoints/best.ckpt"
TP_SCALER="$WS/resources/cocoa_truthpart_log_standard.joblib"
DIRFLAG="cs${CS}"
mkdir -p "$EVALDIR"

# Best-val checkpoint = lowest val_total_loss encoded in the filename.
BEST_CKPT=$(ls "$BASE_ROOT"/hep4m/pflow_cs${CS}*/checkpoints/epoch=*-val_total_loss=*.ckpt 2>/dev/null \
  | sed -E 's/.*val_total_loss=([0-9.]+)\.ckpt/\1\t&/' | sort -n | head -1 | cut -f2-)
[ -n "$BEST_CKPT" ] || { echo "ERROR: no best-val checkpoint under $BASE_ROOT/hep4m/pflow_cs${CS}*/checkpoints" >&2; exit 1; }
echo "[cs$CS] best-val ckpt: $BEST_CKPT"

# eval_hep4m derives its output dir by replacing 'config_m.yml' -> 'inference_ff'
# in config_path_m, so the model arch config MUST be named config_m.yml and live
# in EVALDIR to route outputs here.
cp "$MODEL_CFG" "$EVALDIR/config_m.yml"

cat > "$EVALDIR/inference_ff.yml" <<YAML
init:
  detector: COCOA
  fastforward: true
  device: cuda
  gpu: 0
  precision: highest
  chunk_size: 128
  batch_size: 128
  num_workers: 2
  model:
    config_path_m: ${EVALDIR}/config_m.yml
    checkpoint_path: ${BEST_CKPT}
    modality_dict_path: ${CFGDIR}/modality_dict.yml
    sampling_type: inference

items:
  - info: 'pflow-cs${CS}'
    sampling_dict:
      type: inference
      fixed_input_output_modalities:
        input: [track, topo]
        output: [truthpart]
    start_idx: 0
    reduce_ds: ${REDUCE}
    use_preprocessed_data: true
    use_truth_cardinality: true
    store_tokens: true
    store_pos_tokens: false
    card_topk_dict: {truthpart: 1}
    card_temperature_dict: {truthpart: 1.0}
    top_k_token_dict: {truthpart: 1}
    top_k_gpos_token_dict: {truthpart: 1}
    temperature_token_dict: {truthpart: 1.0}
    temperature_gpos_token_dict: {truthpart: 1.0}
    filepath_dict:
      track: ${SWEEP_TOK}/val/track_data.npy
      topo: ${SWEEP_TOK}/val/topo_data.npy
      truthpart: ${SWEEP_TOK}/val/truthpart_data.npy
    dir_flag: ${DIRFLAG}
    suffix: tokens
YAML

PRED_TOKENS="$EVALDIR/inference_ff/${DIRFLAG}/prediction_tokens.root"
PRED_PHYS="$EVALDIR/prediction_physics.root"
REPORT_DIR="$EVALDIR/report"

echo "[cs$CS] === Stage 1: FF inference (HEP4M env) ==="
cd "$HEP4M"
pixi run python -m hep4m.eval_hep4m -i "$EVALDIR/inference_ff.yml"
[ -f "$PRED_TOKENS" ] || { echo "ERROR: stage1 did not produce $PRED_TOKENS" >&2; exit 1; }

echo "[cs$CS] === Stage 2: decode tokens -> physics (heptokens env) ==="
pixi run --manifest-path "$MANIFEST" python "$WS/scripts/pflow_decode_metrics.py" \
  --pred_root "$PRED_TOKENS" --ckpt "$TP_CKPT" --scaler "$TP_SCALER" \
  --out_root "$PRED_PHYS" --modality truthpart --vocab 128 \
  --token_metrics_json "$EVALDIR/token_metrics.json"

echo "[cs$CS] === Stage 3: pflow_report -> metrics.json (HEP4M env) ==="
pixi run python -c "from hep4m.performance.pflow_report import run_report; run_report('$PRED_PHYS', '$REPORT_DIR', output_modality='truthpart')"
cp "$REPORT_DIR/metrics.json" "$EVALDIR/metrics.json"
echo "[cs$CS] DONE -> $EVALDIR/metrics.json + token_metrics.json"
