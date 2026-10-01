#!/usr/bin/env bash
# Input-tokenizer sweep for the 4M pflow (truthpart) reconstruction.
#
# Fixes the truthpart tokenizer (cb128/cd8/nq3) and cd=8, nq=3; varies the
# INPUT (track+topo) codebook size cs. For one cs value this driver:
#   1. generates per-point HEP4M configs,
#   2. submits an EXPORT job (heptokens env) -> track/topo re-tokenized with the
#      cs tokenizer + truthpart (fixed) into a per-point preprocessed_dir,
#   3. submits a 4M TRAIN job (HEP4M env) that waits (afterok) on the export.
#
# Usage: scripts/pflow_input_sweep.sh <cs> [num_epochs] [train_num_events]
set -euo pipefail

CS="${1:?usage: pflow_input_sweep.sh <cs> [num_epochs] [train_num_events]}"
EPOCHS="${2:-300}"
TRAIN_N="${3:-2000000}"
CD=8
NQ=3

WS=/sdf/data/atlas/u/jkrupa/heptokens
HEP4M="$WS/HEP4M"
MANIFEST="$WS/pyproject.toml"
DATA_DIR=/fs/ddn/sdf/group/atlas/d/jkrupa/treasure/4m/data_new/COCOA_samples_singlejet
LOGDIR="$WS/logs"
ACCT="${SWEEP_ACCT:-atlas:compef@ampere}"
PART="${SWEEP_PART:-ampere}"

TRACK_CKPT="$WS/results/cocoa_scan/tracks/cocoa_tracks_cb${CS}_cd${CD}_nq${NQ}_lr0.001/checkpoints/best.ckpt"
TOPO_CKPT="$WS/results/cocoa_scan/topos/cocoa_topos_cb${CS}_cd${CD}_nq${NQ}_lr0.001/checkpoints/best.ckpt"
for c in "$TRACK_CKPT" "$TOPO_CKPT"; do
  [ -f "$c" ] || { echo "ERROR: missing checkpoint $c" >&2; exit 1; }
done

# Per-point locations.
PP=/fs/ddn/sdf/group/atlas/d/jkrupa/hep4m_sweep/cs${CS}          # preprocessed tokens
CFGDIR="$HEP4M/configs/heptokens_pflow/sweep/cs${CS}"           # generated configs
BASE_ROOT="$WS/results/hep4m_pflow_sweep/cs${CS}"               # 4M outputs
BASECFG="$HEP4M/configs/heptokens_pflow"
mkdir -p "$PP" "$CFGDIR" "$BASE_ROOT" "$LOGDIR"

# Info fraction (x-axis): nq*log2(cs)/(32*cd).
INFO=$(python3 -c "import math;print(f'{$NQ*math.log2($CS)/(32*$CD):.4f}')")
echo "cs=$CS  cd=$CD nq=$NQ  info_fraction=$INFO  epochs=$EPOCHS  train_n=$TRAIN_N"

# ---- generate per-point configs --------------------------------------------
cat > "$CFGDIR/config_m_track.yml" <<YAML
# sweep point cs=${CS}: heptokens track tokenizer model config.
vae_type: heptokens
heptokens_ckpt: ${TRACK_CKPT}

quantizer:
  codebook_size: ${CS}
  num_quantizers: ${NQ}
YAML

cat > "$CFGDIR/config_m_topo.yml" <<YAML
# sweep point cs=${CS}: heptokens topo tokenizer model config.
vae_type: heptokens
heptokens_ckpt: ${TOPO_CKPT}

quantizer:
  codebook_size: ${CS}
  num_quantizers: ${NQ}
YAML

cat > "$CFGDIR/modality_dict.yml" <<YAML
# sweep point cs=${CS}. config_v reused from base (data spec is cs-independent);
# config_m + checkpoint_path point at the cs-specific track/topo tokenizers.
track:
  config_path_v: ${BASECFG}/config_v_track.yml
  config_path_m: ${CFGDIR}/config_m_track.yml
  checkpoint_path: ${TRACK_CKPT}
  codebook_loss_wts: [1, 1, 1]

topo:
  config_path_v: ${BASECFG}/config_v_topo.yml
  config_path_m: ${CFGDIR}/config_m_topo.yml
  checkpoint_path: ${TOPO_CKPT}
  codebook_loss_wts: [1, 1, 1]

truthpart:
  config_path_v: ${BASECFG}/config_v_truthpart.yml
  config_path_m: ${BASECFG}/config_m_truthpart.yml
  checkpoint_path: ${WS}/results/cocoa_scan/truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001/checkpoints/best.ckpt
  codebook_loss_wts: [1, 1, 1]
YAML

# train.yml: copy base, retarget preprocessed_dir/base_root_dir/run_name/budget.
sed -e "s#^preprocessed_dir:.*#preprocessed_dir: ${PP}#" \
    -e "s#^base_root_dir :.*#base_root_dir : ${BASE_ROOT}#" \
    -e "s#^run_name:.*#run_name: pflow_cs${CS}#" \
    -e "s#^num_epochs:.*#num_epochs: ${EPOCHS}#" \
    -e "s#^periodic_checkpoint_interval:.*#periodic_checkpoint_interval: 25#" \
    "$BASECFG/train.yml" > "$CFGDIR/train.yml"

# ---- submit EXPORT job (heptokens env) -------------------------------------
EXPORT_WRAP="set -e; \
pixi run --manifest-path ${MANIFEST} python ${WS}/scripts/build_hep4m_preprocessed.py \
  --data_dir ${DATA_DIR} --split_dir val_100K_cells256 --split_name val \
  --output_dir ${PP} --num_workers 8 --track_ckpt ${TRACK_CKPT} --topo_ckpt ${TOPO_CKPT}; \
pixi run --manifest-path ${MANIFEST} python ${WS}/scripts/build_hep4m_preprocessed.py \
  --data_dir ${DATA_DIR} --split_dir train_topup2_20M_cells256 --split_name train \
  --output_dir ${PP} --num_workers 8 --num_events ${TRAIN_N} --track_ckpt ${TRACK_CKPT} --topo_ckpt ${TOPO_CKPT}"

EXPORT_JID=$(sbatch --parsable -A "$ACCT" -p "$PART" -t "${SWEEP_EXPORT_TIME:-240}" --mem=100G --gres=gpu:1 --cpus-per-task=8 \
  -D "$WS" -J "sweep_export_cs${CS}" -o "$LOGDIR/sweep_export_cs${CS}_%j.log" \
  --wrap="$EXPORT_WRAP")
echo "export job: $EXPORT_JID"

# ---- submit TRAIN job (HEP4M env), waits on export -------------------------
TRAIN_WRAP="export HEP4M_LOGGER=wandb; export WANDB_MODE=${SWEEP_WANDB_MODE:-online}; \
pixi run python -m hep4m.train_hep4m \
  -md ${CFGDIR}/modality_dict.yml -cm configs/fourm/pflow/model.yml -ct ${CFGDIR}/train.yml"

TRAIN_JID=$(sbatch --parsable --dependency=afterok:${EXPORT_JID} \
  -A "$ACCT" -p "$PART" -t "${SWEEP_TRAIN_TIME:-720}" --mem=100G --gres=gpu:1 --cpus-per-task=8 \
  -D "$HEP4M" -J "sweep_train_cs${CS}" -o "$LOGDIR/sweep_train_cs${CS}_%j.log" \
  --wrap="$TRAIN_WRAP")
echo "train job:  $TRAIN_JID  (afterok:$EXPORT_JID)"

# record point metadata for later collection.
echo "${CS},${INFO},${PP},${BASE_ROOT},${EXPORT_JID},${TRAIN_JID}" >> "$WS/results/hep4m_pflow_sweep/points.csv"
