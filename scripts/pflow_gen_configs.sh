#!/usr/bin/env bash
# Generate the per-point HEP4M configs for ONE input-tokenizer sweep point (fixed cs).
#
# Extracted verbatim from scripts/pflow_input_sweep.sh so the Snakemake train rule
# (rule train_pflow_point) produces byte-identical configs to the original driver.
# Writes config_m_track.yml, config_m_topo.yml, modality_dict.yml, train.yml under
# $HEP4M/configs/heptokens_pflow/sweep/cs${CS}.
#
# Usage: scripts/pflow_gen_configs.sh <cs> <tag> <nq> <track_cd> <topo_cd> [num_epochs] [early_stop_patience]
# Env overrides: PFLOW_INPUT_REPR=continuous (+ PFLOW_CONT_HIDDEN="256") feeds raw track/topo
# features instead of token IDs; PFLOW_PP_DIR points at a different preprocessed dir.
set -euo pipefail

CS="${1:?usage: pflow_gen_configs.sh <cs> <tag> <nq> <track_cd> <topo_cd> [num_epochs] [early_stop_patience]}"
TAG="${2:?tag required}"
NQ="${3:?nq required}"
TRACK_CD="${4:?track_cd required}"
TOPO_CD="${5:?topo_cd required}"
EPOCHS="${6:-300}"
PATIENCE="${7:-}"

WS=/sdf/data/atlas/u/jkrupa/heptokens
HEP4M="$WS/HEP4M"

TRACK_CKPT="$WS/results/cocoa_scan/tracks/cocoa_tracks_cb${CS}_cd${TRACK_CD}_nq${NQ}_lr0.001/checkpoints/best.ckpt"
TOPO_CKPT="$WS/results/cocoa_scan/topos/cocoa_topos_cb${CS}_cd${TOPO_CD}_nq${NQ}_lr0.001/checkpoints/best.ckpt"
for c in "$TRACK_CKPT" "$TOPO_CKPT"; do
  [ -f "$c" ] || { echo "ERROR: missing checkpoint $c" >&2; exit 1; }
done

# per-quantizer codebook loss weights (length = nq for the input tokenizers).
WTS="[$(printf '1, %.0s' $(seq "$NQ") | sed 's/, $//')]"

PP="${PFLOW_PP_DIR:-/fs/ddn/sdf/group/atlas/d/jkrupa/hep4m_sweep/${TAG}/cs${CS}}"   # preprocessed tokens
CFGDIR="$HEP4M/configs/heptokens_pflow/sweep/${TAG}/cs${CS}"    # generated configs
BASE_ROOT="/fs/ddn/sdf/group/atlas/d/jkrupa/hep4m_pflow_sweep/${TAG}/cs${CS}"    # 4M outputs (ddn group space)
BASECFG="$HEP4M/configs/heptokens_pflow"
mkdir -p "$CFGDIR" "$BASE_ROOT"

INPUT_REPR="${PFLOW_INPUT_REPR:-tokens}"
CONT_EXTRA=""
RUN_NAME="pflow_cs${CS}"
if [ "$INPUT_REPR" != tokens ]; then
  CONT_EXTRA=$'\n'"  input_repr: ${INPUT_REPR}"$'\n'"  cont_hidden: [${PFLOW_CONT_HIDDEN:-256}]"
  RUN_NAME="pflow_cs${CS}_${TAG}"   # eval globs pflow_cs${CS}*, so the prefix must stay
fi

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
  codebook_loss_wts: ${WTS}${CONT_EXTRA}

topo:
  config_path_v: ${BASECFG}/config_v_topo.yml
  config_path_m: ${CFGDIR}/config_m_topo.yml
  checkpoint_path: ${TOPO_CKPT}
  codebook_loss_wts: ${WTS}${CONT_EXTRA}

truthpart:
  config_path_v: ${BASECFG}/config_v_truthpart.yml
  config_path_m: ${BASECFG}/config_m_truthpart.yml
  checkpoint_path: ${WS}/results/cocoa_scan/truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001/checkpoints/best.ckpt
  codebook_loss_wts: [1, 1, 1]
YAML

# train.yml: copy base, retarget preprocessed_dir/base_root_dir/run_name/budget.
sed -e "s#^preprocessed_dir:.*#preprocessed_dir: ${PP}#" \
    -e "s#^base_root_dir :.*#base_root_dir : ${BASE_ROOT}#" \
    -e "s#^run_name:.*#run_name: ${RUN_NAME}#" \
    -e "s#^num_epochs:.*#num_epochs: ${EPOCHS}#" \
    -e "s#^periodic_checkpoint_interval:.*#periodic_checkpoint_interval: 25#" \
    "$BASECFG/train.yml" > "$CFGDIR/train.yml"
[ -n "$PATIENCE" ] && echo "early_stopping_patience: ${PATIENCE}" >> "$CFGDIR/train.yml"

echo "[cs$CS] wrote configs to $CFGDIR (epochs=$EPOCHS, patience=${PATIENCE:-none}, input=$INPUT_REPR)"
