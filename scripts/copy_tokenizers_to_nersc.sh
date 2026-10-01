#!/usr/bin/env bash
# Copy finished heptokens tokenizers (best.ckpt + full_config.yaml + SUCCESS.txt) to NERSC for nanoHEP
# particle flow (docs/nanohep_pflow_handoff.md). Runs on S3DF; one rsync, so one NERSC login.
#
# Usage:
#   scripts/copy_tokenizers_to_nersc.sh --check                       # readiness only, no transfer
#   NERSC_USER=<user> scripts/copy_tokenizers_to_nersc.sh [--dry-run] [--partial]
#     --partial : copy the finished runs now even if others are still training
#   NERSC_HOST (default dtn01.nersc.gov), DEST (default below) can be overridden.
# Then on NERSC:  cd $DEST && sha256sum -c MANIFEST.sha256
set -euo pipefail

SRC=/sdf/data/atlas/u/jkrupa/heptokens/results/cocoa_scan
DEST=${DEST:-/global/cfs/cdirs/m5386/data/COCOA/tokenizers}
HOST=${NERSC_HOST:-dtn01.nersc.gov}

MODE=copy; DRY=""; PARTIAL=0
for a in "$@"; do
  case $a in
    --check) MODE=check ;;
    --dry-run) DRY=--dry-run ;;
    --partial) PARTIAL=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

runs=(truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001)
for mod in tracks topos; do
  for cb in 256 1024 4096; do
    runs+=("${mod}_e100/cocoa_${mod}_cb${cb}_cd8_nq4_lr0.001_e100"
           "${mod}_e100/cocoa_${mod}_posout_cb${cb}_cd8_nq4_lr0.001_e100")
  done
done

cd "$SRC"
ready=(); pending=()
for r in "${runs[@]}"; do
  if [[ -f $r/SUCCESS.txt && -f $r/checkpoints/best.ckpt && -f $r/full_config.yaml ]]; then
    ready+=("$r")
  else
    pending+=("$r")
  fi
done
echo "ready (${#ready[@]}):";   printf '  %s\n' "${ready[@]}"
echo "pending (${#pending[@]}):"; printf '  %s\n' "${pending[@]+"${pending[@]}"}"

[[ $MODE == check ]] && exit 0
if (( ${#pending[@]} > 0 && PARTIAL == 0 )); then
  echo "Not all runs are finished; rerun later or pass --partial to copy the finished ones." >&2
  exit 1
fi
: "${NERSC_USER:?set NERSC_USER}"

# Stage symlinks + checksum manifest so everything goes in a single rsync (one login).
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
for r in "${ready[@]}"; do
  for f in checkpoints/best.ckpt full_config.yaml SUCCESS.txt; do
    mkdir -p "$stage/$r/$(dirname "$f")"
    ln -s "$SRC/$r/$f" "$stage/$r/$f"
  done
done
(cd "$stage" && find . -type l -printf '%P\n' | sort | xargs sha256sum -- > MANIFEST.sha256)

rsync -avL $DRY --chmod=Dg+rx,Fg+r --rsync-path="mkdir -p $DEST && rsync" \
  "$stage/" "$NERSC_USER@$HOST:$DEST/"
echo "Copied ${#ready[@]} runs to $HOST:$DEST. Verify on NERSC: cd $DEST && sha256sum -c MANIFEST.sha256"
