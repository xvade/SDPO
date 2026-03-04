#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_apptainer_cmd.sh <image.sif> <sdpo_path> <logs_path> <checkpoints_path> <command...>

Example:
  scripts/run_apptainer_cmd.sh \
    /path/to/sdpo-gh200.sif \
    /path/to/SDPO \
    /path/to/logs \
    /path/to/checkpoints \
    "bash run_local_sdpo.sh"
EOF
}

if [[ $# -lt 5 ]]; then
#   usage
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Error: apptainer is not installed or not in PATH." >&2
  exit 1
fi

SIF_PATH="$1"
SDPO_PATH="$2"
LOGS_PATH="$3"
CKPT_PATH="$4"
CERT_PATH="/etc/ssl/certs/ca-bundle.crt"
shift 4
CMD="$*"

if [[ ! -f "$SIF_PATH" ]]; then
  echo "Error: SIF file not found: $SIF_PATH" >&2
  exit 1
fi

if [[ ! -d "$SDPO_PATH" ]]; then
  echo "Error: SDPO path is not a directory: $SDPO_PATH" >&2
  exit 1
fi

mkdir -p "$LOGS_PATH" "$CKPT_PATH"

SIF_ABS="$(realpath "$SIF_PATH")"
SDPO_ABS="$(realpath "$SDPO_PATH")"
LOGS_ABS="$(realpath "$LOGS_PATH")"
CKPT_ABS="$(realpath "$CKPT_PATH")"
CERT_ABS="$(realpath "$CERT_PATH")"


SDPO_CONTAINER_PATH="/users/sgvtc/SDPO"

echo "Running command in container:"
echo "  image: $SIF_ABS"
echo "  sdpo : $SDPO_ABS"
echo "  logs : $LOGS_ABS"
echo "  ckpt : $CKPT_ABS"
echo "  cert : $CERT_ABS"
echo "  cmd  : $CMD"

apptainer exec --nv \
  --bind "${SDPO_ABS}:/users/sgvtc/SDPO" \
  --bind "${LOGS_ABS}:/users/sgvtc/logs" \
  --bind "${CKPT_ABS}:/users/sgvtc/SDPO/checkpoints" \
  --bind "${CERT_ABS}:/etc/ssl/certs/ca-bundle.crt" \
  "$SIF_ABS" \
  bash -lc "cd $SDPO_CONTAINER_PATH && export PYTHONPATH=$SDPO_CONTAINER_PATH:\$PYTHONPATH && ${CMD}"