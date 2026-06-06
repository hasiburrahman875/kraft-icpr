#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-1}"
TARGET="${1:-uavswarm}"

case "$TARGET" in
  uavswarm)
    CONFIG="$ROOT/config_uavswarm.yaml"
    ;;
  w2c|uavswarm-w2c)
    CONFIG="$ROOT/config_w2c.yaml"
    ;;
  muav)
    CONFIG="$ROOT/config_muav.yaml"
    ;;
  muav-eval|muav-fast)
    CONFIG="$ROOT/config_muav_eval_tracks.yaml"
    ;;
  all)
    "$0" uavswarm
    "$0" w2c
    "$0" muav
    exit 0
    ;;
  *)
    echo "Usage: $0 [uavswarm|w2c|muav|muav-eval|all]" >&2
    exit 2
    ;;
esac

"$PYTHON" "$ROOT/reproduce.py" \
  --python "$PYTHON" \
  --config "$CONFIG" \
  --workers "$WORKERS"
