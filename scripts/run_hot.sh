#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_hot.sh <k> [seed] [project_dir]
# Defaults: seed=42, project_dir=/workspace/MoE_Sieve_Experiments

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <k> [seed] [project_dir]"
  exit 1
fi

K="$1"
SEED="${2:-42}"
PROJECT_DIR="${3:-/workspace/MoE_Sieve_Experiments}"

cd "$PROJECT_DIR"

python -m scripts.run \
  --task gsm8k \
  --mode hot \
  --model_tag olmoe \
  --seed "$SEED" \
  --k "$K" \
  --hotmap_dir "$PROJECT_DIR/outputs/hotmaps" \
  --hotmap_template "telemetry_{task}_train_n7473_seed123_global_hotmap_counts_k{k}.json" \
  --cleanup_after_push
