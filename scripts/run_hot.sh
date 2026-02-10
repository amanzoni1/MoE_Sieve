#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_hot.sh <k> [seed] [project_dir]
# Defaults: seed=99, project_dir=/workspace/HELLoRA_Project

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <k> [seed] [project_dir]"
  exit 1
fi

K="$1"
SEED="${2:-99}"
PROJECT_DIR="${3:-/workspace/HELLoRA_Project}"

cd "$PROJECT_DIR"

python -m scripts.run \
  --task gsm8k \
  --mode hot \
  --seed "$SEED" \
  --k "$K" \
  --hotmap_dir "$PROJECT_DIR/outputs/hotmaps" \
  --hotmap_template "telemetry_{task}_train_n7473_seed123_global_hotmap_counts_k{k}.json" \
  --push_to_hub \
  --hub_repo_template "AManzoni/hellora-olmoe-{task}_hot_k{k}_s{seed}" \
  --hub_private \
  --cleanup_after_push
