#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/pod_setup.sh [project_dir]
# Default project_dir: /workspace/HELLoRA_Project

PROJECT_DIR="${1:-/workspace/HELLoRA_Project}"

export HF_HOME="/workspace/.cache/huggingface"
export WANDB_DIR="/workspace/.cache/wandb"
export WANDB_CACHE_DIR="/workspace/.cache/wandb"
mkdir -p "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR"

mkdir -p /workspace

if [[ ! -d "$PROJECT_DIR" ]]; then
  git clone https://github.com/amanzoni1/MoE_HELLoRA.git "$PROJECT_DIR"
else
  echo "Using existing project dir: $PROJECT_DIR"
fi

cd "$PROJECT_DIR"

python -m pip install -U pip
python -m pip install --no-cache-dir -U \
  "transformers==4.57.6" \
  "accelerate==1.12.0" \
  "peft==0.18.1" \
  datasets wandb huggingface_hub

mkdir -p "$PROJECT_DIR/outputs/hotmaps"

echo "Setup complete in $PROJECT_DIR"
