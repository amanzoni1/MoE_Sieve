#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TASK="gsm8k"
MODEL_TAG="olmoe"
SELECTION_MODE="hot"
SEEDS_CSV="42"
KS_CSV="4,8,12,14,16,18,20,22,24,32,48"
DATA_SEED=""
TELEMETRY_PATH=""
HOTMAP_DIR="${REPO_ROOT}/outputs/hotmaps"
HOTMAP_MODE="counts"
HOTMAP_SOURCE="auto"
COVERAGE_PCT="60"
COVERAGE_MIN_K=""
COVERAGE_MAX_K=""
RANDOM_SELECTOR_OFFSET=0
RUN_FULL=1
PIP_INSTALL=1
PUSH_TO_HUB=1
CLEANUP_AFTER_PUSH=1
USE_WANDB=1
HUB_REPO_PREFIX=""
SPIDER_SCHEMA_AUTO=1
SPIDER_TS_REPO="/workspace/test-suite-sql-eval"
EXTRA_RUN_ARGS=()

usage() {
  cat <<EOF
Usage: bash scripts/pod/train_pod.sh [options] [-- extra args for scripts.run]

Options:
  --task <name>                 Dataset key (default: ${TASK})
  --model-tag <tag>             Model tag for run names (default: ${MODEL_TAG})
  --selection-mode <mode>       hot|dyn|random (default: ${SELECTION_MODE})
  --seeds <csv>                 Training seeds as comma-list (default: ${SEEDS_CSV})
  --ks <csv>                    k values as comma-list (default: ${KS_CSV}; ignored for mode=dyn)
  --data-seed <int>             Dataset shuffle seed (default: model seed)
  --telemetry <path>            Explicit telemetry .pt path
  --hotmap-source <mode>        auto|hotmap|telemetry (default: ${HOTMAP_SOURCE})
  --hotmap-dir <path>           Hotmap output dir (default: outputs/hotmaps)
  --hotmap-mode <counts|mass>   Hotmap metric (default: ${HOTMAP_MODE})
  --coverage-pct <float>        Coverage percent for mode=dyn (default: ${COVERAGE_PCT})
  --cov <float>                 Alias for --coverage-pct
  --coverage-min-k <int>        Optional min per-layer k clamp in mode=dyn
  --coverage-max-k <int>        Optional max per-layer k clamp in mode=dyn
  --random-selector-offset <n>  random_seed = model_seed + n (default: ${RANDOM_SELECTOR_OFFSET})
  --no-full                     Skip full-LoRA run
  --no-pip-install              Skip dependency install
  --no-push                     Pass --no_push_to_hub
  --no-cleanup                  Do not pass --cleanup_after_push
  --no-wandb                    Pass --no_wandb
  --hub-repo-prefix <prefix>    Optional repo prefix per run (e.g. myexp_)
  --no-spider-schema-auto       Disable automatic Spider schema setup
  --spider-ts-repo <path>       Spider test-suite repo path (default: ${SPIDER_TS_REPO})
  -h, --help                    Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --selection-mode) SELECTION_MODE="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --ks) KS_CSV="$2"; shift 2 ;;
    --data-seed) DATA_SEED="$2"; shift 2 ;;
    --telemetry) TELEMETRY_PATH="$2"; shift 2 ;;
    --hotmap-source) HOTMAP_SOURCE="$2"; shift 2 ;;
    --hotmap-dir) HOTMAP_DIR="$2"; shift 2 ;;
    --hotmap-mode) HOTMAP_MODE="$2"; shift 2 ;;
    --coverage-pct) COVERAGE_PCT="$2"; shift 2 ;;
    --cov) COVERAGE_PCT="$2"; shift 2 ;;
    --coverage-min-k) COVERAGE_MIN_K="$2"; shift 2 ;;
    --coverage-max-k) COVERAGE_MAX_K="$2"; shift 2 ;;
    --random-selector-offset) RANDOM_SELECTOR_OFFSET="$2"; shift 2 ;;
    --no-full) RUN_FULL=0; shift ;;
    --no-pip-install) PIP_INSTALL=0; shift ;;
    --no-push) PUSH_TO_HUB=0; shift ;;
    --no-cleanup) CLEANUP_AFTER_PUSH=0; shift ;;
    --no-wandb) USE_WANDB=0; shift ;;
    --hub-repo-prefix) HUB_REPO_PREFIX="$2"; shift 2 ;;
    --no-spider-schema-auto) SPIDER_SCHEMA_AUTO=0; shift ;;
    --spider-ts-repo) SPIDER_TS_REPO="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_RUN_ARGS=("$@"); break ;;
    *) die "Unknown argument: $1" ;;
  esac
done

cd "$REPO_ROOT"
export MOE_SIEVE_ROOT_DIR="${MOE_SIEVE_ROOT_DIR:-${REPO_ROOT}/outputs}"
mkdir -p "$MOE_SIEVE_ROOT_DIR"

if [[ "$PIP_INSTALL" -eq 1 ]]; then
  log "Installing training dependencies..."
  install_train_deps
fi

setup_cache_dirs

if [[ "$TASK" == "spider" ]] && [[ "$SPIDER_SCHEMA_AUTO" -eq 1 ]]; then
  log "Preparing Spider schema assets for training..."
  setup_spider_assets "$SPIDER_TS_REPO" 0 0
fi

if ! hf_login_if_token; then
  if [[ "$PUSH_TO_HUB" -eq 1 ]]; then
    die "HF push enabled but HF_TOKEN is missing. Set HF_TOKEN or pass --no-push."
  fi
  log "HF_TOKEN not set; using unauthenticated HF access."
fi

if [[ "$USE_WANDB" -eq 1 ]]; then
  if ! wandb_login_if_key; then
    log "WANDB_API_KEY missing; disabling W&B for this run."
    USE_WANDB=0
  fi
fi

case "$SELECTION_MODE" in
  hot|dyn|random) ;;
  *) die "Invalid --selection-mode '${SELECTION_MODE}' (expected hot|dyn|random)" ;;
esac

if [[ "$SELECTION_MODE" == "hot" || "$SELECTION_MODE" == "dyn" ]]; then
  case "$HOTMAP_SOURCE" in
    auto|hotmap|telemetry) ;;
    *) die "Invalid --hotmap-source '${HOTMAP_SOURCE}' (expected auto|hotmap|telemetry)" ;;
  esac

  if [[ -z "$TELEMETRY_PATH" ]] && [[ "$HOTMAP_SOURCE" != "hotmap" ]]; then
    TELEMETRY_PATH="$(ls -t "${REPO_ROOT}/outputs/telemetry/${TASK}/telemetry_${TASK}_"*"_global.pt" 2>/dev/null | head -n 1 || true)"
  fi
  if [[ "$HOTMAP_SOURCE" == "telemetry" ]] || [[ "$HOTMAP_SOURCE" == "auto" ]]; then
    if [[ -n "$TELEMETRY_PATH" ]] && [[ ! -f "$TELEMETRY_PATH" ]]; then
      die "Telemetry file not found: $TELEMETRY_PATH"
    fi
  fi
fi

if [[ "$SELECTION_MODE" == "dyn" ]]; then
  if [[ -z "${COVERAGE_PCT}" ]]; then
    die "mode=dyn requires --coverage-pct (or --cov)"
  fi
fi

mkdir -p "$HOTMAP_DIR"

split_csv "$KS_CSV" KS
split_csv "$SEEDS_CSV" SEEDS

for seed in "${SEEDS[@]}"; do
  if [[ "$SELECTION_MODE" == "dyn" ]]; then
    cov_tag="$(printf '%s' "$COVERAGE_PCT" | sed 's/[[:space:]]//g; s/\.0*$//; s/\.$//; s/\./p/g')"
    hotmap_cov_path="$(ls -t "${HOTMAP_DIR}"/*"${TASK}"*"_hotmap"*"_cov"*".json" 2>/dev/null | head -n 1 || true)"
    cmd=(
      python3 -m scripts.run
      --task "$TASK"
      --mode dyn
      --model_tag "$MODEL_TAG"
      --seed "$seed"
      -cov "$COVERAGE_PCT"
      --hotmap_mode "$HOTMAP_MODE"
    )
    if [[ -n "$DATA_SEED" ]]; then
      cmd+=(--data_seed "$DATA_SEED")
    fi
    if [[ -n "$COVERAGE_MIN_K" ]]; then
      cmd+=(--coverage_min_k "$COVERAGE_MIN_K")
    fi
    if [[ -n "$COVERAGE_MAX_K" ]]; then
      cmd+=(--coverage_max_k "$COVERAGE_MAX_K")
    fi

    if [[ "$HOTMAP_SOURCE" == "hotmap" ]]; then
      [[ -n "$hotmap_cov_path" ]] || die "No coverage hotmap found for task='${TASK}' in '${HOTMAP_DIR}'"
      cmd+=(--hotmap "$hotmap_cov_path")
    elif [[ "$HOTMAP_SOURCE" == "telemetry" ]]; then
      [[ -n "$TELEMETRY_PATH" ]] || die "Telemetry mode selected but no telemetry file was provided/found."
      cmd+=(--telemetry "$TELEMETRY_PATH")
    else
      if [[ -n "$hotmap_cov_path" ]]; then
        cmd+=(--hotmap "$hotmap_cov_path")
      elif [[ -n "$TELEMETRY_PATH" ]]; then
        cmd+=(--telemetry "$TELEMETRY_PATH")
      else
        die "Auto mode: no prebuilt coverage hotmap in '${HOTMAP_DIR}' and no telemetry file found."
      fi
    fi

    if [[ "$PUSH_TO_HUB" -eq 0 ]]; then
      cmd+=(--no_push_to_hub)
    elif [[ -n "$HUB_REPO_PREFIX" ]]; then
      cmd+=(--hub_repo "${HUB_REPO_PREFIX}${MODEL_TAG}_${TASK}_s${seed}_cov${cov_tag}")
    fi
    if [[ "$CLEANUP_AFTER_PUSH" -eq 1 ]]; then
      cmd+=(--cleanup_after_push)
    fi
    if [[ "$USE_WANDB" -eq 0 ]]; then
      cmd+=(--no_wandb)
    fi
    cmd+=("${EXTRA_RUN_ARGS[@]}")

    log "Launching DYN: seed=${seed}, coverage=${COVERAGE_PCT}%, source=${HOTMAP_SOURCE}"
    "${cmd[@]}"
    continue
  fi

  for k in "${KS[@]}"; do
    hotmap_path="$(ls -t "${HOTMAP_DIR}"/*"${TASK}"*"_hotmap"*"_k${k}.json" 2>/dev/null | head -n 1 || true)"
    cmd=(
      python3 -m scripts.run
      --task "$TASK"
      --mode "$SELECTION_MODE"
      --model_tag "$MODEL_TAG"
      --seed "$seed"
      --k "$k"
    )
    if [[ -n "$DATA_SEED" ]]; then
      cmd+=(--data_seed "$DATA_SEED")
    fi
    if [[ "$SELECTION_MODE" == "random" ]]; then
      random_selector_seed=$((seed + RANDOM_SELECTOR_OFFSET))
      cmd+=(--random_seed "$random_selector_seed")
    else
      cmd+=(--hotmap_mode "$HOTMAP_MODE")
      if [[ "$HOTMAP_SOURCE" == "hotmap" ]]; then
        [[ -n "$hotmap_path" ]] || die "No hotmap found for task='${TASK}' k='${k}' in '${HOTMAP_DIR}'"
        cmd+=(--hotmap "$hotmap_path")
      elif [[ "$HOTMAP_SOURCE" == "telemetry" ]]; then
        [[ -n "$TELEMETRY_PATH" ]] || die "Telemetry mode selected but no telemetry file was provided/found."
        cmd+=(--telemetry "$TELEMETRY_PATH")
      else
        if [[ -n "$hotmap_path" ]]; then
          cmd+=(--hotmap "$hotmap_path")
        elif [[ -n "$TELEMETRY_PATH" ]]; then
          cmd+=(--telemetry "$TELEMETRY_PATH")
        else
          die "Auto mode: no prebuilt hotmap in '${HOTMAP_DIR}' and no telemetry file found."
        fi
      fi
    fi
    if [[ "$PUSH_TO_HUB" -eq 0 ]]; then
      cmd+=(--no_push_to_hub)
    elif [[ -n "$HUB_REPO_PREFIX" ]]; then
      mode_tag="hotk${k}"
      if [[ "$SELECTION_MODE" == "random" ]]; then
        mode_tag="randk${k}"
      fi
      cmd+=(--hub_repo "${HUB_REPO_PREFIX}${MODEL_TAG}_${TASK}_s${seed}_${mode_tag}")
    fi
    if [[ "$CLEANUP_AFTER_PUSH" -eq 1 ]]; then
      cmd+=(--cleanup_after_push)
    fi
    if [[ "$USE_WANDB" -eq 0 ]]; then
      cmd+=(--no_wandb)
    fi
    cmd+=("${EXTRA_RUN_ARGS[@]}")

    if [[ "$SELECTION_MODE" == "random" ]]; then
      log "Launching RANDOM: seed=${seed}, k=${k}"
    else
      log "Launching HOT: seed=${seed}, k=${k}, source=${HOTMAP_SOURCE}"
    fi
    "${cmd[@]}"
  done
done

if [[ "$RUN_FULL" -eq 1 ]]; then
  for seed in "${SEEDS[@]}"; do
    cmd=(
      python3 -m scripts.run
      --task "$TASK"
      --mode full
      --model_tag "$MODEL_TAG"
      --seed "$seed"
    )
    if [[ -n "$DATA_SEED" ]]; then
      cmd+=(--data_seed "$DATA_SEED")
    fi
    if [[ "$PUSH_TO_HUB" -eq 0 ]]; then
      cmd+=(--no_push_to_hub)
    elif [[ -n "$HUB_REPO_PREFIX" ]]; then
      cmd+=(--hub_repo "${HUB_REPO_PREFIX}${MODEL_TAG}_${TASK}_s${seed}_full_lora")
    fi
    if [[ "$CLEANUP_AFTER_PUSH" -eq 1 ]]; then
      cmd+=(--cleanup_after_push)
    fi
    if [[ "$USE_WANDB" -eq 0 ]]; then
      cmd+=(--no_wandb)
    fi
    cmd+=("${EXTRA_RUN_ARGS[@]}")

    log "Launching FULL: seed=${seed}"
    "${cmd[@]}"
  done
fi

log "Training launcher completed."
