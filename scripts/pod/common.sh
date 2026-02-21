#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log() {
  printf '[pod] %s\n' "$*"
}

die() {
  printf '[pod][error] %s\n' "$*" >&2
  exit 1
}

split_csv() {
  local value="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<<"$value"
  local i
  for i in "${!out_ref[@]}"; do
    out_ref[$i]="$(printf '%s' "${out_ref[$i]}" | xargs)"
  done
}

setup_cache_dirs() {
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
  export WANDB_DIR="${WANDB_DIR:-/workspace/.cache/wandb}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/workspace/.cache/wandb}"
  mkdir -p "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR"
}

install_train_deps() {
  python3 -m pip install -U pip
  python3 -m pip install --no-cache-dir -U \
    "transformers==4.57.6" \
    "accelerate==1.12.0" \
    "peft==0.18.1" \
    datasets \
    gdown \
    wandb \
    huggingface_hub
}

install_eval_deps() {
  python3 -m pip install -U pip
  python3 -m pip install --no-cache-dir -U \
    transformers \
    datasets \
    peft \
    accelerate \
    bitsandbytes \
    gdown \
    sqlparse \
    nltk \
    wandb \
    vllm \
    huggingface_hub
}

setup_spider_assets() {
  local ts_repo="${1:-/workspace/test-suite-sql-eval}"
  local need_database="${2:-0}"
  local need_nltk="${3:-0}"
  local tables_json="${ts_repo}/tables.json"
  local db_dir="${ts_repo}/database"
  local have_db_files=0

  if [[ ! -d "$ts_repo" ]]; then
    log "Cloning Spider test-suite evaluator into $ts_repo ..."
    git clone https://github.com/taoyds/test-suite-sql-eval.git "$ts_repo"
  fi

  [[ -f "${ts_repo}/evaluation.py" ]] || die "Missing ${ts_repo}/evaluation.py"

  if [[ ! -f "$tables_json" ]]; then
    command -v gdown >/dev/null 2>&1 || die "gdown is required to fetch Spider tables.json. Install deps or pass --no-pip-install only if assets already exist."
    log "Downloading Spider tables.json ..."
    gdown --fuzzy "https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing" -O /tmp/spider_data.zip
    rm -rf /tmp/spider_data
    mkdir -p /tmp/spider_data
    python3 -m zipfile -e /tmp/spider_data.zip /tmp/spider_data
    cp /tmp/spider_data/spider/tables.json "$tables_json"
    rm -rf /tmp/spider_data/__MACOSX
  fi

  if [[ "$need_database" -eq 1 ]]; then
    if [[ -d "$db_dir" ]] && find "$db_dir" -type f \( -name "*.sqlite" -o -name "*.db" \) ! -name "._*" ! -path "*/__MACOSX/*" -print -quit | grep -q .; then
      have_db_files=1
    fi
    if [[ "$have_db_files" -eq 0 ]]; then
      command -v gdown >/dev/null 2>&1 || die "gdown is required to fetch Spider test-suite databases. Install deps or provide assets manually."
      log "Downloading Spider official test-suite databases ..."
      gdown --fuzzy "https://drive.google.com/file/d/1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view?usp=sharing" -O /tmp/testsuitedatabases.zip
      python3 -m zipfile -e /tmp/testsuitedatabases.zip "$ts_repo"
      rm -rf "${ts_repo}/__MACOSX"
      if ! find "$db_dir" -type f \( -name "*.sqlite" -o -name "*.db" \) ! -name "._*" ! -path "*/__MACOSX/*" -print -quit | grep -q .; then
        die "Spider database download/extract completed but no .sqlite/.db files were found under ${db_dir}"
      fi
    fi
  fi

  if [[ "$need_nltk" -eq 1 ]]; then
    python3 - <<'PY'
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
print("NLTK punkt assets ready.")
PY
  fi

  export SPIDER_TABLES_JSON="$tables_json"
  log "Spider schema source: $SPIDER_TABLES_JSON"
}

hf_login_if_token() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    return 1
  fi
  python3 - <<'PY'
import os
from huggingface_hub import login, whoami
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit(1)
login(token=token)
print("HF login:", whoami()["name"])
PY
}

wandb_login_if_key() {
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    return 1
  fi
  python3 - <<'PY'
import os
import wandb
key = os.environ.get("WANDB_API_KEY")
if not key:
    raise SystemExit(1)
wandb.login(key=key, relogin=True)
print("W&B login: ok")
PY
}
