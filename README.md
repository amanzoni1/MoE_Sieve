# MoE Sieve

LoRA experiment framework for MoE models with three stages:

1. profiling expert usage
2. training (hot/full/random target selection)
3. evaluation (task metrics + optional Spider official test-suite eval)

This README is general onboarding. Spider-specific setup is in its own section below.

## Project Structure

```text
MoE_Sieve/
├── src/
│   ├── __init__.py
│   ├── config.py                 # paths, defaults, global training/eval config
│   ├── data_registry.py          # dataset registry, formatters, Spider prompt/schema logic
│   ├── profiler.py               # profiler execution entry points
│   ├── trainer.py                # core training loop (HF Trainer + PEFT)
│   ├── evaluator.py              # generic eval runner
│   ├── utils_profiling.py        # telemetry + hotmap builders
│   ├── utils_training.py         # target selection (hot/full/random) + sanity helpers
│   ├── utils_eval.py             # merge + inference backends (HF/vLLM)
│   └── eval_tasks/
│       ├── __init__.py
│       ├── gsm8k.py
│       ├── hellaswag.py
│       └── spider.py             # Spider metrics + official evaluator integration
├── scripts/
│   ├── profiling.py              # CLI profiling launcher
│   ├── run.py                    # CLI training launcher (single run)
│   ├── eval.py                   # CLI eval launcher (single/sweep)
│   ├── summarize_eval.py         # aggregate summary json files
│   └── pod/
│       ├── common.sh             # pod setup helpers (deps/cache/login/assets)
│       ├── train_pod.sh          # pod training sweeps
│       └── eval_pod.sh           # pod eval sweeps
├── notebooks/                    # analysis and experiment notebooks
└── outputs/                      # auto-created artifacts: telemetry, runs, eval_results, hotmaps
```

## Quick Start

### Pod bootstrap

```bash
set -euxo pipefail
cd /workspace
git clone https://github.com/amanzoni1/MoE_Sieve.git MoE_Sieve_Project || true
cd /workspace/MoE_Sieve_Project
git pull --ff-only

export HF_TOKEN="YOUR_HF_TOKEN"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
```

### One-time setup (if running `python -m scripts.*` directly)

```bash
source scripts/pod/common.sh
setup_cache_dirs
install_train_deps
hf_login_if_token
wandb_login_if_key
```

If you use `scripts/pod/train_pod.sh` or `scripts/pod/eval_pod.sh`, setup is handled automatically.

## Standard Workflow

### 1) Profiling

```bash
python3 scripts/profiling.py --task spider --seed 123 --data_seed 123 --run_name spider_profile
```

Output:

- `outputs/telemetry/<task>/*.pt`

### 2) Training

Hot run (single):

```bash
python3 -m scripts.run \
  --task spider \
  --mode hot \
  --model_tag olmoe \
  --seed 42 \
  --data_seed 123 \
  --k 12 \
  --telemetry outputs/telemetry/spider/telemetry_spider_train_full_all_full_s123_global.pt
```

Full LoRA (single):

```bash
python3 -m scripts.run \
  --task spider \
  --mode full \
  --model_tag olmoe \
  --seed 42 \
  --data_seed 123
```

Outputs:

- `outputs/runs/<task>/<run_name>/...`
- HF adapter push (if enabled)
- W&B logs (if enabled)

### 3) Evaluation

Single adapter:

```bash
python3 -m scripts.eval \
  --task spider \
  --model allenai/OLMoE-1B-7B-0924 \
  --adapter AManzoni/olmoe_spider_s42_hotk12 \
  --backend vllm \
  --max_new_tokens 256
```

Sweep:

```bash
python3 -m scripts.eval \
  --task spider \
  --model allenai/OLMoE-1B-7B-0924 \
  --adapter_template "AManzoni/olmoe_spider_s{seed}_hotk{k}" \
  --model_seeds "42,99,123" \
  --ks "4,8,12,16" \
  --backend vllm \
  --max_new_tokens 256
```

Outputs:

- `outputs/eval_results/<task>/*.jsonl`
- `outputs/eval_results/<task>/*_summary.json`

## Spider Notes (Important)

Spider differs from other tasks in two ways:

1. Prompting requires schema text from `tables.json` (via `src/data_registry.py`).
2. Official metric uses `test-suite-sql-eval` databases + evaluator.

### Spider assets

```bash
source scripts/pod/common.sh
setup_spider_assets /workspace/test-suite-sql-eval 1 1
```

This ensures:

- `SPIDER_TABLES_JSON` is set
- `tables.json` exists
- test-suite databases exist (if requested)
- NLTK punkt assets exist (if requested)

### Spider prompt/schema sanity check

```bash
python3 - <<'PY'
from datasets import load_dataset
from src.data_registry import validate_spider_schema_or_raise, build_spider_schema_text, build_spider_prompt
ds = load_dataset("spider", split="train[:3]")
validate_spider_schema_or_raise(ds, context="preflight", max_examples=3)
for ex in ds:
    s = build_spider_schema_text(ex)
    p = build_spider_prompt(ex["question"], s, sql_answer=None)
    print(ex["db_id"], len(s), p[:200])
PY
```

### Official Spider eval from `scripts.eval`

```bash
python3 -m scripts.eval \
  --task spider \
  --model allenai/OLMoE-1B-7B-0924 \
  --adapter AManzoni/olmoe_spider_s42_hotk12 \
  --backend vllm \
  --max_new_tokens 256 \
  --ts_run_official \
  --ts_eval_repo /workspace/test-suite-sql-eval \
  --ts_etype exec
```

This writes:

- `*_ts_inputs/gold.txt`
- `*_ts_inputs/pred.txt`
- official eval output/parsed metrics inside summary

## Pod Convenience Launchers

Train sweeps:

```bash
bash scripts/pod/train_pod.sh --help
```

Eval sweeps:

```bash
bash scripts/pod/eval_pod.sh --help
```

These wrappers manage deps, cache dirs, optional Spider assets, HF/W&B login, and run loops over seeds/k.

## Outputs Cheat Sheet

- `outputs/telemetry/`: profiling tensors (`.pt`) and profile manifests
- `outputs/hotmaps/`: generated hotmap jsons
- `outputs/runs/`: training outputs/checkpoints/final adapters
- `outputs/eval_results/`: predictions, summaries, and Spider ts input files

## Troubleshooting

- Missing `datasets`/`nltk`/`sqlparse`: install deps (`install_train_deps` / `install_eval_deps`).
- Spider schema missing: run `setup_spider_assets ... 0 0` or export `SPIDER_TABLES_JSON`.
- Official eval DB errors: run `setup_spider_assets ... 1 1` and verify DB files exist under `database/`.
- Full LoRA OOM:
  - enable gradient checkpointing (`src/config.py`)
  - `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - reduce micro-batch (`--bs`) and increase `--grad_acc` to keep effective batch stable.
- `hf_transfer` error: unset `HF_HUB_ENABLE_HF_TRANSFER` or install `hf_transfer`.

## Reproducibility Notes

- Keep `seed`, `data_seed`, LR, epochs, and effective batch (`bs * grad_acc`) fixed.
- Local Spider official eval is deterministic for fixed `gold/pred/db`.
- vLLM greedy decoding is usually stable; tiny numeric variations can still occur depending on stack/hardware.
