#!/usr/bin/env python3
"""
Profile expert activations for one or more datasets.

Usage:
    # Single dataset
    python scripts/profiling.py --task gsm8k

    # Multiple datasets
    python scripts/profiling.py --task gsm8k wikitext pubmedqa

    # All registered datasets
    python scripts/profiling.py --all

    # Quick test run
    python scripts/profiling.py --all --n_samples 500
"""

import argparse
import json
import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers.utils.import_utils as _tf_import_utils

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils_profiling import make_profile
from src.config import TRAIN_CFG, SYS_CFG
from src.data_registry import DATASETS


def _prepare_remote_code_compat() -> None:
    # Some remote model repos still import this helper from older transformers APIs.
    if not hasattr(_tf_import_utils, "is_torch_fx_available"):
        _tf_import_utils.is_torch_fx_available = lambda: hasattr(torch, "fx")


def _prepare_model_config(model_id: str):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # DeepSeek remote code may expect rope_scaling["type"] while newer configs
    # provide rope_scaling["rope_type"].
    rope_scaling = getattr(cfg, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_scaling = dict(rope_scaling)

        if "type" not in rope_scaling:
            rope_scaling["type"] = rope_scaling.get("rope_type", "linear")

        # Some DeepSeek remote-code versions hard-index `factor`.
        # Keep a safe default when it is omitted by newer config schemas.
        if "factor" not in rope_scaling:
            factor = rope_scaling.get("scaling_factor", 1.0)
            try:
                factor = float(factor)
            except Exception:
                factor = 1.0
            if factor < 1.0:
                factor = 1.0
            rope_scaling["factor"] = factor

        # Normalise optional YaRN fields to float when present.
        for key in ("beta_fast", "beta_slow"):
            if key in rope_scaling:
                try:
                    rope_scaling[key] = float(rope_scaling[key])
                except Exception:
                    pass

        cfg.rope_scaling = rope_scaling
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Profile MoE expert activations.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=str, nargs="+", default=None,
                       choices=list(DATASETS.keys()),
                       help="One or more datasets to profile")
    group.add_argument("--all", action="store_true", help="Profile all registered datasets")
    parser.add_argument("--n_samples", type=int, default=None, help="Limit samples per dataset")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--split", type=str, default=None,
                        help="Override split for all selected datasets (default: use registry split)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--data_seed", type=int, default=None,
                        help="Dataset shuffle seed (default: uses --seed)")
    parser.add_argument("--run_name", type=str, default="default", help="Tag for output file")
    parser.add_argument(
        "--model",
        type=str,
        default=TRAIN_CFG.model_id,
        help=f"HF model id to profile (default: {TRAIN_CFG.model_id})",
    )
    parser.add_argument(
        "--gate_path",
        type=str,
        default=None,
        help="Optional gate module path per layer (e.g., mlp.gate, block_sparse_moe.gate)",
    )
    parser.add_argument("--num_experts", type=int, default=None, help="Optional override for router expert count")
    parser.add_argument("--top_k", type=int, default=None, help="Optional override for experts selected per token")
    parser.add_argument("--fail_fast", action="store_true",
                        help="Stop immediately if one dataset fails")
    args = parser.parse_args()

    # Resolve task list
    if args.all:
        tasks = list(DATASETS.keys())
    else:
        tasks = args.task

    # Load model once
    print(f"Loading model {args.model}...")
    _prepare_remote_code_compat()
    model_cfg = _prepare_model_config(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=model_cfg,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Profile
    results = {}
    for i, task in enumerate(tasks, 1):
        split = args.split or DATASETS[task]["split"]
        header = f"[{i}/{len(tasks)}] {task} (split={split})"
        print(f"\n{'='*60}\n{header}\n{'='*60}")
        t0 = time.time()
        try:
            pt_file = make_profile(
                dataset_key=task,
                model=model,
                tokenizer=tokenizer,
                split=args.split,
                n_samples=args.n_samples,
                bs=args.bs,
                seed=args.seed,
                data_seed=args.data_seed,
                run_name=args.run_name,
                gate_path=args.gate_path,
                num_experts=args.num_experts,
                top_k=args.top_k,
            )
            elapsed = time.time() - t0
            results[task] = {"status": "ok", "file": pt_file, "time": elapsed}
            print(f"  Done in {elapsed:.1f}s -> {pt_file}")
        except Exception as e:
            elapsed = time.time() - t0
            results[task] = {"status": "error", "error": str(e), "time": elapsed}
            print(f"  FAILED in {elapsed:.1f}s: {e}")
            if args.fail_fast:
                break

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for task, r in results.items():
        status = "ok" if r["status"] == "ok" else f"ERROR: {r['error']}"
        print(f"  {task:20s}  {r['time']:6.1f}s  {status}")
    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"\n{n_ok}/{len(results)} succeeded.")

    telemetry_root = SYS_CFG.get_output_dir("telemetry")
    safe_run_name = args.run_name.replace("/", "_")
    ts = time.strftime("%Y%m%d_%H%M%S")
    manifest_path = os.path.join(telemetry_root, f"profile_run_{safe_run_name}_{ts}.json")
    manifest = {
        "model_id": args.model,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "n_samples": args.n_samples,
        "batch_size": args.bs,
        "gate_path": args.gate_path,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "split_override": args.split,
        "tasks": tasks,
        "results": results,
        "telemetry_root": telemetry_root,
        "created_at": ts,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Run manifest: {manifest_path}")
    print(f"Telemetry root: {telemetry_root}")

    if n_ok != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
