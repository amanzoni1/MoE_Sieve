#!/usr/bin/env python3
import os
import re
import time
import argparse
import shutil
from typing import Optional

from .config import TRAIN_CFG
from .eval_tasks import get_task
from .utils_eval import (
    set_repro,
    short_hash,
    merge_and_save,
    run_inference_vllm,
    run_inference_hf,
)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--model", default=TRAIN_CFG.model_id)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model_seed", type=int, default=None)

    parser.add_argument("--merge_dir", default="./merged_models")
    parser.add_argument("--output_dir", default="./eval_results")

    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--cleanup_merge", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=TRAIN_CFG.wandb_project)
    return parser


def _strip_hub_prefix(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.split("/")[-1]


def _infer_model_seed_from_adapter(adapter: Optional[str]) -> Optional[int]:
    if not adapter:
        return None
    name = _strip_hub_prefix(adapter)
    match = re.search(r"(?:^|[_-])s(\d+)(?:$|[_-])", name)
    if not match:
        return None
    return int(match.group(1))


def _infer_selection_mode_from_adapter(adapter: Optional[str]) -> Optional[str]:
    if not adapter:
        return None
    name = _strip_hub_prefix(adapter)
    if "full_lora" in name:
        return "full"
    if re.search(r"rand(?:om)?[_-]?k\d+", name):
        return "random"
    if re.search(r"(?:^|[_-])(?:dyn[_-]?)?cov\d+(?:p\d+)?(?:$|[_-])", name):
        return "dyn"
    if re.search(r"hot[_-]?k\d+", name):
        return "hot"
    return None


def _infer_k_from_adapter(adapter: Optional[str]) -> Optional[int]:
    if not adapter:
        return None
    name = _strip_hub_prefix(adapter)
    match = re.search(r"(?:hot|rand|random)[_-]?k(\d+)", name)
    if not match:
        if "full_lora" in name:
            return 64
        return None
    return int(match.group(1))


def _infer_cov_tag_from_adapter(adapter: Optional[str]) -> Optional[str]:
    if not adapter:
        return None
    name = _strip_hub_prefix(adapter)
    match = re.search(r"(?:^|[_-])(?:dyn[_-]?)?cov(\d+(?:p\d+)?)(?:$|[_-])", name)
    if not match:
        return None
    return match.group(1)


def _infer_model_tag(model_id: str) -> str:
    model_name = model_id.split("/")[-1]
    model_tag = model_name.split("-")[0] if "-" in model_name else model_name
    return model_tag.lower()


def infer_run_name(
    task_name: str,
    model_id: str,
    adapter: Optional[str],
) -> str:
    model_tag = _infer_model_tag(model_id)
    seed = _infer_model_seed_from_adapter(adapter)
    k = _infer_k_from_adapter(adapter)
    cov_tag = _infer_cov_tag_from_adapter(adapter)
    selection = _infer_selection_mode_from_adapter(adapter)

    run_name = f"{model_tag}_{task_name}"
    if seed is not None:
        run_name += f"_s{seed}"
    if selection == "full":
        run_name += "_full_lora"
    elif selection == "random" and k is not None:
        run_name += f"_randk{k}"
    elif selection == "dyn" and cov_tag is not None:
        run_name += f"_cov{cov_tag}"
    elif k is not None:
        run_name += f"_hotk{k}"
    return run_name


def run_eval(task_name: str, args: argparse.Namespace):
    if not args.run_name:
        args.run_name = infer_run_name(
            task_name=task_name,
            model_id=args.model,
            adapter=args.adapter,
        )

    set_repro(args.seed)

    task = get_task(task_name)

    # ── Custom eval path (perplexity tasks that need logits) ─────
    if getattr(task, "CUSTOM_EVAL", False):
        print(f"[EVAL] Running custom eval for '{task_name}'...")
        t0 = time.time()
        task.run_custom_eval(args)
        duration = time.time() - t0
        print(f"Custom eval done in {duration:.2f}s")
        return

    # ── Standard generative eval path (GSM8K-style) ─────────────
    ds, prompts, golds = task.load_data(args)

    # B. Auto-Merge (vLLM only)
    effective_model = args.model
    run_adapter = args.adapter

    merge_path = None
    if args.backend == "vllm" and args.adapter:
        merge_name = f"merged_{short_hash(args.model)}_{short_hash(args.adapter)}"
        merge_path = os.path.join(args.merge_dir, merge_name)

        if not os.path.exists(os.path.join(merge_path, "config.json")):
            print(f"[SETUP] Merging adapter to {merge_path}...")
            merge_and_save(args.model, args.adapter, merge_path)
        else:
            print(f"[SETUP] Using cached merge: {merge_path}")

        effective_model = merge_path
        run_adapter = None

    # C. Inference
    print(f"Starting Inference ({args.backend})...")
    t0 = time.time()

    if args.backend == "vllm":
        outputs = run_inference_vllm(effective_model, prompts, args.max_new_tokens, tp=1)
    else:
        outputs = run_inference_hf(effective_model, run_adapter, prompts, args.max_new_tokens, args.bs)

    duration = time.time() - t0
    print(f"Inference done in {duration:.2f}s")

    task.score_and_save(args, ds, prompts, golds, outputs)

    if args.cleanup_merge and merge_path and os.path.exists(merge_path):
        print(f"[CLEANUP] Removing merged model: {merge_path}")
        shutil.rmtree(merge_path, ignore_errors=True)
