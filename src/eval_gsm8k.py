#!/usr/bin/env python3
import os
import gc
import re
import json
import time
import argparse
import random
import hashlib
import torch
from typing import Optional, List
from datasets import load_dataset
from tqdm.auto import tqdm

# Optional imports
try:
    import wandb
except ImportError:
    wandb = None

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None


# Configuration & Prompting
PLAIN_TEMPLATE = "Question: {q}\nAnswer:"

def build_prompt(q: str, template: str) -> str:
    return template.format(q=q)

def set_repro(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


# Robust Extraction & Normalization
def normalize_number(s: str) -> str:
    if not s:
        return s
    s = s.replace(",", "")
    try:
        f_val = float(s)
        # If it's effectively an integer (42.0), return '42'
        if f_val.is_integer():
            return str(int(f_val))
        return str(f_val)
    except ValueError:
        return s

def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None

    # 1. Try GSM8K standard '####'
    # We take the FIRST match to avoid the "repetition loop" truncation bug.
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text.replace(",", ""))
    if match:
        return normalize_number(match.group(1).strip())

    # 2. Fallback: Last number
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        return normalize_number(numbers[-1].strip())

    return None


# Model Backend: Merge
def merge_and_save(base_model: str, adapter: str, out_dir: str):
    """Merges LoRA into base model safely with sharding."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[MERGE] Loading base: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Use bfloat16 for Ampere (A40/A6000)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )

    print(f"[MERGE] Loading adapter: {adapter}")
    model = PeftModel.from_pretrained(base, adapter)
    model = model.merge_and_unload()

    print(f"[MERGE] Saving to: {out_dir}")
    # Shard size 2GB to prevent OOM
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(out_dir)

    print("[MERGE] Cleaning up VRAM...")

    del model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print("[MERGE] Done. GPU is pristine.")

    return out_dir


# Inference Engines
def run_inference_vllm(model_path: str, prompts: List[str], max_tokens: int, tp: int) -> List[str]:
    if LLM is None:
        raise ImportError("vLLM not installed.")

    print(f"[VLLM] Initializing engine (max_tokens={max_tokens})...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        trust_remote_code=True,
        gpu_memory_utilization=0.85
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        stop=["\nQuestion:", "\n\nQuestion:"]
    )

    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text for out in outputs]

def run_inference_hf(model_path: str, adapter: str, prompts: List[str], max_tokens: int, bs: int) -> List[str]:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[HF] Initializing (Batch Size={bs})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if adapter:
        print(f"[HF] Loading LoRA: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()

    results: List[str] = []

    for i in tqdm(range(0, len(prompts), bs), desc="HF Inference"):
        batch_prompts = prompts[i : i + bs]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        input_seq_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_only = outputs[:, input_seq_len:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        results.extend(decoded)

    return results


# Main Execution
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--merge_dir", default="./merged_models")
    parser.add_argument("--output_dir", default="./eval_results")

    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="hellora-repro")

    args = parser.parse_args()
    set_repro(args.seed)

    # A. Data
    print("Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.n:
        ds = ds.select(range(args.n))

    prompts = [build_prompt(q, PLAIN_TEMPLATE) for q in ds["question"]]
    golds = ds["answer"]

    # B. Auto-Merge (vLLM only)
    effective_model = args.model
    run_adapter = args.adapter

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

    # D. W&B Setup
    if args.wandb and wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"gsm8k/{args.run_name}",
            config=vars(args)
        )
        wb_table = wandb.Table(columns=["question", "gold_answer", "pred_answer", "pred_text"])

    # E. Score & Save
    correct = 0
    wrong_logged = 0
    results = []

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.run_name}.jsonl")

    print("Scoring results...")
    with open(out_file, "w") as f:
        # Use tqdm for scoring progress too, it's fast but good to see
        for q, gold, pred in tqdm(zip(ds["question"], golds, outputs), total=len(prompts), desc="Scoring"):
            # Extract & Normalize
            gold_ans = extract_answer(gold)
            pred_ans = extract_answer(pred)

            # Comparison (String equality after normalization)
            is_correct = (gold_ans is not None) and (pred_ans is not None) and (gold_ans == pred_ans)
            if is_correct:
                correct += 1

            res = {
                "question": q,
                "gold_text": gold,
                "pred_text": pred,
                "gold_ans": gold_ans,
                "pred_ans": pred_ans,
                "correct": is_correct
            }
            f.write(json.dumps(res) + "\n")
            results.append(res)

            # W&B Logging (Only wrong answers, first 100)
            if args.wandb and wandb and (not is_correct) and wrong_logged < 100:
                 wb_table.add_data(q, gold_ans, pred_ans, pred)
                 wrong_logged += 1

    acc = correct / len(ds)
    print(f"\nFinal Accuracy: {acc:.2%}")

    summary = {
        "run_name": args.run_name,
        "model_name": args.model,
        "adapter": args.adapter,
        "dataset": "gsm8k",
        "split": "test",
        "n": len(ds),
        "bs": args.bs,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template": PLAIN_TEMPLATE,
        "correct": correct,
        "total": len(ds),
        "acc": acc,
        "predictions_jsonl": out_file,
    }

    summary_path = os.path.join(args.output_dir, f"{args.run_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, indent=2)

    print("[IO] saved:", summary_path)

    if args.wandb and wandb:
        wandb.save(summary_path)
        wandb.save(out_file)

    if args.wandb and wandb:
        wandb.log({
            "eval/acc": acc,
            "eval/correct": correct,
            "eval/total": len(ds),
            "eval/wrong_examples": wb_table
        })
        wandb.finish()

if __name__ == "__main__":
    main()
