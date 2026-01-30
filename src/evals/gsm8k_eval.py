#!/usr/bin/env python3
import os, re, json, time, argparse, random
from dataclasses import asdict, dataclass
from typing import Optional, List, Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm.auto import tqdm

try:
    import wandb
except Exception:
    wandb = None


# -----------------------------
# Prompting
# -----------------------------
PAPER_STYLE_TEMPLATE = (
    "Question: {q}\n"
    "Answer: Let's think step by step.\n"
    "The final answer is: "
)

PLAIN_TEMPLATE = "Question: {q}\nAnswer:"


def build_prompt(q: str, template: str) -> str:
    return template.format(q=q)


# -----------------------------
# Answer extraction
# -----------------------------
def extract_answer(
    text: str,
    *,
    strict_hash: bool = False,
    prefer_final_answer_marker: bool = True,
    final_answer_marker: str = "The final answer is:",
) -> Optional[str]:
    if text is None:
        return None

    # normalize
    t = text.replace(",", "")

    # 1) Prefer "The final answer is: <num>" (paper-style)
    if prefer_final_answer_marker and final_answer_marker:
        # allow optional spaces after marker
        pat = re.escape(final_answer_marker) + r"\s*([-+]?\d+(?:\.\d+)?)"
        m = re.findall(pat, t)
        if m:
            return m[-1].strip()

    # 2) Prefer "#### <num>" (GSM8K canonical)
    m = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?)", t)
    if m:
        return m[-1].strip()

    if strict_hash:
        return None

    # 3) Fallback: last number-like
    m2 = re.findall(r"[-+]?\d+(?:\.\d+)?", t)
    if m2:
        return m2[-1].strip()

    return None


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def get_device(model) -> torch.device:
    return next(model.parameters()).device


def set_repro(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Greedy decoding is deterministic anyway, but keep this for consistency.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


@dataclass
class EvalSummary:
    run_name: str
    adapter: str
    model_name: str
    dataset: str
    split: str
    n: int
    bs: int
    max_input_len: int
    max_new_tokens: int
    prompt_template: str
    strict_hash: bool
    prefer_final_answer_marker: bool
    final_answer_marker: str
    do_sample: bool
    temperature: float
    correct: int
    total: int
    acc: float
    out_jsonl: str
    out_summary: str
    seconds: float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--adapter", default=None, help="HF repo id or local path. Omit for base model.")
    ap.add_argument("--save_root", default="/workspace/evals")

    ap.add_argument("--prompt_style", choices=["plain", "paper"], default="paper")
    ap.add_argument("--custom_template", default=None, help="If set, overrides prompt_style template.")

    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_input_len", type=int, default=1024)
    ap.add_argument("--n", type=int, default=None, help="Eval first n examples (debug).")

    ap.add_argument("--strict_hash", action="store_true")
    ap.add_argument("--prefer_final_answer_marker", action="store_true")
    ap.add_argument("--final_answer_marker", default="The final answer is:")

    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="hellora-repro")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--log_wrong_k", type=int, default=50)

    args = ap.parse_args()
    set_repro(args.seed)

    # choose prompt template
    if args.custom_template:
        prompt_template = args.custom_template
    else:
        prompt_template = PAPER_STYLE_TEMPLATE if args.prompt_style == "paper" else PLAIN_TEMPLATE

    out_dir = os.path.join(args.save_root, "gsm8k", args.run_name)
    ensure_dir(out_dir)
    out_jsonl = os.path.join(out_dir, "predictions.jsonl")
    out_summary = os.path.join(out_dir, "summary.json")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    adapter_tag = "BASE"
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        adapter_tag = args.adapter

    model.eval()
    device = get_device(model)

    # dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.n is not None:
        ds = ds.select(range(min(args.n, len(ds))))

    # wandb
    wb_run = None
    wrong_table = None
    wrong_logged = 0
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed/importable.")
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"eval/gsm8k/{args.run_name}",
            config={
                "task": "eval",
                "dataset": "gsm8k",
                "split": "test",
                "model_id": args.model,
                "adapter": adapter_tag,
                "prompt_style": args.prompt_style,
                "prompt_template": prompt_template,
                "n": len(ds),
                "bs": args.bs,
                "max_input_len": args.max_input_len,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "temperature": 0.0,
                "seed": args.seed,
                "strict_hash": args.strict_hash,
                "prefer_final_answer_marker": args.prefer_final_answer_marker,
                "final_answer_marker": args.final_answer_marker,
            },
        )
        wrong_table = wandb.Table(columns=["question", "gold_answer", "pred_answer", "pred_text"])

    correct = 0
    total = 0
    t0 = time.time()

    with open(out_jsonl, "w", encoding="utf-8") as f_jsonl, torch.inference_mode():
        for i in tqdm(range(0, len(ds), args.bs), desc="gsm8k eval"):
            batch = ds[i : i + args.bs]
            questions = batch["question"]
            gold_texts = batch["answer"]

            prompts = [build_prompt(q, prompt_template) for q in questions]
            inputs = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_len,
            )
            # move tensors to model device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # per-example prompt lengths (handles left padding correctly)
            prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

            out = model.generate(
                **inputs,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

            # decode per-example generated continuation
            gen_texts: List[str] = []
            for j in range(out.size(0)):
                pl = int(prompt_lens[j])
                gen_ids = out[j, pl:]
                gen_texts.append(tok.decode(gen_ids, skip_special_tokens=True))

            for q, gold, pred_text in zip(questions, gold_texts, gen_texts):
                gold_ans = extract_answer(
                    gold,
                    strict_hash=False,  # gsm8k gold has ####
                    prefer_final_answer_marker=False,
                )
                pred_ans = extract_answer(
                    pred_text,
                    strict_hash=args.strict_hash,
                    prefer_final_answer_marker=args.prefer_final_answer_marker,
                    final_answer_marker=args.final_answer_marker,
                )

                ok = (gold_ans is not None) and (pred_ans is not None) and (gold_ans == pred_ans)
                total += 1
                correct += int(ok)

                rec = {
                    "question": q,
                    "gold_text": gold,
                    "pred_text": pred_text,
                    "gold_answer": gold_ans,
                    "pred_answer": pred_ans,
                    "correct": bool(ok),
                }
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if wrong_table is not None and (not ok) and wrong_logged < args.log_wrong_k:
                    wrong_table.add_data(q, gold_ans, pred_ans, pred_text)
                    wrong_logged += 1

    secs = time.time() - t0
    acc = (correct / total) if total else 0.0

    summary = EvalSummary(
        run_name=args.run_name,
        adapter=adapter_tag,
        model_name=args.model,
        dataset="gsm8k",
        split="test",
        n=len(ds),
        bs=args.bs,
        max_input_len=args.max_input_len,
        max_new_tokens=args.max_new_tokens,
        prompt_template=prompt_template,
        strict_hash=args.strict_hash,
        prefer_final_answer_marker=args.prefer_final_answer_marker,
        final_answer_marker=args.final_answer_marker,
        do_sample=False,
        temperature=0.0,
        correct=correct,
        total=total,
        acc=acc,
        out_jsonl=out_jsonl,
        out_summary=out_summary,
        seconds=secs,
    )

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    if wb_run is not None:
        wandb.log({
            "eval/acc": acc,
            "eval/correct": correct,
            "eval/total": total,
            "eval/seconds": secs,
            "eval/wrong_logged": wrong_logged,
        })
        if wrong_table is not None:
            wandb.log({"eval/wrong_examples": wrong_table})
        wandb.finish()

    print("[IO] saved:", out_jsonl)
    print("[IO] saved:", out_summary)
    print(f"[DONE] acc={acc:.4f}  correct={correct}/{total}  seconds={secs:.1f}")


if __name__ == "__main__":
    main()
