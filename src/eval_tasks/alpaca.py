"""
CodeAlpaca-20k evaluation task — held-out loss / perplexity.

CodeAlpaca has no standard test split, so we carve the last `eval_frac`
(default 10 %) of the dataset as a held-out eval set.  The training
pipeline in data_registry uses shuffle(seed=...) before training, so
the last 10 % of the *unshuffled* dataset is never seen during training
regardless of the training seed.

Metric: perplexity = exp(mean cross-entropy) over the held-out examples,
each formatted with the same instruction template used for training.

Like wikitext.py, this uses CUSTOM_EVAL because it needs logits.
"""

import os
import json
import math

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from ..data_registry import EVAL_DATASETS, DATASETS


# ── Flag: evaluator.py will skip its own inference loop ──────────
CUSTOM_EVAL = True


def _format_example(ex) -> str:
    """Same template as DATASETS['alpaca']['text_fn']."""
    inp = ex.get("input", "") or ""
    return f"{ex['instruction']}\n{inp}\n{ex['output']}"


def add_args(parser):
    parser.add_argument(
        "--eval_frac",
        type=float,
        default=None,
        help="Fraction of dataset to use as held-out eval (default: from registry)",
    )


def load_data(args):
    """Returns (dataset_obj, formatted_texts, None)."""
    print("Loading CodeAlpaca-20k (held-out eval portion)...")
    ds_cfg = EVAL_DATASETS["alpaca"]
    ds = load_dataset(ds_cfg["path"], ds_cfg["name"], split=ds_cfg["split"])

    eval_frac = args.eval_frac if hasattr(args, "eval_frac") and args.eval_frac else ds_cfg.get("eval_frac", 0.10)
    n_total = len(ds)
    n_eval_start = int(n_total * (1 - eval_frac))

    # Take the TAIL as eval (training shuffles, but from full set — the
    # unshuffled tail is the cleanest held-out strategy).
    ds_eval = ds.select(range(n_eval_start, n_total))

    if args.n:
        ds_eval = ds_eval.select(range(min(args.n, len(ds_eval))))

    print(f"[Alpaca] Using {len(ds_eval)} held-out examples (last {eval_frac:.0%} of {n_total})")

    texts = [_format_example(ex) for ex in ds_eval]
    return ds_eval, texts, None


def run_custom_eval(args):
    """Per-example cross-entropy → aggregate perplexity."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    ds_eval, texts, _ = load_data(args)

    # ── Tokenize ─────────────────────────────────────────────────
    print("[Alpaca] Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = 2048  # match training config

    # ── Load model ───────────────────────────────────────────────
    print(f"[Alpaca] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter:
        print(f"[Alpaca] Loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    device = model.device

    # ── Per-example loss ─────────────────────────────────────────
    total_nll = 0.0
    total_tokens = 0
    per_example = []
    bs = args.bs

    for i in tqdm(range(0, len(texts), bs), desc="Alpaca eval"):
        batch_texts = texts[i : i + bs]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Labels: shift is handled internally; mask padding with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Per-example losses (need to recompute from logits)
        logits = out.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        for j in range(per_token_loss.size(0)):
            mask_j = shift_labels[j] != -100
            n_tok = mask_j.sum().item()
            if n_tok == 0:
                continue
            nll_j = per_token_loss[j][mask_j].sum().item()
            total_nll += nll_j
            total_tokens += n_tok
            per_example.append({
                "idx": i + j,
                "n_tokens": n_tok,
                "mean_nll": round(nll_j / n_tok, 6),
                "ppl": round(math.exp(nll_j / n_tok), 4),
            })

    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll)
    print(f"\n[Alpaca] Perplexity: {ppl:.4f}  (mean NLL: {mean_nll:.6f}, tokens: {total_tokens:,})")

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    summary = {
        "run_name": args.run_name,
        "model_name": args.model,
        "adapter": args.adapter,
        "dataset": "alpaca",
        "split": "held_out",
        "n_examples": len(per_example),
        "n_tokens": total_tokens,
        "mean_nll": round(mean_nll, 6),
        "perplexity": round(ppl, 4),
    }

    summary_path = os.path.join(args.output_dir, f"{args.run_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[IO] saved: {summary_path}")

    # Also save per-example breakdown (useful for analysis)
    detail_path = os.path.join(args.output_dir, f"{args.run_name}_detail.jsonl")
    with open(detail_path, "w") as f:
        for rec in per_example:
            f.write(json.dumps(rec) + "\n")
    print(f"[IO] saved: {detail_path}")

    if args.wandb and wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"eval/{args.run_name}",
            config=vars(args),
        )
        wandb.log({
            "eval/perplexity": ppl,
            "eval/mean_nll": mean_nll,
            "eval/n_tokens": total_tokens,
            "eval/n_examples": len(per_example),
        })
        wandb.save(summary_path)
        wandb.finish()
