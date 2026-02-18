"""
WikiText-2 evaluation task — perplexity on the test split.

Unlike GSM8K (generative, exact-match), this is a standard LM eval:
we compute the cross-entropy loss over the full test set and report
perplexity = exp(mean_loss).

Because perplexity requires access to model logits, this task runs
its own forward pass instead of delegating to the shared inference
engines in utils_eval.  The evaluator detects this via `CUSTOM_EVAL = True`.
"""

import os
import json
import math

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from ..data_registry import EVAL_DATASETS


# ── Flag: evaluator.py will skip its own inference loop ──────────
CUSTOM_EVAL = True


def add_args(parser):
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding-window stride for perplexity (default: 512)",
    )


def load_data(args):
    """Returns (dataset_obj, texts_list, None).

    `golds` is None because perplexity has no per-example gold label.
    """
    print("Loading WikiText-2 test split...")
    ds_cfg = EVAL_DATASETS["wikitext"]
    ds = load_dataset(ds_cfg["path"], ds_cfg["name"], split=ds_cfg["split"])
    if args.n:
        ds = ds.select(range(args.n))

    # Filter out empty lines (WikiText has many blank lines)
    texts = [t for t in ds["text"] if t.strip()]
    return ds, texts, None


def run_custom_eval(args):
    """Full perplexity evaluation with sliding window.

    Called directly by evaluator.py when CUSTOM_EVAL is set.
    Returns nothing — saves results to disk (and optionally W&B).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    ds, texts, _ = load_data(args)

    # ── Tokenize the entire test set as a single sequence ────────
    print("[WikiText] Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Concatenate all non-empty lines with double newline (original format)
    full_text = "\n\n".join(texts)
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids  # (1, seq_len)
    seq_len = input_ids.size(1)
    print(f"[WikiText] Total tokens: {seq_len:,}")

    # ── Load model ───────────────────────────────────────────────
    print(f"[WikiText] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter:
        print(f"[WikiText] Loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    max_len = getattr(model.config, "max_position_embeddings", 2048)
    stride = args.stride
    device = model.device

    # ── Sliding-window perplexity ────────────────────────────────
    nlls = []
    n_tokens = 0
    prev_end = 0

    pbar = tqdm(total=seq_len, desc="Perplexity", unit="tok")
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        chunk = input_ids[:, begin:end].to(device)

        # Only count loss for tokens beyond the overlap region
        target_start = max(0, prev_end - begin)
        target_ids = chunk.clone()
        target_ids[:, :target_start] = -100  # mask overlap

        with torch.inference_mode():
            out = model(input_ids=chunk, labels=target_ids)

        # out.loss is mean over non-masked tokens in the chunk
        n_target = (target_ids != -100).sum().item()
        nlls.append(out.loss.float().item() * n_target)
        n_tokens += n_target

        pbar.update(end - prev_end)
        prev_end = end
        if end == seq_len:
            break

    pbar.close()

    mean_nll = sum(nlls) / n_tokens
    ppl = math.exp(mean_nll)
    print(f"\n[WikiText] Perplexity: {ppl:.4f}  (mean NLL: {mean_nll:.6f}, tokens: {n_tokens:,})")

    # ── Save results ─────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    summary = {
        "run_name": args.run_name,
        "model_name": args.model,
        "adapter": args.adapter,
        "dataset": "wikitext",
        "split": "test",
        "n_tokens": n_tokens,
        "stride": stride,
        "max_len": max_len,
        "mean_nll": round(mean_nll, 6),
        "perplexity": round(ppl, 4),
    }

    summary_path = os.path.join(args.output_dir, f"{args.run_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[IO] saved: {summary_path}")

    if args.wandb and wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"eval/{args.run_name}",
            config=vars(args),
        )
        wandb.log({
            "eval/perplexity": ppl,
            "eval/mean_nll": mean_nll,
            "eval/n_tokens": n_tokens,
        })
        wandb.save(summary_path)
        wandb.finish()
