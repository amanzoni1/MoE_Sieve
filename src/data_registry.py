from typing import Optional
from datasets import load_dataset


# Formatters — each returns a single string for profiling/training
def fmt_gsm8k(ex):
    return f"Question: {ex['question']}\nAnswer: {ex['answer']}"

def fmt_alpaca(ex):
    inp = ex.get("input", "") or ""
    return f"{ex['instruction']}\n{inp}\n{ex['output']}"

def fmt_wikitext(ex):
    return ex["text"]

def fmt_arc(ex):
    """ARC-Challenge: question + labeled choices + answer."""
    choices = ex["choices"]
    opts = "\n".join(f"  {l}) {t}" for l, t in zip(choices["label"], choices["text"]))
    return f"Question: {ex['question']}\n{opts}\nAnswer: {ex['answerKey']}"

def fmt_piqa(ex):
    """PIQA: goal + two solutions."""
    correct = ex["sol1"] if ex["label"] == 0 else ex["sol2"]
    return f"Goal: {ex['goal']}\nSolution 1: {ex['sol1']}\nSolution 2: {ex['sol2']}\nAnswer: {correct}"

def fmt_hellaswag(ex):
    """HellaSwag: context + 4 endings."""
    endings_str = "\n".join(f"  {i}) {e}" for i, e in enumerate(ex["endings"]))
    correct = ex["endings"][int(ex["label"])]
    return f"{ex['ctx']}\n{endings_str}\nAnswer: {correct}"

def fmt_boolq(ex):
    """BoolQ: passage + yes/no question."""
    ans = "Yes" if ex["answer"] else "No"
    return f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer: {ans}"

def fmt_pubmedqa(ex):
    """PubMedQA: biomedical question + context + decision."""
    ctx = " ".join(ex["context"]["contexts"]) if isinstance(ex["context"], dict) else str(ex["context"])
    return f"Context: {ctx}\nQuestion: {ex['question']}\nAnswer: {ex['long_answer']}"

def fmt_mbpp(ex):
    """MBPP: code task description + solution."""
    return f"Task: {ex['text']}\nCode:\n{ex['code']}"

def fmt_mmlu(ex):
    """MMLU: question + 4 choices + answer."""
    labels = ["A", "B", "C", "D"]
    opts = "\n".join(f"  {labels[i]}) {c}" for i, c in enumerate(ex["choices"]))
    ans_idx = ex["answer"]
    return f"Question: {ex['question']}\n{opts}\nAnswer: {labels[ans_idx]}"


# ═══════════════════════════════════════════════════════════════════
# Dataset Registry
# ═══════════════════════════════════════════════════════════════════
#
# Every entry has:  path, name, split, text_fn
# Training config (lr, epochs) is optional — when present the
# launcher uses it as default; when absent the user must pass
# --lr / --epochs explicitly.
# ═══════════════════════════════════════════════════════════════════

DATASETS = {
    # ── Datasets with training defaults ──────────────────────────
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
        "text_fn": fmt_gsm8k,
        "lr": 4e-4, "epochs": 3,
    },
    "alpaca": {
        "path": "sahil2801/CodeAlpaca-20k",
        "name": None,
        "split": "train",
        "text_fn": fmt_alpaca,
        "lr": 2e-3, "epochs": 2,
    },
    "wikitext": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "train",
        "text_fn": fmt_wikitext,
        "lr": 4e-4, "epochs": 1,
    },

    # ── Additional datasets (no training defaults yet) ───────────
    "pubmedqa": {
        "path": "qiaojin/PubMedQA",
        "name": "pqa_artificial",       # 211k examples, richest split
        "split": "train",
        "text_fn": fmt_pubmedqa,
    },
    "mbpp": {
        "path": "google-research-datasets/mbpp",
        "name": "full",
        "split": "train",
        "text_fn": fmt_mbpp,
    },
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "train",
        "text_fn": fmt_arc,
    },
    "boolq": {
        "path": "google/boolq",
        "name": None,
        "split": "train",
        "text_fn": fmt_boolq,
    },
    "piqa": {
        "path": "ybisk/piqa",
        "name": None,
        "split": "train",
        "text_fn": fmt_piqa,
    },
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "name": None,
        "split": "train",
        "text_fn": fmt_hellaswag,
    },
    "mmlu": {
        "path": "cais/mmlu",
        "name": "all",
        "split": "test",               # MMLU "test" is the standard eval split
        "text_fn": fmt_mmlu,
    },
}


# ═══════════════════════════════════════════════════════════════════
# Eval Registry (datasets with eval tasks implemented)
# ═══════════════════════════════════════════════════════════════════

EVAL_DATASETS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
    },
    "wikitext": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
    },
    "alpaca": {
        "path": "sahil2801/CodeAlpaca-20k",
        "name": None,
        "split": "train",           # no test split; we carve a held-out portion
        "eval_frac": 0.10,          # last 10 % used for eval
    },
}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def load_and_format_dataset(key: str, tokenizer, max_len: int, n_samples: Optional[int] = None, seed: int = 123):
    """
    Helper used by TRAINER.PY to get tokenized, ready-to-train tensors.
    Requires that the dataset has lr/epochs in the registry OR that
    the caller provides overrides — but that's the launcher's job, not ours.
    """
    cfg = DATASETS[key]
    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"])
    ds = ds.shuffle(seed=seed)

    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    fn = cfg["text_fn"]

    def tokenize(ex):
        return tokenizer(fn(ex), truncation=True, max_length=max_len)

    return ds.map(tokenize, remove_columns=ds.column_names)
