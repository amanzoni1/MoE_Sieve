import json
import os
from typing import Any, Dict, Iterable, List, Optional
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

def _as_clean_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _build_compact_schema(table_names: list, column_names: list) -> str:
    if not isinstance(table_names, list) or not isinstance(column_names, list):
        return ""
    if not table_names or not column_names:
        return ""

    tables = [_as_clean_str(t) for t in table_names]
    if not any(tables):
        return ""

    by_table: Dict[int, List[str]] = {i: [] for i in range(len(tables))}
    for pair in column_names:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            table_idx = int(pair[0])
        except Exception:
            continue

        if table_idx < 0 or table_idx >= len(tables):
            continue

        col_name = _as_clean_str(pair[1])
        if not col_name or col_name == "*":
            continue
        if col_name not in by_table[table_idx]:
            by_table[table_idx].append(col_name)

    lines: List[str] = []
    for idx, table in enumerate(tables):
        if not table:
            continue
        cols = by_table[idx]
        if cols:
            lines.append(f"{table}({', '.join(cols)})")
        else:
            lines.append(f"{table}()")
    return "\n".join(lines)


def _schema_from_spider_obj(obj) -> str:
    if not isinstance(obj, dict):
        return ""
    table_names = obj.get("table_names_original") or obj.get("table_names")
    column_names = obj.get("column_names_original") or obj.get("column_names")
    return _build_compact_schema(table_names, column_names)


_SPIDER_SCHEMA_BY_DB: Optional[Dict[str, str]] = None
_SPIDER_SCHEMA_CACHE_KEY: Optional[tuple] = None


def spider_tables_json_candidates() -> List[str]:
    candidates = []
    env_path = os.environ.get("SPIDER_TABLES_JSON")
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            "/workspace/test-suite-sql-eval/tables.json",
            os.path.join(os.getcwd(), "test-suite-sql-eval", "tables.json"),
            os.path.join(os.getcwd(), "tables.json"),
        ]
    )
    return candidates


def _spider_schema_cache_key(candidates: List[str]) -> tuple:
    keyed = []
    for path in candidates:
        exists = os.path.isfile(path)
        mtime = os.path.getmtime(path) if exists else None
        keyed.append((path, exists, mtime))
    return tuple(keyed)


def _load_spider_schema_by_db() -> Dict[str, str]:
    global _SPIDER_SCHEMA_BY_DB, _SPIDER_SCHEMA_CACHE_KEY
    candidates = spider_tables_json_candidates()
    cache_key = _spider_schema_cache_key(candidates)

    if _SPIDER_SCHEMA_BY_DB is not None and cache_key == _SPIDER_SCHEMA_CACHE_KEY:
        return _SPIDER_SCHEMA_BY_DB

    _SPIDER_SCHEMA_BY_DB = {}
    _SPIDER_SCHEMA_CACHE_KEY = cache_key
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                continue
            for obj in payload:
                if not isinstance(obj, dict):
                    continue
                db_id = _as_clean_str(obj.get("db_id"))
                if not db_id:
                    continue
                schema = _schema_from_spider_obj(obj)
                if schema:
                    _SPIDER_SCHEMA_BY_DB[db_id] = schema
            if _SPIDER_SCHEMA_BY_DB:
                break
        except Exception:
            continue
    return _SPIDER_SCHEMA_BY_DB


def build_spider_schema_text(ex) -> str:
    """
    Build a compact schema string from Spider example metadata.
    Falls back to empty string when schema metadata is unavailable.
    """
    raw_schema = ex.get("schema")
    if isinstance(raw_schema, str) and raw_schema.strip():
        return raw_schema.strip()

    schema = _schema_from_spider_obj(ex)
    if schema:
        return schema

    db_id = _as_clean_str(ex.get("db_id"))
    if not db_id:
        return ""
    return _load_spider_schema_by_db().get(db_id, "")


def build_spider_prompt(question: str, schema_text: str, sql_answer: Optional[str] = None) -> str:
    """
    Shared Spider prompt template used by both training and eval.
    """
    schema = _as_clean_str(schema_text)
    if not schema:
        raise ValueError(
            "Spider schema is required but missing. "
            "Set SPIDER_TABLES_JSON or place tables.json in test-suite-sql-eval."
        )

    parts = [
        "You are a SQLite SQL expert.",
        "Given the database schema, write a SQL query that answers the question.",
    ]
    parts += ["", "Schema:", schema]
    parts += [
        "",
        f"Question: {question}",
        "Return only SQL.",
    ]
    if sql_answer is None:
        parts.append("SQL:")
    else:
        parts.append(f"SQL: {sql_answer}")
    return "\n".join(parts)


def validate_spider_schema_or_raise(
    examples: Iterable[Dict[str, Any]],
    *,
    context: str = "Spider",
    max_examples: Optional[int] = None,
) -> None:
    checked = 0
    missing = 0
    missing_db_ids: List[str] = []
    seen_ids = set()

    for ex in examples:
        schema = build_spider_schema_text(ex)
        checked += 1
        if not schema:
            missing += 1
            db_id = _as_clean_str(ex.get("db_id"))
            if db_id and db_id not in seen_ids and len(missing_db_ids) < 8:
                seen_ids.add(db_id)
                missing_db_ids.append(db_id)
        if max_examples is not None and checked >= max_examples:
            break

    if missing > 0:
        sample = ", ".join(missing_db_ids) if missing_db_ids else "unknown"
        candidates = "\n".join(f"  - {p}" for p in spider_tables_json_candidates())
        raise ValueError(
            f"{context}: missing schema for {missing}/{checked} example(s). "
            f"Sample db_id(s): {sample}\n"
            "Set SPIDER_TABLES_JSON or place tables.json in one of:\n"
            f"{candidates}"
        )


def fmt_spider(ex):
    """Spider: schema-grounded text-to-SQL formatting."""
    schema_text = build_spider_schema_text(ex)
    return build_spider_prompt(ex["question"], schema_text, sql_answer=ex["query"])


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
    # ── Primary k-sweep anchors (cross-dataset experiments) ───────
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",              # 7,473 examples
        "text_fn": fmt_gsm8k,
        "lr": 4e-4, "epochs": 3,      # reference dataset — H=0.911
    },
    "spider": {
        "path": "spider",
        "name": None,
        "split": "train",              # 7,000 examples; eval on validation (1,034)
        "text_fn": fmt_spider,
        "lr": 4e-4, "epochs": 3,      # low-entropy anchor — H=0.874
    },
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "name": None,
        "split": "train",              # 39,905 examples; subsample to ~7,500 via --train_samples
        "text_fn": fmt_hellaswag,
        "lr": 4e-4, "epochs": 3,      # high-entropy anchor — H=0.966
    },

    # ── Profiling-only datasets ─────────────
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

    "mmlu": {
        "path": "cais/mmlu",
        "name": "all",
        "split": "test",               # MMLU "test" is the standard eval split — H=0.974
        "text_fn": fmt_mmlu,
    },
    "boolq": {
        "path": "google/boolq",
        "name": None,
        "split": "train",
        "text_fn": fmt_boolq,
    },
    "pubmedqa": {
        "path": "qiaojin/PubMedQA",
        "name": "pqa_artificial",       # 211k examples, richest split
        "split": "train",
        "text_fn": fmt_pubmedqa,
    },
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "train",
        "text_fn": fmt_arc,
    },
    "piqa": {
        "path": "lighteval/piqa",
        "name": None,
        "split": "train",
        "text_fn": fmt_piqa,
    },
    "mbpp": {
        "path": "google-research-datasets/mbpp",
        "name": "full",
        "split": "train",
        "text_fn": fmt_mbpp,
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
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "name": None,
        "split": "validation",      # labels available; standard dev-time split
    },
    "spider": {
        "path": "spider",
        "name": None,
        "split": "validation",      # official train/validation partition
    },
}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def load_and_format_dataset(
    key: str,
    tokenizer,
    max_len: int,
    n_samples: Optional[int] = None,
    seed: int = 123,
    data_seed: Optional[int] = None,
):
    """
    Helper used by TRAINER.PY to get tokenized, ready-to-train tensors.
    Requires that the dataset has lr/epochs in the registry OR that
    the caller provides overrides — but that's the launcher's job, not ours.
    """
    cfg = DATASETS[key]
    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"])
    data_seed_eff = seed if data_seed is None else data_seed
    ds = ds.shuffle(seed=data_seed_eff)

    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    if key == "spider":
        validate_spider_schema_or_raise(
            ds,
            context=f"Spider training split='{cfg['split']}'",
            max_examples=min(256, len(ds)),
        )

    fn = cfg["text_fn"]

    def tokenize(ex):
        return tokenizer(fn(ex), truncation=True, max_length=max_len)

    return ds.map(tokenize, remove_columns=ds.column_names)
