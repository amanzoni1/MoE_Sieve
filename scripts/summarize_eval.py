import argparse
import glob
import json
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple


def _parse_seed(run_name: str) -> Optional[int]:
    match = re.search(r"(?:^|[_-])s(\d+)(?:$|[_-])", run_name)
    if not match:
        return None
    return int(match.group(1))


def _parse_hotk(run_name: str) -> Optional[int]:
    match = re.search(r"hot[_-]?k(\d+)", run_name)
    if not match:
        return None
    return int(match.group(1))


def _detect_dataset(row: Dict) -> Optional[str]:
    """Detect dataset from summary JSON metadata."""
    return row.get("dataset", None)


def _detect_metric(row: Dict) -> Tuple[str, str]:
    """Return (metric_key, metric_label) based on what's in the summary."""
    if "acc" in row:
        return "acc", "accuracy"
    if "perplexity" in row:
        return "perplexity", "perplexity"
    return "acc", "accuracy"  # fallback


def _load_summaries(input_dir: str, pattern: str) -> List[Dict]:
    files = glob.glob(os.path.join(input_dir, pattern))
    rows = []
    for path in sorted(files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_path"] = path
            rows.append(data)
        except Exception:
            continue
    return rows


def _filter_rows(
    rows: List[Dict],
    seeds: Optional[List[int]],
    ks: Optional[List[int]],
    dataset: Optional[str] = None,
) -> List[Dict]:
    out = []
    for r in rows:
        run_name = r.get("run_name", "")
        seed = _parse_seed(run_name)
        hotk = _parse_hotk(run_name)
        if seeds is not None and seed not in seeds:
            continue
        if ks is not None and hotk not in ks:
            continue
        if dataset is not None and r.get("dataset") != dataset:
            continue
        r["_seed"] = seed
        r["_hotk"] = hotk
        out.append(r)
    return out


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _variance(values: List[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    return statistics.variance(values)


def _t_critical_95(df: int) -> float:
    # Two-tailed 95% t critical values for df=1..30; use normal approx beyond.
    t_crit = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if df <= 0:
        return float("nan")
    if df in t_crit:
        return t_crit[df]
    return 1.96


def _format_accuracy_table(by_k, args) -> List[str]:
    """Format table for accuracy-based tasks (GSM8K)."""
    lines = []
    lines.append("k | n | mean_acc | std_acc")
    lines.append("--|---|----------|--------")
    for k in sorted(by_k.keys()):
        accs = [float(r.get("acc", 0.0)) for r in by_k[k]]
        mean_acc, std_acc = _mean_std(accs)
        var_acc = _variance(accs)
        n = len(accs)
        se_acc = std_acc / (n ** 0.5) if n > 0 else float("nan")
        tcrit = _t_critical_95(n - 1)
        ci_half = tcrit * se_acc if n > 1 else 0.0
        ci_low = mean_acc - ci_half
        ci_high = mean_acc + ci_half
        min_acc = min(accs) if accs else float("nan")
        max_acc = max(accs) if accs else float("nan")

        lines.append(f"{k} | {len(accs)} | {mean_acc:.4f} | {std_acc:.4f}")
        if args.extended:
            lines.append(
                f"  var={var_acc:.6f}  se={se_acc:.4f}  ci95=[{ci_low:.4f}, {ci_high:.4f}]  min={min_acc:.4f}  max={max_acc:.4f}"
            )
        if args.show_seeds:
            seed_str = ", ".join(
                f"s{r.get('_seed', 'na')}={float(r.get('acc', 0.0)):.4f}" for r in by_k[k]
            )
            lines.append(f"  seeds: {seed_str}")
    return lines


def _format_perplexity_table(by_k, args) -> List[str]:
    """Format table for perplexity-based tasks (WikiText, Alpaca)."""
    lines = []
    lines.append("k | n | mean_ppl | std_ppl")
    lines.append("--|---|----------|--------")
    for k in sorted(by_k.keys()):
        ppls = [float(r.get("perplexity", 0.0)) for r in by_k[k]]
        mean_ppl, std_ppl = _mean_std(ppls)
        var_ppl = _variance(ppls)
        n = len(ppls)
        se_ppl = std_ppl / (n ** 0.5) if n > 0 else float("nan")
        tcrit = _t_critical_95(n - 1)
        ci_half = tcrit * se_ppl if n > 1 else 0.0
        ci_low = mean_ppl - ci_half
        ci_high = mean_ppl + ci_half
        min_ppl = min(ppls) if ppls else float("nan")
        max_ppl = max(ppls) if ppls else float("nan")

        lines.append(f"{k} | {len(ppls)} | {mean_ppl:.4f} | {std_ppl:.4f}")
        if args.extended:
            lines.append(
                f"  var={var_ppl:.6f}  se={se_ppl:.4f}  ci95=[{ci_low:.4f}, {ci_high:.4f}]  min={min_ppl:.4f}  max={max_ppl:.4f}"
            )
        if args.show_seeds:
            seed_str = ", ".join(
                f"s{r.get('_seed', 'na')}={float(r.get('perplexity', 0.0)):.4f}" for r in by_k[k]
            )
            lines.append(f"  seeds: {seed_str}")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Summarize eval results across seeds/k-values. Auto-detects metric type."
    )
    parser.add_argument("--input_dir", default="./eval_results")
    parser.add_argument("--pattern", default="*_summary.json")
    parser.add_argument("--seeds", default=None, help="Comma-separated list, e.g. 42,99,123")
    parser.add_argument("--ks", default=None, help="Comma-separated list, e.g. 4,8,12,16")
    parser.add_argument("--dataset", default=None, help="Filter by dataset name (gsm8k, wikitext, alpaca)")
    parser.add_argument("--show_seeds", action="store_true")
    parser.add_argument("--extended", action="store_true", help="Show variance, SE, CI95, min/max per k")
    parser.add_argument("--out", default=None, help="Write report to this file (e.g., ./eval_results/summary.txt)")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",")] if args.seeds else None
    ks = [int(x.strip()) for x in args.ks.split(",")] if args.ks else None

    rows = _load_summaries(args.input_dir, args.pattern)
    rows = _filter_rows(rows, seeds=seeds, ks=ks, dataset=args.dataset)

    if not rows:
        print(f"No summary files found in {args.input_dir} (pattern: {args.pattern}).")
        return

    # Group by dataset, then by k
    by_dataset: Dict[str, List[Dict]] = {}
    for r in rows:
        ds = r.get("dataset", "unknown")
        by_dataset.setdefault(ds, []).append(r)

    all_lines = []
    for ds_name in sorted(by_dataset.keys()):
        ds_rows = by_dataset[ds_name]
        metric_key, metric_label = _detect_metric(ds_rows[0])

        by_k: Dict[int, List[Dict]] = {}
        for r in ds_rows:
            k = r.get("_hotk")
            if k is None:
                continue
            by_k.setdefault(k, []).append(r)

        if not by_k:
            continue

        all_lines.append(f"\n## {ds_name} ({metric_label})\n")

        if metric_key == "acc":
            all_lines.extend(_format_accuracy_table(by_k, args))
        else:
            all_lines.extend(_format_perplexity_table(by_k, args))

    report = "\n".join(all_lines)
    print(report)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
