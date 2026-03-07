import os
import json
import heapq
import torch
from tqdm.auto import tqdm
from typing import Optional, List, Dict, Any
from datasets import load_dataset

from .config import SYS_CFG
from .profiler import ProfilerEngine
from .data_registry import DATASETS

def make_profile(
    dataset_key: str,
    model,
    tokenizer,
    run_name: str = "default",
    split: Optional[str] = None,
    seed: int = 123,
    data_seed: Optional[int] = None,
    n_samples: Optional[int] = None,
    bs: int = 16,
    seq_len: int = 2048,
    bucket_edges: Optional[List[int]] = None,
    store_mass: bool = True,
    gate_path: Optional[str] = None,
    num_experts: Optional[int] = None,
    top_k: Optional[int] = None,
) -> str:
    """
    Runs the ProfilerEngine on a dataset and saves the raw telemetry .pt file.
    If n_samples is None, profiles the entire dataset.
    """
    output_dir = SYS_CFG.get_output_dir(f"telemetry/{dataset_key}")
    ds_cfg = DATASETS[dataset_key]
    text_fn = ds_cfg["text_fn"]

    # Load raw dataset
    target_split = split or ds_cfg["split"]
    raw_ds = load_dataset(ds_cfg["path"], ds_cfg["name"], split=target_split)
    data_seed_eff = seed if data_seed is None else data_seed
    raw_ds = raw_ds.shuffle(seed=data_seed_eff)

    if n_samples is not None:
        raw_ds = raw_ds.select(range(min(n_samples, len(raw_ds))))

    selected_n = len(raw_ds)
    skipped_short = 0
    profiled_examples = 0

    eng = ProfilerEngine(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        seq_len=seq_len,
        bucket_edges=bucket_edges,
        store_mass=store_mass,
        gate_path=gate_path,
        num_experts=num_experts,
        top_k=top_k,
    )

    eng.attach_hooks()
    try:
        buf = []
        pbar = tqdm(total=selected_n, desc=f"Profiling {dataset_key}")
        try:
            for ex in raw_ds:
                pbar.update(1)
                text = text_fn(ex)
                if not isinstance(text, str) or len(text) < 10:
                    skipped_short += 1
                    continue

                buf.append(text)

                if len(buf) == bs:
                    eng._process_batch(buf)
                    profiled_examples += len(buf)
                    buf = []

            if buf:
                eng._process_batch(buf)
                profiled_examples += len(buf)
        finally:
            pbar.close()

    finally:
        eng.detach_hooks()
        torch.cuda.empty_cache()

    sanity = eng.validate_buffer_or_raise()
    print(
        f"[Profiler] sanity ok: layers={sanity['layers_checked']} "
        f"assignments/layer={sanity['assignments_per_layer'][0] if sanity['assignments_per_layer'] else 0}"
    )

    suffix = "bucketed" if bucket_edges is not None else "global"
    count_str = f"n{selected_n}" if n_samples is not None else "full"
    filename = f"telemetry_{dataset_key}_{target_split}_{count_str}_{run_name}_{suffix}.pt"
    pt_path = os.path.join(output_dir, filename)

    payload = {
        "meta": {
            "dataset": dataset_key,
            "split": target_split,
            "seed": seed,
            "data_seed": data_seed_eff,
            "n_requested": n_samples,
            "n_samples": profiled_examples,
            "n_selected": selected_n,
            "n_skipped_short": skipped_short,
            "seq_len": seq_len,
            "layers": eng.num_layers,
            "experts": eng.num_experts,
            "k": eng.top_k,
            "gate_path": gate_path,
            "sanity": sanity,
        }
    }

    for li in range(eng.num_layers):
        layer_buf = eng.data_buffer[li]
        payload[li] = {
            "counts": layer_buf["counts"].clone(),
            "total": torch.tensor(layer_buf["total"], dtype=torch.long),
        }
        if store_mass:
            payload[li]["mass"] = layer_buf["mass"].clone()

    torch.save(payload, pt_path)
    print(f"Telemetry saved: {pt_path}")
    return pt_path


def build_hotmap(
    pt_path: str,
    k: int = 8,
    out_json: Optional[str] = None,
    mode: str = "counts"
) -> str:
    """
    Converts a telemetry .pt file into a JSON Hotmap for training.
    """
    print(f"Building Hotmap (K={k}, Mode={mode}) from {os.path.basename(pt_path)}...")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    layers = _load_telemetry_layers(pt_path, mode=mode)

    hm: Dict[int, List[int]] = {}

    per_layer_meta: List[Dict[str, Any]] = []
    global_selected = 0.0
    global_total = 0.0
    for layer_idx, data_tensor in layers.items():
        top_indices = torch.topk(data_tensor.float(), k=min(k, data_tensor.numel())).indices.tolist()
        hm[int(layer_idx)] = top_indices
        scores = data_tensor.float()
        total = float(scores.sum().item())
        selected = float(scores[top_indices].sum().item()) if top_indices else 0.0
        cov = (selected / total) if total > 0 else 0.0
        global_selected += selected
        global_total += total
        per_layer_meta.append(
            {
                "layer": int(layer_idx),
                "k": int(len(top_indices)),
                "coverage": float(cov),
                "selected_experts": [int(x) for x in top_indices],
            }
        )

    if out_json is None:
        dir_name = os.path.dirname(pt_path)
        base_name = os.path.splitext(os.path.basename(pt_path))[0]
        if mode == "counts":
            # Keep historical filename for backward compatibility.
            out_json = os.path.join(dir_name, f"{base_name}_hotmap_k{k}.json")
        else:
            out_json = os.path.join(dir_name, f"{base_name}_hotmap_{mode}_k{k}.json")

    with open(out_json, "w") as f:
        json.dump({str(l): exps for l, exps in hm.items()}, f, indent=2)

    ks = [len(v) for v in hm.values()]
    coverages = [float(row["coverage"]) for row in per_layer_meta]
    meta = {
        "schema_version": 1,
        "source_telemetry": pt_path,
        "hotmap_json": out_json,
        "method": "topk",
        "score_mode": mode,
        "constraints": {"k": int(k)},
        "layers": int(len(ks)),
        "k_stats": {
            "min": int(min(ks)) if ks else 0,
            "mean": float(sum(ks) / len(ks)) if ks else 0.0,
            "max": int(max(ks)) if ks else 0,
            "uniform": bool(len(set(ks)) == 1) if ks else True,
            "total_slots": int(sum(ks)) if ks else 0,
        },
        "coverage_stats": {
            "min": float(min(coverages)) if coverages else 0.0,
            "mean": float(sum(coverages) / len(coverages)) if coverages else 0.0,
            "max": float(max(coverages)) if coverages else 0.0,
        },
        "global_coverage": float(global_selected / global_total) if global_total > 0 else 0.0,
        "per_layer": per_layer_meta,
    }
    meta_path = _write_hotmap_metadata(out_json, meta)

    print(f"Hotmap saved: {out_json}")
    print(f"Hotmap metadata saved: {meta_path}")
    return out_json


def build_hotmap_by_coverage(
    pt_path: str,
    coverage_pct: float = 60.0,
    out_json: Optional[str] = None,
    mode: str = "counts",
    min_k: Optional[int] = None,
    max_k: Optional[int] = None,
) -> str:
    """
    Builds a per-layer dynamic-k hotmap where each layer uses the minimum number of
    experts needed to reach `coverage_pct` routing mass.
    """
    if not (0.0 < coverage_pct <= 100.0):
        raise ValueError(f"coverage_pct must be in (0, 100], got {coverage_pct}")
    if min_k is not None and min_k <= 0:
        raise ValueError(f"min_k must be > 0, got {min_k}")
    if max_k is not None and max_k <= 0:
        raise ValueError(f"max_k must be > 0, got {max_k}")
    if min_k is not None and max_k is not None and min_k > max_k:
        raise ValueError(f"min_k ({min_k}) cannot exceed max_k ({max_k})")

    print(
        "Building Coverage Hotmap "
        f"(coverage={coverage_pct:.2f}%, mode={mode}, min_k={min_k}, max_k={max_k}) "
        f"from {os.path.basename(pt_path)}..."
    )

    layers = _load_telemetry_layers(pt_path, mode=mode)
    hm: Dict[int, List[int]] = {}
    k_values: List[int] = []
    target = coverage_pct / 100.0

    per_layer_meta: List[Dict[str, Any]] = []
    global_selected = 0.0
    global_total = 0.0
    for layer_idx, data_tensor in layers.items():
        scores = data_tensor.float()
        num_experts = int(scores.numel())

        vals, idx = torch.sort(scores, descending=True)
        total = vals.sum().item()
        if total <= 0:
            k_layer = 1
            picks = [int(idx[0].item())]
        else:
            cumsum = torch.cumsum(vals / total, dim=0)
            meets = (cumsum >= target).nonzero(as_tuple=True)[0]
            k_layer = int(meets[0].item()) + 1 if len(meets) > 0 else num_experts

            if min_k is not None:
                k_layer = max(k_layer, int(min_k))
            if max_k is not None:
                k_layer = min(k_layer, int(max_k))
            k_layer = max(1, min(k_layer, num_experts))

            picks = idx[:k_layer].tolist()

        hm[int(layer_idx)] = sorted(int(x) for x in picks)
        k_values.append(len(hm[int(layer_idx)]))
        total = float(scores.sum().item())
        selected = float(scores[hm[int(layer_idx)]].sum().item()) if hm[int(layer_idx)] else 0.0
        cov = (selected / total) if total > 0 else 0.0
        global_selected += selected
        global_total += total
        per_layer_meta.append(
            {
                "layer": int(layer_idx),
                "k": int(len(hm[int(layer_idx)])),
                "coverage": float(cov),
                "selected_experts": [int(x) for x in hm[int(layer_idx)]],
            }
        )

    if out_json is None:
        dir_name = os.path.dirname(pt_path)
        base_name = os.path.splitext(os.path.basename(pt_path))[0]
        cov_tag = _coverage_tag(coverage_pct)
        clamp_tag = ""
        if min_k is not None or max_k is not None:
            clamp_tag = f"_min{min_k if min_k is not None else 'na'}_max{max_k if max_k is not None else 'na'}"
        out_json = os.path.join(dir_name, f"{base_name}_hotmap_{mode}_cov{cov_tag}{clamp_tag}.json")

    with open(out_json, "w") as f:
        json.dump({str(l): exps for l, exps in hm.items()}, f, indent=2)

    coverages = [float(row["coverage"]) for row in per_layer_meta]
    meta = {
        "schema_version": 1,
        "source_telemetry": pt_path,
        "hotmap_json": out_json,
        "method": "coverage",
        "score_mode": mode,
        "constraints": {
            "coverage_pct": float(coverage_pct),
            "min_k": int(min_k) if min_k is not None else None,
            "max_k": int(max_k) if max_k is not None else None,
        },
        "layers": int(len(k_values)),
        "k_stats": {
            "min": int(min(k_values)) if k_values else 0,
            "mean": float(sum(k_values) / len(k_values)) if k_values else 0.0,
            "max": int(max(k_values)) if k_values else 0,
            "uniform": bool(len(set(k_values)) == 1) if k_values else True,
            "total_slots": int(sum(k_values)) if k_values else 0,
        },
        "coverage_stats": {
            "min": float(min(coverages)) if coverages else 0.0,
            "mean": float(sum(coverages) / len(coverages)) if coverages else 0.0,
            "max": float(max(coverages)) if coverages else 0.0,
        },
        "global_coverage": float(global_selected / global_total) if global_total > 0 else 0.0,
        "per_layer": per_layer_meta,
    }
    meta_path = _write_hotmap_metadata(out_json, meta)

    print(
        f"Hotmap saved: {out_json} | "
        f"k stats -> min={min(k_values)}, mean={sum(k_values)/len(k_values):.2f}, max={max(k_values)}"
    )
    print(f"Hotmap metadata saved: {meta_path}")
    return out_json


def build_hotmap_by_budget(
    pt_path: str,
    budget_per_layer: Optional[float] = None,
    total_budget: Optional[int] = None,
    out_json: Optional[str] = None,
    mode: str = "counts",
    min_k: Optional[int] = 1,
    max_k: Optional[int] = None,
) -> str:
    """
    Builds a per-layer dynamic-k hotmap under an exact global budget.

    Allocation objective:
    - maximize covered routing score with fixed total routed slots
    - score is defined by `mode` ('counts' or 'mass')
    - greedy by marginal gain per extra expert (optimal for this separable monotone objective)
    """
    layers = _load_telemetry_layers(pt_path, mode=mode)
    layer_ids = sorted(layers.keys())
    if not layer_ids:
        raise ValueError(f"No layers found in telemetry: {pt_path}")

    n_layers = len(layer_ids)
    n_experts = int(layers[layer_ids[0]].numel())
    if any(int(t.numel()) != n_experts for t in layers.values()):
        raise ValueError("All layers must have the same number of experts for fixed-budget allocation.")

    k_min = int(min_k) if min_k is not None else 1
    k_max = int(max_k) if max_k is not None else n_experts
    if k_min < 0:
        raise ValueError(f"min_k must be >= 0, got {k_min}")
    if k_max <= 0:
        raise ValueError(f"max_k must be > 0, got {k_max}")
    if k_min > k_max:
        raise ValueError(f"min_k ({k_min}) cannot exceed max_k ({k_max})")
    if k_max > n_experts:
        k_max = n_experts

    if total_budget is None:
        if budget_per_layer is None:
            raise ValueError("Provide either total_budget or budget_per_layer for fixed-budget mode.")
        total_budget = int(round(float(budget_per_layer) * n_layers))
    else:
        total_budget = int(total_budget)

    if total_budget <= 0:
        raise ValueError(f"total_budget must be > 0, got {total_budget}")

    min_total = n_layers * k_min
    max_total = n_layers * k_max
    if total_budget < min_total or total_budget > max_total:
        raise ValueError(
            f"total_budget={total_budget} out of feasible range [{min_total}, {max_total}] "
            f"for n_layers={n_layers}, min_k={k_min}, max_k={k_max}"
        )

    print(
        "Building Fixed-Budget Hotmap "
        f"(mode={mode}, total_budget={total_budget}, avg_budget={total_budget / n_layers:.2f}, "
        f"min_k={k_min}, max_k={k_max}) from {os.path.basename(pt_path)}..."
    )

    # Pre-sort once per layer.
    sorted_idx: Dict[int, torch.Tensor] = {}
    marginal_gains: Dict[int, List[float]] = {}
    for li in layer_ids:
        vals, idx = torch.sort(layers[li].float(), descending=True)
        sorted_idx[li] = idx
        total = float(vals.sum().item())
        if total <= 0:
            gains = [0.0] * int(vals.numel())
        else:
            gains = (vals / total).tolist()
        marginal_gains[li] = gains

    # Initialize all layers at k_min and allocate the remaining slots greedily.
    k_alloc = {li: k_min for li in layer_ids}
    used = n_layers * k_min
    remaining = total_budget - used

    # Max-heap by next marginal gain: (-gain, layer_id)
    heap: List[tuple] = []
    for li in layer_ids:
        if k_alloc[li] < k_max:
            next_gain = marginal_gains[li][k_alloc[li]]
            heapq.heappush(heap, (-float(next_gain), li))

    while remaining > 0:
        if not heap:
            break
        neg_gain, li = heapq.heappop(heap)
        if k_alloc[li] >= k_max:
            continue
        k_alloc[li] += 1
        remaining -= 1
        if k_alloc[li] < k_max:
            next_gain = marginal_gains[li][k_alloc[li]]
            heapq.heappush(heap, (-float(next_gain), li))

    if remaining != 0:
        raise RuntimeError(
            f"Failed to allocate full budget: remaining={remaining}, used={total_budget - remaining}, total={total_budget}"
        )

    hm: Dict[int, List[int]] = {}
    k_values: List[int] = []
    per_layer_meta: List[Dict[str, Any]] = []
    global_selected = 0.0
    global_total = 0.0
    for li in layer_ids:
        k_layer = int(k_alloc[li])
        picks = sorted(int(x) for x in sorted_idx[li][:k_layer].tolist())
        hm[int(li)] = picks
        k_values.append(k_layer)
        scores = layers[li].float()
        total = float(scores.sum().item())
        selected = float(scores[picks].sum().item()) if picks else 0.0
        cov = (selected / total) if total > 0 else 0.0
        global_selected += selected
        global_total += total
        per_layer_meta.append(
            {
                "layer": int(li),
                "k": int(k_layer),
                "coverage": float(cov),
                "selected_experts": [int(x) for x in picks],
            }
        )

    if out_json is None:
        dir_name = os.path.dirname(pt_path)
        base_name = os.path.splitext(os.path.basename(pt_path))[0]
        avg_tag = _coverage_tag(total_budget / n_layers)
        clamp_tag = ""
        if min_k is not None or max_k is not None:
            clamp_tag = f"_min{min_k if min_k is not None else 'na'}_max{max_k if max_k is not None else 'na'}"
        out_json = os.path.join(
            dir_name,
            f"{base_name}_hotmap_{mode}_budget{total_budget}_kavg{avg_tag}{clamp_tag}.json",
        )

    with open(out_json, "w") as f:
        json.dump({str(l): exps for l, exps in hm.items()}, f, indent=2)

    coverages = [float(row["coverage"]) for row in per_layer_meta]
    meta = {
        "schema_version": 1,
        "source_telemetry": pt_path,
        "hotmap_json": out_json,
        "method": "budget",
        "score_mode": mode,
        "constraints": {
            "budget_per_layer_requested": float(budget_per_layer) if budget_per_layer is not None else None,
            "total_budget": int(total_budget),
            "min_k": int(min_k) if min_k is not None else None,
            "max_k": int(max_k) if max_k is not None else None,
        },
        "layers": int(len(k_values)),
        "k_stats": {
            "min": int(min(k_values)) if k_values else 0,
            "mean": float(sum(k_values) / len(k_values)) if k_values else 0.0,
            "max": int(max(k_values)) if k_values else 0,
            "uniform": bool(len(set(k_values)) == 1) if k_values else True,
            "total_slots": int(sum(k_values)) if k_values else 0,
        },
        "coverage_stats": {
            "min": float(min(coverages)) if coverages else 0.0,
            "mean": float(sum(coverages) / len(coverages)) if coverages else 0.0,
            "max": float(max(coverages)) if coverages else 0.0,
        },
        "global_coverage": float(global_selected / global_total) if global_total > 0 else 0.0,
        "per_layer": per_layer_meta,
    }
    meta_path = _write_hotmap_metadata(out_json, meta)

    print(
        f"Hotmap saved: {out_json} | "
        f"k stats -> min={min(k_values)}, mean={sum(k_values)/len(k_values):.2f}, max={max(k_values)}"
    )
    print(f"Hotmap metadata saved: {meta_path}")
    return out_json


def _hotmap_meta_path(out_json: str) -> str:
    if out_json.endswith(".json"):
        return out_json[:-5] + ".meta.json"
    return out_json + ".meta.json"


def _write_hotmap_metadata(out_json: str, payload: Dict[str, Any]) -> str:
    meta_path = _hotmap_meta_path(out_json)
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2)
    return meta_path


def _coverage_tag(coverage_pct: float) -> str:
    s = f"{coverage_pct:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _load_telemetry_layers(pt_path: str, mode: str) -> Dict[int, torch.Tensor]:
    if mode not in ("counts", "mass"):
        raise ValueError("Mode must be 'counts' or 'mass'")

    d = torch.load(pt_path, map_location="cpu")
    layer_objs = {k: v for k, v in d.items() if isinstance(k, int)}
    if not layer_objs:
        raise ValueError(f"No layer tensors found in telemetry: {pt_path}")

    layers: Dict[int, torch.Tensor] = {}
    for layer_idx in sorted(layer_objs.keys()):
        obj = layer_objs[layer_idx]
        if mode not in obj or obj[mode] is None:
            raise ValueError(f"Layer {layer_idx} missing '{mode}' tensor in {pt_path}")
        data_tensor = obj[mode]
        if data_tensor.dim() == 2:
            data_tensor = data_tensor.sum(dim=0)
        if data_tensor.dim() != 1:
            raise ValueError(
                f"Expected 1-D or 2-D tensor for layer {layer_idx}/{mode}, got shape {tuple(data_tensor.shape)}"
            )
        layers[layer_idx] = data_tensor
    return layers
