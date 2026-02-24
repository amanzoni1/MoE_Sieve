import os
import json
import torch
from tqdm.auto import tqdm
from typing import Optional, List, Dict
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

    for layer_idx, data_tensor in layers.items():
        top_indices = torch.topk(data_tensor.float(), k=min(k, data_tensor.numel())).indices.tolist()
        hm[int(layer_idx)] = top_indices

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

    print(f"Hotmap saved: {out_json}")
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

    print(
        f"Hotmap saved: {out_json} | "
        f"k stats -> min={min(k_values)}, mean={sum(k_values)/len(k_values):.2f}, max={max(k_values)}"
    )
    return out_json


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
