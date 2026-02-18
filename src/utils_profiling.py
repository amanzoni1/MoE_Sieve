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
    n_samples: Optional[int] = None,
    bs: int = 16,
    seq_len: int = 2048,
    bucket_edges: Optional[List[int]] = None,
    store_mass: bool = True,
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
    raw_ds = raw_ds.shuffle(seed=seed)

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

    suffix = "bucketed" if bucket_edges is not None else "global"
    count_str = f"n{selected_n}" if n_samples is not None else "full"
    filename = f"telemetry_{dataset_key}_{target_split}_{count_str}_{run_name}_{suffix}.pt"
    pt_path = os.path.join(output_dir, filename)

    payload = {
        "meta": {
            "dataset": dataset_key,
            "split": target_split,
            "seed": seed,
            "n_requested": n_samples,
            "n_samples": profiled_examples,
            "n_selected": selected_n,
            "n_skipped_short": skipped_short,
            "seq_len": seq_len,
            "layers": eng.num_layers,
            "experts": eng.num_experts,
            "k": eng.top_k
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

    d = torch.load(pt_path, map_location="cpu")
    layers = {k: v for k, v in d.items() if isinstance(k, int)}

    if mode not in ("counts", "mass"):
        raise ValueError("Mode must be 'counts' or 'mass'")

    hm: Dict[int, List[int]] = {}

    for layer_idx, obj in layers.items():
        data_tensor = obj[mode]

        if data_tensor.dim() == 2:
            data_tensor = data_tensor.sum(dim=0)

        top_indices = torch.topk(data_tensor.float(), k=k).indices.tolist()
        hm[int(layer_idx)] = top_indices

    if out_json is None:
        dir_name = os.path.dirname(pt_path)
        base_name = os.path.splitext(os.path.basename(pt_path))[0]
        out_json = os.path.join(dir_name, f"{base_name}_hotmap_k{k}.json")

    with open(out_json, "w") as f:
        json.dump({str(l): exps for l, exps in hm.items()}, f, indent=2)

    print(f"Hotmap saved: {out_json}")
    return out_json
