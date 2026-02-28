import os
import json
import random
from typing import Any, List, Dict, Optional


# Target Name Generators (OLMoE/Mixtral specific)
ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
EXPERT_PROJS = ["gate_proj", "up_proj", "down_proj"]
FULL_TARGET_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "gate_proj", "up_proj", "down_proj"]

def targets_attention(num_layers: int) -> List[str]:
    """Returns attention module names for all layers."""
    return [f"model.layers.{l}.self_attn.{p}" for l in range(num_layers) for p in ATTN_PROJS]

def targets_router_gates(num_layers: int) -> List[str]:
    """Returns router gate module names for all layers."""
    return [f"model.layers.{l}.mlp.gate" for l in range(num_layers)]

def targets_hot_experts(hot_map: Dict[str, List[int]]) -> List[str]:
    """Returns ONLY the specific experts listed in the hot_map."""
    targets = []
    for layer_str, experts in hot_map.items():
        layer_idx = int(layer_str)
        for e in experts:
            for p in EXPERT_PROJS:
                targets.append(f"model.layers.{layer_idx}.mlp.experts.{e}.{p}")
    return targets


def full_target_suffixes() -> List[str]:
    """Compact target list for full LoRA via suffix matching in PEFT."""
    return list(FULL_TARGET_SUFFIXES)


def build_random_hotmap(num_layers: int, num_experts: int, k: int, seed: int) -> Dict[str, List[int]]:
    """Build deterministic random top-k experts per layer."""
    if k <= 0:
        raise ValueError(f"random k must be > 0, got {k}")
    if k > num_experts:
        raise ValueError(f"random k={k} exceeds num_experts={num_experts}")

    rng = random.Random(seed)
    out: Dict[str, List[int]] = {}
    for layer_idx in range(num_layers):
        picks = sorted(rng.sample(range(num_experts), k))
        out[str(layer_idx)] = picks
    return out


# Helpers & Validators
def load_hotmap(json_path: str) -> Dict[str, List[int]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Hotmap file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    # Ensure keys are strings and expert ids are int lists for consistency.
    norm: Dict[str, List[int]] = {}
    for k, v in data.items():
        if not isinstance(v, list):
            raise ValueError(f"Hotmap layer '{k}' must map to a list of expert ids.")
        norm[str(k)] = [int(x) for x in v]
    return norm

def infer_hot_k(hotmap_json: Optional[str]) -> Optional[int]:
    """Reads the hotmap to find the K value (e.g., 8, 16)."""
    if not hotmap_json:
        return None
    try:
        hm = load_hotmap(hotmap_json)
        if not hm:
            return None
        k_set = {len(experts) for experts in hm.values()}
        return next(iter(k_set)) if len(k_set) == 1 else None
    except Exception:
        return None


def infer_hotmap_stats(hotmap_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if not hotmap_json:
        return None
    try:
        hm = load_hotmap(hotmap_json)
        if not hm:
            return None
        ks = [len(v) for v in hm.values()]
        active_slots = sum(ks)
        return {
            "layers": len(ks),
            "active_slots": active_slots,
            "k_min": min(ks),
            "k_max": max(ks),
            "k_mean": float(active_slots) / float(len(ks)),
            "k_uniform": len(set(ks)) == 1,
        }
    except Exception:
        return None


def validate_targets(model, targets: List[str]) -> List[str]:
    """Filters out targets that don't strictly exist in the model."""
    existing_modules = set(n for n, _ in model.named_modules())
    kept = [t for t in targets if t in existing_modules]

    if not kept:
        raise RuntimeError("No targets matched! Check your model architecture vs. naming logic.")

    missing = len(targets) - len(kept)
    if missing > 0:
        print(f"[Target Selector] Filtered out {missing} invalid targets (kept {len(kept)}).")

    return kept


def expected_full_linear_module_count(num_layers: int, num_experts: int) -> int:
    # 4 attn projections + 1 router gate + 3 expert projections per expert.
    return num_layers * (len(ATTN_PROJS) + 1 + (len(EXPERT_PROJS) * num_experts))


def match_linear_modules_by_suffix(model, suffixes: List[str]):
    matched = []
    for name, module in model.named_modules():
        if not (hasattr(module, "in_features") and hasattr(module, "out_features")):
            continue
        if any(name.endswith(f".{suffix}") for suffix in suffixes):
            matched.append(module)
    return matched


def full_target_sanity(
    model,
    suffixes: List[str],
    num_layers: int,
    num_experts: int,
    lora_rank: int,
    strict: bool = True,
):
    matched_linear_modules = match_linear_modules_by_suffix(model, suffixes)
    matched_count = len(matched_linear_modules)
    expected_modules = None
    if strict:
        expected_modules = expected_full_linear_module_count(num_layers, num_experts)
        if matched_count != expected_modules:
            raise RuntimeError(
                f"Full-LoRA sanity check failed: matched {matched_count} modules, expected {expected_modules}. "
                "This would break comparability."
            )
    else:
        print(
            "[Full-LoRA sanity] Non-strict mode: skipping fixed expected module count "
            f"(matched {matched_count})."
        )

    # LoRA trainable params per linear = r * (in_features + out_features) when bias='none'.
    expected_trainable_params = sum(
        lora_rank * (int(m.in_features) + int(m.out_features))
        for m in matched_linear_modules
    )
    return {
        "matched_count": matched_count,
        "expected_modules": expected_modules,
        "expected_trainable_params": expected_trainable_params,
    }

# Main Entry Point
def get_targets(
    model,
    mode: str,
    hotmap_json: Optional[str] = None,
    random_k: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> List[str]:
    """
    The Master Function.
    Args:
        model: The HF model object (to read config/layers)
        mode: 'hot', 'dyn', 'random', or 'full'
        hotmap_json: Path to .json file (required if mode in {'hot','dyn'})
    """
    # Auto-detect architecture stats
    if hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.layers

    num_layers = len(layers)
    num_experts = getattr(model.config, "num_experts", 64)

    print(f"Building targets for mode='{mode}' (L={num_layers}, E={num_experts})...")

    if mode == "full":
        targets = full_target_suffixes()
        print(f"   + Attention targets: {len(ATTN_PROJS) * num_layers}")
        print(f"   + Router gate targets: {num_layers}")
        print(f"   + Expert targets (FULL): {num_layers * num_experts * len(EXPERT_PROJS)}")
        print(f"   + Target suffixes (FULL): {len(targets)}")
        return targets

    # Always target Attention & Routers
    attn_targets = targets_attention(num_layers)
    gate_targets = targets_router_gates(num_layers)
    targets = attn_targets + gate_targets

    # Add Experts based on Mode

    if mode in ("hot", "dyn"):
        if not hotmap_json:
            raise ValueError(f"Mode '{mode}' requires a valid 'hotmap_json' path.")

        hot_map = load_hotmap(hotmap_json)
        expert_targets = targets_hot_experts(hot_map)
        targets += expert_targets

        print(f"   + Attention targets: {len(attn_targets)}")
        print(f"   + Router gate targets: {len(gate_targets)}")
        if mode == "dyn":
            print(f"   + Expert targets (DYN): {len(expert_targets)}")
        else:
            print(f"   + Expert targets (HOT): {len(expert_targets)}")

        # Hotmap intent stats (based on JSON content)
        total_slots = num_layers * num_experts
        active_slots = sum(len(v) for v in hot_map.values())
        print(f"   + Hotmap file: {hotmap_json}")
        print(f"   + Active Experts (from hotmap): {active_slots} / {total_slots} ({active_slots/total_slots:.1%})")

    elif mode == "random":
        if random_k is None:
            raise ValueError("Mode 'random' requires random_k (usually from --k).")
        seed_eff = int(random_seed or 0)
        hot_map = build_random_hotmap(num_layers, num_experts, int(random_k), seed_eff)
        expert_targets = targets_hot_experts(hot_map)
        targets += expert_targets

        print(f"   + Attention targets: {len(attn_targets)}")
        print(f"   + Router gate targets: {len(gate_targets)}")
        print(f"   + Expert targets (RANDOM): {len(expert_targets)}")
        print(f"   + Random seed: {seed_eff}")
        print(f"   + Random k: {int(random_k)}")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Validate against actual model
    final_targets = validate_targets(model, targets)
    return final_targets
