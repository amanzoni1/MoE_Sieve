import os
import json
from typing import List, Dict, Optional


# Target Name Generators (OLMoE/Mixtral specific)
ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
EXPERT_PROJS = ["gate_proj", "up_proj", "down_proj"]

def targets_attention(num_layers: int) -> List[str]:
    """Returns attention module names for all layers."""
    return [f"model.layers.{l}.self_attn.{p}" for l in range(num_layers) for p in ATTN_PROJS]

def targets_router_gates(num_layers: int) -> List[str]:
    """Returns router gate module names for all layers."""
    return [f"model.layers.{l}.mlp.gate" for l in range(num_layers)]

def targets_all_experts(num_layers: int, num_experts: int) -> List[str]:
    """Returns ALL expert module names (for Full LoRA)."""
    return [
        f"model.layers.{l}.mlp.experts.{e}.{p}"
        for l in range(num_layers)
        for e in range(num_experts)
        for p in EXPERT_PROJS
    ]

def targets_hot_experts(hot_map: Dict[str, List[int]]) -> List[str]:
    """Returns ONLY the specific experts listed in the hot_map."""
    targets = []
    for layer_str, experts in hot_map.items():
        layer_idx = int(layer_str)
        for e in experts:
            for p in EXPERT_PROJS:
                targets.append(f"model.layers.{layer_idx}.mlp.experts.{e}.{p}")
    return targets


# Helpers & Validators
def load_hotmap(json_path: str) -> Dict[str, List[int]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Hotmap file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    # Ensure keys are strings for consistency
    return {str(k): v for k, v in data.items()}

def infer_hot_k(hotmap_json: Optional[str]) -> Optional[int]:
    """Reads the hotmap to find the K value (e.g., 8, 16)."""
    if not hotmap_json:
        return None
    try:
        hm = load_hotmap(hotmap_json)
        # Check the first layer to guess K
        first_layer_experts = next(iter(hm.values()))
        return len(first_layer_experts)
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

# Main Entry Point
def get_targets(model, mode: str, hotmap_json: Optional[str] = None) -> List[str]:
    """
    The Master Function.
    Args:
        model: The HF model object (to read config/layers)
        mode: 'hot' or 'full'
        hotmap_json: Path to .json file (required if mode='hot')
    """
    # Auto-detect architecture stats
    if hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.layers

    num_layers = len(layers)
    try:
        num_experts = len(layers[0].mlp.experts)
    except AttributeError:
        num_experts = getattr(model.config, "num_experts", 64)

    print(f"🎯 Building targets for mode='{mode}' (L={num_layers}, E={num_experts})...")

    # 1. Always target Attention & Routers
    targets = []
    targets += targets_attention(num_layers)
    targets += targets_router_gates(num_layers)

    # 2. Add Experts based on Mode
    if mode == "full":
        targets += targets_all_experts(num_layers, num_experts)
        print(f"   + Added ALL experts (Full LoRA)")

    elif mode == "hot":
        if not hotmap_json:
            raise ValueError("Mode 'hot' requires a valid 'hotmap_json' path.")
        hot_map = load_hotmap(hotmap_json)
        expert_targets = targets_hot_experts(hot_map)
        targets += expert_targets

        # Calculate stats for logging
        total_slots = num_layers * num_experts
        active_slots = sum(len(v) for v in hot_map.values())
        print(f"   + Added HOT experts from map: {hotmap_json}")
        print(f"   + Active Experts: {active_slots} / {total_slots} ({active_slots/total_slots:.1%})")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 3. Validate against actual model
    final_targets = validate_targets(model, targets)
    return final_targets
