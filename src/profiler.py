import os
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional


class ProfilerEngine:
    """
    Profiles MoE routing from gate logits.

    Responsibilities:
      1. Register Hooks
      2. Capture Logits -> Compute TopK
      3. Store Counts & Mass in Buffer

    This class does NOT handle Data Loading or Saving to disk.
    That is handled by 'utils_profiling.py'.
    """

    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str,
        seq_len: int = 2048,
        bucket_edges: Optional[List[int]] = None,
        gate_getter: Optional[Callable[[Any], Any]] = None,
        gate_path: Optional[str] = None,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
        store_mass: bool = True,
        prob_dtype: torch.dtype = torch.float32,
        renorm_topk_prob: Optional[bool] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.seq_len = int(seq_len)

        # Layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        elif hasattr(model, "layers"):
            self.layers = model.layers
        else:
            raise ValueError("Could not automatically identify model layers.")

        self.num_layers = len(self.layers)
        self.gate_path = gate_path
        self.profiled_layer_indices: List[int] = []

        # Gate module resolution
        if gate_getter is not None:
            self.gate_getter = gate_getter
        else:
            self.gate_path = self.gate_path or self._auto_detect_gate_path_any_layer()
            self.gate_getter = lambda layer: self._resolve_attr_path(layer, self.gate_path)

        first_gate = self._find_first_gate_module()
        gate_out_features = getattr(first_gate, "out_features", None)

        # Router cardinality (experts)
        cfg_experts = self._get_config_first_int(
            ["num_experts", "num_local_experts", "n_routed_experts", "n_experts"]
        )
        if num_experts is not None:
            self.num_experts = int(num_experts)
        elif cfg_experts is not None:
            self.num_experts = int(cfg_experts)
        elif gate_out_features is not None:
            self.num_experts = int(gate_out_features)
        else:
            raise ValueError(
                "Could not infer num_experts from model config or gate module. "
                "Pass num_experts explicitly."
            )

        # Experts selected per token (top-k)
        cfg_top_k = self._get_config_first_int(
            [
                "num_experts_per_tok",
                "num_experts_per_token",
                "num_selected_experts",
                "moe_top_k",
                "router_topk",
                "top_k",
                "num_selects",
            ]
        )
        if top_k is not None:
            self.top_k = int(top_k)
        elif cfg_top_k is not None:
            self.top_k = int(cfg_top_k)
        else:
            # Conservative fallback for unsupported configs.
            self.top_k = min(8, self.num_experts)

        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError(
                f"Invalid top_k={self.top_k} for num_experts={self.num_experts}. "
                "Pass --top_k/--num_experts explicitly."
            )

        self.store_mass = bool(store_mass)
        self.prob_dtype = prob_dtype

        # Buckets
        self.bucket_edges = bucket_edges
        if self.bucket_edges is not None:
            if self.bucket_edges[0] != 0:
                raise ValueError("bucket_edges must start at 0.")
            if self.bucket_edges[-1] < self.seq_len:
                raise ValueError("bucket_edges[-1] must be >= seq_len.")
            self.num_buckets = len(self.bucket_edges) - 1

            # NEW: warn on left padding (bucket positions become meaningless)
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                print("   [Profiler] tokenizer.padding_side='left' detected.")
                print("   Positional bucketing will be skewed; recommended: tokenizer.padding_side='right'.")
        else:
            self.num_buckets = 0

        self._current_attn_mask: Optional[torch.Tensor] = None
        self.hooks = []
        self.data_buffer: Dict[int, Dict[str, Any]] = {}

        # NEW: auto-detect renorm_topk_prob (affects mass only, not indices)
        if renorm_topk_prob is None:
            try:
                mlp = getattr(self.layers[0], "mlp", None)
                renorm_topk_prob = bool(getattr(mlp, "norm_topk_prob", False))
            except Exception:
                renorm_topk_prob = False
        self.renorm_topk_prob = bool(renorm_topk_prob)

        print(
            f"[Profiler] gate_path={self.gate_path or '<custom_getter>'} "
            f"num_experts={self.num_experts} top_k={self.top_k}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def _resolve_attr_path(self, obj: Any, path: str) -> Any:
        cur = obj
        for part in path.split("."):
            if not part:
                continue
            if part.isdigit():
                idx = int(part)
                cur = cur[idx]
                continue
            if not hasattr(cur, part):
                raise AttributeError(f"Path '{path}' not found at '{part}'")
            cur = getattr(cur, part)
        return cur

    def _auto_detect_gate_path(self, first_layer: Any) -> str:
        # Common gate locations across OLMoE/Mixtral-like implementations.
        candidates = [
            "mlp.gate",
            "block_sparse_moe.gate",
            "block_sparse_moe.router",
            "moe.gate",
            "router.gate",
            "router",
        ]
        for path in candidates:
            try:
                mod = self._resolve_attr_path(first_layer, path)
                if hasattr(mod, "register_forward_hook"):
                    return path
            except Exception:
                continue

        # Fallback scan by name.
        for name, mod in first_layer.named_modules():
            lname = name.lower()
            if not lname:
                continue
            # Skip FFN projection names; these are not MoE router gates.
            if lname.endswith("gate_proj") or lname.endswith("up_proj") or lname.endswith("down_proj"):
                continue
            if ("gate" in lname or "router" in lname) and hasattr(mod, "register_forward_hook"):
                return name

        raise ValueError(
            "Could not auto-detect MoE gate module path. Pass gate_path explicitly "
            "(examples: 'mlp.gate', 'block_sparse_moe.gate')."
        )

    def _auto_detect_gate_path_any_layer(self) -> str:
        for layer in self.layers:
            try:
                return self._auto_detect_gate_path(layer)
            except Exception:
                continue
        raise ValueError(
            "Could not auto-detect MoE gate module path on any layer. Pass gate_path explicitly "
            "(examples: 'mlp.gate', 'block_sparse_moe.gate')."
        )

    def _find_first_gate_module(self) -> Any:
        for layer in self.layers:
            try:
                gate = self.gate_getter(layer)
                if hasattr(gate, "register_forward_hook"):
                    return gate
            except Exception:
                continue
        raise ValueError(
            "Could not resolve a valid MoE gate module on any layer with current gate getter/path."
        )

    def _get_config_first_int(self, keys: List[str]) -> Optional[int]:
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return None
        for key in keys:
            val = getattr(cfg, key, None)
            if isinstance(val, (int, float)):
                val_i = int(val)
                if val_i > 0:
                    return val_i
        return None

    def _init_buffer(self):
        if self.bucket_edges is None:
            self.data_buffer = {
                i: {
                    "counts": torch.zeros(self.num_experts, dtype=torch.long),
                    "mass": torch.zeros(self.num_experts, dtype=torch.float32) if self.store_mass else None,
                    "total": 0,
                }
                for i in range(self.num_layers)
            }
        else:
            self.data_buffer = {
                i: {
                    "counts": torch.zeros(self.num_buckets, self.num_experts, dtype=torch.long),
                    "mass": torch.zeros(self.num_buckets, self.num_experts, dtype=torch.float32) if self.store_mass else None,
                    "total": [0 for _ in range(self.num_buckets)],
                }
                for i in range(self.num_layers)
            }

    def _get_hook(self, layer_idx: int):
        def hook(module, input, output):
            logits = output[0] if isinstance(output, tuple) else output  # gate logits

            if self._current_attn_mask is None:
                raise RuntimeError("attention_mask not set; cannot mask padding.")

            B, S = self._current_attn_mask.shape
            # In multi-GPU device_map="auto" runs, different layers can execute
            # on different CUDA devices. Keep mask on the same device as logits.
            m = self._current_attn_mask.to(device=logits.device, dtype=torch.bool)  # [B,S]

            # NEW: safer universal flattening (still supports 2D or 3D logits)
            if logits.dim() == 3:
                logits_flat = logits.reshape(-1, logits.shape[-1])  # [B*S, E]
            elif logits.dim() == 2:
                logits_flat = logits  # [B*S, E]
            else:
                raise RuntimeError(f"Unexpected gate output shape: {tuple(logits.shape)}")

            if logits_flat.shape[1] != self.num_experts:
                raise RuntimeError(f"Gate last-dim {logits_flat.shape[1]} != num_experts {self.num_experts}")
            if logits_flat.shape[0] != B * S:
                raise RuntimeError(f"Gate tokens {logits_flat.shape[0]} != B*S {B*S} (cannot align)")

            flat_mask = m.reshape(-1)          # [B*S]
            logits2d = logits_flat[flat_mask]  # [N,E] (non-pad tokens)
            if logits2d.numel() == 0:
                return

            # Exact HF routing rule: softmax -> topk
            probs2d = F.softmax(logits2d.to(self.prob_dtype), dim=-1)  # [N,E] float32
            top_p, idx = torch.topk(probs2d, k=self.top_k, dim=-1)      # [N,K], [N,K]

            # match HF norm_topk_prob for mass (indices unchanged)
            if self.renorm_topk_prob:
                top_p = top_p / top_p.sum(dim=-1, keepdim=True)

            # counts (hit frequency)
            flat_idx_cpu = idx.reshape(-1).to(torch.long).cpu()
            counts = torch.bincount(flat_idx_cpu, minlength=self.num_experts)

            if self.bucket_edges is None:
                self.data_buffer[layer_idx]["counts"] += counts
                self.data_buffer[layer_idx]["total"] += int(idx.shape[0]) * self.top_k

                # mass (probability mass, top-k only)
                if self.store_mass:
                    mass = torch.zeros(self.num_experts, dtype=torch.float32)
                    mass.index_add_(0, flat_idx_cpu, top_p.reshape(-1).to(torch.float32).cpu())
                    self.data_buffer[layer_idx]["mass"] += mass

            else:
                # bucket by token position
                pos = torch.arange(S, device=logits.device).view(1, S).expand(B, S)  # [B,S]
                pos2d = pos[m]  # [N]

                for bi in range(self.num_buckets):
                    lo, hi = self.bucket_edges[bi], self.bucket_edges[bi + 1]
                    bm = (pos2d >= lo) & (pos2d < hi)
                    if not bm.any():
                        continue

                    idx_b_cpu = idx[bm].reshape(-1).to(torch.long).cpu()
                    counts_b = torch.bincount(idx_b_cpu, minlength=self.num_experts)

                    self.data_buffer[layer_idx]["counts"][bi] += counts_b
                    self.data_buffer[layer_idx]["total"][bi] += int(bm.sum().item()) * self.top_k

                    if self.store_mass:
                        mass_b = torch.zeros(self.num_experts, dtype=torch.float32)
                        mass_b.index_add_(0, idx_b_cpu, top_p[bm].reshape(-1).to(torch.float32).cpu())
                        self.data_buffer[layer_idx]["mass"][bi] += mass_b

        return hook

    def attach_hooks(self):
        self.detach_hooks()
        self._init_buffer()
        self.profiled_layer_indices = []
        print(f"[Profiler] Attaching gate hooks across {self.num_layers} layers...")
        for i, layer in enumerate(self.layers):
            try:
                gate = self.gate_getter(layer)
            except Exception:
                # Mixed dense+MoE stacks may not expose a gate on every layer.
                continue
            self.hooks.append(gate.register_forward_hook(self._get_hook(i)))
            self.profiled_layer_indices.append(i)

        if not self.profiled_layer_indices:
            raise RuntimeError(
                f"Failed to resolve gate module on any layer (gate_path={self.gate_path})."
            )
        print(
            f"[Profiler] Hooked {len(self.profiled_layer_indices)}/{self.num_layers} "
            f"layers: {self.profiled_layer_indices}"
        )

    def detach_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []

    def _process_batch(self, batch_text: List[str]):
        inputs = self.tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.seq_len,
        ).to(self.model.device)

        self._current_attn_mask = inputs.get("attention_mask", None)
        if self._current_attn_mask is None:
            raise RuntimeError("Tokenizer output missing attention_mask")

        try:
            self.model.eval()
            with torch.inference_mode():
                self.model(**inputs)
        finally:
            self._current_attn_mask = None

    def validate_buffer_or_raise(self) -> Dict[str, Any]:
        """
        Sanity-check captured routing telemetry.

        Validates:
          - counts sum equals tracked total assignments
          - mass tensors are finite (when enabled)
          - optional mass bounds with/without top-k renormalization
        """
        layer_totals: List[int] = []
        token_totals: List[int] = []
        layer_indices = (
            self.profiled_layer_indices if self.profiled_layer_indices else list(range(self.num_layers))
        )
        for layer_idx in layer_indices:
            layer_buf = self.data_buffer[layer_idx]
            counts = layer_buf["counts"]
            total_obj = layer_buf["total"]
            mass = layer_buf.get("mass", None)

            if self.bucket_edges is None:
                counted = int(counts.sum().item())
                tracked = int(total_obj)
                if counted != tracked:
                    raise RuntimeError(
                        f"Profiler sanity failed at layer {layer_idx}: "
                        f"counts.sum={counted} != total={tracked}"
                    )
                layer_totals.append(tracked)
                token_totals.append(tracked // self.top_k)

                if self.store_mass and mass is not None:
                    if not torch.isfinite(mass).all():
                        raise RuntimeError(f"Profiler sanity failed at layer {layer_idx}: mass has NaN/Inf values.")

                    mass_sum = float(mass.sum().item())
                    tokens = tracked / float(self.top_k)
                    if self.renorm_topk_prob:
                        tol = max(1e-2, tokens * 1e-3)
                        if abs(mass_sum - tokens) > tol:
                            raise RuntimeError(
                                f"Profiler sanity failed at layer {layer_idx}: "
                                f"renorm mass_sum={mass_sum:.6f} not close to tokens={tokens:.6f} (tol={tol:.6f})"
                            )
                    else:
                        # Without renorm, per-token top-k mass is in (0, 1], so aggregate mass must be in [0, tokens].
                        tol = max(1e-2, tokens * 1e-3)
                        if mass_sum < -tol or mass_sum > tokens + tol:
                            raise RuntimeError(
                                f"Profiler sanity failed at layer {layer_idx}: "
                                f"mass_sum={mass_sum:.6f} outside [0, {tokens:.6f}]"
                            )
            else:
                if not isinstance(total_obj, list):
                    raise RuntimeError(
                        f"Profiler sanity failed at layer {layer_idx}: bucketed totals should be list, got {type(total_obj)}"
                    )
                if counts.dim() != 2:
                    raise RuntimeError(
                        f"Profiler sanity failed at layer {layer_idx}: bucketed counts should be 2D, got {tuple(counts.shape)}"
                    )
                if counts.shape[0] != len(total_obj):
                    raise RuntimeError(
                        f"Profiler sanity failed at layer {layer_idx}: bucket count mismatch "
                        f"counts.shape[0]={counts.shape[0]} vs len(total)={len(total_obj)}"
                    )

                tracked_layer = 0
                for bucket_idx in range(len(total_obj)):
                    counted_b = int(counts[bucket_idx].sum().item())
                    tracked_b = int(total_obj[bucket_idx])
                    if counted_b != tracked_b:
                        raise RuntimeError(
                            f"Profiler sanity failed at layer {layer_idx}, bucket {bucket_idx}: "
                            f"counts.sum={counted_b} != total={tracked_b}"
                        )
                    tracked_layer += tracked_b

                    if self.store_mass and mass is not None:
                        if not torch.isfinite(mass[bucket_idx]).all():
                            raise RuntimeError(
                                f"Profiler sanity failed at layer {layer_idx}, bucket {bucket_idx}: mass has NaN/Inf."
                            )
                        mass_sum_b = float(mass[bucket_idx].sum().item())
                        tokens_b = tracked_b / float(self.top_k)
                        tol_b = max(1e-2, tokens_b * 1e-3)
                        if self.renorm_topk_prob:
                            if abs(mass_sum_b - tokens_b) > tol_b:
                                raise RuntimeError(
                                    f"Profiler sanity failed at layer {layer_idx}, bucket {bucket_idx}: "
                                    f"renorm mass_sum={mass_sum_b:.6f} not close to tokens={tokens_b:.6f} "
                                    f"(tol={tol_b:.6f})"
                                )
                        else:
                            if mass_sum_b < -tol_b or mass_sum_b > tokens_b + tol_b:
                                raise RuntimeError(
                                    f"Profiler sanity failed at layer {layer_idx}, bucket {bucket_idx}: "
                                    f"mass_sum={mass_sum_b:.6f} outside [0, {tokens_b:.6f}]"
                                )

                layer_totals.append(tracked_layer)
                token_totals.append(tracked_layer // self.top_k)

        if layer_totals:
            first_total = layer_totals[0]
            if any(t != first_total for t in layer_totals[1:]):
                raise RuntimeError(
                    "Profiler sanity failed: assignment totals differ across layers "
                    f"(totals={layer_totals})"
                )

        return {
            "layers_checked": len(layer_indices),
            "layer_indices_checked": list(layer_indices),
            "assignments_per_layer": layer_totals,
            "tokens_per_layer": token_totals,
            "renorm_topk_prob": self.renorm_topk_prob,
        }
