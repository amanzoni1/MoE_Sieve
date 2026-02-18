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
        self.num_experts = int(num_experts or getattr(model.config, "num_experts", 64))
        self.top_k = int(top_k or getattr(model.config, "num_experts_per_tok", 8))

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

        # Default OLMoE gate path
        self.gate_getter = gate_getter or (lambda layer: layer.mlp.gate)

        # NEW: auto-detect renorm_topk_prob (affects mass only, not indices)
        if renorm_topk_prob is None:
            try:
                renorm_topk_prob = bool(getattr(self.layers[0].mlp, "norm_topk_prob", False))
            except Exception:
                renorm_topk_prob = False
        self.renorm_topk_prob = bool(renorm_topk_prob)

        os.makedirs(self.output_dir, exist_ok=True)

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
            m = self._current_attn_mask.to(torch.bool)  # [B,S]

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
                pos = torch.arange(S, device=self._current_attn_mask.device).view(1, S).expand(B, S)  # [B,S]
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
        print(f"[Profiler] Attaching gate hooks to {self.num_layers} layers...")
        for i, layer in enumerate(self.layers):
            gate = self.gate_getter(layer)  # e.g., layer.mlp.gate
            self.hooks.append(gate.register_forward_hook(self._get_hook(i)))

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
