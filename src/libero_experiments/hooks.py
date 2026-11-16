"""Neuron intervention hooks."""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class InterventionNormStats:
    """Measures how big the intervention is relative to the residual stream.

    For each intervened layer L it accumulates
      delta = ||W_down[:, ids] @ (alpha - a_orig[ids])||   (change of the FFN output caused by the override)
      resid = ||x_L||                                       (residual stream entering decoder layer L)
    averaged over tokens. Only the first `max_samples_per_layer` forward calls are measured and the sums stay
    on the GPU (no .item() inside hooks), so the effect on per-step latency is negligible.
    """

    def __init__(self, max_samples_per_layer: int = 2000):
        self.max_samples = max_samples_per_layer
        self.delta_sum: Dict[int, torch.Tensor] = {}
        self.delta_n: Dict[int, int] = {}
        self.resid_sum: Dict[int, torch.Tensor] = {}
        self.resid_n: Dict[int, int] = {}

    def wants(self, counts: Dict[int, int], layer: int) -> bool:
        return counts.get(layer, 0) < self.max_samples

    @staticmethod
    def _add(sums: Dict[int, torch.Tensor], counts: Dict[int, int], layer: int, value: torch.Tensor) -> None:
        value = value.detach().float()
        sums[layer] = sums[layer] + value if layer in sums else value
        counts[layer] = counts.get(layer, 0) + 1

    def summary(self) -> Dict:
        per_layer = {}
        for layer in sorted(self.delta_sum):
            if layer not in self.resid_sum:
                continue
            d = (self.delta_sum[layer] / self.delta_n[layer]).item()
            r = (self.resid_sum[layer] / self.resid_n[layer]).item()
            per_layer[layer] = {"delta_norm": d, "resid_norm": r, "ratio": d / r if r > 0 else None,
                                "samples": self.delta_n[layer]}
        ratios = [v["ratio"] for v in per_layer.values() if v["ratio"] is not None]
        return {"per_layer": per_layer, "mean_ratio": sum(ratios) / len(ratios) if ratios else None}


def apply_gate_proj_hooks(
    model,
    flat_index_dict: Dict[int, List[str]],
    intermediate_size: int = 11008,
    coef: float = 1.0,
    norm_stats: Optional[InterventionNormStats] = None,
):
    values_per_layer = {}
    for flat_idx in flat_index_dict:
        layer_idx = flat_idx // intermediate_size
        neuron_idx = flat_idx % intermediate_size
        values_per_layer.setdefault(layer_idx, []).append(neuron_idx)

    print("\nNeurons selected for activation:\n")
    for layer, neurons in values_per_layer.items():
        print(f"  Layer {layer}: Neurons {neurons}")
    print()

    def down_proj_hook(layer_idx, neuron_ids, coef_val):
        def hook_fn(module, input, output):
            # Input: activations of shape (batch, seq_len, num_value_vectors) (where for openvla num_value_vectors = 11008)
            # Output: FFN output of shape (batch, seq_len, embedding_dim) (where for openvla embedding_dim = 4096)
            modified_input = input[0]
            measure = norm_stats is not None and norm_stats.wants(norm_stats.delta_n, layer_idx)
            if measure:
                orig = modified_input[..., neuron_ids].clone()
            modified_input[..., neuron_ids] = coef_val
            output = F.linear(modified_input, module.weight, module.bias)
            if measure:
                delta_act = (coef_val - orig).to(module.weight.dtype)  # (..., k)
                delta = F.linear(delta_act, module.weight[:, neuron_ids])  # (..., d)
                norm_stats._add(norm_stats.delta_sum, norm_stats.delta_n, layer_idx,
                                delta.float().norm(dim=-1).mean())
            return output
        return hook_fn

    def resid_pre_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            if norm_stats.wants(norm_stats.resid_n, layer_idx):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is not None:
                    norm_stats._add(norm_stats.resid_sum, norm_stats.resid_n, layer_idx,
                                    hidden.float().norm(dim=-1).mean())
            return None
        return hook_fn

    hooks = []
    decoder_layers = model.language_model.model.layers
    for layer_idx, neuron_ids in values_per_layer.items():
        layer = decoder_layers[layer_idx]
        hooks.append(layer.mlp.down_proj.register_forward_hook(down_proj_hook(layer_idx, neuron_ids, coef)))
        if norm_stats is not None:
            hooks.append(layer.register_forward_pre_hook(resid_pre_hook(layer_idx), with_kwargs=True))

    return hooks
