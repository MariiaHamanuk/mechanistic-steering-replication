"""Reading and steering mean-pooled VLM-prefix representations of LeRobot pi0/pi0.5 policies.

Paper notation (Buurmeijer et al. 2026):
  X_l in R^{N x d}  hidden state after layer l over the N prefix tokens (images + language[+ state])
  x_l = (1/N) X_l^T 1_N                      mean-pooled representation (Eq. 2)
  g_l(X) = X + 1_N u^T                       broadcast additive intervention (Eq. 5): shifts x_l by exactly u

In LeRobot's PI05Pytorch.sample_actions the prefix goes through
`paligemma.model.language_model` alone (filling the KV cache); the action expert then attends to that
cache during denoising. A forward hook on `language_model.layers[l]` therefore sees exactly X_l of the
VLM prefix, and changing its output changes every later prefix layer and the KV cache the action
expert reads. The hooks fire only in that prefix pass (denoising runs the expert only).

Mean pooling uses only valid (non-padding) prefix tokens: language tokens are padded to
`tokenizer_max_length` and missing cameras are masked. The paper says "mean over the N token rows"
without mentioning padding; pooling over padding would make x_l depend on prompt length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

_LAYER_PATHS = (
    "model.paligemma_with_expert.paligemma.model.language_model.layers",  # lerobot pi0 / pi05
    "model.paligemma_with_expert.paligemma.language_model.model.layers",  # older transformers layout
)


def _getattr_path(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def find_vlm_layers(policy) -> torch.nn.ModuleList:
    for path in _LAYER_PATHS:
        try:
            layers = _getattr_path(policy, path)
        except AttributeError:
            continue
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
            return layers
    raise RuntimeError(
        "Could not find the VLM decoder layers. Print the policy and add the path to _LAYER_PATHS."
    )


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace_hidden(out, new):
    if isinstance(out, tuple):
        return (new, *out[1:])
    return new


class PrefixMask:
    """Captures the prefix padding mask by wrapping `policy.model.embed_prefix`."""

    def __init__(self, policy):
        self.model = policy.model
        self.orig = self.model.embed_prefix
        self.mask: torch.Tensor | None = None  # (B, N) bool

        def wrapped(*args, **kwargs):
            embs, pad_masks, att_masks = self.orig(*args, **kwargs)
            self.mask = pad_masks
            return embs, pad_masks, att_masks

        self.model.embed_prefix = wrapped

    def remove(self):
        self.model.embed_prefix = self.orig

    def weights(self, dtype) -> torch.Tensor:
        """(B, N, 1) averaging weights: 1/n_valid on valid tokens, 0 on padding."""
        m = self.mask.to(dtype)
        return (m / m.sum(dim=1, keepdim=True).clamp(min=1.0)).unsqueeze(-1)


def pooled(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Masked mean over tokens, in float32. h: (B, N, d), w: (B, N, 1) -> (B, d)."""
    return (h.float() * w.float()).sum(dim=1)


class PrefixTap:
    """Records x_l for the selected layers on every prefix pass."""

    def __init__(self, policy, layer_ids=None):
        self.layers = find_vlm_layers(policy)
        self.layer_ids = list(range(len(self.layers))) if layer_ids is None else list(layer_ids)
        self.mask = PrefixMask(policy)
        self.store: dict[int, list[np.ndarray]] = {l: [] for l in self.layer_ids}
        self.handles = [self.layers[l].register_forward_hook(self._make(l)) for l in self.layer_ids]
        self.enabled = True

    def _make(self, l):
        def hook(_mod, _inp, out):
            if self.enabled and self.mask.mask is not None:
                h = _hidden(out)
                x = pooled(h.detach(), self.mask.weights(h.dtype))
                self.store[l].append(x.cpu().numpy().astype(np.float16))
        return hook

    def pop(self) -> dict[int, np.ndarray]:
        out = {l: np.concatenate(v, axis=0) if v else None for l, v in self.store.items()}
        self.store = {l: [] for l in self.layer_ids}
        return out

    def remove(self):
        for h in self.handles:
            h.remove()
        self.mask.remove()


@dataclass
class Probe:
    """Linear observer xi = W.x + b in RAW (unstandardised) representation space."""

    layer: int
    feature: str
    W: np.ndarray  # (d,)
    b: float
    kind: str  # "regression" | "classification" (xi is the logit)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        W = torch.as_tensor(self.W, device=x.device, dtype=torch.float64)
        return x.double() @ W + self.b


@dataclass
class Steer:
    """One intervention at one layer.

    mode="fixed":      u = alpha * W / ||W||   (output-sensitivity sweep; alpha in units of ||x_l||, see below)
    mode="controller": u = argmin ||u||^2 s.t. xi_min <= W.(x+u)+b <= xi_max   (paper Eq. 6-7)
    """

    probe: Probe
    mode: str = "controller"
    alpha: float = 0.0
    xi_min: float = -np.inf
    xi_max: float = np.inf
    margin: float = 0.0  # aim this far inside the interval (the paper's u lands exactly on the boundary)
    stats: dict = field(default_factory=lambda: {"calls": 0, "active": 0, "xi_before": [], "xi_after": [],
                                                 "u_norm": [], "x_norm": []})

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d) float32 pooled representation -> u: (B, d)."""
        W = torch.as_tensor(self.probe.W, device=x.device, dtype=torch.float64)
        x = x.double()
        if self.mode == "fixed":
            # Scale relative to the representation norm so alpha means the same thing at every layer.
            return ((self.alpha * x.norm(dim=1, keepdim=True)) * (W / W.norm())[None, :]).float()
        xi = x @ W + self.probe.b
        target = torch.clamp(xi, min=self.xi_min + self.margin, max=self.xi_max - self.margin)
        return (((target - xi) / (W @ W))[:, None] * W[None, :]).float()


class PrefixSteer:
    """Applies Steer interventions (one per layer) as X_l <- X_l + 1 u^T on valid prefix tokens."""

    def __init__(self, policy, steers: list[Steer]):
        self.layers = find_vlm_layers(policy)
        self.mask = PrefixMask(policy)
        self.steers = steers
        self.enabled = True
        self.handles = [self.layers[s.probe.layer].register_forward_hook(self._make(s)) for s in steers]

    def _make(self, s: Steer):
        def hook(_mod, _inp, out):
            if not self.enabled or self.mask.mask is None:
                return None
            h = _hidden(out)
            w = self.mask.weights(h.dtype)
            x = pooled(h, w)
            u = s.delta(x)
            valid = (w > 0).to(h.dtype)  # broadcast to valid tokens only -> pooled shift is exactly u
            h_new = h + valid * u[:, None, :].to(h.dtype)
            st = s.stats
            st["calls"] += 1
            un = u.norm(dim=1)
            st["active"] += int((un > 0).sum())
            st["u_norm"] += un.tolist()
            st["x_norm"] += x.norm(dim=1).tolist()
            st["xi_before"] += s.probe.value(x).tolist()
            st["xi_after"] += s.probe.value(pooled(h_new, w)).tolist()
            return _replace_hidden(out, h_new)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.mask.remove()
