"""Replication of Buurmeijer et al. 2026, "Observing and Controlling Features in VLA Models" (arXiv:2603.05487).

No official code was released; this package re-implements the method from the paper text:
  * observe  - linear probes on mean-pooled VLM-prefix hidden states (per layer, per feature)
  * control  - additive broadcast intervention X_l <- X_l + 1 u^T, with u either a fixed alpha*W
               (output-sensitivity sweep, paper Sec. V-B) or the minimal closed-form controller
               u = (xi_bound - xi_hat) W / ||W||^2 (paper Eq. 7).
"""
