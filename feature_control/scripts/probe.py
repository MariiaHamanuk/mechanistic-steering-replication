"""Step 2 (observe): fit one linear observer per (feature, layer) - paper Algorithm 1, Fig. 3.

Continuous features: ridge regression (MSE), scored by R^2 on held-out data.
Binary features (gripper state / command): logistic regression (cross-entropy), scored by accuracy.
Held-out split: 10% of EPISODES (paper: "held-out 10%", unit not stated; splitting by episode avoids
near-duplicate neighbouring frames in train and test). Regularisation strength is picked by internal
cross-validation on the training part (the paper does not state its regulariser).

Outputs in the collect dir:
  probe_results.json   score per feature per layer, best layer per feature, chance levels
  probes.npz           raw-space W (F, L, d) and b (F, L) for the steering scripts
  probe_heatmap.png    feature x layer score map (paper Fig. 3 analogue)

  python feature_control/scripts/probe.py --run feature_control/logs/pi05_libero_collect
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from feature_control.common import BINARY, CONTINUOUS  # noqa: E402


def fit_layer(X, labels, train, test, features, scale=False):
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    from sklearn.metrics import accuracy_score, r2_score

    X = X.astype(np.float64)
    mu = X[train].mean(0)
    # Default: centre only, fit in the raw representation space (as the paper's f(x) = W^T x + b).
    # Per-dimension scaling gives huge raw weights on near-constant dimensions; the minimal-norm control
    # shift then concentrates there and barely moves the policy.
    sd = X[train].std(0) + 1e-6 if scale else np.ones(X.shape[1])
    Z = (X - mu) / sd
    res = {}
    # All continuous targets in one RidgeCV fit (one SVD, alpha chosen per target by efficient LOO-CV).
    cont = [f for f in features if f not in BINARY]
    if cont:
        Y = np.stack([labels[f] for f in cont], axis=1)
        m = RidgeCV(alphas=np.logspace(-1, 5, 13), alpha_per_target=True).fit(Z[train], Y[train])
        P = m.predict(Z[test])
        for j, f in enumerate(cont):
            w, b = m.coef_[j], float(m.intercept_[j])
            res[f] = (float(r2_score(Y[test, j], P[:, j])), (w / sd),
                      b - float(np.sum(w * mu / sd)))
    for f in features:
        if f not in BINARY:
            continue
        y = labels[f]
        if f in BINARY:
            if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
                res[f] = None
                continue
            try:
                m = LogisticRegressionCV(Cs=np.logspace(-4, 1, 6), cv=3, max_iter=3000, n_jobs=1).fit(Z[train], y[train])
            except ValueError as e:  # too few examples of one class for 3-fold CV (small runs)
                print(f"[probe] {f}: skipped ({e})")
                res[f] = None
                continue
            score = accuracy_score(y[test], m.predict(Z[test]))
            w, b = m.coef_[0], float(m.intercept_[0])
        else:
            m = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(Z[train], y[train])
            score = r2_score(y[test], m.predict(Z[test]))
            w, b = m.coef_, float(m.intercept_)
        # Back to raw space: xi = w.(x-mu)/sd + b = (w/sd).x + (b - sum(w*mu/sd))
        W_raw = w / sd
        b_raw = b - float(np.sum(w * mu / sd))
        res[f] = (float(score), W_raw, b_raw)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="output dir of collect.py")
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=0, help="parallel layers; 0 = half the CPUs")
    p.add_argument("--scale", action="store_true", help="standardise each dimension before fitting")
    args = p.parse_args()

    run = Path(args.run)
    meta = json.loads((run / "meta.json").read_text())
    lab = dict(np.load(run / "labels.npz"))
    features = CONTINUOUS + BINARY
    L, d = meta["n_layers"], meta["hidden_dim"]

    eps = np.unique(lab["episode_index"])
    rng = np.random.default_rng(args.seed)
    test_eps = rng.choice(eps, size=max(1, int(round(len(eps) * args.test_frac))), replace=False)
    test = np.isin(lab["episode_index"], test_eps)
    train = ~test
    print(f"[probe] {train.sum()} train / {test.sum()} test frames ({len(test_eps)} held-out episodes)")

    scores = {f: [None] * L for f in features}
    Ws = np.zeros((len(features), L, d), dtype=np.float64)
    bs = np.zeros((len(features), L), dtype=np.float64)
    t0 = time.time()
    from joblib import Parallel, delayed

    def one(l):
        X = np.load(run / f"acts_L{l:02d}.npy").astype(np.float32)
        return l, fit_layer(X, lab, train, test, features, scale=args.scale)

    n_jobs = args.jobs or max(1, min(L, (os.cpu_count() or 2) // 2))
    print(f"[probe] fitting {L} layers with {n_jobs} parallel jobs", flush=True)
    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(l) for l in range(L))
    for l, res in sorted(results, key=lambda t: t[0]):
        for fi, f in enumerate(features):
            if res[f] is not None:
                scores[f][l], Ws[fi, l], bs[fi, l] = res[f]
        line = " ".join(f"{f}={scores[f][l]:.2f}" for f in features if scores[f][l] is not None)
        print(f"[probe] layer {l:2d} ({time.time() - t0:.0f}s): {line}", flush=True)

    chance = {f: float(max(lab[f][test].mean(), 1 - lab[f][test].mean())) for f in BINARY}
    best = {}
    for f in features:
        vals = [(s, l) for l, s in enumerate(scores[f]) if s is not None]
        if vals:
            s, l = max(vals)
            best[f] = {"layer": l, "score": s, "metric": "accuracy" if f in BINARY else "R2"}
    out = {"features": features, "scores": scores, "best": best, "chance_accuracy": chance,
           "n_train": int(train.sum()), "n_test": int(test.sum()), "test_episodes": test_eps.tolist(),
           "policy": meta["policy"], "dataset": meta["dataset"]}
    (run / "probe_results.json").write_text(json.dumps(out, indent=2))
    np.savez(run / "probes.npz", W=Ws, b=bs, features=np.array(features))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        M = np.array([[np.nan if s is None else s for s in scores[f]] for f in features])
        fig, ax = plt.subplots(figsize=(1 + 0.45 * L, 0.5 * len(features) + 1.5))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_yticks(range(len(features)), [f + (" (acc)" if f in BINARY else " (R²)") for f in features])
        ax.set_xticks(range(L))
        ax.set_xlabel("layer")
        ax.set_title(f"Linear observability, {meta['policy']} on {meta['dataset']}")
        for i in range(len(features)):
            for j in range(L):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if M[i, j] < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.02)
        fig.tight_layout()
        fig.savefig(run / "probe_heatmap.png", dpi=150)
    except Exception as e:  # plotting is optional
        print(f"[probe] heatmap skipped: {e}")

    print("[probe] best layer per feature:")
    for f, v in best.items():
        print(f"   {f:14s} layer {v['layer']:2d}  {v['metric']} {v['score']:.3f}"
              + (f"  (chance {chance[f]:.2f})" if f in chance else ""))
    print(f"[probe] DONE -> {run / 'probe_results.json'}")


if __name__ == "__main__":
    main()
