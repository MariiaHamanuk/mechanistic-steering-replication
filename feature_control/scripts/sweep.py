"""Step 3 (control, offline): does moving x_l along the observer direction move the actions?

A) Output-sensitivity sweep (paper Sec. V-B, Fig. 4): x_l <- x_l + alpha * ||x_l|| * W/||W|| at one layer,
   for each layer and alpha; report the mean change of the predicted chunk-mean dz and gripper command
   vs the unmodified model on the same frames with the same denoising noise.
   (The paper writes x + alpha*W without giving alpha values; we scale by ||x_l|| so one alpha grid
   is comparable across layers.)
B) Closed-form controller check (paper Eq. 7, Fig. 5): at a chosen layer, ask for xi = W.x+b to lie in
   an interval taken from the label distribution (e.g. dz >= its 90th percentile) and measure
   (i) the observer value after the intervention (should be inside by construction) and
   (ii) whether the policy's ACTUAL predicted dz moves into the interval.

Uses held-out-episode frames from probe.py, so observers are evaluated on data they were not fit on.

  python feature_control/scripts/sweep.py --run feature_control/logs/pi05_libero_collect
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from feature_control.common import collate, load_policy  # noqa: E402
from feature_control.hooks import PrefixSteer, Probe, Steer  # noqa: E402

ACTION_DIM = {"dz": 2, "gripper_cmd": 6}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n_frames", type=int, default=50, help="paper Fig. 4 uses 50 LIBERO episodes")
    p.add_argument("--features", default="dz,gripper_cmd")
    p.add_argument("--alphas", default="-0.5,-0.2,-0.1,0.1,0.2,0.5")
    p.add_argument("--layers", default="all")
    p.add_argument("--ctrl_layer", type=int, default=-1, help="-1 = best observer layer for dz")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="", help="override model dtype (e.g. float32: small controller shifts can be "
                                               "rounded away when added to large bf16 hidden states)")
    p.add_argument("--skip_sweep", action="store_true", help="only run the controller check (B)")
    p.add_argument("--tag", default="", help="suffix for the output files")
    p.add_argument("--margin_sd", type=float, default=0.0,
                   help="controller aims this many label-SDs inside the interval (0 = paper's minimal shift)")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    run = Path(args.run)
    meta = json.loads((run / "meta.json").read_text())
    res = json.loads((run / "probe_results.json").read_text())
    pz = np.load(run / "probes.npz")
    feats = list(pz["features"])
    lab = dict(np.load(run / "labels.npz"))
    L = meta["n_layers"]
    layers = list(range(L)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    sweep_feats = args.features.split(",")
    horizon = meta["label_horizon"]

    test_rows = np.where(np.isin(lab["episode_index"], res["test_episodes"]))[0]
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(test_rows, size=min(args.n_frames, len(test_rows)), replace=False))
    ds = LeRobotDataset(meta["dataset"], episodes=sorted({int(lab["episode_index"][r]) for r in rows}))
    cols = ds.hf_dataset.with_format("numpy")
    ds_ep = np.asarray(cols["episode_index"]).reshape(-1)
    ds_fr = np.asarray(cols["frame_index"]).reshape(-1)
    lookup = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ds_ep, ds_fr))}
    items = [ds[lookup[(int(lab["episode_index"][r]), int(lab["frame_index"][r]))]] for r in rows]

    dtype = args.dtype or meta["dtype"]
    policy, pre, post, cfg = load_policy(meta["policy"], dtype=dtype)
    batches = [pre(collate(items[i: i + args.batch])) for i in range(0, len(items), args.batch)]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    noises = [torch.randn((len(items[i: i + args.batch]), cfg.chunk_size, cfg.max_action_dim), generator=gen)
              for i in range(0, len(items), args.batch)]

    def predict():
        out = []
        for b, n in zip(batches, noises):
            with torch.no_grad():
                out.append(post(policy.predict_action_chunk(b, noise=n)).float().cpu().numpy()[:, :horizon].mean(1))
        return np.concatenate(out)  # (n, action_dim)

    t0 = time.time()
    torch.cuda.synchronize()
    base = predict()
    base_ms = (time.time() - t0) / len(items) * 1000

    def probe(f, l):
        fi = feats.index(f)
        kind = "classification" if f == "gripper_cmd" else "regression"
        return Probe(layer=l, feature=f, W=pz["W"][fi, l], b=float(pz["b"][fi, l]), kind=kind)

    # A) sweep
    sweep = {f: {} for f in sweep_feats}
    for f in ([] if args.skip_sweep else sweep_feats):
        dim = ACTION_DIM[f]
        for l in layers:
            row = {}
            for a in alphas:
                h = PrefixSteer(policy, [Steer(probe(f, l), mode="fixed", alpha=a)])
                pred = predict()
                h.remove()
                row[str(a)] = float((pred[:, dim] - base[:, dim]).mean())
            sweep[f][l] = row
            print(f"[sweep] {f} layer {l:2d}: " + " ".join(f"a={a:+.2f}:{v:+.4f}" for a, v in
                                                          zip(alphas, row.values())), flush=True)

    # B) controller (dz)
    l = args.ctrl_layer if args.ctrl_layer >= 0 else res["best"]["dz"]["layer"]
    q10, q90 = np.percentile(lab["dz"], [10, 90])
    margin = args.margin_sd * float(np.std(lab["dz"]))
    tol = 1e-3 * float(np.std(lab["dz"]))  # landing exactly on the boundary counts as inside
    ctrl = {"layer": l, "observer_R2": res["scores"]["dz"][l], "base_pred_dz_mean": float(base[:, 2].mean()),
            "margin": margin}
    for name, lo, hi in [("dz_high", q90, np.inf), ("dz_low", -np.inf, q10)]:
        s = Steer(probe("dz", l), mode="controller", xi_min=lo, xi_max=hi, margin=margin)
        h = PrefixSteer(policy, [s])
        torch.cuda.synchronize()
        t1 = time.time()
        pred = predict()
        steer_ms = (time.time() - t1) / len(items) * 1000
        h.remove()
        inside = lambda v: (v >= lo - tol) & (v <= hi + tol)  # noqa: E731
        ctrl[name] = {
            "interval": [None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi)],
            "observer_inside_before": float(inside(np.array(s.stats["xi_before"])).mean()),
            "observer_inside_after": float(inside(np.array(s.stats["xi_after"])).mean()),
            "pred_dz_inside_base": float(inside(base[:, 2]).mean()),
            "pred_dz_inside_steered": float(inside(pred[:, 2]).mean()),
            "pred_dz_mean_steered": float(pred[:, 2].mean()),
            "mean_rel_u_norm": float(np.mean(np.array(s.stats["u_norm"]) / np.array(s.stats["x_norm"]))),
            "overhead_pct": (steer_ms / base_ms - 1) * 100,
        }
        print(f"[ctrl] {name} @L{l}: observer in-interval {ctrl[name]['observer_inside_before']:.0%} -> "
              f"{ctrl[name]['observer_inside_after']:.0%}; predicted dz in-interval "
              f"{ctrl[name]['pred_dz_inside_base']:.0%} -> {ctrl[name]['pred_dz_inside_steered']:.0%}; "
              f"mean dz {base[:, 2].mean():+.4f} -> {pred[:, 2].mean():+.4f}", flush=True)

    out = {"n_frames": len(items), "alphas": alphas, "alpha_units": "fraction of ||x_l||", "dtype": dtype,
           "base_ms_per_frame": base_ms, "sweep": sweep, "controller": ctrl}
    (run / f"sweep_results{args.tag}.json").write_text(json.dumps(out, indent=2))
    if args.skip_sweep:
        print(f"[sweep] DONE (controller only) -> {run / f'sweep_results{args.tag}.json'}")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(sweep_feats), figsize=(6 * len(sweep_feats), 4))
        for ax, f in zip(np.atleast_1d(axes), sweep_feats):
            for a in alphas:
                ax.plot(layers, [sweep[f][l][str(a)] for l in layers], marker="o", ms=3, label=f"α={a:+g}")
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_xlabel("intervention layer")
            ax.set_ylabel(f"mean Δ predicted {f}")
            ax.set_title(f"{f}: output sensitivity")
        np.atleast_1d(axes)[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(run / "sweep.png", dpi=150)
    except Exception as e:
        print(f"[sweep] plot skipped: {e}")
    print(f"[sweep] DONE -> {run / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
