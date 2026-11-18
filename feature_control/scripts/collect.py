"""Step 1 (observe): collect mean-pooled VLM-prefix activations x_l for every layer + paper labels.

Runs the frozen policy on frames of a LeRobot dataset (default: lerobot/libero with lerobot/pi05_libero)
and saves, for K frames:
  acts_L{l}.npy   (K, d) float16   x_l for each layer l
  labels.npz      paper features (state, chunk-averaged actions, binary gripper state/command)
                  + episode/frame/task indices + the policy's own predicted chunk-mean actions
  meta.json       settings, timing, memory

  python feature_control/scripts/collect.py --out feature_control/logs/pi05_libero_collect
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

from feature_control.common import FEATURES, collate, episode_labels, load_policy  # noqa: E402
from feature_control.hooks import PrefixTap  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="lerobot/pi05_libero")
    p.add_argument("--dataset", default="lerobot/libero")
    p.add_argument("--n_episodes", type=int, default=400, help="random episodes (of 1693 in lerobot/libero)")
    p.add_argument("--stride", type=int, default=10, help="take every k-th frame (10 = 1 frame/s at 10 fps)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--horizon", type=int, default=0, help="action-label averaging horizon; 0 = policy chunk_size")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=0, help="debug: stop after this many frames")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    total = LeRobotDatasetMetadata(args.dataset).total_episodes
    rng = np.random.default_rng(args.seed)
    episodes = sorted(rng.choice(total, size=min(args.n_episodes, total), replace=False).tolist())
    ds = LeRobotDataset(args.dataset, episodes=episodes)
    print(f"[collect] {len(episodes)} episodes, {len(ds)} frames loaded in {time.time() - t0:.0f}s")

    cols = ds.hf_dataset.with_format("numpy")
    ep_idx = np.asarray(cols["episode_index"]).reshape(-1)
    fr_idx = np.asarray(cols["frame_index"]).reshape(-1)
    task_idx = np.asarray(cols["task_index"]).reshape(-1)
    states = np.stack(cols["observation.state"]).astype(np.float64)
    actions = np.stack(cols["action"]).astype(np.float64)

    policy, pre, post, cfg = load_policy(args.policy, dtype=args.dtype)
    horizon = args.horizon or int(cfg.chunk_size)
    aperture = states[:, 6] - states[:, 7]
    grip_thresh = float((aperture.min() + aperture.max()) / 2)

    # Labels for every frame, then keep every `stride`-th frame of each episode.
    labels = {k: np.zeros(len(ds), dtype=np.float32) for k in FEATURES}
    for e in np.unique(ep_idx):
        rows = np.where(ep_idx == e)[0]
        rows = rows[np.argsort(fr_idx[rows])]
        lab = episode_labels(states[rows], actions[rows], horizon, grip_thresh)
        for k in FEATURES:
            labels[k][rows] = lab[k]
    keep = np.where(fr_idx % args.stride == 0)[0]
    if args.max_frames:
        keep = keep[: args.max_frames]
    print(f"[collect] {len(keep)} frames to run (stride {args.stride}), label horizon {horizon}")

    tap = PrefixTap(policy)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    pred_mean = []
    t_inf = 0.0
    torch.cuda.reset_peak_memory_stats()
    for bi in range(0, len(keep), args.batch):
        rows = keep[bi: bi + args.batch]
        items = [ds[int(i)] for i in rows]
        for i, it in zip(rows, items):
            assert int(it["episode_index"]) == ep_idx[i] and int(it["frame_index"]) == fr_idx[i], "index mismatch"
        batch = pre(collate(items))
        noise = torch.randn((len(rows), cfg.chunk_size, cfg.max_action_dim), generator=gen)
        torch.cuda.synchronize()
        ts = time.time()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch, noise=noise)
        torch.cuda.synchronize()
        t_inf += time.time() - ts
        chunk = post(chunk).float().cpu().numpy()  # (B, chunk, action_dim), unnormalised
        pred_mean.append(chunk[:, :horizon].mean(axis=1))
        done = min(bi + args.batch, len(keep))
        if (bi // args.batch) % 20 == 0:
            el = time.time() - t0
            print(f"[collect] {done}/{len(keep)} frames, {t_inf / done * 1000:.0f} ms/frame inference, "
                  f"elapsed {el / 60:.1f} min", flush=True)

    acts = tap.pop()
    tap.remove()
    for l, a in acts.items():
        assert a.shape[0] == len(keep), f"layer {l}: {a.shape[0]} != {len(keep)} (hook fired more than once per batch?)"
        np.save(out / f"acts_L{l:02d}.npy", a)
    np.savez(
        out / "labels.npz",
        episode_index=ep_idx[keep], frame_index=fr_idx[keep], task_index=task_idx[keep],
        pred_action_mean=np.concatenate(pred_mean), **{k: v[keep] for k, v in labels.items()},
    )
    meta = {
        "paper": "Buurmeijer et al. 2026, arXiv:2603.05487",
        "policy": args.policy, "dataset": args.dataset, "dtype": args.dtype,
        "n_episodes": len(episodes), "episodes": episodes, "stride": args.stride, "n_frames": int(len(keep)),
        "label_horizon": horizon, "gripper_open_threshold": grip_thresh,
        "n_layers": len(acts), "hidden_dim": int(next(iter(acts.values())).shape[1]),
        "inference_ms_per_frame": t_inf / len(keep) * 1000,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu": torch.cuda.get_device_name(0), "wall_min": (time.time() - t0) / 60,
        "seed": args.seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[collect] DONE: {len(keep)} frames x {len(acts)} layers -> {out}  ({meta['wall_min']:.1f} min)")


if __name__ == "__main__":
    main()
