"""Shared helpers: loading a LeRobot policy and computing the paper's state/action labels."""

from __future__ import annotations

import numpy as np
import torch

# Paper Sec. V-A: state s = (x, y, z, phi, theta, psi, g) and actions (dx, dy, dz, dphi, dtheta, dpsi, dg).
CONTINUOUS = ["x", "y", "z", "roll", "pitch", "yaw", "dx", "dy", "dz", "droll", "dpitch", "dyaw"]
BINARY = ["gripper_state", "gripper_cmd"]
FEATURES = CONTINUOUS + BINARY


def load_policy(path: str, device: str = "cuda", dtype: str = "bfloat16"):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    if hasattr(cfg, "dtype"):
        cfg.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}}
    )
    return policy, pre, post, cfg


def _euler_from_rotvec(rv: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(rv).as_euler("xyz")


def episode_labels(states: np.ndarray, actions: np.ndarray, horizon: int, grip_thresh: float) -> dict:
    """Labels for every frame of one LIBERO episode.

    states:  (T, 8) = eef pos (3) + eef axis-angle (3) + gripper finger qpos (2)   [lerobot/libero]
    actions: (T, 7) = delta pos (3) + delta rot (3) + gripper (-1 open, +1 close)
    Action labels are averaged over the next `horizon` actions (paper: "averaged over the chunk"),
    truncated at the end of the episode.
    """
    T = len(states)
    eul = _euler_from_rotvec(states[:, 3:6].astype(np.float64))
    aperture = states[:, 6] - states[:, 7]
    csum = np.vstack([np.zeros((1, actions.shape[1])), np.cumsum(actions, axis=0)])
    ends = np.minimum(np.arange(T) + horizon, T)
    mean_act = (csum[ends] - csum[np.arange(T)]) / (ends - np.arange(T))[:, None]
    out = {
        "x": states[:, 0], "y": states[:, 1], "z": states[:, 2],
        "roll": eul[:, 0], "pitch": eul[:, 1], "yaw": eul[:, 2],
        "dx": mean_act[:, 0], "dy": mean_act[:, 1], "dz": mean_act[:, 2],
        "droll": mean_act[:, 3], "dpitch": mean_act[:, 4], "dyaw": mean_act[:, 5],
        "gripper_state": (aperture > grip_thresh).astype(np.float32),  # 1 = open
        "gripper_cmd": (mean_act[:, 6] > 0).astype(np.float32),  # 1 = close
    }
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def collate(items: list[dict]) -> dict:
    """Stack LeRobotDataset items into a batch the policy preprocessor accepts."""
    batch = {}
    for k in items[0]:
        v = items[0][k]
        if isinstance(v, torch.Tensor):
            batch[k] = torch.stack([it[k] for it in items])
        else:
            batch[k] = [it[k] for it in items]
    return batch
