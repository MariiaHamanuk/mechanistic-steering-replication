import os
import argparse
import logging
from pathlib import Path

import datetime as _datetime

# Python 3.10 compatibility for openpi's datetime.UTC usage.
if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc

import sys as _sys
import types as _types

import jax as _jax
import inspect as _inspect

# Older JAX builds may not expose api_util.debug_info, which Flax expects.
if not hasattr(_jax.api_util, "debug_info"):
    def _debug_info(*_args, **_kwargs):
        return None
    _jax.api_util.debug_info = _debug_info

# Older JAX builds may not accept `debug_info` in lu.wrap_init.
def _patch_wrap_init(module) -> None:
    if not hasattr(module, "wrap_init"):
        return
    _wrap_init_sig = _inspect.signature(module.wrap_init)
    if "debug_info" in _wrap_init_sig.parameters:
        return
    _orig_wrap_init = module.wrap_init

    def _wrap_init(fun, *args, **kwargs):
        kwargs.pop("debug_info", None)
        return _orig_wrap_init(fun, *args, **kwargs)

    module.wrap_init = _wrap_init


_patch_wrap_init(getattr(_jax, "lu", None))
_patch_wrap_init(getattr(_jax, "linear_util", None))
try:
    import jax._src.linear_util as _linear_util
except Exception:  # pragma: no cover - best effort compatibility
    _linear_util = None
_patch_wrap_init(_linear_util)

# Shim missing lerobot module path used by openpi training code.
if "lerobot.common.datasets.lerobot_dataset" not in _sys.modules:
    _lerobot_root = _types.ModuleType("lerobot")
    _lerobot_common = _types.ModuleType("lerobot.common")
    _lerobot_common_datasets = _types.ModuleType("lerobot.common.datasets")
    _stub = _types.ModuleType("lerobot.common.datasets.lerobot_dataset")

    class LeRobotDatasetMetadata:  # noqa: N801 - keep upstream name
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("LeRobotDatasetMetadata is unavailable in this lerobot version.")

    class LeRobotDataset:  # noqa: N801 - keep upstream name
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("LeRobotDataset is unavailable in this lerobot version.")

    _stub.LeRobotDatasetMetadata = LeRobotDatasetMetadata
    _stub.LeRobotDataset = LeRobotDataset
    _sys.modules["lerobot"] = _lerobot_root
    _sys.modules["lerobot.common"] = _lerobot_common
    _sys.modules["lerobot.common.datasets"] = _lerobot_common_datasets
    _sys.modules["lerobot.common.datasets.lerobot_dataset"] = _stub

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import yaml
from openpi.policies import policy_config as _policy_config
from openpi.policies import droid_policy as _droid_policy
from openpi.shared import download
from openpi.training import config as _config

_SCRIPT_DIR = Path(__file__).resolve().parent
_OPENPI_ROOT = _SCRIPT_DIR.parent.parent
_REPO_ROOT = _OPENPI_ROOT.parent
_sys.path.insert(0, str(_SCRIPT_DIR))
import utils as _utils

import time as _time

_sys.path.insert(0, str(_REPO_ROOT / "src"))
from libero_experiments import experiment_log as _xlog  # sheet-compatible logging (numpy-only module)


def _resolve_run_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    repo_path = _REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    openpi_path = _OPENPI_ROOT / path
    if openpi_path.exists():
        return openpi_path
    return path


def _setup_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging_handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="w"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=logging_handlers,
    )
    return logging.getLogger("openpi.experiments")


def _build_hyperactivate_mask(target_vectors, num_layers, hidden_dim):
    mask = jnp.zeros((num_layers, hidden_dim), dtype=jnp.bool)
    for target_vector in target_vectors:
        layer_idx = target_vector // hidden_dim
        neuron_idx = target_vector % hidden_dim
        mask = mask.at[layer_idx, neuron_idx].set(True)
    return mask


def _save_action_plot(action_no: np.ndarray, action_with: np.ndarray | None, diff: np.ndarray | None, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(action_no))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, action_no, marker="o", label="no_intervention")
    if action_with is not None:
        ax.plot(x, action_with, marker="o", label="with_intervention")
    if diff is not None:
        ax.bar(x, diff, alpha=0.3, label="diff")
    ax.set_xlabel("Action dimension")
    ax.set_ylabel("Value")
    ax.set_title("Action + Intervention Difference")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_jsonl(path: Path, row: dict):
    _utils.write_jsonl(path, row)


def _decode_instruction(raw) -> str:
    if hasattr(raw, "numpy"):
        raw = raw.numpy()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, np.ndarray):
        if raw.dtype.kind in {"S", "O"}:
            return str(raw.item())
        if raw.shape == ():
            return str(raw.item())
    return str(raw)


def _peak_device_memory_gb():
    try:
        stats = _jax.devices()[0].memory_stats() or {}
        peak = stats.get("peak_bytes_in_use")
        return peak / 1024**3 if peak else None
    except Exception:
        return None


def _layers_str(target_vectors, hidden_dim):
    counts = {}
    for v in target_vectors:
        counts[v // hidden_dim] = counts.get(v // hidden_dim, 0) + 1
    return ", ".join(f"L{layer}" + (f"×{c}" if c > 1 else "") for layer, c in sorted(counts.items()))


def _sheet_row(cfg, meta, run_name, run_dir, start_dt, hours, status, error_msg, samples, timings,
               target_vectors, git, versions, hardware, command):
    model_cfg = cfg.get("model", {})
    env_cfg = cfg.get("env", {})
    iv = cfg.get("intervention", {})
    enabled = iv.get("enabled", True)
    hidden_dim = iv.get("hidden_dim", 16384)
    price = meta.get("price_per_hour")
    fmt = _xlog._fmt

    n = len(samples)
    lat_no = float(np.mean(timings["no"]) * 1000) if timings["no"] else None
    lat_with = float(np.mean(timings["with"]) * 1000) if timings["with"] else None
    overhead = (lat_with - lat_no) / lat_no * 100 if (lat_no and lat_with) else None

    speed = obs_diff = ""
    if n:
        v_no = float(np.mean([s["joint_vel_norm_no"] for s in samples]))
        speed = f"‖q̇‖ (7 суглобів, сер. по чанку): {v_no:.3f}"
        if enabled:
            v_with = float(np.mean([s["joint_vel_norm_with"] for s in samples]))
            speed += f" → {v_with:.3f} з втручанням"
            d0 = np.array([s["diff_first"] for s in samples])
            mean_d = d0.mean(axis=0)
            top = np.argsort(-np.abs(mean_d))[:3]
            top_s = ", ".join(f"dim{j} {mean_d[j]:+.3f}" for j in top)
            obs_diff = (
                f"‖Δa₀‖ сер. {np.mean([s['diff_first_norm'] for s in samples]):.3f}; "
                f"‖ΔA‖ по чанку сер. {np.mean([s['diff_chunk_rms'] for s in samples]):.3f}; "
                f"найбільші зсуви першої дії: {top_s} (dim0–6 суглоби, dim7 гріпер)"
            )

    anomalies = [a for a in [obs_diff] if a]
    if status != "completed":
        anomalies.append(f"запуск не завершено: {status}" + (f" ({error_msg})" if error_msg else ""))
    if git.get("dirty"):
        anomalies.append("незакомічені зміни в коді")
    anomalies.append("офлайн-експеримент: кадри DROID, дії не виконуються, success rate не вимірюється")
    if meta.get("notes"):
        anomalies.append(meta["notes"])

    steps = [s["step"] for s in samples]
    prompts = []
    for s in samples:
        if s["prompt"] not in prompts:
            prompts.append(s["prompt"])

    row = {
        "exp_id": meta.get("exp_id") or run_name,
        "datetime": start_dt.strftime("%Y-%m-%d %H:%M (UTC%z)"),
        "author": meta.get("author") or "",
        "goal": meta.get("goal") or "",
        "status": status,
        "repo_commit": f"{git.get('remote') or _REPO_ROOT} @ {(git.get('commit') or '?')[:10]}"
        + (" (dirty)" if git.get("dirty") else ""),
        "command": command,
        "seed": "jax.random.key(0); greedy decoding (temperature 0)",
        "versions": ", ".join(f"{k} {v}" for k, v in versions.items()),
        "artifacts": meta.get("artifacts_url") or str(run_dir),
        "where": meta.get("where") or hardware.get("platform", ""),
        "gpu": hardware.get("gpu", ""),
        "cpu_ram": hardware.get("cpu_ram", ""),
        "hours": fmt(hours, 2),
        "cost": fmt(price * hours, 2) if price else "",
        "checkpoint": f"s3://openpi-assets/checkpoints/{model_cfg.get('checkpoint_name', 'pi0_fast_droid')} "
        f"(config {model_cfg.get('config_name', 'pi0_fast_hooked')})",
        "model_type": "π₀-FAST (PaliGemma: SigLIP + Gemma 2B), авторегресивні FAST action-токени, JAX",
        "precision": "bf16",
        "quant": "немає",
        "sim_task": f"офлайн, {env_cfg.get('dataset_name', 'droid_100')} ({env_cfg.get('dataset_split', 'train')}), "
        f"епізод {env_cfg.get('episode_index', 0)}, кроки {steps}",
        "n_rollouts": f"{n} кадрів (офлайн)",
        "n_init_states": "—",
        "instruction": " | ".join(prompts),
        "method": "FFN override: активації вибраних нейронів = α (вхід w_linear, gemma_fast_hooked)"
        if enabled else "без втручання",
        "vector": f"{iv.get('dict_name')} (нейронів: {len(target_vectors)}; шари {_layers_str(target_vectors, hidden_dim)}; "
        f"Gemma 2B у π₀-FAST)" if enabled else "—",
        "site": "PaliGemma Gemma 2B, layers[L].mlp (вхід down-проєкції), усі токени" if enabled else "—",
        "alpha": fmt(float(iv.get("coef", 1.0)), 2) if enabled else "0",
        "norm_ratio": "",
        "controls": meta.get("controls") or "парне порівняння: ті самі кадри без втручання",
        "success_rate": "н/з (офлайн)",
        "n_trials": str(n),
        "ci": "",
        "speed": speed,
        "latency_ms": fmt(lat_with if enabled else lat_no, 0),
        "peak_mem_gb": fmt(_peak_device_memory_gb(), 1),
        "overhead_pct": fmt(overhead, 1),
        "observations": "; ".join(anomalies),
        "conclusion": meta.get("conclusion") or "",
        "confidence": meta.get("confidence") or "",
        "next_step": meta.get("next_step") or "",
    }
    return row, {"latency_no_ms": lat_no, "latency_with_ms": lat_with, "overhead_pct": overhead}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values (dot notation), e.g. env.show_every=10",
    )
    parser.add_argument(
        "--interventions",
        default=str(_OPENPI_ROOT / "experiments/configs/interventions/dictionaries.yaml"),
        help="Path to intervention dictionaries YAML",
    )
    sheet = parser.add_argument_group("experiment sheet metadata (or EXP_* environment variables)")
    for name in ["exp_id", "author", "goal", "notes", "controls", "where", "artifacts_url"]:
        sheet.add_argument(f"--{name}")
    sheet.add_argument("--price_per_hour", type=float)
    args = parser.parse_args()
    meta = _xlog.build_meta(args)

    overrides = _utils.parse_overrides(args.override)
    cfg_path = _resolve_run_path(args.config)
    cfg = _utils.load_config(cfg_path, overrides)
    intervention_dict = _utils.load_intervention_dict(
        cfg.get("intervention", {}).get("dict_name", "high_cluster"),
        _resolve_run_path(args.interventions),
    )

    logging_cfg = cfg.get("logging", {})
    intervention_cfg = cfg.get("intervention", {})
    env_cfg = cfg.get("env", {})
    model_cfg = cfg.get("model", {})

    run_name = logging_cfg.get("run_name") or (
        f"INTERVENTION-{intervention_cfg.get('dict_name', 'unknown')}-coef{intervention_cfg.get('coef', 1.0)}-"
        f"{_datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
    )
    root_dir = _resolve_run_path(logging_cfg.get("root_dir", _OPENPI_ROOT / "experiments/logs"))
    run_dir = root_dir / run_name
    logger = _setup_logger(run_dir / "stdout.log")

    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    start_dt = _datetime.datetime.now().astimezone()
    t_start = _time.time()
    git, versions, hardware = _xlog.git_info(), _xlog.package_versions(), _xlog.hardware_info()
    command = _xlog.command_string()
    samples, timings = [], {"no": [], "with": []}
    status, error_msg = "interrupted", None

    try:
        logger.info("Getting model")
        train_config = _config.get_config(model_cfg.get("config_name", "pi0_fast_hooked"))
        checkpoint_dir = download.maybe_download(
            f"s3://openpi-assets/checkpoints/{model_cfg.get('checkpoint_name', 'pi0_fast_droid')}"
        )
        logger.info("Creating policy")
        policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)

        use_fake_data = env_cfg.get("use_fake_data", False) or env_cfg.get("dataset_name") == "fake"
        if use_fake_data:
            logger.info("Using fake dataset example")
            ds = [{"steps": [{"observation": _droid_policy.make_droid_example(), "language_instruction": "do something"}]}]
        else:
            logger.info("Loading dataset")
            ds = tfds.load(
                env_cfg.get("dataset_name", "droid_100"),
                data_dir=env_cfg.get("dataset_data_dir", "gs://gresearch/robotics/"),
                split=env_cfg.get("dataset_split", "train"),
            )

        hyperactivate_mask = None
        if intervention_cfg.get("enabled", True):
            hyperactivate_mask = _build_hyperactivate_mask(
                intervention_dict,
                intervention_cfg.get("num_layers", 18),
                intervention_cfg.get("hidden_dim", 16384),
            )

        # Original behaviour = 1 sample (the first matching step). env.num_samples > 1 keeps going.
        num_samples = int(env_cfg.get("num_samples", 1) or 1)
        max_steps = env_cfg.get("max_steps")
        show_every = env_cfg.get("show_every", 50)
        warmup = logging_cfg.get("warmup", True)
        warmed_up = False

        logger.info("Running inference")
        for i, episode in enumerate(ds):
            if i < env_cfg.get("episode_index", 0):
                continue
            if i > env_cfg.get("episode_index", 0):
                break
            for step_idx, step in enumerate(episode["steps"]):
                if max_steps is not None and step_idx >= max_steps:
                    break
                if step_idx % show_every != 0:
                    continue

                obs = step["observation"]
                instruction = _decode_instruction(step["language_instruction"])
                if "prompt" in obs and instruction in {"", "None"}:
                    instruction = _decode_instruction(obs["prompt"])
                static_example = {
                    "observation/exterior_image_1_left": np.asarray(obs["observation/exterior_image_1_left"])
                    if "observation/exterior_image_1_left" in obs
                    else obs["exterior_image_1_left"].numpy(),
                    "observation/wrist_image_left": np.asarray(obs["observation/wrist_image_left"])
                    if "observation/wrist_image_left" in obs
                    else obs["wrist_image_left"].numpy(),
                    "observation/joint_position": np.asarray(obs["observation/joint_position"])
                    if "observation/joint_position" in obs
                    else obs["joint_position"].numpy(),
                    "observation/gripper_position": np.asarray(obs["observation/gripper_position"])
                    if "observation/gripper_position" in obs
                    else obs["gripper_position"].numpy(),
                    "prompt": instruction,
                }

                if warmup and not warmed_up:
                    # First calls include JIT compilation; run them once untimed so latency is per-step cost.
                    logger.info("Warm-up (JIT compile, not timed)")
                    policy.infer(static_example)
                    if intervention_cfg.get("enabled", True):
                        policy.infer(static_example, hyperactivate_mask=hyperactivate_mask,
                                     hyperactivate_coeff=intervention_cfg.get("coef", 1.0))
                    warmed_up = True

                t0 = _time.perf_counter()
                result_no = policy.infer(static_example)
                timings["no"].append(_time.perf_counter() - t0)
                chunk_no = np.asarray(result_no["actions"])
                action_no = chunk_no[0]
                logger.info("No intervention: Step %s -> Action: %s", step_idx, action_no)

                action_with = None
                diff = None
                chunk_with = None
                if intervention_cfg.get("enabled", True):
                    t0 = _time.perf_counter()
                    result_with = policy.infer(
                        static_example,
                        hyperactivate_mask=hyperactivate_mask,
                        hyperactivate_coeff=intervention_cfg.get("coef", 1.0),
                    )
                    timings["with"].append(_time.perf_counter() - t0)
                    chunk_with = np.asarray(result_with["actions"])
                    action_with = chunk_with[0]
                    diff = action_with - action_no
                    logger.info("With intervention: Step %s -> Action: %s", step_idx, action_with)
                    logger.info("Intervention difference: %s", diff)

                if logging_cfg.get("save_actions", True):
                    _write_jsonl(
                        run_dir / "artifacts" / "actions.jsonl",
                        {
                            "step": int(step_idx),
                            "prompt": instruction,
                            "action_no_intervention": np.asarray(action_no).tolist(),
                            "action_with_intervention": None if action_with is None else np.asarray(action_with).tolist(),
                            "diff": None if diff is None else np.asarray(diff).tolist(),
                            "chunk_no_intervention": chunk_no.tolist(),
                            "chunk_with_intervention": None if chunk_with is None else chunk_with.tolist(),
                        },
                    )

                if logging_cfg.get("save_images", True):
                    _save_action_plot(
                        np.asarray(action_no),
                        None if action_with is None else np.asarray(action_with),
                        None if diff is None else np.asarray(diff),
                        run_dir / "artifacts" / "images" / f"step_{step_idx:04d}.png",
                    )

                sample = {
                    "step": int(step_idx),
                    "prompt": instruction,
                    "joint_vel_norm_no": float(np.linalg.norm(chunk_no[:, :7], axis=-1).mean()),
                }
                if chunk_with is not None:
                    h = min(len(chunk_no), len(chunk_with))
                    sample.update({
                        "joint_vel_norm_with": float(np.linalg.norm(chunk_with[:, :7], axis=-1).mean()),
                        "diff_first": np.asarray(diff).tolist(),
                        "diff_first_norm": float(np.linalg.norm(diff)),
                        "diff_chunk_rms": float(np.sqrt(np.mean(np.sum((chunk_with[:h] - chunk_no[:h]) ** 2, axis=-1)))),
                    })
                samples.append(sample)
                logger.info("[exp-log] sample %s/%s (step %s) done", len(samples), num_samples, step_idx)

                if len(samples) >= num_samples:
                    break
        status = "completed"
    except KeyboardInterrupt:
        status, error_msg = "interrupted", "Ctrl+C"
        raise
    except Exception as exc:
        status, error_msg = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        hours = (_time.time() - t_start) / 3600
        row, extra = _sheet_row(cfg, meta, run_name, run_dir, start_dt, hours, status, error_msg, samples, timings,
                                intervention_dict, git, versions, hardware, command)
        _xlog.write_run_outputs(run_dir, {
            "run_id": run_name, "status": status, "start": start_dt.isoformat(timespec="seconds"),
            "config": cfg, "meta": meta, "git": git, "versions": versions, "hardware": hardware,
            "samples": samples, "timing": extra, "row": row,
        })
        logger.info("[exp-log] DONE (%s): %s samples | row: %s", status, len(samples), run_dir / "sheet_row.tsv")


if __name__ == "__main__":
    main()
