"""Experiment logging that mirrors the `experiments` tab of the vla_steering Google Sheet.

For every run this module writes into the run directory:
  run_info.json    - environment / hardware / config metadata (written at start, survives crashes)
  episodes.jsonl   - one JSON line per finished episode (success, steps, speed, jerk, latency, ...)
  progress.json    - live status: episodes done, success so far, elapsed time, ETA, money spent
  summary.json     - all aggregated metrics + the finished sheet row
  sheet_row.tsv    - the row, ready to paste into the sheet (click a cell in column A, Ctrl+V)
  sheet_row.csv    - same row with the header line (for File -> Import)

and regenerates, in the logs root, a combined table of all runs:
  experiments_table.tsv / experiments_table.csv
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# robosuite / LIBERO control frequency: one policy step = 1/20 s.
CONTROL_HZ = 20.0

REPO_ROOT = Path(__file__).resolve().parents[2]

# (internal key, column header exactly as in row 2 of the sheet), in sheet order.
COLUMNS: List[tuple] = [
    ("exp_id", "ID експерименту*"),
    ("datetime", "Дата й час*"),
    ("author", "Автор*"),
    ("goal", "Мета / гіпотеза*"),
    ("status", "Статус"),
    ("repo_commit", "Репозиторій і commit*"),
    ("command", "Конфіг і команда запуску*"),
    ("seed", "Seed*"),
    ("versions", "Версії (Python, PyTorch, CUDA, lerobot)"),
    ("artifacts", "Логи й артефакти (посилання)*"),
    ("where", "Де запускали*"),
    ("gpu", "GPU (модель, VRAM, кількість)*"),
    ("cpu_ram", "CPU / RAM"),
    ("hours", "Тривалість (год)"),
    ("cost", "Ціна ($)"),
    ("checkpoint", "Назва і чекпойнт (HF id)*"),
    ("model_type", "Тип моделі"),
    ("precision", "Точність*"),
    ("quant", "Метод квантизації"),
    ("sim_task", "Симулятор і задача (suite, task id)*"),
    ("n_rollouts", "Кількість роллаутів*"),
    ("n_init_states", "Початкових станів"),
    ("instruction", "Інструкція (промпт)"),
    ("method", "Метод*"),
    ("vector", "Вектор (риса, модель, шар)*"),
    ("site", "Місце втручання*"),
    ("alpha", "α*"),
    ("norm_ratio", "‖αv‖/‖x‖"),
    ("controls", "Контролі (random vector, target/off-target)"),
    ("success_rate", "Success rate (%)*"),
    ("n_trials", "Кількість спроб*"),
    ("ci", "Довірчий інтервал"),
    ("speed", "Швидкість середня"),
    ("sat", "SAT"),
    ("path_eff", "Path efficiency"),
    ("jerk", "Jerk"),
    ("clearance", "Clearance"),
    ("intent_rate", "Steering intent rate"),
    ("phase_metrics", "Метрики по фазах (підхід, контакт, транспорт)"),
    ("probe_acc", "Точність проби по шарах"),
    ("cos_sim", "Cosine similarity VLM–VLA"),
    ("judge", "Збіг судді з людьми"),
    ("latency_ms", "Затримка на крок (мс)"),
    ("peak_mem_gb", "Пікова пам'ять (ГБ)"),
    ("overhead_pct", "Оверхед steering (%)"),
    ("observations", "Припущення і спостереження (аномалії)*"),
    ("conclusion", "Висновок*"),
    ("confidence", "Довіра*"),
    ("next_step", "Наступний крок"),
]
COLUMN_KEYS = [k for k, _ in COLUMNS]
COLUMN_HEADERS = [h for _, h in COLUMNS]

# Concept (first token of the intervention dict name) -> (action index, expected sign).
# Action layout in LIBERO: [dx, dy, dz, droll, dpitch, dyaw, gripper].
CONCEPT_DIRECTIONS = {
    "up": (2, +1),
    "high": (2, +1),
    "raise": (2, +1),
    "down": (2, -1),
    "low": (2, -1),
}


# ----------------------------------------------------------------------------- helpers


def _run(cmd: Sequence[str], cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.check_output(list(cmd), cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return f"{x:.{nd}f}"
    return str(x)


def _nanmean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a binomial proportion (works for small n and k=0/n)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def concept_from_dict_name(dict_name: Optional[str]) -> Optional[str]:
    if not dict_name or dict_name == "blank":
        return None
    return dict_name.split("_")[0].lower()


# ----------------------------------------------------------------------------- environment info


def git_info() -> Dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], REPO_ROOT)
    remote = _run(["git", "config", "--get", "remote.origin.url"], REPO_ROOT)
    dirty = bool(_run(["git", "status", "--porcelain", "--untracked-files=no"], REPO_ROOT))
    return {"remote": remote, "commit": commit, "dirty": dirty}


def package_versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    if "torch" in sys.modules or importlib.util.find_spec("torch") is not None:
        try:
            import torch

            out["torch"] = torch.__version__
            out["cuda"] = torch.version.cuda or "cpu"
        except Exception:
            pass
    pkgs = ["transformers", "tokenizers", "timm", "flash-attn", "robosuite", "mujoco", "libero", "lerobot",
            "jax", "jaxlib", "flax"]
    for pkg in pkgs:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if driver:
        out["nvidia_driver"] = driver.splitlines()[0]
    return out


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"hostname": socket.gethostname()}
    # nvidia-smi first: works in any env (the JAX env has a CPU-only torch).
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if smi:
        gpus = [line.split(",") for line in smi.splitlines() if line.strip()]
        name, mem_mb = gpus[0][0].strip(), float(gpus[0][1])
        info["gpu"] = f"{name}, {mem_mb / 1024:.0f} GB, x{len(gpus)}"
    else:
        try:
            import torch

            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                props = torch.cuda.get_device_properties(0)
                info["gpu"] = f"{props.name}, {props.total_memory / 1024**3:.0f} GB, x{n}"
            else:
                info["gpu"] = "CPU only"
        except Exception:
            info["gpu"] = "unknown"

    cpu_model = platform.processor() or ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        n_cpu = len(os.sched_getaffinity(0))  # cores actually available to this container
    except AttributeError:
        n_cpu = os.cpu_count()
    ram_gb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    ram_gb = int(line.split()[1]) / 1024**2
                    break
    except OSError:
        pass
    info["cpu_ram"] = f"{cpu_model}, {n_cpu} cores" + (f", {ram_gb:.0f} GB RAM" if ram_gb else "")

    vast_label = os.environ.get("VAST_CONTAINERLABEL") or os.environ.get("CONTAINER_ID")
    info["platform"] = f"vast.ai ({vast_label})" if vast_label else info["hostname"]
    return info


# ----------------------------------------------------------------------------- per-episode tracking


class EpisodeTracker:
    """Collects per-step data for one rollout and turns it into behaviour metrics."""

    def __init__(self, task_id: int, task_description: str, episode_idx: int, max_steps: int):
        self.task_id = task_id
        self.task_description = task_description
        self.episode_idx = episode_idx
        self.max_steps = max_steps
        self.positions: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.latencies: List[float] = []
        self.error: Optional[str] = None
        self._t0 = time.time()

    def add_step(self, eef_pos: np.ndarray, action: np.ndarray, latency_s: float) -> None:
        self.positions.append(np.asarray(eef_pos, dtype=np.float64).copy())
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.latencies.append(latency_s)

    def finish(self, success: bool, final_eef_pos: Optional[np.ndarray], concept: Optional[str]) -> Dict[str, Any]:
        pos = list(self.positions)
        if final_eef_pos is not None and pos:
            pos.append(np.asarray(final_eef_pos, dtype=np.float64))
        pos_arr = np.array(pos) if pos else np.zeros((0, 3))
        act = np.array(self.actions) if self.actions else np.zeros((0, 7))

        mean_speed = path_eff = jerk = path_len = None
        if len(pos_arr) >= 2:
            seg = np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)
            path_len = float(seg.sum())
            mean_speed = float(seg.mean() * CONTROL_HZ)  # m/s
            disp = float(np.linalg.norm(pos_arr[-1] - pos_arr[0]))
            path_eff = disp / path_len if path_len > 1e-9 else None
        if len(pos_arr) >= 4:
            jerk = float(np.linalg.norm(np.diff(pos_arr, n=3, axis=0), axis=1).mean() * CONTROL_HZ**3)  # m/s^3

        intent_rate = None
        if concept in CONCEPT_DIRECTIONS and len(act):
            idx, sign = CONCEPT_DIRECTIONS[concept]
            intent_rate = float(np.mean(np.sign(act[:, idx]) == sign))

        n_steps = len(self.actions)
        return {
            "task_id": self.task_id,
            "task": self.task_description,
            "episode_idx": self.episode_idx,
            "success": bool(success),
            "steps": n_steps,
            "timeout": (not success) and self.error is None and n_steps >= self.max_steps,
            "error": self.error,
            "duration_s": time.time() - self._t0,
            "mean_speed_mps": mean_speed,
            "path_length_m": path_len,
            "path_efficiency": path_eff,
            "mean_jerk": jerk,
            "intent_rate": intent_rate,
            "mean_dz": float(act[:, 2].mean()) if len(act) else None,
            "frac_dz_pos": float(np.mean(act[:, 2] > 0)) if len(act) else None,
            "frac_dz_neg": float(np.mean(act[:, 2] < 0)) if len(act) else None,
            "mean_latency_ms": float(np.mean(self.latencies) * 1000) if self.latencies else None,
        }


# ----------------------------------------------------------------------------- run-level logger


class ExperimentLogger:
    def __init__(
        self,
        cfg: Any,
        run_dir: str,
        run_id: str,
        meta: Dict[str, Any],
        command: str,
        intervention_dict: Optional[Dict[int, List[str]]] = None,
        intermediate_size: int = 11008,
    ):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.meta = meta
        self.command = command
        self.intervention_dict = intervention_dict or {}
        self.intermediate_size = intermediate_size
        self.concept = concept_from_dict_name(cfg.intervention.dict_name) if cfg.intervention.enabled else None

        self.start_time = time.time()
        self.start_dt = datetime.now().astimezone()
        self.episodes: List[Dict[str, Any]] = []
        self.task_descriptions: List[str] = []
        self.planned_episodes: Optional[int] = None

        self.git = git_info()
        self.versions = package_versions()
        self.hardware = hardware_info()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.episodes_path.write_text("", encoding="utf-8")
        self._write_json(
            "run_info.json",
            {
                "run_id": self.run_id,
                "start": self.start_dt.isoformat(timespec="seconds"),
                "command": self.command,
                "meta": self.meta,
                "config": _cfg_to_dict(cfg),
                "git": self.git,
                "versions": self.versions,
                "hardware": self.hardware,
                "intervention_neurons": {str(k): v for k, v in self.intervention_dict.items()},
            },
        )
        if self.git.get("dirty"):
            print("[exp-log] WARNING: repo has uncommitted changes to tracked files; commit hash alone won't reproduce this run.")

    # -- live progress -------------------------------------------------------

    def set_plan(self, n_tasks: int, trials_per_task: int) -> None:
        self.planned_episodes = n_tasks * trials_per_task

    def log_episode(self, ep: Dict[str, Any]) -> None:
        self.episodes.append(ep)
        if ep["task"] not in self.task_descriptions:
            self.task_descriptions.append(ep["task"])
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

        n = len(self.episodes)
        k = sum(e["success"] for e in self.episodes)
        elapsed = time.time() - self.start_time
        avg_ep = np.mean([e["duration_s"] for e in self.episodes])
        remaining = (self.planned_episodes - n) * avg_ep if self.planned_episodes else None
        price = self.meta.get("price_per_hour")
        spent = price * elapsed / 3600 if price else None
        eta_cost = price * (elapsed + (remaining or 0)) / 3600 if price else None

        progress = {
            "run_id": self.run_id,
            "episodes_done": n,
            "episodes_planned": self.planned_episodes,
            "successes": k,
            "success_rate": k / n,
            "elapsed_min": elapsed / 60,
            "eta_min": remaining / 60 if remaining is not None else None,
            "spent_usd": spent,
            "projected_total_usd": eta_cost,
            "last_episode": ep,
            "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        self._write_json("progress.json", progress)

        status = "SUCCESS" if ep["success"] else ("ERROR" if ep["error"] else ("TIMEOUT" if ep["timeout"] else "FAIL"))
        line = (
            f"[exp-log] {n}/{self.planned_episodes or '?'} | task {ep['task_id']} ep {ep['episode_idx']}: {status} "
            f"({ep['steps']} steps, {ep['duration_s'] / 60:.1f} min, {_fmt(ep['mean_latency_ms'], 0)} ms/step) | "
            f"SR {k}/{n} = {100 * k / n:.0f}% | elapsed {elapsed / 60:.0f} min"
        )
        if remaining is not None:
            line += f" | ETA {remaining / 60:.0f} min"
        if spent is not None:
            line += f" | ~${spent:.2f} spent, ~${eta_cost:.2f} total"
        print(line, flush=True)

    # -- final row -----------------------------------------------------------

    def finalize(self, status: str, norm_summary: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> Dict[str, Any]:
        cfg = self.cfg
        end = time.time()
        hours = (end - self.start_time) / 3600
        n = len(self.episodes)
        k = sum(e["success"] for e in self.episodes)
        lo, hi = wilson_ci(k, n)

        peak_mem_gb = None
        try:
            import torch

            if torch.cuda.is_available():
                peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        except Exception:
            pass

        all_latency = _nanmean([e["mean_latency_ms"] for e in self.episodes])
        agg = {
            "episodes": n,
            "successes": k,
            "success_rate": k / n if n else None,
            "ci95": [lo, hi],
            "mean_speed_mps": _nanmean([e["mean_speed_mps"] for e in self.episodes]),
            "path_efficiency": _nanmean([e["path_efficiency"] for e in self.episodes]),
            "mean_jerk": _nanmean([e["mean_jerk"] for e in self.episodes]),
            "intent_rate": _nanmean([e["intent_rate"] for e in self.episodes]),
            "mean_dz": _nanmean([e["mean_dz"] for e in self.episodes]),
            "frac_dz_pos": _nanmean([e["frac_dz_pos"] for e in self.episodes]),
            "frac_dz_neg": _nanmean([e["frac_dz_neg"] for e in self.episodes]),
            "mean_steps": _nanmean([float(e["steps"]) for e in self.episodes]),
            "latency_ms": all_latency,
            "peak_mem_gb": peak_mem_gb,
            "timeouts": sum(1 for e in self.episodes if e["timeout"]),
            "errors": sum(1 for e in self.episodes if e["error"]),
            "hours": hours,
        }
        if norm_summary:
            agg["norms"] = norm_summary

        baseline = self._find_baseline() if cfg.intervention.enabled else None
        overhead = None
        if cfg.intervention.enabled and baseline and baseline.get("latency_ms") and all_latency:
            overhead = (all_latency - baseline["latency_ms"]) / baseline["latency_ms"] * 100
        elif not cfg.intervention.enabled:
            overhead = 0.0

        row = self._build_row(status, agg, baseline, overhead, error)
        summary = {"run_id": self.run_id, "status": status, "metrics": agg, "baseline_run": baseline, "row": row,
                   "meta": self.meta, "checkpoint": cfg.model.checkpoint, "suite": cfg.env.task_suite_name,
                   "intervention_enabled": cfg.intervention.enabled, "start": self.start_dt.isoformat(timespec="seconds")}
        self._write_json("summary.json", summary)
        _write_table(self.run_dir / "sheet_row", [row], with_header_csv=True)
        rebuild_combined_table(self.run_dir.parent)

        print(
            f"[exp-log] DONE ({status}): SR {k}/{n} = {_fmt(agg['success_rate'] and 100 * agg['success_rate'], 1)}% "
            f"[95% CI {100 * lo:.0f}-{100 * hi:.0f}%] | {hours * 60:.0f} min"
            + (f" | ~${row['cost']}" if row["cost"] else "")
        )
        print(f"[exp-log] Row for the sheet: {self.run_dir / 'sheet_row.tsv'}")
        print(f"[exp-log] All runs table:   {self.run_dir.parent / 'experiments_table.tsv'}")
        return summary

    def _build_row(self, status, agg, baseline, overhead, error) -> Dict[str, str]:
        cfg = self.cfg
        m = self.meta
        enabled = cfg.intervention.enabled
        price = m.get("price_per_hour")

        layers: Dict[int, int] = {}
        for flat_idx in self.intervention_dict:
            layers[flat_idx // self.intermediate_size] = layers.get(flat_idx // self.intermediate_size, 0) + 1
        layer_str = ", ".join(f"L{layer}×{cnt}" for layer, cnt in sorted(layers.items()))

        if cfg.model.load_in_4bit:
            precision, quant = "4-bit (bf16 compute)", "bitsandbytes 4-bit"
        elif cfg.model.load_in_8bit:
            precision, quant = "int8 (bf16 compute)", "bitsandbytes 8-bit"
        else:
            precision, quant = "bf16", "немає"

        task_ids = sorted({e["task_id"] for e in self.episodes})
        task_range = f"{task_ids[0]}–{task_ids[-1]}" if task_ids else "—"

        # behaviour cells
        speed = f"{_fmt(agg['mean_speed_mps'], 4)} м/с" if agg["mean_speed_mps"] is not None else ""
        jerk = f"{_fmt(agg['mean_jerk'], 1)} м/с³" if agg["mean_jerk"] is not None else ""
        if agg["intent_rate"] is not None:
            intent = f"{100 * agg['intent_rate']:.1f}% кроків у напрямку «{self.concept}»"
        else:
            intent = (
                f"dz>0: {_fmt(agg['frac_dz_pos'] and 100 * agg['frac_dz_pos'], 1)}%, "
                f"dz<0: {_fmt(agg['frac_dz_neg'] and 100 * agg['frac_dz_neg'], 1)}%"
                if agg["frac_dz_pos"] is not None
                else ""
            )

        norm_ratio = ""
        if agg.get("norms") and agg["norms"].get("mean_ratio") is not None:
            norm_ratio = _fmt(agg["norms"]["mean_ratio"], 3)

        controls = m.get("controls") or ""
        if not controls:
            if enabled and baseline:
                controls = f"baseline: {baseline['run_id']} (SR {_fmt(100 * baseline['success_rate'], 1)}%)"
            elif enabled:
                controls = "baseline ще не запускали"
            else:
                controls = "це і є baseline (без втручання)"

        anomalies = []
        if agg["errors"]:
            anomalies.append(f"{agg['errors']} епізод(ів) з exception")
        if agg["timeouts"]:
            anomalies.append(f"{agg['timeouts']} епізод(ів) дійшли до ліміту кроків")
        if status != "completed":
            anomalies.append(f"запуск не завершено: {status}" + (f" ({error})" if error else ""))
        if self.planned_episodes and len(self.episodes) < self.planned_episodes:
            anomalies.append(f"виконано {len(self.episodes)}/{self.planned_episodes} епізодів")
        if self.git.get("dirty"):
            anomalies.append("незакомічені зміни в коді")
        if agg["mean_steps"] is not None:
            anomalies.append(f"сер. довжина епізоду {agg['mean_steps']:.0f} кроків")
        if m.get("notes"):
            anomalies.append(m["notes"])

        commit = self.git.get("commit") or "?"
        repo = self.git.get("remote") or str(REPO_ROOT)
        versions = ", ".join(f"{k} {v}" for k, v in self.versions.items())

        values = {
            "exp_id": m.get("exp_id") or self.run_id,
            "datetime": self.start_dt.strftime("%Y-%m-%d %H:%M (UTC%z)"),
            "author": m.get("author") or "",
            "goal": m.get("goal") or "",
            "status": status,
            "repo_commit": f"{repo} @ {commit[:10]}" + (" (dirty)" if self.git.get("dirty") else ""),
            "command": self.command,
            "seed": str(cfg.env.seed),
            "versions": versions,
            "artifacts": m.get("artifacts_url") or str(self.run_dir),
            "where": m.get("where") or self.hardware.get("platform", ""),
            "gpu": self.hardware.get("gpu", ""),
            "cpu_ram": self.hardware.get("cpu_ram", ""),
            "hours": _fmt(agg["hours"], 2),
            "cost": _fmt(price * agg["hours"], 2) if price else "",
            "checkpoint": cfg.model.checkpoint,
            "model_type": "OpenVLA (Llama-2 7B + DINOv2/SigLIP), авторегресивні дискретні action-токени"
            if cfg.model.family == "openvla"
            else cfg.model.family,
            "precision": precision,
            "quant": quant,
            "sim_task": f"LIBERO, {cfg.env.task_suite_name}, task {task_range}",
            "n_rollouts": str(self.planned_episodes or len(self.episodes)),
            "n_init_states": f"{cfg.env.num_trials_per_task} на задачу (init states 0–{cfg.env.num_trials_per_task - 1})",
            "instruction": " | ".join(self.task_descriptions),
            "method": "FFN neuron override: активації вибраних нейронів = α (вхід down_proj)" if enabled else "baseline (без втручання)",
            "vector": f"{cfg.intervention.dict_name} (нейронів: {len(self.intervention_dict)}; шари {layer_str}; OpenVLA LLM)" if enabled else "—",
            "site": "language_model.model.layers[L].mlp.down_proj (input), усі токени" if enabled else "—",
            "alpha": _fmt(float(cfg.intervention.coef), 2) if enabled else "0",
            "norm_ratio": norm_ratio,
            "controls": controls,
            "success_rate": _fmt(agg["success_rate"] * 100, 1) if agg["success_rate"] is not None else "",
            "n_trials": str(agg["episodes"]),
            "ci": f"95% Wilson: {100 * agg['ci95'][0]:.1f}–{100 * agg['ci95'][1]:.1f}%" if agg["episodes"] else "",
            "speed": speed,
            "sat": "",
            "path_eff": _fmt(agg["path_efficiency"], 3),
            "jerk": jerk,
            "clearance": "",
            "intent_rate": intent,
            "phase_metrics": "",
            "probe_acc": "",
            "cos_sim": "",
            "judge": "",
            "latency_ms": _fmt(agg["latency_ms"], 0),
            "peak_mem_gb": _fmt(agg["peak_mem_gb"], 1),
            "overhead_pct": _fmt(overhead, 1) if overhead is not None else "",
            "observations": "; ".join(anomalies),
            "conclusion": m.get("conclusion") or "",
            "confidence": m.get("confidence") or "",
            "next_step": m.get("next_step") or "",
        }
        return {k: values.get(k, "") for k in COLUMN_KEYS}

    def _find_baseline(self) -> Optional[Dict[str, Any]]:
        """Most recent completed baseline run with the same checkpoint and suite in the same logs root."""
        best = None
        for path in self.run_dir.parent.glob("*/summary.json"):
            try:
                s = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (
                s.get("intervention_enabled") is False
                and s.get("status") == "completed"
                and s.get("checkpoint") == self.cfg.model.checkpoint
                and s.get("suite") == self.cfg.env.task_suite_name
            ):
                if best is None or s.get("start", "") > best.get("start", ""):
                    best = s
        if not best:
            return None
        return {
            "run_id": best["run_id"],
            "success_rate": best["metrics"]["success_rate"],
            "latency_ms": best["metrics"]["latency_ms"],
            "start": best.get("start"),
        }

    def _write_json(self, name: str, data: Any) -> None:
        tmp = self.run_dir / (name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.run_dir / name)


# ----------------------------------------------------------------------------- tables


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    from dataclasses import asdict, is_dataclass

    return asdict(cfg) if is_dataclass(cfg) else dict(cfg)


def _tsv_cell(v: str) -> str:
    return str(v).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _write_table(base_path: Path, rows: List[Dict[str, str]], with_header_csv: bool = True) -> None:
    """Write <base>.tsv (no header, paste-ready) and <base>.csv (with header, for import)."""
    with open(str(base_path) + ".tsv", "w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write("\t".join(_tsv_cell(row.get(k, "")) for k in COLUMN_KEYS) + "\n")
    with open(str(base_path) + ".csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if with_header_csv:
            w.writerow(COLUMN_HEADERS)
        for row in rows:
            w.writerow([row.get(k, "") for k in COLUMN_KEYS])


def rebuild_combined_table(logs_root: Path) -> Optional[Path]:
    """Collect sheet rows of every run under logs_root into experiments_table.{tsv,csv}, oldest first."""
    summaries = []
    for path in Path(logs_root).glob("*/summary.json"):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not summaries:
        return None
    summaries.sort(key=lambda s: s.get("start", ""))
    out = Path(logs_root) / "experiments_table"
    _write_table(out, [s["row"] for s in summaries])
    return out.with_suffix(".tsv")


def write_run_outputs(run_dir: Path, summary: Dict[str, Any]) -> None:
    """For runs that build their own row (e.g. pi0): write summary.json + sheet_row.* and refresh the combined table.

    `summary` must contain "row" (dict keyed by COLUMN_KEYS), "run_id", "status" and "start" (ISO time).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    row = {k: str(summary["row"].get(k, "") if summary["row"].get(k) is not None else "") for k in COLUMN_KEYS}
    summary = dict(summary, row=row)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_table(run_dir / "sheet_row", [row])
    rebuild_combined_table(run_dir.parent)


def build_meta(args: Any) -> Dict[str, Any]:
    """Merge CLI args with EXP_* environment variables (CLI wins)."""

    def pick(name: str, env: str) -> Optional[str]:
        val = getattr(args, name, None)
        return val if val not in (None, "") else os.environ.get(env) or None

    price = pick("price_per_hour", "EXP_PRICE_PER_HOUR")
    return {
        "exp_id": pick("exp_id", "EXP_ID"),
        "author": pick("author", "EXP_AUTHOR"),
        "goal": pick("goal", "EXP_GOAL"),
        "notes": pick("notes", "EXP_NOTES"),
        "controls": pick("controls", "EXP_CONTROLS"),
        "where": pick("where", "EXP_WHERE"),
        "artifacts_url": pick("artifacts_url", "EXP_ARTIFACTS_URL"),
        "price_per_hour": float(price) if price else None,
    }


def command_string() -> str:
    return "python " + " ".join(sys.argv)
