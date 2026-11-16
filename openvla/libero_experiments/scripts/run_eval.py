"""CLI entrypoint for a single LIBERO eval run."""

import argparse

from libero_experiments.config import load_config, parse_overrides
from libero_experiments.eval_libero import eval_libero
from libero_experiments.experiment_log import build_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values (dot notation), e.g. env.num_trials_per_task=5",
    )
    parser.add_argument(
        "--interventions",
        default="libero_experiments/configs/interventions/dictionaries.yaml",
        help="Path to intervention dictionaries YAML",
    )

    # Experiment-sheet metadata. Each can also be set once via an environment variable (in brackets).
    sheet = parser.add_argument_group("experiment sheet metadata")
    sheet.add_argument("--exp_id", help="ID експерименту, e.g. EXP-001 [EXP_ID]; default: run directory name")
    sheet.add_argument("--author", help="Автор [EXP_AUTHOR]")
    sheet.add_argument("--goal", help="Мета / гіпотеза [EXP_GOAL]")
    sheet.add_argument("--notes", help="Extra observations appended to the anomalies column [EXP_NOTES]")
    sheet.add_argument("--controls", help="Контролі; default: auto-links the latest baseline run [EXP_CONTROLS]")
    sheet.add_argument("--where", help="Де запускали; default: auto-detected (vast.ai / hostname) [EXP_WHERE]")
    sheet.add_argument("--artifacts_url", help="Link to uploaded logs; default: local run dir [EXP_ARTIFACTS_URL]")
    sheet.add_argument("--price_per_hour", type=float, help="Instance $/hour, for cost + live spend [EXP_PRICE_PER_HOUR]")
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    cfg = load_config(args.config, overrides)

    eval_libero(cfg, args.interventions, meta=build_meta(args))


if __name__ == "__main__":
    main()
