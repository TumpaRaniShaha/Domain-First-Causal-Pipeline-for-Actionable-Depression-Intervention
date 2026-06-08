"""Command-line entrypoint for the depression causal feature-selection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.causal.feature_selection import PipelineConfig, run_pipeline
from src.utils.helpers import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR


def _bool_arg(value: str) -> bool:
    if value.lower() in {"1", "true", "yes", "y"}:
        return True
    if value.lower() in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--row-subsample", type=int, default=30000)
    parser.add_argument("--shap-sample", type=int, default=2000)
    parser.add_argument("--pi-repeats", type=int, default=2)
    parser.add_argument("--ga-generations", type=int, default=8)
    parser.add_argument("--ga-population", type=int, default=12)
    parser.add_argument("--binary-outcome", type=_bool_arg, default=True)
    parser.add_argument("--force-gam", type=_bool_arg, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        row_subsample=args.row_subsample,
        shap_sample=args.shap_sample,
        pi_repeats=args.pi_repeats,
        ga_generations=args.ga_generations,
        ga_population=args.ga_population,
        binary_outcome=args.binary_outcome,
        force_gam=args.force_gam,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
