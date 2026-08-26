"""RSETO-IPA NLL-weight sensitivity on synthetic data."""

import argparse
import copy
import random
from pathlib import Path

from spline_sensitivity_common import (
    add_common_arguments,
    run_sensitivity,
    sensitivity_data_tag,
)


LAMBDA_TEST_LIST = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_DIMS = [14, 19, 24]
DEFAULT_FOLDS = list(range(10))


def glr_random_state_array():
    random.seed(42)
    return random.sample(range(1, 101), 10)


def run_default_grid(args):
    random_state_array = glr_random_state_array()
    invalid_folds = [fold for fold in args.folds if fold not in range(10)]
    if invalid_folds:
        raise ValueError(f"folds must lie in [0, 9], got {invalid_folds}.")
    if any(dim < 1 for dim in args.dims):
        raise ValueError("Every dimension must be positive.")

    jobs = [(dim, fold) for dim in args.dims for fold in args.folds]
    for job_index, (dim, fold) in enumerate(jobs, start=1):
        random_state = random_state_array[fold]
        run_args = copy.copy(args)
        run_args.dim = dim
        run_args.fold = fold
        run_args.job_index = job_index
        run_args.job_count = len(jobs)
        data_tag = sensitivity_data_tag(args)
        run_args.output_root = args.output_parent / (
            f"spline_sensitivity_{data_tag}_dim{dim}_seed{random_state}"
        )
        print(
            f"\n[job {job_index}/{len(jobs)}] sweep=lambda "
            f"dim={dim} fold={fold} random_state={random_state} "
            f"output={run_args.output_root}",
            flush=True,
        )
        run_sensitivity(run_args, "lambda", run_args.lambda_test_list)


def build_parser():
    parser = add_common_arguments(argparse.ArgumentParser())
    parser.add_argument(
        "--data_synthetic",
        choices=["exp5", "van-havre", "izbicki-bimodal"],
        default="exp5",
        help="Synthetic data type to use (default: exp5).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs for every model.",
    )
    parser.add_argument(
        "--use-early-stopping",
        "--ifearlystop",
        "--ifearflystop",
        dest="use_early_stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop on validation newsvendor loss and restore the best checkpoint.",
    )
    parser.add_argument(
        "--early-stopping",
        "--early-stopping-patience",
        dest="early_stopping",
        type=int,
        default=20,
        help="Validation patience when early stopping is enabled.",
    )
    parser.add_argument(
        "--samples-per-replication",
        "--mnum",
        "--m-num",
        dest="samples_per_replication",
        type=int,
        default=128,
        help="Initial samples m0 per IPA replication (default: 128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print the current hyperparameter and timing every N epochs; 0 disables it.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=DEFAULT_DIMS,
        help="Feature dimensions. Defaults to the five GLR dimensions.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=DEFAULT_FOLDS,
        help="GLR fold indices; folds 0-9 map to the ten seed-42 random states.",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("analysis_outputs"),
        help="Parent directory for per-dimension, per-seed result directories.",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run only --dim/--fold and use --output-root instead of the default grid.",
    )
    parser.add_argument(
        "--lambda-test-list",
        type=float,
        nargs="+",
        default=LAMBDA_TEST_LIST,
        help="NLL weights lambda in lambda*NLL + (1-lambda)*IPA.",
    )
    parser.add_argument(
        "--fixed-replications",
        type=int,
        default=16,
        help="Fixed independent IPA replications R while varying lambda.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.single_run:
        run_sensitivity(args, "lambda", args.lambda_test_list)
    else:
        run_default_grid(args)


if __name__ == "__main__":
    main()
