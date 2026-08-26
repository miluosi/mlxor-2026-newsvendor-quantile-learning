"""Train and test only GenDFL and QFlow on the four d3group datasets."""

from __future__ import annotations

import argparse

from model.gendfl_spline import GenDFLSplineNewsvendor
from model.spline_qfr import SplineQFRNewsvendor
from real_world_d3group_gendfl_common import (
    SplineModelSpec,
    add_common_arguments,
    run_real_world,
    validate_common_arguments,
)


def train_gendfl(model, train_loader, validation_loader, args, checkpoint_path):
    return model.train_gendfl_spline(
        train_loader,
        validation_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_name="projected_sgd",
        step_size_exponent=args.step_size_exponent,
        training_seed=args.seed,
        parameter_box_lower=args.parameter_box_lower,
        parameter_box_upper=args.parameter_box_upper,
        stop_early=args.use_early_stopping,
        restore_best=args.use_early_stopping,
        early_stopping=args.early_stopping,
        warmup_epochs=args.warmup_epochs,
        min_delta_relative=args.min_delta_relative,
        checkpoint_path=checkpoint_path,
        verbose=args.verbose_training,
    )


def train_qflow(model, train_loader, validation_loader, args, checkpoint_path):
    return model.train_spline_qfr(
        train_loader,
        validation_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_name="projected_sgd",
        step_size_exponent=args.step_size_exponent,
        training_seed=args.seed,
        parameter_box_lower=args.parameter_box_lower,
        parameter_box_upper=args.parameter_box_upper,
        stop_early=args.use_early_stopping,
        restore_best=args.use_early_stopping,
        early_stopping=args.early_stopping,
        warmup_epochs=args.warmup_epochs,
        min_delta_relative=args.min_delta_relative,
        num_tau=args.qflow_levels,
        validation_num_tau=args.qflow_validation_levels,
        checkpoint_path=checkpoint_path,
        verbose=args.verbose_training,
    )


def gendfl_settings(args):
    return {
        "training_objective": "conditional_nll_only",
        "optimizer": "projected_sgd",
        "use_early_stopping": bool(args.use_early_stopping),
        "early_stopping_patience": int(args.early_stopping),
        "step_size_exponent": float(args.step_size_exponent),
        "parameter_box": [
            float(args.parameter_box_lower),
            float(args.parameter_box_upper),
        ],
    }


def qflow_settings(args):
    return {
        "training_objective": "random_tau_integrated_pinball",
        "num_tau": int(args.qflow_levels),
        "validation_num_tau": int(args.qflow_validation_levels),
        "optimizer": "projected_sgd",
        "use_early_stopping": bool(args.use_early_stopping),
        "early_stopping_patience": int(args.early_stopping),
        "step_size_exponent": float(args.step_size_exponent),
        "parameter_box": [
            float(args.parameter_box_lower),
            float(args.parameter_box_upper),
        ],
    }


MODEL_SPECS = [
    SplineModelSpec(
        key="gendfl",
        display_name="GenDFL",
        model_class=GenDFLSplineNewsvendor,
        trainer=train_gendfl,
        settings=gendfl_settings,
    ),
    SplineModelSpec(
        key="qflow",
        display_name="QFlow",
        model_class=SplineQFRNewsvendor,
        trainer=train_qflow,
        settings=qflow_settings,
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_arguments(
        argparse.ArgumentParser(description=__doc__),
        default_output_dir="analysis_outputs/d3_real_world_gendfl_benchmark",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs for every model.",
    )
    parser.add_argument(
        "--use-early-stopping",
        "--ifearlystop",
        "--ifearflystop",
        dest="use_early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop on validation newsvendor loss and restore the best checkpoint.",
    )
    parser.add_argument(
        "--early-stopping",
        "--early-stopping-patience",
        dest="early_stopping",
        type=int,
        default=50,
        help="Validation patience when early stopping is enabled.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64).",
    )
    parser.add_argument("--qflow-levels", type=int, default=16)
    parser.add_argument("--qflow-validation-levels", type=int, default=99)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)
    if min(args.qflow_levels, args.qflow_validation_levels) < 1:
        parser.error("QFlow level counts must be positive.")
    run_real_world(
        args,
        MODEL_SPECS,
        experiment_name="d3group_gendfl_qflow_benchmark",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
