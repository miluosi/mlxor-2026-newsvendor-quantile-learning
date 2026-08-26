"""Train and test only GenDFL-SQETO-IPA on the four d3group datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from model.rseto_ipa_spline import RSETOIPASplineNewsvendor
from real_world_d3group_gendfl_common import (
    SplineModelSpec,
    add_common_arguments,
    run_real_world,
    validate_common_arguments,
)


def train_sqeto_ipa(model, train_loader, validation_loader, args, checkpoint_path):
    return model.train_rseto_ipa_spline(
        train_loader,
        validation_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        step_size_exponent=args.step_size_exponent,
        early_stopping=args.early_stopping,
        warmup_epochs=args.warmup_epochs,
        min_delta_relative=args.min_delta_relative,
        replications=args.simulation_number,
        samples_per_replication=args.samples_per_replication,
        m_growth=args.m_growth,
        m_growth_exponent=args.m_growth_exponent,
        smoothing_mu=args.smoothing_mu,
        fidelity_weight=args.lambda_weight,
        training_seed=args.seed,
        parameter_box_lower=args.parameter_box_lower,
        parameter_box_upper=args.parameter_box_upper,
        stop_early=args.use_early_stopping,
        restore_best=args.use_early_stopping,
        max_simulation_values=args.max_simulation_values,
        diagnostic_interval=args.diagnostic_interval,
        finite_check_interval=args.finite_check_interval,
        train_data_on_device=args.train_data_on_device,
        checkpoint_path=checkpoint_path,
        verbose=args.verbose_training,
    )


def sqeto_ipa_settings(args):
    return {
        "simulation_number": int(args.simulation_number),
        "samples_per_replication": int(args.samples_per_replication),
        "m_growth": float(args.m_growth),
        "m_growth_exponent": float(args.m_growth_exponent),
        "lambda": float(args.lambda_weight),
        "smoothing_mu": float(args.smoothing_mu),
        "optimizer": "projected_sgd",
        "use_early_stopping": bool(args.use_early_stopping),
        "early_stopping_patience": int(args.early_stopping),
        "acceleration": "screen_and_replay",
        "diagnostic_interval": int(args.diagnostic_interval),
        "finite_check_interval": int(args.finite_check_interval),
        "train_data_on_device": bool(args.train_data_on_device),
        "step_size_exponent": float(args.step_size_exponent),
        "parameter_box": [
            float(args.parameter_box_lower),
            float(args.parameter_box_upper),
        ],
    }


MODEL_SPEC = SplineModelSpec(
    key="gendfl_sqeto_ipa",
    display_name="GenDFL-SQETO-IPA",
    model_class=RSETOIPASplineNewsvendor,
    trainer=train_sqeto_ipa,
    settings=sqeto_ipa_settings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_arguments(
        argparse.ArgumentParser(description=__doc__),
        default_output_dir=None,
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
    parser.add_argument(
        "--simulation-number",
        "--simulation_number",
        dest="simulation_number",
        type=int,
        default=16,
        help="Number R of independent IPA simulations per context.",
    )
    parser.add_argument(
        "--lambda",
        "--Lambda",
        dest="lambda_weight",
        type=float,
        default=0.5,
        help="NLL weight in lambda*NLL + (1-lambda)*IPA.",
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
    parser.add_argument("--m-growth", type=float, default=1.0)
    parser.add_argument("--m-growth-exponent", type=float, default=0.25)
    parser.add_argument(
        "--max-simulation-values",
        type=int,
        default=1048576,
        help="Maximum BRm values per no-grad screening chunk; tuned for a 24GB RTX 4090.",
    )
    parser.add_argument("--diagnostic-interval", type=int, default=100)
    parser.add_argument("--finite-check-interval", type=int, default=100)
    parser.add_argument(
        "--train-data-on-device",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--smoothing-mu", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir is None:
        lambda_slug = f"{args.lambda_weight:g}".replace(".", "p")
        args.output_dir = Path(
            "analysis_outputs/d3_real_world_gendfl_sqeto_ipa"
        ) / f"simulation_{args.simulation_number}_lambda_{lambda_slug}"
    validate_common_arguments(parser, args)
    if args.simulation_number < 1:
        parser.error("--simulation-number must be positive.")
    if args.samples_per_replication < 1:
        parser.error("--samples-per-replication must be positive.")
    if not 0.0 <= args.lambda_weight <= 1.0:
        parser.error("--lambda must lie in [0, 1].")
    if args.smoothing_mu <= 0:
        parser.error("--smoothing-mu must be positive.")
    if min(args.m_growth, args.m_growth_exponent) <= 0:
        parser.error("--m-growth and --m-growth-exponent must be positive.")
    if args.max_simulation_values < 1:
        parser.error("--max-simulation-values must be positive.")
    if min(args.diagnostic_interval, args.finite_check_interval) < 1:
        parser.error("Diagnostic and finite-check intervals must be positive.")
    run_real_world(
        args,
        [MODEL_SPEC],
        experiment_name="d3group_gendfl_sqeto_ipa",
        extra_metadata={"sqeto_ipa": sqeto_ipa_settings(args)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
