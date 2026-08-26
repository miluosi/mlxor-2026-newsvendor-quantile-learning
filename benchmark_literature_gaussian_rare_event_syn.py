"""Literature-grounded rare-event Gaussian-mixture newsvendor benchmark.

The conditional demand remains a Gaussian mixture.  Its standardized mixture
uses the rare-event GMM simulation design of Wang et al. (JMLR, 2024): a 10%
minor component and component means -1.5 and +1.5 with unit variance.  We map
the minor component to high demand so that it is decision-relevant at a 0.95
newsvendor critical ratio.  A bounded common context shift is the only
contextual extension; it does not change the component probability, separation,
or within-component variance.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from benchmark_shared_spline_flow_syn import (
    build_models,
    evaluate_models,
    gradient_diagnostics,
    model_arguments,
    parse_float_list,
    parse_int_list,
    plot_method_comparison,
    train_models,
)


LITERATURE = {
    "rare_event_gmm": {
        "citation": "Wang et al. (2024), Gaussian Mixture Model with Rare Events Data",
        "url": "https://jmlr.org/papers/volume25/23-1245/23-1245.pdf",
        "borrowed_design": (
            "10% minor component; standardized component means -1.5 and +1.5; "
            "unit component variance"
        ),
    },
    "separated_rare_component_gmm": {
        "citation": (
            "van Havre et al. (2015), Overfitting Bayesian Mixture Models with "
            "an Unknown Number of Components"
        ),
        "url": "https://doi.org/10.1371/journal.pone.0131739",
        "borrowed_design": (
            "Simulation 4: weights (0.60, 0.39, 0.01), means (6, 10, 20), "
            "and variances (1, 1, 0.5)"
        ),
    },
    "gaussian_mixture_newsvendor": {
        "citation": (
            "Esteban-Perez and Morales (2022), Partition-based distributionally "
            "robust optimization via optimal transport with order cone constraints"
        ),
        "url": "https://doi.org/10.1007/s10288-021-00484-z",
        "borrowed_design": "Gaussian-mixture demand and a 0.10 mixture weight in a newsvendor experiment",
    },
    "rare_demand_newsvendor": {
        "citation": "Ulubayova et al. (2026), Imbalanced neural newsvendor",
        "url": "https://doi.org/10.1007/s11081-026-10077-6",
        "borrowed_design": "Rare extreme demands are evaluated under asymmetric newsvendor costs",
    },
    "quantile_ipa": {
        "citation": "Jiang and Fu (2015), On Estimating Quantile Sensitivities via IPA",
        "url": "https://doi.org/10.1287/opre.2015.1356",
        "borrowed_design": "Batched IPA estimation of quantile sensitivities",
    },
}


class LiteratureGaussianRareEventDGP:
    """Conditional two-Gaussian mixture with a literature-calibrated rare mode."""

    STANDARDIZED_LOW_MEAN = -1.5
    STANDARDIZED_HIGH_MEAN = 1.5

    def __init__(
        self,
        context_dim,
        *,
        seed=42,
        rare_probability=0.10,
        component_sigma=50.0,
        demand_location=300.0,
        context_amplitude=25.0,
    ):
        self.context_dim = int(context_dim)
        self.rare_probability = float(rare_probability)
        self.component_sigma = float(component_sigma)
        self.demand_location = float(demand_location)
        self.context_amplitude = float(context_amplitude)
        if self.context_dim < 1:
            raise ValueError("context_dim must be positive.")
        if not 0.0 < self.rare_probability < 1.0:
            raise ValueError("rare_probability must lie strictly between zero and one.")
        if self.component_sigma <= 0.0 or self.context_amplitude < 0.0:
            raise ValueError("component_sigma must be positive and context_amplitude nonnegative.")

        rng = np.random.default_rng(int(seed))
        context_vector = rng.normal(size=self.context_dim)
        self.context_vector = context_vector / max(np.linalg.norm(context_vector), 1e-12)

    @property
    def regular_probability(self):
        return 1.0 - self.rare_probability

    def context_projection(self, context):
        context = np.asarray(context, dtype=np.float64)
        return context @ self.context_vector

    def parameters(self, context):
        context = np.asarray(context, dtype=np.float64)
        common_shift = self.demand_location + self.context_amplitude * np.tanh(
            self.context_projection(context)
        )
        mean_regular = common_shift + self.STANDARDIZED_LOW_MEAN * self.component_sigma
        mean_rare = common_shift + self.STANDARDIZED_HIGH_MEAN * self.component_sigma
        shape = mean_regular.shape
        regular_probability = np.full(shape, self.regular_probability, dtype=np.float64)
        sigma_regular = np.full(shape, self.component_sigma, dtype=np.float64)
        sigma_rare = np.full(shape, self.component_sigma, dtype=np.float64)
        return regular_probability, mean_regular, mean_rare, sigma_regular, sigma_rare

    def generation_config(self):
        return {
            "family": "conditional_two_component_gaussian_rare_event_mixture",
            "context_distribution": "standard_normal",
            "rare_component_role": "upper_demand_component",
            "rare_probability": self.rare_probability,
            "standardized_component_means": [
                self.STANDARDIZED_LOW_MEAN,
                self.STANDARDIZED_HIGH_MEAN,
            ],
            "standardized_component_standard_deviations": [1.0, 1.0],
            "component_sigma": self.component_sigma,
            "demand_location": self.demand_location,
            "context_shift": "context_amplitude * tanh(v^T x)",
            "context_amplitude": self.context_amplitude,
            "context_vector_norm": float(np.linalg.norm(self.context_vector)),
            "literature_calibrated": [
                "rare_probability",
                "standardized_component_means",
                "standardized_component_standard_deviations",
            ],
            "project_specific_extensions": [
                "minor component assigned to the upper-demand regime",
                "bounded shared contextual location shift",
                "positive affine demand-unit calibration",
            ],
        }

    def sample(self, num_samples, seed, *, return_component=False):
        rng = np.random.default_rng(int(seed))
        context = rng.normal(size=(int(num_samples), self.context_dim))
        regular_probability, mean_regular, mean_rare, sigma_regular, sigma_rare = self.parameters(
            context
        )
        is_rare = rng.uniform(size=len(context)) >= regular_probability
        demand = np.where(
            is_rare,
            rng.normal(mean_rare, sigma_rare),
            rng.normal(mean_regular, sigma_regular),
        )
        result = (
            context.astype(np.float32),
            demand.reshape(-1, 1).astype(np.float32),
        )
        if return_component:
            return (*result, is_rare.astype(np.int8))
        return result

    def cdf(self, value, context):
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        weight, mean_regular, mean_rare, sigma_regular, sigma_rare = self.parameters(context)
        return weight * ndtr((value - mean_regular) / sigma_regular) + (
            1.0 - weight
        ) * ndtr((value - mean_rare) / sigma_rare)

    def density(self, value, context):
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        weight, mean_regular, mean_rare, sigma_regular, sigma_rare = self.parameters(context)
        normalizer = math.sqrt(2.0 * math.pi)
        regular_density = np.exp(-0.5 * ((value - mean_regular) / sigma_regular) ** 2) / (
            normalizer * sigma_regular
        )
        rare_density = np.exp(-0.5 * ((value - mean_rare) / sigma_rare) ** 2) / (
            normalizer * sigma_rare
        )
        return weight * regular_density + (1.0 - weight) * rare_density

    def quantile(self, alpha, context, iterations=80):
        context = np.asarray(context, dtype=np.float64)
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one.")
        _, mean_regular, mean_rare, sigma_regular, sigma_rare = self.parameters(context)
        lower = np.minimum(mean_regular - 10.0 * sigma_regular, mean_rare - 10.0 * sigma_rare)
        upper = np.maximum(mean_regular + 10.0 * sigma_regular, mean_rare + 10.0 * sigma_rare)
        for _ in range(int(iterations)):
            midpoint = 0.5 * (lower + upper)
            move_lower = self.cdf(midpoint, context) < alpha
            lower = np.where(move_lower, midpoint, lower)
            upper = np.where(move_lower, upper, midpoint)
        return 0.5 * (lower + upper)

    def expected_newsvendor_cost(self, decision, context, cost_under, cost_over):
        decision = np.asarray(decision, dtype=np.float64).reshape(-1)
        weight, mean_regular, mean_rare, sigma_regular, sigma_rare = self.parameters(context)

        def component_cost(mean, sigma):
            standardized = (decision - mean) / sigma
            cdf = ndtr(standardized)
            density = np.exp(-0.5 * standardized**2) / math.sqrt(2.0 * math.pi)
            shortage = sigma * density + (mean - decision) * (1.0 - cdf)
            overage = sigma * density + (decision - mean) * cdf
            return float(cost_under) * shortage + float(cost_over) * overage

        return weight * component_cost(mean_regular, sigma_regular) + (
            1.0 - weight
        ) * component_cost(mean_rare, sigma_rare)


class LiteratureSeparatedRareGaussianDGP:
    """Conditional affine version of van Havre et al.'s Gaussian Sim 4."""

    RAW_WEIGHTS = np.asarray([0.60, 0.39, 0.01], dtype=np.float64)
    RAW_MEANS = np.asarray([6.0, 10.0, 20.0], dtype=np.float64)
    RAW_VARIANCES = np.asarray([1.0, 1.0, 0.5], dtype=np.float64)

    def __init__(
        self,
        context_dim,
        *,
        seed=42,
        demand_scale=20.0,
        demand_shift=30.0,
        context_amplitude=25.0,
    ):
        self.context_dim = int(context_dim)
        self.demand_scale = float(demand_scale)
        self.demand_shift = float(demand_shift)
        self.context_amplitude = float(context_amplitude)
        self.rare_probability = float(self.RAW_WEIGHTS[-1])
        self.regular_probability = 1.0 - self.rare_probability
        if self.context_dim < 1:
            raise ValueError("context_dim must be positive.")
        if self.demand_scale <= 0.0 or self.context_amplitude < 0.0:
            raise ValueError("demand_scale must be positive and context_amplitude nonnegative.")
        rng = np.random.default_rng(int(seed))
        context_vector = rng.normal(size=self.context_dim)
        self.context_vector = context_vector / max(np.linalg.norm(context_vector), 1e-12)

    def context_projection(self, context):
        context = np.asarray(context, dtype=np.float64)
        return context @ self.context_vector

    def component_parameters(self, context):
        context = np.asarray(context, dtype=np.float64)
        common_shift = self.context_amplitude * np.tanh(self.context_projection(context))
        means = (
            self.demand_shift
            + self.demand_scale * self.RAW_MEANS[None, :]
            + common_shift[:, None]
        )
        sigmas = np.broadcast_to(
            self.demand_scale * np.sqrt(self.RAW_VARIANCES)[None, :],
            means.shape,
        )
        weights = np.broadcast_to(self.RAW_WEIGHTS[None, :], means.shape)
        return weights, means, sigmas

    def generation_config(self):
        return {
            "family": "conditional_three_component_gaussian_rare_event_mixture",
            "literature_design": "van_havre_2015_simulation_4",
            "context_distribution": "standard_normal",
            "rare_component_role": "upper_demand_component",
            "rare_probability": self.rare_probability,
            "raw_component_weights": self.RAW_WEIGHTS.tolist(),
            "raw_component_means": self.RAW_MEANS.tolist(),
            "raw_component_variances": self.RAW_VARIANCES.tolist(),
            "demand_scale": self.demand_scale,
            "demand_shift": self.demand_shift,
            "context_shift": "context_amplitude * tanh(v^T x)",
            "context_amplitude": self.context_amplitude,
            "context_vector_norm": float(np.linalg.norm(self.context_vector)),
            "literature_calibrated": [
                "raw_component_weights",
                "raw_component_means",
                "raw_component_variances",
            ],
            "project_specific_extensions": [
                "the separated 1% component is treated as high demand",
                "bounded shared contextual location shift",
                "positive affine demand-unit calibration",
            ],
        }

    def sample(
        self,
        num_samples,
        seed,
        *,
        return_component=False,
        return_component_index=False,
    ):
        if return_component and return_component_index:
            raise ValueError(
                "Choose either the binary rare indicator or the full component index."
            )
        rng = np.random.default_rng(int(seed))
        context = rng.normal(size=(int(num_samples), self.context_dim))
        weights, means, sigmas = self.component_parameters(context)
        uniforms = rng.uniform(size=len(context))
        component_index = np.sum(
            uniforms[:, None] > np.cumsum(weights, axis=1),
            axis=1,
        )
        rows = np.arange(len(context))
        demand = rng.normal(means[rows, component_index], sigmas[rows, component_index])
        is_rare = component_index == (weights.shape[1] - 1)
        result = (
            context.astype(np.float32),
            demand.reshape(-1, 1).astype(np.float32),
        )
        if return_component_index:
            return (*result, component_index.astype(np.int8))
        if return_component:
            return (*result, is_rare.astype(np.int8))
        return result

    def cdf(self, value, context):
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        weights, means, sigmas = self.component_parameters(context)
        return np.sum(weights * ndtr((value[:, None] - means) / sigmas), axis=1)

    def density(self, value, context):
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        weights, means, sigmas = self.component_parameters(context)
        standardized = (value[:, None] - means) / sigmas
        component_density = np.exp(-0.5 * standardized**2) / (
            math.sqrt(2.0 * math.pi) * sigmas
        )
        return np.sum(weights * component_density, axis=1)

    def quantile(self, alpha, context, iterations=80):
        context = np.asarray(context, dtype=np.float64)
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one.")
        _, means, sigmas = self.component_parameters(context)
        lower = np.min(means - 10.0 * sigmas, axis=1)
        upper = np.max(means + 10.0 * sigmas, axis=1)
        for _ in range(int(iterations)):
            midpoint = 0.5 * (lower + upper)
            move_lower = self.cdf(midpoint, context) < alpha
            lower = np.where(move_lower, midpoint, lower)
            upper = np.where(move_lower, upper, midpoint)
        return 0.5 * (lower + upper)

    def expected_newsvendor_cost(self, decision, context, cost_under, cost_over):
        decision = np.asarray(decision, dtype=np.float64).reshape(-1)
        weights, means, sigmas = self.component_parameters(context)
        standardized = (decision[:, None] - means) / sigmas
        cdf = ndtr(standardized)
        density = np.exp(-0.5 * standardized**2) / math.sqrt(2.0 * math.pi)
        shortage = sigmas * density + (means - decision[:, None]) * (1.0 - cdf)
        overage = sigmas * density + (decision[:, None] - means) * cdf
        component_cost = float(cost_under) * shortage + float(cost_over) * overage
        return np.sum(weights * component_cost, axis=1)


def dgp_checks(dgp, context, alpha, observed_component):
    audit_context = np.asarray(context[: min(len(context), 32)], dtype=np.float64)
    oracle = dgp.quantile(alpha, audit_context)
    cdf_error = np.abs(dgp.cdf(oracle, audit_context) - alpha)
    if hasattr(dgp, "component_parameters"):
        weights, means, sigmas = dgp.component_parameters(audit_context)
        lower = float(np.min(means - 10.0 * sigmas))
        upper = float(np.max(means + 10.0 * sigmas))
        regular_probability = float(np.mean(np.sum(weights[:, :-1], axis=1)))
    else:
        weight, mean_regular, mean_rare, sigma_regular, sigma_rare = dgp.parameters(
            audit_context
        )
        lower = float(
            np.min(
                np.minimum(
                    mean_regular - 10 * sigma_regular,
                    mean_rare - 10 * sigma_rare,
                )
            )
        )
        upper = float(
            np.max(
                np.maximum(
                    mean_regular + 10 * sigma_regular,
                    mean_rare + 10 * sigma_rare,
                )
            )
        )
        regular_probability = float(np.mean(weight))
    grid = np.linspace(lower, upper, 20000)
    integrals = []
    for row in audit_context:
        repeated = np.repeat(row[None, :], len(grid), axis=0)
        integrals.append(float(np.trapezoid(dgp.density(grid, repeated), grid)))
    return {
        "maximum_oracle_cdf_error": float(np.max(cdf_error)),
        "maximum_density_integral_error": float(np.max(np.abs(np.asarray(integrals) - 1.0))),
        "minimum_density_at_critical_fractile": float(np.min(dgp.density(oracle, audit_context))),
        "configured_rare_probability": dgp.rare_probability,
        "observed_rare_fraction": float(np.mean(observed_component)),
        "mean_regular_probability": regular_probability,
    }


def plot_target(dgp, context, demand, component, alpha, output_path):
    context = np.asarray(context, dtype=np.float64)
    demand = np.asarray(demand, dtype=np.float64).reshape(-1)
    component = np.asarray(component, dtype=bool)
    oracle = dgp.quantile(alpha, context)
    projection = dgp.context_projection(context)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    bins = np.linspace(float(demand.min()), float(demand.max()), 55)
    axes[0].hist(demand[~component], bins=bins, density=True, alpha=0.72, label="Regular component")
    axes[0].hist(demand[component], bins=bins, density=True, alpha=0.72, label="Rare high component")
    axes[0].axvline(np.quantile(demand, alpha), color="black", linestyle="--", linewidth=1.4)
    axes[0].set_title(f"Rare-event Gaussian mixture (rare={component.mean():.3f})")
    axes[0].set_xlabel("Demand y")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)

    axes[1].scatter(projection[~component], demand[~component], s=10, alpha=0.32, label="Regular")
    axes[1].scatter(projection[component], demand[component], s=14, alpha=0.65, label="Rare high")
    order = np.argsort(projection)
    axes[1].plot(
        projection[order],
        oracle[order],
        color="black",
        linewidth=1.6,
        label=f"Oracle q({alpha:.3f}|x)",
    )
    axes[1].set_title("Conditional upper-tail decision")
    axes[1].set_xlabel("Fixed context projection v^T x")
    axes[1].set_ylabel("Demand / order quantity")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/literature_van_havre_sim4_alpha0995"),
    )
    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--context-dim", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument(
        "--literature-design",
        choices=("wang_2024_two_component", "van_havre_2015_sim4"),
        default="van_havre_2015_sim4",
    )
    parser.add_argument("--rare-probability", type=float, default=0.10)
    parser.add_argument("--component-sigma", type=float, default=50.0)
    parser.add_argument("--demand-location", type=float, default=300.0)
    parser.add_argument("--demand-scale", type=float, default=20.0)
    parser.add_argument("--demand-shift", type=float, default=30.0)
    parser.add_argument("--context-amplitude", type=float, default=25.0)
    parser.add_argument("--training-seeds", type=parse_int_list, default=parse_int_list("42,43,44"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-delta-relative", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--cost-under",
        type=float,
        default=None,
        help=(
            "Underage cost. By default this is 199 for van Havre Sim 4 "
            "(alpha=0.995) and 19 for the Wang two-component design (alpha=0.95)."
        ),
    )
    parser.add_argument("--cost-over", type=float, default=1.0)
    parser.add_argument("--num-transforms", type=int, default=1)
    parser.add_argument("--num-bins", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--tail-bound", type=float, default=4.0)
    parser.add_argument("--tau-eps", type=float, default=1e-5)
    parser.add_argument("--qfr-levels", type=int, default=16)
    parser.add_argument("--validation-qfr-levels", type=int, default=99)
    parser.add_argument("--ipa-replicates", type=int, default=16)
    parser.add_argument("--ipa-samples", type=int, default=128)
    parser.add_argument("--smoothing-mu", type=float, default=0.05)
    parser.add_argument("--fidelity-weight", type=float, default=0.5)
    parser.add_argument("--validation-seed", type=int, default=1701)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-tau-levels", type=int, default=99)
    parser.add_argument("--evaluation-tau-eps", type=float, default=0.01)
    parser.add_argument(
        "--evaluation-alphas",
        type=parse_float_list,
        default=parse_float_list("0.50,0.90,0.95"),
    )
    parser.add_argument("--gradient-trials", type=int, default=1)
    parser.add_argument("--gradient-contexts", type=int, default=4)
    parser.add_argument("--inference-samples", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--inference-seed", type=int, default=20260815, help=argparse.SUPPRESS)
    return parser


def main():
    args = build_parser().parse_args()
    if args.cost_under is None:
        args.cost_under = (
            199.0 if args.literature_design == "van_havre_2015_sim4" else 19.0
        )
    if args.num_samples < 20 or not args.training_seeds:
        raise ValueError("num_samples must be at least 20 and training_seeds cannot be empty.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.literature_design == "van_havre_2015_sim4":
        dgp = LiteratureSeparatedRareGaussianDGP(
            args.context_dim,
            seed=args.data_seed,
            demand_scale=args.demand_scale,
            demand_shift=args.demand_shift,
            context_amplitude=args.context_amplitude,
        )
    else:
        dgp = LiteratureGaussianRareEventDGP(
            args.context_dim,
            seed=args.data_seed,
            rare_probability=args.rare_probability,
            component_sigma=args.component_sigma,
            demand_location=args.demand_location,
            context_amplitude=args.context_amplitude,
        )
    context, demand, component = dgp.sample(
        args.num_samples,
        seed=args.data_seed + 1,
        return_component=True,
    )
    indices = np.arange(args.num_samples)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.data_seed,
        stratify=component,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=args.val_size,
        random_state=args.data_seed,
        stratify=component[train_val_idx],
    )
    split = np.full(args.num_samples, "train", dtype=object)
    split[val_idx] = "validation"
    split[test_idx] = "test"
    dataset = pd.DataFrame(context, columns=[f"x_{j}" for j in range(args.context_dim)])
    dataset["demand"] = demand.reshape(-1)
    dataset["is_rare_high_component"] = component
    dataset["split"] = split
    dataset.to_csv(args.output_dir / "generated_dataset.csv", index=False)

    context_scaler = StandardScaler().fit(context[train_idx])
    target_scaler = StandardScaler().fit(demand[train_idx])
    context_scaled = context_scaler.transform(context).astype(np.float32)
    demand_scaled = target_scaler.transform(demand).astype(np.float32)
    alpha = args.cost_under / (args.cost_under + args.cost_over)
    checks = dgp_checks(dgp, context[test_idx], alpha, component)
    target_sign_check = {
        "minimum": float(np.min(demand)),
        "maximum": float(np.max(demand)),
        "mean": float(np.mean(demand)),
        "standard_deviation": float(np.std(demand)),
        "nonpositive_count": int(np.count_nonzero(demand <= 0.0)),
        "all_strictly_positive": bool(np.all(demand > 0.0)),
    }
    plot_target(
        dgp,
        context,
        demand,
        component,
        alpha,
        args.output_dir / "target_distribution.png",
    )

    detail_rows = []
    multi_ratio_rows = []
    gradient_rows = []
    consistency_rows = []
    for seed in args.training_seeds:
        print(
            f"[run] seed={seed} device={device} samples={args.num_samples} "
            f"rare={dgp.rare_probability:.3f} alpha={alpha:.3f}"
        )
        common = model_arguments(args.context_dim, len(train_idx), alpha, seed, args)
        models, parameter_counts, initial_state = build_models(common, device, seed)
        common_noise = torch.randn(
            min(4, len(test_idx)),
            11,
            1,
            device=device,
            generator=torch.Generator(device=device).manual_seed(seed + 99),
        )
        common_context = torch.as_tensor(
            context_scaled[test_idx[: len(common_noise)]],
            device=device,
        )
        initial_outputs = [
            model.sample_from_base_noise(common_context, common_noise).detach().cpu()
            for model in models.values()
        ]
        max_initial_difference = max(
            float((initial_outputs[0] - output).abs().max())
            for output in initial_outputs[1:]
        )
        histories, elapsed = train_models(
            models,
            (context_scaled[train_idx], demand_scaled[train_idx]),
            (context_scaled[val_idx], demand_scaled[val_idx]),
            seed,
            args,
            args.output_dir,
        )
        rows, ratio_rows, predictions = evaluate_models(
            models,
            context_scaled[test_idx],
            demand_scaled[test_idx],
            context[test_idx],
            demand[test_idx],
            target_scaler,
            dgp,
            alpha,
            seed,
            elapsed,
            parameter_counts,
            args,
        )
        detail_rows.extend(rows)
        multi_ratio_rows.extend(ratio_rows)
        predictions.to_csv(args.output_dir / f"predictions_seed{seed}.csv", index=False)
        if args.gradient_trials > 0:
            gradient_rows.extend(
                gradient_diagnostics(
                    models["rseto_ipa_spline"],
                    torch.as_tensor(
                        context_scaled[val_idx[: args.gradient_contexts]],
                        device=device,
                    ),
                    torch.as_tensor(
                        demand_scaled[val_idx[: args.gradient_contexts]],
                        device=device,
                    ),
                    seed,
                    args,
                )
            )
        consistency_rows.append(
            {
                "seed": seed,
                "same_parameter_count": len(set(parameter_counts.values())) == 1,
                "max_initial_output_difference": max_initial_difference,
                "parameter_count": next(iter(parameter_counts.values())),
                "state_key_count": len(initial_state),
            }
        )
        with (args.output_dir / f"histories_seed{seed}.json").open("w") as handle:
            json.dump(histories, handle, indent=2, allow_nan=True)

    detail = pd.DataFrame(detail_rows)
    multi_ratio = pd.DataFrame(multi_ratio_rows)
    gradient_detail = pd.DataFrame(gradient_rows)
    consistency = pd.DataFrame(consistency_rows)
    summary = (
        detail.groupby(["method", "method_label"], as_index=False)
        .agg(
            newsvendor_cost_mean=("newsvendor_cost", "mean"),
            expected_newsvendor_cost_mean=("expected_newsvendor_cost", "mean"),
            normalized_regret_mean=("normalized_regret", "mean"),
            critical_quantile_mae_mean=("critical_quantile_mae", "mean"),
            service_level_mean=("service_level", "mean"),
            scaled_nll_mean=("scaled_nll", "mean"),
            scaled_crps_mean=("scaled_crps", "mean"),
            calibration_error_mean=("calibration_error", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            parameter_count=("parameter_count", "first"),
        )
        .sort_values("expected_newsvendor_cost_mean")
    )
    paired = detail.pivot(index="seed", columns="method", values="expected_newsvendor_cost").reset_index()
    paired["rseto_improvement_over_gendfl_percent"] = 100.0 * (
        paired["gendfl_spline"] - paired["rseto_ipa_spline"]
    ) / paired["gendfl_spline"]
    paired["rseto_improvement_over_qflow_percent"] = 100.0 * (
        paired["spline_qfr"] - paired["rseto_ipa_spline"]
    ) / paired["spline_qfr"]

    detail.to_csv(args.output_dir / "detail.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_expected_cost_by_seed.csv", index=False)
    multi_ratio.to_csv(args.output_dir / "multi_ratio_detail.csv", index=False)
    gradient_detail.to_csv(args.output_dir / "gradient_diagnostics.csv", index=False)
    consistency.to_csv(args.output_dir / "consistency_checks.csv", index=False)
    plot_method_comparison(detail, args.output_dir / "expected_cost_by_seed.png")
    with pd.ExcelWriter(args.output_dir / "results.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        detail.to_excel(writer, sheet_name="detail", index=False)
        paired.to_excel(writer, sheet_name="paired", index=False)
        multi_ratio.to_excel(writer, sheet_name="multiple_alphas", index=False)
        gradient_detail.to_excel(writer, sheet_name="gradient_diagnostics", index=False)
        consistency.to_excel(writer, sheet_name="consistency", index=False)

    configuration = vars(args).copy()
    configuration["output_dir"] = str(args.output_dir)
    configuration["device"] = str(device)
    configuration["alpha"] = alpha
    configuration["split_sizes"] = {
        "train": len(train_idx),
        "validation": len(val_idx),
        "test": len(test_idx),
    }
    configuration["data_generation"] = {
        "parameter_seed": int(args.data_seed),
        "observation_seed": int(args.data_seed) + 1,
        **dgp.generation_config(),
        "target_sign_check": target_sign_check,
    }
    configuration["dgp_checks"] = checks
    configuration["literature_basis"] = copy.deepcopy(LITERATURE)
    with (args.output_dir / "config_and_dgp_checks.json").open("w") as handle:
        json.dump(configuration, handle, indent=2, allow_nan=True)

    print("\n" + summary.to_string(index=False))
    print("\n" + paired.to_string(index=False))
    print("\ndgp_checks=" + json.dumps(checks, indent=2))
    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
