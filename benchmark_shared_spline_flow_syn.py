"""Fair synthetic comparison of three shared-backbone spline methods."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.special import ndtr
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model.gendfl_spline import GenDFLSplineNewsvendor
from model.rseto_ipa_spline import RSETOIPASplineNewsvendor
from model.spline_qfr import SplineQFRNewsvendor, pinball_loss


METHOD_LABELS = {
    "gendfl_spline": "Gen-DFL-Spline",
    "spline_qfr": "Spline-QFR",
    "rseto_ipa_spline": "RSETO-IPA-Spline",
}

SYNTHETIC_DATA_PROFILES = {
    "original": {
        "weight_logit_bias": 0.0,
        "separation_base": 1.7,
        "separation_amplitude": 0.35,
        "demand_scale": 1.0,
        "demand_shift": 0.0,
    },
    "rare_tail": {
        "weight_logit_bias": 2.2,
        "separation_base": 5.0,
        "separation_amplitude": 1.0,
        "demand_scale": 1.0,
        "demand_shift": 10.0,
    },
}


def resolve_synthetic_data_profile(args):
    """Resolve a named DGP profile while allowing explicit parameter overrides."""
    profile = SYNTHETIC_DATA_PROFILES[str(args.synthetic_data_mode)]
    for argument_name, profile_name in (
        ("mixture_weight_logit_bias", "weight_logit_bias"),
        ("mixture_separation_base", "separation_base"),
        ("mixture_separation_amplitude", "separation_amplitude"),
        ("mixture_demand_scale", "demand_scale"),
        ("mixture_demand_shift", "demand_shift"),
    ):
        value = getattr(args, argument_name)
        if value is None:
            setattr(args, argument_name, float(profile[profile_name]))
    return args


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


class ConditionalGaussianMixture1D:
    """A reproducible two-mode conditional target without point masses."""

    def __init__(
        self,
        context_dim,
        seed=42,
        *,
        weight_logit_bias=0.0,
        separation_base=1.7,
        separation_amplitude=0.35,
        demand_scale=1.0,
        demand_shift=0.0,
    ):
        self.context_dim = int(context_dim)
        if self.context_dim < 1:
            raise ValueError("context_dim must be positive.")
        rng = np.random.default_rng(int(seed))

        def unit_vector():
            vector = rng.normal(size=self.context_dim)
            return vector / max(np.linalg.norm(vector), 1e-12)

        self.weight_vector = 0.9 * unit_vector()
        self.common_vector = 0.8 * unit_vector()
        self.separation_vector = 0.35 * unit_vector()
        self.sigma1_vector = 0.9 * unit_vector()
        self.sigma2_vector = 0.9 * unit_vector()
        self.weight_logit_bias = float(weight_logit_bias)
        self.separation_base = float(separation_base)
        self.separation_amplitude = float(separation_amplitude)
        self.demand_scale = float(demand_scale)
        if self.demand_scale <= 0.0:
            raise ValueError("demand_scale must be positive.")
        self.demand_shift = float(demand_shift)

    def generation_config(self):
        return {
            "family": "context_dependent_two_component_gaussian_mixture",
            "context_distribution": "standard_normal",
            "weight_logit_bias": self.weight_logit_bias,
            "separation_base": self.separation_base,
            "separation_amplitude": self.separation_amplitude,
            "demand_scale": self.demand_scale,
            "demand_shift": self.demand_shift,
            "weight_vector_norm": float(np.linalg.norm(self.weight_vector)),
            "common_vector_norm": float(np.linalg.norm(self.common_vector)),
            "separation_vector_norm": float(np.linalg.norm(self.separation_vector)),
            "sigma1_vector_norm": float(np.linalg.norm(self.sigma1_vector)),
            "sigma2_vector_norm": float(np.linalg.norm(self.sigma2_vector)),
        }

    def parameters(self, context):
        context = np.asarray(context, dtype=np.float64)
        common = self.demand_scale * (context @ self.common_vector) + self.demand_shift
        separation = self.demand_scale * (
            self.separation_base
            + self.separation_amplitude * np.tanh(context @ self.separation_vector)
        )
        weight = sigmoid(self.weight_logit_bias + context @ self.weight_vector)
        mean1 = common - separation
        mean2 = common + separation
        sigma1 = self.demand_scale * (
            0.30 + 0.45 * sigmoid(context @ self.sigma1_vector)
        )
        sigma2 = self.demand_scale * (
            0.30 + 0.45 * sigmoid(context @ self.sigma2_vector)
        )
        return weight, mean1, mean2, sigma1, sigma2

    def sample(self, num_samples, seed):
        rng = np.random.default_rng(int(seed))
        context = rng.normal(size=(int(num_samples), self.context_dim))
        weight, mean1, mean2, sigma1, sigma2 = self.parameters(context)
        choose_first = rng.uniform(size=len(context)) < weight
        demand = np.where(
            choose_first,
            rng.normal(mean1, sigma1),
            rng.normal(mean2, sigma2),
        )
        return context.astype(np.float32), demand.reshape(-1, 1).astype(np.float32)

    def cdf(self, value, context):
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        weight, mean1, mean2, sigma1, sigma2 = self.parameters(context)
        return weight * ndtr((value - mean1) / sigma1) + (1.0 - weight) * ndtr(
            (value - mean2) / sigma2
        )

    def density(self, value, context):
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        weight, mean1, mean2, sigma1, sigma2 = self.parameters(context)
        normalizer = math.sqrt(2.0 * math.pi)
        density1 = np.exp(-0.5 * ((value - mean1) / sigma1) ** 2) / (
            normalizer * sigma1
        )
        density2 = np.exp(-0.5 * ((value - mean2) / sigma2) ** 2) / (
            normalizer * sigma2
        )
        return weight * density1 + (1.0 - weight) * density2

    def quantile(self, alpha, context, iterations=80):
        context = np.asarray(context, dtype=np.float64)
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one.")
        _, mean1, mean2, sigma1, sigma2 = self.parameters(context)
        lower = np.minimum(mean1 - 10.0 * sigma1, mean2 - 10.0 * sigma2)
        upper = np.maximum(mean1 + 10.0 * sigma1, mean2 + 10.0 * sigma2)
        for _ in range(int(iterations)):
            midpoint = 0.5 * (lower + upper)
            move_lower = self.cdf(midpoint, context) < alpha
            lower = np.where(move_lower, midpoint, lower)
            upper = np.where(move_lower, upper, midpoint)
        return 0.5 * (lower + upper)

    def expected_newsvendor_cost(self, decision, context, cost_under, cost_over):
        decision = np.asarray(decision, dtype=np.float64).reshape(-1)
        weight, mean1, mean2, sigma1, sigma2 = self.parameters(context)

        def component_cost(mean, sigma):
            standardized = (decision - mean) / sigma
            cdf = ndtr(standardized)
            density = np.exp(-0.5 * standardized**2) / math.sqrt(2.0 * math.pi)
            shortage = sigma * density + (mean - decision) * (1.0 - cdf)
            overage = sigma * density + (decision - mean) * cdf
            return float(cost_under) * shortage + float(cost_over) * overage

        return weight * component_cost(mean1, sigma1) + (1.0 - weight) * component_cost(
            mean2, sigma2
        )


def plot_mixture_target(mixture, context, demand, alpha, output_path):
    """Plot marginal and conditional structure of the synthetic target."""
    context = np.asarray(context, dtype=np.float64)
    demand = np.asarray(demand, dtype=np.float64).reshape(-1)
    weight, mean1, mean2, sigma1, sigma2 = mixture.parameters(context)
    oracle = mixture.quantile(alpha, context)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axis = axes[0, 0]
    axis.hist(demand, bins=55, density=True, color="#277da1", alpha=0.75)
    axis.axvline(np.quantile(demand, alpha), color="black", linestyle="--", linewidth=1.5)
    axis.set_title(
        f"Marginal target y (excess kurtosis={kurtosis(demand, fisher=True):.2f})"
    )
    axis.set_xlabel("Target demand y")
    axis.set_ylabel("Density")
    axis.grid(alpha=0.18)

    axis = axes[0, 1]
    center = np.zeros((1, mixture.context_dim), dtype=np.float64)
    center_weight, center_mean1, center_mean2, center_sigma1, center_sigma2 = (
        mixture.parameters(center)
    )
    low = float(np.minimum(center_mean1 - 6.0 * center_sigma1, center_mean2 - 6.0 * center_sigma2)[0])
    high = float(np.maximum(center_mean1 + 6.0 * center_sigma1, center_mean2 + 6.0 * center_sigma2)[0])
    grid = np.linspace(low, high, 1200)
    normalizer = math.sqrt(2.0 * math.pi)

    def density(values, mean, sigma):
        return np.exp(-0.5 * ((values - mean) / sigma) ** 2) / (normalizer * sigma)

    density1 = float(center_weight[0]) * density(
        grid, float(center_mean1[0]), float(center_sigma1[0])
    )
    density2 = (1.0 - float(center_weight[0])) * density(
        grid, float(center_mean2[0]), float(center_sigma2[0])
    )
    center_quantile = float(mixture.quantile(alpha, center)[0])
    axis.plot(grid, density1, color="#277da1", label="lower-demand component")
    axis.plot(grid, density2, color="#f8961e", label="rare high-demand component")
    axis.plot(grid, density1 + density2, color="black", linewidth=2.0, label="mixture")
    axis.axvline(
        center_quantile,
        color="#d1495b",
        linestyle="--",
        linewidth=1.6,
        label=f"oracle Q({alpha:.2f})={center_quantile:.2f}",
    )
    axis.set_title(f"Conditional density at x=0 (high-mode p={1-center_weight[0]:.3f})")
    axis.set_xlabel("Target demand y")
    axis.set_ylabel("Weighted density")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.18)

    axis = axes[1, 0]
    axis.hist(1.0 - weight, bins=35, color="#f8961e", alpha=0.8)
    axis.axvline(float(np.mean(1.0 - weight)), color="black", linestyle="--")
    axis.set_title("Context-dependent probability of the high-demand mode")
    axis.set_xlabel("P(high-demand component | x)")
    axis.set_ylabel("Contexts")
    axis.grid(alpha=0.18)

    axis = axes[1, 1]
    projection = context @ mixture.weight_vector
    axis.scatter(projection, oracle, s=10, alpha=0.35, color="#43aa8b")
    axis.set_title(f"Oracle contextual target quantile Q(x, {alpha:.2f})")
    axis.set_xlabel("Context projection controlling mixture weight")
    axis.set_ylabel("Oracle target quantile")
    axis.grid(alpha=0.18)

    figure.suptitle(
        "Rare-tail conditional Gaussian-mixture dataset",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_method_comparison(detail, output_path):
    """Plot paired expected newsvendor costs across training seeds."""
    order = ["gendfl_spline", "spline_qfr", "rseto_ipa_spline"]
    pivot = detail.pivot(index="seed", columns="method", values="expected_newsvendor_cost")
    pivot = pivot.reindex(columns=order)
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    positions = np.arange(len(order))
    colors = ["#222222", "#277da1", "#d1495b"]
    for seed, row in pivot.iterrows():
        axis.plot(positions, row.to_numpy(), color="#888888", alpha=0.55, linewidth=1.0)
        axis.scatter(positions, row.to_numpy(), color=colors, s=36, zorder=3)
    means = pivot.mean(axis=0).to_numpy()
    axis.plot(positions, means, color="black", linewidth=2.2, marker="D", markersize=6)
    axis.set_xticks(positions, ["GenDFL (NLL)", "QFlow", "RSETO-IPA"])
    axis.set_ylabel("Oracle expected newsvendor cost")
    axis.set_title("Paired comparison across training initializations")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def make_loader(context, demand, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.as_tensor(context), torch.as_tensor(demand)),
        batch_size=min(int(batch_size), len(context)),
        shuffle=bool(shuffle),
        generator=generator,
    )


def model_arguments(feature_dim, train_len, alpha, seed, args):
    return {
        "targetdim": 1,
        "labeldim": int(feature_dim),
        "latent": 1,
        "data_len": int(train_len),
        "epoch": int(args.epochs),
        "quantiles": float(alpha),
        "target_quantile": float(alpha),
        "cost_under": float(args.cost_under),
        "cost_over": float(args.cost_over),
        "random_seed": int(seed),
        "num_transforms": int(args.num_transforms),
        "num_bins": int(args.num_bins),
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "tail_bound": float(args.tail_bound),
        "tau_eps": float(args.tau_eps),
    }


def build_models(common, device, seed):
    set_seed(seed)
    gendfl = GenDFLSplineNewsvendor(**common).to(device)
    shared_state = copy.deepcopy(gendfl.state_dict())
    qfr = SplineQFRNewsvendor(**common).to(device)
    ipa = RSETOIPASplineNewsvendor(**common).to(device)
    qfr.load_state_dict(copy.deepcopy(shared_state), strict=True)
    ipa.load_state_dict(copy.deepcopy(shared_state), strict=True)
    models = {
        "gendfl_spline": gendfl,
        "spline_qfr": qfr,
        "rseto_ipa_spline": ipa,
    }
    counts = {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in models.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Shared-backbone parameter counts differ: {counts}")
    parameter_ids = [{id(parameter) for parameter in model.parameters()} for model in models.values()]
    if any(parameter_ids[i] & parameter_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Models unexpectedly share live parameter objects.")
    return models, counts, shared_state


def train_models(models, train_data, val_data, seed, args, output_dir):
    x_train, y_train = train_data
    x_val, y_val = val_data
    histories = {}
    elapsed = {}
    for name, model in models.items():
        set_seed(seed)
        train_loader = make_loader(x_train, y_train, args.batch_size, True, seed + 1000)
        val_loader = make_loader(x_val, y_val, args.batch_size, False, seed + 1001)
        checkpoint = output_dir / f"{name}_seed{seed}.pth"
        start = time.perf_counter()
        if name == "gendfl_spline":
            histories[name] = model.train_gendfl_spline(
                train_loader,
                val_loader,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                early_stopping=args.early_stopping,
                warmup_epochs=args.warmup_epochs,
                min_delta_relative=args.min_delta_relative,
                checkpoint_path=checkpoint,
            )
        elif name == "spline_qfr":
            histories[name] = model.train_spline_qfr(
                train_loader,
                val_loader,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                early_stopping=args.early_stopping,
                warmup_epochs=args.warmup_epochs,
                min_delta_relative=args.min_delta_relative,
                num_tau=args.qfr_levels,
                validation_num_tau=args.validation_qfr_levels,
                checkpoint_path=checkpoint,
            )
        else:
            histories[name] = model.train_rseto_ipa_spline(
                train_loader,
                val_loader,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                early_stopping=args.early_stopping,
                warmup_epochs=args.warmup_epochs,
                min_delta_relative=args.min_delta_relative,
                replications=args.ipa_replicates,
                samples_per_replication=args.ipa_samples,
                smoothing_mu=args.smoothing_mu,
                fidelity_weight=args.fidelity_weight,
                validation_seed=args.validation_seed,
                checkpoint_path=checkpoint,
            )
        elapsed[name] = time.perf_counter() - start
    return histories, elapsed


def inverse_target(values, scaler):
    values = np.asarray(values).reshape(-1, 1)
    return scaler.inverse_transform(values).reshape(-1)


def evaluate_models(
    models,
    x_test_scaled,
    y_test_scaled,
    x_test_raw,
    y_test_raw,
    target_scaler,
    mixture,
    alpha,
    seed,
    elapsed,
    parameter_counts,
    args,
):
    rows = []
    multi_ratio_rows = []
    predictions = pd.DataFrame({"y_true": y_test_raw.reshape(-1)})
    oracle_decision = mixture.quantile(alpha, x_test_raw)
    oracle_expected_cost = mixture.expected_newsvendor_cost(
        oracle_decision,
        x_test_raw,
        args.cost_under,
        args.cost_over,
    )
    tau_grid = torch.linspace(args.evaluation_tau_eps, 1.0 - args.evaluation_tau_eps, args.evaluation_tau_levels)

    for name, model in models.items():
        model.eval()
        device = next(model.parameters()).device
        method_decisions = []
        exact_decisions = []
        nll_sum = 0.0
        pinball_sum = 0.0
        calibration_sum = np.zeros(len(tau_grid), dtype=np.float64)
        observation_count = 0
        with torch.no_grad():
            for start in range(0, len(x_test_scaled), args.evaluation_batch_size):
                end = min(start + args.evaluation_batch_size, len(x_test_scaled))
                context = torch.as_tensor(x_test_scaled[start:end], device=device)
                demand = torch.as_tensor(y_test_scaled[start:end], device=device)
                exact_decision = model.critical_quantile_decision(context)
                method_decision = exact_decision
                batch_tau = tau_grid.to(device=device, dtype=context.dtype)
                batch_quantiles = model.quantile(batch_tau, context)
                expanded_tau = batch_tau.reshape(1, -1, 1).expand_as(batch_quantiles)
                expanded_target = demand[:, None, :].expand_as(batch_quantiles)
                batch_size = end - start
                nll_sum += float(model.generative_loss(demand, context)) * batch_size
                pinball_sum += float(
                    pinball_loss(expanded_target, batch_quantiles, expanded_tau).mean()
                ) * batch_size
                calibration_sum += (
                    (demand[:, None, :] <= batch_quantiles)
                    .float()
                    .mean(dim=0)
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                    * batch_size
                )
                observation_count += batch_size
                method_decisions.append(method_decision.cpu().numpy())
                exact_decisions.append(exact_decision.cpu().numpy())

        method_decision_raw = inverse_target(np.vstack(method_decisions), target_scaler)
        exact_decision_raw = inverse_target(np.vstack(exact_decisions), target_scaler)
        predictions[name] = method_decision_raw
        predictions[f"{name}_exact_map"] = exact_decision_raw
        realized_difference = y_test_raw.reshape(-1) - method_decision_raw
        underage = float(np.mean(args.cost_under * np.maximum(realized_difference, 0.0)))
        overage = float(np.mean(args.cost_over * np.maximum(-realized_difference, 0.0)))
        expected_cost = mixture.expected_newsvendor_cost(
            method_decision_raw,
            x_test_raw,
            args.cost_under,
            args.cost_over,
        )
        exact_map_expected_cost = mixture.expected_newsvendor_cost(
            exact_decision_raw,
            x_test_raw,
            args.cost_under,
            args.cost_over,
        )
        integrated_pinball = pinball_sum / observation_count
        empirical_calibration = calibration_sum / observation_count
        rows.append(
            {
                "seed": seed,
                "method": name,
                "method_label": METHOD_LABELS[name],
                "newsvendor_cost": underage + overage,
                "underage_cost": underage,
                "overage_cost": overage,
                "expected_newsvendor_cost": float(np.mean(expected_cost)),
                "normalized_regret": float(
                    np.mean(expected_cost - oracle_expected_cost)
                    / max(float(np.mean(oracle_expected_cost)), 1e-12)
                ),
                "service_level": float(np.mean(y_test_raw.reshape(-1) <= method_decision_raw)),
                "coverage_error": float(
                    abs(np.mean(y_test_raw.reshape(-1) <= method_decision_raw) - alpha)
                ),
                "critical_quantile_mae": float(np.mean(np.abs(method_decision_raw - oracle_decision))),
                "exact_map_newsvendor_cost": float(np.mean(exact_map_expected_cost)),
                "scaled_nll": nll_sum / observation_count,
                "scaled_integrated_pinball": integrated_pinball,
                "scaled_crps": 2.0 * integrated_pinball,
                "calibration_error": float(
                    np.mean(np.abs(empirical_calibration - tau_grid.numpy()))
                ),
                "elapsed_seconds": elapsed[name],
                "parameter_count": parameter_counts[name],
                "inference_mode": "direct_exact_spline_quantile",
                "inference_samples": 0,
            }
        )

        with torch.no_grad():
            context_all = torch.as_tensor(x_test_scaled, device=device)
            for ratio in args.evaluation_alphas:
                predicted = model.quantile(float(ratio), context_all)[:, 0, :].cpu().numpy()
                predicted_raw = inverse_target(predicted, target_scaler)
                ratio_oracle = mixture.quantile(ratio, x_test_raw)
                ratio_cost_under = float(ratio)
                ratio_cost_over = 1.0 - float(ratio)
                predicted_cost = mixture.expected_newsvendor_cost(
                    predicted_raw,
                    x_test_raw,
                    ratio_cost_under,
                    ratio_cost_over,
                )
                oracle_cost = mixture.expected_newsvendor_cost(
                    ratio_oracle,
                    x_test_raw,
                    ratio_cost_under,
                    ratio_cost_over,
                )
                multi_ratio_rows.append(
                    {
                        "seed": seed,
                        "method": name,
                        "method_label": METHOD_LABELS[name],
                        "alpha": float(ratio),
                        "expected_newsvendor_cost": float(np.mean(predicted_cost)),
                        "normalized_regret": float(
                            np.mean(predicted_cost - oracle_cost)
                            / max(float(np.mean(oracle_cost)), 1e-12)
                        ),
                        "critical_quantile_mae": float(
                            np.mean(np.abs(predicted_raw - ratio_oracle))
                        ),
                    }
                )
    return rows, multi_ratio_rows, predictions


def flatten_gradients(loss, model):
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def gradient_diagnostics(model, context, demand, seed, args):
    model.eval()
    exact_loss, _ = model.exact_task_objective(context, demand, args.smoothing_mu)
    exact_gradient = flatten_gradients(exact_loss, model).detach()
    exact_norm = float(exact_gradient.norm())
    rows = []
    replication_grid = sorted(set([1, min(4, args.ipa_replicates), args.ipa_replicates]))
    sample_grid = sorted(set([8, min(16, args.ipa_samples), args.ipa_samples]))
    for replications in replication_grid:
        for samples_per_replication in sample_grid:
            estimates = []
            start = time.perf_counter()
            for trial in range(args.gradient_trials):
                generator = torch.Generator(device=context.device).manual_seed(
                    int(seed + 100000 + trial)
                )
                _, details = model.rseto_ipa_objective(
                    context,
                    demand,
                    replications=replications,
                    samples_per_replication=samples_per_replication,
                    smoothing_mu=args.smoothing_mu,
                    fidelity_weight=0.0,
                    generator=generator,
                )
                estimates.append(flatten_gradients(details["ipa_task_loss"], model).detach())
            elapsed = time.perf_counter() - start
            stacked = torch.stack(estimates)
            mean_estimate = stacked.mean(dim=0)
            difference = stacked - exact_gradient
            cosine = torch.nn.functional.cosine_similarity(
                stacked,
                exact_gradient.unsqueeze(0).expand_as(stacked),
                dim=1,
                eps=1e-12,
            )
            rows.append(
                {
                    "seed": seed,
                    "replications": replications,
                    "samples_per_replication": samples_per_replication,
                    "trials": args.gradient_trials,
                    "exact_gradient_norm": exact_norm,
                    "relative_bias": float(
                        (mean_estimate - exact_gradient).norm() / (exact_gradient.norm() + 1e-12)
                    ),
                    "mse": float(difference.pow(2).sum(dim=1).mean()),
                    "variance": float((stacked - mean_estimate).pow(2).sum(dim=1).mean()),
                    "cosine_similarity_mean": float(cosine.mean()),
                    "elapsed_seconds": elapsed,
                    "peak_memory_mb": (
                        float(torch.cuda.max_memory_allocated(context.device) / 1024**2)
                        if context.is_cuda
                        else float("nan")
                    ),
                }
            )
    return rows


def mixture_checks(mixture, context, alpha):
    context = np.asarray(context[: min(len(context), 16)])
    oracle = mixture.quantile(alpha, context)
    cdf_error = np.abs(mixture.cdf(oracle, context) - alpha)
    _, mean1, mean2, sigma1, sigma2 = mixture.parameters(context)
    grid_lower = float(np.min(np.minimum(mean1 - 10.0 * sigma1, mean2 - 10.0 * sigma2)))
    grid_upper = float(np.max(np.maximum(mean1 + 10.0 * sigma1, mean2 + 10.0 * sigma2)))
    grid = np.linspace(grid_lower, grid_upper, 10000)
    integrals = []
    for row in context:
        repeated_context = np.repeat(row[None, :], len(grid), axis=0)
        integrals.append(float(np.trapezoid(mixture.density(grid, repeated_context), grid)))
    density_at_quantile = mixture.density(oracle, context)
    return {
        "maximum_oracle_cdf_error": float(np.max(cdf_error)),
        "maximum_density_integral_error": float(np.max(np.abs(np.asarray(integrals) - 1.0))),
        "minimum_density_at_critical_fractile": float(np.min(density_at_quantile)),
        "mean_density_at_critical_fractile": float(np.mean(density_at_quantile)),
    }


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/shared_spline_flow_syn"),
    )
    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--context-dim", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument(
        "--synthetic-data-mode",
        choices=tuple(SYNTHETIC_DATA_PROFILES),
        default="original",
        help=(
            "Synthetic target profile. 'original' preserves the previous balanced "
            "two-mode DGP; 'rare_tail' creates a rare, distant high-demand mode."
        ),
    )
    parser.add_argument(
        "--mixture-weight-logit-bias",
        type=float,
        default=None,
        help="Override the selected profile's lower-mode logit bias.",
    )
    parser.add_argument(
        "--mixture-separation-base",
        type=float,
        default=None,
        help="Override the selected profile's base half-distance between modes.",
    )
    parser.add_argument(
        "--mixture-separation-amplitude",
        type=float,
        default=None,
        help="Override the selected profile's context-dependent separation amplitude.",
    )
    parser.add_argument(
        "--mixture-demand-scale",
        type=float,
        default=None,
        help=(
            "Override the selected profile's positive target scale. This multiplies "
            "context effects, mode separation, and component standard deviations."
        ),
    )
    parser.add_argument(
        "--mixture-demand-shift",
        type=float,
        default=None,
        help=(
            "Override the selected profile's additive target shift. The rare-tail "
            "profile uses 10 so its fixed benchmark sample is strictly positive."
        ),
    )
    parser.add_argument(
        "--require-positive-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail before training unless every generated target is strictly positive.",
    )
    parser.add_argument("--training-seeds", type=parse_int_list, default=parse_int_list("42,43,44"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-delta-relative", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cost-under", type=float, default=7.0)
    parser.add_argument("--cost-over", type=float, default=3.0)
    parser.add_argument("--num-transforms", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--tail-bound", type=float, default=4.0)
    parser.add_argument("--tau-eps", type=float, default=1e-5)
    parser.add_argument("--qfr-levels", type=int, default=16)
    parser.add_argument("--validation-qfr-levels", type=int, default=99)
    parser.add_argument("--ipa-replicates", type=int, default=16)
    parser.add_argument("--ipa-samples", type=int, default=128)
    parser.add_argument("--smoothing-mu", type=float, default=0.05)
    parser.add_argument("--fidelity-weight", type=float, default=0.5)
    parser.add_argument("--validation-seed", type=int, default=1701)
    parser.add_argument(
        "--inference-samples",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=20260807,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-tau-levels", type=int, default=99)
    parser.add_argument("--evaluation-tau-eps", type=float, default=0.01)
    parser.add_argument(
        "--evaluation-alphas",
        type=parse_float_list,
        default=parse_float_list("0.25,0.50,0.75,0.90"),
    )
    parser.add_argument("--gradient-trials", type=int, default=8)
    parser.add_argument("--gradient-contexts", type=int, default=16)
    args = resolve_synthetic_data_profile(parser.parse_args())

    if args.num_samples < 20 or not args.training_seeds:
        raise ValueError("num_samples must be at least 20 and training_seeds cannot be empty.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixture = ConditionalGaussianMixture1D(
        args.context_dim,
        seed=args.data_seed,
        weight_logit_bias=args.mixture_weight_logit_bias,
        separation_base=args.mixture_separation_base,
        separation_amplitude=args.mixture_separation_amplitude,
        demand_scale=args.mixture_demand_scale,
        demand_shift=args.mixture_demand_shift,
    )
    context, demand = mixture.sample(args.num_samples, seed=args.data_seed + 1)
    target_sign_check = {
        "minimum": float(np.min(demand)),
        "maximum": float(np.max(demand)),
        "nonpositive_count": int(np.count_nonzero(demand <= 0.0)),
        "all_strictly_positive": bool(np.all(demand > 0.0)),
    }
    if args.require_positive_target and not target_sign_check["all_strictly_positive"]:
        raise ValueError(
            "--require-positive-target was set, but generated demand contains "
            f"{target_sign_check['nonpositive_count']} nonpositive values "
            f"(minimum={target_sign_check['minimum']:.6f})."
        )
    indices = np.arange(args.num_samples)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.data_seed,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=args.val_size,
        random_state=args.data_seed,
    )
    context_scaler = StandardScaler().fit(context[train_idx])
    target_scaler = StandardScaler().fit(demand[train_idx])
    context_scaled = context_scaler.transform(context).astype(np.float32)
    demand_scaled = target_scaler.transform(demand).astype(np.float32)
    alpha = args.cost_under / (args.cost_under + args.cost_over)
    checks = mixture_checks(mixture, context[test_idx], alpha)
    plot_mixture_target(
        mixture,
        context,
        demand,
        alpha,
        args.output_dir / "target_distribution.png",
    )

    detail_rows = []
    multi_ratio_rows = []
    gradient_rows = []
    consistency_rows = []
    for seed in args.training_seeds:
        print(f"[run] seed={seed} device={device} samples={args.num_samples}")
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
            float((initial_outputs[0] - output).abs().max()) for output in initial_outputs[1:]
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
            mixture,
            alpha,
            seed,
            elapsed,
            parameter_counts,
            args,
        )
        detail_rows.extend(rows)
        multi_ratio_rows.extend(ratio_rows)
        predictions.to_csv(args.output_dir / f"predictions_seed{seed}.csv", index=False)
        gradient_context = torch.as_tensor(
            context_scaled[val_idx[: args.gradient_contexts]],
            device=device,
        )
        gradient_demand = torch.as_tensor(
            demand_scaled[val_idx[: args.gradient_contexts]],
            device=device,
        )
        gradient_rows.extend(
            gradient_diagnostics(
                models["rseto_ipa_spline"],
                gradient_context,
                gradient_demand,
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
    detail.to_csv(args.output_dir / "detail.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    plot_method_comparison(detail, args.output_dir / "expected_cost_by_seed.png")
    paired_expected_cost = detail.pivot(
        index="seed",
        columns="method",
        values="expected_newsvendor_cost",
    ).reset_index()
    paired_expected_cost["rseto_improvement_over_gendfl_percent"] = 100.0 * (
        paired_expected_cost["gendfl_spline"]
        - paired_expected_cost["rseto_ipa_spline"]
    ) / paired_expected_cost["gendfl_spline"]
    paired_expected_cost["rseto_improvement_over_qflow_percent"] = 100.0 * (
        paired_expected_cost["spline_qfr"]
        - paired_expected_cost["rseto_ipa_spline"]
    ) / paired_expected_cost["spline_qfr"]
    paired_expected_cost.to_csv(
        args.output_dir / "paired_expected_cost_by_seed.csv",
        index=False,
    )
    multi_ratio.to_csv(args.output_dir / "multi_ratio_detail.csv", index=False)
    gradient_detail.to_csv(args.output_dir / "gradient_diagnostics.csv", index=False)
    consistency.to_csv(args.output_dir / "consistency_checks.csv", index=False)
    with pd.ExcelWriter(args.output_dir / "results.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        detail.to_excel(writer, sheet_name="detail", index=False)
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
        "profile": args.synthetic_data_mode,
        "parameter_seed": int(args.data_seed),
        "observation_seed": int(args.data_seed) + 1,
        **mixture.generation_config(),
        "target_sign_check": target_sign_check,
    }
    configuration["mixture_checks"] = checks
    with (args.output_dir / "config_and_mixture_checks.json").open("w") as handle:
        json.dump(configuration, handle, indent=2, allow_nan=True)
    print("\n" + summary.to_string(index=False))
    print("\nmixture_checks=" + json.dumps(checks, indent=2))
    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
