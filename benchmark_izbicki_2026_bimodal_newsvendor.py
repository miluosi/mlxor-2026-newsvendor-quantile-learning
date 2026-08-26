"""Benchmark GenDFL, QFlow, and RSETO-IPA on an all-active bimodal DGP.

The two-dimensional case reproduces ``make_bimodal_full`` from the public
repository accompanying arXiv:2603.26611. Higher dimensions use a project-specific
all-active projection extension. The newsvendor protocol is also project-specific.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import ScalarFormatter
from scipy.special import ndtr
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model.gendfl_spline import GenDFLSplineNewsvendor
from model.rseto_ipa_spline import RSETOIPASplineNewsvendor
from model.spline_qfr import SplineQFRNewsvendor


PUBLIC_REPOSITORY = "https://github.com/rizbicki/tabDensityComparisons"
PUBLIC_COMMIT = "fab6559adfa1a5fe45224f89baf209e276819382"
PUBLIC_SOURCE_FILE = "plot_bimodal_illustration.py"
METHODS = {
    "gendfl_spline": ("GenDFL", GenDFLSplineNewsvendor),
    "spline_qfr": ("QFlow", SplineQFRNewsvendor),
    "rseto_ipa_spline": ("RSETO-IPA", RSETOIPASplineNewsvendor),
}


def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested):
    requested = str(requested).lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


class IzbickiBimodalFullDGP:
    """All-active high-dimensional extension of the Figure-1 bimodal DGP.

    At two dimensions this is exactly the public ``make_bimodal_full`` model.
    Above two dimensions, every feature enters two normalized dense projections
    before the public mean, scale, and mixture-weight formulas are applied.
    """

    max_samples = 25_000

    def __init__(self, context_dim=2):
        self.context_dim = int(context_dim)
        if self.context_dim < 2:
            raise ValueError("The public bimodal DGP requires at least two features.")
        self.projection1, self.projection2 = self._build_projection_vectors(
            self.context_dim
        )

    @staticmethod
    def _build_projection_vectors(context_dim):
        if int(context_dim) == 2:
            return (
                np.asarray([1.0, 0.0], dtype=np.float64),
                np.asarray([0.0, 1.0], dtype=np.float64),
            )
        projection1 = np.ones(int(context_dim), dtype=np.float64)
        projection1 /= np.linalg.norm(projection1)
        projection2 = np.where(
            np.arange(int(context_dim)) % 2 == 0,
            1.0,
            -1.0,
        )
        projection2 -= np.dot(projection2, projection1) * projection1
        projection2 /= np.linalg.norm(projection2)
        if np.any(np.abs(projection1) <= 1e-12) or np.any(
            np.abs(projection2) <= 1e-12
        ):
            raise RuntimeError("All-active projection contains a zero coefficient.")
        return projection1, projection2

    def generation_config(self):
        return {
            "family": "izbicki_2026_bimodal_full_all_active_projection_v1",
            "public_repository": PUBLIC_REPOSITORY,
            "public_commit": PUBLIC_COMMIT,
            "public_source_file": PUBLIC_SOURCE_FILE,
            "context_distribution": "standard_normal",
            "effective_coordinates": ["u1=a^T*x", "u2=b^T*x"],
            "projection1": self.projection1.tolist(),
            "projection2": self.projection2.tolist(),
            "projection_vectors_are_orthonormal": True,
            "all_context_dimensions_active": True,
            "mu1": "2*u1 + u2",
            "mu2": "-u1 + 0.5*u2",
            "sigma1": "0.3 + 0.2*sigmoid(u1)",
            "sigma2": "0.3 + 0.3*sigmoid(u2)",
            "component1_probability": "sigmoid(u1 - 0.5*u2)",
            "maximum_nested_sample_size": self.max_samples,
            "context_dim": self.context_dim,
            "active_context_dimensions": self.context_dim,
            "nuisance_context_dimensions": 0,
            "multidimensional_extension": (
                "all dimensions enter two deterministic normalized dense projections; "
                "the two-dimensional case uses the public coordinates exactly"
            ),
        }

    def parameters(self, context):
        context = np.asarray(context, dtype=np.float64)
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError(
                f"context must have shape [n, {self.context_dim}], got {context.shape}."
            )
        effective1 = context @ self.projection1
        effective2 = context @ self.projection2
        mean1 = 2.0 * effective1 + effective2
        mean2 = -effective1 + 0.5 * effective2
        sigma1 = 0.3 + 0.2 * sigmoid(effective1)
        sigma2 = 0.3 + 0.3 * sigmoid(effective2)
        weight = sigmoid(effective1 - 0.5 * effective2)
        return weight, mean1, mean2, sigma1, sigma2

    def sample(self, num_samples, seed, *, return_component=False):
        num_samples = int(num_samples)
        if not 1 <= num_samples <= self.max_samples:
            raise ValueError(
                f"num_samples must lie in [1, {self.max_samples}] to preserve "
                "the public repository's nested-sample protocol."
            )
        rng = np.random.RandomState(int(seed) + 1000)
        context_all = rng.randn(self.max_samples, self.context_dim)
        epsilon1_all = rng.randn(self.max_samples)
        epsilon2_all = rng.randn(self.max_samples)
        uniform_all = rng.rand(self.max_samples)

        context = context_all[:num_samples]
        weight, mean1, mean2, sigma1, sigma2 = self.parameters(context)
        component1 = uniform_all[:num_samples] < weight
        demand = np.where(
            component1,
            mean1 + sigma1 * epsilon1_all[:num_samples],
            mean2 + sigma2 * epsilon2_all[:num_samples],
        )
        result = (
            context.astype(np.float32),
            demand.reshape(-1, 1).astype(np.float32),
        )
        if return_component:
            return (*result, component1.astype(np.int8))
        return result

    def cdf(self, value, context):
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        weight, mean1, mean2, sigma1, sigma2 = self.parameters(context)
        return weight * ndtr((value - mean1) / sigma1) + (
            1.0 - weight
        ) * ndtr((value - mean2) / sigma2)

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
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one.")
        context = np.asarray(context, dtype=np.float64)
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

        return weight * component_cost(mean1, sigma1) + (
            1.0 - weight
        ) * component_cost(mean2, sigma2)


def make_loader(context, demand, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(
            torch.as_tensor(context, dtype=torch.float32),
            torch.as_tensor(demand, dtype=torch.float32),
        ),
        batch_size=min(int(batch_size), len(context)),
        shuffle=bool(shuffle),
        generator=generator,
    )


def build_data(dgp, sample_size, seed, args):
    context_pool, demand_pool = dgp.sample(sample_size, seed=seed)
    train_indices, validation_indices = train_test_split(
        np.arange(sample_size),
        test_size=0.25,
        random_state=int(seed),
    )
    context_test, demand_test = dgp.sample(
        args.test_samples,
        seed=int(seed) + 100_000,
    )
    context_scaler = StandardScaler().fit(context_pool[train_indices])
    target_scaler = StandardScaler().fit(demand_pool[train_indices])
    context_pool_scaled = context_scaler.transform(context_pool).astype(np.float32)
    demand_pool_scaled = target_scaler.transform(demand_pool).astype(np.float32)
    context_test_scaled = context_scaler.transform(context_test).astype(np.float32)
    demand_test_scaled = target_scaler.transform(demand_test).astype(np.float32)
    return {
        "x_train": context_pool_scaled[train_indices],
        "y_train": demand_pool_scaled[train_indices],
        "x_validation": context_pool_scaled[validation_indices],
        "y_validation": demand_pool_scaled[validation_indices],
        "x_test": context_test_scaled,
        "y_test": demand_test_scaled,
        "x_test_raw": context_test,
        "y_test_raw": demand_test,
        "target_scaler": target_scaler,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }


def build_models(args, train_size, seed, device):
    alpha = args.cost_under / (args.cost_under + args.cost_over)
    kwargs = {
        "targetdim": 1,
        "labeldim": args.context_dim,
        "latent": 1,
        "data_len": int(train_size),
        "epoch": args.epochs,
        "quantiles": alpha,
        "target_quantile": alpha,
        "cost_under": args.cost_under,
        "cost_over": args.cost_over,
        "random_seed": int(seed),
        "num_transforms": args.num_transforms,
        "num_bins": args.num_bins,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "tail_bound": args.tail_bound,
        "tau_eps": args.tau_eps,
    }
    set_seed(seed)
    template = GenDFLSplineNewsvendor(**kwargs)
    initial_state = copy.deepcopy(template.state_dict())
    models = {}
    for method, (_, model_class) in METHODS.items():
        set_seed(seed)
        model = model_class(**kwargs)
        model.load_state_dict(copy.deepcopy(initial_state), strict=True)
        models[method] = model.to(device)
    parameter_counts = {
        method: sum(parameter.numel() for parameter in model.parameters())
        for method, model in models.items()
    }
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"Shared-backbone parameter counts differ: {parameter_counts}")
    initial_hash = hashlib.sha256()
    for name, tensor in initial_state.items():
        initial_hash.update(name.encode("utf-8"))
        initial_hash.update(np.ascontiguousarray(tensor.numpy()).tobytes())
    return models, parameter_counts, initial_hash.hexdigest()


def train_model(method, model, data, seed, args, checkpoint):
    train_loader = make_loader(
        data["x_train"], data["y_train"], args.batch_size, True, seed + 1000
    )
    validation_loader = make_loader(
        data["x_validation"],
        data["y_validation"],
        args.batch_size,
        False,
        seed + 1001,
    )
    common = {
        "num_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "step_size_exponent": args.step_size_exponent,
        "training_seed": seed + 2000,
        "parameter_box_lower": args.parameter_box_lower,
        "parameter_box_upper": args.parameter_box_upper,
        "stop_early": args.use_early_stopping,
        "restore_best": args.use_early_stopping,
        "early_stopping": args.early_stopping,
        "warmup_epochs": args.warmup_epochs,
        "min_delta_relative": args.min_delta_relative,
        "checkpoint_path": checkpoint,
    }
    if method == "gendfl_spline":
        return model.train_gendfl_spline(
            train_loader,
            validation_loader,
            optimizer_name="projected_sgd",
            **common,
        )
    if method == "spline_qfr":
        return model.train_spline_qfr(
            train_loader,
            validation_loader,
            optimizer_name="projected_sgd",
            num_tau=args.qfr_levels,
            validation_num_tau=args.validation_qfr_levels,
            **common,
        )
    return model.train_rseto_ipa_spline(
        train_loader,
        validation_loader,
        replications=args.ipa_replications,
        samples_per_replication=args.ipa_samples,
        m_growth=args.m_growth,
        m_growth_exponent=args.m_growth_exponent,
        smoothing_mu=args.smoothing_mu,
        fidelity_weight=args.fidelity_weight,
        max_simulation_values=args.max_simulation_values,
        diagnostic_interval=args.diagnostic_interval,
        finite_check_interval=args.finite_check_interval,
        train_data_on_device=args.train_data_on_device,
        **common,
    )


def predict_quantile(model, context, batch_size):
    device = next(model.parameters()).device
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(context), int(batch_size)):
            batch = torch.as_tensor(
                context[start : start + int(batch_size)],
                dtype=torch.float32,
                device=device,
            )
            predictions.append(model.critical_quantile_decision(batch).cpu().numpy())
    return np.vstack(predictions)


def evaluate_model(method, model, data, dgp, elapsed, history, args):
    alpha = args.cost_under / (args.cost_under + args.cost_over)
    prediction_scaled = predict_quantile(
        model,
        data["x_test"],
        args.evaluation_batch_size,
    )
    prediction = data["target_scaler"].inverse_transform(prediction_scaled).reshape(-1)
    demand = data["y_test_raw"].reshape(-1)
    oracle = dgp.quantile(alpha, data["x_test_raw"])
    expected_cost = dgp.expected_newsvendor_cost(
        prediction,
        data["x_test_raw"],
        args.cost_under,
        args.cost_over,
    )
    oracle_cost = dgp.expected_newsvendor_cost(
        oracle,
        data["x_test_raw"],
        args.cost_under,
        args.cost_over,
    )
    realized_cost = (
        args.cost_under * np.maximum(demand - prediction, 0.0)
        + args.cost_over * np.maximum(prediction - demand, 0.0)
    )
    return {
        "method": method,
        "method_label": METHODS[method][0],
        "expected_newsvendor_cost": float(np.mean(expected_cost)),
        "oracle_expected_cost": float(np.mean(oracle_cost)),
        "normalized_regret": float(
            np.mean(expected_cost - oracle_cost)
            / max(float(np.mean(oracle_cost)), 1e-12)
        ),
        "realized_newsvendor_cost": float(np.mean(realized_cost)),
        "critical_quantile_mae": float(np.mean(np.abs(prediction - oracle))),
        "service_level": float(np.mean(demand <= prediction)),
        "coverage_error": float(abs(np.mean(demand <= prediction) - alpha)),
        "elapsed_seconds": float(elapsed),
        "epochs_ran": int(history["epochs_ran"]),
        "best_epoch": int(history["best_epoch"]),
        "best_val_newsvendor": float(history["best_val_newsvendor"]),
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def argument_fingerprint(args, sample_size, seed):
    ignored = {"force", "output_dir", "device"}
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ignored
    }
    payload.update(
        sample_size=int(sample_size),
        seed=int(seed),
        public_commit=PUBLIC_COMMIT,
    )
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def run_job(dgp, sample_size, seed, args, device):
    job_dir = args.output_dir / f"n{sample_size}" / f"seed{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)
    fingerprint, job_config = argument_fingerprint(args, sample_size, seed)
    result_path = job_dir / "detail.csv"
    config_path = job_dir / "config.json"
    if result_path.exists() and config_path.exists() and not args.force:
        existing = json.loads(config_path.read_text())
        if existing.get("fingerprint") == fingerprint:
            print(f"[reuse] n={sample_size} seed={seed}", flush=True)
            return pd.read_csv(result_path)

    data = build_data(dgp, sample_size, seed, args)
    models, parameter_counts, initial_hash = build_models(
        args,
        len(data["x_train"]),
        seed,
        device,
    )
    rows = []
    histories = {}
    for method, model in models.items():
        print(f"  [train] n={sample_size} seed={seed} method={method}", flush=True)
        checkpoint = job_dir / f"{method}.pth"
        set_seed(seed)
        start = time.perf_counter()
        history = train_model(method, model, data, seed, args, checkpoint)
        synchronize(device)
        elapsed = time.perf_counter() - start
        histories[method] = history
        row = evaluate_model(method, model, data, dgp, elapsed, history, args)
        row.update(
            sample_size=int(sample_size),
            training_size=len(data["x_train"]),
            validation_size=len(data["x_validation"]),
            test_size=len(data["x_test"]),
            seed=int(seed),
            parameter_count=parameter_counts[method],
            initial_checkpoint_sha256=initial_hash,
        )
        rows.append(row)
        torch.save(model.state_dict(), checkpoint)
    detail = pd.DataFrame(rows)
    detail.to_csv(result_path, index=False)
    (job_dir / "histories.json").write_text(
        json.dumps(histories, indent=2, allow_nan=True)
    )
    config_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "arguments": job_config,
                "data_generation": dgp.generation_config(),
                "split_protocol": {
                    "total_sample_size": int(sample_size),
                    "train_fraction": 0.75,
                    "validation_fraction": 0.25,
                    "independent_oracle_test_contexts": int(args.test_samples),
                    "context_and_target_standardized_from_training_split": True,
                },
                "newsvendor_adaptation": {
                    "cost_under": args.cost_under,
                    "cost_over": args.cost_over,
                    "critical_ratio": args.cost_under
                    / (args.cost_under + args.cost_over),
                },
            },
            indent=2,
        )
    )
    return detail


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    valid_indices = np.flatnonzero(np.isfinite(p_values))
    if not len(valid_indices):
        return adjusted
    order = valid_indices[np.argsort(p_values[valid_indices])]
    running = 0.0
    count = len(valid_indices)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_significance(detail):
    rows = []
    for sample_size, group in detail.groupby("sample_size"):
        pivot = group.pivot(
            index="seed",
            columns="method",
            values="expected_newsvendor_cost",
        )
        ipa = pivot["rseto_ipa_spline"].to_numpy(dtype=float)
        for baseline in ("gendfl_spline", "spline_qfr"):
            baseline_values = pivot[baseline].to_numpy(dtype=float)
            difference = baseline_values - ipa
            sample_count = len(difference)
            mean_difference = float(np.mean(difference))
            if sample_count >= 2:
                standard_error = float(
                    np.std(difference, ddof=1) / math.sqrt(sample_count)
                )
                critical = float(student_t.ppf(0.975, sample_count - 1))
                t_p_value = float(
                    ttest_1samp(difference, 0.0, alternative="greater").pvalue
                )
                cohens_dz = float(
                    mean_difference
                    / max(float(np.std(difference, ddof=1)), 1e-12)
                )
                try:
                    wilcoxon_p = float(
                        wilcoxon(
                            difference,
                            alternative="greater",
                            zero_method="wilcox",
                        ).pvalue
                    )
                except ValueError:
                    wilcoxon_p = float("nan")
            else:
                standard_error = float("nan")
                critical = float("nan")
                t_p_value = float("nan")
                wilcoxon_p = float("nan")
                cohens_dz = float("nan")
            rows.append(
                {
                    "sample_size": int(sample_size),
                    "baseline": baseline,
                    "baseline_label": METHODS[baseline][0],
                    "comparand": "rseto_ipa_spline",
                    "comparand_label": "RSETO-IPA",
                    "paired_runs": sample_count,
                    "baseline_cost_mean": float(np.mean(baseline_values)),
                    "rseto_cost_mean": float(np.mean(ipa)),
                    "mean_cost_reduction": mean_difference,
                    "ci95_lower": mean_difference - critical * standard_error,
                    "ci95_upper": mean_difference + critical * standard_error,
                    "mean_relative_improvement_percent": float(
                        np.mean(100.0 * difference / baseline_values)
                    ),
                    "rseto_win_count": int(np.count_nonzero(difference > 0.0)),
                    "paired_t_one_sided_p": t_p_value,
                    "wilcoxon_one_sided_p": wilcoxon_p,
                    "cohens_dz": cohens_dz,
                }
            )
    result = pd.DataFrame(rows)
    result["paired_t_holm_p"] = holm_adjust(result["paired_t_one_sided_p"])
    result["significant_ipa_advantage_0.05"] = (
        (result["paired_t_holm_p"] < 0.05) & (result["ci95_lower"] > 0.0)
    )
    return result


def plot_results(detail, significance, output_dir):
    method_order = list(METHODS)
    colors = {
        "gendfl_spline": "#222222",
        "spline_qfr": "#277da1",
        "rseto_ipa_spline": "#d1495b",
    }
    summary = (
        detail.groupby(["sample_size", "method"], as_index=False)
        .agg(
            mean=("expected_newsvendor_cost", "mean"),
            std=("expected_newsvendor_cost", "std"),
            count=("expected_newsvendor_cost", "size"),
        )
        .sort_values("sample_size")
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.2), dpi=180, constrained_layout=True)
    for method in method_order:
        method_data = summary[summary["method"] == method]
        error = 1.96 * method_data["std"] / np.sqrt(method_data["count"])
        axis.errorbar(
            method_data["sample_size"],
            method_data["mean"],
            yerr=error,
            marker="o",
            linewidth=2.2,
            capsize=4,
            color=colors[method],
            label=METHODS[method][0],
        )
    axis.set_xscale("log")
    axis.set_xticks(sorted(detail["sample_size"].unique()))
    axis.get_xaxis().set_major_formatter(ScalarFormatter())
    axis.set_xlabel("Total sample size n")
    axis.set_ylabel("Oracle expected newsvendor cost")
    axis.set_title("All-active conditional bimodal DGP: paired mean and 95% CI")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_dir / "expected_cost_comparison.png", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=180, constrained_layout=True)
    for axis, baseline in zip(axes, ("gendfl_spline", "spline_qfr")):
        subset = significance[significance["baseline"] == baseline].sort_values(
            "sample_size"
        )
        center = subset["mean_relative_improvement_percent"].to_numpy()
        absolute_center = subset["mean_cost_reduction"].to_numpy()
        lower = subset["ci95_lower"].to_numpy()
        upper = subset["ci95_upper"].to_numpy()
        scale = np.divide(
            center,
            absolute_center,
            out=np.zeros_like(center),
            where=np.abs(absolute_center) > 1e-12,
        )
        error = np.vstack(
            (
                (absolute_center - lower) * np.abs(scale),
                (upper - absolute_center) * np.abs(scale),
            )
        )
        axis.errorbar(
            subset["sample_size"],
            center,
            yerr=error,
            marker="o",
            linewidth=2.0,
            capsize=4,
            color="#d1495b",
        )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        axis.set_xscale("log")
        axis.set_xticks(subset["sample_size"])
        axis.get_xaxis().set_major_formatter(ScalarFormatter())
        axis.set_xlabel("Sample size n")
        axis.set_ylabel("RSETO-IPA improvement (%)")
        axis.set_title(f"RSETO-IPA vs {METHODS[baseline][0]}")
        axis.grid(alpha=0.25)
    figure.savefig(output_dir / "paired_ipa_improvement.png", bbox_inches="tight")
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/izbicki_2026_bimodal_newsvendor"),
    )
    parser.add_argument("--sample-sizes", type=parse_int_list, default=[50, 200, 2000])
    parser.add_argument(
        "--training-seeds",
        type=parse_int_list,
        default=list(range(42, 52)),
    )
    parser.add_argument("--context-dim", type=int, default=2)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping", type=int, default=20)
    parser.add_argument(
        "--use-early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-delta-relative", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--step-size-exponent", type=float, default=0.6)
    parser.add_argument("--parameter-box-lower", type=float, default=-10.0)
    parser.add_argument("--parameter-box-upper", type=float, default=10.0)
    parser.add_argument("--cost-under", type=float, default=19.0)
    parser.add_argument("--cost-over", type=float, default=1.0)
    parser.add_argument("--num-transforms", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--tail-bound", type=float, default=4.0)
    parser.add_argument("--tau-eps", type=float, default=1e-5)
    parser.add_argument("--qfr-levels", type=int, default=16)
    parser.add_argument("--validation-qfr-levels", type=int, default=99)
    parser.add_argument("--ipa-replications", type=int, default=16)
    parser.add_argument("--ipa-samples", type=int, default=128)
    parser.add_argument("--m-growth", type=float, default=1.0)
    parser.add_argument("--m-growth-exponent", type=float, default=0.25)
    parser.add_argument("--smoothing-mu", type=float, default=0.05)
    parser.add_argument("--fidelity-weight", type=float, default=0.5)
    parser.add_argument("--max-simulation-values", type=int, default=1_048_576)
    parser.add_argument("--diagnostic-interval", type=int, default=100)
    parser.add_argument("--finite-check-interval", type=int, default=100)
    parser.add_argument(
        "--train-data-on-device",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force", action="store_true")
    return parser


def validate_args(args):
    if not args.sample_sizes or not args.training_seeds:
        raise ValueError("sample_sizes and training_seeds cannot be empty.")
    if min(args.sample_sizes) < 8 or max(args.sample_sizes) > 25_000:
        raise ValueError("sample sizes must lie in [8, 25000].")
    if not 1 <= args.test_samples <= 25_000:
        raise ValueError("test_samples must lie in [1, 25000].")
    if args.context_dim < 2:
        raise ValueError("context_dim must be at least two.")
    if min(args.epochs, args.early_stopping, args.batch_size) < 1:
        raise ValueError("training controls must be positive.")
    if not 0.0 < args.cost_under or not 0.0 < args.cost_over:
        raise ValueError("newsvendor costs must be positive.")
    if not 0.5 < args.step_size_exponent <= 1.0:
        raise ValueError("step_size_exponent must lie in (0.5, 1].")


def main():
    args = build_parser().parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dgp = IzbickiBimodalFullDGP(args.context_dim)
    all_results = []
    total_jobs = len(args.sample_sizes) * len(args.training_seeds)
    job_index = 0
    for sample_size in args.sample_sizes:
        for seed in args.training_seeds:
            job_index += 1
            print(
                f"\n[job {job_index}/{total_jobs}] n={sample_size} seed={seed} "
                f"device={device}",
                flush=True,
            )
            all_results.append(run_job(dgp, sample_size, seed, args, device))

    detail = pd.concat(all_results, ignore_index=True)
    significance = paired_significance(detail)
    paired = detail.pivot_table(
        index=["sample_size", "seed"],
        columns="method",
        values="expected_newsvendor_cost",
    ).reset_index()
    summary = (
        detail.groupby(["sample_size", "method", "method_label"], as_index=False)
        .agg(
            expected_cost_mean=("expected_newsvendor_cost", "mean"),
            expected_cost_std=("expected_newsvendor_cost", "std"),
            normalized_regret_mean=("normalized_regret", "mean"),
            quantile_mae_mean=("critical_quantile_mae", "mean"),
            coverage_error_mean=("coverage_error", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            epochs_ran_mean=("epochs_ran", "mean"),
        )
        .sort_values(["sample_size", "expected_cost_mean"])
    )
    detail.to_csv(args.output_dir / "detail.csv", index=False)
    paired.to_csv(args.output_dir / "paired_expected_cost.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    significance.to_csv(args.output_dir / "paired_significance.csv", index=False)
    with pd.ExcelWriter(args.output_dir / "results.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        significance.to_excel(writer, sheet_name="significance", index=False)
        paired.to_excel(writer, sheet_name="paired", index=False)
        detail.to_excel(writer, sheet_name="detail", index=False)
    plot_results(detail, significance, args.output_dir)
    configuration = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": str(device),
        "data_generation": dgp.generation_config(),
        "source_illustration_sample_sizes": [50, 200, 2000],
        "project_specific_adaptations": {
            "newsvendor_cost_under": args.cost_under,
            "newsvendor_cost_over": args.cost_over,
            "critical_ratio": args.cost_under / (args.cost_under + args.cost_over),
            "paired_training_seeds": args.training_seeds,
            "shared_spline_backbone": True,
            "shared_initial_checkpoint_within_pair": True,
            "optimizer": "projected_sgd",
            "early_stopping_metric": "unsmoothed_newsvendor_loss",
            "target_standardization": True,
            "all_context_dimensions_active_above_two": True,
        },
        "significance_protocol": {
            "primary_outcome": "oracle expected newsvendor cost",
            "paired_difference": "baseline minus RSETO-IPA",
            "test": "one-sided paired t-test",
            "familywise_adjustment": "Holm across all sample-size/baseline comparisons",
            "robustness_test": "one-sided paired Wilcoxon signed-rank",
        },
    }
    (args.output_dir / "config.json").write_text(json.dumps(configuration, indent=2))
    print("\n" + summary.to_string(index=False))
    print("\n" + significance.to_string(index=False))
    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
