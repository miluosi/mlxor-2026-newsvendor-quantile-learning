"""Search for fixed-mixture settings where RSETO-IPA helps with scarce data."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model.gendfl_spline import GenDFLSplineNewsvendor
from model.rseto_ipa_spline import RSETOIPASplineNewsvendor
from model.spline_qfr import SplineQFRNewsvendor
from spline_sensitivity_common import make_cost_protocol
from synthetic_fixed_dgp import make_toy_mixture_parameters, makettoy_multi_exp


METHODS = {
    "gendfl_spline": GenDFLSplineNewsvendor,
    "spline_qfr": SplineQFRNewsvendor,
    "rseto_ipa_spline": RSETOIPASplineNewsvendor,
}


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def resolve_device(requested):
    requested = str(requested).lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_int_list(value):
    return [int(item) for item in str(value).split(",") if item.strip()]


def parse_settings(value):
    settings = []
    for item in str(value).split(","):
        dim, fold = item.split(":")
        settings.append((int(dim), int(fold)))
    return settings


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


def sample_fixed_dgp(parameters, dim, seed, size):
    labelled, _ = makettoy_multi_exp(
        size,
        dim,
        seed,
        num_exps=len(parameters.probabilities),
        sample_random_state=seed,
        parameters=parameters,
    )
    return labelled[:, :dim], labelled[:, dim], labelled[:, dim + 1].astype(int)


def mixture_quantile(parameters, context, alpha, iterations=70):
    means = context @ parameters.weights.T + parameters.intercepts
    lower = means.min(axis=1) - 10.0 * parameters.noise_scale
    upper = means.max(axis=1) + 10.0 * parameters.noise_scale
    for _ in range(int(iterations)):
        midpoint = 0.5 * (lower + upper)
        cdf = (
            ndtr((midpoint[:, None] - means) / parameters.noise_scale)
            * parameters.probabilities
        ).sum(axis=1)
        move_lower = cdf < alpha
        lower = np.where(move_lower, midpoint, lower)
        upper = np.where(move_lower, upper, midpoint)
    return 0.5 * (lower + upper)


def mixture_expected_cost(parameters, context, decision, cost_under, cost_over):
    decision = np.asarray(decision).reshape(-1, 1)
    means = context @ parameters.weights.T + parameters.intercepts
    sigma = float(parameters.noise_scale)
    standardized = (decision - means) / sigma
    cdf = ndtr(standardized)
    density = np.exp(-0.5 * standardized**2) / math.sqrt(2.0 * math.pi)
    shortage = sigma * density + (means - decision) * (1.0 - cdf)
    overage = sigma * density + (decision - means) * cdf
    component_cost = float(cost_under) * shortage + float(cost_over) * overage
    return component_cost @ parameters.probabilities


def predict_quantile(model, context_scaled, alpha, target_scaler, batch_size):
    loader = DataLoader(
        TensorDataset(torch.as_tensor(context_scaled, dtype=torch.float32)),
        batch_size=min(int(batch_size), len(context_scaled)),
        shuffle=False,
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for (context,) in loader:
            scaled = model.quantile(float(alpha), context.to(model._device()))[:, 0, :]
            predictions.append(scaled.cpu().numpy())
    scaled = np.concatenate(predictions, axis=0)
    return target_scaler.inverse_transform(scaled)[:, 0]


def empirical_cost(demand, decision, cost_under, cost_over):
    return float(
        np.mean(
            float(cost_under) * np.maximum(demand - decision, 0.0)
            + float(cost_over) * np.maximum(decision - demand, 0.0)
        )
    )


def model_kwargs(dim, train_size, epochs, alpha, cost_under, cost_over, seed, args):
    return {
        "targetdim": 1,
        "labeldim": int(dim),
        "latent": 1,
        "data_len": int(train_size),
        "epoch": int(epochs),
        "quantiles": float(alpha),
        "target_quantile": float(alpha),
        "cost_under": float(cost_under),
        "cost_over": float(cost_over),
        "random_seed": int(seed),
        "num_transforms": int(args.num_transforms),
        "num_bins": int(args.num_bins),
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "tail_bound": float(args.tail_bound),
        "tau_eps": float(args.tau_eps),
    }


def train_model(method, model, train_loader, validation_loader, seed, args):
    common = {
        "num_epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "step_size_exponent": float(args.step_size_exponent),
        "training_seed": int(seed),
        "parameter_box_lower": float(args.parameter_box_lower),
        "parameter_box_upper": float(args.parameter_box_upper),
        "stop_early": False,
        "restore_best": False,
        "early_stopping": max(1, int(args.epochs)),
        "warmup_epochs": 0,
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
            num_tau=int(args.qflow_levels),
            validation_num_tau=int(args.qflow_validation_levels),
            **common,
        )
    return model.train_rseto_ipa_spline(
        train_loader,
        validation_loader,
        replications=int(args.replications),
        samples_per_replication=int(args.samples_per_replication),
        m_growth=float(args.m_growth),
        m_growth_exponent=float(args.m_growth_exponent),
        smoothing_mu=float(args.smoothing_mu),
        fidelity_weight=float(args.fidelity_weight),
        max_simulation_values=int(args.max_simulation_values),
        train_data_on_device=bool(args.train_data_on_device),
        **common,
    )


def run_setting(dim, fold, train_size, args, device):
    random_states, single_costs, series_costs = make_cost_protocol()
    parameter_seed = random_states[fold]
    cost_under, cost_over_signed = single_costs[fold]
    cost_over = abs(cost_over_signed)
    alpha = cost_under / (cost_under + cost_over)
    parameters = make_toy_mixture_parameters(dim, parameter_seed, num_exps=5)
    context_train, demand_train, _ = sample_fixed_dgp(
        parameters,
        dim,
        parameter_seed + 100_000 + train_size,
        train_size,
    )
    validation_size = max(200, int(math.ceil(0.25 * train_size)))
    context_validation, demand_validation, _ = sample_fixed_dgp(
        parameters,
        dim,
        parameter_seed + 200_000,
        validation_size,
    )
    context_test, demand_test, labels_test = sample_fixed_dgp(
        parameters,
        dim,
        parameter_seed + 300_000,
        int(args.test_size),
    )

    context_scaler = StandardScaler().fit(context_train)
    target_scaler = StandardScaler().fit(demand_train.reshape(-1, 1))
    train_x = context_scaler.transform(context_train).astype(np.float32)
    validation_x = context_scaler.transform(context_validation).astype(np.float32)
    test_x = context_scaler.transform(context_test).astype(np.float32)
    train_y = target_scaler.transform(demand_train.reshape(-1, 1)).astype(np.float32)
    validation_y = target_scaler.transform(demand_validation.reshape(-1, 1)).astype(
        np.float32
    )

    initialization_seed = parameter_seed + 400_000 + train_size
    kwargs = model_kwargs(
        dim,
        train_size,
        args.epochs,
        alpha,
        cost_under,
        cost_over,
        initialization_seed,
        args,
    )
    set_seed(initialization_seed)
    template = GenDFLSplineNewsvendor(**kwargs)
    initial_state = copy.deepcopy(template.state_dict())
    rows = []
    predictions = {}
    for method, model_class in METHODS.items():
        set_seed(initialization_seed)
        model = model_class(**kwargs)
        model.load_state_dict(copy.deepcopy(initial_state), strict=True)
        model.to(device)
        train_loader = make_loader(
            train_x,
            train_y,
            args.batch_size,
            True,
            initialization_seed + 1,
        )
        validation_loader = make_loader(
            validation_x,
            validation_y,
            args.batch_size,
            False,
            initialization_seed + 2,
        )
        start = time.perf_counter()
        history = train_model(
            method,
            model,
            train_loader,
            validation_loader,
            initialization_seed + 3,
            args,
        )
        elapsed = time.perf_counter() - start
        metric1_prediction = predict_quantile(
            model,
            test_x,
            alpha,
            target_scaler,
            args.evaluation_batch_size,
        )
        predictions[method] = metric1_prediction
        metric1 = empirical_cost(
            demand_test,
            metric1_prediction,
            cost_under,
            cost_over,
        )
        expected_metric1 = float(
            mixture_expected_cost(
                parameters,
                context_test,
                metric1_prediction,
                cost_under,
                cost_over,
            ).mean()
        )
        metric2_costs = []
        expected_metric2_costs = []
        for metric_cost_under, metric_cost_over_signed in series_costs[fold]:
            metric_cost_over = abs(metric_cost_over_signed)
            metric_alpha = metric_cost_under / (metric_cost_under + metric_cost_over)
            decision = predict_quantile(
                model,
                test_x,
                metric_alpha,
                target_scaler,
                args.evaluation_batch_size,
            )
            metric2_costs.append(
                empirical_cost(
                    demand_test,
                    decision,
                    metric_cost_under,
                    metric_cost_over,
                )
            )
            expected_metric2_costs.append(
                float(
                    mixture_expected_cost(
                        parameters,
                        context_test,
                        decision,
                        metric_cost_under,
                        metric_cost_over,
                    ).mean()
                )
            )
        rows.append(
            {
                "dim": int(dim),
                "fold": int(fold),
                "parameter_seed": int(parameter_seed),
                "train_size": int(train_size),
                "validation_size": int(validation_size),
                "test_size": int(args.test_size),
                "alpha": float(alpha),
                "method": method,
                "metric1": metric1,
                "expected_metric1": expected_metric1,
                "metric2": float(np.mean(metric2_costs)),
                "expected_metric2": float(np.mean(expected_metric2_costs)),
                "epochs_ran": int(history["epochs_ran"]),
                "steps_ran": int(history["steps_ran"]),
                "elapsed_seconds": elapsed,
                "training_objective": history.get("training_objective", "nll_plus_ipa"),
                "demand_excess_kurtosis": float(
                    kurtosis(demand_test, fisher=True, bias=False)
                ),
                "effective_components": float(
                    np.exp(
                        -np.sum(
                            parameters.probabilities
                            * np.log(parameters.probabilities + 1e-15)
                        )
                    )
                ),
            }
        )

    oracle_decision = mixture_quantile(parameters, context_test, alpha)
    oracle_expected_metric1 = float(
        mixture_expected_cost(
            parameters,
            context_test,
            oracle_decision,
            cost_under,
            cost_over,
        ).mean()
    )
    for row in rows:
        row["oracle_expected_metric1"] = oracle_expected_metric1
        row["expected_metric1_regret"] = (
            row["expected_metric1"] - oracle_expected_metric1
        )
    return rows


def add_improvements(results):
    output = results.copy()
    for (_, _, train_size), group in output.groupby(["dim", "fold", "train_size"]):
        baseline = group[group["method"] == "gendfl_spline"].iloc[0]
        indices = group.index
        for metric in ["metric1", "expected_metric1", "metric2", "expected_metric2"]:
            output.loc[indices, f"{metric}_improvement_over_gendfl_percent"] = (
                100.0 * (float(baseline[metric]) - output.loc[indices, metric])
                / float(baseline[metric])
            )
    return output


def plot_results(results, output_path):
    settings = results[["dim", "fold", "parameter_seed"]].drop_duplicates()
    figure, axes = plt.subplots(
        len(settings),
        2,
        figsize=(11, 4.0 * len(settings)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {
        "gendfl_spline": "#222222",
        "spline_qfr": "#277da1",
        "rseto_ipa_spline": "#d1495b",
    }
    for row_index, setting in enumerate(settings.itertuples(index=False)):
        subset = results[(results.dim == setting.dim) & (results.fold == setting.fold)]
        for column_index, metric in enumerate(["expected_metric1", "expected_metric2"]):
            axis = axes[row_index, column_index]
            for method, group in subset.groupby("method"):
                group = group.sort_values("train_size")
                axis.plot(
                    group["train_size"],
                    group[metric],
                    marker="o",
                    linewidth=1.8,
                    color=colors[method],
                    label=method,
                )
            axis.set_xscale("log")
            axis.set_xlabel("Training observations")
            axis.set_ylabel("Oracle expected newsvendor cost")
            axis.set_title(
                f"dim={setting.dim}, seed={setting.parameter_seed}: {metric}"
            )
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/spline_low_data_shape_search"),
    )
    parser.add_argument(
        "--settings",
        type=parse_settings,
        default=parse_settings("4:1,19:2"),
        help="Comma-separated dim:fold pairs.",
    )
    parser.add_argument(
        "--train-sizes",
        type=parse_int_list,
        default=parse_int_list("200,500,1000,2000"),
    )
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--step-size-exponent", type=float, default=0.6)
    parser.add_argument("--parameter-box-lower", type=float, default=-10.0)
    parser.add_argument("--parameter-box-upper", type=float, default=10.0)
    parser.add_argument("--num-transforms", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--tail-bound", type=float, default=4.0)
    parser.add_argument("--tau-eps", type=float, default=1e-5)
    parser.add_argument("--qflow-levels", type=int, default=16)
    parser.add_argument("--qflow-validation-levels", type=int, default=99)
    parser.add_argument("--replications", type=int, default=16)
    parser.add_argument("--samples-per-replication", type=int, default=128)
    parser.add_argument("--m-growth", type=float, default=1.0)
    parser.add_argument("--m-growth-exponent", type=float, default=0.25)
    parser.add_argument("--fidelity-weight", type=float, default=0.5)
    parser.add_argument("--smoothing-mu", type=float, default=0.05)
    parser.add_argument("--max-simulation-values", type=int, default=1_048_576)
    parser.add_argument(
        "--train-data-on-device",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "low_data_shape_results.csv"
    existing = pd.read_csv(result_path) if result_path.exists() and not args.force else pd.DataFrame()
    device = resolve_device(args.device)
    rows = existing.to_dict("records")
    completed = (
        set(zip(existing.dim, existing.fold, existing.train_size, existing.method))
        if not existing.empty
        else set()
    )
    jobs = [(dim, fold, size) for dim, fold in args.settings for size in args.train_sizes]
    for job_index, (dim, fold, train_size) in enumerate(jobs, start=1):
        required = {(dim, fold, train_size, method) for method in METHODS}
        if required.issubset(completed):
            continue
        print(
            f"[job {job_index}/{len(jobs)}] dim={dim} fold={fold} "
            f"train_size={train_size} device={device}",
            flush=True,
        )
        job_rows = run_setting(dim, fold, train_size, args, device)
        rows.extend(job_rows)
        results = add_improvements(pd.DataFrame(rows))
        results.to_csv(result_path, index=False)
        print(
            results[
                (results.dim == dim)
                & (results.fold == fold)
                & (results.train_size == train_size)
            ][
                [
                    "method",
                    "expected_metric1",
                    "expected_metric1_improvement_over_gendfl_percent",
                    "expected_metric2",
                    "elapsed_seconds",
                ]
            ].to_string(index=False),
            flush=True,
        )
    results = add_improvements(pd.DataFrame(rows))
    results.to_csv(result_path, index=False)
    plot_results(results, args.output_dir / "low_data_expected_cost.png")
    configuration = vars(args).copy()
    configuration["output_dir"] = str(args.output_dir)
    configuration["settings"] = [list(setting) for setting in args.settings]
    configuration["device"] = str(device)
    (args.output_dir / "config.json").write_text(json.dumps(configuration, indent=2))
    print(f"Saved to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
