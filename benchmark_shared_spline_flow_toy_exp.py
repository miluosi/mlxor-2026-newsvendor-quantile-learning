"""Compare shared-spline Gen-DFL, QFR, and RSETO-IPA on toy exp 1/5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from benchmark_shared_spline_flow_syn import (
    METHOD_LABELS,
    build_models,
    model_arguments,
    set_seed,
    train_models,
)
from model.spline_qfr import pinball_loss
from run_generative_newsvendor_toy_exp import makettoy_multi_exp


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def inverse_target(values, scaler):
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).reshape(-1)


def evaluate_models(
    models,
    context_scaled,
    demand_scaled,
    demand_raw,
    target_scaler,
    alpha,
    seed,
    elapsed,
    parameter_counts,
    histories,
    args,
):
    rows = []
    predictions = pd.DataFrame({"y_true": np.asarray(demand_raw).reshape(-1)})
    tau_grid = torch.linspace(
        args.evaluation_tau_eps,
        1.0 - args.evaluation_tau_eps,
        args.evaluation_tau_levels,
    )
    for name, model in models.items():
        model.eval()
        device = next(model.parameters()).device
        method_decisions = []
        exact_decisions = []
        nll_sum = 0.0
        pinball_sum = 0.0
        observation_count = 0
        with torch.no_grad():
            for start in range(0, len(context_scaled), args.evaluation_batch_size):
                end = min(start + args.evaluation_batch_size, len(context_scaled))
                condition = torch.as_tensor(context_scaled[start:end], device=device)
                target = torch.as_tensor(demand_scaled[start:end], device=device)
                exact_decision = model.critical_quantile_decision(condition)
                method_decision = exact_decision

                batch_tau = tau_grid.to(device=device, dtype=condition.dtype)
                quantiles = model.quantile(batch_tau, condition)
                expanded_tau = batch_tau.reshape(1, -1, 1).expand_as(quantiles)
                expanded_target = target[:, None, :].expand_as(quantiles)
                batch_size = end - start
                nll_sum += float(model.generative_loss(target, condition)) * batch_size
                pinball_sum += float(
                    pinball_loss(expanded_target, quantiles, expanded_tau).mean()
                ) * batch_size
                observation_count += batch_size
                method_decisions.append(method_decision.cpu().numpy())
                exact_decisions.append(exact_decision.cpu().numpy())

        decision_raw = inverse_target(np.vstack(method_decisions), target_scaler)
        exact_decision_raw = inverse_target(np.vstack(exact_decisions), target_scaler)
        predictions[name] = decision_raw
        predictions[f"{name}_exact_map"] = exact_decision_raw
        demand_vector = np.asarray(demand_raw).reshape(-1)
        difference = demand_vector - decision_raw
        exact_difference = demand_vector - exact_decision_raw
        underage = float(np.mean(args.cost_under * np.maximum(difference, 0.0)))
        overage = float(np.mean(args.cost_over * np.maximum(-difference, 0.0)))
        exact_map_cost = float(
            np.mean(
                args.cost_under * np.maximum(exact_difference, 0.0)
                + args.cost_over * np.maximum(-exact_difference, 0.0)
            )
        )
        coverage = float(np.mean(demand_vector <= decision_raw))
        rows.append(
            {
                "seed": seed,
                "method": name,
                "method_label": METHOD_LABELS[name],
                "newsvendor_cost": underage + overage,
                "underage_cost": underage,
                "overage_cost": overage,
                "coverage": coverage,
                "coverage_error": abs(coverage - alpha),
                "exact_map_newsvendor_cost": exact_map_cost,
                "scaled_nll": nll_sum / observation_count,
                "scaled_integrated_pinball": pinball_sum / observation_count,
                "scaled_crps": 2.0 * pinball_sum / observation_count,
                "elapsed_seconds": elapsed[name],
                "epochs_ran": histories[name]["epochs_ran"],
                "best_epoch": histories[name]["best_epoch"],
                "parameter_count": parameter_counts[name],
                "inference_mode": "direct_exact_spline_quantile",
                "inference_samples": 0,
            }
        )
    return rows, predictions


def run_experiment(num_exps, seed, args, device):
    raw_data, _ = makettoy_multi_exp(
        num_samples=args.num_samples,
        num_features=args.dim,
        random_state=args.data_seed,
        num_exps=num_exps,
    )
    data = raw_data[:, :-1].astype(np.float32)
    peak_labels = raw_data[:, -1].astype(np.int64)
    indices = np.arange(len(data))
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
    context_scaler = StandardScaler().fit(data[train_idx, :-1])
    target_scaler = StandardScaler().fit(data[train_idx, -1:])
    context_scaled = context_scaler.transform(data[:, :-1]).astype(np.float32)
    demand_scaled = target_scaler.transform(data[:, -1:]).astype(np.float32)
    alpha = args.cost_under / (args.cost_under + args.cost_over)
    common = model_arguments(args.dim, len(train_idx), alpha, seed, args)
    models, parameter_counts, _ = build_models(common, device, seed)

    exp_dir = args.output_dir / f"exp{num_exps}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    histories, elapsed = train_models(
        models,
        (context_scaled[train_idx], demand_scaled[train_idx]),
        (context_scaled[val_idx], demand_scaled[val_idx]),
        seed,
        args,
        exp_dir,
    )
    rows, predictions = evaluate_models(
        models,
        context_scaled[test_idx],
        demand_scaled[test_idx],
        data[test_idx, -1],
        target_scaler,
        alpha,
        seed,
        elapsed,
        parameter_counts,
        histories,
        args,
    )
    for row in rows:
        row["num_exps"] = num_exps
        row["feature_dim"] = args.dim
        row["train_size"] = len(train_idx)
        row["validation_size"] = len(val_idx)
        row["test_size"] = len(test_idx)
    predictions.insert(1, "peak_label", peak_labels[test_idx])
    predictions.to_csv(exp_dir / f"predictions_seed{seed}.csv", index=False)
    with (exp_dir / f"histories_seed{seed}.json").open("w") as handle:
        json.dump(histories, handle, indent=2, allow_nan=True)
    consistency = {
        "num_exps": num_exps,
        "seed": seed,
        "same_parameter_count": len(set(parameter_counts.values())) == 1,
        "parameter_counts": parameter_counts,
        "split_sizes": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "peak_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(peak_labels, return_counts=True))
        },
    }
    with (exp_dir / f"consistency_seed{seed}.json").open("w") as handle:
        json.dump(consistency, handle, indent=2)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/shared_spline_flow_toy_exp1_exp5"),
    )
    parser.add_argument("--experiments", type=parse_int_list, default=parse_int_list("1,5"))
    parser.add_argument("--training-seeds", type=parse_int_list, default=parse_int_list("42"))
    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=42)
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
    parser.add_argument("--gendfl-scenarios", type=int, default=128)
    parser.add_argument("--gendfl-decision-weight", type=float, default=1.0)
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
    args = parser.parse_args()

    if not args.experiments or not args.training_seeds:
        raise ValueError("experiments and training_seeds cannot be empty.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for num_exps in args.experiments:
        for seed in args.training_seeds:
            print(f"[run] exp={num_exps} seed={seed} device={device}")
            set_seed(seed)
            rows.extend(run_experiment(num_exps, seed, args, device))

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["num_exps", "method", "method_label"], as_index=False)
        .agg(
            newsvendor_cost_mean=("newsvendor_cost", "mean"),
            underage_cost_mean=("underage_cost", "mean"),
            overage_cost_mean=("overage_cost", "mean"),
            coverage_mean=("coverage", "mean"),
            coverage_error_mean=("coverage_error", "mean"),
            exact_map_newsvendor_cost_mean=("exact_map_newsvendor_cost", "mean"),
            scaled_nll_mean=("scaled_nll", "mean"),
            scaled_integrated_pinball_mean=("scaled_integrated_pinball", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            epochs_ran_mean=("epochs_ran", "mean"),
            parameter_count=("parameter_count", "first"),
        )
        .sort_values(["num_exps", "newsvendor_cost_mean"])
    )
    detail.to_csv(args.output_dir / "detail.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    with pd.ExcelWriter(args.output_dir / "results.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        detail.to_excel(writer, sheet_name="detail", index=False)
    configuration = vars(args).copy()
    configuration["output_dir"] = str(args.output_dir)
    configuration["device"] = str(device)
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(configuration, handle, indent=2)
    print("\n" + summary.to_string(index=False))
    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
