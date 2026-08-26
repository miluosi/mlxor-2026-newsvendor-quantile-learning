import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model.newsvendor_gendfl_conditional_flow import GenDFLConditionalFlowNewsvendor
from model.newsvendor_quantile_flow import AffineQuantileFlowNewsvendor
from run_generative_newsvendor_toy_exp import makettoy_multi_exp


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_xy_loader(x, y, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.as_tensor(x), torch.as_tensor(y)),
        batch_size=min(int(batch_size), len(x)),
        shuffle=bool(shuffle),
        generator=generator,
    )


def make_regularized_loader(x, y, batch_size, shuffle, seed):
    combined = torch.as_tensor(np.hstack([x, y]).astype(np.float32))
    indices = torch.arange(len(combined), dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(combined, indices),
        batch_size=min(int(batch_size), len(combined)),
        shuffle=bool(shuffle),
        generator=generator,
    )


def model_arguments(feature_dim, train_len, args, alpha, seed):
    return {
        "targetdim": 1,
        "labeldim": int(feature_dim),
        "latent": 1,
        "data_len": int(train_len),
        "epoch": int(args.epochs),
        "quantiles": float(alpha),
        "target_quantile": float(alpha),
        "lambda1": float(args.ipa_lambda),
        "lambda_gradient": float(args.ipa_lambda),
        "samplingnumber": int(args.ipa_samples),
        "cost_under": float(args.cost_under),
        "cost_over": float(args.cost_over),
        "random_seed": int(seed),
        "innerloop": 1,
        "hidden_dim": 32,
    }


def evaluate_model(
    model,
    method,
    x_test_scaled,
    y_test_scaled,
    y_test_raw,
    y_scaler,
    alpha,
    cost_under,
    cost_over,
    inference_samples,
    inference_seed,
    batch_size=256,
):
    model.eval()
    device = next(model.parameters()).device
    decisions_scaled = []
    nll_values = []
    qfr_values = []
    tau_grid = torch.linspace(0.01, 0.99, 99, device=device)
    torch.manual_seed(int(inference_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(inference_seed))
    with torch.no_grad():
        for start in range(0, len(x_test_scaled), int(batch_size)):
            end = min(start + int(batch_size), len(x_test_scaled))
            condition = torch.as_tensor(
                x_test_scaled[start:end],
                dtype=torch.float32,
                device=device,
            )
            target = torch.as_tensor(
                y_test_scaled[start:end],
                dtype=torch.float32,
                device=device,
            )
            if method in {"gendfl_nll", "gendfl_ipa"}:
                generated_distribution = model.sample(
                    int(inference_samples),
                    condition,
                )
                decision = torch.quantile(
                    generated_distribution,
                    float(alpha),
                    dim=1,
                ).reshape(-1, 1)
            else:
                decision = model.quantile(alpha, condition)[:, 0, :]
            decisions_scaled.append(decision.cpu().numpy())
            nll_values.append(float(model.generative_loss(target, condition)))

            tau = tau_grid.reshape(1, -1, 1).expand(len(condition), -1, -1)
            quantiles = model.quantile(tau, condition)
            residual = target[:, None, :] - quantiles
            qfr_loss = torch.maximum(tau * residual, (tau - 1.0) * residual).mean()
            qfr_values.append(float(qfr_loss))

    decisions_scaled = np.vstack(decisions_scaled)
    decisions_raw = y_scaler.inverse_transform(decisions_scaled).reshape(-1)
    y_test_raw = np.asarray(y_test_raw).reshape(-1)
    difference = y_test_raw - decisions_raw
    underage = float(np.mean(cost_under * np.maximum(difference, 0.0)))
    overage = float(np.mean(cost_over * np.maximum(-difference, 0.0)))
    return {
        "method": method,
        "newsvendor_cost": underage + overage,
        "underage_cost": underage,
        "overage_cost": overage,
        "coverage": float(np.mean(y_test_raw <= decisions_raw)),
        "scaled_nll": float(np.mean(nll_values)),
        "scaled_integrated_pinball": float(np.mean(qfr_values)),
        "inference_mode": (
            "sample_distribution_then_quantile"
            if method in {"gendfl_nll", "gendfl_ipa"}
            else "direct_quantile_function"
        ),
        "inference_samples": (
            int(inference_samples)
            if method in {"gendfl_nll", "gendfl_ipa"}
            else 0
        ),
        "q_pred": decisions_raw,
    }


def serializable_history(history):
    result = {}
    for key, value in history.items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def run_seed(seed, data, args, output_dir, device):
    train_val, test = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.data_seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=args.val_size,
        random_state=args.data_seed,
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(train[:, :-1]).astype(np.float32)
    x_val = x_scaler.transform(val[:, :-1]).astype(np.float32)
    x_test = x_scaler.transform(test[:, :-1]).astype(np.float32)
    y_train = y_scaler.fit_transform(train[:, -1:]).astype(np.float32)
    y_val = y_scaler.transform(val[:, -1:]).astype(np.float32)
    y_test = y_scaler.transform(test[:, -1:]).astype(np.float32)

    alpha = args.cost_under / (args.cost_under + args.cost_over)
    common = model_arguments(x_train.shape[1], len(x_train), args, alpha, seed)
    set_seed(seed)
    initial_gendfl = GenDFLConditionalFlowNewsvendor(**common).to(device)
    initial_state = copy.deepcopy(initial_gendfl.state_dict())
    quantile_flow = AffineQuantileFlowNewsvendor(
        **common,
        sigma_min=args.sigma_min,
        tau_eps=args.tau_eps,
    ).to(device)
    quantile_flow.load_state_dict(copy.deepcopy(initial_state), strict=True)

    rows = []
    predictions = pd.DataFrame({"y_true": test[:, -1].astype(np.float32)})
    parameter_count = sum(parameter.numel() for parameter in initial_gendfl.parameters())
    if parameter_count != sum(parameter.numel() for parameter in quantile_flow.parameters()):
        raise RuntimeError("GenDFL and Quantile-Flow parameter counts differ.")

    nll_model = initial_gendfl
    start = time.perf_counter()
    nll_history = nll_model.train_conditional_flow(
        make_xy_loader(x_train, y_train, args.batch_size, True, seed + 1000),
        make_xy_loader(x_val, y_val, args.batch_size, False, seed + 1001),
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping,
    )
    nll_seconds = time.perf_counter() - start
    nll_result = evaluate_model(
        nll_model,
        "gendfl_nll",
        x_test,
        y_test,
        test[:, -1],
        y_scaler,
        alpha,
        args.cost_under,
        args.cost_over,
        args.inference_samples,
        args.inference_seed,
    )
    predictions["gendfl_nll"] = nll_result.pop("q_pred")
    rows.append(
        {
            "seed": seed,
            **nll_result,
            "epochs_ran": nll_history["epochs_ran"],
            "elapsed_seconds": nll_seconds,
            "parameter_count": parameter_count,
        }
    )

    ipa_model = copy.deepcopy(nll_model)
    set_seed(seed + 20000)
    start = time.perf_counter()
    ipa_history = ipa_model.train_regularized_ipa(
        make_regularized_loader(x_train, y_train, args.batch_size, True, seed + 2000),
        make_regularized_loader(x_val, y_val, args.batch_size, False, seed + 2001),
        num_epochs=args.epochs,
        early_stopping=args.early_stopping,
        regularization_lambda=args.ipa_lambda,
        learning_rate=args.learning_rate,
        k=args.ipa_replicates,
        num_samples=args.ipa_samples,
        use_vmap=True,
        vmap_chunk_size=args.vmap_chunk_size,
        max_grad_norm=None,
    )
    ipa_seconds = time.perf_counter() - start
    ipa_result = evaluate_model(
        ipa_model,
        "gendfl_ipa",
        x_test,
        y_test,
        test[:, -1],
        y_scaler,
        alpha,
        args.cost_under,
        args.cost_over,
        args.inference_samples,
        args.inference_seed,
    )
    predictions["gendfl_ipa"] = ipa_result.pop("q_pred")
    rows.append(
        {
            "seed": seed,
            **ipa_result,
            "epochs_ran": ipa_history["epochs_ran"],
            "elapsed_seconds": ipa_seconds,
            "parameter_count": parameter_count,
        }
    )

    set_seed(seed)
    start = time.perf_counter()
    qfr_history = quantile_flow.train_quantile_flow(
        make_xy_loader(x_train, y_train, args.batch_size, True, seed + 3000),
        make_xy_loader(x_val, y_val, args.batch_size, False, seed + 3001),
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping,
        num_quantile_levels=args.qfr_levels,
        validation_quantile_levels=args.validation_qfr_levels,
        max_grad_norm=None,
    )
    qfr_seconds = time.perf_counter() - start
    qfr_result = evaluate_model(
        quantile_flow,
        "quantile_flow",
        x_test,
        y_test,
        test[:, -1],
        y_scaler,
        alpha,
        args.cost_under,
        args.cost_over,
        args.inference_samples,
        args.inference_seed,
    )
    predictions["quantile_flow"] = qfr_result.pop("q_pred")
    rows.append(
        {
            "seed": seed,
            **qfr_result,
            "epochs_ran": qfr_history["epochs_ran"],
            "elapsed_seconds": qfr_seconds,
            "parameter_count": parameter_count,
        }
    )

    predictions.to_csv(output_dir / f"predictions_seed{seed}.csv", index=False)
    histories = {
        "gendfl_nll": serializable_history(nll_history),
        "gendfl_ipa": serializable_history(ipa_history),
        "quantile_flow": serializable_history(qfr_history),
    }
    (output_dir / f"histories_seed{seed}.json").write_text(
        json.dumps(histories, indent=2, allow_nan=True)
    )
    torch.save(nll_model.state_dict(), output_dir / f"gendfl_nll_seed{seed}.pth")
    torch.save(ipa_model.state_dict(), output_dir / f"gendfl_ipa_seed{seed}.pth")
    torch.save(quantile_flow.state_dict(), output_dir / f"quantile_flow_seed{seed}.pth")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/gendfl_quantile_flow_syn"),
    )
    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--num-exps", type=int, default=1)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--training-seeds", type=str, default="42,43,44")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cost-under", type=float, default=7.0)
    parser.add_argument("--cost-over", type=float, default=3.0)
    parser.add_argument("--ipa-lambda", type=float, default=0.5)
    parser.add_argument("--ipa-replicates", type=int, default=8)
    parser.add_argument("--ipa-samples", type=int, default=32)
    parser.add_argument("--vmap-chunk-size", type=int, default=4)
    parser.add_argument("--qfr-levels", type=int, default=32)
    parser.add_argument("--validation-qfr-levels", type=int, default=99)
    parser.add_argument("--inference-samples", type=int, default=4096)
    parser.add_argument("--inference-seed", type=int, default=20260807)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--tau-eps", type=float, default=1e-4)
    parser.add_argument("--large-difference-pct", type=float, default=10.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, _ = makettoy_multi_exp(
        num_samples=args.num_samples,
        num_features=args.dim,
        random_state=args.data_seed,
        num_exps=args.num_exps,
    )
    data = data[:, :-1].astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(value.strip()) for value in args.training_seeds.split(",") if value.strip()]
    rows = []
    for seed in seeds:
        print(f"[run] dataset_seed={args.data_seed} training_seed={seed} device={device}")
        rows.extend(run_seed(seed, data, args, args.output_dir, device))

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("method", as_index=False)
        .agg(
            newsvendor_cost_mean=("newsvendor_cost", "mean"),
            newsvendor_cost_std=("newsvendor_cost", "std"),
            underage_cost_mean=("underage_cost", "mean"),
            overage_cost_mean=("overage_cost", "mean"),
            coverage_mean=("coverage", "mean"),
            scaled_nll_mean=("scaled_nll", "mean"),
            scaled_integrated_pinball_mean=("scaled_integrated_pinball", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            epochs_ran_mean=("epochs_ran", "mean"),
            parameter_count=("parameter_count", "first"),
        )
        .sort_values("newsvendor_cost_mean")
    )
    baseline_cost = float(
        summary.loc[summary["method"] == "gendfl_nll", "newsvendor_cost_mean"].iloc[0]
    )
    summary["cost_delta_vs_gendfl_pct"] = (
        100.0 * (summary["newsvendor_cost_mean"] - baseline_cost) / baseline_cost
    )
    summary["within_large_difference_threshold"] = (
        summary["cost_delta_vs_gendfl_pct"].abs() <= args.large_difference_pct
    )
    consistency = {
        "large_difference_threshold_pct": args.large_difference_pct,
        "all_methods_within_threshold": bool(
            summary["within_large_difference_threshold"].all()
        ),
        "maximum_absolute_cost_delta_pct": float(
            summary["cost_delta_vs_gendfl_pct"].abs().max()
        ),
        "same_parameter_count": bool(summary["parameter_count"].nunique() == 1),
        "inference": {
            "gendfl_nll": "sample_distribution_then_quantile",
            "gendfl_ipa": "sample_distribution_then_quantile",
            "quantile_flow": "direct_quantile_function",
            "gendfl_samples_per_x": args.inference_samples,
            "inference_seed": args.inference_seed,
        },
        "dataset": {
            "num_samples": args.num_samples,
            "feature_dim": args.dim,
            "num_exps": args.num_exps,
            "data_seed": args.data_seed,
            "training_seeds": seeds,
        },
    }
    detail.to_csv(args.output_dir / "detail.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    with pd.ExcelWriter(args.output_dir / "results.xlsx") as writer:
        detail.to_excel(writer, sheet_name="detail", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    (args.output_dir / "consistency_check.json").write_text(
        json.dumps(consistency, indent=2)
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args) | {"output_dir": str(args.output_dir)}, indent=2)
    )
    print("\n", summary.to_string(index=False))
    print("\n", json.dumps(consistency, indent=2))
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
