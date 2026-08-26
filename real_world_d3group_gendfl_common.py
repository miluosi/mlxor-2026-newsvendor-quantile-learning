"""Shared d3group real-world protocol for the spline GenDFL experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from real_world_d3group_test import (
    COST_PAIRS,
    DATASET_FEATURES,
    FEATURE_ALIASES,
    GroupData,
    build_feature_frame,
    load_dataset,
    parse_max_groups,
    selected_datasets,
    split_group_data,
)


METRIC2_COST_PAIRS = COST_PAIRS


@dataclass(frozen=True)
class SplineModelSpec:
    key: str
    display_name: str
    model_class: type
    trainer: Callable
    settings: Callable[[argparse.Namespace], dict]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def service_level(cost_under: float, cost_over: float) -> float:
    return float(cost_under) / (float(cost_under) + float(cost_over))


def newsvendor_cost(
    demand: np.ndarray,
    prediction: np.ndarray,
    cost_under: float,
    cost_over: float,
) -> float:
    demand = np.asarray(demand, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if demand.shape != prediction.shape:
        raise ValueError(
            f"Demand and prediction shapes differ: {demand.shape} != {prediction.shape}."
        )
    point_loss = np.where(
        demand > prediction,
        float(cost_under) * (demand - prediction),
        float(cost_over) * (prediction - demand),
    )
    return float(np.mean(point_loss))


def metric2_cost(
    demand: np.ndarray,
    predictions: np.ndarray,
    cost_pairs=METRIC2_COST_PAIRS,
) -> tuple[float, np.ndarray]:
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[0] != len(cost_pairs):
        raise ValueError(
            "Metric 2 predictions must have shape [number of cost pairs, test size]."
        )
    by_cost = np.asarray(
        [
            newsvendor_cost(demand, predictions[index], cost_under, cost_over)
            for index, (cost_under, cost_over) in enumerate(cost_pairs)
        ],
        dtype=float,
    )
    return float(by_cost.mean()), by_cost


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def state_dict_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_or_create_shared_initialization(
    model,
    args: argparse.Namespace,
    dataset: str,
    feature_combo: list[str],
    group,
    group_index: int,
    cost_under: float,
    cost_over: float,
    seed: int,
) -> tuple[Path, str]:
    architecture = {
        "dataset": dataset,
        "feature_combo": feature_combo,
        "group": str(group),
        "group_index": int(group_index),
        "cost_under": float(cost_under),
        "cost_over": float(cost_over),
        "seed": int(seed),
        "labeldim": int(model.labeldim),
        "num_transforms": int(args.num_transforms),
        "num_bins": int(args.num_bins),
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "tail_bound": float(args.tail_bound),
    }
    architecture_key = hashlib.sha256(
        json.dumps(architecture, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    group_name = safe_slug(group)
    initialization_dir = (
        Path(args.shared_initialization_dir) / dataset / args.feature_combi
    )
    initialization_path = initialization_dir / (
        f"group{group_index:04d}_{group_name}_{architecture_key}.pth"
    )
    metadata_path = initialization_path.with_suffix(".json")
    generated_hash = state_dict_sha256(model)
    if initialization_path.exists():
        state = torch.load(initialization_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        loaded_hash = state_dict_sha256(model)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata != {**architecture, "initial_state_sha256": loaded_hash}:
                raise RuntimeError(
                    f"Shared initialization metadata mismatch: {initialization_path}"
                )
        if loaded_hash != generated_hash:
            raise RuntimeError(
                "Deterministic initialization differs from the shared checkpoint; "
                "check model architecture and random-seed handling."
            )
    else:
        initialization_dir.mkdir(parents=True, exist_ok=True)
        cpu_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        torch.save(cpu_state, initialization_path)
        loaded_hash = generated_hash
        metadata_path.write_text(
            json.dumps(
                {**architecture, "initial_state_sha256": loaded_hash},
                indent=2,
            ),
            encoding="utf-8",
        )
    return initialization_path, loaded_hash


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def safe_slug(value) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return slug or "group"


def make_loaders(
    data: GroupData,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(
        torch.as_tensor(data.X_fit_model, dtype=torch.float32),
        torch.as_tensor(data.y_fit_model, dtype=torch.float32),
    )
    validation_dataset = TensorDataset(
        torch.as_tensor(data.X_val_model, dtype=torch.float32),
        torch.as_tensor(data.y_val_model, dtype=torch.float32),
    )
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training and validation partitions must both be non-empty.")
    generator = torch.Generator().manual_seed(int(seed))
    return (
        DataLoader(
            train_dataset,
            batch_size=min(int(batch_size), len(train_dataset)),
            shuffle=True,
            generator=generator,
        ),
        DataLoader(
            validation_dataset,
            batch_size=min(int(batch_size), len(validation_dataset)),
            shuffle=False,
        ),
    )


def model_kwargs(
    data: GroupData,
    args: argparse.Namespace,
    target_quantile: float,
    cost_under: float,
    cost_over: float,
    seed: int,
) -> dict:
    return {
        "targetdim": 1,
        "labeldim": data.n_features,
        "latent": 1,
        "data_len": len(data.y_fit_model),
        "epoch": int(args.epochs),
        "quantiles": float(target_quantile),
        "target_quantile": float(target_quantile),
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


def predict_exact_quantiles(
    model,
    context: np.ndarray,
    quantile_levels: list[float],
    scaler_y,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    device = model._device()
    context_tensor = torch.as_tensor(context, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(context_tensor),
        batch_size=min(int(batch_size), len(context_tensor)),
        shuffle=False,
    )
    chunks = []
    with torch.no_grad():
        for (context_batch,) in loader:
            context_batch = context_batch.to(device)
            scaled = model.quantile(quantile_levels, context_batch)
            chunks.append(scaled.squeeze(-1).detach().cpu().numpy())
    scaled_prediction = np.concatenate(chunks, axis=0)
    raw_prediction = scaler_y.inverse_transform(
        scaled_prediction.reshape(-1, 1)
    ).reshape(scaled_prediction.shape)
    return raw_prediction.T


def common_training_signature(
    spec: SplineModelSpec,
    data: GroupData,
    args: argparse.Namespace,
    dataset: str,
    feature_combo: list[str],
    group,
    cost_under: float,
    cost_over: float,
    seed: int,
    initialization_path: Path,
    initial_state_sha256: str,
) -> dict:
    return {
        "model": spec.key,
        "dataset": dataset,
        "feature_combo": feature_combo,
        "group": str(group),
        "cost_under": float(cost_under),
        "cost_over": float(cost_over),
        "target_quantile": service_level(cost_under, cost_over),
        "seed": int(seed),
        "initialization_path": str(initialization_path),
        "initial_state_sha256": initial_state_sha256,
        "train_validation_sha256": array_sha256(
            data.X_fit_model,
            data.y_fit_model,
            data.X_val_model,
            data.y_val_model,
        ),
        "epochs": int(args.epochs),
        "use_early_stopping": bool(args.use_early_stopping),
        "early_stopping": int(args.early_stopping),
        "warmup_epochs": int(args.warmup_epochs),
        "min_delta_relative": float(args.min_delta_relative),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "optimizer": "projected_sgd",
        "step_size_exponent": float(args.step_size_exponent),
        "parameter_box": [
            float(args.parameter_box_lower),
            float(args.parameter_box_upper),
        ],
        "architecture": {
            "num_transforms": int(args.num_transforms),
            "num_bins": int(args.num_bins),
            "hidden_dim": int(args.hidden_dim),
            "hidden_layers": int(args.hidden_layers),
            "tail_bound": float(args.tail_bound),
            "tau_eps": float(args.tau_eps),
        },
        "model_settings": spec.settings(args),
    }


def train_or_load(
    spec: SplineModelSpec,
    model,
    data: GroupData,
    args: argparse.Namespace,
    checkpoint_path: Path,
    history_path: Path,
    metadata_path: Path,
    signature: dict,
    seed: int,
) -> tuple[dict, float, bool]:
    can_reuse = (
        not args.force
        and checkpoint_path.exists()
        and history_path.exists()
        and metadata_path.exists()
        and json.loads(metadata_path.read_text()) == signature
    )
    if can_reuse:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(resolve_device(args.device))
        return json.loads(history_path.read_text()), 0.0, True

    train_loader, validation_loader = make_loaders(data, args.batch_size, seed)
    start = time.perf_counter()
    history = spec.trainer(
        model,
        train_loader,
        validation_loader,
        args,
        checkpoint_path,
    )
    if model._device().type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    history_path.write_text(
        json.dumps(json_ready(history), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    metadata_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return history, elapsed, False


def evaluate_one_model(
    spec: SplineModelSpec,
    data: GroupData,
    args: argparse.Namespace,
    dataset: str,
    feature_combo: list[str],
    group,
    group_index: int,
    cost_index: int,
    cost_under: float,
    cost_over: float,
    output_dir: Path,
) -> dict:
    target_quantile = service_level(cost_under, cost_over)
    model_seed = int(args.seed) + group_index * 100 + cost_index
    set_seed(model_seed)
    model = spec.model_class(
        **model_kwargs(
            data,
            args,
            target_quantile,
            cost_under,
            cost_over,
            model_seed,
        )
    )
    initialization_path, initial_state_sha256 = load_or_create_shared_initialization(
        model,
        args,
        dataset,
        feature_combo,
        group,
        group_index,
        cost_under,
        cost_over,
        model_seed,
    )
    model.to(resolve_device(args.device))

    group_name = safe_slug(group)
    cost_name = f"cu{cost_under:g}_co{cost_over:g}"
    artifact_stem = f"group{group_index:04d}_{group_name}_{spec.key}_{cost_name}"
    checkpoint_path = output_dir / "checkpoints" / f"{artifact_stem}.pth"
    history_path = output_dir / "histories" / f"{artifact_stem}.json"
    metadata_path = output_dir / "checkpoints" / f"{artifact_stem}_metadata.json"
    signature = common_training_signature(
        spec,
        data,
        args,
        dataset,
        feature_combo,
        group,
        cost_under,
        cost_over,
        model_seed,
        initialization_path,
        initial_state_sha256,
    )
    history, elapsed, reused = train_or_load(
        spec,
        model,
        data,
        args,
        checkpoint_path,
        history_path,
        metadata_path,
        signature,
        model_seed,
    )

    metric1_prediction = predict_exact_quantiles(
        model,
        data.X_test_model,
        [target_quantile],
        data.scaler_y,
        args.evaluation_batch_size,
    )[0]
    metric2_levels = [
        service_level(metric_cu, metric_co)
        for metric_cu, metric_co in METRIC2_COST_PAIRS
    ]
    metric2_predictions = predict_exact_quantiles(
        model,
        data.X_test_model,
        metric2_levels,
        data.scaler_y,
        args.evaluation_batch_size,
    )
    metric1 = newsvendor_cost(
        data.y_test_raw,
        metric1_prediction,
        cost_under,
        cost_over,
    )
    metric2, metric2_by_cost = metric2_cost(
        data.y_test_raw,
        metric2_predictions,
    )
    prediction_path = output_dir / "predictions" / f"{artifact_stem}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        y_true=data.y_test_raw,
        metric1_prediction=metric1_prediction,
        metric2_predictions=metric2_predictions,
        metric2_by_cost=metric2_by_cost,
        metric2_service_levels=np.asarray(metric2_levels),
    )
    return {
        "dataset": dataset,
        "feature combi": str(feature_combo),
        "group": str(group),
        "model": spec.display_name,
        "model key": spec.key,
        "cu": float(cost_under),
        "co": float(cost_over),
        "sl": target_quantile,
        "metric 1": metric1,
        "metric 2": metric2,
        "epochs ran": history.get("epochs_ran"),
        "best epoch": history.get("best_epoch"),
        "best val newsvendor": history.get("best_val_newsvendor"),
        "elapsed seconds": elapsed,
        "training reused": reused,
        "initialization seed": model_seed,
        "initialization path": str(initialization_path),
        "initial state sha256": initial_state_sha256,
        "prediction path": str(prediction_path),
        "checkpoint path": str(checkpoint_path),
        "error": None,
    }


def failed_result_row(
    spec: SplineModelSpec,
    dataset: str,
    feature_combo: list[str],
    group,
    cost_under: float,
    cost_over: float,
    error: Exception,
) -> dict:
    return {
        "dataset": dataset,
        "feature combi": str(feature_combo),
        "group": str(group),
        "model": spec.display_name,
        "model key": spec.key,
        "cu": float(cost_under),
        "co": float(cost_over),
        "sl": service_level(cost_under, cost_over),
        "metric 1": float("nan"),
        "metric 2": float("nan"),
        "epochs ran": None,
        "best epoch": None,
        "best val newsvendor": None,
        "elapsed seconds": float("nan"),
        "training reused": False,
        "initialization seed": None,
        "initialization path": None,
        "initial state sha256": None,
        "prediction path": None,
        "checkpoint path": None,
        "error": str(error),
    }


def write_outputs(
    output_dir: Path,
    dataset: str,
    feature_combi: str,
    results: pd.DataFrame,
    metadata: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "results_detail.csv"
    summary_path = output_dir / "summary_by_model_sl.csv"
    workbook_path = output_dir / f"{dataset}_{feature_combi}_results.xlsx"
    metadata_path = output_dir / "run_metadata.json"
    results.to_csv(detail_path, index=False)
    summary = (
        results.groupby(["model", "cu", "co", "sl"], as_index=False)[
            ["metric 1", "metric 2", "elapsed seconds"]
        ]
        .mean(numeric_only=True)
        .sort_values(["model", "sl"])
    )
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps(json_ready(metadata), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    metadata_frame = pd.DataFrame(
        [
            {
                "key": key,
                "value": json.dumps(json_ready(value), ensure_ascii=False)
                if isinstance(value, (dict, list, tuple))
                else value,
            }
            for key, value in metadata.items()
        ]
    )
    with pd.ExcelWriter(workbook_path) as writer:
        results.to_excel(writer, sheet_name="detail", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        metadata_frame.to_excel(writer, sheet_name="metadata", index=False)
    return {
        "dataset": dataset,
        "detail": str(detail_path),
        "summary": str(summary_path),
        "workbook": str(workbook_path),
        "metadata": str(metadata_path),
    }


def run_one_dataset(
    args: argparse.Namespace,
    dataset: str,
    model_specs: list[SplineModelSpec],
    experiment_name: str,
    extra_metadata: dict,
) -> dict:
    set_seed(args.seed)
    d3_root = Path(args.d3_root)
    output_dir = Path(args.output_dir) / dataset / args.feature_combi
    X, y = load_dataset(d3_root, dataset, args.auto_download)
    X_features, feature_combo = build_feature_frame(X, dataset, args.feature_combi)
    grouped = X_features.groupby(["store", "item"])
    groups = list(grouped.groups.keys())
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    print(
        f"\n[dataset] {dataset} feature={feature_combo} "
        f"models={[spec.key for spec in model_specs]} groups={len(groups)}",
        flush=True,
    )

    if args.summary_only:
        detail_path = output_dir / "results_detail.csv"
        if not detail_path.exists():
            raise FileNotFoundError(detail_path)
        existing = pd.read_csv(detail_path)
        metadata = {
            "experiment": experiment_name,
            "dataset": dataset,
            "feature_combi": args.feature_combi,
            "feature_categories": feature_combo,
            "models": [spec.display_name for spec in model_specs],
            **extra_metadata,
        }
        return write_outputs(
            output_dir,
            dataset,
            args.feature_combi,
            existing,
            metadata,
        )

    rows = []
    for group_index, group in enumerate(groups):
        print(f"[group {group_index + 1}/{len(groups)}] {group}", flush=True)
        data = split_group_data(
            group,
            grouped.get_group(group),
            y,
            args.val_fraction,
        )
        for cost_index, (cost_under, cost_over) in enumerate(COST_PAIRS):
            for spec in model_specs:
                try:
                    row = evaluate_one_model(
                        spec,
                        data,
                        args,
                        dataset,
                        feature_combo,
                        group,
                        group_index,
                        cost_index,
                        cost_under,
                        cost_over,
                        output_dir,
                    )
                    print(
                        f"  {spec.display_name} sl={row['sl']:.2f} "
                        f"metric1={row['metric 1']:.6f} "
                        f"metric2={row['metric 2']:.6f}",
                        flush=True,
                    )
                except Exception as error:
                    row = failed_result_row(
                        spec,
                        dataset,
                        feature_combo,
                        group,
                        cost_under,
                        cost_over,
                        error,
                    )
                    print(
                        f"[warn] {spec.display_name} group={group} "
                        f"sl={row['sl']:.2f}: {error}",
                        flush=True,
                    )
                rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "results_detail.csv", index=False)

    results = pd.DataFrame(rows)
    metadata = {
        "experiment": experiment_name,
        "dataset": dataset,
        "feature_combi": args.feature_combi,
        "feature_categories": feature_combo,
        "groups": [str(group) for group in groups],
        "models": [spec.display_name for spec in model_specs],
        "cost_pairs": COST_PAIRS,
        "metric2_cost_pairs": METRIC2_COST_PAIRS,
        "metric2_service_levels": [
            service_level(cost_under, cost_over)
            for cost_under, cost_over in METRIC2_COST_PAIRS
        ],
        "split": "first 75% train, last 25% test; validation is the final slice of train",
        "metric1": "Exact spline quantile for the row cost pair.",
        "metric2": "Average cost after recomputing an exact spline quantile for every metric2 cost pair.",
        "shared_initialization_dir": str(Path(args.shared_initialization_dir)),
        "arguments": vars(args),
        **extra_metadata,
    }
    output = write_outputs(
        output_dir,
        dataset,
        args.feature_combi,
        results,
        metadata,
    )
    print(f"[dataset done] {dataset}: {output['workbook']}", flush=True)
    return output


def run_real_world(
    args: argparse.Namespace,
    model_specs: list[SplineModelSpec],
    experiment_name: str,
    extra_metadata: dict | None = None,
) -> list[dict]:
    outputs = []
    for dataset in selected_datasets(args.dataset):
        outputs.append(
            run_one_dataset(
                args,
                dataset,
                model_specs,
                experiment_name,
                extra_metadata or {},
            )
        )
    print("\n[all done]", flush=True)
    for output in outputs:
        print(f"{output['dataset']}: {output['workbook']}", flush=True)
    return outputs


def add_common_arguments(
    parser: argparse.ArgumentParser,
    default_output_dir: str | Path | None,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--dataset",
        choices=["all", *sorted(DATASET_FEATURES)],
        default="all",
        help="Dataset to evaluate. Default: all four d3group datasets.",
    )
    parser.add_argument(
        "--feature-combi",
        choices=sorted(FEATURE_ALIASES),
        default="calendar",
        help="Feature combination: calendar, calendar_lag, or full.",
    )
    parser.add_argument("--d3-root", type=Path, default=Path("d3group_data"))
    parser.add_argument(
        "--auto-download",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-groups", type=parse_max_groups, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--shared-initialization-dir",
        type=Path,
        default=Path("analysis_outputs/d3_real_world_gendfl_initializations"),
        help="Shared initial checkpoints used by GenDFL, QFlow, and RSETO-IPA.",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-delta-relative", type=float, default=0.0)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--step-size-exponent",
        type=float,
        default=0.6,
        help="Exponent a in gamma_k = learning_rate / (k + 1)^a.",
    )
    parser.add_argument("--parameter-box-lower", type=float, default=-10.0)
    parser.add_argument("--parameter-box-upper", type=float, default=10.0)
    parser.add_argument("--num-transforms", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--tail-bound", type=float, default=4.0)
    parser.add_argument("--tau-eps", type=float, default=1e-5)
    parser.add_argument("--verbose-training", action="store_true")
    return parser


def validate_common_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error("--val-fraction must lie in [0, 1).")
    positive = {
        "epochs": args.epochs,
        "early-stopping": args.early_stopping,
        "batch-size": args.batch_size,
        "evaluation-batch-size": args.evaluation_batch_size,
        "learning-rate": args.learning_rate,
        "num-transforms": args.num_transforms,
        "num-bins": args.num_bins,
        "hidden-dim": args.hidden_dim,
        "hidden-layers": args.hidden_layers,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error(f"These arguments must be positive: {invalid}.")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("--warmup-epochs must lie in [0, epochs).")
    if args.min_delta_relative < 0:
        parser.error("--min-delta-relative must be nonnegative.")
    if not 0.5 < args.step_size_exponent <= 1.0:
        parser.error("--step-size-exponent must lie in (0.5, 1].")
    if args.parameter_box_lower >= args.parameter_box_upper:
        parser.error("The parameter box lower bound must be below its upper bound.")
    if args.output_dir is None:
        parser.error("--output-dir must be set.")
