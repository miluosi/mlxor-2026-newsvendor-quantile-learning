"""Real-world newsvendor tests aligned with d3group/ddnv.

This script mirrors the real-data protocol in
https://github.com/d3group/A-structured-evaluation-of-data-driven-newsvendor-approaches:

* datasets: m5, SID, yaz, bakery
* grouping: one model per (store, item)
* split: first 75% train, last 25% test within each group
* feature sets: calendar / calendar+lag / calendar+lag+dataset-specific features
* costs: (cu, co) = (9,1), (7.5,2.5), (5,5), (2.5,7.5), (1,9)
* metric 1: average newsvendor cost for the row's (cu, co)
* metric 2: series newsvendor loss over a fixed sl_list

The default run uses every local interface, every group, and all four real-world
datasets. Use --dataset and --max-groups for a smaller smoke test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


D3_REPO_RAW = (
    "https://raw.githubusercontent.com/d3group/"
    "A-structured-evaluation-of-data-driven-newsvendor-approaches/main"
)

ALL_MODELS = (
    "saa",
    "erm",
    "lightgbm",
    "benchmark",
    "end_to_end",
    "sipa",
    "lripa",
    "ipaonly",
    "sglr",
    "lrglr",
    "glronly",
)

MODEL_DISPLAY_NAMES = {
    "saa": "SAA",
    "lightgbm": "LightGBM",
    "benchmark": "Benchmark",
    "end_to_end": "EndToEndVAE",
    "sipa": "SIPA",
    "lripa": "LRIPA",
    "ipaonly": "IPAOnly",
    "sglr": "SGLR",
    "lrglr": "LRGLR",
    "glronly": "GLROnly",
}

FEATURE_CATEGORIES = {
    "calendar": ["weekday", "month", "year"],
    "lag": [
        "demand__sum_values_7",
        "demand__median_7",
        "demand__mean_7",
        "demand__standard_deviation_7",
        "demand__variance_7",
        "demand__root_mean_square_7",
        "demand__maximum_7",
        "demand__absolute_maximum_7",
        "demand__minimum_7",
        "demand__sum_values_14",
        "demand__median_14",
        "demand__mean_14",
        "demand__standard_deviation_14",
        "demand__variance_14",
        "demand__root_mean_square_14",
        "demand__maximum_14",
        "demand__absolute_maximum_14",
        "demand__minimum_14",
        "demand__sum_values_28",
        "demand__median_28",
        "demand__mean_28",
        "demand__standard_deviation_28",
        "demand__variance_28",
        "demand__root_mean_square_28",
        "demand__maximum_28",
        "demand__absolute_maximum_28",
        "demand__minimum_28",
    ],
    "special_yaz": [
        "is_holiday",
        "is_closed",
        "wind",
        "clouds",
        "rain",
        "sunshine",
        "temperature",
    ],
    "special_m5": [
        "is_sporting_event",
        "is_cultural_event",
        "is_national_event",
        "is_religious_event",
        "is_snap_day",
    ],
    "special_bakery": [
        "is_schoolholiday",
        "is_holiday",
        "is_holiday_next2days",
        "rain",
        "temperature",
        "promotion_currentweek",
        "promotion_lastweek",
    ],
}

DATASET_FEATURES = {
    "m5": [
        ["calendar"],
        ["calendar", "lag"],
        ["calendar", "lag", "special_m5"],
    ],
    "SID": [
        ["calendar"],
        ["calendar", "lag"],
    ],
    "yaz": [
        ["calendar"],
        ["calendar", "lag"],
        ["calendar", "lag", "special_yaz"],
    ],
    "bakery": [
        ["calendar"],
        ["calendar", "lag"],
        ["calendar", "lag", "special_bakery"],
    ],
}

FEATURE_ALIASES = {
    "calendar": 0,
    "calendar_lag": 1,
    "full": -1,
}

COST_PAIRS = [
    (9.0, 1.0),
    (7.5, 2.5),
    (5.0, 5.0),
    (2.5, 7.5),
    (1.0, 9.0),
]
METRIC2_COST_PAIRS = COST_PAIRS
ALL_DATASETS = tuple(DATASET_FEATURES)


@dataclass
class GroupData:
    group: tuple
    n_features: int
    X_train_model: np.ndarray
    X_val_model: np.ndarray
    X_fit_model: np.ndarray
    X_test_model: np.ndarray
    y_train_model: np.ndarray
    y_val_model: np.ndarray
    y_fit_model: np.ndarray
    y_test_raw: np.ndarray
    y_train_raw: np.ndarray
    scaler_y: StandardScaler


def _to_1d_numpy(values) -> np.ndarray:
    if isinstance(values, (pd.Series, pd.DataFrame)):
        values = values.to_numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def pandas_loss(alpha: float, beta: float) -> Callable:
    """Return a pandas-compatible newsvendor loss.

    beta may be passed as either positive overage cost or the old negative form.
    """

    def loss(y_true, y_pred) -> float:
        y_true_np = _to_1d_numpy(y_true)
        y_pred_np = _to_1d_numpy(y_pred)
        error = y_true_np - y_pred_np
        over_cost = abs(beta)
        loss_values = np.where(error > 0, alpha * error, over_cost * (-error))
        return float(pd.Series(loss_values).mean())

    return loss


def pandas_loss_series(alphalist: Iterable[float], betalist: Iterable[float]) -> Callable:
    """Return average newsvendor loss over a list of cost pairs."""

    def loss(y_true, y_pred) -> float:
        y_true_np = _to_1d_numpy(y_true)
        y_pred_np = _to_1d_numpy(y_pred)
        error = y_true_np - y_pred_np
        loss_values = []
        for alpha, beta in zip(alphalist, betalist):
            over_cost = abs(beta)
            loss_values.append(np.where(error > 0, alpha * error, over_cost * (-error)))
        return float(pd.Series(np.mean(loss_values, axis=0)).mean())

    return loss


def metric2_series_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alphalist: Iterable[float],
    betalist: Iterable[float],
) -> float:
    """Series loss for either one order quantity or one quantity per cost pair."""

    alpha_list = list(alphalist)
    beta_list = list(betalist)
    pred_np = np.asarray(y_pred, dtype=float)
    y_true_series = pd.Series(_to_1d_numpy(y_true))

    if pred_np.ndim == 1:
        return pandas_loss_series(alpha_list, beta_list)(y_true_series, pd.Series(pred_np))

    if pred_np.ndim != 2:
        raise ValueError("metric 2 predictions must be 1D or 2D.")

    if pred_np.shape[0] != len(alpha_list):
        if pred_np.shape[1] == len(alpha_list):
            pred_np = pred_np.T
        else:
            raise ValueError(
                "metric 2 prediction rows must match the number of alpha/beta pairs."
            )

    losses = [
        pandas_loss(alpha, beta)(y_true_series, pd.Series(pred_np[idx]))
        for idx, (alpha, beta) in enumerate(zip(alpha_list, beta_list))
    ]
    return float(np.mean(losses))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@contextmanager
def pushd(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def service_level(cu: float, co: float) -> float:
    return cu / (cu + co)


def cost_pair_key(cu: float, co: float) -> tuple[float, float]:
    return round(float(cu), 10), round(float(co), 10)


def split_cost_pairs(cost_pairs: Iterable[tuple[float, float]]) -> tuple[list[float], list[float]]:
    pairs = list(cost_pairs)
    return [cu for cu, _ in pairs], [co for _, co in pairs]


def service_levels_for_cost_pairs(cost_pairs: Iterable[tuple[float, float]]) -> list[float]:
    return [service_level(cu, co) for cu, co in cost_pairs]


def average_costs(y_true: np.ndarray, y_pred: np.ndarray, cu: float, co: float) -> float:
    return pandas_loss(cu, co)(pd.Series(y_true), pd.Series(y_pred))


def prescriptiveness_score(model_cost: float, saa_cost: float) -> float:
    if saa_cost == 0 or not np.isfinite(saa_cost):
        return float("nan")
    return float(1.0 - model_cost / saa_cost)


def empirical_quantile(values: np.ndarray, sl: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    try:
        return float(np.quantile(values, sl, method="linear"))
    except TypeError:
        return float(np.quantile(values, sl, interpolation="linear"))


def parse_max_groups(value: str | None) -> int | None:
    if value is None or value.lower() == "all":
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--max-groups must be a positive integer or 'all'")
    return parsed


def parse_models(raw: str) -> list[str]:
    models = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [model for model in models if model not in ALL_MODELS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown models: {invalid}. Valid: {list(ALL_MODELS)}")
    deduped = []
    for model in models:
        if model not in deduped:
            deduped.append(model)
    if "saa" not in deduped:
        deduped.insert(0, "saa")
    return deduped


def selected_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return list(ALL_DATASETS)
    return [dataset_arg]


def feature_combo_for(dataset: str, feature_combi: str) -> list[str]:
    if feature_combi not in FEATURE_ALIASES:
        raise ValueError(f"Unknown feature combination: {feature_combi}")
    idx = FEATURE_ALIASES[feature_combi]
    combos = DATASET_FEATURES[dataset]
    if idx == -1:
        return combos[-1]
    if idx >= len(combos):
        raise ValueError(f"{dataset} does not define feature combination {feature_combi}")
    return combos[idx]


def ensure_d3_file(data_root: Path, rel_path: str, auto_download: bool) -> Path:
    path = data_root / rel_path
    if path.exists():
        return path
    if not auto_download:
        raise FileNotFoundError(
            f"Missing {path}. Pass --auto-download or point --d3-root to a cloned d3group repo."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{D3_REPO_RAW}/{rel_path}"
    print(f"[download] {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def load_dataset(d3_root: Path, dataset: str, auto_download: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_path = ensure_d3_file(d3_root, f"Data/final/{dataset}_data.csv.zip", auto_download)
    y_path = ensure_d3_file(d3_root, f"Data/final/{dataset}_target.csv.zip", auto_download)
    return pd.read_csv(X_path), pd.read_csv(y_path)


def load_reference_results(
    d3_root: Path,
    dataset: str,
    auto_download: bool,
) -> pd.DataFrame | None:
    try:
        path = ensure_d3_file(d3_root, f"Results/best_results_{dataset}.csv.zip", auto_download)
    except Exception as exc:
        print(f"[warn] Could not load d3 reference results: {exc}")
        return None
    return pd.read_csv(path)


def build_feature_frame(X: pd.DataFrame, dataset: str, feature_combi: str) -> tuple[pd.DataFrame, list[str]]:
    combo = feature_combo_for(dataset, feature_combi)
    cols: list[str] = []
    for category in combo:
        cols.extend(FEATURE_CATEGORIES[category])

    required = cols + ["store", "item"]
    missing = [col for col in required if col not in X.columns]
    if missing:
        raise KeyError(f"{dataset} is missing expected columns: {missing}")

    X_cols = X[required].copy()
    dummy_cols = [col for col in ["weekday", "month"] if col in X_cols.columns]
    X_cols = pd.get_dummies(X_cols, columns=dummy_cols)
    return X_cols, combo


def split_group_data(
    group: tuple,
    X_group: pd.DataFrame,
    y: pd.DataFrame,
    val_fraction: float,
) -> GroupData:
    y_group = y.iloc[X_group.index.values.tolist()].to_numpy(dtype=np.float32).reshape(-1, 1)
    X_values = X_group.drop(["store", "item"], axis=1).to_numpy(dtype=np.float32)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_values,
        y_group,
        train_size=0.75,
        shuffle=False,
    )

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler_X.transform(X_test_raw).astype(np.float32)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw).astype(np.float32)

    if val_fraction > 0:
        split_at = max(1, int(math.floor((1.0 - val_fraction) * len(X_train_scaled))))
        split_at = min(split_at, len(X_train_scaled) - 1)
        X_fit_model = X_train_scaled[:split_at]
        y_fit_model = y_train_scaled[:split_at]
        X_val_model = X_train_scaled[split_at:]
        y_val_model = y_train_scaled[split_at:]
    else:
        X_fit_model = X_train_scaled
        y_fit_model = y_train_scaled
        X_val_model = X_train_scaled
        y_val_model = y_train_scaled

    return GroupData(
        group=group,
        n_features=X_train_scaled.shape[1],
        X_train_model=X_train_scaled,
        X_val_model=X_val_model,
        X_fit_model=X_fit_model,
        X_test_model=X_test_scaled,
        y_train_model=y_train_scaled,
        y_val_model=y_val_model,
        y_fit_model=y_fit_model,
        y_test_raw=y_test_raw.reshape(-1),
        y_train_raw=y_train_raw.reshape(-1),
        scaler_y=scaler_y,
    )


def model_matrix(X_model: np.ndarray, y_model: np.ndarray) -> np.ndarray:
    return np.hstack([X_model.astype(np.float32), y_model.astype(np.float32)]).astype(np.float32)


def predict_saa(data: GroupData, sl: float) -> np.ndarray:
    pred = empirical_quantile(data.y_train_raw, sl)
    return np.full_like(data.y_test_raw, pred, dtype=float)


def torch_device(args: argparse.Namespace):
    import torch

    return torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))


def make_tensor_loaders(data: GroupData, args: argparse.Namespace):
    import torch
    from torch.utils.data import DataLoader

    train_arr = torch.tensor(model_matrix(data.X_fit_model, data.y_fit_model), dtype=torch.float32)
    val_arr = torch.tensor(model_matrix(data.X_val_model, data.y_val_model), dtype=torch.float32)
    train_loader = DataLoader(train_arr, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_arr, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


def make_indexed_loaders(data: GroupData, args: argparse.Namespace):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    train_arr = torch.tensor(model_matrix(data.X_fit_model, data.y_fit_model), dtype=torch.float32)
    val_arr = torch.tensor(model_matrix(data.X_val_model, data.y_val_model), dtype=torch.float32)
    indices = torch.arange(train_arr.shape[0], dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(train_arr, indices),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_arr, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


def fit_predict_lightgbm(
    data: GroupData,
    sl: float,
    seed: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    n_jobs: int,
) -> np.ndarray:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError(
            "lightgbm is not installed. Install it or run without --models lightgbm."
        ) from exc

    model = LGBMRegressor(
        objective="quantile",
        alpha=sl,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=20,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )
    model.fit(data.X_train_model, data.y_train_model.reshape(-1))
    pred_scaled = model.predict(data.X_test_model).reshape(-1, 1)
    return data.scaler_y.inverse_transform(pred_scaled).reshape(-1)


def predict_generative_model(model, data: GroupData, args: argparse.Namespace, sl: float) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    preds: list[float] = []
    with torch.no_grad():
        for row in data.X_test_model:
            condition = torch.tensor(
                np.tile(row, (args.e2e_samples, 1)),
                dtype=torch.float32,
                device=device,
            )
            z = torch.randn(args.e2e_samples, args.latent, device=device)
            generated_scaled = model.decode(z, condition).detach().cpu().numpy().reshape(-1, 1)
            generated_raw = data.scaler_y.inverse_transform(generated_scaled).reshape(-1)
            preds.append(empirical_quantile(generated_raw, sl))
    return np.asarray(preds, dtype=float)


def predict_generative_model_series(
    model,
    data: GroupData,
    args: argparse.Namespace,
    sl_list: list[float],
) -> np.ndarray:
    import torch
    device = next(model.parameters()).device
    preds: list[list[float]] = [[] for _ in sl_list]
    with torch.no_grad():
        for row in data.X_test_model:
            condition = torch.tensor(
                np.tile(row, (args.e2e_samples, 1)),
                dtype=torch.float32,
                device=device,
            )
            z = torch.randn(args.e2e_samples, args.latent, device=device)
            generated_scaled = model.decode(z, condition).detach().cpu().numpy().reshape(-1, 1)
            generated_raw = data.scaler_y.inverse_transform(generated_scaled).reshape(-1)
            for idx, sl in enumerate(sl_list):
                preds[idx].append(empirical_quantile(generated_raw, sl))
    return np.asarray(preds, dtype=float)


def predict_point_model(model, data: GroupData) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        X_test = torch.tensor(data.X_test_model, dtype=torch.float32, device=device)
        pred_scaled = model(X_test).detach().cpu().numpy().reshape(-1, 1)
    return data.scaler_y.inverse_transform(pred_scaled).reshape(-1)


def fit_predict_benchmark(
    data: GroupData,
    sl: float,
    args: argparse.Namespace,
    group_idx: int,
) -> np.ndarray:
    import torch

    from model.benchmark import Benchmark

    train_loader, val_loader = make_tensor_loaders(data, args)
    checkpoint_dir = Path(args.run_output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_name = (
        f"d3_benchmark_{args.dataset}_{args.feature_combi}_g{group_idx}_"
        f"sl{int(round(sl * 100))}_seed{args.seed}"
    )
    model_path = checkpoint_dir / f"{save_name}.pth"

    model = Benchmark(
        alpha=sl,
        input_size=1,
        con_size=data.n_features,
        randnumber=args.seed + group_idx,
    ).to(torch_device(args))
    model.train(
        num_epochs=args.epochs,
        targetdim=1,
        traindata_loader=train_loader,
        valdata_loader=val_loader,
        early_stopping=args.early_stopping,
        model_save_path=str(model_path),
    )
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=torch_device(args)))
    else:
        print(f"[warn] Missing benchmark checkpoint {model_path}; using final epoch weights.")
    return predict_point_model(model, data)


def fit_predict_vae(
    data: GroupData,
    sl: float,
    sl_list: list[float],
    args: argparse.Namespace,
    group_idx: int,
    model_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from model.VAE_end_to_end import VAE_end_to_end

    train_loader, val_loader = make_tensor_loaders(data, args)
    save_name = (
        f"d3_{model_key}_{args.dataset}_{args.feature_combi}_g{group_idx}_"
        f"sl{int(round(sl * 100))}_seed{args.seed}"
    )
    checkpoint_dir = Path(args.run_output_dir) / "checkpoints"

    model = VAE_end_to_end(
        targetdim=1,
        labeldim=data.n_features,
        latent=args.latent,
        quantiles=sl,
        lambda1=args.lambda1,
        samplingnumber=args.e2e_samples,
    ).to(torch_device(args))

    with pushd(checkpoint_dir):
        if model_key in {"end_to_end", "lripa"}:
            model.trainconvae_sgd_2(
                num_epochs=args.epochs,
                targetdim=1,
                traindata_loader=train_loader,
                valdata_loader=val_loader,
                early_stopping=args.early_stopping,
                save_name=save_name,
                randomnumber=args.seed,
            )
        elif model_key == "sipa":
            model.trainconvae_sgd(
                num_epochs=args.epochs,
                targetdim=1,
                traindata_loader=train_loader,
                valdata_loader=val_loader,
                early_stopping=args.early_stopping,
                save_name=save_name,
                randomnumber=args.seed,
            )
        elif model_key == "ipaonly":
            model.traindecoderonly(
                num_epochs=args.epochs,
                targetdim=1,
                traindata_loader=train_loader,
                valdata_loader=val_loader,
                early_stopping=args.early_stopping,
                save_name=save_name,
                randomnumber=args.seed,
            )
        else:
            raise ValueError(f"Unsupported VAE model key: {model_key}")

        model_path = Path(model.get_save_path(save_name, args.seed, "MODEL"))
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=torch_device(args)))
        else:
            print(f"[warn] Missing {model_key} checkpoint {model_path}; using final epoch weights.")

    model.eval()
    pred = predict_generative_model(model, data, args, sl)
    pred_series = predict_generative_model_series(model, data, args, sl_list)
    return pred, pred_series


def fit_predict_glr(
    data: GroupData,
    sl: float,
    sl_list: list[float],
    cu: float,
    co: float,
    args: argparse.Namespace,
    group_idx: int,
    model_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from model.VAE_GLR_Model import VAE_GLR_Model

    train_loader, val_loader = make_indexed_loaders(data, args)
    save_name = (
        f"d3_{model_key}_{args.dataset}_{args.feature_combi}_g{group_idx}_"
        f"sl{int(round(sl * 100))}_seed{args.seed}"
    )
    checkpoint_dir = Path(args.run_output_dir) / "checkpoints"

    model = VAE_GLR_Model(
        targetdim=1,
        labeldim=data.n_features,
        latent=args.latent,
        data_len=len(data.y_fit_model),
        epoch=args.epochs,
        quantiles=sl,
        samplingnumber=args.glr_samples,
        target_quantile=sl,
        cost_under=cu,
        cost_over=co,
        random_seed=args.seed + group_idx,
        innerloop=args.glr_innerloop,
    ).to(torch_device(args))

    ifonlyglr = model_key == "glronly"
    iftwoupdate = model_key == "sglr"
    with pushd(checkpoint_dir):
        model.train_step_sqo_vectorized_SGD_LR_globalsingle(
            train_loader,
            val_loader,
            args.early_stopping,
            args.batch_size,
            ifdecoderonly=False,
            save_tag=save_name,
            ifonlyglr=ifonlyglr,
            iftwoupdate=iftwoupdate,
        )
        model_path = Path(model.get_save_path(save_name))
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=torch_device(args)))
        else:
            print(f"[warn] Missing {model_key} checkpoint {model_path}; using final epoch weights.")

    model.eval()
    pred = predict_generative_model(model, data, args, sl)
    pred_series = predict_generative_model_series(model, data, args, sl_list)
    return pred, pred_series


def fit_predict_model(
    model_key: str,
    data: GroupData,
    sl: float,
    sl_list: list[float],
    cu: float,
    co: float,
    args: argparse.Namespace,
    group_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    if model_key == "lightgbm":
        pred = fit_predict_lightgbm(
            data,
            sl,
            seed=args.seed + group_idx,
            n_estimators=args.lgb_n_estimators,
            learning_rate=args.lgb_learning_rate,
            num_leaves=args.lgb_num_leaves,
            n_jobs=args.n_jobs,
        )
        return pred, pred
    if model_key == "benchmark":
        pred = fit_predict_benchmark(data, sl, args, group_idx)
        return pred, pred
    if model_key in {"end_to_end", "sipa", "lripa", "ipaonly"}:
        return fit_predict_vae(data, sl, sl_list, args, group_idx, model_key)
    if model_key in {"sglr", "lrglr", "glronly"}:
        return fit_predict_glr(data, sl, sl_list, cu, co, args, group_idx, model_key)
    raise ValueError(f"Unsupported model: {model_key}")


def evaluate_predictions(
    dataset: str,
    feature_combo: list[str],
    group: tuple,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_metric2: np.ndarray,
    cu: float,
    co: float,
    alpha_list: Iterable[float],
    beta_list: Iterable[float],
    saa_cost: float,
    elapsed: float,
    error: str | None = None,
) -> dict:
    if error:
        avg = float("nan")
        sop = float("nan")
    else:
        avg = np.round(pandas_loss(cu, co)(pd.Series(y_true), pd.Series(y_pred)), 4)
        sop = np.round(metric2_series_loss(y_true, y_pred_metric2, alpha_list, beta_list), 4)

    return {
        "dataset": dataset,
        "feature combi": str(feature_combo),
        "group": str(group),
        "model": model_name,
        "cu": cu,
        "co": co,
        "sl": service_level(cu, co),
        "metric 1": avg,
        "metric 2": sop,
        "saa average costs": round(saa_cost, 4),
        "elapsed seconds": round(elapsed, 3),
        "error": error,
    }


def reference_subset(
    reference: pd.DataFrame | None,
    feature_combo: list[str],
    groups: Iterable[tuple],
) -> pd.DataFrame | None:
    if reference is None:
        return None
    group_set = {group if isinstance(group, str) else str(group) for group in groups}
    ref = reference[
        (reference["feature combi"] == str(feature_combo))
        & (reference["group"].isin(group_set))
    ].copy()
    if ref.empty:
        return None
    ref["source"] = "d3_reference"
    return ref


def add_reference_metrics(
    reference: pd.DataFrame,
    metric2_cost_pairs: Iterable[tuple[float, float]],
    use_metric2_proxy: bool = False,
) -> pd.DataFrame:
    reference_metrics = reference.copy()
    if "metric 1" not in reference_metrics.columns and "average costs" in reference_metrics.columns:
        reference_metrics["metric 1"] = reference_metrics["average costs"]

    if "metric 2" in reference_metrics.columns:
        return reference_metrics

    if not use_metric2_proxy:
        reference_metrics["metric 2"] = np.nan
        return reference_metrics

    wanted_pairs = {cost_pair_key(cu, co) for cu, co in metric2_cost_pairs}
    cost_keys = [
        cost_pair_key(cu, co)
        for cu, co in zip(reference_metrics["cu"], reference_metrics["co"])
    ]
    metric2_source = reference_metrics[pd.Series(cost_keys, index=reference_metrics.index).isin(wanted_pairs)]

    metric2_by_model_group = (
        metric2_source.groupby(["dataset", "feature combi", "group", "model"], as_index=False)[
            "metric 1"
        ]
        .mean(numeric_only=True)
        .rename(columns={"metric 1": "metric 2"})
    )
    return reference_metrics.merge(
        metric2_by_model_group,
        on=["dataset", "feature combi", "group", "model"],
        how="left",
    )


def write_alignment_summary(
    output_dir: Path,
    results: pd.DataFrame,
    reference: pd.DataFrame | None,
    dataset: str,
    feature_combo: list[str],
    metric2_cost_pairs: Iterable[tuple[float, float]],
    use_reference_metric2_proxy: bool,
) -> Path:
    summary_rows = []
    ours = (
        results.groupby(["model", "sl"], as_index=False)[
            ["metric 1", "metric 2"]
        ]
        .mean(numeric_only=True)
        .assign(source="ours")
    )
    summary_rows.append(ours)

    if reference is not None and not reference.empty:
        reference_metrics = add_reference_metrics(
            reference,
            metric2_cost_pairs,
            use_metric2_proxy=use_reference_metric2_proxy,
        )
        ref_summary = (
            reference_metrics.groupby(["model", "sl"], as_index=False)[
                ["metric 1", "metric 2"]
            ]
            .mean(numeric_only=True)
            .assign(source="d3_reference")
        )
        summary_rows.append(ref_summary)

    summary = pd.concat(summary_rows, ignore_index=True)
    summary.insert(0, "dataset", dataset)
    summary.insert(1, "feature combi", str(feature_combo))
    path = output_dir / "summary_by_model_sl.csv"
    summary.to_csv(path, index=False)
    return path


def build_metadata(
    run_args: argparse.Namespace,
    d3_root: Path,
    feature_combo: list[str],
    groups: Iterable,
    metric2_cost_pairs: Iterable[tuple[float, float]],
    sl_list: list[float],
    detail_path: Path,
    summary_path: Path,
    workbook_path: Path,
) -> dict:
    reference_metric2_note = (
        "For d3 reference methods, metric 2 is a proxy derived by averaging published average costs over metric2_cost_pairs; raw predictions are unavailable."
        if run_args.reference_metric2_proxy
        else "For d3 reference methods, metric 2 is left NaN because raw predictions are unavailable; published average costs cannot produce comparable series loss."
    )
    return {
        "dataset": run_args.dataset,
        "feature_combi": run_args.feature_combi,
        "feature_categories": feature_combo,
        "groups": [group if isinstance(group, str) else str(group) for group in groups],
        "cost_pairs": COST_PAIRS,
        "metric2_cost_pairs": list(metric2_cost_pairs),
        "metric2_service_levels": sl_list,
        "models": run_args.models,
        "model_display_names": MODEL_DISPLAY_NAMES,
        "d3_root": str(d3_root),
        "outputs": {
            "detail": str(detail_path),
            "summary": str(summary_path),
            "workbook": str(workbook_path),
        },
        "notes": [
            "SAA rows should match d3group reference rows for the same group/sl.",
            "metric 1 is computed with pandas_loss(cu, co) for the row's service level.",
            "metric 2 is the series loss over metric2_cost_pairs.",
            "For VAE/GLR models, metric 2 uses one generated quantile prediction for each metric2_service_level.",
            "For point-prediction models, metric 2 reuses the row's single prediction for all metric2_cost_pairs.",
            reference_metric2_note,
            "LightGBM is not a d3group estimator; compare it against d3 LR/DL/RFW/etc. on the aligned split and metric.",
            "Neural models use a validation slice from the d3 75% train window for early stopping.",
        ],
    }


def write_dataset_workbook(
    workbook_path: Path,
    results: pd.DataFrame,
    summary_path: Path,
    metadata: dict,
) -> Path:
    summary = pd.read_csv(summary_path)
    metadata_rows = [
        {
            "key": key,
            "value": json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list, tuple))
            else value,
        }
        for key, value in metadata.items()
    ]
    metadata_frame = pd.DataFrame(metadata_rows)

    try:
        with pd.ExcelWriter(workbook_path) as writer:
            results.to_excel(writer, sheet_name="detail", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            metadata_frame.to_excel(writer, sheet_name="metadata", index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Writing .xlsx output requires openpyxl or xlsxwriter. "
            "Install one of them, or use the CSV/JSON outputs already written."
        ) from exc
    return workbook_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *sorted(DATASET_FEATURES)],
        default="all",
        help="Dataset to evaluate. Default: all four d3 real-world datasets.",
    )
    parser.add_argument(
        "--feature-combi",
        choices=sorted(FEATURE_ALIASES),
        default="calendar",
        help="d3group feature combination: calendar, calendar_lag, or full.",
    )
    parser.add_argument(
        "--d3-root",
        default="d3group_data",
        help="Directory containing Data/final and Results, or a cloned d3group repo.",
    )
    parser.add_argument(
        "--auto-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download d3group final data/results from GitHub when missing.",
    )
    parser.add_argument(
        "--models",
        type=parse_models,
        default=parse_models("saa,lightgbm,benchmark,end_to_end,sipa,lripa,ipaonly,sglr,lrglr,glronly"),
        help=f"Comma-separated subset of {','.join(ALL_MODELS)}. SAA is always added.",
    )
    parser.add_argument(
        "--max-groups",
        type=parse_max_groups,
        default=None,
        help="Number of (store, item) groups to evaluate, or 'all'. Default: all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="analysis_outputs/d3_real_world")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild summary CSV/XLSX from existing results_detail.csv files without retraining.",
    )
    parser.add_argument(
        "--reference-metric2-proxy",
        action="store_true",
        help=(
            "Fill d3 reference metric 2 with the old proxy average of published "
            "average costs. Off by default because it is not comparable to series loss."
        ),
    )

    parser.add_argument("--lgb-n-estimators", type=int, default=300)
    parser.add_argument("--lgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgb-num-leaves", type=int, default=31)
    parser.add_argument("--n-jobs", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent", type=int, default=5)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--e2e-samples", type=int, default=200)
    parser.add_argument("--glr-samples", type=int, default=100)
    parser.add_argument("--glr-innerloop", type=int, default=1)
    parser.add_argument("--device", default=None, help="Force torch device, e.g. cpu or cuda.")
    return parser


def append_result_row(
    rows: list[dict],
    args: argparse.Namespace,
    feature_combo: list[str],
    group: tuple,
    model_key: str,
    data: GroupData,
    pred: np.ndarray,
    pred_metric2: np.ndarray,
    cu: float,
    co: float,
    alpha_list: Iterable[float],
    beta_list: Iterable[float],
    saa_cost: float,
    elapsed: float,
    error: str | None,
) -> None:
    rows.append(
        evaluate_predictions(
            args.dataset,
            feature_combo,
            group,
            MODEL_DISPLAY_NAMES[model_key],
            data.y_test_raw,
            pred,
            pred_metric2,
            cu,
            co,
            alpha_list,
            beta_list,
            saa_cost,
            elapsed=elapsed,
            error=error,
        )
    )


def run_single_dataset(args: argparse.Namespace, dataset: str) -> dict:
    run_args = argparse.Namespace(**vars(args))
    run_args.dataset = dataset
    set_seed(run_args.seed)

    d3_root = Path(run_args.d3_root)
    output_dir = Path(run_args.output_dir) / run_args.dataset / run_args.feature_combi
    output_dir.mkdir(parents=True, exist_ok=True)
    run_args.run_output_dir = str(output_dir)

    print(f"\n[config] dataset={run_args.dataset}, feature={run_args.feature_combi}, models={run_args.models}")
    X, y = load_dataset(d3_root, run_args.dataset, run_args.auto_download)
    reference = load_reference_results(d3_root, run_args.dataset, run_args.auto_download)
    X_features, feature_combo = build_feature_frame(X, run_args.dataset, run_args.feature_combi)

    grouped = X_features.groupby(["store", "item"])
    groups = list(grouped.groups.keys())
    max_groups = run_args.max_groups
    if max_groups is not None:
        groups = groups[:max_groups]
    print(f"[data] X={X.shape}, y={y.shape}, groups selected={len(groups)}")

    selected_reference = reference_subset(reference, feature_combo, groups)
    rows: list[dict] = []
    metric2_cost_pairs = METRIC2_COST_PAIRS
    metric2_alphalist, metric2_betalist = split_cost_pairs(metric2_cost_pairs)
    sl_list = service_levels_for_cost_pairs(metric2_cost_pairs)
    for group_idx, group in enumerate(groups):
        print(f"\n[group {group_idx + 1}/{len(groups)}] {group}")
        data = split_group_data(group, grouped.get_group(group), y, run_args.val_fraction)
        for cu, co in COST_PAIRS:
            sl = service_level(cu, co)
            saa_pred = predict_saa(data, sl)
            saa_pred_series = np.vstack([predict_saa(data, metric_sl) for metric_sl in sl_list])
            saa_cost = pandas_loss(cu, co)(pd.Series(data.y_test_raw), pd.Series(saa_pred))
            append_result_row(
                rows,
                run_args,
                feature_combo,
                group,
                "saa",
                data,
                saa_pred,
                saa_pred_series,
                cu,
                co,
                metric2_alphalist,
                metric2_betalist,
                saa_cost,
                elapsed=0.0,
                error=None,
            )

            for model_key in run_args.models:
                if model_key == "saa":
                    continue
                start = time.time()
                error = None
                try:
                    pred, pred_series = fit_predict_model(
                        model_key,
                        data,
                        sl,
                        sl_list,
                        cu,
                        co,
                        run_args,
                        group_idx,
                    )
                except Exception as exc:
                    pred = np.full_like(data.y_test_raw, np.nan, dtype=float)
                    pred_series = pred
                    error = str(exc)
                    print(f"[warn] {model_key} failed for group={group}, sl={sl:.2f}: {error}")
                append_result_row(
                    rows,
                    run_args,
                    feature_combo,
                    group,
                    model_key,
                    data,
                    pred,
                    pred_series,
                    cu,
                    co,
                    metric2_alphalist,
                    metric2_betalist,
                    saa_cost,
                    elapsed=time.time() - start,
                    error=error,
                )

    results = pd.DataFrame(rows)
    detail_path = output_dir / "results_detail.csv"
    results.to_csv(detail_path, index=False)
    summary_path = write_alignment_summary(
        output_dir,
        results,
        selected_reference,
        run_args.dataset,
        feature_combo,
        metric2_cost_pairs,
        run_args.reference_metric2_proxy,
    )

    workbook_path = output_dir / f"{run_args.dataset}_{run_args.feature_combi}_results.xlsx"
    metadata = build_metadata(
        run_args,
        d3_root,
        feature_combo,
        groups,
        metric2_cost_pairs,
        sl_list,
        detail_path,
        summary_path,
        workbook_path,
    )
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_dataset_workbook(workbook_path, results, summary_path, metadata)

    print("\n[dataset done]")
    print(f"detail:  {detail_path}")
    print(f"summary: {summary_path}")
    print(f"meta:    {metadata_path}")
    print(f"xlsx:    {workbook_path}")
    print("\n[summary]")
    print(pd.read_csv(summary_path).to_string(index=False))
    return {
        "dataset": run_args.dataset,
        "detail": str(detail_path),
        "summary": str(summary_path),
        "metadata": str(metadata_path),
        "workbook": str(workbook_path),
    }


def rebuild_single_dataset_outputs(args: argparse.Namespace, dataset: str) -> dict:
    run_args = argparse.Namespace(**vars(args))
    run_args.dataset = dataset

    d3_root = Path(run_args.d3_root)
    output_dir = Path(run_args.output_dir) / run_args.dataset / run_args.feature_combi
    detail_path = output_dir / "results_detail.csv"
    if not detail_path.exists():
        raise FileNotFoundError(
            f"Missing {detail_path}. Run training first or point --output-dir to existing outputs."
        )

    results = pd.read_csv(detail_path)
    feature_combo = feature_combo_for(run_args.dataset, run_args.feature_combi)
    groups = results["group"].dropna().drop_duplicates().tolist()
    reference = load_reference_results(d3_root, run_args.dataset, run_args.auto_download)
    selected_reference = reference_subset(reference, feature_combo, groups)

    metric2_cost_pairs = METRIC2_COST_PAIRS
    sl_list = service_levels_for_cost_pairs(metric2_cost_pairs)
    summary_path = write_alignment_summary(
        output_dir,
        results,
        selected_reference,
        run_args.dataset,
        feature_combo,
        metric2_cost_pairs,
        run_args.reference_metric2_proxy,
    )

    workbook_path = output_dir / f"{run_args.dataset}_{run_args.feature_combi}_results.xlsx"
    metadata = build_metadata(
        run_args,
        d3_root,
        feature_combo,
        groups,
        metric2_cost_pairs,
        sl_list,
        detail_path,
        summary_path,
        workbook_path,
    )
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_dataset_workbook(workbook_path, results, summary_path, metadata)

    print(f"[summary rebuilt] {run_args.dataset}: {workbook_path}")
    return {
        "dataset": run_args.dataset,
        "detail": str(detail_path),
        "summary": str(summary_path),
        "metadata": str(metadata_path),
        "workbook": str(workbook_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0 <= args.val_fraction < 1:
        parser.error("--val-fraction must be in [0, 1).")

    datasets = selected_datasets(args.dataset)
    print(f"[run] datasets={datasets}")
    if args.summary_only:
        outputs = [rebuild_single_dataset_outputs(args, dataset) for dataset in datasets]
    else:
        outputs = [run_single_dataset(args, dataset) for dataset in datasets]

    print("\n[all done]")
    for output in outputs:
        print(f"{output['dataset']}: {output['workbook']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
