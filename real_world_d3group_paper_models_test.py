"""Retest the d3group paper models with metric 1 and metric 2.

This file intentionally evaluates the methods used by the d3group paper,
plus the ERM baseline used in the broader d3group-aligned tests:

    SAA, ERM, LR, DTW, RFW, KNNW, KW, DL

The data protocol is shared with real_world_d3group_test.py:

* datasets: m5, SID, yaz, bakery
* grouping: one model per (store, item)
* split: first 75% train, last 25% test within each group
* feature sets: calendar / calendar+lag / full
* costs: (cu, co) = (9,1), (7.5,2.5), (5,5), (2.5,7.5), (1,9)
* metric 1: newsvendor average cost for the row's (cu, co)
* metric 2: average series loss over the full cost-pair list, reusing the
  current row's single prediction for every cost pair

If ddop is installed, --engine auto uses the original ddop estimators from the
d3group script. Otherwise, the script falls back to local implementations with
the same method labels and aligned loss protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from gurobipy import GRB, Model, quicksum
import gurobipy as gp
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor

from real_world_d3group_test import (
    ALL_DATASETS,
    COST_PAIRS,
    DATASET_FEATURES,
    FEATURE_ALIASES,
    METRIC2_COST_PAIRS,
    add_reference_metrics,
    build_feature_frame,
    empirical_quantile,
    load_dataset,
    load_reference_results,
    metric2_series_loss,
    pandas_loss,
    parse_max_groups,
    reference_subset,
    selected_datasets,
    service_level,
    service_levels_for_cost_pairs,
    set_seed,
    split_cost_pairs,
    split_group_data,
    write_dataset_workbook,
)


def solveconditionalgurobi(Y, X, cu, co, X_test):
    dim_x = X.shape[1]
    n_samples = X.shape[0]
    model = Model("linear_regression_l1")
    model.setParam("OutputFlag", 0)
    W_plus = [model.addVar(lb = 0, vtype=GRB.CONTINUOUS, name=f"Wplus_{k}") 
                for k in range(dim_x)]
    W_minus = [model.addVar(lb = 0, vtype=GRB.CONTINUOUS, name=f"Wminus_{k}") 
                for k in range(dim_x)]
    b = model.addVar(lb=-GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")
    z = [model.addVar(lb = 0, vtype=GRB.CONTINUOUS, name=f"z_{i}") 
            for i in range(n_samples)]
    for i in range(n_samples):
        expr = gp.LinExpr()
        for k in range(dim_x):
            expr += (W_plus[k] - W_minus[k]) * X[i, k]
        expr += b
        model.addConstr(z[i] >= cu * (Y[i].item() - expr))
        model.addConstr(z[i] >= co * (expr - Y[i].item()))
    model.setObjective(quicksum(z[i] for i in range(n_samples)), GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"ERM solve failed with Gurobi status {model.Status}")
    W_opt = np.zeros((dim_x,))
    for k in range(dim_x):
        W_opt[k] = W_plus[k].X - W_minus[k].X
    b_opt = b.X
    y_pred = W_opt @ X_test.T + b_opt
    return y_pred.reshape(-1)

def fit_erm_model(
    model_key: str,
    y_train: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    cu: float,
    co: float,
) -> np.ndarray:
    if model_key != "erm":
        raise ValueError(f"Unsupported model: {model_key}")
    return solveconditionalgurobi(
        Y=y_train,
        X=X_train,
        cu=cu,
        co=co,
        X_test=X_test,
    )









PAPER_MODELS = ("saa", "erm", "lr", "dtw", "rfw", "knnw", "kw", "dl")

MODEL_DISPLAY_NAMES = {
    "saa": "SAA",
    "erm": "ERM",
    "lr": "LR",
    "dtw": "DTW",
    "rfw": "RFW",
    "knnw": "KNNW",
    "kw": "KW",
    "dl": "DL",
}


def parse_models(raw: str) -> list[str]:
    models = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [model for model in models if model not in PAPER_MODELS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown paper models: {invalid}. Valid: {list(PAPER_MODELS)}"
        )
    deduped = []
    for model in models:
        if model not in deduped:
            deduped.append(model)
    return deduped


def parse_optional_int(raw: str) -> int | None:
    if raw.lower() in {"none", "null"}:
        return None
    return int(raw)


def parse_neurons(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or raw.strip().lower() in {"", "auto", "none"}:
        return None
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        return None
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("--dl-neurons must contain positive integers")
    return values


def ddop_is_available() -> bool:
    try:
        import ddop  # noqa: F401

        return True
    except Exception:
        return False


def resolve_engine(args: argparse.Namespace) -> str:
    if args.engine == "local":
        return "local"
    if args.engine == "ddop":
        if not ddop_is_available():
            raise RuntimeError("Requested --engine ddop, but package 'ddop' is not installed.")
        return "ddop"
    return "ddop" if ddop_is_available() else "local"


def clean_params(params: dict | None) -> dict:
    if not params:
        return {}
    cleaned = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, tuple):
            value = tuple(int(v) if isinstance(v, np.integer) else v for v in value)
        cleaned[key] = value
    return cleaned


def params_to_json(params: dict | None) -> str:
    return json.dumps(clean_params(params), sort_keys=True)


def inverse_target(data, pred_scaled: np.ndarray) -> np.ndarray:
    pred_scaled = np.asarray(pred_scaled, dtype=float).reshape(-1, 1)
    return data.scaler_y.inverse_transform(pred_scaled).reshape(-1)


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan")
    if weights is None:
        return empirical_quantile(values, quantile)

    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != values.size:
        raise ValueError("values and weights must have the same length")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return empirical_quantile(values, quantile)

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    threshold = quantile * cumulative[-1]
    return float(values[np.searchsorted(cumulative, threshold, side="left")])


def default_neurons(n_features: int) -> tuple[int, int]:
    first = max(4, int(2 * n_features))
    second = max(4, int(n_features))
    return first, second


def default_params(model_key: str, n_features: int, args: argparse.Namespace) -> dict:
    if model_key == "dtw":
        return {
            "max_depth": args.tree_max_depth,
            "min_samples_split": args.tree_min_samples_split,
            "min_samples_leaf": args.tree_min_samples_leaf,
        }
    if model_key == "rfw":
        return {
            "max_depth": args.rf_max_depth,
            "min_samples_split": args.rf_min_samples_split,
            "min_samples_leaf": args.rf_min_samples_leaf,
            "n_estimators": args.rf_n_estimators,
        }
    if model_key == "knnw":
        return {"n_neighbors": args.knn_k}
    if model_key == "kw":
        return {"kernel_bandwidth": args.kw_bandwidth}
    if model_key == "dl":
        return {
            "optimizer": "adam",
            "neurons": args.dl_neurons or default_neurons(n_features),
            "epochs": args.dl_epochs,
        }
    if model_key == "lr":
        return {"alpha": args.lr_alpha}
    return {}


def paper_grid(model_key: str, n_features: int, profile: str) -> list[dict]:
    if profile == "small":
        if model_key == "dtw":
            return list(
                ParameterGrid(
                    {
                        "max_depth": [None, 4, 8],
                        "min_samples_split": [2, 16],
                    }
                )
            )
        if model_key == "rfw":
            return list(
                ParameterGrid(
                    {
                        "max_depth": [None, 6],
                        "min_samples_split": [2, 16],
                        "n_estimators": [20, 50],
                    }
                )
            )
        if model_key == "knnw":
            return list(ParameterGrid({"n_neighbors": [4, 16, 64]}))
        if model_key == "kw":
            upper = max(0.5, math.sqrt(max(n_features, 1) / 2))
            return list(ParameterGrid({"kernel_bandwidth": [0.5, min(1.0, upper), upper]}))
        if model_key == "dl":
            return list(
                ParameterGrid(
                    {
                        "optimizer": ["adam"],
                        "neurons": [default_neurons(n_features)],
                        "epochs": [10, 100],
                    }
                )
            )
        return [{}]

    if model_key == "dtw":
        return list(
            ParameterGrid(
                {
                    "max_depth": [None, 2, 4, 6, 8, 10],
                    "min_samples_split": [2, 4, 6, 8, 16, 32, 64],
                }
            )
        )
    if model_key == "rfw":
        return list(
            ParameterGrid(
                {
                    "max_depth": [None, 2, 4, 6, 8, 10],
                    "min_samples_split": [2, 4, 6, 8, 16, 32, 64],
                    "n_estimators": [10, 20, 50, 100],
                }
            )
        )
    if model_key == "knnw":
        return list(ParameterGrid({"n_neighbors": [1, 2, 4, 8, 16, 32, 64, 128]}))
    if model_key == "kw":
        upper = max(0.5, math.sqrt(max(n_features, 1) / 2))
        values = np.arange(0.5, int(upper) + 0.25, 0.25)
        if values.size == 0:
            values = np.array([0.5])
        return list(ParameterGrid({"kernel_bandwidth": values.tolist()}))
    if model_key == "dl":
        n = n_features
        neurons = [
            (round(0.5 * n), round(0.5 * 0.5 * n)),
            (round(0.5 * n), round(0.5 * 1 * n)),
            (1 * n, round(1 * 0.5 * n)),
            (1 * n, 1 * n),
            (2 * n, round(2 * 0.5 * n)),
            (2 * n, 2 * n),
            (3 * n, round(3 * 0.5 * n)),
            (3 * n, 3 * n),
        ]
        neurons = [tuple(max(1, int(v)) for v in pair) for pair in neurons]
        return list(ParameterGrid({"optimizer": ["adam"], "neurons": neurons, "epochs": [10, 100, 200]}))
    return [{}]


def fit_predict_lr_local(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    sl: float,
    args: argparse.Namespace,
    params: dict,
) -> np.ndarray:
    alpha = float(params.get("alpha", args.lr_alpha))
    try:
        model = QuantileRegressor(quantile=sl, alpha=alpha, solver=args.lr_solver)
        model.fit(X_train, y_train)
        return model.predict(X_test)
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor

        fallback = GradientBoostingRegressor(
            loss="quantile",
            alpha=sl,
            n_estimators=args.lr_fallback_estimators,
            max_depth=3,
            learning_rate=0.05,
            random_state=args.seed,
        )
        fallback.fit(X_train, y_train)
        return fallback.predict(X_test)


def fit_predict_dtw_local(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    sl: float,
    args: argparse.Namespace,
    params: dict,
    seed: int,
) -> np.ndarray:
    model = DecisionTreeRegressor(
        max_depth=params.get("max_depth", args.tree_max_depth),
        min_samples_split=int(params.get("min_samples_split", args.tree_min_samples_split)),
        min_samples_leaf=int(params.get("min_samples_leaf", args.tree_min_samples_leaf)),
        random_state=seed,
    )
    model.fit(X_train, y_train)
    train_leaf = model.apply(X_train)
    test_leaf = model.apply(X_test)
    leaf_values: dict[int, np.ndarray] = {}
    for leaf in np.unique(train_leaf):
        leaf_values[int(leaf)] = y_train[train_leaf == leaf]
    fallback = empirical_quantile(y_train, sl)
    preds = [
        weighted_quantile(leaf_values.get(int(leaf), np.array([fallback])), sl)
        for leaf in test_leaf
    ]
    return np.asarray(preds, dtype=float)


def fit_predict_knnw_local(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    sl: float,
    args: argparse.Namespace,
    params: dict,
) -> np.ndarray:
    k = min(int(params.get("n_neighbors", args.knn_k)), len(y_train))
    k = max(1, k)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_test)
    preds = []
    for row_dist, row_idx in zip(distances, indices):
        if args.knn_weighting == "distance":
            weights = 1.0 / np.maximum(row_dist, 1e-8)
        else:
            weights = np.ones_like(row_dist)
        preds.append(weighted_quantile(y_train[row_idx], sl, weights))
    return np.asarray(preds, dtype=float)


def estimate_bandwidth(X_train: np.ndarray, args: argparse.Namespace) -> float:
    sample = X_train
    if len(sample) > args.bandwidth_sample:
        rng = np.random.default_rng(args.seed)
        sample = sample[rng.choice(len(sample), size=args.bandwidth_sample, replace=False)]
    distances = pairwise_distances(sample)
    upper = distances[np.triu_indices_from(distances, k=1)]
    upper = upper[np.isfinite(upper) & (upper > 0)]
    if upper.size == 0:
        return 1.0
    return float(np.median(upper))


def fit_predict_kw_local(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    sl: float,
    args: argparse.Namespace,
    params: dict,
) -> np.ndarray:
    bandwidth = float(params.get("kernel_bandwidth", args.kw_bandwidth))
    if bandwidth <= 0:
        bandwidth = estimate_bandwidth(X_train, args)

    preds = []
    for start in range(0, len(X_test), args.kernel_chunk_size):
        end = min(start + args.kernel_chunk_size, len(X_test))
        distances = pairwise_distances(X_test[start:end], X_train)
        weights = np.exp(-0.5 * (distances / max(bandwidth, 1e-8)) ** 2)
        preds.extend(weighted_quantile(y_train, sl, row) for row in weights)
    return np.asarray(preds, dtype=float)


def fit_predict_rfw_local(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    sl: float,
    args: argparse.Namespace,
    params: dict,
    seed: int,
) -> np.ndarray:
    forest = RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", args.rf_n_estimators)),
        max_depth=params.get("max_depth", args.rf_max_depth),
        min_samples_split=int(params.get("min_samples_split", args.rf_min_samples_split)),
        min_samples_leaf=int(params.get("min_samples_leaf", args.rf_min_samples_leaf)),
        random_state=seed,
        n_jobs=args.n_jobs,
    )
    forest.fit(X_train, y_train)

    weight_maps = []
    for tree in forest.estimators_:
        train_leaf = tree.apply(X_train)
        test_leaf = tree.apply(X_test)
        leaf_to_indices: dict[int, np.ndarray] = {}
        for leaf in np.unique(train_leaf):
            leaf_to_indices[int(leaf)] = np.flatnonzero(train_leaf == leaf)
        weight_maps.append((test_leaf, leaf_to_indices))

    preds = []
    for test_idx in range(len(X_test)):
        weights = np.zeros(len(y_train), dtype=float)
        for test_leaf, leaf_to_indices in weight_maps:
            indices = leaf_to_indices.get(int(test_leaf[test_idx]))
            if indices is not None and indices.size:
                weights[indices] += 1.0 / indices.size
        preds.append(weighted_quantile(y_train, sl, weights))
    return np.asarray(preds, dtype=float)


def fit_predict_dl_local(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    sl: float,
    args: argparse.Namespace,
    params: dict,
    seed: int,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    neurons = tuple(int(v) for v in params.get("neurons", args.dl_neurons or default_neurons(X_train.shape[1])))
    epochs = int(params.get("epochs", args.dl_epochs))

    layers: list[nn.Module] = []
    in_features = X_train.shape[1]
    for width in neurons:
        layers.append(nn.Linear(in_features, width))
        layers.append(nn.ReLU())
        in_features = width
    layers.append(nn.Linear(in_features, 1))
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.dl_learning_rate, weight_decay=args.dl_weight_decay)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=args.batch_size,
        shuffle=True,
    )

    use_val = (
        X_val is not None
        and y_val is not None
        and args.val_fraction > 0
        and len(X_val) > 0
    )
    if use_val:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32, device=device)
    else:
        X_val_tensor = None
        y_val_tensor = None

    def pinball(pred, target):
        err = target - pred
        return torch.maximum(sl * err, (sl - 1.0) * err).mean()

    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    for _epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = pinball(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.dl_clip_grad)
            optimizer.step()

        if use_val:
            model.eval()
            with torch.no_grad():
                val_loss = float(pinball(model(X_val_tensor), y_val_tensor).detach().cpu())
            if val_loss + args.dl_min_delta < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= args.dl_early_stopping:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        pred = model(X_test_tensor).detach().cpu().numpy().reshape(-1)
    return pred


def make_ddop_estimator(model_key: str, seed: int):
    from ddop.newsvendor import (
        DeepLearningNewsvendor,
        DecisionTreeWeightedNewsvendor,
        GaussianWeightedNewsvendor,
        KNeighborsWeightedNewsvendor,
        LinearRegressionNewsvendor,
        RandomForestWeightedNewsvendor,
        SampleAverageApproximationNewsvendor,
    )

    if model_key == "saa":
        return SampleAverageApproximationNewsvendor()
    if model_key == "lr":
        return LinearRegressionNewsvendor()
    if model_key == "dtw":
        return DecisionTreeWeightedNewsvendor(random_state=seed)
    if model_key == "rfw":
        return RandomForestWeightedNewsvendor(random_state=seed)
    if model_key == "knnw":
        return KNeighborsWeightedNewsvendor()
    if model_key == "kw":
        return GaussianWeightedNewsvendor()
    if model_key == "dl":
        return DeepLearningNewsvendor(random_state=seed)
    raise ValueError(f"Unsupported ddop model: {model_key}")


def ddop_set_params(estimator, params: dict, cu: float, co: float):
    params = dict(params)
    params["cu"] = cu
    params["co"] = co
    try:
        valid = set(estimator.get_params(deep=True))
        params = {key: value for key, value in params.items() if key in valid}
    except Exception:
        pass
    return estimator.set_params(**params)


def fit_predict_ddop(
    model_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cu: float,
    co: float,
    params: dict,
    seed: int,
) -> np.ndarray:
    estimator = make_ddop_estimator(model_key, seed)
    estimator = ddop_set_params(estimator, params, cu, co)
    try:
        estimator.fit(X=X_train, y=y_train)
    except TypeError:
        if model_key == "saa":
            estimator.fit(y_train)
        else:
            estimator.fit(X_train, y_train)

    if model_key == "saa":
        try:
            pred = estimator.predict(X_test.shape[0])
        except Exception:
            pred = estimator.predict(X_test)
    else:
        pred = estimator.predict(X_test)
    return np.asarray(pred, dtype=float).reshape(-1)


def fit_predict_scaled(
    model_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    sl: float,
    cu: float,
    co: float,
    args: argparse.Namespace,
    params: dict,
    seed: int,
    engine: str,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    if model_key == "saa":
        pred = empirical_quantile(y_train, sl)
        return np.full(len(X_test), pred, dtype=float)
    if model_key == "erm":
        return fit_erm_model("erm", y_train, X_train, X_test, cu, co)
    if engine == "ddop":
        return fit_predict_ddop(model_key, X_train, y_train, X_test, cu, co, params, seed)
    if model_key == "lr":
        return fit_predict_lr_local(X_train, y_train, X_test, sl, args, params)
    if model_key == "dtw":
        return fit_predict_dtw_local(X_train, y_train, X_test, sl, args, params, seed)
    if model_key == "rfw":
        return fit_predict_rfw_local(X_train, y_train, X_test, sl, args, params, seed)
    if model_key == "knnw":
        return fit_predict_knnw_local(X_train, y_train, X_test, sl, args, params)
    if model_key == "kw":
        return fit_predict_kw_local(X_train, y_train, X_test, sl, args, params)
    if model_key == "dl":
        return fit_predict_dl_local(
            X_train,
            y_train,
            X_test,
            sl,
            args,
            params,
            seed,
            X_val=X_val,
            y_val=y_val,
        )
    raise ValueError(f"Unsupported paper model: {model_key}")


def cv_score_params(
    model_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cu: float,
    co: float,
    sl: float,
    args: argparse.Namespace,
    params: dict,
    seed: int,
    engine: str,
) -> float:
    n_splits = min(args.cv_folds, len(y_train))
    if n_splits < 2:
        return float("-inf")

    losses = []
    cv = KFold(n_splits=n_splits, shuffle=False)
    for fold_idx, (fit_idx, val_idx) in enumerate(cv.split(X_train)):
        pred = fit_predict_scaled(
            model_key,
            X_train[fit_idx],
            y_train[fit_idx],
            X_train[val_idx],
            sl,
            cu,
            co,
            args,
            params,
            seed + fold_idx,
            engine,
        )
        losses.append(pandas_loss(cu, co)(pd.Series(y_train[val_idx]), pd.Series(pred)))
    return -float(np.mean(losses))


def select_params(
    model_key: str,
    data,
    cu: float,
    co: float,
    sl: float,
    args: argparse.Namespace,
    seed: int,
    engine: str,
) -> dict:
    params = default_params(model_key, data.n_features, args)
    if not args.tune or model_key in {"saa", "erm", "lr"}:
        return params

    candidates = paper_grid(model_key, data.n_features, args.grid_profile)
    if args.max_grid_candidates is not None:
        candidates = candidates[: args.max_grid_candidates]
    if not candidates:
        return params

    y_train = data.y_train_model.reshape(-1)
    best_score = float("-inf")
    best_params = candidates[0]
    for candidate_idx, candidate in enumerate(candidates):
        candidate = clean_params(candidate)
        candidate.update(
            {
                key: value
                for key, value in params.items()
                if key in {"min_samples_leaf"} and key not in candidate
            }
        )
        try:
            score = cv_score_params(
                model_key,
                data.X_train_model,
                y_train,
                cu,
                co,
                sl,
                args,
                candidate,
                seed + candidate_idx * 1000,
                engine,
            )
        except Exception as exc:
            print(f"[warn] tuning failed model={model_key}, params={candidate}: {exc}")
            score = float("-inf")
        if score > best_score:
            best_score = score
            best_params = candidate
    return clean_params(best_params)


def make_result_rows(
    dataset: str,
    feature_combo: list[str],
    group,
    model_key: str,
    data,
    predictions: list[np.ndarray],
    params_by_sl: list[dict],
    errors_by_sl: list[str | None],
    saa_costs: list[float],
    elapsed_by_sl: list[float],
    engine: str,
    metric2_alphalist: Iterable[float],
    metric2_betalist: Iterable[float],
) -> list[dict]:
    rows = []
    for idx, (cu, co) in enumerate(COST_PAIRS):
        error = errors_by_sl[idx]
        if error:
            metric1 = float("nan")
            metric2 = float("nan")
        else:
            metric1 = round(
                pandas_loss(cu, co)(
                    pd.Series(data.y_test_raw),
                    pd.Series(predictions[idx]),
                ),
                4,
            )
            metric2 = round(
                metric2_series_loss(
                    data.y_test_raw,
                    predictions[idx],
                    metric2_alphalist,
                    metric2_betalist,
                ),
                4,
            )
        rows.append(
            {
                "dataset": dataset,
                "feature combi": str(feature_combo),
                "group": str(group),
                "model": MODEL_DISPLAY_NAMES[model_key],
                "cu": cu,
                "co": co,
                "sl": service_level(cu, co),
                "metric 1": metric1,
                "metric 2": metric2,
                "average costs": metric1,
                "saa average costs": round(saa_costs[idx], 4),
                "engine": engine if model_key != "saa" else "closed_form",
                "best params": params_to_json(params_by_sl[idx]),
                "elapsed seconds": round(elapsed_by_sl[idx], 3),
                "error": error,
            }
        )
    return rows


def predict_model_all_service_levels(
    model_key: str,
    data,
    args: argparse.Namespace,
    group_idx: int,
    engine: str,
) -> tuple[list[np.ndarray], list[dict], list[str | None], list[float]]:
    predictions = []
    params_by_sl = []
    errors_by_sl = []
    elapsed_by_sl = []
    y_train_scaled = data.y_train_model.reshape(-1)
    y_val_scaled = data.y_val_model.reshape(-1)

    for sl_idx, (cu, co) in enumerate(COST_PAIRS):
        sl = service_level(cu, co)
        seed = args.seed + group_idx * 100 + sl_idx
        start = time.time()
        params = {}
        error = None
        try:
            params = select_params(model_key, data, cu, co, sl, args, seed, engine)
            pred_scaled = fit_predict_scaled(
                model_key,
                data.X_train_model,
                y_train_scaled,
                data.X_test_model,
                sl,
                cu,
                co,
                args,
                params,
                seed,
                engine,
                X_val=data.X_val_model,
                y_val=y_val_scaled,
            )
            pred_raw = inverse_target(data, pred_scaled)
        except Exception as exc:
            error = str(exc)
            pred_raw = np.full_like(data.y_test_raw, np.nan, dtype=float)
            print(f"[warn] {model_key} failed for group={data.group}, sl={sl:.2f}: {error}")

        predictions.append(pred_raw)
        params_by_sl.append(params)
        errors_by_sl.append(error)
        elapsed_by_sl.append(time.time() - start)

    return predictions, params_by_sl, errors_by_sl, elapsed_by_sl


def write_summary(
    output_dir: Path,
    results: pd.DataFrame,
    reference: pd.DataFrame | None,
    dataset: str,
    feature_combo: list[str],
    metric2_cost_pairs: Iterable[tuple[float, float]],
    selected_models: Iterable[str],
    include_reference: bool,
    use_reference_metric2_proxy: bool,
) -> Path:
    summary_rows = [
        results.groupby(["model", "sl"], as_index=False)[["metric 1", "metric 2"]]
        .mean(numeric_only=True)
        .assign(source="paper_model_retest")
    ]

    if include_reference and reference is not None and not reference.empty:
        selected_display_names = {
            MODEL_DISPLAY_NAMES[model_key] for model_key in selected_models
        }
        reference_metrics = add_reference_metrics(
            reference,
            metric2_cost_pairs,
            use_metric2_proxy=use_reference_metric2_proxy,
        )
        reference_metrics = reference_metrics[
            reference_metrics["model"].isin(selected_display_names)
        ].copy()
        if not reference_metrics.empty:
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
    path = output_dir / "paper_models_summary_by_model_sl.csv"
    summary.to_csv(path, index=False)
    return path


def build_metadata(
    args: argparse.Namespace,
    dataset: str,
    d3_root: Path,
    feature_combo: list[str],
    groups: Iterable,
    engine: str,
    detail_path: Path,
    summary_path: Path,
    workbook_path: Path,
) -> dict:
    sl_list = service_levels_for_cost_pairs(METRIC2_COST_PAIRS)
    reference_metric2_note = (
        "d3 reference metric 2 is a proxy average of published average costs because raw predictions are unavailable."
        if args.reference_metric2_proxy
        else "d3 reference metric 2 is left NaN because raw predictions are unavailable; each retested row reuses its single prediction across all metric-2 cost pairs."
    )
    return {
        "dataset": dataset,
        "feature_combi": args.feature_combi,
        "feature_categories": feature_combo,
        "groups": [group if isinstance(group, str) else str(group) for group in groups],
        "models": args.models,
        "model_display_names": MODEL_DISPLAY_NAMES,
        "engine": engine,
        "ddop_available": ddop_is_available(),
        "tune": args.tune,
        "grid_profile": args.grid_profile if args.tune else None,
        "cost_pairs": COST_PAIRS,
        "metric2_cost_pairs": METRIC2_COST_PAIRS,
        "metric2_service_levels": sl_list,
        "d3_root": str(d3_root),
        "outputs": {
            "detail": str(detail_path),
            "summary": str(summary_path),
            "workbook": str(workbook_path),
        },
        "notes": [
            "This script evaluates SAA, ERM, and the paper models: LR, DTW, RFW, KNNW, KW, DL.",
            "metric 1 is pandas_loss(cu, co) for each row's service level.",
            "metric 2 is pandas_loss_series over all cost pairs, reusing the current row's single prediction.",
            "Each service-level row therefore has its own metric 2 value.",
            "Use --engine ddop after installing ddop to force the original d3group estimator classes.",
            "Use --tune --grid-profile paper to run the d3group-style parameter grids; this can be very slow.",
            reference_metric2_note,
        ],
    }


def run_single_dataset(args: argparse.Namespace, dataset: str, engine: str) -> dict:
    set_seed(args.seed)
    d3_root = Path(args.d3_root)
    output_dir = Path(args.output_dir) / dataset / args.feature_combi
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[config] dataset={dataset}, feature={args.feature_combi}, "
        f"models={args.models}, engine={engine}, tune={args.tune}"
    )
    X, y = load_dataset(d3_root, dataset, args.auto_download)
    reference = load_reference_results(d3_root, dataset, args.auto_download)
    X_features, feature_combo = build_feature_frame(X, dataset, args.feature_combi)

    grouped = X_features.groupby(["store", "item"])
    groups = list(grouped.groups.keys())
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    print(f"[data] X={X.shape}, y={y.shape}, groups selected={len(groups)}")

    selected_reference = reference_subset(reference, feature_combo, groups)
    metric2_alphalist, metric2_betalist = split_cost_pairs(METRIC2_COST_PAIRS)
    rows: list[dict] = []

    for group_idx, group in enumerate(groups):
        print(f"\n[group {group_idx + 1}/{len(groups)}] {group}")
        data = split_group_data(group, grouped.get_group(group), y, args.val_fraction)
        saa_predictions, _, _, _ = predict_model_all_service_levels("saa", data, args, group_idx, engine)
        saa_costs = [
            pandas_loss(cu, co)(pd.Series(data.y_test_raw), pd.Series(saa_predictions[idx]))
            for idx, (cu, co) in enumerate(COST_PAIRS)
        ]

        for model_key in args.models:
            predictions, params_by_sl, errors_by_sl, elapsed_by_sl = predict_model_all_service_levels(
                model_key,
                data,
                args,
                group_idx,
                engine,
            )
            rows.extend(
                make_result_rows(
                    dataset,
                    feature_combo,
                    group,
                    model_key,
                    data,
                    predictions,
                    params_by_sl,
                    errors_by_sl,
                    saa_costs,
                    elapsed_by_sl,
                    engine,
                    metric2_alphalist,
                    metric2_betalist,
                )
            )

    results = pd.DataFrame(rows)
    detail_path = output_dir / "paper_models_results_detail.csv"
    results.to_csv(detail_path, index=False)
    summary_path = write_summary(
        output_dir,
        results,
        selected_reference,
        dataset,
        feature_combo,
        METRIC2_COST_PAIRS,
        args.models,
        args.include_reference,
        args.reference_metric2_proxy,
    )

    workbook_path = output_dir / f"{dataset}_{args.feature_combi}_paper_models_results.xlsx"
    metadata = build_metadata(
        args,
        dataset,
        d3_root,
        feature_combo,
        groups,
        engine,
        detail_path,
        summary_path,
        workbook_path,
    )
    metadata_path = output_dir / "paper_models_run_metadata.json"
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
        "dataset": dataset,
        "detail": str(detail_path),
        "summary": str(summary_path),
        "metadata": str(metadata_path),
        "workbook": str(workbook_path),
    }


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
        default=parse_models(",".join(PAPER_MODELS)),
        help=f"Comma-separated subset of {','.join(PAPER_MODELS)}.",
    )
    parser.add_argument(
        "--max-groups",
        type=parse_max_groups,
        default=None,
        help="Number of (store, item) groups to evaluate, or 'all'. Default: all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Validation slice from the 75%% train window for local DL early stopping. Default: 0.",
    )
    parser.add_argument("--output-dir", default="analysis_outputs/d3_paper_models")
    parser.add_argument(
        "--engine",
        choices=["auto", "ddop", "local"],
        default="auto",
        help="auto uses ddop when installed, otherwise local implementations.",
    )
    parser.add_argument(
        "--include-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include d3 published reference metric 1 rows in summary.",
    )
    parser.add_argument(
        "--reference-metric2-proxy",
        action="store_true",
        help="Fill d3 reference metric 2 with the old non-comparable proxy average.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run CV parameter selection. Off by default because full paper grids are expensive.",
    )
    parser.add_argument(
        "--grid-profile",
        choices=["small", "paper"],
        default="small",
        help="Grid used when --tune is enabled. paper mirrors the d3group grid.",
    )
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--max-grid-candidates", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)

    parser.add_argument("--lr-alpha", type=float, default=0.0)
    parser.add_argument("--lr-solver", default="highs")
    parser.add_argument("--lr-fallback-estimators", type=int, default=100)

    parser.add_argument("--tree-max-depth", type=parse_optional_int, default=None)
    parser.add_argument("--tree-min-samples-split", type=int, default=2)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=1)

    parser.add_argument("--rf-max-depth", type=parse_optional_int, default=None)
    parser.add_argument("--rf-min-samples-split", type=int, default=2)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-n-estimators", type=int, default=100)

    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--knn-weighting", choices=["uniform", "distance"], default="uniform")

    parser.add_argument(
        "--kw-bandwidth",
        type=float,
        default=-1.0,
        help="Gaussian kernel bandwidth. <=0 estimates a median-distance bandwidth.",
    )
    parser.add_argument("--bandwidth-sample", type=int, default=512)
    parser.add_argument("--kernel-chunk-size", type=int, default=256)

    parser.add_argument("--dl-epochs", type=int, default=100)
    parser.add_argument("--dl-neurons", type=parse_neurons, default=None)
    parser.add_argument("--dl-learning-rate", type=float, default=1e-3)
    parser.add_argument("--dl-weight-decay", type=float, default=0.0)
    parser.add_argument("--dl-early-stopping", type=int, default=10)
    parser.add_argument("--dl-min-delta", type=float, default=1e-5)
    parser.add_argument("--dl-clip-grad", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="Force torch device, e.g. cpu or cuda.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.val_fraction < 1:
        parser.error("--val-fraction must be in [0, 1).")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2.")
    if args.knn_k < 1:
        parser.error("--knn-k must be positive.")

    engine = resolve_engine(args)
    datasets = selected_datasets(args.dataset)
    print(f"[run] datasets={datasets}, engine={engine}")
    outputs = [run_single_dataset(args, dataset, engine) for dataset in datasets]

    print("\n[all done]")
    for output in outputs:
        print(f"{output['dataset']}: {output['workbook']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
