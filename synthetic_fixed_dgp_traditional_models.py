"""Compare learned baselines and the Bayes oracle on synthetic newsvendor data.

All models receive only the context ``x``. One global ``(cu, co)`` pair sets the
training loss for a complete fold; costs are never appended to sample features.
Training and test observations are sampled independently from one shared
conditional mixture of Gaussians. Learned baselines reuse their fold-level
Metric-1 prediction under every Metric-2 cost pair. Oracle-GMM instead evaluates
the true conditional-mixture quantile separately for every cost pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.projected_sa import (
    project_parameter_box,
    projected_sgd_step,
    robbins_monro_step_size,
)
from model.shared_spline_flow import softplus_mlp
from benchmark_literature_gaussian_rare_event_syn import (
    LiteratureSeparatedRareGaussianDGP,
)
from synthetic_fixed_dgp import (
    ToyMixtureParameters,
    make_toy_mixture_parameters,
    makettoy_multi_exp,
)
from spline_sensitivity_common import VAN_HAVRE_COST, build_van_havre_data


MODEL_NAMES = ("erm", "lightgbm", "end_to_end", "oracle_gmm")
MODEL_ALIASES = {
    "benchmark": "end_to_end",
    "eto": "end_to_end",
    "oracle": "oracle_gmm",
    "bayes_oracle": "oracle_gmm",
    "true_dgp_oracle": "oracle_gmm",
}
TRAIN_SAMPLE_SEED_OFFSET = 10_000
TEST_SAMPLE_SEED_OFFSET = 20_000


def parse_int_list(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_models(raw: str) -> list[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    models = [MODEL_ALIASES.get(value, value) for value in values]
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown models {unknown}; valid models are {list(MODEL_NAMES)}"
        )
    return list(dict.fromkeys(models))


def normalize_cost_pairs(cost_pairs) -> np.ndarray:
    pairs = np.asarray(cost_pairs, dtype=np.float64)
    if pairs.ndim == 1:
        pairs = pairs.reshape(1, -1)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("cost pairs must have shape [n, 2]")
    pairs = pairs.copy()
    if np.any(pairs[:, 0] <= 0.0) or np.any(pairs[:, 1] == 0.0):
        raise ValueError("cu must be positive and co must be nonzero")
    pairs[:, 1] = -np.abs(pairs[:, 1])
    return pairs


def make_cost_protocol(folds: int, test_cost_count: int) -> tuple[list[int], np.ndarray, list[np.ndarray]]:
    if folds > 100:
        raise ValueError("folds cannot exceed 100 with the GLR seed protocol")
    seed_rng = random.Random(42)
    random_states = seed_rng.sample(range(1, 101), folds)

    cost_rng = random.Random(128)
    metric1_costs = normalize_cost_pairs(
        [(cost_rng.randint(1, 10), cost_rng.randint(-10, -1)) for _ in range(folds)]
    )
    metric2_costs = []
    for fold in range(folds):
        fold_costs = []
        for cost_index in range(test_cost_count):
            rng = np.random.RandomState(cost_index * 10 + fold)
            alpha = float(rng.uniform(0.1, 0.9))
            fold_costs.append((10.0 * alpha, -10.0 * (1.0 - alpha)))
        metric2_costs.append(normalize_cost_pairs(fold_costs))
    return random_states, metric1_costs, metric2_costs


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def exact_row_overlap(left: np.ndarray, right: np.ndarray) -> int:
    left_rows = {row.tobytes() for row in np.ascontiguousarray(left)}
    return sum(row.tobytes() in left_rows for row in np.ascontiguousarray(right))


def newsvendor_loss(
    demand: np.ndarray,
    prediction: np.ndarray,
    cost_pair: np.ndarray,
) -> np.ndarray:
    pair = normalize_cost_pairs(cost_pair)[0]
    demand = np.asarray(demand, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    return pair[0] * np.maximum(demand - prediction, 0.0) + abs(pair[1]) * np.maximum(
        prediction - demand, 0.0
    )


@dataclass
class FixedDGPFold:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    X_test_raw: np.ndarray
    y_test: np.ndarray
    x_scaler: StandardScaler | None
    y_scaler: StandardScaler
    parameters: ToyMixtureParameters | None
    metadata: dict


def build_fixed_dgp_fold(
    *,
    train_samples: int,
    test_samples: int | None,
    validation_size: float,
    dim: int,
    num_exps: int,
    random_state: int,
    split_seed: int,
) -> FixedDGPFold:
    parameters = make_toy_mixture_parameters(dim, random_state, num_exps)
    train_sample_seed = random_state + TRAIN_SAMPLE_SEED_OFFSET
    test_sample_seed = random_state + TEST_SAMPLE_SEED_OFFSET
    train_pool, train_weights = makettoy_multi_exp(
        train_samples,
        dim,
        random_state,
        num_exps,
        sample_random_state=train_sample_seed,
        parameters=parameters,
    )
    train_pool = np.asarray(train_pool[:, :-1], dtype=np.float32)
    train_raw, validation_raw = train_test_split(
        train_pool,
        test_size=validation_size,
        random_state=split_seed,
    )
    resolved_test_samples = (
        int(train_raw.shape[0] / 2) if test_samples is None else int(test_samples)
    )
    test_data, test_weights = makettoy_multi_exp(
        resolved_test_samples,
        dim,
        random_state,
        num_exps,
        sample_random_state=test_sample_seed,
        parameters=parameters,
    )
    test_raw = np.asarray(test_data[:, :-1], dtype=np.float32)

    overlap = exact_row_overlap(train_pool[:, :dim], test_raw[:, :dim])
    if overlap:
        raise RuntimeError(f"train/test leakage audit found {overlap} exact contexts")
    if not np.array_equal(train_weights, test_weights):
        raise RuntimeError("train and test do not share the same W")

    x_scaler = StandardScaler().fit(train_raw[:, :dim])
    y_scaler = StandardScaler().fit(train_raw[:, dim : dim + 1])

    def transform(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        context = x_scaler.transform(rows[:, :dim]).astype(np.float32)
        demand = y_scaler.transform(rows[:, dim : dim + 1]).reshape(-1)
        return context, demand.astype(np.float32)

    X_train, y_train = transform(train_raw)
    X_validation, y_validation = transform(validation_raw)
    X_test, _ = transform(test_raw)
    metadata = {
        "data_protocol": "fixed_parameter_conditional_mixture_v2",
        "dim": dim,
        "num_exps": num_exps,
        "parameter_random_state": random_state,
        "train_sample_random_state": train_sample_seed,
        "test_sample_random_state": test_sample_seed,
        "split_random_state": split_seed,
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_validation.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_test_exact_context_overlap": overlap,
        "same_weights": True,
        "same_intercepts": True,
        "same_component_probabilities": True,
        "hashes": {
            "train_raw": array_sha256(train_raw),
            "validation_raw": array_sha256(validation_raw),
            "test_raw": array_sha256(test_raw),
        },
        "mixture_parameters": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in asdict(parameters).items()
        },
    }
    return FixedDGPFold(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        X_test_raw=test_raw[:, :dim].astype(np.float64),
        y_test=test_raw[:, dim].astype(np.float64),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        parameters=parameters,
        metadata=metadata,
    )


def build_van_havre_fold(
    *,
    dim: int,
    fold: int,
    reference_data: str | Path,
    glr_path: str | Path,
) -> FixedDGPFold:
    """Adapt the shared GenDFL Van Havre protocol for traditional models."""
    aligned = build_van_havre_data(
        dim=int(dim),
        fold=int(fold),
        walmart_path=reference_data,
        glr_path=glr_path,
    )
    train_pool_context = np.concatenate(
        (aligned.train_raw[:, :-1], aligned.validation_raw[:, :-1]),
        axis=0,
    )
    overlap = exact_row_overlap(train_pool_context, aligned.test_raw[:, :-1])
    if overlap:
        raise RuntimeError(
            f"Van Havre train/test leakage audit found {overlap} exact contexts"
        )
    distribution = aligned.alignment["distribution_alignment"]
    metadata = dict(aligned.alignment)
    metadata.update(
        {
            "train_test_exact_context_overlap": overlap,
            "same_weights": bool(distribution["same_weights"]),
            "same_intercepts": bool(distribution["same_intercepts"]),
            "same_component_probabilities": bool(
                distribution["same_component_probabilities"]
            ),
        }
    )
    return FixedDGPFold(
        X_train=np.asarray(aligned.train_scaled[:, :-1], dtype=np.float32),
        y_train=np.asarray(aligned.train_scaled[:, -1], dtype=np.float32),
        X_validation=np.asarray(
            aligned.validation_scaled[:, :-1], dtype=np.float32
        ),
        y_validation=np.asarray(
            aligned.validation_scaled[:, -1], dtype=np.float32
        ),
        X_test=np.asarray(aligned.test_scaled[:, :-1], dtype=np.float32),
        X_test_raw=np.asarray(aligned.test_raw[:, :-1], dtype=np.float64),
        y_test=np.asarray(aligned.test_raw[:, -1], dtype=np.float64),
        x_scaler=None,
        y_scaler=aligned.target_scaler,
        parameters=None,
        metadata=metadata,
    )


class BayesConditionalMixtureOracle:
    """True-DGP conditional Gaussian-mixture quantile oracle."""

    def __init__(
        self,
        *,
        data: FixedDGPFold,
        cost_pair,
        bisection_iterations: int = 64,
        tail_standard_deviations: float = 12.0,
    ):
        pair = normalize_cost_pairs(cost_pair)[0]
        self.target_quantile = float(pair[0] / (pair[0] + abs(pair[1])))
        self.bisection_iterations = int(bisection_iterations)
        self.tail_standard_deviations = float(tail_standard_deviations)
        self.data_protocol = str(data.metadata["data_protocol"])
        self.parameters = data.parameters
        self.van_havre_dgp = None

        if self.parameters is not None:
            self.oracle_family = "fixed_parameter_conditional_mixture"
        elif self.data_protocol == "literature_van_havre_2015_sim4_conditional_v1":
            config = data.metadata["mixture_parameters"]
            self.oracle_family = "van_havre_sim4_conditional_mixture"
            self.van_havre_dgp = LiteratureSeparatedRareGaussianDGP(
                context_dim=int(data.metadata["dim"]),
                seed=int(data.metadata["parameter_random_state"]),
                demand_scale=float(config["demand_scale"]),
                demand_shift=float(config["demand_shift"]),
                context_amplitude=float(config["context_amplitude"]),
            )
        else:
            raise ValueError(
                "Oracle-GMM does not support data protocol "
                f"{self.data_protocol!r}."
            )
        if self.bisection_iterations < 1:
            raise ValueError("bisection_iterations must be positive")
        if self.tail_standard_deviations <= 0.0:
            raise ValueError("tail_standard_deviations must be positive")

    def fit(self, *_args, **_kwargs) -> "BayesConditionalMixtureOracle":
        """No-op: the oracle receives the true DGP rather than fitted parameters."""

        return self

    def conditional_parameters(
        self,
        context: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        context = np.asarray(context, dtype=np.float64)
        if context.ndim != 2:
            raise ValueError("context must have shape [n, d]")

        if self.parameters is not None:
            weights = np.broadcast_to(
                np.asarray(self.parameters.probabilities, dtype=np.float64)[None, :],
                (context.shape[0], len(self.parameters.probabilities)),
            )
            means = (
                context @ np.asarray(self.parameters.weights, dtype=np.float64).T
                + np.asarray(self.parameters.intercepts, dtype=np.float64)[None, :]
            )
            sigmas = np.full_like(means, float(self.parameters.noise_scale))
        else:
            weights, means, sigmas = self.van_havre_dgp.component_parameters(context)

        weights = np.asarray(weights, dtype=np.float64)
        means = np.asarray(means, dtype=np.float64)
        sigmas = np.asarray(sigmas, dtype=np.float64)
        if weights.shape != means.shape or sigmas.shape != means.shape:
            raise RuntimeError("Conditional mixture arrays must share shape [n, K].")
        if np.any(sigmas <= 0.0):
            raise ValueError("Oracle-GMM requires positive component standard deviations.")
        if not np.allclose(weights.sum(axis=1), 1.0):
            raise RuntimeError("Conditional mixture weights must sum to one.")
        return weights, means, sigmas

    def predict_quantiles(
        self,
        context: np.ndarray,
        quantiles,
    ) -> np.ndarray:
        """Return exact numerical mixture quantiles with shape [Q, n]."""

        quantiles = np.asarray(quantiles, dtype=np.float64).reshape(-1)
        if quantiles.size < 1 or np.any((quantiles <= 0.0) | (quantiles >= 1.0)):
            raise ValueError("quantiles must lie strictly between zero and one")
        weights, means, sigmas = self.conditional_parameters(context)
        lower = np.min(
            means - self.tail_standard_deviations * sigmas,
            axis=1,
        )[:, None]
        upper = np.max(
            means + self.tail_standard_deviations * sigmas,
            axis=1,
        )[:, None]
        lower = np.broadcast_to(lower, (len(means), len(quantiles))).copy()
        upper = np.broadcast_to(upper, (len(means), len(quantiles))).copy()

        for _ in range(self.bisection_iterations):
            midpoint = 0.5 * (lower + upper)
            standardized = (
                midpoint[:, :, None] - means[:, None, :]
            ) / sigmas[:, None, :]
            cdf = np.sum(weights[:, None, :] * ndtr(standardized), axis=2)
            below = cdf < quantiles[None, :]
            lower = np.where(below, midpoint, lower)
            upper = np.where(below, upper, midpoint)
        return (0.5 * (lower + upper)).T

    def predict(self, context: np.ndarray) -> np.ndarray:
        return self.predict_quantiles(context, [self.target_quantile])[0]

    def predict_for_costs(self, context: np.ndarray, cost_pairs) -> np.ndarray:
        pairs = normalize_cost_pairs(cost_pairs)
        quantiles = pairs[:, 0] / (pairs[:, 0] + np.abs(pairs[:, 1]))
        return self.predict_quantiles(context, quantiles)

    def configuration(self) -> dict:
        return {
            "model": "oracle_gmm",
            "oracle_family": self.oracle_family,
            "data_protocol": self.data_protocol,
            "target_quantile": self.target_quantile,
            "bisection_iterations": self.bisection_iterations,
            "tail_standard_deviations": self.tail_standard_deviations,
            "uses_true_mixture_parameters": True,
            "observes_latent_component": False,
            "observes_future_noise": False,
        }


class CostAwareERM:
    """Linear ERM with one global newsvendor cost pair, solved by Gurobi."""

    def __init__(self, cost_pair, output_flag: int = 0, threads: int = 0):
        pair = normalize_cost_pairs(cost_pair)[0]
        self.cost_under = float(pair[0])
        self.cost_over_signed = float(pair[1])
        self.output_flag = int(output_flag)
        self.threads = int(threads)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, features: np.ndarray, demand: np.ndarray) -> "CostAwareERM":
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise RuntimeError("gurobipy is required for the ERM model") from exc

        features = np.asarray(features, dtype=np.float64)
        demand = np.asarray(demand, dtype=np.float64).reshape(-1)
        model = gp.Model("fixed_dgp_erm")
        model.Params.OutputFlag = self.output_flag
        if self.threads > 0:
            model.Params.Threads = self.threads
        weights = model.addMVar(features.shape[1], lb=-GRB.INFINITY, name="weights")
        intercept = model.addVar(lb=-GRB.INFINITY, name="intercept")
        loss = model.addMVar(features.shape[0], lb=0.0, name="loss")
        prediction = features @ weights + intercept
        residual = demand - prediction
        model.addConstr(loss >= residual * self.cost_under, name="underage")
        # co is signed and negative, so co * (y - q) is overage cost.
        model.addConstr(loss >= residual * self.cost_over_signed, name="overage")
        model.setObjective(loss.sum() / features.shape[0], GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"ERM solve failed with Gurobi status {model.Status}")
        self.coef_ = np.asarray(weights.X, dtype=np.float64)
        self.intercept_ = float(intercept.X)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict")
        return np.asarray(features, dtype=np.float64) @ self.coef_ + self.intercept_


class CostAwareLightGBM:
    """LightGBM quantile regression with one global newsvendor cost pair."""

    def __init__(
        self,
        *,
        n_estimators: int,
        learning_rate: float,
        num_leaves: int,
        min_child_samples: int,
        cost_pair,
        random_state: int,
        n_jobs: int,
    ):
        pair = normalize_cost_pairs(cost_pair)[0]
        self.target_quantile = float(pair[0] / (pair[0] + abs(pair[1])))
        self.kwargs = {
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "min_child_samples": int(min_child_samples),
            "random_state": int(random_state),
            "n_jobs": int(n_jobs),
        }
        self.model = None

    def fit(self, features: np.ndarray, demand: np.ndarray) -> "CostAwareLightGBM":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("lightgbm is required for the LightGBM model") from exc

        features = np.asarray(features, dtype=np.float64)
        demand = np.asarray(demand, dtype=np.float64).reshape(-1)
        self.model = LGBMRegressor(
            objective="quantile",
            alpha=self.target_quantile,
            verbosity=-1,
            **self.kwargs,
        )
        self.model.fit(features, demand)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit must be called before predict")
        return np.asarray(self.model.predict(features), dtype=np.float64)


class CostAwareNetwork(nn.Module):
    """Direct-decision network using the GenDFL conditioner architecture."""

    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        self.network = softplus_mlp(
            input_dim=int(input_dim),
            output_dim=1,
            hidden_dim=int(hidden_dim),
            hidden_layers=int(hidden_layers),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class CostAwareEndToEnd:
    """Neural decision model trained with one global unsmoothed NV loss."""

    def __init__(
        self,
        *,
        input_dim: int,
        cost_pair,
        hidden_dim: int,
        hidden_layers: int,
        learning_rate: float,
        step_size_exponent: float,
        parameter_box_lower: float,
        parameter_box_upper: float,
        batch_size: int,
        epochs: int,
        random_state: int,
        device: str,
    ):
        pair = normalize_cost_pairs(cost_pair)[0]
        self.cost_under = float(pair[0])
        self.cost_over = float(abs(pair[1]))
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.learning_rate = float(learning_rate)
        self.step_size_exponent = float(step_size_exponent)
        self.parameter_box_lower = float(parameter_box_lower)
        self.parameter_box_upper = float(parameter_box_upper)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.random_state = int(random_state)
        self.device = torch.device(device)
        set_seed(self.random_state)
        self.model = CostAwareNetwork(
            self.input_dim,
            self.hidden_dim,
            self.hidden_layers,
        ).to(self.device)
        self.history_: list[dict] = []
        self.best_epoch_: int | None = None
        self.steps_ran_: int = 0

    def _loss(
        self,
        prediction: torch.Tensor,
        demand: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.cost_under * torch.relu(demand - prediction)
            + self.cost_over * torch.relu(prediction - demand)
        ).mean()

    def fit(
        self,
        features: np.ndarray,
        demand: np.ndarray,
        *,
        validation_features: np.ndarray,
        validation_demand: np.ndarray,
    ) -> "CostAwareEndToEnd":
        train_context = torch.as_tensor(features, dtype=torch.float32)
        train_demand = torch.as_tensor(
            demand.reshape(-1, 1), dtype=torch.float32
        )
        validation_loader = DataLoader(
            TensorDataset(
                torch.as_tensor(validation_features, dtype=torch.float32),
                torch.as_tensor(
                    validation_demand.reshape(-1, 1), dtype=torch.float32
                ),
            ),
            batch_size=min(self.batch_size, len(validation_features)),
            shuffle=False,
        )
        parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        robbins_monro_step_size(
            0, self.learning_rate, self.step_size_exponent
        )
        project_parameter_box(
            parameters,
            self.parameter_box_lower,
            self.parameter_box_upper,
        )
        sample_count = len(train_context)
        batch_size = min(self.batch_size, sample_count)
        steps_per_epoch = int(np.ceil(sample_count / batch_size))
        batch_rng = torch.Generator(device="cpu").manual_seed(
            self.random_state + 2000
        )
        global_step = 0
        best_loss = float("inf")

        for epoch in range(self.epochs):
            self.model.train()
            train_total = 0.0
            train_count = 0
            first_step_size = None
            last_step_size = None
            for _ in range(steps_per_epoch):
                indices = torch.randperm(sample_count, generator=batch_rng)[:batch_size]
                batch_features = train_context.index_select(0, indices).to(self.device)
                batch_demand = train_demand.index_select(0, indices).to(self.device)
                prediction = self.model(batch_features)
                loss = self._loss(prediction, batch_demand)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite end-to-end loss encountered")
                for parameter in parameters:
                    parameter.grad = None
                loss.backward()
                if any(
                    parameter.grad is None
                    or not torch.isfinite(parameter.grad).all()
                    for parameter in parameters
                ):
                    raise FloatingPointError("Non-finite end-to-end gradient encountered")
                step_size = robbins_monro_step_size(
                    global_step,
                    self.learning_rate,
                    self.step_size_exponent,
                )
                projected_sgd_step(
                    parameters,
                    step_size,
                    self.parameter_box_lower,
                    self.parameter_box_upper,
                )
                if first_step_size is None:
                    first_step_size = step_size
                last_step_size = step_size
                global_step += 1
                train_total += float(loss.detach()) * batch_features.shape[0]
                train_count += batch_features.shape[0]

            self.model.eval()
            validation_total = 0.0
            validation_count = 0
            with torch.no_grad():
                for batch_features, batch_demand in validation_loader:
                    batch_features = batch_features.to(self.device)
                    batch_demand = batch_demand.to(self.device)
                    loss = self._loss(self.model(batch_features), batch_demand)
                    validation_total += float(loss) * batch_features.shape[0]
                    validation_count += batch_features.shape[0]
            train_loss = train_total / max(train_count, 1)
            validation_loss = validation_total / max(validation_count, 1)
            if validation_loss < best_loss:
                best_loss = validation_loss
                self.best_epoch_ = epoch
            self.history_.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "step_size_first": first_step_size,
                    "step_size_last": last_step_size,
                }
            )
        self.steps_ran_ = global_step
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        predictions = []
        loader = DataLoader(
            torch.as_tensor(features, dtype=torch.float32),
            batch_size=self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            for batch in loader:
                predictions.append(self.model(batch.to(self.device)).cpu().numpy())
        return np.concatenate(predictions, axis=0).reshape(-1).astype(np.float64)


def make_model(
    name: str,
    args: argparse.Namespace,
    input_dim: int,
    seed: int,
    cost_pair: np.ndarray,
    data: FixedDGPFold | None = None,
):
    if name == "erm":
        return CostAwareERM(cost_pair, args.gurobi_output, args.gurobi_threads)
    if name == "lightgbm":
        return CostAwareLightGBM(
            n_estimators=args.lgb_n_estimators,
            learning_rate=args.lgb_learning_rate,
            num_leaves=args.lgb_num_leaves,
            min_child_samples=args.lgb_min_child_samples,
            cost_pair=cost_pair,
            random_state=seed,
            n_jobs=args.n_jobs,
        )
    if name == "end_to_end":
        return CostAwareEndToEnd(
            input_dim=input_dim,
            cost_pair=cost_pair,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            learning_rate=args.learning_rate,
            step_size_exponent=args.step_size_exponent,
            parameter_box_lower=args.parameter_box_lower,
            parameter_box_upper=args.parameter_box_upper,
            batch_size=args.batch_size,
            epochs=args.epochs,
            random_state=seed,
            device=args.device,
        )
    if name == "oracle_gmm":
        if data is None:
            raise ValueError("data is required to construct Oracle-GMM")
        return BayesConditionalMixtureOracle(data=data, cost_pair=cost_pair)
    raise ValueError(name)


def predict_raw_demand(
    model,
    context: np.ndarray,
    y_scaler: StandardScaler,
) -> np.ndarray:
    scaled_prediction = model.predict(np.asarray(context, dtype=np.float64))
    return y_scaler.inverse_transform(scaled_prediction.reshape(-1, 1)).reshape(-1)


def evaluate_model(
    model,
    data: FixedDGPFold,
    metric1_cost: np.ndarray,
    metric2_costs: np.ndarray,
) -> dict:
    if isinstance(model, BayesConditionalMixtureOracle):
        all_costs = np.concatenate(
            (
                normalize_cost_pairs(metric1_cost),
                normalize_cost_pairs(metric2_costs),
            ),
            axis=0,
        )
        all_predictions = model.predict_for_costs(data.X_test_raw, all_costs)
        metric1_prediction = all_predictions[0]
        metric2_predictions = all_predictions[1:]
        metric1_point_loss = newsvendor_loss(
            data.y_test, metric1_prediction, metric1_cost
        )
        metric2_point_losses = []
        per_cost_rows = []
        for cost_index, (pair, prediction) in enumerate(
            zip(normalize_cost_pairs(metric2_costs), metric2_predictions)
        ):
            point_loss = newsvendor_loss(data.y_test, prediction, pair)
            metric2_point_losses.append(point_loss)
            per_cost_rows.append(
                {
                    "cost_index": cost_index,
                    "cu": pair[0],
                    "co": pair[1],
                    "target_quantile": pair[0] / (pair[0] + abs(pair[1])),
                    "average_cost": float(point_loss.mean()),
                }
            )
        return {
            "metric1": float(metric1_point_loss.mean()),
            "metric2": float(np.mean(metric2_point_losses)),
            "metric1_prediction": metric1_prediction,
            "metric1_point_loss": metric1_point_loss,
            "metric2_predictions": metric2_predictions,
            "metric2_point_losses": np.asarray(metric2_point_losses),
            "metric2_by_cost": pd.DataFrame(per_cost_rows),
        }

    metric1_prediction = predict_raw_demand(model, data.X_test, data.y_scaler)
    metric1_point_loss = newsvendor_loss(
        data.y_test, metric1_prediction, metric1_cost
    )
    metric2_point_losses = []
    per_cost_rows = []
    for cost_index, pair in enumerate(normalize_cost_pairs(metric2_costs)):
        point_loss = newsvendor_loss(data.y_test, metric1_prediction, pair)
        metric2_point_losses.append(point_loss)
        per_cost_rows.append(
            {
                "cost_index": cost_index,
                "cu": pair[0],
                "co": pair[1],
                "target_quantile": pair[0] / (pair[0] + abs(pair[1])),
                "average_cost": float(point_loss.mean()),
            }
        )
    return {
        "metric1": float(metric1_point_loss.mean()),
        "metric2": float(np.mean(metric2_point_losses)),
        "metric1_prediction": metric1_prediction,
        "metric1_point_loss": metric1_point_loss,
        "metric2_predictions": np.repeat(
            metric1_prediction.reshape(1, -1), len(metric2_point_losses), axis=0
        ),
        "metric2_point_losses": np.asarray(metric2_point_losses),
        "metric2_by_cost": pd.DataFrame(per_cost_rows),
    }


def save_model_artifacts(
    output_dir: Path,
    stem: str,
    model,
    data: FixedDGPFold,
    evaluation: dict,
) -> tuple[str, str]:
    prediction_dir = output_dir / "predictions"
    model_dir = output_dir / "models"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{stem}.npz"
    np.savez_compressed(
        prediction_path,
        y_true=data.y_test,
        metric1_prediction=evaluation["metric1_prediction"],
        metric1_point_loss=evaluation["metric1_point_loss"],
        metric2_predictions=evaluation["metric2_predictions"],
        metric2_point_losses=evaluation["metric2_point_losses"],
    )

    model_path = ""
    if isinstance(model, CostAwareERM):
        path = model_dir / f"{stem}.npz"
        np.savez(path, coef=model.coef_, intercept=model.intercept_)
        model_path = str(path)
    elif isinstance(model, CostAwareLightGBM):
        path = model_dir / f"{stem}.txt"
        model.model.booster_.save_model(str(path))
        model_path = str(path)
    elif isinstance(model, CostAwareEndToEnd):
        path = model_dir / f"{stem}.pth"
        torch.save(model.model.state_dict(), path)
        model_path = str(path)
        pd.DataFrame(model.history_).to_csv(
            model_dir / f"{stem}_history.csv", index=False
        )
    elif isinstance(model, BayesConditionalMixtureOracle):
        path = model_dir / f"{stem}.json"
        path.write_text(
            json.dumps(model.configuration(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        model_path = str(path)
    return str(prediction_path), model_path


def aggregate_results(detail: pd.DataFrame) -> pd.DataFrame:
    valid = detail[detail["error"].eq("")].copy()
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(
            ["data_synthetic", "num_exps", "dim", "model"],
            as_index=False,
        )
        .agg(
            folds=("fold", "nunique"),
            metric1_mean=("metric1", "mean"),
            metric2_mean=("metric2", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
        )
        .sort_values(["num_exps", "dim", "metric2_mean", "model"])
        .reset_index(drop=True)
    )


def write_npy_results(output_dir: Path, detail: pd.DataFrame) -> list[Path]:
    """Write fold-ordered Metric 1/2 arrays in the legacy analysis format."""

    npy_dir = output_dir / "npy_results"
    npy_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    valid = detail[detail["error"].fillna("").eq("")]
    for (num_exps, dim, model_name), group in valid.groupby(
        ["num_exps", "dim", "model"], sort=True
    ):
        ordered = group.sort_values("fold")
        stem = (
            f"fixed_dgp_{model_name}_exp{int(num_exps)}_dim{int(dim)}"
        )
        metric1_path = npy_dir / f"{stem}.npy"
        metric2_path = npy_dir / f"{stem}_series.npy"
        np.save(metric1_path, ordered["metric1"].to_numpy(dtype=float))
        np.save(metric2_path, ordered["metric2"].to_numpy(dtype=float))
        saved_paths.extend((metric1_path, metric2_path))
    return saved_paths


def merge_result_frames(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    """Replace matching runs while preserving unrelated existing results."""

    if existing.empty:
        return new.copy()
    if new.empty:
        return existing.copy()
    if "data_synthetic" in new.columns:
        default_data_type = str(new["data_synthetic"].iloc[0])
        if "data_synthetic" not in existing.columns:
            existing = existing.copy()
            existing.insert(0, "data_synthetic", default_data_type)
        else:
            existing = existing.copy()
            existing["data_synthetic"] = existing["data_synthetic"].replace(
                "", default_data_type
            ).fillna(default_data_type)
    missing = [
        column
        for column in key_columns
        if column not in existing.columns or column not in new.columns
    ]
    if missing:
        raise ValueError(f"Cannot append results; missing key columns: {missing}")
    new_keys = pd.MultiIndex.from_frame(new[key_columns])
    existing_keys = pd.MultiIndex.from_frame(existing[key_columns])
    retained = existing.loc[~existing_keys.isin(new_keys)]
    return pd.concat([retained, new], ignore_index=True, sort=False)


def write_results(
    output_dir: Path,
    rows: list[dict],
    cost_frames: list[pd.DataFrame],
    *,
    append: bool = False,
) -> None:
    detail = pd.DataFrame(rows)
    per_cost = (
        pd.concat(cost_frames, ignore_index=True) if cost_frames else pd.DataFrame()
    )
    detail_path = output_dir / "fixed_dgp_results_detail.csv"
    per_cost_path = output_dir / "fixed_dgp_metric2_by_cost.csv"
    if append and detail_path.exists():
        existing_detail = pd.read_csv(detail_path, keep_default_na=False)
        detail = merge_result_frames(
            existing_detail,
            detail,
            ["data_synthetic", "num_exps", "dim", "fold", "random_state", "model"],
        )
    if append and per_cost_path.exists():
        existing_per_cost = pd.read_csv(per_cost_path, keep_default_na=False)
        per_cost = merge_result_frames(
            existing_per_cost,
            per_cost,
            [
                "data_synthetic",
                "num_exps",
                "dim",
                "fold",
                "random_state",
                "model",
                "cost_index",
            ],
        )
    detail = detail.sort_values(
        ["num_exps", "dim", "fold", "model"], ignore_index=True
    )
    if not per_cost.empty:
        per_cost = per_cost.sort_values(
            ["num_exps", "dim", "fold", "model", "cost_index"],
            ignore_index=True,
        )
    detail.to_csv(detail_path, index=False)
    per_cost.to_csv(per_cost_path, index=False)
    aggregate_results(detail).to_csv(
        output_dir / "fixed_dgp_results_summary.csv", index=False
    )
    write_npy_results(output_dir, detail)


def run_one_setting(
    args: argparse.Namespace,
    *,
    num_exps: int,
    dim: int,
    fold: int,
    random_state: int,
    metric1_cost: np.ndarray,
    metric2_costs: np.ndarray,
) -> tuple[list[dict], list[pd.DataFrame]]:
    if args.data_synthetic == "van-havre":
        data = build_van_havre_fold(
            dim=dim,
            fold=fold,
            reference_data=args.reference_data,
            glr_path=args.glr_path,
        )
        setting_stem = (
            f"van_havre_dim{dim}_fold{fold:02d}_seed{random_state}"
        )
    else:
        data = build_fixed_dgp_fold(
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            validation_size=args.validation_size,
            dim=dim,
            num_exps=num_exps,
            random_state=random_state,
            split_seed=args.split_seed,
        )
        setting_stem = (
            f"exp{num_exps}_dim{dim}_fold{fold:02d}_seed{random_state}"
        )
    metadata_dir = args.output_dir / "data_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / f"{setting_stem}.json").write_text(
        json.dumps(data.metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[setting] exp={num_exps}, dim={dim}, fold={fold}, seed={random_state}, "
        f"train={data.X_train.shape[0]}, validation={data.X_validation.shape[0]}, "
        f"test={data.X_test.shape[0]}, input_dim={data.X_train.shape[1]}, "
        f"global_cost=({metric1_cost[0]:.4f}, {metric1_cost[1]:.4f})"
    )

    rows = []
    cost_frames = []
    for model_name in args.models:
        started = time.perf_counter()
        metric1 = float("nan")
        metric2 = float("nan")
        prediction_path = ""
        model_path = ""
        error = ""
        best_epoch = np.nan
        optimizer_name = ""
        steps_ran = np.nan
        parameter_count = np.nan
        model = None
        try:
            model = make_model(
                model_name,
                args,
                input_dim=data.X_train.shape[1],
                seed=random_state,
                cost_pair=metric1_cost,
                data=data,
            )
            if isinstance(model, CostAwareEndToEnd):
                optimizer_name = "projected_sgd"
                model.fit(
                    data.X_train,
                    data.y_train,
                    validation_features=data.X_validation,
                    validation_demand=data.y_validation,
                )
                best_epoch = model.best_epoch_
                steps_ran = model.steps_ran_
                parameter_count = sum(
                    parameter.numel() for parameter in model.model.parameters()
                )
            elif isinstance(model, CostAwareLightGBM):
                optimizer_name = "lightgbm_quantile"
                steps_ran = args.lgb_n_estimators
                model.fit(data.X_train, data.y_train)
            elif isinstance(model, BayesConditionalMixtureOracle):
                optimizer_name = "true_dgp_quantile_bisection"
                steps_ran = model.bisection_iterations
                model.fit()
            else:
                optimizer_name = "gurobi_exact"
                model.fit(data.X_train, data.y_train)
            evaluation = evaluate_model(
                model, data, metric1_cost, metric2_costs
            )
            metric1 = evaluation["metric1"]
            metric2 = evaluation["metric2"]
            stem = f"{setting_stem}_{model_name}"
            prediction_path, model_path = save_model_artifacts(
                args.output_dir, stem, model, data, evaluation
            )
            cost_frame = evaluation["metric2_by_cost"].copy()
            cost_frame.insert(0, "model", model_name)
            cost_frame.insert(0, "random_state", random_state)
            cost_frame.insert(0, "fold", fold)
            cost_frame.insert(0, "dim", dim)
            cost_frame.insert(0, "num_exps", num_exps)
            cost_frame.insert(0, "data_synthetic", args.data_synthetic)
            cost_frames.append(cost_frame)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[error] model={model_name}: {error}")
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "data_synthetic": args.data_synthetic,
                "num_exps": num_exps,
                "dim": dim,
                "fold": fold,
                "random_state": random_state,
                "model": model_name,
                "metric1": metric1,
                "metric2": metric2,
                "metric1_cu": float(metric1_cost[0]),
                "metric1_co": float(metric1_cost[1]),
                "training_cost_is_global": not isinstance(
                    model, BayesConditionalMixtureOracle
                ),
                "costs_are_model_features": False,
                "metric2_cost_pairs": int(metric2_costs.shape[0]),
                "metric2_reuses_metric1_prediction": not isinstance(
                    model, BayesConditionalMixtureOracle
                ),
                "configured_epochs": (
                    0 if isinstance(model, BayesConditionalMixtureOracle) else int(args.epochs)
                ),
                "epochs_ran": (
                    len(model.history_)
                    if isinstance(model, CostAwareEndToEnd)
                    else (0 if isinstance(model, BayesConditionalMixtureOracle) else np.nan)
                ),
                "use_early_stopping": False,
                "optimizer": optimizer_name,
                "steps_ran": steps_ran,
                "learning_rate": (
                    float(args.learning_rate)
                    if isinstance(model, CostAwareEndToEnd)
                    else np.nan
                ),
                "step_size_exponent": (
                    float(args.step_size_exponent)
                    if isinstance(model, CostAwareEndToEnd)
                    else np.nan
                ),
                "parameter_box_lower": (
                    float(args.parameter_box_lower)
                    if isinstance(model, CostAwareEndToEnd)
                    else np.nan
                ),
                "parameter_box_upper": (
                    float(args.parameter_box_upper)
                    if isinstance(model, CostAwareEndToEnd)
                    else np.nan
                ),
                "hidden_dim": (
                    int(args.hidden_dim)
                    if isinstance(model, CostAwareEndToEnd)
                    else np.nan
                ),
                "hidden_layers": (
                    int(args.hidden_layers)
                    if isinstance(model, CostAwareEndToEnd)
                    else np.nan
                ),
                "parameter_count": parameter_count,
                "train_rows": int(data.X_train.shape[0]),
                "validation_rows": int(data.X_validation.shape[0]),
                "test_rows": int(data.X_test.shape[0]),
                "data_protocol": data.metadata["data_protocol"],
                "train_test_exact_context_overlap": data.metadata[
                    "train_test_exact_context_overlap"
                ],
                "same_weights": data.metadata["same_weights"],
                "same_intercepts": data.metadata["same_intercepts"],
                "same_component_probabilities": data.metadata[
                    "same_component_probabilities"
                ],
                "best_epoch": best_epoch,
                "elapsed_seconds": elapsed,
                "prediction_path": prediction_path,
                "model_path": model_path,
                "error": error,
            }
        )
        print(
            f"[result] model={model_name}, metric1={metric1:.6f}, "
            f"metric2={metric2:.6f}, seconds={elapsed:.2f}"
        )
    return rows, cost_frames


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-synthetic",
        "--data_synthetic",
        dest="data_synthetic",
        choices=["exp5", "van-havre"],
        default="exp5",
        help="Synthetic DGP. Van Havre reuses the shared GenDFL data protocol.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/fixed_dgp_erm_lightgbm_end_to_end"),
    )
    parser.add_argument(
        "--append-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add or replace selected model rows without deleting existing models.",
    )
    parser.add_argument("--models", type=parse_models, default=list(MODEL_NAMES))
    parser.add_argument("--num-exps-list", type=parse_int_list, default=[5])
    parser.add_argument("--dims", type=parse_int_list, default=[4, 9, 14, 19, 24])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--reference-data", type=Path, default=Path("Walmart.csv"))
    parser.add_argument("--glr-path", type=Path, default=Path("GLR_lr2.py"))
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-cost-count", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--step-size-exponent", type=float, default=0.6)
    parser.add_argument("--parameter-box-lower", type=float, default=-10.0)
    parser.add_argument("--parameter-box-upper", type=float, default=10.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--lgb-n-estimators", type=int, default=50)
    parser.add_argument("--lgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgb-num-leaves", type=int, default=31)
    parser.add_argument("--lgb-min-child-samples", type=int, default=20)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="LightGBM worker count; one avoids local OpenMP/threadpool conflicts.",
    )
    parser.add_argument("--gurobi-output", type=int, choices=[0, 1], default=0)
    parser.add_argument("--gurobi-threads", type=int, default=0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.folds <= 100:
        raise ValueError("--folds must be between 1 and 100")
    if args.train_samples < 2:
        raise ValueError("--train-samples must be at least 2")
    if args.test_samples is not None and args.test_samples < 1:
        raise ValueError("--test-samples must be positive")
    if not 0.0 < args.validation_size < 1.0:
        raise ValueError("--validation-size must be between 0 and 1")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if min(args.hidden_dim, args.hidden_layers) < 1:
        raise ValueError("hidden dimension and layer count must be positive")
    if not 0.5 < args.step_size_exponent <= 1.0:
        raise ValueError("step-size exponent must lie in (0.5, 1]")
    if args.parameter_box_lower >= args.parameter_box_upper:
        raise ValueError("parameter box lower bound must be below its upper bound")
    if args.test_cost_count < 1:
        raise ValueError("--test-cost-count must be positive")
    if args.data_synthetic == "van-havre":
        if args.folds > 10:
            raise ValueError("Van Havre supports the ten aligned GLR folds only")
        if args.test_cost_count != 10:
            raise ValueError("Van Havre Metric 2 requires the ten aligned cost pairs")


def main() -> None:
    args = build_parser().parse_args()
    if args.train_samples is None:
        args.train_samples = 2 * pd.read_csv(args.reference_data).shape[0]
    if args.data_synthetic == "van-havre":
        args.num_exps_list = [3]
    args.device = resolve_device(args.device)
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config["reference_data"] = str(config["reference_data"])
    config["glr_path"] = str(config["glr_path"])
    config_name = "append_config.json" if args.append_results else "config.json"
    (args.output_dir / config_name).write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    random_states, metric1_costs, metric2_costs = make_cost_protocol(
        args.folds, args.test_cost_count
    )
    if args.data_synthetic == "van-havre":
        metric1_costs = normalize_cost_pairs(
            np.repeat(
                np.asarray(VAN_HAVRE_COST, dtype=np.float64).reshape(1, 2),
                args.folds,
                axis=0,
            )
        )
    pd.DataFrame(metric1_costs, columns=["cu", "co"]).assign(
        fold=np.arange(args.folds)
    )[["fold", "cu", "co"]].to_csv(
        args.output_dir / "metric1_costs.csv", index=False
    )
    pd.concat(
        [
            pd.DataFrame(costs, columns=["cu", "co"]).assign(fold=fold)
            for fold, costs in enumerate(metric2_costs)
        ],
        ignore_index=True,
    )[["fold", "cu", "co"]].to_csv(
        args.output_dir / "metric2_costs.csv", index=False
    )

    rows: list[dict] = []
    cost_frames: list[pd.DataFrame] = []
    total_settings = len(args.num_exps_list) * len(args.dims) * args.folds
    setting_index = 0
    for num_exps in args.num_exps_list:
        for dim in args.dims:
            for fold, random_state in enumerate(random_states):
                setting_index += 1
                print(f"\n[job {setting_index}/{total_settings}]")
                new_rows, new_cost_frames = run_one_setting(
                    args,
                    num_exps=num_exps,
                    dim=dim,
                    fold=fold,
                    random_state=random_state,
                    metric1_cost=metric1_costs[fold],
                    metric2_costs=metric2_costs[fold],
                )
                rows.extend(new_rows)
                cost_frames.extend(new_cost_frames)
                write_results(
                    args.output_dir,
                    rows,
                    cost_frames,
                    append=args.append_results,
                )

    detail = pd.DataFrame(rows)
    summary = aggregate_results(detail)
    print("\n[summary]")
    print(summary.to_string(index=False))
    print(f"[saved] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
