"""Shared synthetic-data protocols for spline RSETO sensitivity tests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from benchmark_literature_gaussian_rare_event_syn import (
    LiteratureSeparatedRareGaussianDGP,
)
from benchmark_izbicki_2026_bimodal_newsvendor import IzbickiBimodalFullDGP
from model.gendfl_spline import GenDFLSplineNewsvendor
from model.rseto_ipa_spline import RSETOIPASplineNewsvendor
from model.spline_qfr import SplineQFRNewsvendor
from synthetic_fixed_dgp import (
    make_toy_mixture_parameters,
    makettoy_multi_exp,
)


DEFAULT_RANDOM_STATES = [82, 15, 4, 95, 36, 32, 29, 18, 14, 87]
DEFAULT_SINGLE_COSTS = [
    [4, -4],
    [9, -5],
    [3, -3],
    [1, -8],
    [6, -2],
    [5, -1],
    [9, -6],
    [2, -9],
    [1, -6],
    [6, -9],
]

TRAIN_SAMPLE_SEED_OFFSET = 10_000
TEST_SAMPLE_SEED_OFFSET = 20_000
VAN_HAVRE_COST = [199.0, -1.0]
IZBICKI_BIMODAL_COST = [19.0, -1.0]


@dataclass
class GLRAlignedData:
    train_raw: np.ndarray
    validation_raw: np.ndarray
    test_raw: np.ndarray
    train_scaled: np.ndarray
    validation_scaled: np.ndarray
    test_scaled: np.ndarray
    target_scaler: StandardScaler
    alignment: dict

    @property
    def train_context(self):
        return self.train_scaled[:, :-1]

    @property
    def validation_context(self):
        return self.validation_scaled[:, :-1]

    @property
    def test_context(self):
        return self.test_scaled[:, :-1]

    @property
    def test_demand(self):
        return self.test_raw[:, -1]


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested="auto"):
    requested = str(requested).lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_cost_protocol():
    random.seed(42)
    random_states = random.sample(range(1, 101), 10)
    random.seed(128)
    single_costs = [
        [random.randint(1, 10), random.randint(-10, -1)]
        for _ in range(10)
    ]
    test_costs = [[] for _ in range(10)]
    for fold in range(10):
        for index in range(10):
            np.random.seed(index * 10 + fold)
            target_quantile = np.random.uniform(0.1, 0.9)
            cost_under = target_quantile * 10.0
            cost_over_signed = -(10.0 - cost_under)
            test_costs[fold].append([cost_under, cost_over_signed])
    if random_states != DEFAULT_RANDOM_STATES or single_costs != DEFAULT_SINGLE_COSTS:
        raise RuntimeError("The reconstructed GLR seed/cost protocol has changed.")
    return random_states, single_costs, test_costs


def array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def state_dict_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_fixed_dgp_data(
    *,
    dim=4,
    fold=0,
    walmart_path="Walmart.csv",
    glr_path="GLR_lr2.py",
):
    random_states, single_costs, test_costs = make_cost_protocol()
    random_state = random_states[int(fold)]
    walmart_rows = pd.read_csv(walmart_path).shape[0]
    num_samples = walmart_rows * 2

    parameters = make_toy_mixture_parameters(
        int(dim),
        random_state,
        num_exps=5,
    )
    train_sample_random_state = random_state + TRAIN_SAMPLE_SEED_OFFSET
    test_sample_random_state = random_state + TEST_SAMPLE_SEED_OFFSET

    labelled_data, train_weights = makettoy_multi_exp(
        num_samples=num_samples,
        num_features=int(dim),
        random_state=random_state,
        num_exps=5,
        sample_random_state=train_sample_random_state,
        parameters=parameters,
    )
    data = labelled_data[:, :-1]
    train_raw, validation_raw = train_test_split(
        data,
        test_size=0.1,
        random_state=42,
    )
    labelled_test, test_weights = makettoy_multi_exp(
        int(train_raw.shape[0] / 2),
        int(dim),
        random_state,
        num_exps=5,
        sample_random_state=test_sample_random_state,
        parameters=parameters,
    )
    test_raw = labelled_test[:, :-1]

    distribution_alignment = {
        "same_weights": bool(np.array_equal(train_weights, test_weights)),
        "same_intercepts": True,
        "same_component_probabilities": True,
        "independent_sample_seeds": bool(
            train_sample_random_state != test_sample_random_state
        ),
    }
    if not all(distribution_alignment.values()):
        raise RuntimeError(
            "Train and test data do not share one fixed DGP: "
            f"{distribution_alignment}"
        )

    train_raw = np.asarray(train_raw, dtype=np.float32)
    validation_raw = np.asarray(validation_raw, dtype=np.float32)
    test_raw = np.asarray(test_raw, dtype=np.float32)
    combined_scaler = StandardScaler()
    target_scaler = StandardScaler()
    train_scaled = combined_scaler.fit_transform(train_raw).astype(np.float32)
    validation_scaled = combined_scaler.transform(validation_raw).astype(np.float32)
    test_scaled = combined_scaler.transform(test_raw).astype(np.float32)
    target_scaler.fit(train_raw[:, -1].reshape(-1, 1))
    if not np.allclose(combined_scaler.mean_[-1], target_scaler.mean_[0]):
        raise RuntimeError("Combined and target scaler means differ.")
    if not np.allclose(combined_scaler.scale_[-1], target_scaler.scale_[0]):
        raise RuntimeError("Combined and target scaler scales differ.")

    single_cost = single_costs[int(fold)]
    target_quantile = single_cost[0] / (single_cost[0] + abs(single_cost[1]))
    alignment = {
        "source": str(Path(__file__).resolve()),
        "legacy_cost_protocol_source": str(Path(glr_path).resolve()),
        "data_protocol": "fixed_parameter_conditional_mixture_v2",
        "walmart_rows": walmart_rows,
        "num_samples": num_samples,
        "num_exps": 5,
        "dim": int(dim),
        "fold": int(fold),
        "random_state": random_state,
        "parameter_random_state": random_state,
        "train_sample_random_state": train_sample_random_state,
        "test_sample_random_state": test_sample_random_state,
        "split_random_state": 42,
        "single_cost": single_cost,
        "target_quantile": target_quantile,
        "series_costs": test_costs[int(fold)],
        "shapes": {
            "train": list(train_raw.shape),
            "validation": list(validation_raw.shape),
            "test": list(test_raw.shape),
        },
        "sha256": {
            "train_raw": array_sha256(train_raw),
            "validation_raw": array_sha256(validation_raw),
            "test_raw": array_sha256(test_raw),
            "train_scaled": array_sha256(train_scaled),
            "validation_scaled": array_sha256(validation_scaled),
            "test_scaled": array_sha256(test_scaled),
        },
        "mixture_parameters": {
            "mean_x": parameters.mean_x.tolist(),
            "intercepts": parameters.intercepts.tolist(),
            "weights": parameters.weights.tolist(),
            "probabilities": parameters.probabilities.tolist(),
        },
        "distribution_alignment": distribution_alignment,
    }
    return GLRAlignedData(
        train_raw=train_raw,
        validation_raw=validation_raw,
        test_raw=test_raw,
        train_scaled=train_scaled,
        validation_scaled=validation_scaled,
        test_scaled=test_scaled,
        target_scaler=target_scaler,
        alignment=alignment,
    )


def build_van_havre_data(
    *,
    dim=4,
    fold=0,
    walmart_path="Walmart.csv",
    glr_path="GLR_lr2.py",
):
    """Build independent train/test samples from van Havre et al.'s Sim 4 GMM."""

    random_states, _, test_costs = make_cost_protocol()
    random_state = random_states[int(fold)]
    walmart_rows = pd.read_csv(walmart_path).shape[0]
    num_samples = walmart_rows * 2
    train_sample_random_state = random_state + TRAIN_SAMPLE_SEED_OFFSET
    test_sample_random_state = random_state + TEST_SAMPLE_SEED_OFFSET

    dgp = LiteratureSeparatedRareGaussianDGP(
        int(dim),
        seed=random_state,
        demand_scale=20.0,
        demand_shift=30.0,
        context_amplitude=25.0,
    )
    train_context, train_demand, train_component = dgp.sample(
        num_samples,
        seed=train_sample_random_state,
        return_component=True,
    )
    train_pool = np.column_stack((train_context, train_demand[:, 0]))
    train_indices, validation_indices = train_test_split(
        np.arange(num_samples),
        test_size=0.1,
        random_state=42,
    )
    train_raw = train_pool[train_indices]
    validation_raw = train_pool[validation_indices]

    test_context, test_demand, test_component = dgp.sample(
        int(train_raw.shape[0] / 2),
        seed=test_sample_random_state,
        return_component=True,
    )
    test_raw = np.column_stack((test_context, test_demand[:, 0]))

    train_raw = np.asarray(train_raw, dtype=np.float32)
    validation_raw = np.asarray(validation_raw, dtype=np.float32)
    test_raw = np.asarray(test_raw, dtype=np.float32)
    combined_scaler = StandardScaler()
    target_scaler = StandardScaler()
    train_scaled = combined_scaler.fit_transform(train_raw).astype(np.float32)
    validation_scaled = combined_scaler.transform(validation_raw).astype(np.float32)
    test_scaled = combined_scaler.transform(test_raw).astype(np.float32)
    target_scaler.fit(train_raw[:, -1].reshape(-1, 1))
    if not np.allclose(combined_scaler.mean_[-1], target_scaler.mean_[0]):
        raise RuntimeError("Combined and target scaler means differ.")
    if not np.allclose(combined_scaler.scale_[-1], target_scaler.scale_[0]):
        raise RuntimeError("Combined and target scaler scales differ.")

    target_quantile = VAN_HAVRE_COST[0] / (
        VAN_HAVRE_COST[0] + abs(VAN_HAVRE_COST[1])
    )
    generation_config = dgp.generation_config()
    distribution_alignment = {
        "same_weights": True,
        "same_intercepts": True,
        "same_component_probabilities": True,
        "same_context_vector": True,
        "independent_sample_seeds": bool(
            train_sample_random_state != test_sample_random_state
        ),
    }
    alignment = {
        "source": str(Path(__file__).resolve()),
        "legacy_cost_protocol_source": str(Path(glr_path).resolve()),
        "data_protocol": "literature_van_havre_2015_sim4_conditional_v1",
        "literature_source": "https://doi.org/10.1371/journal.pone.0131739",
        "walmart_rows": walmart_rows,
        "num_samples": num_samples,
        "num_exps": 3,
        "dim": int(dim),
        "fold": int(fold),
        "random_state": random_state,
        "parameter_random_state": random_state,
        "train_sample_random_state": train_sample_random_state,
        "test_sample_random_state": test_sample_random_state,
        "split_random_state": 42,
        "single_cost": VAN_HAVRE_COST.copy(),
        "target_quantile": target_quantile,
        "series_costs": test_costs[int(fold)],
        "shapes": {
            "train": list(train_raw.shape),
            "validation": list(validation_raw.shape),
            "test": list(test_raw.shape),
        },
        "sha256": {
            "train_raw": array_sha256(train_raw),
            "validation_raw": array_sha256(validation_raw),
            "test_raw": array_sha256(test_raw),
            "train_scaled": array_sha256(train_scaled),
            "validation_scaled": array_sha256(validation_scaled),
            "test_scaled": array_sha256(test_scaled),
        },
        "mixture_parameters": generation_config,
        "observed_rare_fraction": {
            "train": float(np.mean(train_component[train_indices])),
            "validation": float(np.mean(train_component[validation_indices])),
            "test": float(np.mean(test_component)),
        },
        "target_sign_check": {
            "train_minimum": float(train_raw[:, -1].min()),
            "validation_minimum": float(validation_raw[:, -1].min()),
            "test_minimum": float(test_raw[:, -1].min()),
            "all_strictly_positive": bool(
                np.all(train_raw[:, -1] > 0.0)
                and np.all(validation_raw[:, -1] > 0.0)
                and np.all(test_raw[:, -1] > 0.0)
            ),
        },
        "distribution_alignment": distribution_alignment,
    }
    if not all(distribution_alignment.values()):
        raise RuntimeError(
            "Train and test data do not share one van Havre DGP: "
            f"{distribution_alignment}"
        )
    return GLRAlignedData(
        train_raw=train_raw,
        validation_raw=validation_raw,
        test_raw=test_raw,
        train_scaled=train_scaled,
        validation_scaled=validation_scaled,
        test_scaled=test_scaled,
        target_scaler=target_scaler,
        alignment=alignment,
    )


def build_izbicki_bimodal_data(
    *,
    dim=4,
    fold=0,
    walmart_path="Walmart.csv",
    glr_path="GLR_lr2.py",
):
    """Build independent samples from the all-active conditional bimodal DGP."""

    if int(dim) < 2:
        raise ValueError("The Izbicki bimodal DGP requires dim >= 2.")
    random_states, _, test_costs = make_cost_protocol()
    random_state = random_states[int(fold)]
    walmart_rows = pd.read_csv(walmart_path).shape[0]
    num_samples = walmart_rows * 2
    train_sample_random_state = random_state + TRAIN_SAMPLE_SEED_OFFSET
    test_sample_random_state = random_state + TEST_SAMPLE_SEED_OFFSET

    dgp = IzbickiBimodalFullDGP(context_dim=int(dim))
    train_context, train_demand, train_component = dgp.sample(
        num_samples,
        seed=train_sample_random_state,
        return_component=True,
    )
    train_pool = np.column_stack((train_context, train_demand[:, 0]))
    train_indices, validation_indices = train_test_split(
        np.arange(num_samples),
        test_size=0.1,
        random_state=42,
    )
    train_raw = train_pool[train_indices]
    validation_raw = train_pool[validation_indices]

    test_context, test_demand, test_component = dgp.sample(
        int(train_raw.shape[0] / 2),
        seed=test_sample_random_state,
        return_component=True,
    )
    test_raw = np.column_stack((test_context, test_demand[:, 0]))

    train_raw = np.asarray(train_raw, dtype=np.float32)
    validation_raw = np.asarray(validation_raw, dtype=np.float32)
    test_raw = np.asarray(test_raw, dtype=np.float32)
    combined_scaler = StandardScaler()
    target_scaler = StandardScaler()
    train_scaled = combined_scaler.fit_transform(train_raw).astype(np.float32)
    validation_scaled = combined_scaler.transform(validation_raw).astype(np.float32)
    test_scaled = combined_scaler.transform(test_raw).astype(np.float32)
    target_scaler.fit(train_raw[:, -1].reshape(-1, 1))
    if not np.allclose(combined_scaler.mean_[-1], target_scaler.mean_[0]):
        raise RuntimeError("Combined and target scaler means differ.")
    if not np.allclose(combined_scaler.scale_[-1], target_scaler.scale_[0]):
        raise RuntimeError("Combined and target scaler scales differ.")

    target_quantile = IZBICKI_BIMODAL_COST[0] / (
        IZBICKI_BIMODAL_COST[0] + abs(IZBICKI_BIMODAL_COST[1])
    )
    generation_config = dgp.generation_config()
    distribution_alignment = {
        "same_conditional_parameter_formulas": True,
        "same_all_active_projection_definition": True,
        "independent_sample_seeds": bool(
            train_sample_random_state != test_sample_random_state
        ),
    }
    alignment = {
        "source": str(Path(__file__).resolve()),
        "legacy_cost_protocol_source": str(Path(glr_path).resolve()),
        "data_protocol": "izbicki_2026_bimodal_full_all_active_projection_v1",
        "public_repository": generation_config["public_repository"],
        "public_commit": generation_config["public_commit"],
        "public_source_file": generation_config["public_source_file"],
        "walmart_rows": walmart_rows,
        "num_samples": num_samples,
        "num_exps": 2,
        "dim": int(dim),
        "active_context_dimensions": int(dim),
        "nuisance_context_dimensions": 0,
        "fold": int(fold),
        "random_state": random_state,
        "train_sample_random_state": train_sample_random_state,
        "test_sample_random_state": test_sample_random_state,
        "split_random_state": 42,
        "single_cost": IZBICKI_BIMODAL_COST.copy(),
        "target_quantile": target_quantile,
        "series_costs": test_costs[int(fold)],
        "shapes": {
            "train": list(train_raw.shape),
            "validation": list(validation_raw.shape),
            "test": list(test_raw.shape),
        },
        "sha256": {
            "train_raw": array_sha256(train_raw),
            "validation_raw": array_sha256(validation_raw),
            "test_raw": array_sha256(test_raw),
            "train_scaled": array_sha256(train_scaled),
            "validation_scaled": array_sha256(validation_scaled),
            "test_scaled": array_sha256(test_scaled),
        },
        "mixture_parameters": generation_config,
        "observed_component1_fraction": {
            "train": float(np.mean(train_component[train_indices])),
            "validation": float(np.mean(train_component[validation_indices])),
            "test": float(np.mean(test_component)),
        },
        "distribution_alignment": distribution_alignment,
    }
    if not all(distribution_alignment.values()):
        raise RuntimeError(
            "Train and test data do not share one Izbicki bimodal DGP: "
            f"{distribution_alignment}"
        )
    return GLRAlignedData(
        train_raw=train_raw,
        validation_raw=validation_raw,
        test_raw=test_raw,
        train_scaled=train_scaled,
        validation_scaled=validation_scaled,
        test_scaled=test_scaled,
        target_scaler=target_scaler,
        alignment=alignment,
    )


def sensitivity_data_tag(args):
    data_type = getattr(args, "data_synthetic", "exp5")
    data_tags = {
        "exp5": "iid_exp5",
        "van-havre": "van_havre_sim4",
        "izbicki-bimodal": "izbicki_bimodal",
    }
    try:
        return data_tags[data_type]
    except KeyError as exc:
        raise ValueError(f"Unknown synthetic data type: {data_type!r}.") from exc


def build_sensitivity_data(args):
    builders = {
        "iid_exp5": build_fixed_dgp_data,
        "van_havre_sim4": build_van_havre_data,
        "izbicki_bimodal": build_izbicki_bimodal_data,
    }
    builder = builders[sensitivity_data_tag(args)]
    return builder(
        dim=args.dim,
        fold=args.fold,
        walmart_path=args.walmart_path,
        glr_path=args.glr_path,
    )


def build_glr_aligned_data(**kwargs):
    """Backward-compatible name for the corrected fixed-DGP data builder."""
    return build_fixed_dgp_data(**kwargs)


def make_loader(data, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    tensor = torch.as_tensor(data, dtype=torch.float32)
    return DataLoader(
        TensorDataset(tensor),
        batch_size=min(int(batch_size), len(tensor)),
        shuffle=bool(shuffle),
        generator=generator,
    )


def model_kwargs(data, args):
    single_cost = data.alignment["single_cost"]
    return {
        "targetdim": 1,
        "labeldim": int(args.dim),
        "latent": 1,
        "data_len": len(data.train_scaled),
        "epoch": int(args.epochs),
        "quantiles": float(data.alignment["target_quantile"]),
        "target_quantile": float(data.alignment["target_quantile"]),
        "cost_under": float(single_cost[0]),
        "cost_over": float(abs(single_cost[1])),
        "random_seed": int(data.alignment["random_state"]),
        "num_transforms": int(args.num_transforms),
        "num_bins": int(args.num_bins),
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "tail_bound": float(args.tail_bound),
        "tau_eps": float(args.tau_eps),
    }


def pointwise_newsvendor_cost(y_true, decision, cost_under, cost_over_signed):
    error = np.asarray(y_true).reshape(-1) - np.asarray(decision).reshape(-1)
    return np.where(
        error > 0,
        float(cost_under) * error,
        float(cost_over_signed) * error,
    )


def predict_exact_quantile(model, context, target_scaler, quantile, batch_size):
    device = next(model.parameters()).device
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(context), int(batch_size)):
            end = min(start + int(batch_size), len(context))
            context_batch = torch.as_tensor(
                context[start:end],
                dtype=torch.float32,
                device=device,
            )
            quantile_scaled = model.quantile(float(quantile), context_batch)[:, 0, :]
            predictions.append(quantile_scaled.cpu().numpy())
    prediction_scaled = np.vstack(predictions)
    return target_scaler.inverse_transform(prediction_scaled).reshape(-1)


def evaluate_metric1_metric2(model, data, batch_size):
    y_true = data.test_demand
    single_cost = data.alignment["single_cost"]
    validation_critical_prediction = predict_exact_quantile(
        model,
        data.validation_context,
        data.target_scaler,
        data.alignment["target_quantile"],
        batch_size,
    )
    metric1_prediction = predict_exact_quantile(
        model,
        data.test_context,
        data.target_scaler,
        data.alignment["target_quantile"],
        batch_size,
    )
    metric1_values = pointwise_newsvendor_cost(
        y_true,
        metric1_prediction,
        single_cost[0],
        single_cost[1],
    )

    metric2_predictions = []
    metric2_point_losses = []
    for cost_under, cost_over_signed in data.alignment["series_costs"]:
        quantile = cost_under / (cost_under + abs(cost_over_signed))
        prediction = predict_exact_quantile(
            model,
            data.test_context,
            data.target_scaler,
            quantile,
            batch_size,
        )
        metric2_predictions.append(prediction)
        metric2_point_losses.append(
            pointwise_newsvendor_cost(
                y_true,
                prediction,
                cost_under,
                cost_over_signed,
            )
        )
    metric2_predictions = np.asarray(metric2_predictions)
    metric2_point_losses = np.asarray(metric2_point_losses)
    return {
        "metric1": float(metric1_values.mean()),
        "metric2": float(metric2_point_losses.mean()),
        "validation_critical_prediction": validation_critical_prediction,
        "validation_negative_quantile_count": int(
            np.count_nonzero(validation_critical_prediction < 0.0)
        ),
        "validation_negative_quantile_rate": float(
            np.mean(validation_critical_prediction < 0.0)
        ),
        "validation_min_critical_quantile": float(
            validation_critical_prediction.min()
        ),
        "test_negative_quantile_count": int(
            np.count_nonzero(metric1_prediction < 0.0)
        ),
        "test_negative_quantile_rate": float(
            np.mean(metric1_prediction < 0.0)
        ),
        "test_min_critical_quantile": float(metric1_prediction.min()),
        "metric1_prediction": metric1_prediction,
        "metric1_point_loss": metric1_values,
        "metric2_predictions": metric2_predictions,
        "metric2_point_losses": metric2_point_losses,
        "metric2_by_cost": metric2_point_losses.mean(axis=1),
    }


def _history_to_json(history):
    result = {}
    for key, value in history.items():
        if isinstance(value, np.generic):
            result[key] = value.item()
        elif isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def _make_epoch_progress_callback(label, interval):
    """Create a timing-only callback that never participates in training math."""
    interval = int(interval)
    if interval <= 0:
        return None
    started_at = time.perf_counter()
    window_started_at = started_at
    previous_epoch = 0

    def report(
        *,
        epoch,
        total_epochs,
        train_value,
        validation_value,
        current_m=None,
    ):
        nonlocal window_started_at, previous_epoch
        completed_epochs = int(epoch) + 1
        if completed_epochs % interval != 0 and completed_epochs != int(total_epochs):
            return
        now = time.perf_counter()
        window_epochs = max(completed_epochs - previous_epoch, 1)
        window_seconds = now - window_started_at
        seconds_per_epoch = window_seconds / window_epochs
        fields = [
            "[epoch-progress]",
            label,
            f"epoch={completed_epochs}/{int(total_epochs)}",
            f"train={float(train_value):.6f}",
            f"validation={float(validation_value):.6f}",
        ]
        if current_m is not None:
            fields.append(f"current_m={int(current_m)}")
        fields.extend(
            [
                f"window_seconds={window_seconds:.2f}",
                f"seconds_per_epoch={seconds_per_epoch:.2f}",
                f"elapsed_seconds={now - started_at:.2f}",
            ]
        )
        print(" ".join(fields), flush=True)
        window_started_at = now
        previous_epoch = completed_epochs

    return report


def _progress_label(args, data, method_label):
    job_index = getattr(args, "job_index", None)
    job_count = getattr(args, "job_count", None)
    job = f"{job_index}/{job_count}" if job_index is not None else "single"
    return (
        f"job={job} data={sensitivity_data_tag(args)} "
        f"dim={int(args.dim)} fold={int(args.fold)} "
        f"seed={int(data.alignment['random_state'])} {method_label}"
    )


def _train_model(
    model,
    method,
    data,
    args,
    checkpoint_path,
    ipa_config=None,
    progress_label=None,
):
    train_loader = make_loader(
        data.train_scaled,
        args.batch_size,
        True,
        data.alignment["random_state"] + 1000,
    )
    validation_loader = make_loader(
        data.validation_scaled,
        args.batch_size,
        False,
        data.alignment["random_state"] + 1001,
    )
    common = {
        "num_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "early_stopping": args.early_stopping,
        "warmup_epochs": args.warmup_epochs,
        "min_delta_relative": args.min_delta_relative,
        "checkpoint_path": checkpoint_path,
    }
    fair_training = {
        "step_size_exponent": args.step_size_exponent,
        "training_seed": data.alignment["random_state"] + 2000,
        "parameter_box_lower": args.parameter_box_lower,
        "parameter_box_upper": args.parameter_box_upper,
        "stop_early": bool(args.use_early_stopping),
        "restore_best": bool(args.use_early_stopping),
    }
    progress_callback = _make_epoch_progress_callback(
        _progress_label(args, data, progress_label or f"method={method}"),
        getattr(args, "progress_interval", 0),
    )
    if method == "gendfl_spline":
        return model.train_gendfl_spline(
            train_loader,
            validation_loader,
            optimizer_name="projected_sgd",
            epoch_callback=progress_callback,
            **fair_training,
            **common,
        )
    if method == "spline_qfr":
        return model.train_spline_qfr(
            train_loader,
            validation_loader,
            num_tau=args.qfr_levels,
            validation_num_tau=args.validation_qfr_levels,
            optimizer_name="projected_sgd",
            epoch_callback=progress_callback,
            **fair_training,
            **common,
        )
    if method == "rseto_ipa_spline":
        if ipa_config is None:
            raise ValueError("ipa_config is required for RSETO-IPA.")
        return model.train_rseto_ipa_spline(
            train_loader,
            validation_loader,
            replications=int(ipa_config["replications"]),
            samples_per_replication=int(ipa_config["samples_per_replication"]),
            m_growth=float(args.m_growth),
            m_growth_exponent=float(args.m_growth_exponent),
            smoothing_mu=float(ipa_config["smoothing_mu"]),
            fidelity_weight=float(ipa_config["fidelity_weight"]),
            max_simulation_values=int(args.max_simulation_values),
            diagnostic_interval=int(args.diagnostic_interval),
            finite_check_interval=int(args.finite_check_interval),
            train_data_on_device=bool(args.train_data_on_device),
            epoch_callback=progress_callback,
            **fair_training,
            **common,
        )
    raise ValueError(f"Unknown method: {method}")


def _load_or_train_baseline(
    *,
    model,
    method,
    data,
    args,
    checkpoint_path,
    history_path,
    metadata_path,
):
    expected_metadata = {
        "method": method,
        "initial_checkpoint_sha256": state_dict_sha256(model),
        "train_sha256": data.alignment["sha256"]["train_scaled"],
        "validation_sha256": data.alignment["sha256"]["validation_scaled"],
        "random_state": data.alignment["random_state"],
        "single_cost": data.alignment["single_cost"],
        "target_quantile": data.alignment["target_quantile"],
        "epochs": int(args.epochs),
        "early_stopping": int(args.early_stopping),
        "use_early_stopping": bool(args.use_early_stopping),
        "warmup_epochs": int(args.warmup_epochs),
        "min_delta_relative": float(args.min_delta_relative),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "training_protocol": {
            "optimizer": "projected_sgd",
            "step_size_exponent": float(args.step_size_exponent),
            "parameter_box_lower": float(args.parameter_box_lower),
            "parameter_box_upper": float(args.parameter_box_upper),
            "stop_early": bool(args.use_early_stopping),
            "restore_best": bool(args.use_early_stopping),
            "epochs": int(args.epochs),
            "evaluation_samples": 0,
        },
        "method_settings": (
            {
                "training_objective": "conditional_nll_only",
            }
            if method == "gendfl_spline"
            else {
                "training_objective": "random_tau_integrated_pinball",
                "num_tau": int(args.qfr_levels),
                "validation_num_tau": int(args.validation_qfr_levels),
            }
        ),
        "architecture": {
            "num_transforms": int(args.num_transforms),
            "num_bins": int(args.num_bins),
            "hidden_dim": int(args.hidden_dim),
            "hidden_layers": int(args.hidden_layers),
        },
    }
    can_reuse = False
    if checkpoint_path.exists() and history_path.exists() and metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text())
        can_reuse = existing_metadata == expected_metadata and not args.force
    if can_reuse:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(resolve_device(args.device))
        return json.loads(history_path.read_text()), 0.0, True

    start = time.perf_counter()
    history = _train_model(model, method, data, args, checkpoint_path)
    if model._device().type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    torch.save(model.state_dict(), checkpoint_path)
    history_path.write_text(json.dumps(_history_to_json(history), indent=2, allow_nan=True))
    metadata_path.write_text(json.dumps(expected_metadata, indent=2))
    return history, elapsed, False


def _save_model_outputs(output_dir, method_id, metrics, history, model):
    prediction_path = output_dir / "predictions" / f"{method_id}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        validation_critical_prediction=metrics["validation_critical_prediction"],
        metric1_prediction=metrics["metric1_prediction"],
        metric1_point_loss=metrics["metric1_point_loss"],
        metric2_predictions=metrics["metric2_predictions"],
        metric2_point_losses=metrics["metric2_point_losses"],
        metric2_by_cost=metrics["metric2_by_cost"],
    )
    np.save(output_dir / f"{method_id}_metric1.npy", np.asarray([metrics["metric1"]]))
    np.save(output_dir / f"{method_id}_metric2.npy", np.asarray([metrics["metric2"]]))
    history_path = output_dir / "histories" / f"{method_id}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(_history_to_json(history), indent=2, allow_nan=True))
    checkpoint_path = output_dir / "checkpoints" / f"{method_id}.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return prediction_path, history_path, checkpoint_path


def _estimate_final_ipa_gradient_variance(model, data, args, ipa_config, history):
    """Run one deterministic post-training variance diagnostic on validation data."""
    batch_size = min(
        int(args.gradient_variance_batch_size),
        len(data.validation_scaled),
    )
    validation_batch = torch.as_tensor(
        data.validation_scaled[:batch_size],
        dtype=torch.float32,
        device=model._device(),
    )
    final_m = history.get("final_m")
    if final_m is None:
        final_m = ipa_config["samples_per_replication"]
    result = model.estimate_batch_ipa_gradient_variance(
        validation_batch[:, :-1],
        validation_batch[:, -1:],
        replications=int(ipa_config["replications"]),
        samples_per_replication=int(final_m),
        smoothing_mu=float(ipa_config["smoothing_mu"]),
        diagnostic_repeats=int(args.gradient_variance_repeats),
        max_simulation_values=int(args.max_simulation_values),
        seed=(
            int(data.alignment["random_state"])
            + int(args.gradient_variance_seed_offset)
        ),
    )
    ipa_weight = 1.0 - float(ipa_config["fidelity_weight"])
    result["weighted_ipa_gradient_variance_trace"] = (
        ipa_weight * ipa_weight * result["ipa_gradient_variance_trace"]
    )
    result["ipa_gradient_weight"] = ipa_weight
    # Explicit aliases used by the sensitivity result tables. The underlying
    # quantity is the trace of the covariance of the final batched IPA gradient.
    result["ipa_batched_gradient_variance"] = result[
        "ipa_gradient_variance_trace"
    ]
    result["weighted_ipa_batched_gradient_variance"] = result[
        "weighted_ipa_gradient_variance_trace"
    ]
    return result


def run_sensitivity(args, sweep_type, sweep_values):
    if args.epochs < 1 or args.early_stopping < 1 or args.batch_size < 1:
        raise ValueError("epochs, early_stopping, and batch_size must be positive.")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup_epochs must lie in [0, epochs).")
    if args.samples_per_replication < 1:
        raise ValueError("samples_per_replication must be positive.")
    if not 0.5 < float(args.step_size_exponent) <= 1.0:
        raise ValueError("step_size_exponent must lie in (0.5, 1].")
    if float(args.m_growth) <= 0.0 or float(args.m_growth_exponent) <= 0.0:
        raise ValueError("m_growth and m_growth_exponent must be positive.")
    if float(args.parameter_box_lower) >= float(args.parameter_box_upper):
        raise ValueError("parameter box lower bound must be below its upper bound.")
    if int(args.max_simulation_values) < 1:
        raise ValueError("max_simulation_values must be positive.")
    if min(int(args.diagnostic_interval), int(args.finite_check_interval)) < 1:
        raise ValueError("Diagnostic and finite-check intervals must be positive.")
    if int(args.gradient_variance_repeats) < 2:
        raise ValueError("gradient_variance_repeats must be at least 2.")
    if int(args.gradient_variance_batch_size) < 1:
        raise ValueError("gradient_variance_batch_size must be positive.")
    if int(getattr(args, "progress_interval", 0)) < 0:
        raise ValueError("progress_interval must be nonnegative.")
    if sweep_type == "simulation_num":
        if any(int(value) < 1 for value in sweep_values):
            raise ValueError("Every simulation count must be positive.")
        if not 0.0 <= float(args.fixed_lambda) <= 1.0:
            raise ValueError("fixed_lambda must lie in [0, 1].")
    elif sweep_type == "lambda":
        if any(not 0.0 <= float(value) <= 1.0 for value in sweep_values):
            raise ValueError("Every lambda must lie in [0, 1].")
        if int(args.fixed_replications) < 1:
            raise ValueError("fixed_replications must be positive.")
    elif sweep_type == "m":
        if any(int(value) < 1 for value in sweep_values):
            raise ValueError("Every initial within-replication sample count must be positive.")
        if int(args.fixed_replications) < 1:
            raise ValueError("fixed_replications must be positive.")
        if not 0.0 <= float(args.fixed_lambda) <= 1.0:
            raise ValueError("fixed_lambda must lie in [0, 1].")
    else:
        raise ValueError(f"Unknown sweep_type: {sweep_type}")

    print(
        "[training-config] "
        f"data={sensitivity_data_tag(args)} "
        f"epochs={args.epochs} "
        f"early_stopping={bool(args.use_early_stopping)} "
        f"patience={args.early_stopping} "
        f"mnum={args.samples_per_replication}",
        flush=True,
    )

    default_fixed_root = Path(
        "analysis_outputs/spline_sensitivity_iid_exp5_dim4_seed82"
    )
    data_tag = sensitivity_data_tag(args)
    if data_tag != "iid_exp5" and Path(args.output_root) == default_fixed_root:
        random_state = make_cost_protocol()[0][int(args.fold)]
        args.output_root = Path("analysis_outputs") / (
            f"spline_sensitivity_{data_tag}_dim{args.dim}_seed{random_state}"
        )

    output_dir = Path(args.output_root) / sweep_type
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = Path(args.output_root) / "shared_baselines"
    shared_dir.mkdir(parents=True, exist_ok=True)
    data = build_sensitivity_data(args)
    (output_dir / "data_alignment.json").write_text(
        json.dumps(data.alignment, indent=2)
    )
    np.savez_compressed(
        output_dir / "aligned_data.npz",
        train_raw=data.train_raw,
        validation_raw=data.validation_raw,
        test_raw=data.test_raw,
        train_scaled=data.train_scaled,
        validation_scaled=data.validation_scaled,
        test_scaled=data.test_scaled,
    )

    device = resolve_device(args.device)
    kwargs = model_kwargs(data, args)
    set_seed(data.alignment["random_state"])
    template = GenDFLSplineNewsvendor(**kwargs)
    initial_state = copy.deepcopy(template.state_dict())
    initial_checkpoint_hash = state_dict_sha256(template)
    parameter_count = sum(parameter.numel() for parameter in template.parameters())

    rows = []
    baseline_specs = [
        ("gendfl_spline", GenDFLSplineNewsvendor),
        ("spline_qfr", SplineQFRNewsvendor),
    ]
    for method, model_class in baseline_specs:
        set_seed(data.alignment["random_state"])
        model = model_class(**kwargs)
        model.load_state_dict(copy.deepcopy(initial_state), strict=True)
        model.to(device)
        checkpoint = shared_dir / f"{method}.pth"
        history_path = shared_dir / f"{method}_history.json"
        metadata_path = shared_dir / f"{method}_metadata.json"
        history, elapsed, reused = _load_or_train_baseline(
            model=model,
            method=method,
            data=data,
            args=args,
            checkpoint_path=checkpoint,
            history_path=history_path,
            metadata_path=metadata_path,
        )
        metrics = evaluate_metric1_metric2(model, data, args.evaluation_batch_size)
        paths = _save_model_outputs(output_dir, method, metrics, history, model)
        rows.append(
            {
                "method": method,
                "sweep_type": sweep_type,
                "sweep_value": np.nan,
                "metric1": metrics["metric1"],
                "metric2": metrics["metric2"],
                "validation_negative_quantile_count": metrics[
                    "validation_negative_quantile_count"
                ],
                "validation_negative_quantile_rate": metrics[
                    "validation_negative_quantile_rate"
                ],
                "validation_min_critical_quantile": metrics[
                    "validation_min_critical_quantile"
                ],
                "test_negative_quantile_count": metrics[
                    "test_negative_quantile_count"
                ],
                "test_negative_quantile_rate": metrics[
                    "test_negative_quantile_rate"
                ],
                "test_min_critical_quantile": metrics[
                    "test_min_critical_quantile"
                ],
                "replications": np.nan,
                "samples_per_replication": np.nan,
                "final_samples_per_replication": np.nan,
                "configured_epochs": int(args.epochs),
                "use_early_stopping": bool(args.use_early_stopping),
                "early_stopping_patience": int(args.early_stopping),
                "fidelity_weight": np.nan,
                "smoothing_mu": np.nan,
                "optimizer": history.get("optimizer"),
                "steps_ran": history.get("steps_ran"),
                "step_size_exponent": history.get("step_size_exponent"),
                "evaluation_samples": 0,
                "evaluation_observations": len(data.test_context),
                "metric2_cost_pairs": len(data.alignment["series_costs"]),
                "batch_size": int(args.batch_size),
                "initial_checkpoint_sha256": initial_checkpoint_hash,
                "epochs_ran": history["epochs_ran"],
                "best_epoch": history["best_epoch"],
                "best_val_newsvendor": history["best_val_newsvendor"],
                "elapsed_seconds": elapsed,
                "baseline_reused": reused,
                "training_reused": reused,
                "parameter_count": parameter_count,
                "prediction_path": str(paths[0]),
                "checkpoint_path": str(paths[2]),
            }
        )

    for value in sweep_values:
        if sweep_type == "simulation_num":
            ipa_config = {
                "replications": int(value),
                "samples_per_replication": int(args.samples_per_replication),
                "fidelity_weight": float(args.fixed_lambda),
                "smoothing_mu": float(args.smoothing_mu),
            }
            method_id = f"rseto_ipa_R{int(value)}"
        elif sweep_type == "lambda":
            ipa_config = {
                "replications": int(args.fixed_replications),
                "samples_per_replication": int(args.samples_per_replication),
                "fidelity_weight": float(value),
                "smoothing_mu": float(args.smoothing_mu),
            }
            method_id = f"rseto_ipa_lambda{float(value):.1f}"
        elif sweep_type == "m":
            ipa_config = {
                "replications": int(args.fixed_replications),
                "samples_per_replication": int(value),
                "fidelity_weight": float(args.fixed_lambda),
                "smoothing_mu": float(args.smoothing_mu),
            }
            method_id = f"rseto_ipa_m{int(value)}"
        else:
            raise ValueError(f"Unknown sweep_type: {sweep_type}")

        set_seed(data.alignment["random_state"])
        model = RSETOIPASplineNewsvendor(**kwargs)
        model.load_state_dict(copy.deepcopy(initial_state), strict=True)
        model.to(device)
        checkpoint = output_dir / "checkpoints" / f"{method_id}.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        history_cache = output_dir / "histories" / f"{method_id}.json"
        metadata_path = output_dir / "checkpoints" / f"{method_id}_metadata.json"
        sweep_metadata = {
            "method": method_id,
            "initial_checkpoint_sha256": state_dict_sha256(model),
            "train_sha256": data.alignment["sha256"]["train_scaled"],
            "validation_sha256": data.alignment["sha256"]["validation_scaled"],
            "random_state": data.alignment["random_state"],
            "single_cost": data.alignment["single_cost"],
            "epochs": int(args.epochs),
            "early_stopping": int(args.early_stopping),
            "use_early_stopping": bool(args.use_early_stopping),
            "warmup_epochs": int(args.warmup_epochs),
            "min_delta_relative": float(args.min_delta_relative),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "training_protocol": {
                "optimizer": "projected_sgd",
                "step_size_exponent": float(args.step_size_exponent),
                "m_growth": float(args.m_growth),
                "m_growth_exponent": float(args.m_growth_exponent),
                "parameter_box_lower": float(args.parameter_box_lower),
                "parameter_box_upper": float(args.parameter_box_upper),
                "max_simulation_values": int(args.max_simulation_values),
                "acceleration": "screen_and_replay",
                "diagnostic_interval": int(args.diagnostic_interval),
                "finite_check_interval": int(args.finite_check_interval),
                "train_data_on_device": bool(args.train_data_on_device),
                "stop_early": bool(args.use_early_stopping),
                "restore_best": bool(args.use_early_stopping),
                "epochs": int(args.epochs),
                "evaluation_samples": 0,
                "evaluation_observations": len(data.test_context),
                "metric2_cost_pairs": len(data.alignment["series_costs"]),
                "batch_size": int(args.batch_size),
                "initial_checkpoint_sha256": initial_checkpoint_hash,
            },
            "ipa_config": ipa_config,
            "architecture": {
                "num_transforms": int(args.num_transforms),
                "num_bins": int(args.num_bins),
                "hidden_dim": int(args.hidden_dim),
                "hidden_layers": int(args.hidden_layers),
            },
        }
        reused = False
        if checkpoint.exists() and history_cache.exists() and metadata_path.exists() and not args.force:
            reused = json.loads(metadata_path.read_text()) == sweep_metadata
        if reused:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.to(device)
            history = json.loads(history_cache.read_text())
            elapsed = 0.0
        else:
            equivalent_checkpoint = None
            equivalent_history = None
            equivalent_candidates = [
                (
                    "simulation_num",
                    f"rseto_ipa_R{ipa_config['replications']}",
                ),
                (
                    "lambda",
                    f"rseto_ipa_lambda{ipa_config['fidelity_weight']:.1f}",
                ),
                (
                    "m",
                    f"rseto_ipa_m{ipa_config['samples_per_replication']}",
                ),
            ]
            for candidate_sweep, equivalent_id in equivalent_candidates:
                if args.force or candidate_sweep == sweep_type:
                    continue
                equivalent_dir = Path(args.output_root) / candidate_sweep
                candidate_checkpoint = (
                    equivalent_dir / "checkpoints" / f"{equivalent_id}.pth"
                )
                candidate_history = equivalent_dir / "histories" / f"{equivalent_id}.json"
                candidate_metadata = (
                    equivalent_dir
                    / "checkpoints"
                    / f"{equivalent_id}_metadata.json"
                )
                if (
                    candidate_checkpoint.exists()
                    and candidate_history.exists()
                    and candidate_metadata.exists()
                ):
                    candidate_config = json.loads(candidate_metadata.read_text())
                    comparable_candidate = {
                        key: item
                        for key, item in candidate_config.items()
                        if key != "method"
                    }
                    comparable_target = {
                        key: item
                        for key, item in sweep_metadata.items()
                        if key != "method"
                    }
                    if comparable_candidate == comparable_target:
                        equivalent_checkpoint = candidate_checkpoint
                        equivalent_history = candidate_history
                        break

            if equivalent_checkpoint is not None:
                state = torch.load(
                    equivalent_checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(state)
                model.to(device)
                history = json.loads(equivalent_history.read_text())
                elapsed = 0.0
                reused = True
            else:
                start = time.perf_counter()
                history = _train_model(
                    model,
                    "rseto_ipa_spline",
                    data,
                    args,
                    checkpoint,
                    ipa_config=ipa_config,
                    progress_label=(
                        f"method={method_id} R={ipa_config['replications']} "
                        f"m0={ipa_config['samples_per_replication']} "
                        f"lambda={ipa_config['fidelity_weight']:.3g}"
                    ),
                )
                if device.type == "mps":
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - start
        gradient_variance = _estimate_final_ipa_gradient_variance(
            model,
            data,
            args,
            ipa_config,
            history,
        )
        history["final_ipa_gradient_variance"] = gradient_variance
        np.save(
            output_dir / f"{method_id}_ipa_batched_gradient_variance.npy",
            np.asarray([gradient_variance["ipa_batched_gradient_variance"]]),
        )
        np.save(
            output_dir
            / f"{method_id}_weighted_ipa_batched_gradient_variance.npy",
            np.asarray(
                [gradient_variance["weighted_ipa_batched_gradient_variance"]]
            ),
        )
        metrics = evaluate_metric1_metric2(model, data, args.evaluation_batch_size)
        paths = _save_model_outputs(output_dir, method_id, metrics, history, model)
        metadata_path.write_text(json.dumps(sweep_metadata, indent=2))
        rows.append(
            {
                "method": method_id,
                "sweep_type": sweep_type,
                "sweep_value": float(value),
                "metric1": metrics["metric1"],
                "metric2": metrics["metric2"],
                "validation_negative_quantile_count": metrics[
                    "validation_negative_quantile_count"
                ],
                "validation_negative_quantile_rate": metrics[
                    "validation_negative_quantile_rate"
                ],
                "validation_min_critical_quantile": metrics[
                    "validation_min_critical_quantile"
                ],
                "test_negative_quantile_count": metrics[
                    "test_negative_quantile_count"
                ],
                "test_negative_quantile_rate": metrics[
                    "test_negative_quantile_rate"
                ],
                "test_min_critical_quantile": metrics[
                    "test_min_critical_quantile"
                ],
                "replications": ipa_config["replications"],
                "samples_per_replication": ipa_config["samples_per_replication"],
                "final_samples_per_replication": history.get("final_m"),
                "configured_epochs": int(args.epochs),
                "use_early_stopping": bool(args.use_early_stopping),
                "early_stopping_patience": int(args.early_stopping),
                "fidelity_weight": ipa_config["fidelity_weight"],
                "smoothing_mu": ipa_config["smoothing_mu"],
                "optimizer": history.get("optimizer"),
                "steps_ran": history.get("steps_ran"),
                "step_size_exponent": history.get("step_size_exponent"),
                "evaluation_samples": 0,
                "evaluation_observations": len(data.test_context),
                "metric2_cost_pairs": len(data.alignment["series_costs"]),
                "batch_size": int(args.batch_size),
                "initial_checkpoint_sha256": initial_checkpoint_hash,
                "epochs_ran": history["epochs_ran"],
                "best_epoch": history["best_epoch"],
                "best_val_newsvendor": history["best_val_newsvendor"],
                "elapsed_seconds": elapsed,
                "baseline_reused": False,
                "training_reused": reused,
                "parameter_count": parameter_count,
                "prediction_path": str(paths[0]),
                "checkpoint_path": str(paths[2]),
                **gradient_variance,
            }
        )
        print(
            f"[{sweep_type}] value={value} metric1={metrics['metric1']:.6f} "
            f"metric2={metrics['metric2']:.6f} epochs={history['epochs_ran']} "
            f"ipa_grad_var={gradient_variance['ipa_gradient_variance_trace']:.6e}"
        )

    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "detail.csv", index=False)
    with pd.ExcelWriter(output_dir / "results.xlsx") as writer:
        detail.to_excel(writer, sheet_name="summary", index=False)
        gradient_variance_columns = [
            "method",
            "sweep_type",
            "sweep_value",
            "replications",
            "samples_per_replication",
            "final_samples_per_replication",
            "fidelity_weight",
            "ipa_gradient_weight",
            "ipa_batched_gradient_variance",
            "weighted_ipa_batched_gradient_variance",
            "ipa_gradient_variance_trace",
            "weighted_ipa_gradient_variance_trace",
            "ipa_gradient_variance_mean_per_parameter",
            "ipa_gradient_std_norm",
            "ipa_gradient_mean_norm",
            "ipa_gradient_relative_variance",
            "ipa_gradient_loss_mean",
            "ipa_gradient_loss_variance",
            "gradient_variance_repeats",
            "gradient_variance_batch_size",
            "gradient_variance_replications",
            "gradient_variance_samples_per_replication",
            "gradient_variance_seed",
            "gradient_parameter_count",
        ]
        detail.loc[
            detail["method"].str.startswith("rseto_ipa_"),
            gradient_variance_columns,
        ].to_excel(writer, sheet_name="ipa_gradient_variance", index=False)
        pd.DataFrame(
            data.alignment["series_costs"],
            columns=["cost_under", "cost_over_signed"],
        ).assign(
            target_quantile=lambda frame: frame["cost_under"]
            / (frame["cost_under"] + frame["cost_over_signed"].abs())
        ).to_excel(writer, sheet_name="metric2_costs", index=False)
    np.save(output_dir / "metric1.npy", detail["metric1"].to_numpy())
    np.save(output_dir / "metric2.npy", detail["metric2"].to_numpy())

    sweep_rows = detail[detail["sweep_value"].notna()].sort_values("sweep_value")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    for axis, metric in zip(axes, ["metric1", "metric2"]):
        axis.plot(
            sweep_rows["sweep_value"],
            sweep_rows[metric],
            marker="o",
            linewidth=2.0,
            label="RSETO-IPA-Spline",
        )
        for _, baseline in detail[detail["sweep_value"].isna()].iterrows():
            axis.axhline(
                baseline[metric],
                linestyle="--",
                linewidth=1.3,
                label=baseline["method"],
            )
        x_labels = {
            "simulation_num": "Replications R",
            "lambda": "NLL weight lambda",
            "m": "Initial samples per replication m0",
        }
        axis.set_xlabel(x_labels[sweep_type])
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(output_dir / f"{sweep_type}_metric1_metric2.png", bbox_inches="tight")
    plt.close(fig)

    configuration = vars(args).copy()
    configuration.update(
        sweep_type=sweep_type,
        sweep_values=[float(value) for value in sweep_values],
        device=str(device),
        parameter_count=parameter_count,
    )
    for key, value in list(configuration.items()):
        if isinstance(value, Path):
            configuration[key] = str(value)
    (output_dir / "config.json").write_text(json.dumps(configuration, indent=2))
    print("\n" + detail.to_string(index=False))
    print(f"\nSaved to {output_dir.resolve()}")
    return detail


def add_common_arguments(parser):
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_outputs/spline_sensitivity_iid_exp5_dim4_seed82"),
    )
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--walmart-path", type=Path, default=Path("Walmart.csv"))
    parser.add_argument("--glr-path", type=Path, default=Path("GLR_lr2.py"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-delta-relative", type=float, default=0.0)
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
    parser.add_argument("--qfr-levels", type=int, default=16)
    parser.add_argument("--validation-qfr-levels", type=int, default=99)
    parser.add_argument("--m-growth", type=float, default=1.0)
    parser.add_argument("--m-growth-exponent", type=float, default=0.25)
    parser.add_argument(
        "--max-simulation-values",
        type=int,
        default=1048576,
        help="Maximum BRm values per no-grad screening chunk; tuned for a 24GB RTX 4090.",
    )
    parser.add_argument("--diagnostic-interval", type=int, default=100)
    parser.add_argument("--finite-check-interval", type=int, default=100)
    parser.add_argument(
        "--gradient-variance-repeats",
        type=int,
        default=8,
        help="Independent post-training batch IPA gradients used to estimate variance.",
    )
    parser.add_argument(
        "--gradient-variance-batch-size",
        type=int,
        default=16,
        help="Fixed validation-batch size for the post-training gradient diagnostic.",
    )
    parser.add_argument(
        "--gradient-variance-seed-offset",
        type=int,
        default=30000,
        help="Offset added to the experiment seed for the variance diagnostic.",
    )
    parser.add_argument(
        "--train-data-on-device",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--smoothing-mu", type=float, default=0.05)
    return parser
