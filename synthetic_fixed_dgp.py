"""Fixed-DGP synthetic data used by the newsvendor experiments.

The distribution parameters are drawn once for each experiment setting. Train,
validation, and test observations then use independent sampling streams while
sharing the same context mean, component probabilities, slopes, and intercepts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyMixtureParameters:
    """Parameters of a conditional mixture of linear Gaussian demands."""

    mean_x: np.ndarray
    intercepts: np.ndarray
    weights: np.ndarray
    probabilities: np.ndarray
    noise_scale: float = 10.0


def make_toy_mixture_parameters(
    num_features: int,
    random_state: int,
    num_exps: int = 5,
    noise_scale: float = 10.0,
) -> ToyMixtureParameters:
    """Draw one DGP independently of the requested number of observations."""

    num_features = int(num_features)
    num_exps = int(num_exps)
    noise_scale = float(noise_scale)
    if min(num_features, num_exps) < 1:
        raise ValueError("num_features and num_exps must be positive.")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be nonnegative.")

    parameter_rng = np.random.RandomState(int(random_state))
    return ToyMixtureParameters(
        mean_x=parameter_rng.uniform(-50.0, 50.0, size=num_features),
        intercepts=parameter_rng.uniform(0.0, 250.0, size=num_exps),
        weights=parameter_rng.normal(0.0, 1.0, size=(num_exps, num_features)),
        probabilities=parameter_rng.dirichlet(np.ones(num_exps)),
        noise_scale=noise_scale,
    )


def _validate_parameters(
    parameters: ToyMixtureParameters,
    num_features: int,
    num_exps: int,
) -> np.ndarray:
    expected_shapes = {
        "mean_x": (num_features,),
        "intercepts": (num_exps,),
        "weights": (num_exps, num_features),
        "probabilities": (num_exps,),
    }
    for name, expected_shape in expected_shapes.items():
        if np.asarray(getattr(parameters, name)).shape != expected_shape:
            raise ValueError(f"parameters.{name} must have shape {expected_shape}.")

    probabilities = np.asarray(parameters.probabilities, dtype=float)
    if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("Mixture probabilities must be nonnegative and sum to one.")
    if float(parameters.noise_scale) < 0.0:
        raise ValueError("parameters.noise_scale must be nonnegative.")
    return probabilities


def makettoy_multi_exp(
    num_samples: int,
    num_features: int,
    random_state: int,
    num_exps: int = 5,
    *,
    sample_random_state: int | None = None,
    parameters: ToyMixtureParameters | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate iid observations from a fixed conditional mixture.

    For every observation, the function independently draws a context ``x_i``
    and a component ``z_i``. The same component selects both ``W[z_i]`` and
    ``b[z_i]`` before Gaussian observation noise is added:

    ``y_i = x_i.T @ W[z_i] + b[z_i] + epsilon_i``.

    ``random_state`` controls the DGP parameters. ``sample_random_state`` only
    controls observations, so sample size changes cannot alter ``W`` or ``b``.
    The returned columns are ``[x, y, component_label]``.
    """

    num_samples = int(num_samples)
    num_features = int(num_features)
    num_exps = int(num_exps)
    if min(num_samples, num_features, num_exps) < 1:
        raise ValueError("num_samples, num_features, and num_exps must be positive.")

    if parameters is None:
        parameters = make_toy_mixture_parameters(
            num_features=num_features,
            random_state=random_state,
            num_exps=num_exps,
        )
    probabilities = _validate_parameters(parameters, num_features, num_exps)

    sample_seed = (
        int(random_state)
        if sample_random_state is None
        else int(sample_random_state)
    )
    sample_rng = np.random.RandomState(sample_seed)
    context = sample_rng.normal(
        loc=np.asarray(parameters.mean_x),
        scale=1.0,
        size=(num_samples, num_features),
    )
    labels = sample_rng.choice(num_exps, size=num_samples, p=probabilities)
    selected_weights = np.asarray(parameters.weights)[labels]
    selected_intercepts = np.asarray(parameters.intercepts)[labels]
    demand = (
        np.einsum("ij,ij->i", context, selected_weights)
        + selected_intercepts
        + sample_rng.normal(0.0, parameters.noise_scale, size=num_samples)
    )

    combined = np.column_stack((context, demand, labels))
    combined = combined[sample_rng.permutation(num_samples)]
    return combined, np.asarray(parameters.weights).copy()
