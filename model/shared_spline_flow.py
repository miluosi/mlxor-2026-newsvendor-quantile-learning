"""Shared one-dimensional conditional rational-quadratic spline flow."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)
from torch import Tensor


@dataclass(frozen=True)
class SplineFlowConfig:
    context_dim: int
    num_transforms: int = 4
    num_bins: int = 16
    hidden_dim: int = 64
    hidden_layers: int = 2
    tail_bound: float = 4.0
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3

    def __post_init__(self):
        if self.context_dim < 1:
            raise ValueError("context_dim must be positive.")
        if min(self.num_transforms, self.num_bins, self.hidden_dim, self.hidden_layers) < 1:
            raise ValueError("Spline architecture sizes must be positive.")
        if self.tail_bound <= 0:
            raise ValueError("tail_bound must be positive.")
        if self.num_bins * self.min_bin_width >= 1.0:
            raise ValueError("num_bins * min_bin_width must be less than one.")
        if self.num_bins * self.min_bin_height >= 1.0:
            raise ValueError("num_bins * min_bin_height must be less than one.")
        if self.min_derivative <= 0:
            raise ValueError("min_derivative must be positive.")


def softplus_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_layers: int,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(current_dim, hidden_dim), nn.Softplus()])
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class ConditionalRQSLayer1D(nn.Module):
    def __init__(self, config: SplineFlowConfig):
        super().__init__()
        self.config = config
        output_dim = 3 * config.num_bins - 1
        self.parameter_dim = output_dim
        self.parameter_net = softplus_mlp(
            input_dim=config.context_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
        )

    def encode_condition(self, context: Tensor):
        return self.parameter_net(context)

    def _spline_parameters(self, values: Tensor, raw_parameters: Tensor):
        if raw_parameters.ndim != 2 or raw_parameters.shape[0] != values.shape[0]:
            raise ValueError("Encoded condition must have shape [B, P].")
        if raw_parameters.shape[1] != self.parameter_dim:
            raise ValueError("Encoded condition has the wrong parameter dimension.")
        parameter_shape = (
            values.shape[0],
            *((1,) * (values.ndim - 2)),
            self.parameter_dim,
        )
        raw_parameters = raw_parameters.view(parameter_shape).expand(
            *values.shape[:-1],
            self.parameter_dim,
        )
        num_bins = self.config.num_bins
        widths = raw_parameters[..., :num_bins]
        heights = raw_parameters[..., num_bins : 2 * num_bins]
        derivatives = raw_parameters[..., 2 * num_bins :]
        return widths, heights, derivatives

    def forward_with_encoded(
        self,
        values: Tensor,
        encoded_condition: Tensor,
        *,
        inverse: bool,
    ):
        widths, heights, derivatives = self._spline_parameters(
            values,
            encoded_condition,
        )
        outputs, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=values.squeeze(-1),
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=bool(inverse),
            tails="linear",
            tail_bound=self.config.tail_bound,
            min_bin_width=self.config.min_bin_width,
            min_bin_height=self.config.min_bin_height,
            min_derivative=self.config.min_derivative,
        )
        return outputs.unsqueeze(-1), logabsdet

    def forward(self, values: Tensor, context: Tensor, *, inverse: bool):
        return self.forward_with_encoded(
            values,
            self.encode_condition(context),
            inverse=inverse,
        )


class SharedConditionalSplineFlow(nn.Module):
    """Monotone conditional transport shared by NLL, QFR, and RSETO-IPA."""

    def __init__(self, config: SplineFlowConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [ConditionalRQSLayer1D(config) for _ in range(config.num_transforms)]
        )

    def encode_condition(self, context: Tensor):
        if context.ndim != 2:
            raise ValueError("context must have shape [B, C].")
        return tuple(layer.encode_condition(context) for layer in self.layers)

    def _validate_encoded_values(self, encoded_condition, values: Tensor):
        if values.ndim < 2 or values.shape[-1] != 1:
            raise ValueError("values must have shape [B, ..., 1].")
        if len(encoded_condition) != len(self.layers):
            raise ValueError("Encoded condition must contain one tensor per spline layer.")
        if any(encoded.shape[0] != values.shape[0] for encoded in encoded_condition):
            raise ValueError("Encoded condition and values must share the batch dimension.")

    def base_to_data_from_encoded(self, z: Tensor, encoded_condition):
        self._validate_encoded_values(encoded_condition, z)
        values = z
        total_logabsdet = torch.zeros_like(values[..., 0])
        for layer, encoded in zip(self.layers, encoded_condition):
            values, logabsdet = layer.forward_with_encoded(
                values,
                encoded,
                inverse=False,
            )
            total_logabsdet = total_logabsdet + logabsdet
        return values, total_logabsdet

    def base_to_data(self, z: Tensor, context: Tensor):
        return self.base_to_data_from_encoded(z, self.encode_condition(context))

    def data_to_base_from_encoded(self, target: Tensor, encoded_condition):
        self._validate_encoded_values(encoded_condition, target)
        values = target
        total_logabsdet = torch.zeros_like(values[..., 0])
        for layer, encoded in zip(
            reversed(self.layers),
            reversed(encoded_condition),
        ):
            values, logabsdet = layer.forward_with_encoded(
                values,
                encoded,
                inverse=True,
            )
            total_logabsdet = total_logabsdet + logabsdet
        return values, total_logabsdet

    def data_to_base(self, target: Tensor, context: Tensor):
        return self.data_to_base_from_encoded(
            target,
            self.encode_condition(context),
        )

    def log_prob_from_encoded(self, target: Tensor, encoded_condition):
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        z, inverse_logabsdet = self.data_to_base_from_encoded(
            target,
            encoded_condition,
        )
        base_log_prob = -0.5 * z.squeeze(-1).pow(2) - 0.5 * math.log(2.0 * math.pi)
        return base_log_prob + inverse_logabsdet

    def log_prob(self, target: Tensor, context: Tensor):
        return self.log_prob_from_encoded(target, self.encode_condition(context))

    def sample_from_encoded_condition(self, encoded_condition, z: Tensor):
        samples, _ = self.base_to_data_from_encoded(z, encoded_condition)
        return samples

    def sample_from_base_noise(self, context: Tensor, z: Tensor):
        return self.sample_from_encoded_condition(self.encode_condition(context), z)

    def quantile(self, context: Tensor, tau: Tensor, tau_eps: float = 1e-5):
        tau_eps = float(tau_eps)
        if not 0.0 < tau_eps < 0.5:
            raise ValueError("tau_eps must lie strictly between zero and 0.5.")
        tau = tau.clamp(tau_eps, 1.0 - tau_eps)
        normal = torch.distributions.Normal(
            torch.zeros((), device=tau.device, dtype=tau.dtype),
            torch.ones((), device=tau.device, dtype=tau.dtype),
        )
        return self.sample_from_base_noise(context, normal.icdf(tau))

    def sample(
        self,
        num_samples: int,
        context: Tensor,
        generator: torch.Generator | None = None,
    ):
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        z = torch.randn(
            context.shape[0],
            num_samples,
            1,
            device=context.device,
            dtype=context.dtype,
            generator=generator,
        )
        return self.sample_from_base_noise(context, z)
