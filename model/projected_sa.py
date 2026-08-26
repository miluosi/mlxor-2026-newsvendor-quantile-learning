"""Shared projected stochastic-approximation utilities for spline experiments."""

from __future__ import annotations

import torch


def robbins_monro_step_size(iteration, gamma0, exponent):
    """Return gamma_k = gamma0 / (k + 1)^exponent."""
    iteration = int(iteration)
    gamma0 = float(gamma0)
    exponent = float(exponent)
    if iteration < 0 or gamma0 <= 0.0:
        raise ValueError("iteration must be nonnegative and gamma0 must be positive.")
    if not 0.5 < exponent <= 1.0:
        raise ValueError("Robbins-Monro exponent must lie in (0.5, 1].")
    return gamma0 / ((iteration + 1) ** exponent)


def training_tensors(train_loader, targetdim=1):
    """Extract CPU tensors for fresh uniform-without-replacement mini-batches."""
    dataset = getattr(train_loader, "dataset", None)
    tensors = getattr(dataset, "tensors", None)
    if tensors is None:
        raise TypeError(
            "Projected-SA training requires a TensorDataset so every iteration "
            "can draw a fresh uniform subset."
        )
    if len(tensors) == 2 and tensors[1].is_floating_point():
        condition, demand = tensors
    elif len(tensors) == 1:
        packed = tensors[0]
        targetdim = int(targetdim)
        condition = packed[:, :-targetdim]
        demand = packed[:, -targetdim:]
    else:
        raise ValueError(
            "Training TensorDataset must contain (context, demand) or one packed tensor."
        )
    if condition.ndim != 2 or demand.shape[0] != condition.shape[0]:
        raise ValueError("Invalid training context/demand shapes.")
    if demand.ndim == 1:
        demand = demand.unsqueeze(-1)
    if demand.ndim != 2 or demand.shape[1] != int(targetdim):
        raise ValueError("Unexpected training target shape.")
    return condition.detach().cpu(), demand.detach().cpu()


def gradient_norm(gradients):
    """Return the Euclidean norm and largest absolute coordinate."""
    reference = next((gradient for gradient in gradients if gradient is not None), None)
    if reference is None:
        return 0.0, 0.0
    squared = torch.zeros((), device=reference.device, dtype=torch.float32)
    maximum = torch.zeros((), device=reference.device, dtype=torch.float32)
    for gradient in gradients:
        if gradient is None:
            continue
        gradient32 = gradient.detach().float()
        squared += gradient32.square().sum()
        maximum = torch.maximum(maximum, gradient32.abs().max())
    return float(squared.sqrt().cpu()), float(maximum.cpu())


@torch.no_grad()
def project_parameter_box(parameters, lower, upper, *, return_hit_rate=False):
    """Project parameters, optionally returning a device-side boundary-hit rate."""
    lower = float(lower)
    upper = float(upper)
    if lower >= upper:
        raise ValueError("parameter_box_lower must be less than parameter_box_upper.")
    parameters = list(parameters)
    if not parameters:
        return None
    hit = torch.zeros((), device=parameters[0].device, dtype=torch.int64)
    total = 0
    for parameter in parameters:
        parameter.clamp_(min=lower, max=upper)
        if return_hit_rate:
            hit.add_(((parameter == lower) | (parameter == upper)).sum())
        total += parameter.numel()
    if not return_hit_rate:
        return None
    return hit.to(torch.float32) / max(total, 1)


def projected_sgd_step(
    parameters,
    step_size,
    lower,
    upper,
    *,
    return_hit_rate=False,
):
    """Apply one no-momentum SGD update and project every coordinate."""
    parameters = list(parameters)
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is None:
                raise RuntimeError("Every trainable parameter must have a gradient.")
            parameter.add_(parameter.grad, alpha=-float(step_size))
    return project_parameter_box(
        parameters,
        lower,
        upper,
        return_hit_rate=return_hit_rate,
    )
