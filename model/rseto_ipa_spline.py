"""RSETO-IPA-Spline with explicit [B, R, m, 1] simulation axes."""

import copy
import math
from pathlib import Path

import numpy as np
import torch

from model.gendfl_spline import SplineConditionalNewsvendorBase
from model.projected_sa import (
    project_parameter_box,
    projected_sgd_step,
    robbins_monro_step_size,
    training_tensors,
)


def increasing_sample_size(iteration, initial_m, growth, exponent):
    """Return an unbounded integer schedule m_k starting at initial_m."""
    iteration = int(iteration)
    initial_m = int(initial_m)
    growth = float(growth)
    exponent = float(exponent)
    if iteration < 0 or initial_m < 1 or growth <= 0.0 or exponent <= 0.0:
        raise ValueError(
            "Need iteration >= 0, initial_m >= 1, growth > 0, and exponent > 0."
        )
    return initial_m + int(math.floor(growth * (iteration**exponent)))


def smooth_newsvendor_loss(q, demand, shortage_cost, overage_cost, smoothing_mu):
    smoothing_mu = float(smoothing_mu)
    if min(float(shortage_cost), float(overage_cost), smoothing_mu) <= 0.0:
        raise ValueError("Costs and smoothing_mu must be positive.")
    zero = torch.zeros((), device=q.device, dtype=q.dtype)
    return (
        shortage_cost
        * smoothing_mu
        * torch.logaddexp(zero, (demand - q) / smoothing_mu)
        + overage_cost
        * smoothing_mu
        * torch.logaddexp(zero, (q - demand) / smoothing_mu)
    )


def exact_spline_newsvendor_objective(model, condition, demand, smoothing_mu):
    """Exact scalar-flow task oracle at the critical fractile."""
    if demand.ndim == 1:
        demand = demand.unsqueeze(-1)
    exact_quantile = model.critical_quantile_decision(condition)
    loss = smooth_newsvendor_loss(
        exact_quantile,
        demand,
        model.cu,
        model.co,
        smoothing_mu,
    ).mean()
    return loss, exact_quantile


def _gather_selected_noise(noise, selected_index):
    """Gather one latent draw per [context, replication] order statistic."""
    latent_dim = noise.shape[-1]
    gather_index = selected_index[..., None, None].expand(
        *selected_index.shape,
        1,
        latent_dim,
    )
    return noise.gather(dim=2, index=gather_index).squeeze(2)


def _slice_encoded_condition(encoded_condition, start, end):
    return tuple(encoded[start:end] for encoded in encoded_condition)


@torch.no_grad()
def screen_selected_base_noise(
    backbone,
    condition,
    *,
    replications,
    samples_per_replication,
    target_quantile,
    max_simulation_values,
    generator=None,
    noise_device=None,
    base_noise=None,
    collect_diagnostics=False,
    finite_check=False,
):
    """Select each empirical quantile's base noise without an autograd graph."""
    replications = int(replications)
    samples_per_replication = int(samples_per_replication)
    max_simulation_values = int(max_simulation_values)
    if min(replications, samples_per_replication, max_simulation_values) < 1:
        raise ValueError("R, m, and max_simulation_values must be positive.")
    if samples_per_replication > max_simulation_values:
        raise ValueError("max_simulation_values must be at least m.")

    batch_size = condition.shape[0]
    device = condition.device
    dtype = condition.dtype
    latent_dim = 1 if base_noise is None else int(base_noise.shape[-1])
    expected_shape = (batch_size, replications, samples_per_replication, latent_dim)
    if base_noise is not None and tuple(base_noise.shape) != expected_shape:
        raise ValueError(
            f"base_noise must have shape {expected_shape}, got {tuple(base_noise.shape)}."
        )
    noise_device = device if noise_device is None else torch.device(noise_device)
    order_index = max(
        1,
        min(
            samples_per_replication,
            math.ceil(float(target_quantile) * samples_per_replication),
        ),
    )
    replication_chunk = min(
        replications,
        max(1, max_simulation_values // samples_per_replication),
    )
    context_chunk = min(
        batch_size,
        max(
            1,
            max_simulation_values
            // (replication_chunk * samples_per_replication),
        ),
    )

    selected_noise = torch.empty(
        batch_size,
        replications,
        latent_dim,
        device=device,
        dtype=dtype,
    )
    selected_value = torch.empty(
        batch_size,
        replications,
        device=device,
        dtype=dtype,
    )
    selected_indices = torch.empty(
        batch_size,
        replications,
        device=device,
        dtype=torch.int64,
    )
    encoded_condition = backbone.encode_condition(condition)
    tie_count = torch.zeros((), device=device, dtype=torch.int64)
    gap_sum = torch.zeros((), device=device, dtype=torch.float32)
    gap_count = torch.zeros((), device=device, dtype=torch.int64)
    finite_flag = torch.ones((), device=device, dtype=torch.bool)
    chunk_count = 0

    for context_start in range(0, batch_size, context_chunk):
        context_end = min(context_start + context_chunk, batch_size)
        encoded_chunk = _slice_encoded_condition(
            encoded_condition,
            context_start,
            context_end,
        )
        for replication_start in range(0, replications, replication_chunk):
            replication_end = min(
                replication_start + replication_chunk,
                replications,
            )
            chunk_shape = (
                context_end - context_start,
                replication_end - replication_start,
                samples_per_replication,
                latent_dim,
            )
            if base_noise is None:
                noise = torch.randn(
                    chunk_shape,
                    device=noise_device,
                    dtype=dtype,
                    generator=generator,
                ).to(device, non_blocking=True)
            else:
                noise = base_noise[
                    context_start:context_end,
                    replication_start:replication_end,
                ].to(device=device, dtype=dtype, non_blocking=True)
            generated = backbone.sample_from_encoded_condition(
                encoded_chunk,
                noise,
            ).squeeze(-1)
            if finite_check:
                finite_flag.logical_and_(torch.isfinite(generated).all())
            quantile, selected_index = torch.kthvalue(
                generated,
                k=order_index,
                dim=-1,
            )
            selected_noise[
                context_start:context_end,
                replication_start:replication_end,
            ] = _gather_selected_noise(noise, selected_index)
            selected_value[
                context_start:context_end,
                replication_start:replication_end,
            ] = quantile
            selected_indices[
                context_start:context_end,
                replication_start:replication_end,
            ] = selected_index

            if collect_diagnostics:
                gaps = []
                if order_index > 1:
                    lower = torch.kthvalue(
                        generated,
                        k=order_index - 1,
                        dim=-1,
                    ).values
                    gaps.append(quantile - lower)
                if order_index < samples_per_replication:
                    upper = torch.kthvalue(
                        generated,
                        k=order_index + 1,
                        dim=-1,
                    ).values
                    gaps.append(upper - quantile)
                if gaps:
                    min_gap = torch.stack(gaps).amin(dim=0)
                    tie_count.add_((min_gap == 0).sum())
                    gap_sum.add_(min_gap.float().sum())
                    gap_count.add_(min_gap.numel())
            chunk_count += 1

    return selected_noise, selected_value, {
        "order_index": order_index,
        "selected_index": selected_indices,
        "tie_count": tie_count,
        "gap_sum": gap_sum,
        "gap_count": gap_count,
        "finite": finite_flag,
        "chunk_count": chunk_count,
        "context_chunk": context_chunk,
        "replication_chunk": replication_chunk,
    }


def replay_selected_quantiles(
    backbone,
    condition,
    selected_noise,
    *,
    encoded_condition=None,
):
    """Replay only the selected BR latent paths with gradients enabled."""
    if selected_noise.ndim != 3 or selected_noise.shape[0] != condition.shape[0]:
        raise ValueError("selected_noise must have shape [B, R, latent_dim].")
    if encoded_condition is None:
        encoded_condition = backbone.encode_condition(condition)
    replay = backbone.sample_from_encoded_condition(
        encoded_condition,
        selected_noise.unsqueeze(2),
    )
    return replay.squeeze(-1).squeeze(-1)


def _gradient_statistics(parameters):
    reference = next(
        (parameter.grad for parameter in parameters if parameter.grad is not None),
        None,
    )
    if reference is None:
        raise RuntimeError("No gradients are available for the projected update.")
    squared = torch.zeros((), device=reference.device, dtype=torch.float32)
    maximum = torch.zeros((), device=reference.device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        squared.add_(gradient.square().sum())
        maximum = torch.maximum(maximum, gradient.abs().max())
    return squared.sqrt(), maximum


def _gradients_are_finite(parameters):
    reference = next(
        (parameter.grad for parameter in parameters if parameter.grad is not None),
        None,
    )
    if reference is None:
        return torch.zeros((), dtype=torch.bool)
    finite = torch.ones((), device=reference.device, dtype=torch.bool)
    for parameter in parameters:
        if parameter.grad is None:
            finite.fill_(False)
        else:
            finite.logical_and_(torch.isfinite(parameter.grad).all())
    return finite


class RSETOIPASplineNewsvendor(SplineConditionalNewsvendorBase):
    @staticmethod
    def _make_noise_generator(device, seed):
        if device.type == "mps":
            return torch.Generator(device="cpu").manual_seed(int(seed)), torch.device("cpu")
        return torch.Generator(device=device).manual_seed(int(seed)), device

    def rseto_ipa_objective(
        self,
        condition,
        demand,
        *,
        replications,
        samples_per_replication,
        smoothing_mu,
        fidelity_weight,
        generator=None,
        base_noise=None,
    ):
        replications = int(replications)
        samples_per_replication = int(samples_per_replication)
        fidelity_weight = float(fidelity_weight)
        if replications < 1 or samples_per_replication < 1:
            raise ValueError("replications and samples_per_replication must be positive.")
        if not 0.0 <= fidelity_weight <= 1.0:
            raise ValueError("fidelity_weight must lie in [0, 1].")
        if demand.ndim == 2:
            demand_vector = demand.squeeze(-1)
        else:
            demand_vector = demand
        expected_shape = (
            condition.shape[0],
            replications,
            samples_per_replication,
            1,
        )
        if base_noise is None:
            base_noise = torch.randn(
                expected_shape,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
        else:
            base_noise = base_noise.to(device=condition.device, dtype=condition.dtype)
            if tuple(base_noise.shape) != expected_shape:
                raise ValueError(
                    f"base_noise must have shape {expected_shape}, got {tuple(base_noise.shape)}."
                )
        generated = self.backbone.sample_from_base_noise(condition, base_noise)
        order_index = max(
            1,
            min(
                samples_per_replication,
                math.ceil(self.target_quantile * samples_per_replication),
            ),
        )
        selected_quantile, selected_index = torch.kthvalue(
            generated.squeeze(-1),
            k=order_index,
            dim=-1,
        )
        if order_index > 1:
            lower_quantile = torch.kthvalue(
                generated.squeeze(-1),
                k=order_index - 1,
                dim=-1,
            ).values
            lower_gap = selected_quantile - lower_quantile
        else:
            lower_gap = torch.full_like(selected_quantile, float("nan"))
        if order_index < samples_per_replication:
            upper_quantile = torch.kthvalue(
                generated.squeeze(-1),
                k=order_index + 1,
                dim=-1,
            ).values
            upper_gap = upper_quantile - selected_quantile
        else:
            upper_gap = torch.full_like(selected_quantile, float("nan"))
        ipa_task_loss = smooth_newsvendor_loss(
            selected_quantile,
            demand_vector[:, None],
            self.cu,
            self.co,
            smoothing_mu,
        ).mean()
        fidelity_loss = self.generative_loss(demand, condition)
        total_loss = fidelity_weight * fidelity_loss + (1.0 - fidelity_weight) * ipa_task_loss
        return total_loss, {
            "fidelity_loss": fidelity_loss,
            "ipa_task_loss": ipa_task_loss,
            "base_noise": base_noise,
            "generated": generated,
            "selected_index": selected_index,
            "selected_quantile": selected_quantile,
            "order_index": order_index,
            "lower_order_gap": lower_gap,
            "upper_order_gap": upper_gap,
        }

    def rseto_ipa_replay_objective(
        self,
        condition,
        demand,
        *,
        selected_noise,
        smoothing_mu,
        fidelity_weight,
    ):
        """Joint NLL/IPA objective using only pre-screened latent paths."""
        fidelity_weight = float(fidelity_weight)
        if not 0.0 <= fidelity_weight <= 1.0:
            raise ValueError("fidelity_weight must lie in [0, 1].")
        if demand.ndim == 1:
            demand = demand.unsqueeze(-1)

        encoded_condition = self.backbone.encode_condition(condition)
        if fidelity_weight < 1.0:
            if selected_noise is None:
                raise ValueError("selected_noise is required when fidelity_weight < 1.")
            selected_quantile = replay_selected_quantiles(
                self.backbone,
                condition,
                selected_noise,
                encoded_condition=encoded_condition,
            )
            ipa_task_loss = smooth_newsvendor_loss(
                selected_quantile,
                demand[:, 0, None],
                self.cu,
                self.co,
                smoothing_mu,
            ).mean()
        else:
            selected_quantile = None
            ipa_task_loss = condition.new_zeros(())

        if fidelity_weight > 0.0:
            fidelity_loss = -self.backbone.log_prob_from_encoded(
                demand,
                encoded_condition,
            ).mean()
        else:
            fidelity_loss = condition.new_zeros(())
        total_loss = (
            fidelity_weight * fidelity_loss
            + (1.0 - fidelity_weight) * ipa_task_loss
        )
        return total_loss, {
            "fidelity_loss": fidelity_loss,
            "ipa_task_loss": ipa_task_loss,
            "selected_quantile": selected_quantile,
        }

    def exact_task_objective(self, condition, demand, smoothing_mu=0.05):
        return exact_spline_newsvendor_objective(
            self,
            condition,
            demand,
            smoothing_mu,
        )

    def estimate_batch_ipa_gradient_variance(
        self,
        condition,
        demand,
        *,
        replications,
        samples_per_replication,
        smoothing_mu=0.05,
        diagnostic_repeats=8,
        max_simulation_values=1048576,
        seed=31701,
    ):
        """Estimate variance of the batch IPA gradient averaged over R replications.

        Each diagnostic repeat independently constructs the same estimator used by
        training: an average over a fixed data batch and ``replications`` empirical
        quantiles, with ``samples_per_replication`` draws per quantile. The sample
        variance across repeats therefore measures the variance of the averaged
        gradient itself, which is the quantity expected to decrease as R grows.
        """
        replications = int(replications)
        samples_per_replication = int(samples_per_replication)
        diagnostic_repeats = int(diagnostic_repeats)
        max_simulation_values = int(max_simulation_values)
        if min(
            replications,
            samples_per_replication,
            max_simulation_values,
        ) < 1:
            raise ValueError("R, m, and max_simulation_values must be positive.")
        if diagnostic_repeats < 2:
            raise ValueError("diagnostic_repeats must be at least 2.")
        if samples_per_replication > max_simulation_values:
            raise ValueError("max_simulation_values must be at least m.")
        if condition.ndim != 2 or condition.shape[0] < 1:
            raise ValueError("condition must be a non-empty [B, d] tensor.")
        if demand.ndim == 1:
            demand = demand.unsqueeze(-1)
        if demand.ndim != 2 or demand.shape != (condition.shape[0], 1):
            raise ValueError("demand must have shape [B] or [B, 1].")

        device = self._device()
        condition = condition.to(device=device, dtype=next(self.parameters()).dtype)
        demand = demand.to(device=device, dtype=condition.dtype)
        parameters = tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        generator, noise_device = self._make_noise_generator(device, seed)
        gradient_vectors = []
        losses = []
        was_training = self.training
        self.eval()
        try:
            with torch.enable_grad():
                for _ in range(diagnostic_repeats):
                    selected_noise, _, _ = screen_selected_base_noise(
                        self.backbone,
                        condition,
                        replications=replications,
                        samples_per_replication=samples_per_replication,
                        target_quantile=self.target_quantile,
                        max_simulation_values=max_simulation_values,
                        generator=generator,
                        noise_device=noise_device,
                    )
                    selected_quantile = replay_selected_quantiles(
                        self.backbone,
                        condition,
                        selected_noise,
                    )
                    ipa_loss = smooth_newsvendor_loss(
                        selected_quantile,
                        demand[:, 0, None],
                        self.cu,
                        self.co,
                        smoothing_mu,
                    ).mean()
                    gradients = torch.autograd.grad(
                        ipa_loss,
                        parameters,
                        allow_unused=True,
                    )
                    gradient_vectors.append(
                        torch.cat(
                            [
                                (
                                    torch.zeros_like(parameter)
                                    if gradient is None
                                    else gradient
                                )
                                .detach()
                                .float()
                                .reshape(-1)
                                .cpu()
                                for parameter, gradient in zip(parameters, gradients)
                            ]
                        )
                    )
                    losses.append(float(ipa_loss.detach().cpu()))
        finally:
            self.train(was_training)

        gradient_matrix = torch.stack(gradient_vectors, dim=0)
        coordinate_variance = gradient_matrix.var(dim=0, unbiased=True)
        mean_gradient = gradient_matrix.mean(dim=0)
        variance_trace = coordinate_variance.sum()
        mean_gradient_squared_norm = mean_gradient.square().sum()
        parameter_count = gradient_matrix.shape[1]
        return {
            "ipa_gradient_variance_trace": float(variance_trace),
            "ipa_gradient_variance_mean_per_parameter": float(
                variance_trace / max(parameter_count, 1)
            ),
            "ipa_gradient_std_norm": float(variance_trace.sqrt()),
            "ipa_gradient_mean_norm": float(mean_gradient_squared_norm.sqrt()),
            "ipa_gradient_relative_variance": float(
                variance_trace / mean_gradient_squared_norm.clamp_min(1e-24)
            ),
            "ipa_gradient_loss_mean": float(np.mean(losses)),
            "ipa_gradient_loss_variance": float(np.var(losses, ddof=1)),
            "gradient_variance_repeats": diagnostic_repeats,
            "gradient_variance_batch_size": int(condition.shape[0]),
            "gradient_variance_replications": replications,
            "gradient_variance_samples_per_replication": samples_per_replication,
            "gradient_variance_seed": int(seed),
            "gradient_parameter_count": int(parameter_count),
        }

    def train_rseto_ipa_spline(
        self,
        train_loader,
        val_loader,
        *,
        num_epochs=None,
        learning_rate=1e-3,
        step_size_exponent=0.6,
        stop_early=True,
        restore_best=True,
        early_stopping=20,
        warmup_epochs=0,
        min_delta_relative=0.0,
        replications=16,
        samples_per_replication=128,
        m_growth=1.0,
        m_growth_exponent=0.25,
        smoothing_mu=0.05,
        fidelity_weight=0.5,
        validation_seed=1701,
        training_seed=None,
        parameter_box_lower=-10.0,
        parameter_box_upper=10.0,
        max_simulation_values=262144,
        diagnostic_interval=100,
        finite_check_interval=100,
        train_data_on_device=True,
        checkpoint_path=None,
        verbose=False,
        epoch_callback=None,
    ):
        """Train using the projected stochastic-approximation recursion in Theorem 2.

        ``samples_per_replication`` is the initial value m_0. The actual sample
        count follows the unbounded ``increasing_sample_size`` schedule. Large
        [B, R, m_k, 1] simulations are split into vectorized chunks; gradients
        from every chunk are accumulated at the same pre-update parameter state.
        """
        num_epochs = self.epoch if num_epochs is None else int(num_epochs)
        early_stopping = int(early_stopping)
        warmup_epochs = int(warmup_epochs)
        min_delta_relative = float(min_delta_relative)
        replications = int(replications)
        samples_per_replication = int(samples_per_replication)
        max_simulation_values = int(max_simulation_values)
        diagnostic_interval = int(diagnostic_interval)
        finite_check_interval = int(finite_check_interval)
        parameter_box_lower = float(parameter_box_lower)
        parameter_box_upper = float(parameter_box_upper)
        if min(
            num_epochs,
            early_stopping,
            replications,
            samples_per_replication,
            max_simulation_values,
            diagnostic_interval,
            finite_check_interval,
        ) < 1 or learning_rate <= 0:
            raise ValueError("Training arguments must be positive.")
        if not 0 <= warmup_epochs < num_epochs:
            raise ValueError("warmup_epochs must lie in [0, num_epochs).")
        if min_delta_relative < 0:
            raise ValueError("min_delta_relative must be nonnegative.")
        if not 0.0 <= float(fidelity_weight) <= 1.0:
            raise ValueError("fidelity_weight must lie in [0, 1].")
        if parameter_box_lower >= parameter_box_upper:
            raise ValueError("parameter_box_lower must be less than parameter_box_upper.")
        robbins_monro_step_size(0, learning_rate, step_size_exponent)
        increasing_sample_size(
            0,
            samples_per_replication,
            m_growth,
            m_growth_exponent,
        )

        train_context, train_demand = training_tensors(
            train_loader,
            targetdim=self.targetdim,
        )
        sample_count = len(train_context)
        batch_size = min(int(train_loader.batch_size or sample_count), sample_count)
        steps_per_epoch = math.ceil(sample_count / batch_size)
        device = self._device()
        if train_data_on_device:
            train_context = train_context.to(device)
            train_demand = train_demand.to(device)
        training_seed = self.random_seed if training_seed is None else int(training_seed)
        batch_rng = torch.Generator(device="cpu").manual_seed(training_seed)
        ipa_rng, noise_device = self._make_noise_generator(device, training_seed + 1)
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        project_parameter_box(
            parameters,
            parameter_box_lower,
            parameter_box_upper,
        )

        history = {
            "epoch": [],
            "train_total": [],
            "train_nll": [],
            "train_ipa": [],
            "val_total": [],
            "val_nll": [],
            "val_ipa": [],
            "val_newsvendor": [],
            "best_epoch": -1,
            "fidelity_weight": float(fidelity_weight),
            "optimizer": "projected_sgd",
            "gamma0": float(learning_rate),
            "step_size_exponent": float(step_size_exponent),
            "initial_m": samples_per_replication,
            "m_growth": float(m_growth),
            "m_growth_exponent": float(m_growth_exponent),
            "replications": replications,
            "parameter_box_lower": parameter_box_lower,
            "parameter_box_upper": parameter_box_upper,
            "max_simulation_values": max_simulation_values,
            "acceleration": "screen_and_replay",
            "diagnostic_interval": diagnostic_interval,
            "finite_check_interval": finite_check_interval,
            "train_data_on_device": bool(train_data_on_device),
            "steps_per_epoch": steps_per_epoch,
            "stop_early": bool(stop_early),
            "restore_best": bool(restore_best),
            "step_size_first": [],
            "step_size_last": [],
            "m_first": [],
            "m_last": [],
            "grad_norm": [],
            "fidelity_grad_norm": [],
            "ipa_grad_norm": [],
            "max_abs_grad": [],
            "projection_hit_rate": [],
            "exact_tie_rate": [],
            "mean_min_neighbor_gap": [],
            "max_replay_error": [],
            "screening_chunks_per_step": [],
            "early_stopping": early_stopping,
            "warmup_epochs": warmup_epochs,
            "min_delta_relative": min_delta_relative,
        }
        best_val_newsvendor = float("inf")
        best_total_at_newsvendor = float("inf")
        best_state = None
        patience = 0
        global_step = 0
        for epoch in range(num_epochs):
            self.train()
            epoch_total = torch.zeros((), device=device, dtype=torch.float32)
            epoch_nll = torch.zeros((), device=device, dtype=torch.float32)
            epoch_ipa = torch.zeros((), device=device, dtype=torch.float32)
            epoch_grad_norm = torch.zeros((), device=device, dtype=torch.float32)
            epoch_max_abs_grad = torch.zeros((), device=device, dtype=torch.float32)
            epoch_projection_hit = torch.zeros((), device=device, dtype=torch.float32)
            epoch_ties = torch.zeros((), device=device, dtype=torch.int64)
            epoch_quantiles = torch.zeros((), device=device, dtype=torch.int64)
            epoch_gap_sum = torch.zeros((), device=device, dtype=torch.float32)
            epoch_gap_count = torch.zeros((), device=device, dtype=torch.int64)
            epoch_max_replay_error = torch.zeros(
                (),
                device=device,
                dtype=torch.float32,
            )
            epoch_diagnostic_steps = 0
            epoch_screening_chunks = 0
            first_step_size = None
            last_step_size = None
            first_m = None
            last_m = None
            for _ in range(steps_per_epoch):
                step_size = robbins_monro_step_size(
                    global_step,
                    learning_rate,
                    step_size_exponent,
                )
                current_m = increasing_sample_size(
                    global_step,
                    samples_per_replication,
                    m_growth,
                    m_growth_exponent,
                )
                if current_m > max_simulation_values:
                    raise ValueError(
                        "max_simulation_values must be at least the current m_k so "
                        "one complete order statistic can be computed."
                    )
                indices = torch.randperm(sample_count, generator=batch_rng)[:batch_size]
                if train_data_on_device:
                    indices = indices.to(device, non_blocking=True)
                    condition = train_context.index_select(0, indices)
                    target = train_demand.index_select(0, indices)
                else:
                    condition = train_context.index_select(0, indices).to(
                        device,
                        non_blocking=True,
                    )
                    target = train_demand.index_select(0, indices).to(
                        device,
                        non_blocking=True,
                    )

                do_diagnostics = global_step % diagnostic_interval == 0
                do_finite_check = global_step % finite_check_interval == 0

                for parameter in parameters:
                    parameter.grad = None
                selected_noise = None
                selected_value_screen = None
                screening = None
                if float(fidelity_weight) < 1.0:
                    (
                        selected_noise,
                        selected_value_screen,
                        screening,
                    ) = screen_selected_base_noise(
                        self.backbone,
                        condition,
                        replications=replications,
                        samples_per_replication=current_m,
                        target_quantile=self.target_quantile,
                        max_simulation_values=max_simulation_values,
                        generator=ipa_rng,
                        noise_device=noise_device,
                        collect_diagnostics=do_diagnostics,
                        finite_check=do_finite_check,
                    )
                    epoch_screening_chunks += screening["chunk_count"]

                total_loss, details = self.rseto_ipa_replay_objective(
                    condition,
                    target,
                    selected_noise=selected_noise,
                    smoothing_mu=smoothing_mu,
                    fidelity_weight=fidelity_weight,
                )
                total_loss.backward()

                if do_finite_check:
                    finite = torch.isfinite(total_loss.detach())
                    if screening is not None:
                        finite.logical_and_(screening["finite"])
                    finite.logical_and_(_gradients_are_finite(parameters))
                    if not bool(finite.item()):
                        raise FloatingPointError(
                            "Non-finite RSETO screen/replay loss or gradient."
                        )

                if do_diagnostics:
                    grad_norm, max_abs_grad = _gradient_statistics(parameters)
                    epoch_grad_norm.add_(grad_norm)
                    epoch_max_abs_grad = torch.maximum(
                        epoch_max_abs_grad,
                        max_abs_grad,
                    )
                    epoch_diagnostic_steps += 1
                    if screening is not None:
                        epoch_ties.add_(screening["tie_count"])
                        epoch_quantiles.add_(screening["gap_count"])
                        epoch_gap_sum.add_(screening["gap_sum"])
                        epoch_gap_count.add_(screening["gap_count"])
                        replay_error = (
                            details["selected_quantile"].detach()
                            - selected_value_screen
                        ).abs().max().float()
                        epoch_max_replay_error = torch.maximum(
                            epoch_max_replay_error,
                            replay_error,
                        )

                projection_hit = projected_sgd_step(
                    parameters,
                    step_size,
                    parameter_box_lower,
                    parameter_box_upper,
                    return_hit_rate=do_diagnostics,
                )
                if projection_hit is not None:
                    epoch_projection_hit.add_(projection_hit)

                epoch_total.add_(total_loss.detach().float())
                epoch_nll.add_(details["fidelity_loss"].detach().float())
                epoch_ipa.add_(details["ipa_task_loss"].detach().float())
                if first_step_size is None:
                    first_step_size = step_size
                    first_m = current_m
                last_step_size = step_size
                last_m = current_m
                global_step += 1

            train_value = float((epoch_total / steps_per_epoch).cpu())
            history["epoch"].append(epoch)
            history["train_total"].append(train_value)
            history["train_nll"].append(float((epoch_nll / steps_per_epoch).cpu()))
            history["train_ipa"].append(float((epoch_ipa / steps_per_epoch).cpu()))
            history["step_size_first"].append(first_step_size)
            history["step_size_last"].append(last_step_size)
            history["m_first"].append(first_m)
            history["m_last"].append(last_m)
            diagnostic_denominator = max(epoch_diagnostic_steps, 1)
            history["grad_norm"].append(
                float((epoch_grad_norm / diagnostic_denominator).cpu())
            )
            history["fidelity_grad_norm"].append(float("nan"))
            history["ipa_grad_norm"].append(float("nan"))
            history["max_abs_grad"].append(float(epoch_max_abs_grad.cpu()))
            history["projection_hit_rate"].append(
                float((epoch_projection_hit / diagnostic_denominator).cpu())
            )
            history["exact_tie_rate"].append(
                float(
                    (
                        epoch_ties.float()
                        / epoch_quantiles.clamp_min(1).float()
                    ).cpu()
                )
            )
            history["mean_min_neighbor_gap"].append(
                float(
                    (
                        epoch_gap_sum
                        / epoch_gap_count.clamp_min(1).float()
                    ).cpu()
                )
            )
            history["max_replay_error"].append(
                float(epoch_max_replay_error.cpu())
            )
            history["screening_chunks_per_step"].append(
                epoch_screening_chunks / steps_per_epoch
            )
            if epoch < warmup_epochs:
                history["val_total"].append(float("nan"))
                history["val_nll"].append(float("nan"))
                history["val_ipa"].append(float("nan"))
                history["val_newsvendor"].append(float("nan"))
                continue

            validation = self.evaluate_exact_newsvendor(val_loader)
            val_newsvendor = validation["newsvendor_loss"]
            val_nll = validation["nll"]
            val_total = (
                float(fidelity_weight) * val_nll
                + (1.0 - float(fidelity_weight)) * val_newsvendor
            )
            history["val_total"].append(val_total)
            history["val_nll"].append(val_nll)
            history["val_ipa"].append(val_newsvendor)
            history["val_newsvendor"].append(val_newsvendor)
            if epoch_callback is not None:
                epoch_callback(
                    epoch=epoch,
                    total_epochs=num_epochs,
                    train_value=train_value,
                    validation_value=val_newsvendor,
                    current_m=last_m,
                )
            if verbose:
                print(
                    f"epoch={epoch} train_total={train_value:.6f} "
                    f"val_newsvendor={val_newsvendor:.6f} "
                    f"gamma={last_step_size:.3e} m={last_m}"
                )
            relative_improvement = (
                (best_val_newsvendor - val_newsvendor)
                / max(abs(best_val_newsvendor), 1e-12)
                if np.isfinite(best_val_newsvendor)
                else float("inf")
            )
            if np.isfinite(val_newsvendor) and (
                not np.isfinite(best_val_newsvendor)
                or relative_improvement > min_delta_relative
            ):
                best_val_newsvendor = val_newsvendor
                best_total_at_newsvendor = val_total
                history["best_epoch"] = epoch
                best_state = copy.deepcopy(self.state_dict())
                patience = 0
                if checkpoint_path is not None:
                    checkpoint_path = Path(checkpoint_path)
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, checkpoint_path)
            else:
                patience += 1
                if stop_early and patience >= early_stopping:
                    break
        if best_state is not None and restore_best:
            self.load_state_dict(best_state)
        history["best_val_newsvendor"] = best_val_newsvendor
        history["best_val_total"] = best_total_at_newsvendor
        history["epochs_ran"] = len(history["epoch"])
        history["steps_ran"] = global_step
        history["final_step_size"] = (
            history["step_size_last"][-1] if history["step_size_last"] else None
        )
        history["final_m"] = history["m_last"][-1] if history["m_last"] else None
        history["checkpoint_path"] = str(checkpoint_path) if checkpoint_path else None
        return history
