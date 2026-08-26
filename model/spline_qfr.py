"""QFlow trained by random-quantile pinball loss on the shared spline backbone."""

import copy
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


def pinball_loss(target, prediction, tau):
    residual = target - prediction
    return torch.maximum(tau * residual, (tau - 1.0) * residual)


class SplineQFRNewsvendor(SplineConditionalNewsvendorBase):
    def qfr_objective(self, condition, demand, num_tau=16, tau_eps=None, generator=None, tau=None):
        tau_eps = self.tau_eps if tau_eps is None else float(tau_eps)
        if demand.ndim == 1:
            demand = demand.unsqueeze(-1)
        if tau is None:
            tau = torch.rand(
                condition.shape[0],
                int(num_tau),
                1,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
            tau = tau_eps + (1.0 - 2.0 * tau_eps) * tau
        else:
            tau = self._prepare_quantile_levels(tau, condition, tau_eps)
        quantiles = self.backbone.quantile(condition, tau, tau_eps=tau_eps)
        target = demand[:, None, :].expand_as(quantiles)
        return pinball_loss(target, quantiles, tau).mean(), {
            "tau": tau,
            "quantiles": quantiles,
        }

    def train_spline_qfr(
        self,
        train_loader,
        val_loader,
        *,
        num_epochs=None,
        learning_rate=1e-3,
        optimizer_name="projected_sgd",
        step_size_exponent=0.6,
        training_seed=None,
        parameter_box_lower=-10.0,
        parameter_box_upper=10.0,
        stop_early=True,
        restore_best=True,
        early_stopping=20,
        warmup_epochs=0,
        min_delta_relative=0.0,
        num_tau=16,
        validation_num_tau=99,
        checkpoint_path=None,
        verbose=False,
        epoch_callback=None,
    ):
        num_epochs = self.epoch if num_epochs is None else int(num_epochs)
        early_stopping = int(early_stopping)
        warmup_epochs = int(warmup_epochs)
        min_delta_relative = float(min_delta_relative)
        if (
            min(num_epochs, early_stopping, int(num_tau), int(validation_num_tau)) < 1
            or learning_rate <= 0
        ):
            raise ValueError("Training arguments must be positive.")
        if not 0 <= warmup_epochs < num_epochs:
            raise ValueError("warmup_epochs must lie in [0, num_epochs).")
        if min_delta_relative < 0:
            raise ValueError("min_delta_relative must be nonnegative.")
        optimizer_name = str(optimizer_name).lower()
        if optimizer_name not in {"adam", "projected_sgd"}:
            raise ValueError("optimizer_name must be 'adam' or 'projected_sgd'.")
        optimizer = (
            torch.optim.Adam(self.parameters(), lr=float(learning_rate))
            if optimizer_name == "adam"
            else None
        )
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        global_step = 0
        if optimizer_name == "projected_sgd":
            robbins_monro_step_size(0, learning_rate, step_size_exponent)
            project_parameter_box(
                parameters,
                parameter_box_lower,
                parameter_box_upper,
            )
            train_context, train_demand = training_tensors(
                train_loader,
                targetdim=self.targetdim,
            )
            sample_count = len(train_context)
            batch_size = min(int(train_loader.batch_size or sample_count), sample_count)
            steps_per_epoch = int(np.ceil(sample_count / batch_size))
            training_seed = self.random_seed if training_seed is None else int(training_seed)
            batch_rng = torch.Generator(device="cpu").manual_seed(training_seed)
            objective_rng = (
                torch.Generator(device=self._device()).manual_seed(training_seed + 2)
                if self._device().type != "mps"
                else None
            )
        else:
            steps_per_epoch = len(train_loader)
            batch_rng = None
            objective_rng = None
        validation_tau = torch.linspace(
            self.tau_eps,
            1.0 - self.tau_eps,
            int(validation_num_tau),
            device=self._device(),
        )
        history = {
            "epoch": [],
            "train_qfr": [],
            "val_qfr": [],
            "val_newsvendor": [],
            "best_epoch": -1,
            "training_objective": "random_tau_integrated_pinball",
            "num_random_tau_per_observation": int(num_tau),
            "optimizer": optimizer_name,
            "gamma0": float(learning_rate),
            "step_size_exponent": float(step_size_exponent),
            "parameter_box_lower": float(parameter_box_lower),
            "parameter_box_upper": float(parameter_box_upper),
            "steps_per_epoch": steps_per_epoch,
            "stop_early": bool(stop_early),
            "restore_best": bool(restore_best),
            "step_size_first": [],
            "step_size_last": [],
            "early_stopping": early_stopping,
            "warmup_epochs": warmup_epochs,
            "min_delta_relative": min_delta_relative,
        }
        best_val_newsvendor = float("inf")
        best_qfr_at_newsvendor = float("inf")
        best_state = None
        patience = 0
        for epoch in range(num_epochs):
            self.train()
            train_losses = []
            epoch_step_sizes = []
            batches = train_loader if optimizer is not None else range(steps_per_epoch)
            for batch in batches:
                if optimizer is not None:
                    condition, target = self._split_spline_batch(batch)
                else:
                    indices = torch.randperm(sample_count, generator=batch_rng)[:batch_size]
                    condition = train_context.index_select(0, indices).to(self._device())
                    target = train_demand.index_select(0, indices).to(self._device())
                loss, _ = self.qfr_objective(
                    condition,
                    target,
                    num_tau=num_tau,
                    generator=objective_rng,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite Spline-QFR loss encountered.")
                for parameter in parameters:
                    parameter.grad = None
                loss.backward()
                if optimizer is not None:
                    optimizer.step()
                else:
                    step_size = robbins_monro_step_size(
                        global_step,
                        learning_rate,
                        step_size_exponent,
                    )
                    projected_sgd_step(
                        parameters,
                        step_size,
                        parameter_box_lower,
                        parameter_box_upper,
                    )
                    epoch_step_sizes.append(step_size)
                    global_step += 1
                train_losses.append(float(loss.detach()))

            train_value = float(np.mean(train_losses)) if train_losses else float("inf")
            history["epoch"].append(epoch)
            history["train_qfr"].append(train_value)
            history["step_size_first"].append(
                epoch_step_sizes[0] if epoch_step_sizes else float(learning_rate)
            )
            history["step_size_last"].append(
                epoch_step_sizes[-1] if epoch_step_sizes else float(learning_rate)
            )
            if epoch < warmup_epochs:
                history["val_qfr"].append(float("nan"))
                history["val_newsvendor"].append(float("nan"))
                continue

            self.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    condition, target = self._split_spline_batch(batch)
                    loss, _ = self.qfr_objective(condition, target, tau=validation_tau)
                    val_losses.append(float(loss))
            val_qfr = float(np.mean(val_losses)) if val_losses else float("inf")
            val_newsvendor = self.evaluate_exact_newsvendor(val_loader)["newsvendor_loss"]
            history["val_qfr"].append(val_qfr)
            history["val_newsvendor"].append(val_newsvendor)
            if epoch_callback is not None:
                epoch_callback(
                    epoch=epoch,
                    total_epochs=num_epochs,
                    train_value=train_value,
                    validation_value=val_newsvendor,
                )
            if verbose:
                print(
                    f"epoch={epoch} train_qfr={train_value:.6f} "
                    f"val_newsvendor={val_newsvendor:.6f}"
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
                best_qfr_at_newsvendor = val_qfr
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
        history["best_val_qfr"] = best_qfr_at_newsvendor
        history["epochs_ran"] = len(history["epoch"])
        history["steps_ran"] = (
            global_step if optimizer is None else len(history["epoch"]) * steps_per_epoch
        )
        history["checkpoint_path"] = str(checkpoint_path) if checkpoint_path else None
        return history
