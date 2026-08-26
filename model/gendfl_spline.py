"""Gen-DFL-Spline model using the shared conditional RQS backbone."""

import copy
from pathlib import Path

import numpy as np
import torch

from model.generative_newsvendor_base import GenerativeNewsvendorBase
from model.projected_sa import (
    project_parameter_box,
    projected_sgd_step,
    robbins_monro_step_size,
    training_tensors,
)
from model.shared_spline_flow import SharedConditionalSplineFlow, SplineFlowConfig


class SplineConditionalNewsvendorBase(GenerativeNewsvendorBase):
    """Common adapter from the shared scalar spline to newsvendor interfaces."""

    def __init__(
        self,
        targetdim,
        labeldim,
        latent=1,
        data_len=0,
        epoch=100,
        quantiles=0.5,
        lambda1=0.5,
        lambda_gradient=0.5,
        samplingnumber=100,
        target_quantile=None,
        cost_under=10.0,
        cost_over=5.0,
        random_seed=0,
        innerloop=1,
        num_transforms=4,
        num_bins=16,
        hidden_dim=64,
        hidden_layers=2,
        tail_bound=4.0,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
        tau_eps=1e-5,
    ):
        super().__init__()
        if int(targetdim) != 1 or int(latent) != 1:
            raise ValueError("The shared spline flow requires targetdim=latent=1.")
        self.spline_config = SplineFlowConfig(
            context_dim=int(labeldim),
            num_transforms=int(num_transforms),
            num_bins=int(num_bins),
            hidden_dim=int(hidden_dim),
            hidden_layers=int(hidden_layers),
            tail_bound=float(tail_bound),
            min_bin_width=float(min_bin_width),
            min_bin_height=float(min_bin_height),
            min_derivative=float(min_derivative),
        )
        self.backbone = SharedConditionalSplineFlow(self.spline_config)
        self.tau_eps = float(tau_eps)
        if not 0.0 < self.tau_eps < 0.5:
            raise ValueError("tau_eps must lie strictly between zero and 0.5.")
        self._init_newsvendor_base(
            targetdim=targetdim,
            labeldim=labeldim,
            latent=latent,
            data_len=data_len,
            epoch=epoch,
            quantiles=quantiles,
            lambda1=lambda1,
            lambda_gradient=lambda_gradient,
            samplingnumber=samplingnumber,
            target_quantile=target_quantile,
            cost_under=cost_under,
            cost_over=cost_over,
            random_seed=random_seed,
            innerloop=innerloop,
        )

    def decode(self, z, condition):
        return self.backbone.sample_from_base_noise(condition, z)

    def generative_loss(self, y_true, condition):
        return -self.backbone.log_prob(y_true, condition).mean()

    def sample_from_base_noise(self, condition, z):
        return self.backbone.sample_from_base_noise(condition, z)

    def sample(self, num_samples, condition, generator=None):
        return self.backbone.sample(num_samples, condition, generator=generator)

    def quantile(self, tau, condition, tau_eps=None):
        tau_eps = self.tau_eps if tau_eps is None else float(tau_eps)
        tau = self._prepare_quantile_levels(tau, condition, tau_eps)
        return self.backbone.quantile(condition, tau, tau_eps=tau_eps)

    @staticmethod
    def _prepare_quantile_levels(tau, condition, tau_eps):
        tau = torch.as_tensor(tau, device=condition.device, dtype=condition.dtype)
        batch_size = condition.shape[0]
        if tau.ndim == 0:
            tau = tau.reshape(1, 1, 1).expand(batch_size, 1, 1)
        elif tau.ndim == 1:
            tau = tau.reshape(1, -1, 1).expand(batch_size, -1, -1)
        elif tau.ndim == 2:
            if tau.shape[0] != batch_size:
                raise ValueError("A two-dimensional tau must have shape [B, K].")
            tau = tau.unsqueeze(-1)
        elif tau.ndim == 3:
            if tau.shape[0] != batch_size or tau.shape[-1] != 1:
                raise ValueError("A three-dimensional tau must have shape [B, K, 1].")
        else:
            raise ValueError("tau must be scalar, [K], [B, K], or [B, K, 1].")
        return tau.clamp(float(tau_eps), 1.0 - float(tau_eps))

    def critical_quantile_decision(self, condition):
        return self.quantile(self.target_quantile, condition)[:, 0, :]

    def exact_newsvendor_loss(self, condition, demand, reduction="mean"):
        """Unsmoothed newsvendor loss at the exact spline quantile."""
        if demand.ndim == 1:
            demand = demand.unsqueeze(-1)
        decision = self.critical_quantile_decision(condition)
        point_losses = (
            self.cu * torch.relu(demand - decision)
            + self.co * torch.relu(decision - demand)
        )
        if reduction == "none":
            return point_losses, decision
        if reduction == "sum":
            return point_losses.sum(), decision
        if reduction == "mean":
            return point_losses.mean(), decision
        raise ValueError("reduction must be 'none', 'sum', or 'mean'.")

    def evaluate_exact_newsvendor(self, data_loader):
        """Evaluate exact-quantile unsmoothed cost and NLL by observation."""
        self.eval()
        total_newsvendor = 0.0
        total_nll = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in data_loader:
                condition, target = self._split_spline_batch(batch)
                point_losses, _ = self.exact_newsvendor_loss(
                    condition,
                    target,
                    reduction="none",
                )
                batch_count = int(target.numel())
                total_newsvendor += float(point_losses.sum())
                total_nll += float(-self.backbone.log_prob(target, condition).sum())
                total_count += batch_count
        denominator = max(total_count, 1)
        return {
            "newsvendor_loss": total_newsvendor / denominator,
            "nll": total_nll / denominator,
            "count": total_count,
        }

    def _split_spline_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2 and isinstance(batch[1], torch.Tensor) and batch[1].is_floating_point():
                condition, target = batch
                condition = condition.to(self._device())
                target = target.to(self._device())
                if target.ndim == 1:
                    target = target.unsqueeze(1)
                return condition, target
            batch = batch[0]
        return self._split_batch(batch, targetdim=1)

    def train_spline_nll(
        self,
        train_loader,
        val_loader,
        *,
        num_epochs=None,
        learning_rate=1e-3,
        early_stopping=10,
        checkpoint_path=None,
        verbose=False,
    ):
        num_epochs = self.epoch if num_epochs is None else int(num_epochs)
        early_stopping = int(early_stopping)
        if num_epochs < 1 or early_stopping < 1 or learning_rate <= 0:
            raise ValueError("Training arguments must be positive.")
        optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        history = {"epoch": [], "train_nll": [], "val_nll": [], "best_epoch": -1}
        best_value = float("inf")
        best_state = None
        patience = 0
        for epoch in range(num_epochs):
            self.train()
            train_losses = []
            for batch in train_loader:
                condition, target = self._split_spline_batch(batch)
                loss = self.generative_loss(target, condition)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite spline NLL encountered.")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach()))

            self.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    condition, target = self._split_spline_batch(batch)
                    val_losses.append(float(self.generative_loss(target, condition)))
            train_nll = float(np.mean(train_losses)) if train_losses else float("inf")
            val_nll = float(np.mean(val_losses)) if val_losses else float("inf")
            history["epoch"].append(epoch)
            history["train_nll"].append(train_nll)
            history["val_nll"].append(val_nll)
            if verbose:
                print(f"epoch={epoch} train_nll={train_nll:.6f} val_nll={val_nll:.6f}")
            if np.isfinite(val_nll) and val_nll < best_value:
                best_value = val_nll
                history["best_epoch"] = epoch
                best_state = copy.deepcopy(self.state_dict())
                patience = 0
                if checkpoint_path is not None:
                    checkpoint_path = Path(checkpoint_path)
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, checkpoint_path)
            else:
                patience += 1
                if patience >= early_stopping:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        history["best_val_nll"] = best_value
        history["epochs_ran"] = len(history["epoch"])
        history["checkpoint_path"] = str(checkpoint_path) if checkpoint_path else None
        return history

    def train_gendfl_spline(
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
        checkpoint_path=None,
        verbose=False,
        epoch_callback=None,
    ):
        """Fit the GenDFL conditional generator using NLL only.

        Exact unsmoothed newsvendor loss is evaluated only to implement the
        shared early-stopping protocol. It is never part of the training
        objective or gradient.
        """
        num_epochs = self.epoch if num_epochs is None else int(num_epochs)
        early_stopping = int(early_stopping)
        warmup_epochs = int(warmup_epochs)
        min_delta_relative = float(min_delta_relative)
        if min(num_epochs, early_stopping) < 1 or learning_rate <= 0:
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
        else:
            steps_per_epoch = len(train_loader)
            batch_rng = None
        history = {
            "epoch": [],
            "train_nll": [],
            "val_nll": [],
            "val_newsvendor": [],
            "best_epoch": -1,
            "training_objective": "conditional_nll_only",
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
        best_nll_at_newsvendor = float("inf")
        best_state = None
        patience = 0
        for epoch in range(num_epochs):
            self.train()
            train_nll = []
            epoch_step_sizes = []
            batches = train_loader if optimizer is not None else range(steps_per_epoch)
            for batch in batches:
                if optimizer is not None:
                    condition, target = self._split_spline_batch(batch)
                else:
                    indices = torch.randperm(sample_count, generator=batch_rng)[:batch_size]
                    condition = train_context.index_select(0, indices).to(self._device())
                    target = train_demand.index_select(0, indices).to(self._device())
                nll = self.generative_loss(target, condition)
                if not torch.isfinite(nll):
                    raise FloatingPointError("Non-finite GenDFL spline NLL encountered.")
                for parameter in parameters:
                    parameter.grad = None
                nll.backward()
                for parameter in self.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Non-finite GenDFL spline gradient encountered.")
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
                train_nll.append(float(nll.detach()))

            train_value = float(np.mean(train_nll)) if train_nll else float("inf")
            history["epoch"].append(epoch)
            history["train_nll"].append(train_value)
            history["step_size_first"].append(
                epoch_step_sizes[0] if epoch_step_sizes else float(learning_rate)
            )
            history["step_size_last"].append(
                epoch_step_sizes[-1] if epoch_step_sizes else float(learning_rate)
            )
            if epoch < warmup_epochs:
                history["val_nll"].append(float("nan"))
                history["val_newsvendor"].append(float("nan"))
                continue

            validation = self.evaluate_exact_newsvendor(val_loader)
            val_newsvendor = validation["newsvendor_loss"]
            val_nll = validation["nll"]
            history["val_nll"].append(val_nll)
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
                    f"epoch={epoch} train_nll={train_value:.6f} "
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
                best_nll_at_newsvendor = val_nll
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
        history["best_val_nll"] = best_nll_at_newsvendor
        history["epochs_ran"] = len(history["epoch"])
        history["steps_ran"] = (
            global_step if optimizer is None else len(history["epoch"]) * steps_per_epoch
        )
        history["checkpoint_path"] = str(checkpoint_path) if checkpoint_path else None
        return history


class GenDFLSplineNewsvendor(SplineConditionalNewsvendorBase):
    """Shared spline conditional generator trained only by likelihood."""
