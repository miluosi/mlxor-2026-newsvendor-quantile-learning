"""Affine quantile-flow regression with the GenDFL ConditionalFlow backbone."""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.newsvendor_gendfl_conditional_flow import GenDFLConditionalFlowNewsvendor


def pinball_loss(target, quantile, tau):
    """Elementwise pinball loss with broadcast-compatible tensors."""
    residual = target - quantile
    return torch.maximum(tau * residual, (tau - 1.0) * residual)


class AffineQuantileFlowNewsvendor(GenDFLConditionalFlowNewsvendor):
    """Integrated-pinball benchmark using exactly the GenDFL MLP backbone.

    The class adds no trainable parameters. The final two backbone outputs are
    interpreted as location and raw scale, as specified by the benchmark plan.
    """

    def __init__(self, *args, sigma_min=1e-4, tau_eps=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_min = float(sigma_min)
        self.tau_eps = float(tau_eps)
        if self.sigma_min <= 0:
            raise ValueError("sigma_min must be positive.")
        if not 0.0 < self.tau_eps < 0.5:
            raise ValueError("tau_eps must lie strictly between zero and 0.5.")

    def location_scale(self, condition):
        output = self.flow.net(condition)
        location = output[:, 0:1]
        scale = self.sigma_min + F.softplus(output[:, 1:2])
        return location, scale

    def decode(self, z, condition):
        location, scale = self.location_scale(condition)
        return location + scale * z[:, :1]

    def generative_loss(self, y_true, condition):
        """Implied Gaussian NLL, provided only as a distributional metric."""
        location, scale = self.location_scale(condition)
        standardized = (y_true - location) / scale
        return (
            0.5 * standardized.pow(2)
            + torch.log(scale)
            + 0.5 * np.log(2.0 * np.pi)
        ).mean()

    def quantile(self, tau, condition, tau_eps=None):
        tau_eps = self.tau_eps if tau_eps is None else float(tau_eps)
        tau = self._prepare_quantile_levels(tau, condition, tau_eps)
        location, scale = self.location_scale(condition)
        standard_normal = torch.distributions.Normal(
            condition.new_zeros(()),
            condition.new_ones(()),
        )
        base_quantile = standard_normal.icdf(tau)
        return location[:, None, :] + scale[:, None, :] * base_quantile

    def integrated_pinball_loss(
        self,
        target,
        condition,
        num_quantile_levels=32,
        tau=None,
    ):
        """Monte Carlo integrated pinball loss over quantile levels."""
        if target.ndim == 1:
            target = target.unsqueeze(1)
        if tau is None:
            num_quantile_levels = int(num_quantile_levels)
            if num_quantile_levels < 1:
                raise ValueError("num_quantile_levels must be positive.")
            tau = torch.rand(
                condition.shape[0],
                num_quantile_levels,
                1,
                device=condition.device,
                dtype=condition.dtype,
            )
            tau = self.tau_eps + (1.0 - 2.0 * self.tau_eps) * tau
        else:
            tau = self._prepare_quantile_levels(tau, condition, self.tau_eps)
        predicted_quantiles = self.quantile(tau, condition)
        losses = pinball_loss(target[:, None, :], predicted_quantiles, tau)
        return {
            "loss": losses.mean(),
            "tau": tau,
            "quantiles": predicted_quantiles,
            "point_losses": losses,
        }

    def critical_quantile_decision(self, condition):
        return self.quantile(self.target_quantile, condition)[:, 0, :]

    def train_quantile_flow(
        self,
        train_loader,
        val_loader,
        *,
        num_epochs=None,
        learning_rate=1e-3,
        early_stopping=10,
        num_quantile_levels=32,
        validation_quantile_levels=99,
        max_grad_norm=1.0,
        checkpoint_path=None,
        verbose=False,
    ):
        """Train by integrated pinball loss and select by validation QFR loss."""
        num_epochs = self.epoch if num_epochs is None else int(num_epochs)
        early_stopping = int(early_stopping)
        num_quantile_levels = int(num_quantile_levels)
        validation_quantile_levels = int(validation_quantile_levels)
        if min(num_epochs, early_stopping, num_quantile_levels, validation_quantile_levels) < 1:
            raise ValueError("Epoch, patience, and quantile-level counts must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        validation_tau = torch.linspace(
            self.tau_eps,
            1.0 - self.tau_eps,
            validation_quantile_levels,
            device=self._device(),
        )
        history = {
            "epoch": [],
            "train_qfr_loss": [],
            "val_qfr_loss": [],
            "val_critical_newsvendor_loss": [],
            "best_epoch": -1,
        }
        best_value = float("inf")
        best_state = None
        patience = 0

        for epoch in range(num_epochs):
            self.train()
            train_losses = []
            for batch in train_loader:
                condition, target = self._split_conditional_flow_batch(batch)
                qfr_loss = self.integrated_pinball_loss(
                    target,
                    condition,
                    num_quantile_levels=num_quantile_levels,
                )["loss"]
                optimizer.zero_grad()
                qfr_loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        max_norm=float(max_grad_norm),
                    )
                optimizer.step()
                train_losses.append(float(qfr_loss.detach()))

            self.eval()
            val_qfr_losses = []
            val_newsvendor_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    condition, target = self._split_conditional_flow_batch(batch)
                    val_qfr_losses.append(
                        float(
                            self.integrated_pinball_loss(
                                target,
                                condition,
                                tau=validation_tau,
                            )["loss"]
                        )
                    )
                    decision = self.critical_quantile_decision(condition)
                    val_newsvendor_losses.append(float(self.newsvendor_loss(decision, target)))
            train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
            val_loss = float(np.mean(val_qfr_losses)) if val_qfr_losses else float("inf")
            val_newsvendor = (
                float(np.mean(val_newsvendor_losses))
                if val_newsvendor_losses
                else float("inf")
            )
            history["epoch"].append(epoch)
            history["train_qfr_loss"].append(train_loss)
            history["val_qfr_loss"].append(val_loss)
            history["val_critical_newsvendor_loss"].append(val_newsvendor)
            if verbose:
                print(
                    f"epoch={epoch} train_qfr={train_loss:.6f} "
                    f"val_qfr={val_loss:.6f} val_nv={val_newsvendor:.6f}"
                )

            if np.isfinite(val_loss) and val_loss < best_value:
                best_value = val_loss
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
        history["best_val_qfr_loss"] = best_value
        history["epochs_ran"] = len(history["epoch"])
        history["checkpoint_path"] = (
            str(checkpoint_path) if checkpoint_path is not None else None
        )
        return history
