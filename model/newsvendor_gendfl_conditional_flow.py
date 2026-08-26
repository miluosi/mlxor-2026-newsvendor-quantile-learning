import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.generative_newsvendor_base import GenerativeNewsvendorBase


class ConditionalFlow(nn.Module):
    """Exact scalar ConditionalFlow from gen_dfl-main."""

    def __init__(self, c_dim, x_dim):
        super().__init__()
        hidden = 32
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, c, x):
        out = self.net(x)
        mu = out[:, 0:1]
        log_var = out[:, 1:2]
        sigma = torch.exp(0.5 * log_var)
        z = (c - mu) / (sigma + 1e-8)
        log_det = -0.5 * log_var.squeeze(-1)
        return z, log_det

    def sample(self, num_samples, x):
        with torch.no_grad():
            out = self.net(x)
            mu = out[:, 0:1]
            log_var = out[:, 1:2]
            sigma = torch.exp(0.5 * log_var)
            batch_size = x.shape[0]
            z = torch.randn(batch_size, num_samples, 1, device=x.device)
            mu_expand = mu.unsqueeze(1)
            sigma_expand = sigma.unsqueeze(1)
            c = mu_expand + sigma_expand * z
        return c.squeeze(-1)


GenDFLConditionalFlow = ConditionalFlow


def pretrain_flow(flow_model, loader, num_epochs=10, lr=1e-3, device="cpu"):
    """Exact pure-NLL training loop from gen_dfl-main."""
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=lr)
    flow_model.to(device)
    nll_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, c_batch in loader:
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)
            z, log_det = flow_model(c_batch, x_batch)
            log_prob = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * z.size(1) * np.log(2 * np.pi)
            loss = -(log_prob + log_det).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        nll_losses.append(avg_loss)
        print(f"[Pretrain] Epoch {epoch+1}/{num_epochs}, NLL: {avg_loss:.4f}")
    return nll_losses


class GenDFLConditionalFlowNewsvendor(GenerativeNewsvendorBase):
    """Adapt the scalar gen_dfl conditional flow to newsvendor IPA/GLR."""

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
        hidden_dim=32,
    ):
        super().__init__()
        if int(targetdim) != 1:
            raise ValueError("GenDFLConditionalFlowNewsvendor requires targetdim=1.")
        if int(latent) != 1:
            raise ValueError("The scalar gen_dfl conditional flow requires latent=1.")
        if int(hidden_dim) != 32:
            raise ValueError("Exact gen_dfl ConditionalFlow requires hidden_dim=32.")
        self.flow = ConditionalFlow(c_dim=1, x_dim=labeldim)
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
        out = self.flow.net(condition)
        mu = out[:, 0:1]
        log_var = out[:, 1:2]
        sigma = torch.exp(0.5 * log_var)
        return mu + sigma * z[:, :1]

    def quantile(self, tau, condition, tau_eps=1e-4):
        """Return the exact affine-Gaussian quantile with shape [B, K, 1]."""
        out = self.flow.net(condition)
        mu = out[:, 0:1]
        log_var = out[:, 1:2]
        sigma = torch.exp(0.5 * log_var)
        tau = self._prepare_quantile_levels(tau, condition, tau_eps)
        standard_normal = torch.distributions.Normal(
            condition.new_zeros(()),
            condition.new_ones(()),
        )
        return mu[:, None, :] + sigma[:, None, :] * standard_normal.icdf(tau)

    @staticmethod
    def _prepare_quantile_levels(tau, condition, tau_eps):
        tau_eps = float(tau_eps)
        if not 0.0 < tau_eps < 0.5:
            raise ValueError("tau_eps must lie strictly between zero and 0.5.")
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
            if tau.shape[0] != batch_size or tau.shape[2] != 1:
                raise ValueError("A three-dimensional tau must have shape [B, K, 1].")
        else:
            raise ValueError("tau must be scalar, [K], [B, K], or [B, K, 1].")
        return tau.clamp(tau_eps, 1.0 - tau_eps)

    def generative_loss(self, y_true, condition):
        z, log_det = self.flow(y_true, condition)
        log_prob = -0.5 * torch.sum(z.pow(2), dim=1)
        log_prob = log_prob - 0.5 * z.shape[1] * np.log(2.0 * np.pi)
        return -(log_prob + log_det).mean()

    def sample(self, num_samples, condition):
        """Expose the gen_dfl sample(num_samples, x) interface."""
        return self.flow.sample(num_samples, condition)

    def transform_to_noise(self, target, condition):
        """Expose gen_dfl forward(c, x) without changing decode-based inference."""
        return self.flow(target, condition)

    def _split_conditional_flow_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2 and not (
                isinstance(batch[1], torch.Tensor)
                and batch[1].dtype
                in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
            ):
                condition, target = batch
                condition = condition.to(self._device())
                target = target.to(self._device())
                if target.ndim == 1:
                    target = target.unsqueeze(1)
                return condition, target
            batch = batch[0]
        return self._split_batch(batch, targetdim=1)

    def train_conditional_flow(
        self,
        train_loader,
        val_loader=None,
        num_epochs=None,
        learning_rate=1e-3,
        early_stopping=None,
        checkpoint_path=None,
        verbose=False,
    ):
        """Train only p(target | condition) with conditional-flow NLL."""
        num_epochs = self.epoch if num_epochs is None else int(num_epochs)
        if num_epochs < 1 or learning_rate <= 0:
            raise ValueError("num_epochs and learning_rate must be positive.")
        if early_stopping is not None and int(early_stopping) < 1:
            raise ValueError("early_stopping must be positive when provided.")

        optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        history = {
            "epoch": [],
            "train_nll": [],
            "val_nll": [],
            "best_epoch": -1,
        }
        best_loss = float("inf")
        best_state = None
        patience = 0

        for epoch in range(num_epochs):
            self.train()
            train_losses = []
            for batch in train_loader:
                condition, target = self._split_conditional_flow_batch(batch)
                nll = self.generative_loss(target, condition)
                optimizer.zero_grad()
                nll.backward()
                optimizer.step()
                train_losses.append(float(nll.detach()))

            train_nll = float(np.mean(train_losses)) if train_losses else float("inf")
            if val_loader is None:
                val_nll = train_nll
            else:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        condition, target = self._split_conditional_flow_batch(batch)
                        val_losses.append(float(self.generative_loss(target, condition)))
                val_nll = float(np.mean(val_losses)) if val_losses else float("inf")

            history["epoch"].append(epoch)
            history["train_nll"].append(train_nll)
            history["val_nll"].append(val_nll)
            if verbose:
                print(f"epoch={epoch} train_nll={train_nll:.6f} val_nll={val_nll:.6f}")

            if np.isfinite(val_nll) and val_nll < best_loss:
                best_loss = val_nll
                history["best_epoch"] = epoch
                best_state = copy.deepcopy(self.state_dict())
                patience = 0
                if checkpoint_path is not None:
                    checkpoint_path = Path(checkpoint_path)
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, checkpoint_path)
            else:
                patience += 1
                if early_stopping is not None and patience >= int(early_stopping):
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        history["best_val_nll"] = best_loss
        history["epochs_ran"] = len(history["epoch"])
        history["checkpoint_path"] = str(checkpoint_path) if checkpoint_path is not None else None
        return history

    def pretrain_flow(self, *args, **kwargs):
        """Alias matching the gen_dfl pretraining terminology."""
        return self.train_conditional_flow(*args, **kwargs)
