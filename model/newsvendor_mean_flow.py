import torch
import torch.nn as nn

from model.generative_newsvendor_base import GenerativeNewsvendorBase


class _VelocityNet(nn.Module):
    def __init__(self, targetdim, labeldim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(targetdim + labeldim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, targetdim),
        )

    def forward(self, x_t, condition, t):
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([x_t, condition, t], dim=1))


class ConditionalMeanFlowNewsvendor(GenerativeNewsvendorBase):
    """One-step conditional mean/rectified-flow model."""

    def __init__(
        self,
        targetdim,
        labeldim,
        latent=None,
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
        hidden_dim=64,
    ):
        super().__init__()
        latent = int(latent or targetdim)
        self.velocity = _VelocityNet(targetdim, labeldim, hidden_dim)
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
        if z.shape[1] != self.targetdim:
            z = z[:, :self.targetdim]
        t0 = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        return z + self.velocity(z, condition, t0)

    def generative_loss(self, y_true, condition):
        z0 = torch.randn_like(y_true)
        t = torch.rand(y_true.shape[0], 1, device=y_true.device, dtype=y_true.dtype)
        x_t = (1.0 - t) * z0 + t * y_true
        target_velocity = y_true - z0
        pred_velocity = self.velocity(x_t, condition, t)
        return torch.nn.functional.mse_loss(pred_velocity, target_velocity)
