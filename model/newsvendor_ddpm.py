import torch
import torch.nn as nn

from model.generative_newsvendor_base import GenerativeNewsvendorBase


class _EpsilonNet(nn.Module):
    def __init__(self, targetdim, labeldim, n_steps, hidden_dim):
        super().__init__()
        self.time_embed = nn.Embedding(n_steps, hidden_dim)
        self.input = nn.Linear(targetdim + labeldim, hidden_dim)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, targetdim),
        )

    def forward(self, x_t, condition, t_idx):
        h = self.input(torch.cat([x_t, condition], dim=1)) + self.time_embed(t_idx)
        return self.net(h)


class ConditionalDDPMNewsvendor(GenerativeNewsvendorBase):
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
        T=20,
        beta_1=1e-4,
        beta_T=0.02,
    ):
        super().__init__()
        latent = int(latent or targetdim)
        self.T = int(T)
        self.eps_model = _EpsilonNet(targetdim, labeldim, self.T, hidden_dim)
        betas = torch.linspace(beta_1, beta_T, self.T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_prev_bars = torch.cat([torch.ones(1), alpha_bars[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_prev_bars", alpha_prev_bars)
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
            x = z[:, :self.targetdim]
        else:
            x = z
        for idx in reversed(range(self.T)):
            t_idx = torch.full((x.shape[0],), idx, device=x.device, dtype=torch.long)
            eps = self.eps_model(x, condition, t_idx)
            x = torch.sqrt(1.0 / self.alphas[idx]) * (
                x - self.betas[idx] / torch.sqrt(1.0 - self.alpha_bars[idx]) * eps
            )
        return x

    def generative_loss(self, y_true, condition):
        t_idx = torch.randint(0, self.T, (y_true.shape[0],), device=y_true.device)
        alpha_bar = self.alpha_bars[t_idx].view(-1, 1)
        eps = torch.randn_like(y_true)
        x_t = torch.sqrt(alpha_bar) * y_true + torch.sqrt(1.0 - alpha_bar) * eps
        eps_pred = self.eps_model(x_t, condition, t_idx)
        return torch.nn.functional.mse_loss(eps_pred, eps)
