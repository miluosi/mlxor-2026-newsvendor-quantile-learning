import torch
import torch.nn as nn

from model.generative_newsvendor_base import GenerativeNewsvendorBase


class _Encoder(nn.Module):
    def __init__(self, targetdim, labeldim, latent, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(targetdim + labeldim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent)
        self.logvar = nn.Linear(hidden_dim, latent)

    def forward(self, y, condition):
        h = self.net(torch.cat([y, condition], dim=1))
        return self.mu(h), self.logvar(h)


class _Decoder(nn.Module):
    def __init__(self, targetdim, labeldim, latent, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent + labeldim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, targetdim),
        )

    def forward(self, z, condition):
        return self.net(torch.cat([z, condition], dim=1))


class ConditionalVAENewsvendor(GenerativeNewsvendorBase):
    def __init__(
        self,
        targetdim,
        labeldim,
        latent,
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
        self.encoder = _Encoder(targetdim, labeldim, latent, hidden_dim)
        self.decoder = _Decoder(targetdim, labeldim, latent, hidden_dim)
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
        return self.decoder(z, condition)

    def generative_loss(self, y_true, condition):
        mu, logvar = self.encoder(y_true, condition)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z, condition)
        recon_loss = torch.nn.functional.mse_loss(recon, y_true, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld
