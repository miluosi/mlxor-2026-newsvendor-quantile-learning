import torch
import torch.nn as nn

from model.generative_newsvendor_base import GenerativeNewsvendorBase


class _ConditionalAffineFlow(nn.Module):
    def __init__(self, targetdim, labeldim, hidden_dim):
        super().__init__()
        self.shift = nn.Sequential(
            nn.Linear(labeldim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, targetdim),
        )
        self.log_scale = nn.Sequential(
            nn.Linear(labeldim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, targetdim),
        )

    def params(self, condition):
        shift = self.shift(condition)
        log_scale = torch.clamp(self.log_scale(condition), -5.0, 5.0)
        return shift, log_scale


class ConditionalRealNVPNewsvendor(GenerativeNewsvendorBase):
    """Conditional affine RealNVP layer for scalar/multivariate newsvendor targets."""

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
        self.flow = _ConditionalAffineFlow(targetdim, labeldim, hidden_dim)
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
        shift, log_scale = self.flow.params(condition)
        return shift + torch.exp(log_scale) * z

    def generative_loss(self, y_true, condition):
        shift, log_scale = self.flow.params(condition)
        z = (y_true - shift) * torch.exp(-log_scale)
        log_prob = -0.5 * (z.pow(2) + torch.log(y_true.new_tensor(2.0 * torch.pi)))
        log_prob = log_prob.sum(dim=1) - log_scale.sum(dim=1)
        return -log_prob.mean()
