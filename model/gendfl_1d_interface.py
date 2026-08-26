"""Public training interface for the scalar gen_dfl conditional flow.

This module uses the source-aligned scalar ``ConditionalFlow`` together with
the project's regularized newsvendor trainers. It intentionally supports only
one-dimensional targets.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.newsvendor_gendfl_conditional_flow import (
    GenDFLConditionalFlowNewsvendor,
    pretrain_flow,
)


SUPPORTED_METHODS = {"nll", "ipa", "glr"}


def _as_float_tensor(value, name):
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains a non-finite value.")
    return tensor


def _validate_xy(x, y, name):
    x = _as_float_tensor(x, f"{name} x")
    y = _as_float_tensor(y, f"{name} y")
    if x.ndim != 2:
        raise ValueError(f"{name} x must have shape [N, feature_dim].")
    if y.ndim == 1:
        y = y.unsqueeze(1)
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"{name} y must have shape [N] or [N, 1].")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{name} x and y must contain the same number of rows.")
    if x.shape[0] == 0:
        raise ValueError(f"{name} data must not be empty.")
    return x, y


def _regularized_loader(x, y, batch_size, shuffle):
    combined = torch.cat([x, y], dim=1)
    indices = torch.arange(len(combined), dtype=torch.long)
    return DataLoader(
        TensorDataset(combined, indices),
        batch_size=min(int(batch_size), len(combined)),
        shuffle=bool(shuffle),
    )


def _validation_loader(x, y, batch_size):
    combined = torch.cat([x, y], dim=1)
    return DataLoader(
        TensorDataset(combined),
        batch_size=min(int(batch_size), len(combined)),
        shuffle=False,
    )


def make_gendfl_1d_model(
    feature_dim,
    data_len,
    cost_under,
    cost_over,
    *,
    target_quantile=None,
    epochs=100,
    regularization_lambda=0.5,
    sampling_number=100,
    glr_inner_steps=1,
    random_seed=0,
):
    """Construct the source-aligned scalar flow with newsvendor adapters."""
    feature_dim = int(feature_dim)
    cost_under = float(cost_under)
    cost_over = float(cost_over)
    if feature_dim < 1:
        raise ValueError("feature_dim must be positive.")
    if cost_under <= 0 or cost_over <= 0:
        raise ValueError("cost_under and cost_over must be positive.")
    if target_quantile is None:
        target_quantile = cost_under / (cost_under + cost_over)
    target_quantile = float(target_quantile)
    if not 0.0 < target_quantile < 1.0:
        raise ValueError("target_quantile must lie strictly between zero and one.")

    return GenDFLConditionalFlowNewsvendor(
        targetdim=1,
        labeldim=feature_dim,
        latent=1,
        data_len=int(data_len),
        epoch=int(epochs),
        quantiles=target_quantile,
        target_quantile=target_quantile,
        lambda1=float(regularization_lambda),
        lambda_gradient=float(regularization_lambda),
        samplingnumber=int(sampling_number),
        cost_under=cost_under,
        cost_over=cost_over,
        random_seed=int(random_seed),
        innerloop=int(glr_inner_steps),
        hidden_dim=32,
    )


def train_gendfl_1d(
    x_train,
    y_train,
    x_val=None,
    y_val=None,
    *,
    method="ipa",
    cost_under=7.0,
    cost_over=3.0,
    target_quantile=None,
    regularization_lambda=0.5,
    epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    early_stopping=10,
    sampling_number=100,
    ipa_replicates=8,
    glr_inner_steps=1,
    use_vmap=True,
    vmap_chunk_size=None,
    max_grad_norm=1.0,
    shuffle=True,
    random_seed=0,
    device=None,
    checkpoint_path=None,
    verbose=False,
):
    """Train scalar gen_dfl by NLL, NLL+IPA, or NLL+GLR.

    ``method='ipa'`` optimizes
    ``NLL + regularization_lambda * IPA_newsvendor_loss``.
    ``method='glr'`` applies
    ``grad(NLL) + regularization_lambda * estimated_GLR_gradient``.
    ``method='nll'`` calls the source-aligned pure-NLL pretraining loop.

    Returns:
        A pair ``(model, history)``. The model's inference interface is unchanged;
        use :func:`predict_gendfl_1d_quantile` for a sampled quantile decision.
    """
    method = str(method).lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {sorted(SUPPORTED_METHODS)}.")
    if int(epochs) < 1 or int(batch_size) < 1:
        raise ValueError("epochs and batch_size must be positive integers.")
    if float(learning_rate) <= 0:
        raise ValueError("learning_rate must be positive.")

    x_train, y_train = _validate_xy(x_train, y_train, "training")
    if (x_val is None) != (y_val is None):
        raise ValueError("x_val and y_val must either both be provided or both be omitted.")
    if x_val is None:
        x_val, y_val = x_train, y_train
    else:
        x_val, y_val = _validate_xy(x_val, y_val, "validation")
        if x_val.shape[1] != x_train.shape[1]:
            raise ValueError("Training and validation feature dimensions must match.")

    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = make_gendfl_1d_model(
        feature_dim=x_train.shape[1],
        data_len=len(x_train),
        cost_under=cost_under,
        cost_over=cost_over,
        target_quantile=target_quantile,
        epochs=epochs,
        regularization_lambda=regularization_lambda,
        sampling_number=sampling_number,
        glr_inner_steps=glr_inner_steps,
        random_seed=random_seed,
    ).to(device)

    if method == "nll":
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=min(int(batch_size), len(x_train)),
            shuffle=bool(shuffle),
        )
        nll_losses = pretrain_flow(
            model.flow,
            loader,
            num_epochs=int(epochs),
            lr=float(learning_rate),
            device=device,
        )
        history = {
            "method": "nll",
            "nll_loss": nll_losses,
            "epochs_ran": len(nll_losses),
            "checkpoint_path": None,
        }
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            history["checkpoint_path"] = str(checkpoint_path)
        return model, history

    train_loader = _regularized_loader(x_train, y_train, batch_size, shuffle)
    val_loader = _validation_loader(x_val, y_val, batch_size)
    common_arguments = {
        "traindata_loader": train_loader,
        "valdata_loader": val_loader,
        "num_epochs": int(epochs),
        "early_stopping": int(early_stopping),
        "regularization_lambda": float(regularization_lambda),
        "learning_rate": float(learning_rate),
        "num_samples": int(sampling_number),
        "use_vmap": bool(use_vmap),
        "vmap_chunk_size": vmap_chunk_size,
        "max_grad_norm": max_grad_norm,
        "checkpoint_path": checkpoint_path,
        "verbose": bool(verbose),
    }
    if method == "ipa":
        history = model.train_regularized_ipa(
            k=int(ipa_replicates),
            **common_arguments,
        )
    else:
        history = model.train_regularized_glr(
            glr_inner_steps=int(glr_inner_steps),
            **common_arguments,
        )
    return model, history


def predict_gendfl_1d_quantile(model, x, num_samples=1000):
    """Return one sampled critical-quantile decision for every input row."""
    x = _as_float_tensor(x, "inference x")
    if x.ndim != 2 or x.shape[1] != model.labeldim:
        raise ValueError(f"x must have shape [N, {model.labeldim}].")
    if int(num_samples) < 1:
        raise ValueError("num_samples must be positive.")
    model.eval()
    with torch.no_grad():
        condition = x.to(model._device())
        decision = model.sample_quantile_decision(
            condition,
            num_samples=int(num_samples),
            requires_grad=False,
        )
    return decision.cpu()
