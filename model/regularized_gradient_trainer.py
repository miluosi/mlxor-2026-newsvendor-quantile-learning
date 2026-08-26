import copy
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import RandomSampler


class RegularizedGenerativeTrainer:
    """Jointly optimize a conditional generator with IPA or GLR regularization.

    The model must provide the interface implemented by
    ``GenerativeNewsvendorBase``. This trainer intentionally supports only the
    regularized update used by gen_dfl-style training:

        generator gradient + regularization_lambda * decision gradient.

    It does not implement generator-only, IPA-only, GLR-only, or two-optimizer
    update modes.
    """

    METHODS = {"ipa", "glr"}

    def __init__(
        self,
        model,
        method,
        regularization_lambda,
        learning_rate=1e-3,
        use_vmap=True,
        k=8,
        num_samples=None,
        vmap_chunk_size=None,
        glr_inner_steps=None,
        max_grad_norm=1.0,
    ):
        method = str(method).lower()
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {sorted(self.METHODS)}.")
        if getattr(model, "targetdim", None) != 1:
            raise ValueError("The regularized IPA/GLR trainer currently requires targetdim=1.")
        if regularization_lambda < 0:
            raise ValueError("regularization_lambda must be nonnegative.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.model = model
        self.method = method
        self.regularization_lambda = float(regularization_lambda)
        self.learning_rate = float(learning_rate)
        self.use_vmap = bool(use_vmap)
        self.k = int(k)
        self.num_samples = int(num_samples or model.samplingnumber)
        self.vmap_chunk_size = vmap_chunk_size
        self.glr_inner_steps = glr_inner_steps
        self.max_grad_norm = max_grad_norm
        if self.k < 1 or self.num_samples < 1:
            raise ValueError("k and num_samples must be positive integers.")

    @staticmethod
    def _looks_like_indices(value):
        return (
            isinstance(value, torch.Tensor)
            and value.ndim <= 1
            and value.dtype in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
        )

    def _unpack_batch(self, batch, batch_idx, batch_size):
        explicit_indices = False
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2 and self._looks_like_indices(batch[1]):
                data, global_indices = batch
                global_indices = batch[1].detach().cpu().numpy()
                explicit_indices = True
            elif len(batch) == 1:
                data = batch[0]
                global_indices = None
            else:
                raise ValueError(
                    "Expected a combined [condition, target] tensor or a "
                    "(combined_tensor, integer_index) batch."
                )
        else:
            data = batch
            global_indices = None

        if global_indices is None:
            start = batch_idx * batch_size
            global_indices = np.arange(start, start + data.shape[0])
        return data, global_indices, explicit_indices

    def _ipa_result(self, condition, y_true):
        return self.model.batched_ipa_regularizer(
            condition,
            y_true,
            k=self.k,
            num_samples=self.num_samples,
            use_vmap=self.use_vmap,
            vmap_chunk_size=self.vmap_chunk_size,
        )

    def _set_combined_glr_gradient(self, generative_loss, glr_result, optimizer):
        named_parameters = self.model.generation_named_parameters()
        parameters = list(named_parameters.values())
        generative_gradients_raw = torch.autograd.grad(
            generative_loss,
            parameters,
            create_graph=False,
            allow_unused=True,
        )
        generative_gradients = OrderedDict(
            (
                name,
                gradient.detach() if gradient is not None else torch.zeros_like(parameter),
            )
            for (name, parameter), gradient in zip(named_parameters.items(), generative_gradients_raw)
        )

        optimizer.zero_grad()
        for name, parameter in named_parameters.items():
            parameter.grad = (
                generative_gradients[name]
                + self.regularization_lambda * glr_result["gradient"][name]
            ).clone()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=float(self.max_grad_norm))
        gradient_norm = torch.sqrt(
            sum(torch.sum(parameter.grad.detach().pow(2)) for parameter in parameters if parameter.grad is not None)
        )
        optimizer.step()
        return float(gradient_norm)

    def _validation_metrics(self, val_loader):
        self.model.eval()
        generative_losses = []
        regularizer_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                condition, y_true = self.model._split_batch(data, self.model.targetdim)
                generative_losses.append(float(self.model.generative_loss(y_true, condition)))
                regularizer_losses.append(float(self._ipa_result(condition, y_true)["loss"]))
        self.model.train()
        generative_loss = float(np.mean(generative_losses)) if generative_losses else float("inf")
        regularizer_loss = float(np.mean(regularizer_losses)) if regularizer_losses else float("inf")
        return generative_loss, regularizer_loss

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs,
        early_stopping=10,
        checkpoint_path=None,
        verbose=False,
    ):
        """Train with one joint regularized update per minibatch."""
        num_epochs = int(num_epochs)
        early_stopping = int(early_stopping)
        if num_epochs < 1 or early_stopping < 1:
            raise ValueError("num_epochs and early_stopping must be positive integers.")

        dataset_len = len(train_loader.dataset)
        batch_size = int(train_loader.batch_size or dataset_len)
        if self.method == "glr":
            self.model.data_len = dataset_len
            self.model._reset_glr_state(dataset_len)
            self.model.k_step = 1.0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        history = {
            "epoch": [],
            "generative_loss": [],
            "regularizer_loss": [],
            "total_loss": [],
            "val_generative_loss": [],
            "val_regularizer_loss": [],
            "val_total_loss": [],
            "combined_gradient_norm": [],
            "gradient_seconds": [],
            "best_epoch": -1,
            "method": self.method,
            "use_vmap": self.use_vmap,
            "regularization_lambda": self.regularization_lambda,
        }
        best_value = float("inf")
        best_state = None
        patience = 0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_generative = []
            epoch_regularizer = []
            epoch_total = []
            epoch_gradient_norms = []
            epoch_gradient_seconds = []

            for batch_idx, batch in enumerate(train_loader):
                data, global_indices, explicit_indices = self._unpack_batch(batch, batch_idx, batch_size)
                if (
                    self.method == "glr"
                    and isinstance(train_loader.sampler, RandomSampler)
                    and not explicit_indices
                ):
                    raise ValueError(
                        "GLR with a shuffled DataLoader requires batches of "
                        "(combined_tensor, integer_index) so q/D state follows each observation."
                    )
                condition, y_true = self.model._split_batch(data, self.model.targetdim)
                generative_loss = self.model.generative_loss(y_true, condition)
                gradient_start = time.perf_counter()

                if self.method == "ipa":
                    ipa_result = self._ipa_result(condition, y_true)
                    regularizer_loss = ipa_result["loss"]
                    total_loss = generative_loss + self.regularization_lambda * regularizer_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=float(self.max_grad_norm),
                        )
                    gradient_norm = torch.sqrt(
                        sum(
                            torch.sum(parameter.grad.detach().pow(2))
                            for parameter in self.model.parameters()
                            if parameter.grad is not None
                        )
                    )
                    optimizer.step()
                    gradient_norm = float(gradient_norm)
                else:
                    glr_result = self.model.regularized_glr_gradient(
                        condition,
                        y_true,
                        global_indices,
                        use_vmap=self.use_vmap,
                        vmap_chunk_size=self.vmap_chunk_size,
                        inner_steps=self.glr_inner_steps,
                    )
                    gradient_norm = self._set_combined_glr_gradient(
                        generative_loss,
                        glr_result,
                        optimizer,
                    )
                    with torch.no_grad():
                        regularizer_loss = self._ipa_result(condition, y_true)["loss"]
                    total_loss = generative_loss.detach() + self.regularization_lambda * regularizer_loss

                epoch_gradient_seconds.append(time.perf_counter() - gradient_start)
                epoch_generative.append(float(generative_loss.detach()))
                epoch_regularizer.append(float(regularizer_loss.detach()))
                epoch_total.append(float(total_loss.detach()))
                epoch_gradient_norms.append(gradient_norm)

            val_generative, val_regularizer = self._validation_metrics(val_loader)
            val_total = val_generative + self.regularization_lambda * val_regularizer
            history["epoch"].append(epoch)
            history["generative_loss"].append(float(np.mean(epoch_generative)))
            history["regularizer_loss"].append(float(np.mean(epoch_regularizer)))
            history["total_loss"].append(float(np.mean(epoch_total)))
            history["val_generative_loss"].append(val_generative)
            history["val_regularizer_loss"].append(val_regularizer)
            history["val_total_loss"].append(val_total)
            history["combined_gradient_norm"].append(float(np.mean(epoch_gradient_norms)))
            history["gradient_seconds"].append(float(np.mean(epoch_gradient_seconds)))

            if verbose:
                backend = "vmap" if self.use_vmap else "loop"
                print(
                    f"epoch={epoch} method={self.method}/{backend} "
                    f"train_total={history['total_loss'][-1]:.6f} "
                    f"val_total={val_total:.6f}"
                )

            if np.isfinite(val_total) and val_total < best_value:
                best_value = val_total
                history["best_epoch"] = epoch
                patience = 0
                best_state = copy.deepcopy(self.model.state_dict())
                if checkpoint_path is not None:
                    checkpoint_path = Path(checkpoint_path)
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, checkpoint_path)
            else:
                patience += 1
                if patience >= early_stopping:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        history["best_val_total_loss"] = best_value
        history["epochs_ran"] = len(history["epoch"])
        history["checkpoint_path"] = str(checkpoint_path) if checkpoint_path is not None else None
        return history
