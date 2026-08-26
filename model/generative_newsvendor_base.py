import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if hasattr(torch, "func"):
    functional_call = torch.func.functional_call
    torch_func_grad = torch.func.grad
    vmap = torch.func.vmap
else:
    from torch.nn.utils.stateless import functional_call
    from functorch import grad as torch_func_grad, vmap


class GenerativeNewsvendorBase(nn.Module):
    """Shared IPA and GLR loops for conditional generative newsvendor models."""

    def _init_newsvendor_base(
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
        save_loss="gen_glr_pth",
        save_xlsx="gen_glr_xlsx",
    ):
        self.targetdim = int(targetdim)
        self.labeldim = int(labeldim)
        self.latent = int(latent)
        self.data_len = int(data_len or 0)
        self.epoch = int(epoch)
        self.quantiles = quantiles
        self.lambda1 = float(lambda1)
        self.lambda_gradient = float(lambda_gradient)
        self.samplingnumber = int(samplingnumber)
        self.target_quantile = float(target_quantile if target_quantile is not None else quantiles)
        self.cu = float(cost_under)
        self.co = float(cost_over)
        self.random_seed = random_seed
        self.innerloop = int(innerloop)
        self.k_step = 1.0
        self.save_loss = save_loss
        self.save_xlsx = save_xlsx
        self.loss_history = {}
        self.D_hat = []
        self.q_hat = []
        self.q_hat_list = []
        self._reset_glr_state(self.data_len)

    def forward(self, z, condition):
        return self.decode(z, condition)

    def decode(self, z, condition):
        raise NotImplementedError

    def generative_loss(self, y_true, condition):
        raise NotImplementedError

    def generation_named_parameters(self):
        return OrderedDict((name, p) for name, p in self.named_parameters() if p.requires_grad)

    def generation_parameters(self):
        return list(self.generation_named_parameters().values())

    def _device(self):
        return next(self.parameters()).device

    def get_save_path(self, save_tag=None):
        name = f"{self.__class__.__name__}_{self.labeldim}_{self.epoch}_{self.innerloop}_{self.random_seed}"
        if save_tag:
            name = f"{name}_{save_tag}"
        return os.path.join(self.save_loss, f"{name}.pth")

    def get_save_xlsx_path(self, save_tag=None):
        name = f"{self.__class__.__name__}_{self.labeldim}_{self.epoch}_{self.innerloop}_{self.random_seed}"
        if save_tag:
            name = f"{name}_{save_tag}"
        return os.path.join(self.save_xlsx, f"{name}.xlsx")

    def get_ipa_save_path(self, save_name, randomnumber, directory="MODEL"):
        save_name = save_name or self.__class__.__name__
        return os.path.join(directory, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_best_model.pth")

    def _reset_glr_state(self, data_len=None):
        if data_len is not None:
            self.data_len = int(data_len)
        params = self.generation_parameters()
        device = params[0].device if params else torch.device("cpu")
        self.q_hat = [[torch.tensor(0.0, device=device)] for _ in range(max(self.data_len, 0))]
        self.q_hat_list = [[torch.tensor(0.0, device=device)] for _ in range(max(self.data_len, 0))]
        self.D_hat = [
            [torch.zeros_like(param, device=device) for param in params]
            for _ in range(max(self.data_len, 0))
        ]

    def _ensure_glr_state(self, data_len):
        params = self.generation_parameters()
        needs_reset = len(self.D_hat) != int(data_len)
        if self.D_hat and params:
            needs_reset = needs_reset or len(self.D_hat[0]) != len(params)
        if needs_reset:
            self._reset_glr_state(data_len)
        self._sync_auxiliary_state_device()

    def _sync_auxiliary_state_device(self):
        device = self._device()
        self.q_hat = [
            [v.to(device) if isinstance(v, torch.Tensor) else torch.tensor(float(v), device=device) for v in row]
            for row in self.q_hat
        ]
        self.q_hat_list = [
            [v.to(device) if isinstance(v, torch.Tensor) else torch.tensor(float(v), device=device) for v in row]
            for row in self.q_hat_list
        ]
        self.D_hat = [[v.to(device) for v in row] for row in self.D_hat]

    def _split_batch(self, batch, targetdim=None):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        targetdim = self.targetdim if targetdim is None else int(targetdim)
        batch = batch.to(self._device())
        if targetdim == 1:
            y_true = batch[:, -1].reshape(-1, 1)
            condition = batch[:, :-1]
        else:
            y_true = batch[:, -targetdim:]
            condition = batch[:, :-targetdim]
        return condition, y_true

    def _batch_with_indices(self, batch_data, batch_idx, batch_size):
        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
            data, global_indices = batch_data
            global_indices = global_indices.detach().cpu().numpy()
        else:
            data = batch_data
            global_indices = np.arange(batch_idx * batch_size, batch_idx * batch_size + data.shape[0])
        return data, global_indices

    def newsvendor_loss(self, q_value, y_true):
        diff = y_true - q_value
        return torch.where(diff > 0, self.cu * diff, self.co * (-diff)).mean()

    def sample_many(self, condition, num_samples=None, requires_grad=True):
        num_samples = int(num_samples or self.samplingnumber)
        batch_size = condition.shape[0]
        condition_rep = condition[:, None, :].expand(batch_size, num_samples, condition.shape[1])
        condition_rep = condition_rep.reshape(batch_size * num_samples, condition.shape[1])
        z = torch.randn(
            batch_size * num_samples,
            self.latent,
            device=condition.device,
            dtype=condition.dtype,
            requires_grad=requires_grad,
        )
        generated = self.decode(z, condition_rep)
        return generated.view(batch_size, num_samples, self.targetdim)

    def sample_quantile_decision(self, condition, num_samples=None, requires_grad=True):
        generated = self.sample_many(condition, num_samples=num_samples, requires_grad=requires_grad)
        return torch.quantile(generated, self.target_quantile, dim=1)

    def batched_ipa_regularizer(
        self,
        condition,
        y_true,
        k=8,
        num_samples=None,
        use_vmap=True,
        vmap_chunk_size=None,
        latent_samples=None,
    ):
        """Return the mean of K scalar-target IPA losses.

        Each replicate generates ``num_samples`` conditional draws for every
        observation and uses the exact ceil(alpha * M)-th order statistic. The
        gradient of the returned mean is therefore the arithmetic mean of the
        K replicate IPA gradients.
        """
        if self.targetdim != 1:
            raise ValueError("Regularized IPA currently requires targetdim=1.")
        k = int(k)
        num_samples = int(num_samples or self.samplingnumber)
        if k < 1 or num_samples < 1:
            raise ValueError("k and num_samples must both be positive integers.")
        if not 0.0 < self.target_quantile < 1.0:
            raise ValueError("target_quantile must lie strictly between zero and one.")

        batch_size = condition.shape[0]
        expected_shape = (k, batch_size, num_samples, self.latent)
        if latent_samples is None:
            latent_samples = torch.randn(
                expected_shape,
                device=condition.device,
                dtype=condition.dtype,
            )
        else:
            latent_samples = latent_samples.to(device=condition.device, dtype=condition.dtype)
            if tuple(latent_samples.shape) != expected_shape:
                raise ValueError(
                    f"latent_samples must have shape {expected_shape}, got {tuple(latent_samples.shape)}."
                )

        condition_rep = condition[:, None, :].expand(batch_size, num_samples, condition.shape[1])
        condition_rep = condition_rep.reshape(batch_size * num_samples, condition.shape[1])

        def decode_replicate(z_replicate):
            generated = self.decode(
                z_replicate.reshape(batch_size * num_samples, self.latent),
                condition_rep,
            )
            return generated.reshape(batch_size, num_samples, self.targetdim)

        if use_vmap:
            chunk_size = k if vmap_chunk_size is None else int(vmap_chunk_size)
            if chunk_size < 1:
                raise ValueError("vmap_chunk_size must be positive when provided.")
            generated_chunks = []
            for start_idx in range(0, k, chunk_size):
                generated_chunks.append(vmap(decode_replicate)(latent_samples[start_idx:start_idx + chunk_size]))
            generated = torch.cat(generated_chunks, dim=0)
        else:
            generated = torch.stack(
                [decode_replicate(latent_samples[replicate_idx]) for replicate_idx in range(k)],
                dim=0,
            )

        order_index = max(1, min(num_samples, int(np.ceil(self.target_quantile * num_samples))))
        order_quantiles = torch.kthvalue(
            generated.squeeze(-1),
            order_index,
            dim=2,
        ).values.unsqueeze(-1)
        diff = y_true.unsqueeze(0) - order_quantiles
        point_losses = torch.where(diff > 0, self.cu * diff, self.co * (-diff))
        replicate_losses = point_losses.mean(dim=(1, 2))
        return {
            "loss": replicate_losses.mean(),
            "order_quantiles": order_quantiles,
            "replicate_losses": replicate_losses,
            "order_index": order_index,
            "k": k,
            "num_samples": num_samples,
            "use_vmap": bool(use_vmap),
        }

    def _build_q_tensor_for_batch(self, global_indices, q_local, device, dtype):
        q_values = torch.zeros(len(global_indices), 1, device=device, dtype=dtype)
        for idx, global_idx in enumerate(global_indices):
            if global_idx < len(self.q_hat) and float(self.q_hat[global_idx][0]) != 0.0:
                q_values[idx, 0] = float(self.q_hat[global_idx][0])
            elif q_local is not None:
                q_values[idx, 0] = q_local[idx, 0].to(device=device, dtype=dtype)
        return q_values

    def _glr_innerloop(
        self,
        condition,
        y_true,
        q_values,
        use_vmap=True,
        vmap_chunk_size=None,
        latent_samples=None,
        latent_dimensions=None,
    ):
        if self.targetdim != 1:
            raise ValueError("GLR globalsingle currently expects scalar targetdim=1.")

        batch_size = condition.shape[0]
        device = condition.device
        params = self.generation_named_parameters()
        epsilon = y_true.new_tensor(1e-6)
        neg_weight = y_true.new_tensor(-self.cu / (self.cu + self.co))
        pos_weight = y_true.new_tensor(self.co / (self.cu + self.co))

        def single_output(param_dict, z_i, condition_i):
            return functional_call(
                self,
                param_dict,
                (z_i.unsqueeze(0), condition_i.unsqueeze(0)),
            ).reshape(())

        def single_surrogate(param_dict, z_i, condition_i, y_i, q_i, dim_mask):
            def output_fn(latent):
                return single_output(param_dict, latent, condition_i)

            y_pred_i = output_fn(z_i)
            grad_z_i = torch_func_grad(output_fn)(z_i)
            h_prime_i = torch.sum(grad_z_i * dim_mask)

            def h_prime_fn(latent):
                return torch.sum(torch_func_grad(output_fn)(latent) * dim_mask)

            h_double_prime_vec_i = torch_func_grad(h_prime_fn)(z_i)
            h_double_prime_i = torch.sum(h_double_prime_vec_i * dim_mask)
            score_i = -torch.sum(z_i * dim_mask)
            safe_h_prime_i = torch.where(
                torch.abs(h_prime_i) < epsilon,
                torch.where(h_prime_i >= 0, epsilon, -epsilon),
                h_prime_i,
            )
            h_prime_inv_i = torch.clamp(1.0 / safe_h_prime_i, -100.0, 100.0)
            psi_2_i = torch.clamp(
                h_prime_inv_i * (score_i - h_double_prime_i * h_prime_inv_i),
                -100.0,
                100.0,
            )
            indicator_i = (y_pred_i <= q_i).to(y_pred_i.dtype)
            final_w_i = indicator_i * torch.where(q_i - y_i < 0, neg_weight, pos_weight)
            surrogate_loss_i = (y_pred_i * psi_2_i + h_prime_i * h_prime_inv_i) * final_w_i
            g2_i = torch.clamp(torch.abs(psi_2_i * indicator_i), min=1e-4, max=10.0)
            return surrogate_loss_i, (y_pred_i, g2_i)

        grad_fn = torch_func_grad(single_surrogate, argnums=0, has_aux=True)
        if latent_samples is None:
            latent_samples = torch.randn(batch_size, self.latent, device=device, dtype=condition.dtype)
        else:
            latent_samples = latent_samples.to(device=device, dtype=condition.dtype)
            expected_shape = (batch_size, self.latent)
            if tuple(latent_samples.shape) != expected_shape:
                raise ValueError(
                    f"latent_samples must have shape {expected_shape}, got {tuple(latent_samples.shape)}."
                )
        if latent_dimensions is None:
            latent_dimensions = torch.randint(0, self.latent, (batch_size,), device=device)
        else:
            latent_dimensions = latent_dimensions.to(device=device, dtype=torch.long)
            if tuple(latent_dimensions.shape) != (batch_size,):
                raise ValueError(
                    f"latent_dimensions must have shape {(batch_size,)}, got {tuple(latent_dimensions.shape)}."
                )
            if torch.any(latent_dimensions < 0) or torch.any(latent_dimensions >= self.latent):
                raise ValueError("latent_dimensions contains an invalid latent coordinate.")
        dim_masks = torch.zeros(batch_size, self.latent, device=device, dtype=latent_samples.dtype)
        dim_masks.scatter_(1, latent_dimensions.unsqueeze(1), 1.0)

        grad_chunks = {name: [] for name in params.keys()}
        y_pred_chunks = []
        g2_chunks = []

        if use_vmap:
            default_chunk_size = int(os.environ.get("GEN_NEWSVENDOR_GLR_CHUNK_SIZE", "8"))
            chunk_size = default_chunk_size if vmap_chunk_size is None else int(vmap_chunk_size)
            chunk_size = max(1, min(chunk_size, batch_size))
            for start_idx in range(0, batch_size, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size)
                per_sample_grads_chunk, (y_pred_chunk, g2_chunk) = vmap(
                    grad_fn,
                    in_dims=(None, 0, 0, 0, 0, 0),
                )(
                    params,
                    latent_samples[start_idx:end_idx],
                    condition[start_idx:end_idx],
                    y_true[start_idx:end_idx].squeeze(1),
                    q_values[start_idx:end_idx].squeeze(1),
                    dim_masks[start_idx:end_idx],
                )
                for name in params.keys():
                    grad_chunks[name].append(per_sample_grads_chunk[name].detach())
                y_pred_chunks.append(y_pred_chunk.detach())
                g2_chunks.append(g2_chunk.detach())
        else:
            for sample_idx in range(batch_size):
                per_sample_grad, (y_pred, g2) = grad_fn(
                    params,
                    latent_samples[sample_idx],
                    condition[sample_idx],
                    y_true[sample_idx, 0],
                    q_values[sample_idx, 0],
                    dim_masks[sample_idx],
                )
                for name in params.keys():
                    grad_chunks[name].append(per_sample_grad[name].detach().unsqueeze(0))
                y_pred_chunks.append(y_pred.detach().reshape(1))
                g2_chunks.append(g2.detach().reshape(1))

        per_sample_grads = OrderedDict((name, torch.cat(chunks, dim=0)) for name, chunks in grad_chunks.items())
        y_pred_batch = torch.cat(y_pred_chunks, dim=0).view(batch_size, 1)
        g2_batch = torch.cat(g2_chunks, dim=0).view(batch_size, 1)
        return per_sample_grads, y_pred_batch, g2_batch

    def _vectorized_glr_innerloop(
        self,
        condition,
        y_true,
        q_values,
        vmap_chunk_size=None,
        latent_samples=None,
        latent_dimensions=None,
    ):
        return self._glr_innerloop(
            condition,
            y_true,
            q_values,
            use_vmap=True,
            vmap_chunk_size=vmap_chunk_size,
            latent_samples=latent_samples,
            latent_dimensions=latent_dimensions,
        )

    def _loop_glr_innerloop(
        self,
        condition,
        y_true,
        q_values,
        latent_samples=None,
        latent_dimensions=None,
    ):
        return self._glr_innerloop(
            condition,
            y_true,
            q_values,
            use_vmap=False,
            latent_samples=latent_samples,
            latent_dimensions=latent_dimensions,
        )

    def regularized_glr_gradient(
        self,
        condition,
        y_true,
        global_indices,
        use_vmap=True,
        vmap_chunk_size=None,
        inner_steps=None,
    ):
        """Update per-observation GLR state and return its batch gradient."""
        if self.targetdim != 1:
            raise ValueError("Regularized GLR currently requires targetdim=1.")
        inner_steps = self.innerloop if inner_steps is None else int(inner_steps)
        if inner_steps < 1:
            raise ValueError("inner_steps must be a positive integer.")
        global_indices = np.asarray(global_indices, dtype=int)
        self._ensure_glr_state(self.data_len)

        with torch.no_grad():
            q_local = None
            if any(
                global_idx < len(self.q_hat) and float(self.q_hat[global_idx][0]) == 0.0
                for global_idx in global_indices
            ):
                q_local = self.sample_quantile_decision(
                    condition,
                    self.samplingnumber,
                    requires_grad=False,
                )

        q_values = None
        y_pred_batch = torch.zeros_like(y_true)
        g2_batch = torch.ones_like(y_true)
        alpha_k = 1 / (self.k_step ** 0.55)
        beta_k = 1 / (self.k_step ** 0.6)
        for _ in range(inner_steps):
            q_values = self._build_q_tensor_for_batch(
                global_indices,
                q_local,
                condition.device,
                y_true.dtype,
            )
            per_sample_grads, y_pred_batch, g2_batch = self._glr_innerloop(
                condition,
                y_true,
                q_values,
                use_vmap=use_vmap,
                vmap_chunk_size=vmap_chunk_size,
            )
            for sample_idx, global_idx in enumerate(global_indices):
                if global_idx >= len(self.D_hat):
                    continue
                g2_i = float(g2_batch[sample_idx, 0])
                for param_name, d_val in zip(self.generation_named_parameters().keys(), self.D_hat[global_idx]):
                    g1_val = per_sample_grads[param_name][sample_idx]
                    d_val.add_(alpha_k * (g1_val / g2_i - d_val))
                    d_val.clamp_(-1.0, 1.0)

        with torch.no_grad():
            for sample_idx, global_idx in enumerate(global_indices):
                if global_idx >= len(self.q_hat):
                    continue
                if float(self.q_hat[global_idx][0]) == 0.0:
                    q_new = float(q_values[sample_idx, 0])
                else:
                    q_current = float(self.q_hat[global_idx][0])
                    indicator = float(y_pred_batch[sample_idx, 0] <= q_current)
                    q_new = q_current + beta_k * (self.target_quantile - indicator)
                self.q_hat[global_idx] = [q_new]
                self.q_hat_list[global_idx].append(q_new)

        named_params = self.generation_named_parameters()
        average_gradient = [torch.zeros_like(param) for param in named_params.values()]
        valid_count = 0
        for global_idx in global_indices:
            if global_idx >= len(self.D_hat):
                continue
            for param_idx, d_val in enumerate(self.D_hat[global_idx]):
                average_gradient[param_idx].add_(d_val)
            valid_count += 1
        if valid_count:
            average_gradient = [(gradient / valid_count).detach() for gradient in average_gradient]
        self.k_step += 1
        return {
            "gradient": OrderedDict(zip(named_params.keys(), average_gradient)),
            "q_values": q_values.detach(),
            "generated_values": y_pred_batch.detach(),
            "g2_values": g2_batch.detach(),
            "use_vmap": bool(use_vmap),
        }

    def _trainconvae_sgd_impl(
        self,
        num_epochs,
        targetdim,
        traindata_loader,
        valdata_loader,
        early_stopping,
        save_name=None,
        randomnumber=None,
        if_test_lambda=False,
        if_test_sample=False,
        gen_lr=1e-3,
        ipa_lr=1e-3,
        update_mode="separate",
        selection_metric="newsvendor",
    ):
        if update_mode not in {"separate", "linear_combination"}:
            raise ValueError("update_mode must be 'separate' or 'linear_combination'.")
        if selection_metric not in {"generative", "newsvendor", "total"}:
            raise ValueError("selection_metric must be 'generative', 'newsvendor', or 'total'.")

        save_dir = "lambda" if if_test_lambda else ("sample" if if_test_sample else "MODEL")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs("lossrecord", exist_ok=True)
        best_model_path = self.get_ipa_save_path(save_name, randomnumber, save_dir)
        loss_csv_path = os.path.join(
            "lossrecord",
            f"{save_name or self.__class__.__name__}_{self.targetdim}_{self.labeldim}_{self.lambda1}_{self.samplingnumber}_{randomnumber}_loss_history.xlsx",
        )
        gen_optimizer = torch.optim.Adam(self.parameters(), lr=gen_lr)
        ipa_optimizer = torch.optim.SGD(self.generation_parameters(), lr=ipa_lr)
        self.loss_history = {
            "train_loss": [],
            "val_loss": [],
            "generative_loss": [],
            "newsvendor_loss": [],
            "val_generative_loss": [],
            "val_newsvendor_loss": [],
            "total_loss": [],
            "val_selection_loss": [],
            "newsvendor_gradient": [],
            "epoch": [],
            "best_loss_epoch": -1,
        }
        best_loss = float("inf")
        early_stopping_counter = 0
        innerloop_times = []
        gradient_times = []

        for epoch in range(num_epochs):
            epoch_gen_losses = []
            epoch_nv_losses = []
            epoch_total_losses = []
            epoch_grad_norms = []
            train_total = 0.0

            for batch in traindata_loader:
                condition, y_true = self._split_batch(batch, targetdim)
                gen_loss = self.generative_loss(y_true, condition)
                start_time = time.time()
                q_value = self.sample_quantile_decision(condition, self.samplingnumber, requires_grad=True)
                nv_loss = self.newsvendor_loss(q_value, y_true)

                if update_mode == "linear_combination":
                    total_loss = (1.0 - self.lambda1) * gen_loss + self.lambda1 * nv_loss
                    gen_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    gen_optimizer.step()
                else:
                    gen_optimizer.zero_grad()
                    gen_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    gen_optimizer.step()

                    ipa_optimizer.zero_grad()
                    q_value = self.sample_quantile_decision(condition, self.samplingnumber, requires_grad=True)
                    nv_loss = self.newsvendor_loss(q_value, y_true)
                    (self.lambda1 * nv_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.generation_parameters(), max_norm=1.0)
                    ipa_optimizer.step()
                innerloop_times.append(time.time() - start_time)

                with torch.no_grad():
                    q_eval = self.sample_quantile_decision(condition, self.samplingnumber, requires_grad=False)
                    nv_eval = self.newsvendor_loss(q_eval, y_true)

                grad_start = time.time()
                q_probe = self.sample_quantile_decision(condition, self.samplingnumber, requires_grad=True)
                nv_probe = self.newsvendor_loss(q_probe, y_true)
                nv_grads = torch.autograd.grad(
                    nv_probe,
                    self.generation_parameters(),
                    retain_graph=False,
                    allow_unused=True,
                )
                gradient_times.append(time.time() - grad_start)
                grad_norm = sum(torch.norm(g) for g in nv_grads if g is not None).item()

                total_value = gen_loss.item() + self.lambda1 * nv_eval.item()
                epoch_gen_losses.append(gen_loss.item())
                epoch_nv_losses.append(nv_eval.item())
                epoch_total_losses.append(total_value)
                epoch_grad_norms.append(grad_norm)
                train_total += total_value

            val_gen_losses = []
            val_nv_losses = []
            with torch.no_grad():
                for val_batch in valdata_loader:
                    condition, y_true = self._split_batch(val_batch, targetdim)
                    val_gen_losses.append(self.generative_loss(y_true, condition).item())
                    q_val = self.sample_quantile_decision(condition, self.samplingnumber, requires_grad=False)
                    val_nv_losses.append(self.newsvendor_loss(q_val, y_true).item())

            val_gen_loss = float(np.mean(val_gen_losses)) if val_gen_losses else float("inf")
            val_nv_loss = float(np.mean(val_nv_losses)) if val_nv_losses else float("inf")
            val_total_loss = val_gen_loss + self.lambda1 * val_nv_loss
            if selection_metric == "generative":
                val_loss = val_gen_loss
            elif selection_metric == "newsvendor":
                val_loss = val_nv_loss
            else:
                val_loss = val_total_loss
            self.loss_history["generative_loss"].append(float(np.mean(epoch_gen_losses)))
            self.loss_history["newsvendor_loss"].append(float(np.mean(epoch_nv_losses)))
            self.loss_history["newsvendor_gradient"].append(float(np.mean(epoch_grad_norms)))
            self.loss_history["total_loss"].append(float(np.mean(epoch_total_losses)))
            self.loss_history["train_loss"].append(train_total)
            self.loss_history["val_generative_loss"].append(val_gen_loss)
            self.loss_history["val_newsvendor_loss"].append(val_nv_loss)
            self.loss_history["val_loss"].append(val_loss)
            self.loss_history["val_selection_loss"].append(val_loss)
            self.loss_history["epoch"].append(epoch)

            if epoch % 20 == 0:
                print(
                    f"epoch: {epoch}, Train Loss: {train_total:.4f}, "
                    f"Val Selection Loss ({selection_metric}): {val_loss:.4f}, "
                    f"Gen Loss: {np.mean(epoch_gen_losses):.4f}, Newsvendor Loss: {np.mean(epoch_nv_losses):.4f}"
                )

            if val_loss < best_loss:
                best_loss = val_loss
                early_stopping_counter = 0
                self.loss_history["best_loss_epoch"] = epoch
                torch.save(self.state_dict(), best_model_path)
                print(f"epoch: {epoch}, find new best loss: Val Selection Loss ({selection_metric}): {best_loss:.4f}")
                print(f"best model saved to: {best_model_path}")
                print("-" * 10)
            else:
                early_stopping_counter += 1

            if early_stopping_counter == early_stopping:
                print(f"Early stopping after {epoch} epochs")
                break

        self.loss_history["avg_innerloop_time"] = float(np.mean(innerloop_times)) if innerloop_times else 0.0
        self.loss_history["avg_gradient_time"] = float(np.mean(gradient_times)) if gradient_times else 0.0
        pd.DataFrame(self.loss_history).to_excel(loss_csv_path, index=False)
        return self.loss_history, best_model_path, self.loss_history["avg_innerloop_time"]

    def train_generator_only(
        self,
        num_epochs,
        targetdim,
        traindata_loader,
        valdata_loader,
        early_stopping,
        save_name=None,
        randomnumber=None,
        gen_lr=1e-3,
        selection_metric="newsvendor",
    ):
        if selection_metric not in {"generative", "newsvendor", "total"}:
            raise ValueError("selection_metric must be 'generative', 'newsvendor', or 'total'.")
        os.makedirs("MODEL", exist_ok=True)
        os.makedirs("lossrecord", exist_ok=True)
        best_model_path = self.get_ipa_save_path(save_name, randomnumber, "MODEL")
        loss_csv_path = os.path.join(
            "lossrecord",
            f"{save_name or self.__class__.__name__}_{self.targetdim}_{self.labeldim}_generator_only_{self.samplingnumber}_{randomnumber}_loss_history.xlsx",
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=gen_lr)
        self.loss_history = {
            "train_loss": [],
            "val_loss": [],
            "generative_loss": [],
            "newsvendor_loss": [],
            "val_generative_loss": [],
            "val_newsvendor_loss": [],
            "total_loss": [],
            "val_selection_loss": [],
            "newsvendor_gradient": [],
            "epoch": [],
            "best_loss_epoch": -1,
        }
        best_loss = float("inf")
        early_stopping_counter = 0
        gradient_times = []

        for epoch in range(num_epochs):
            epoch_gen_losses = []
            epoch_nv_losses = []
            epoch_grad_norms = []
            train_total = 0.0

            for batch in traindata_loader:
                condition, y_true = self._split_batch(batch, targetdim)
                gen_loss = self.generative_loss(y_true, condition)
                optimizer.zero_grad()
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    q_eval = self.sample_quantile_decision(condition, self.samplingnumber, requires_grad=False)
                    nv_eval = self.newsvendor_loss(q_eval, y_true)

                grad_start = time.time()
                q_probe = self.sample_quantile_decision(condition, self.samplingnumber, requires_grad=True)
                nv_probe = self.newsvendor_loss(q_probe, y_true)
                nv_grads = torch.autograd.grad(
                    nv_probe,
                    self.generation_parameters(),
                    retain_graph=False,
                    allow_unused=True,
                )
                gradient_times.append(time.time() - grad_start)
                grad_norm = sum(torch.norm(g) for g in nv_grads if g is not None).item()

                epoch_gen_losses.append(gen_loss.item())
                epoch_nv_losses.append(nv_eval.item())
                epoch_grad_norms.append(grad_norm)
                train_total += gen_loss.item()

            val_gen_losses = []
            val_nv_losses = []
            with torch.no_grad():
                for val_batch in valdata_loader:
                    condition, y_true = self._split_batch(val_batch, targetdim)
                    val_gen_losses.append(self.generative_loss(y_true, condition).item())
                    q_val = self.sample_quantile_decision(condition, self.samplingnumber, requires_grad=False)
                    val_nv_losses.append(self.newsvendor_loss(q_val, y_true).item())

            val_gen_loss = float(np.mean(val_gen_losses)) if val_gen_losses else float("inf")
            val_nv_loss = float(np.mean(val_nv_losses)) if val_nv_losses else float("inf")
            val_total_loss = val_gen_loss + self.lambda1 * val_nv_loss
            if selection_metric == "generative":
                val_loss = val_gen_loss
            elif selection_metric == "newsvendor":
                val_loss = val_nv_loss
            else:
                val_loss = val_total_loss
            self.loss_history["generative_loss"].append(float(np.mean(epoch_gen_losses)))
            self.loss_history["newsvendor_loss"].append(float(np.mean(epoch_nv_losses)))
            self.loss_history["newsvendor_gradient"].append(float(np.mean(epoch_grad_norms)))
            self.loss_history["total_loss"].append(float(np.mean(epoch_gen_losses)))
            self.loss_history["train_loss"].append(train_total)
            self.loss_history["val_generative_loss"].append(val_gen_loss)
            self.loss_history["val_newsvendor_loss"].append(val_nv_loss)
            self.loss_history["val_loss"].append(val_loss)
            self.loss_history["val_selection_loss"].append(val_loss)
            self.loss_history["epoch"].append(epoch)

            if epoch % 20 == 0:
                print(
                    f"epoch: {epoch}, Train Gen Loss: {train_total:.4f}, "
                    f"Val Selection Loss ({selection_metric}): {val_loss:.4f}, "
                    f"Newsvendor Loss: {np.mean(epoch_nv_losses):.4f}"
                )

            if val_loss < best_loss:
                best_loss = val_loss
                early_stopping_counter = 0
                self.loss_history["best_loss_epoch"] = epoch
                torch.save(self.state_dict(), best_model_path)
                print(
                    f"epoch: {epoch}, find new best generator checkpoint: "
                    f"Val Selection Loss ({selection_metric}): {best_loss:.4f}"
                )
                print(f"best model saved to: {best_model_path}")
                print("-" * 10)
            else:
                early_stopping_counter += 1

            if early_stopping_counter == early_stopping:
                print(f"Early stopping after {epoch} epochs")
                break

        self.loss_history["avg_innerloop_time"] = 0.0
        self.loss_history["avg_gradient_time"] = float(np.mean(gradient_times)) if gradient_times else 0.0
        pd.DataFrame(self.loss_history).to_excel(loss_csv_path, index=False)
        return self.loss_history, best_model_path, self.loss_history["avg_innerloop_time"]

    def trainconvae_sgd(
        self,
        num_epochs,
        targetdim,
        traindata_loader,
        valdata_loader,
        early_stopping,
        save_name=None,
        save_interval=50,
        randomnumber=None,
        if_test_lambda=False,
        if_test_sample=False,
        gen_lr=1e-3,
        ipa_lr=1e-3,
        update_mode="separate",
        selection_metric="newsvendor",
    ):
        return self._trainconvae_sgd_impl(
            num_epochs,
            targetdim,
            traindata_loader,
            valdata_loader,
            early_stopping,
            save_name=save_name,
            randomnumber=randomnumber,
            if_test_lambda=if_test_lambda,
            if_test_sample=if_test_sample,
            gen_lr=gen_lr,
            ipa_lr=ipa_lr,
            update_mode=update_mode,
            selection_metric=selection_metric,
        )

    def trainconvae_sgd_linear_combination(self, *args, **kwargs):
        kwargs["update_mode"] = "linear_combination"
        return self.trainconvae_sgd(*args, **kwargs)

    def trainconvae_sgd_separate_update(self, *args, **kwargs):
        kwargs["update_mode"] = "separate"
        return self.trainconvae_sgd(*args, **kwargs)

    def train_step_sqo_vectorized_SGD_LR_globalsingle(
        self,
        data_loader,
        valdata_loader,
        early_stopping,
        batch_size,
        ifdecoderonly=False,
        ifsave=False,
        save_tag=None,
        ifonlyglr=False,
        iftwoupdate=False,
        gen_lr=5e-4,
        selection_metric="newsvendor",
    ):
        if selection_metric not in {"generative", "newsvendor", "total"}:
            raise ValueError("selection_metric must be 'generative', 'newsvendor', or 'total'.")
        dataset_len = self.data_len or len(data_loader.dataset)
        self._ensure_glr_state(dataset_len)
        optimizer = torch.optim.Adam(self.parameters(), lr=gen_lr)
        gen_named_params = self.generation_named_parameters()
        gen_params = list(gen_named_params.values())
        best_loss = float("inf")
        early_stopping_counter = 0
        gen_loss_list = []
        nv_loss_list = []
        val_gen_loss_list = []
        val_nv_loss_list = []
        best_loss_epoch = -1
        start_time_list = []
        gradient_time_list = []
        xlsx_save_pth = self.get_save_xlsx_path(save_tag)
        save_pth = self.get_save_path(save_tag)
        os.makedirs(self.save_loss, exist_ok=True)
        os.makedirs(self.save_xlsx, exist_ok=True)

        for epoch in range(self.epoch):
            sum_gen_loss = 0.0
            sum_nv_loss = 0.0

            for batch_idx, batch_data in enumerate(data_loader):
                data, global_indices = self._batch_with_indices(batch_data, batch_idx, batch_size)
                condition, y_true = self._split_batch(data, self.targetdim)
                current_batch_size = y_true.shape[0]
                gen_loss = self.generative_loss(y_true, condition)
                grad_gen_raw = torch.autograd.grad(
                    gen_loss,
                    gen_params,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_gen = [
                    grad.detach().clone() if grad is not None else torch.zeros_like(param)
                    for param, grad in zip(gen_params, grad_gen_raw)
                ]
                if iftwoupdate and not ifonlyglr:
                    optimizer.zero_grad()
                    for param, grad in zip(gen_params, grad_gen):
                        param.grad = grad.clone()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

                with torch.no_grad():
                    q_local = None
                    if any(global_idx < len(self.q_hat) and float(self.q_hat[global_idx][0]) == 0.0 for global_idx in global_indices):
                        q_local = self.sample_quantile_decision(condition, self.samplingnumber, requires_grad=False)
                    q_eval = self.sample_quantile_decision(condition, self.samplingnumber, requires_grad=False)
                    sum_nv_loss += self.newsvendor_loss(q_eval, y_true).item()

                sum_gen_loss += gen_loss.item()
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx}, Gen Loss: {gen_loss.item():.4f}")

                q_values = None
                y_pred_batch = torch.zeros(current_batch_size, 1, device=condition.device, dtype=y_true.dtype)
                beta_k = 1 / (self.k_step ** 0.6)
                start_time = time.time()
                for _ in range(self.innerloop):
                    q_values = self._build_q_tensor_for_batch(global_indices, q_local, condition.device, y_true.dtype)
                    grad_start = time.time()
                    per_sample_grads, y_pred_batch, g2_batch = self._vectorized_glr_innerloop(condition, y_true, q_values)
                    gradient_time_list.append(time.time() - grad_start)
                    alpha_k = 1 / (self.k_step ** 0.55)
                    beta_k = 1 / (self.k_step ** 0.6)

                    for sample_idx, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.D_hat):
                            continue
                        g2_i = g2_batch[sample_idx, 0].item()
                        for param_idx, (param_name, d_val) in enumerate(zip(gen_named_params.keys(), self.D_hat[global_idx])):
                            g1_val = per_sample_grads[param_name][sample_idx]
                            update = g1_val / g2_i - d_val
                            d_val.add_(alpha_k * update)
                            d_val.clamp_(-1.0, 1.0)
                start_time_list.append(time.time() - start_time)

                if q_values is None:
                    q_values = self._build_q_tensor_for_batch(global_indices, q_local, condition.device, y_true.dtype)

                with torch.no_grad():
                    for sample_idx, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.q_hat):
                            continue
                        if float(self.q_hat[global_idx][0]) == 0.0:
                            q_init = q_values[sample_idx, 0].item()
                            self.q_hat[global_idx] = [q_init]
                            self.q_hat_list[global_idx].append(q_init)
                        else:
                            q_hat_current = float(self.q_hat[global_idx][0])
                            indicator_val = float(y_pred_batch[sample_idx, 0].item() <= q_hat_current)
                            q_hat_new = q_hat_current + beta_k * (self.target_quantile - indicator_val)
                            self.q_hat[global_idx] = [q_hat_new]
                            self.q_hat_list[global_idx].append(q_hat_new)

                avg_D = [torch.zeros_like(param) for param in gen_params]
                valid_count = 0
                for global_idx in global_indices:
                    if global_idx < len(self.D_hat):
                        for param_idx, d_val in enumerate(self.D_hat[global_idx]):
                            avg_D[param_idx] += d_val.to(avg_D[param_idx].device)
                        valid_count += 1
                if valid_count > 0:
                    avg_D = [(d_val / valid_count).detach() for d_val in avg_D]

                optimizer.zero_grad()
                for param, d_val in zip(gen_params, avg_D):
                    param.grad = d_val.clone() * self.lambda_gradient
                if not ifonlyglr and not iftwoupdate:
                    for param, grad in zip(gen_params, grad_gen):
                        if param.grad is None:
                            param.grad = grad.clone()
                        else:
                            param.grad += grad
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                self.k_step += 1

            gen_epoch = sum_gen_loss / max(len(data_loader), 1)
            nv_epoch = sum_nv_loss / max(len(data_loader), 1)
            gen_loss_list.append(gen_epoch)
            nv_loss_list.append(nv_epoch)

            val_gen_losses = []
            val_nv_losses = []
            with torch.no_grad():
                for val_batch in valdata_loader:
                    condition, y_true = self._split_batch(val_batch, self.targetdim)
                    val_gen_losses.append(self.generative_loss(y_true, condition).item())
                    q_val = self.sample_quantile_decision(condition, self.samplingnumber, requires_grad=False)
                    val_nv_losses.append(self.newsvendor_loss(q_val, y_true).item())
            val_gen_loss = float(np.mean(val_gen_losses)) if val_gen_losses else float("inf")
            val_nv_loss = float(np.mean(val_nv_losses)) if val_nv_losses else float("inf")
            val_total_loss = val_gen_loss + self.lambda_gradient * val_nv_loss
            val_gen_loss_list.append(val_gen_loss)
            val_nv_loss_list.append(val_nv_loss)

            if selection_metric == "generative":
                val_selection_loss = val_gen_loss
            elif selection_metric == "newsvendor":
                val_selection_loss = val_nv_loss
            else:
                val_selection_loss = val_total_loss

            if val_selection_loss < best_loss and not np.isnan(gen_epoch):
                best_loss = val_selection_loss
                early_stopping_counter = 0
                best_loss_epoch = epoch
                if not ifsave:
                    torch.save(self.state_dict(), save_pth)
                print("-" * 10)
            else:
                early_stopping_counter += 1

            if early_stopping_counter == early_stopping:
                print(f"Early stopping after {epoch} epochs")
                break
            if np.isnan(gen_epoch):
                print("NaN detected in generative loss, stopping training.")
                break

        if ifsave:
            pd.DataFrame({
                "generative_loss": gen_loss_list,
                "newsvendor_loss": nv_loss_list,
                "val_generative_loss": val_gen_loss_list,
                "val_newsvendor_loss": val_nv_loss_list,
            }).to_excel(xlsx_save_pth, index=False)

        return {
            "generative_loss": gen_loss_list,
            "newsvendor_loss": nv_loss_list,
            "val_generative_loss": val_gen_loss_list,
            "val_newsvendor_loss": val_nv_loss_list,
            "val_selection_loss": (
                val_gen_loss_list if selection_metric == "generative"
                else val_nv_loss_list if selection_metric == "newsvendor"
                else [g + self.lambda_gradient * n for g, n in zip(val_gen_loss_list, val_nv_loss_list)]
            ),
            "best_loss_epoch": best_loss_epoch,
            "best_model_path": save_pth if not ifsave else None,
            "avg_innerloop_time": float(np.mean(start_time_list)) if start_time_list else 0.0,
            "avg_gradient_time": float(np.mean(gradient_time_list)) if gradient_time_list else 0.0,
            "message": f"GLR globalsingle training completed: {self.epoch} epochs, {len(data_loader)} batches per epoch",
        }

    def make_regularized_trainer(
        self,
        method,
        regularization_lambda=None,
        learning_rate=1e-3,
        use_vmap=True,
        k=8,
        num_samples=None,
        vmap_chunk_size=None,
        glr_inner_steps=None,
        max_grad_norm=1.0,
    ):
        """Build the shared regularized trainer for this generative model."""
        from model.regularized_gradient_trainer import RegularizedGenerativeTrainer

        if regularization_lambda is None:
            regularization_lambda = self.lambda1 if method == "ipa" else self.lambda_gradient
        return RegularizedGenerativeTrainer(
            model=self,
            method=method,
            regularization_lambda=regularization_lambda,
            learning_rate=learning_rate,
            use_vmap=use_vmap,
            k=k,
            num_samples=num_samples,
            vmap_chunk_size=vmap_chunk_size,
            glr_inner_steps=glr_inner_steps,
            max_grad_norm=max_grad_norm,
        )

    def train_regularized_ipa(
        self,
        traindata_loader,
        valdata_loader,
        num_epochs=None,
        early_stopping=10,
        regularization_lambda=None,
        learning_rate=1e-3,
        k=8,
        num_samples=None,
        use_vmap=True,
        vmap_chunk_size=None,
        max_grad_norm=1.0,
        checkpoint_path=None,
        verbose=False,
    ):
        trainer = self.make_regularized_trainer(
            method="ipa",
            regularization_lambda=regularization_lambda,
            learning_rate=learning_rate,
            use_vmap=use_vmap,
            k=k,
            num_samples=num_samples,
            vmap_chunk_size=vmap_chunk_size,
            max_grad_norm=max_grad_norm,
        )
        return trainer.fit(
            traindata_loader,
            valdata_loader,
            num_epochs=self.epoch if num_epochs is None else num_epochs,
            early_stopping=early_stopping,
            checkpoint_path=checkpoint_path,
            verbose=verbose,
        )

    def train_regularized_glr(
        self,
        traindata_loader,
        valdata_loader,
        num_epochs=None,
        early_stopping=10,
        regularization_lambda=None,
        learning_rate=1e-3,
        num_samples=None,
        use_vmap=True,
        vmap_chunk_size=None,
        glr_inner_steps=None,
        max_grad_norm=1.0,
        checkpoint_path=None,
        verbose=False,
    ):
        trainer = self.make_regularized_trainer(
            method="glr",
            regularization_lambda=regularization_lambda,
            learning_rate=learning_rate,
            use_vmap=use_vmap,
            k=1,
            num_samples=num_samples,
            vmap_chunk_size=vmap_chunk_size,
            glr_inner_steps=glr_inner_steps,
            max_grad_norm=max_grad_norm,
        )
        return trainer.fit(
            traindata_loader,
            valdata_loader,
            num_epochs=self.epoch if num_epochs is None else num_epochs,
            early_stopping=early_stopping,
            checkpoint_path=checkpoint_path,
            verbose=verbose,
        )

    def train_regularized_ipa_vmap(self, *args, **kwargs):
        kwargs["use_vmap"] = True
        return self.train_regularized_ipa(*args, **kwargs)

    def train_regularized_ipa_loop(self, *args, **kwargs):
        kwargs["use_vmap"] = False
        return self.train_regularized_ipa(*args, **kwargs)

    def train_regularized_glr_vmap(self, *args, **kwargs):
        kwargs["use_vmap"] = True
        return self.train_regularized_glr(*args, **kwargs)

    def train_regularized_glr_loop(self, *args, **kwargs):
        kwargs["use_vmap"] = False
        return self.train_regularized_glr(*args, **kwargs)
