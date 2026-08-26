import torch
import os
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torchvision.utils import save_image
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time

from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
import numpy as np
from functools import partial



def loss_function(recon_x, x, mu, logvar):
    reconstruction_function = nn.MSELoss(size_average=False)
    MSE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD

class Encoder(nn.Module):
    def __init__(self, input_dim, con_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim+con_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)
        self.linear4 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
    def forward(self, x,con_x):
        x = torch.cat((x, con_x), dim=1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x1 = self.linear3(x)
        x2 = self.linear4(x)
        return x1,x2

class Decoder(nn.Module):
    _ACTIVATIONS = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'silu': nn.SiLU,
        'softplus': nn.Softplus,
        'softmax': partial(nn.Softmax, dim=-1),
    }

    def __init__(self, output_dim, con_dim, hidden_dim, latent_dim, activation='softplus'):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim+con_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.activation_name = activation.lower()
        act_cls = self._ACTIVATIONS.get(self.activation_name)
        if act_cls is None:
            raise ValueError(f"Unsupported activation '{activation}'. Choose from: {list(self._ACTIVATIONS.keys())}")
        self.act = act_cls()

    def forward(self, x, con_x):
        x = torch.cat((x, con_x), dim=1)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x

class VAE_end_to_end(nn.Module):
    def __init__(self, targetdim, labeldim, latent, quantiles=0.5, lambda1=0.5, iftorchsort=False, samplingnumber=100, decoder_activation='softplus'):


        super(VAE_end_to_end, self).__init__()
        self.fc1 = nn.Linear(targetdim + labeldim, 32)
        self.fc11 = nn.Linear(32, 64)
        self.fc21 = nn.Linear(64, latent)  # mean
        self.fc22 = nn.Linear(64, latent)  # var
        self.fc3 = nn.Linear(latent + labeldim, 32)
        self.fc31 = nn.Linear(32, 64)
        self.targetdim = targetdim
        self.labeldim = labeldim
        self.fc4 = nn.Linear(64, targetdim)
        self.latent = latent
        self.quantiles = quantiles
        self.encoder = Encoder(targetdim, labeldim, 64, latent)
        self.decoder = Decoder(targetdim, labeldim, 64, latent, activation=decoder_activation)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3) # redundant, already defined above
        self.lambda1 = lambda1
        self.iftorchsort = iftorchsort
        self.samplingnumber = samplingnumber
        self.save_loss = "lossrecord"
        # 损失记录
        self.loss_history = {
            'train_loss': [],
            'val_loss': [],
            'vae_loss': [],
            'quantile_loss': [],
            'total_loss': [],
            'quantile_gradient': []
        }
    
    def get_save_path(self, save_name, randomnumber, vae_pth_dir="ipa_pth"):
        """
        获取模型保存路径
        """
        return os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_best_model.pth")
        
    def encode(self, x, condition):  # 编码层
        return self.encoder(x,condition)


    def quantile_loss(self, y_pred, y_true):
        """
        修正的分位数损失函数，确保损失值为非负且在合理量级
        """
        # 确保输入张量在同一设备上且类型一致
        if y_pred.device != y_true.device:
            y_true = y_true.to(y_pred.device)
        if y_pred.dtype != y_true.dtype:
            y_true = y_true.to(y_pred.dtype)
        
        # 调试信息：检查输入张量的梯度状态
        #print(f"Debug quantile_loss: y_pred.requires_grad={y_pred.requires_grad}, y_true.requires_grad={y_true.requires_grad}")
        
        # 确保quantiles也是合适的类型和设备
        if isinstance(self.quantiles, torch.Tensor):
            quantiles = self.quantiles.to(y_pred.device).to(y_pred.dtype)
        else:
            quantiles = torch.tensor(self.quantiles, device=y_pred.device, dtype=y_pred.dtype)
        
        diff = y_true - y_pred
        # 修正：使用绝对值确保损失为非负，并添加合理的缩放
        quantile_loss = torch.where(diff >= 0, 
                                   quantiles * torch.abs(diff), 
                                   (1.0 - quantiles) * torch.abs(diff))
        return torch.mean(quantile_loss)

    def smooth_quantile_loss(self, y_pred, y_true, alpha=None, delta=None):
        """平滑报童问题损失函数"""
        if alpha is None:
            alpha = self.quantiles
        if delta is None:
            delta = self.smooth_delta
        return smooth_pinball_loss(y_pred, y_true, alpha, delta)


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # e**(x*0.5)
        eps = torch.FloatTensor(std.size()).normal_()
        device = mu.device
        if torch.cuda.is_available() and device.type == 'cuda':
            eps = Variable(eps.cuda())
        else:
            eps = Variable(eps.to(device))
        # 避免原地操作：使用add代替add_
        return eps.mul(std).add(mu)

    def decode(self, z, condition):  # 解码层
        return self.decoder(z,condition)

    def forward(self, x, condition):
        mu, logvar = self.encode(x, condition)  # 编码
        z = self.reparametrize(mu, logvar)  # 重新参数化成正态分布
        return self.decode(z, condition), mu, logvar  # 解码，同时输出均值方差



    def line_search(self,vae_loss, quantile_loss, lambda_cur, s, max_step=1.0, n_steps=10):
        best_lambda, best_loss = lambda_cur, 1e9
        for i in range(n_steps+1):
            gamma = i / n_steps * max_step
            lambda_cand = (1-gamma)*lambda_cur + gamma*s
            lambda_cand = torch.clamp(lambda_cand, 0.0, 1.0)
            loss_cand = (1-lambda_cand)*vae_loss + lambda_cand*quantile_loss
            if loss_cand < best_loss:
                best_loss, best_lambda = loss_cand, lambda_cand
        return best_lambda



    
    

    def trainconvae(self, num_epochs, targetdim, traindata_loader, valdata_loader, early_stopping, 
                    ipa_update_mode='batch', save_name=None, save_interval=50, randomnumber = None,if_test_lambda = False):
        best_loss = float('inf')
        early_stopping_counter = 0
        
        
        # 确保保存目录存在
        import os
        if if_test_lambda:
            save_pth_dir = "lambda"
        else:
            save_pth_dir = "lossrecord"

        
        # 确保保存目录存在
        import os
        vae_pth_dir = "MODEL"
        os.makedirs(vae_pth_dir, exist_ok=True)
        best_model_path = os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_best_model.pth")

        os.makedirs(save_pth_dir, exist_ok=True)
        loss_csv_path = os.path.join(save_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_loss_history.xlsx")
        self.loss_history = {
            'train_loss': [],
            'val_loss': [],
            'vae_loss': [],
            'quantile_loss': [],
            'ipa_loss': [],
            'val_vae_loss': [],
            'val_quantile_loss': [],
            'total_loss': [],
            'quantile_gradient': [],  # 当 ipa_update_mode == 'batch' 时记录当轮平均梯度范数
            'epoch': [],
            'best_loss_epoch': -1
        }
        
        # 创建分离的优化器：VAE优化器（所有参数）和Decoder优化器（仅decoder参数）
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        
        whole_losslist = []
        for epoch in range(num_epochs):
            whole_loss = 0
            loss_new = 0
            epoch_vae_losses = []
            epoch_quantile_losses = []
            epoch_quantile_grads = []
            total_loss = []
            ipa_losses = []
            for i, batch in enumerate(traindata_loader):
                # 处理数据加载器返回的数据格式
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # 如果是tuple或list，取第一个元素
                
                batch_size = batch.shape[0]
                device = next(self.parameters()).device
                batch = batch.to(device)
                im = batch[:, -1].reshape(-1, 1).to(device)
                im_label = batch[:, :-1].to(device)
                
                # 第一步：VAE损失和梯度更新（影响所有参数）
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                vae_loss.backward()
                vae_optimizer.step()
                with torch.no_grad():
                    z_samples = torch.randn(batch_size * self.samplingnumber, self.latent).to(device)
                    im_label_repeated = im_label.repeat(self.samplingnumber, 1)
                    
                    # Generate images for all samples in one forward pass
                    generate_ims = self.decode(z_samples, im_label_repeated)
                    
                    # Reshape to (batch_size, samplingnumber, ...)
                    generate_ims = generate_ims.view(batch_size, self.samplingnumber, *generate_ims.shape[1:])
                    
                    # Compute quantiles across the sampling dimension (dim=1)
                    generate_quantiles = torch.quantile(generate_ims, self.quantiles, dim=1)
                    
                    # Assuming generate_im is the mean or a single sample for VAE loss, but since only training decoder with quantile loss,
                    # we might not need VAE loss. If needed, compute it separately.
                    # For now, assuming quantile_loss is the primary loss, and vae_loss is optional or zero.
                    # If vae_loss is required, add it here (e.g., using a single sample or mean).
                    
                    # Compute quantile loss (assuming self.quantile_loss takes predicted quantiles and target)
                    # Adjust if it takes mean or something else; based on code, it was using generate_im (single sample?) but now using quantiles.
                    ipa_loss = self.quantile_loss(generate_quantiles, im)  # Pinball loss or similar
                    ipa_losses.append(ipa_loss.item())


                    z_sample_new = torch.randn(im_label.shape[0], self.latent).to(device)
                    z_sample_new.requires_grad_(True)
                    generate_im_new = self.decode(z_sample_new, im_label)
                    pinball_loss = self.quantile_loss(generate_im_new, im)  # 假设quantile_loss是pinball loss
                    epoch_quantile_losses.append(pinball_loss.item())


                # 重新生成样本（需要梯度用于分位数计算）
                z_sample_new = torch.randn(im_label.shape[0], self.latent).to(device)
                z_sample_new.requires_grad_(True)
                generate_im_new = self.decode(z_sample_new, im_label)
                gen_quantile_new = torch.quantile(generate_im_new, self.quantiles, dim=0)
                decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
                quantile_grads = torch.autograd.grad(gen_quantile_new, decoder_params, grad_outputs=torch.ones_like(gen_quantile_new), retain_graph=True)
                quantile_grad_norm = sum(torch.norm(g) for g in quantile_grads).item()

                z_samples = torch.randn(batch_size * self.samplingnumber, self.latent).to(device)
                im_label_repeated = im_label.repeat(self.samplingnumber, 1)
                
                # Generate images for all samples in one forward pass
                generate_ims = self.decode(z_samples, im_label_repeated)
                
                # Reshape to (batch_size, samplingnumber, ...)
                generate_ims = generate_ims.view(batch_size, self.samplingnumber, *generate_ims.shape[1:])
                
                # Compute quantiles across the sampling dimension (dim=1)
                generate_quantiles = torch.quantile(generate_ims, self.quantiles, dim=1)
                

                ipa_loss = self.quantile_loss(generate_quantiles, im)* self.lambda1  # Pinball loss or similar



                epoch_quantile_grads.append(quantile_grad_norm)
                epoch_vae_losses.append(vae_loss.item())
                whole_loss += vae_loss.item()
                total_loss.append(vae_loss.item())
            self.loss_history['vae_loss'].append(np.mean(epoch_vae_losses))
            self.loss_history['quantile_loss'].append(np.mean(epoch_quantile_losses))
            self.loss_history['ipa_loss'].append(np.mean(ipa_losses))
            self.loss_history['quantile_gradient'].append(np.mean(epoch_quantile_grads))
            self.loss_history['total_loss'].append(np.mean(total_loss))
            self.loss_history['train_loss'].append(whole_loss)
            self.loss_history['epoch'].append(epoch)
            val_loss = 0
            val_vae_losses = []
            val_quantile_losses = []
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    if targetdim == 1:
                        batch = val_batch.to(device)
                        im = batch[:, -1].reshape(-1, 1).to(device)
                        im_label = batch[:, :-1].to(device)
                    else:
                        batch = val_batch.to(device)
                        im = batch[:, -targetdim:].to(device)
                        im_label = batch[:, :-targetdim].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    val_loss += val_vae_loss.item()
                    # 计算验证分位数损失
                    device = next(self.parameters()).device
                    val_generate_im = self.decode(torch.randn(im_label.shape[0], self.latent).to(device), im_label)
                    val_quantile_loss = self.quantile_loss(val_generate_im, im)
                    
                    val_vae_losses.append(val_vae_loss.item())
                    val_quantile_losses.append(val_quantile_loss.item())
            
            self.loss_history['val_vae_loss'].append(np.mean(val_vae_losses))
            self.loss_history['val_quantile_loss'].append(np.mean(val_quantile_losses))
            val_loss /= len(valdata_loader)
            self.loss_history['val_loss'].append(val_loss)
            if (epoch) % 20 == 0:
                print('epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, VAE Loss: {:.4f}, Quantile Loss: {:.4f}'.format(
                    epoch, whole_loss, val_loss, np.mean(epoch_vae_losses), np.mean(epoch_quantile_losses)))
            
            loss_new = val_loss
            if loss_new < best_loss:
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                
                # 保存最佳模型（简化版本）
                torch.save(self.state_dict(), best_model_path)
                
                
                print('epoch: {}, find new best loss: Val Loss: {:.4f}'.format(epoch, best_loss))
                print(f'最佳模型已保存到: {best_model_path}')
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
        pd.DataFrame(self.loss_history).to_excel(loss_csv_path, index=False)

    def batched_ipa(
        self,
        im_label,
        im,
        k=8,
        m=None,
        vmap_chunk_size=None,
        use_vmap=True,
        latent_samples=None,
    ):
        """Return the K-replication IPA training loss for every condition.

        For each condition ``x_i``, this method runs ``k`` independent
        replications. Each replication decodes ``m`` latent samples, selects
        the ``ceil(alpha * m)`` order statistic, and evaluates its pinball
        loss against ``y_i``. The returned loss is the mean over replications,
        observations, and target dimensions. Therefore one backward pass gives
        exactly the arithmetic mean of the K IPA gradients.

        This method is training-only. It does not change any inference path.
        ``latent_samples`` is exposed for deterministic gradient-equivalence
        tests and must have shape ``[k, batch, m, latent]`` when supplied.
        """

        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        m = self.samplingnumber if m is None else m
        if not isinstance(m, int) or m < 1:
            raise ValueError("m must be a positive integer")
        if vmap_chunk_size is not None and (
            not isinstance(vmap_chunk_size, int) or vmap_chunk_size < 1
        ):
            raise ValueError("vmap_chunk_size must be a positive integer or None")

        if im_label.ndim != 2:
            raise ValueError("im_label must have shape [batch, condition_dim]")
        if im.ndim == 1:
            im = im.unsqueeze(1)
        if im.ndim != 2:
            raise ValueError("im must have shape [batch, target_dim]")
        if im_label.shape[0] != im.shape[0]:
            raise ValueError("im_label and im must have the same batch size")
        if im_label.shape[1] != self.labeldim:
            raise ValueError(
                f"expected {self.labeldim} condition features, got {im_label.shape[1]}"
            )
        if im.shape[1] != self.targetdim:
            raise ValueError(
                f"expected {self.targetdim} target features, got {im.shape[1]}"
            )

        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype
        im_label = im_label.to(device=device, dtype=dtype)
        im = im.to(device=device, dtype=dtype)
        batch_size = im.shape[0]

        if isinstance(self.quantiles, torch.Tensor):
            if self.quantiles.numel() != 1:
                raise ValueError("batched_ipa currently requires one scalar alpha")
            alpha = float(self.quantiles.detach().cpu().item())
        else:
            alpha = float(self.quantiles)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")

        expected_latent_shape = (k, batch_size, m, self.latent)
        if latent_samples is None:
            latent_samples = torch.randn(
                expected_latent_shape,
                device=device,
                dtype=dtype,
            )
        else:
            if tuple(latent_samples.shape) != expected_latent_shape:
                raise ValueError(
                    "latent_samples must have shape "
                    f"{expected_latent_shape}, got {tuple(latent_samples.shape)}"
                )
            latent_samples = latent_samples.to(device=device, dtype=dtype)

        condition_group = im_label[:, None, :].expand(
            batch_size, m, self.labeldim
        )
        flat_conditions = condition_group.reshape(batch_size * m, self.labeldim)

        def decode_one_replication(replication_latents):
            decoded = self.decode(
                replication_latents.reshape(batch_size * m, self.latent),
                flat_conditions,
            )
            return decoded.reshape(batch_size, m, self.targetdim)

        if use_vmap:
            vmap_impl = getattr(torch, "vmap", None)
            if vmap_impl is None:
                from torch.func import vmap as vmap_impl
            vmap_kwargs = {}
            if vmap_chunk_size is not None:
                vmap_kwargs["chunk_size"] = vmap_chunk_size
            generated = vmap_impl(
                decode_one_replication,
                in_dims=0,
                out_dims=0,
                **vmap_kwargs,
            )(latent_samples)
        else:
            all_conditions = condition_group.unsqueeze(0).expand(
                k, batch_size, m, self.labeldim
            )
            generated = self.decode(
                latent_samples.reshape(k * batch_size * m, self.latent),
                all_conditions.reshape(k * batch_size * m, self.labeldim),
            ).reshape(k, batch_size, m, self.targetdim)

        order_index = min(m, max(1, int(np.ceil(alpha * m))))
        order_quantiles = torch.kthvalue(
            generated,
            k=order_index,
            dim=2,
        ).values
        target = im.unsqueeze(0).expand(k, batch_size, self.targetdim)
        difference = target - order_quantiles
        alpha_tensor = difference.new_tensor(alpha)
        loss_values = torch.where(
            difference >= 0,
            alpha_tensor * difference,
            (1.0 - alpha_tensor) * (-difference),
        )
        replicate_losses = loss_values.mean(dim=(1, 2))
        loss = replicate_losses.mean()
        return {
            "loss": loss,
            "order_quantiles": order_quantiles,
            "replicate_losses": replicate_losses,
            "order_index": order_index,
            "k": k,
            "m": m,
        }

    def _split_batched_ipa_batch(self, batch, targetdim):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        device = next(self.parameters()).device
        batch = batch.to(device)
        if targetdim == 1:
            im = batch[:, -1:].reshape(-1, 1)
            im_label = batch[:, :-1]
        else:
            im = batch[:, -targetdim:]
            im_label = batch[:, :-targetdim]
        return im_label, im

    def _select_batched_ipa_decoder_parameters(
        self,
        fine_tune_mode,
        custom_trainable_prefixes=None,
    ):
        selected = []
        prefixes = custom_trainable_prefixes or []
        for name, parameter in self.decoder.named_parameters():
            if fine_tune_mode == "all":
                should_train = True
            elif fine_tune_mode == "last_layer":
                should_train = name.startswith("linear3")
            elif fine_tune_mode == "bias_only":
                should_train = name.endswith("bias")
            elif fine_tune_mode == "custom":
                should_train = any(name.startswith(prefix) for prefix in prefixes)
            else:
                raise ValueError(
                    "fine_tune_mode must be all, last_layer, bias_only, or custom"
                )
            parameter.requires_grad = should_train
            if should_train:
                selected.append(parameter)
        if not selected:
            raise ValueError("no decoder parameters were selected for IPA training")
        return selected

    def _train_batched_ipa_strategy(
        self,
        strategy,
        num_epochs,
        targetdim,
        traindata_loader,
        valdata_loader,
        early_stopping,
        k=8,
        m=None,
        save_name=None,
        randomnumber=None,
        learning_rate=1e-3,
        decoder_lr=1e-3,
        pretrain_save_name=None,
        pretrain_model_path=None,
        fine_tune_mode="all",
        custom_trainable_prefixes=None,
        vmap_chunk_size=None,
        use_vmap=True,
        verbose=True,
    ):
        """Shared training loop for batched LR-IPA, IPA-only, and two-stage IPA."""

        strategy = strategy.lower()
        valid_strategies = {"lr_ipa", "ipa_only", "two_stage_ipa"}
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {sorted(valid_strategies)}")
        if targetdim != self.targetdim:
            raise ValueError(
                f"targetdim={targetdim} does not match model targetdim={self.targetdim}"
            )
        if num_epochs < 1 or early_stopping < 1:
            raise ValueError("num_epochs and early_stopping must be positive")
        m = self.samplingnumber if m is None else m

        device = next(self.parameters()).device
        for parameter in self.parameters():
            parameter.requires_grad = True

        if strategy == "two_stage_ipa":
            if pretrain_model_path is None:
                if pretrain_save_name is None:
                    raise ValueError(
                        "two_stage_ipa requires pretrain_model_path or "
                        "pretrain_save_name"
                    )
                pretrain_model_path = self.get_save_path(
                    pretrain_save_name,
                    randomnumber,
                    "MODEL",
                )
            if not os.path.exists(pretrain_model_path):
                raise FileNotFoundError(
                    f"two-stage IPA checkpoint does not exist: {pretrain_model_path}"
                )
            self.load_state_dict(torch.load(pretrain_model_path, map_location=device))

        if strategy == "lr_ipa":
            trainable_parameters = [
                parameter for parameter in self.parameters() if parameter.requires_grad
            ]
            optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
        else:
            for parameter in self.parameters():
                parameter.requires_grad = False
            trainable_parameters = self._select_batched_ipa_decoder_parameters(
                fine_tune_mode=fine_tune_mode,
                custom_trainable_prefixes=custom_trainable_prefixes,
            )
            optimizer = torch.optim.SGD(trainable_parameters, lr=decoder_lr)

        save_name = save_name or f"batched_{strategy}"
        randomnumber = 0 if randomnumber is None else randomnumber
        model_dir = "MODEL"
        loss_dir = "lossrecord"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(loss_dir, exist_ok=True)
        best_model_path = self.get_save_path(
            save_name,
            randomnumber,
            model_dir,
        )
        loss_path = os.path.join(
            loss_dir,
            f"{save_name}_{self.targetdim}_{self.labeldim}_K{k}_M{m}_"
            f"{randomnumber}_loss_history.xlsx",
        )

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "vae_loss": [],
            "ipa_loss": [],
            "val_vae_loss": [],
            "val_ipa_loss": [],
            "ipa_gradient_norm": [],
            "k": [],
            "m": [],
        }
        best_loss = float("inf")
        best_epoch = -1
        stale_epochs = 0

        for epoch in range(num_epochs):
            self.train()
            train_objectives = []
            train_vae_losses = []
            train_ipa_losses = []
            gradient_norms = []

            for batch in traindata_loader:
                im_label, im = self._split_batched_ipa_batch(batch, targetdim)
                optimizer.zero_grad(set_to_none=True)

                if strategy == "lr_ipa":
                    reconstruction, mu, logvar = self.forward(im, im_label)
                    vae_loss = loss_function(
                        reconstruction,
                        im,
                        mu,
                        logvar,
                    ) / im.shape[0]
                else:
                    vae_loss = im.new_zeros(())

                ipa_result = self.batched_ipa(
                    im_label=im_label,
                    im=im,
                    k=k,
                    m=m,
                    vmap_chunk_size=vmap_chunk_size,
                    use_vmap=use_vmap,
                )
                ipa_loss = ipa_result["loss"]
                objective = (
                    vae_loss + self.lambda1 * ipa_loss
                    if strategy == "lr_ipa"
                    else self.lambda1 * ipa_loss
                )
                objective.backward()
                squared_norm = objective.new_zeros(())
                for parameter in trainable_parameters:
                    if parameter.grad is not None:
                        squared_norm = squared_norm + parameter.grad.detach().pow(2).sum()
                gradient_norms.append(float(torch.sqrt(squared_norm).cpu()))
                optimizer.step()

                train_objectives.append(float(objective.detach().cpu()))
                train_vae_losses.append(float(vae_loss.detach().cpu()))
                train_ipa_losses.append(float(ipa_loss.detach().cpu()))

            self.eval()
            val_objectives = []
            val_vae_losses = []
            val_ipa_losses = []
            with torch.no_grad():
                for batch in valdata_loader:
                    im_label, im = self._split_batched_ipa_batch(batch, targetdim)
                    if strategy == "lr_ipa":
                        reconstruction, mu, logvar = self.forward(im, im_label)
                        val_vae_loss = loss_function(
                            reconstruction,
                            im,
                            mu,
                            logvar,
                        ) / im.shape[0]
                    else:
                        val_vae_loss = im.new_zeros(())
                    val_ipa_result = self.batched_ipa(
                        im_label=im_label,
                        im=im,
                        k=k,
                        m=m,
                        vmap_chunk_size=vmap_chunk_size,
                        use_vmap=use_vmap,
                    )
                    val_ipa_loss = val_ipa_result["loss"]
                    val_objective = (
                        val_vae_loss + self.lambda1 * val_ipa_loss
                        if strategy == "lr_ipa"
                        else self.lambda1 * val_ipa_loss
                    )
                    val_objectives.append(float(val_objective.cpu()))
                    val_vae_losses.append(float(val_vae_loss.cpu()))
                    val_ipa_losses.append(float(val_ipa_loss.cpu()))

            if not train_objectives or not val_objectives:
                raise ValueError("training and validation loaders must be non-empty")
            train_loss = float(np.mean(train_objectives))
            val_loss = float(np.mean(val_objectives))
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["vae_loss"].append(float(np.mean(train_vae_losses)))
            history["ipa_loss"].append(float(np.mean(train_ipa_losses)))
            history["val_vae_loss"].append(float(np.mean(val_vae_losses)))
            history["val_ipa_loss"].append(float(np.mean(val_ipa_losses)))
            history["ipa_gradient_norm"].append(float(np.mean(gradient_norms)))
            history["k"].append(k)
            history["m"].append(m)

            if verbose and (epoch == 0 or (epoch + 1) % 20 == 0):
                print(
                    f"Batched {strategy} epoch {epoch + 1}: "
                    f"train={train_loss:.6f}, val={val_loss:.6f}, "
                    f"ipa={history['ipa_loss'][-1]:.6f}, K={k}, M={m}"
                )

            if val_loss < best_loss - 1e-12:
                best_loss = val_loss
                best_epoch = epoch
                stale_epochs = 0
                torch.save(self.state_dict(), best_model_path)
            else:
                stale_epochs += 1
                if stale_epochs >= early_stopping:
                    break

        self.loss_history = history
        self.batched_ipa_best_epoch = best_epoch
        self.batched_ipa_best_loss = best_loss
        pd.DataFrame(history).to_excel(loss_path, index=False)
        return {
            "loss_history": history,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "best_model_path": best_model_path,
            "loss_path": loss_path,
            "strategy": strategy,
            "k": k,
            "m": m,
        }

    def trainconvae_batched_lr_ipa(
        self,
        num_epochs,
        targetdim,
        traindata_loader,
        valdata_loader,
        early_stopping,
        k=8,
        m=None,
        **kwargs,
    ):
        """Jointly update VAE and averaged K-replication IPA gradients."""

        return self._train_batched_ipa_strategy(
            strategy="lr_ipa",
            num_epochs=num_epochs,
            targetdim=targetdim,
            traindata_loader=traindata_loader,
            valdata_loader=valdata_loader,
            early_stopping=early_stopping,
            k=k,
            m=m,
            **kwargs,
        )

    def traindecoderonly_batched_ipa(
        self,
        num_epochs,
        targetdim,
        traindata_loader,
        valdata_loader,
        early_stopping,
        k=8,
        m=None,
        **kwargs,
    ):
        """Train only the decoder with averaged K-replication IPA gradients."""

        return self._train_batched_ipa_strategy(
            strategy="ipa_only",
            num_epochs=num_epochs,
            targetdim=targetdim,
            traindata_loader=traindata_loader,
            valdata_loader=valdata_loader,
            early_stopping=early_stopping,
            k=k,
            m=m,
            **kwargs,
        )

    def trainconvae_batched_two_stage_ipa(
        self,
        num_epochs,
        targetdim,
        traindata_loader,
        valdata_loader,
        early_stopping,
        k=8,
        m=None,
        **kwargs,
    ):
        """Load a pretrained VAE and fine-tune its decoder with batched IPA."""

        return self._train_batched_ipa_strategy(
            strategy="two_stage_ipa",
            num_epochs=num_epochs,
            targetdim=targetdim,
            traindata_loader=traindata_loader,
            valdata_loader=valdata_loader,
            early_stopping=early_stopping,
            k=k,
            m=m,
            **kwargs,
        )


    def traindecoderonly(self, num_epochs, targetdim, traindata_loader, valdata_loader, early_stopping, 
                    ipa_update_mode='batch', save_name=None, save_interval=50, randomnumber = None):
        best_loss = float('inf')
        early_stopping_counter = 0
        
        print(f"训练模式: IPA更新方式={ipa_update_mode}")
        
        # 确保保存目录存在
        import os
        vae_pth_dir = "MODEL"
        os.makedirs(vae_pth_dir, exist_ok=True)
        best_model_path = os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_best_model.pth")
        save_pth_dir = "lossrecord"
        os.makedirs(save_pth_dir, exist_ok=True)
        loss_csv_path = os.path.join(save_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_loss_history.xlsx")
        self.loss_history = {
            'train_loss': [],
            'val_loss': [],
            'vae_loss': [],
            'quantile_loss': [],
            'total_loss': [],
            'val_vae_loss': [],
            'val_quantile_loss': [],
            'quantile_gradient': [],  # 当 ipa_update_mode == 'batch' 时记录当轮平均梯度范数
            'epoch': [],
            'best_loss_epoch': -1
        }
        
        # 创建分离的优化器：VAE优化器（所有参数）和Decoder优化器（仅decoder参数）
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=1e-3)
        for p in self.encoder.parameters():
            p.requires_grad = False
        whole_losslist = []
        vae_losslist = []
        vae_losslist_val = []
        quantile_losslist = []
        quantile_losslist_val = []
        for epoch in range(num_epochs):
            whole_loss = 0
            epoch_vae_losses = []
            epoch_quantile_losses = []
            epoch_quantile_grads = []
            total_loss = []
            quantile_loss_train = []
            for batch in traindata_loader:
                # Handle data loader format
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                batch_size = batch.shape[0]
                device = next(self.parameters()).device
                batch = batch.to(device)
                im = batch[:, -1].reshape(-1, 1).to(device)  # Target
                im_label = batch[:, :-1].to(device)  # Conditions
                
                # Since we only train decoder, no need for encoder_optimizer
                decoder_optimizer.zero_grad()
                

                zsample_test = torch.randn(batch_size, self.latent).to(device)
                generate_im_test = self.decode(zsample_test, im_label)  
                quantile_loss_test = self.quantile_loss(generate_im_test, im)
                quantile_loss_train.append(quantile_loss_test.item())
                



                # Efficiently sample multiple z for the entire batch to compute quantiles
                # Generate samplingnumber samples per batch item
                z_samples = torch.randn(batch_size * self.samplingnumber, self.latent).to(device)
                im_label_repeated = im_label.repeat(self.samplingnumber, 1)
                
                # Generate images for all samples in one forward pass
                generate_ims = self.decode(z_samples, im_label_repeated)
                
                # Reshape to (batch_size, samplingnumber, ...)
                generate_ims = generate_ims.view(batch_size, self.samplingnumber, *generate_ims.shape[1:])
                
                # Compute quantiles across the sampling dimension (dim=1)
                generate_quantiles = torch.quantile(generate_ims, self.quantiles, dim=1)
                
                # Assuming generate_im is the mean or a single sample for VAE loss, but since only training decoder with quantile loss,
                # we might not need VAE loss. If needed, compute it separately.
                # For now, assuming quantile_loss is the primary loss, and vae_loss is optional or zero.
                # If vae_loss is required, add it here (e.g., using a single sample or mean).
                
                # Compute quantile loss (assuming self.quantile_loss takes predicted quantiles and target)
                # Adjust if it takes mean or something else; based on code, it was using generate_im (single sample?) but now using quantiles.
                quantile_loss = self.quantile_loss(generate_quantiles, im)  # Pinball loss or similar
                
                # Backward and step only on decoder
                quantile_loss.backward()
                decoder_optimizer.step()
                
                # To compute gradient norm for logging, regenerate a small set if needed (to save memory)
                # But to reduce computation, we can reuse or subsample
                # Here, we compute grad norm on a new small batch to avoid large memory
                z_sample_new = torch.randn(batch_size, self.latent).to(device)  # Smaller, single sample per item for grad check
                generate_im_new = self.decode(z_sample_new, im_label)
                generate_im_new.requires_grad_(True)
                
                # For quantile grad, need multiple samples; but to save space, use smaller samplingnumber if possible, or skip if not critical
                # Assuming we need it, use the same efficient way but with smaller batch if memory is issue
                gen_quantile_new = torch.quantile(generate_im_new.unsqueeze(1).repeat(1, self.samplingnumber, 1), self.quantiles, dim=1)  # Wait, this is wrong; need multiple decodes
                
                # Proper way: regenerate multiple for grad norm, but to optimize, perhaps compute on subset
                subset_size = min(8, batch_size)  # Use a small subset to reduce memory for logging only
                z_samples_subset = torch.randn(subset_size * self.samplingnumber, self.latent).to(device)
                im_label_subset = im_label[:subset_size].repeat(self.samplingnumber, 1)
                generate_ims_subset = self.decode(z_samples_subset, im_label_subset).view(subset_size, self.samplingnumber, -1)
                gen_quantile_new = torch.quantile(generate_ims_subset, self.quantiles, dim=1)
                gen_quantile_new.requires_grad_(True)
                
                decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
                quantile_grads = torch.autograd.grad(gen_quantile_new, decoder_params, grad_outputs=torch.ones_like(gen_quantile_new), retain_graph=True, allow_unused=True)
                quantile_grad_norm = sum(torch.norm(g) for g in quantile_grads if g is not None).item()
                
                # 计算 VAE loss（仅用于记录，不反向传播）
                with torch.no_grad():
                    recon_im, mu, logvar = self.forward(im, im_label)
                    vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                epoch_quantile_grads.append(quantile_grad_norm)
                epoch_vae_losses.append(vae_loss.item())
                epoch_quantile_losses.append(quantile_loss.item())
                whole_loss += vae_loss.item() + quantile_loss.item()
                total_loss.append(vae_loss.item() + self.lambda1 * quantile_loss.item())
            quantile_losslist.append(np.mean(quantile_loss_train))
            # Epoch-level logging
            self.loss_history['vae_loss'].append(np.mean(epoch_vae_losses))
            self.loss_history['quantile_loss'].append(np.mean(epoch_quantile_losses))
            self.loss_history['quantile_gradient'].append(np.mean(epoch_quantile_grads))
            self.loss_history['total_loss'].append(np.mean(total_loss))
            self.loss_history['train_loss'].append(whole_loss)
            self.loss_history['epoch'].append(epoch)

            val_loss = 0
            val_vae_losses = []
            val_quantile_losses = []
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    if targetdim == 1:
                        batch = val_batch.to(device)
                        im = batch[:, -1].reshape(-1, 1).to(device)
                        im_label = batch[:, :-1].to(device)
                    else:
                        batch = val_batch.to(device)
                        im = batch[:, -targetdim:].to(device)
                        im_label = batch[:, :-targetdim].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    val_loss += val_vae_loss.item()
                    # 计算验证分位数损失
                    device = next(self.parameters()).device
                    val_generate_im = self.decode(torch.randn(im_label.shape[0], self.latent).to(device), im_label)
                    val_quantile_loss = self.quantile_loss(val_generate_im, im)
                    
                    val_vae_losses.append(val_vae_loss.item())
                    val_quantile_losses.append(val_quantile_loss.item())
            
            self.loss_history['val_vae_loss'].append(np.mean(val_vae_losses))
            self.loss_history['val_quantile_loss'].append(np.mean(val_quantile_losses))
            val_loss /= len(valdata_loader)
            self.loss_history['val_loss'].append(val_loss)

            # 填充返回列表（每轮追加 epoch 均值）
            vae_losslist.append(np.mean(epoch_vae_losses))
            vae_losslist_val.append(np.mean(val_vae_losses))
            quantile_losslist_val.append(np.mean(val_quantile_losses))

            print('epoch: {}, Train QuantileLoss: {:.4f}, Val Loss: {:.4f}, Val QuantileLoss: {:.4f}, VAE Loss: {:.4f}'.format(
                epoch, np.mean(epoch_quantile_losses), val_loss, np.mean(val_quantile_losses), np.mean(epoch_vae_losses)))
            
            loss_new = val_loss
            if loss_new < best_loss:
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                
                # 保存最佳模型（简化版本）
                torch.save(self.state_dict(), best_model_path)
                
                
                print('epoch: {}, find new best loss: Val Loss: {:.4f}'.format(epoch, best_loss))
                print(f'最佳模型已保存到: {best_model_path}')
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
        pd.DataFrame(self.loss_history).to_excel(loss_csv_path, index=False)
        return{
                'loss_history': self.loss_history,
                'vae_losslist': vae_losslist,
                'vae_losslist_val': vae_losslist_val,
                'quantile_losslist': quantile_losslist,
                'quantile_losslist_val': quantile_losslist_val
        }



    def trainconvae_ipa(self, num_epochs, targetdim, traindata_loader, valdata_loader, early_stopping, 
                    ipa_update_mode='batch', save_name=None, save_interval=50,randomnumber = None):
        best_loss = float('inf')
        early_stopping_counter = 0
        
        print(f"训练模式: IPA更新方式={ipa_update_mode}")
        
        # 确保保存目录存在
        import os
        vae_pth_dir = "MODEL"
        os.makedirs(vae_pth_dir, exist_ok=True)
        best_model_path = os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_best_model.pth")
        loss_csv_path = os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_loss_history.xlsx")
        self.loss_history = {
            'train_loss': [],
            'val_loss': [],
            'vae_loss': [],
            'quantile_loss': [],
            'total_loss': [],
            'quantile_gradient': [],  # 当 ipa_update_mode == 'batch' 时记录当轮平均梯度范数
            'epoch': [],
            'best_loss_epoch': -1
        }
        
        # 创建分离的优化器：VAE优化器（所有参数）和Decoder优化器（仅decoder参数）
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        
        whole_losslist = []
        for epoch in range(num_epochs):
            whole_loss = 0
            loss_new = 0
            epoch_vae_losses = []
            epoch_quantile_losses = []
            epoch_quantile_grads = []
            total_loss = []
            
            # 修复：累积梯度用于平均计算
            decoder_optimizer.zero_grad()  # 在epoch开始时清零梯度
            accumulated_quantile_loss = 0.0


            accumulated_grads = {}
            for name, param in self.decoder.named_parameters():  # 假设decoder是self.decoder
                if param.requires_grad:
                    accumulated_grads[name] = torch.zeros_like(param, device=param.device)



            for i, batch in enumerate(traindata_loader):
                # 处理数据加载器返回的数据格式
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # 如果是tuple或list，取第一个元素
                
                batch_size = batch.shape[0]
                device = next(self.parameters()).device
                batch = batch.to(device)
                im = batch[:, -1].reshape(-1, 1).to(device)
                im_label = batch[:, :-1].to(device)
                
                # 第一步：VAE损失和梯度更新（影响所有参数）
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                vae_loss.backward()
                vae_optimizer.step()
                
                # 第二步：计算pinball loss（假设quantile_loss即pinball loss）
                z_sample = torch.randn(im_label.shape[0], self.latent).to(device)
                z_sample.requires_grad_(True)
                generate_im = self.decode(z_sample, im_label)
                pinball_loss = self.quantile_loss(generate_im, im)  # 假设quantile_loss是pinball loss
                
                # 累积pinball loss用于平均计算
                accumulated_quantile_loss += pinball_loss.item()
                
                # 计算当前batch的scaled pinball loss梯度
                scaled_pinball_loss = pinball_loss * self.lambda1
                scaled_pinball_loss.backward()
                
                # 累积当前batch的梯度到accumulated_grads
                for name, param in self.decoder.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        accumulated_grads[name] += param.grad.clone().detach() / len(traindata_loader)  # 除以批次数进行平均
                
                # 清零当前decoder的梯度，以防干扰下一个batch
                decoder_optimizer.zero_grad()
                
                # 计算用于统计的量化梯度

                # 重新计算梯度用于统计
                z_sample_new = torch.randn(im_label.shape[0], self.latent).to(device)
                z_sample_new.requires_grad_(True)
                generate_im_new = self.decode(z_sample_new, im_label)
                gen_quantile_new = torch.quantile(generate_im_new, self.quantiles, dim=0)
                decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
                quantile_grads = torch.autograd.grad(gen_quantile_new, decoder_params, grad_outputs=torch.ones_like(gen_quantile_new), retain_graph=True)
                quantile_grad_norm = sum(torch.norm(g) for g in quantile_grads).item()
                epoch_quantile_grads.append(quantile_grad_norm)

                
                epoch_vae_losses.append(vae_loss.item())
                epoch_quantile_losses.append(pinball_loss.item())
                whole_loss += vae_loss.item()
                total_loss.append(vae_loss.item() + self.lambda1 * pinball_loss.item())

            # 在epoch结束时，将平均梯度加到decoder参数上
            for name, param in self.decoder.named_parameters():
                if param.requires_grad:
                    param.grad = accumulated_grads[name]  # 设置为平均梯度
            decoder_optimizer.step()
            
            self.loss_history['vae_loss'].append(np.mean(epoch_vae_losses))
            self.loss_history['quantile_loss'].append(np.mean(epoch_quantile_losses))
            self.loss_history['quantile_gradient'].append(np.mean(epoch_quantile_grads))
            self.loss_history['total_loss'].append(np.mean(total_loss))
            self.loss_history['train_loss'].append(whole_loss)
            self.loss_history['epoch'].append(epoch)
            
            val_loss = 0
            val_vae_losses = []
            val_quantile_losses = []
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    if targetdim == 1:
                        batch = val_batch.to(device)
                        im = batch[:, -1].reshape(-1, 1).to(device)
                        im_label = batch[:, :-1].to(device)
                    else:
                        batch = val_batch.to(device)
                        im = batch[:, -targetdim:].to(device)
                        im_label = batch[:, :-targetdim].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    val_loss += val_vae_loss.item()
                    # 计算验证分位数损失
                    device = next(self.parameters()).device
                    val_generate_im = self.decode(torch.randn(im_label.shape[0], self.latent).to(device), im_label)
                    val_quantile_loss = self.quantile_loss(val_generate_im, im)
                    
                    val_vae_losses.append(val_vae_loss.item())
                    val_quantile_losses.append(val_quantile_loss.item())
            
            
            val_loss /= len(valdata_loader)
            self.loss_history['val_loss'].append(val_loss)
            if (epoch) % 20 == 0:
                print('epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, VAE Loss: {:.4f}, Quantile Loss: {:.4f}'.format(
                    epoch, whole_loss, val_loss, np.mean(epoch_vae_losses), np.mean(epoch_quantile_losses)))
                print('Avg Quantile Loss for epoch: {:.4f}'.format(accumulated_quantile_loss / len(traindata_loader)))
            
            loss_new = val_loss
            if loss_new < best_loss:
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                
                # 保存最佳模型（简化版本）
                #torch.save(self.state_dict(), best_model_path)
                
                
                print('epoch: {}, find new best loss: Val Loss: {:.4f}'.format(epoch, best_loss))
                print(f'最佳模型已保存到: {best_model_path}')
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
        pd.DataFrame(self.loss_history).to_excel(loss_csv_path, index=False)



    def trainconvae_quantileipa(self, num_epochs, targetdim, traindata_loader, valdata_loader, early_stopping, 
                    ipa_update_mode='batch', save_name=None, save_interval=50,randomnumber = None):
        best_loss = float('inf')
        early_stopping_counter = 0
        
        print(f"训练模式: IPA更新方式={ipa_update_mode}")
        
        # 确保保存目录存在
        import os
        vae_pth_dir = "MODEL"
        os.makedirs(vae_pth_dir, exist_ok=True)
        best_model_path = os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_best_model.pth")
        loss_csv_path = os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_loss_history.xlsx")
        self.loss_history = {
            'train_loss': [],
            'val_loss': [],
            'vae_loss': [],
            'quantile_loss': [],
            'total_loss': [],
            'quantile_gradient': [],  # 当 ipa_update_mode == 'batch' 时记录当轮平均梯度范数
            'epoch': [],
            'best_loss_epoch': -1
        }
        
        # 创建分离的优化器：VAE优化器（所有参数）和Decoder优化器（仅decoder参数）
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        
        whole_losslist = []
        for epoch in range(num_epochs):
            whole_loss = 0
            loss_new = 0
            epoch_vae_losses = []
            epoch_quantile_losses = []
            epoch_quantile_grads = []
            total_loss = []
            
            # 修复：累积梯度用于平均计算
            decoder_optimizer.zero_grad()  # 在epoch开始时清零梯度
            accumulated_quantile_loss = 0.0


            accumulated_grads = {}
            for name, param in self.decoder.named_parameters():  # 假设decoder是self.decoder
                if param.requires_grad:
                    accumulated_grads[name] = torch.zeros_like(param, device=param.device)



            for i, batch in enumerate(traindata_loader):
                # 处理数据加载器返回的数据格式
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # 如果是tuple或list，取第一个元素
                
                batch_size = batch.shape[0]
                device = next(self.parameters()).device
                batch = batch.to(device)
                im = batch[:, -1].reshape(-1, 1).to(device)
                im_label = batch[:, :-1].to(device)
                
                # 第一步：VAE损失和梯度更新（影响所有参数）
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                vae_loss.backward()
                vae_optimizer.step()
                
                # 第二步：计算pinball loss（假设quantile_loss即pinball loss）
                z_sample = torch.randn(im_label.shape[0], self.latent).to(device)
                z_sample.requires_grad_(True)
                generate_im = self.decode(z_sample, im_label)
                pinball_loss = self.quantile_loss(generate_im, im)  # 假设quantile_loss是pinball loss
                

                im.requires_grad_(True)
                sample_quantile = torch.quantile(im, self.quantiles, dim=0)
                gen_quantile = torch.quantile(generate_im, self.quantiles, dim=0)
                quantile_loss = torch.abs(sample_quantile - gen_quantile).mean()



                # 累积pinball loss用于平均计算
                accumulated_quantile_loss += quantile_loss.item()
                
                # 计算当前batch的scaled pinball loss梯度
                scaled_pinball_loss = quantile_loss * self.lambda1
                scaled_pinball_loss.backward()
                
                # 累积当前batch的梯度到accumulated_grads
                for name, param in self.decoder.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        accumulated_grads[name] += param.grad.clone().detach() / len(traindata_loader)  # 除以批次数进行平均
                
                # 清零当前decoder的梯度，以防干扰下一个batch
                decoder_optimizer.zero_grad()
                
                # 计算用于统计的量化梯度

                # 重新计算梯度用于统计
                z_sample_new = torch.randn(im_label.shape[0], self.latent).to(device)
                z_sample_new.requires_grad_(True)
                generate_im_new = self.decode(z_sample_new, im_label)
                gen_quantile_new = torch.quantile(generate_im_new, self.quantiles, dim=0)
                decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
                quantile_grads = torch.autograd.grad(gen_quantile_new, decoder_params, grad_outputs=torch.ones_like(gen_quantile_new), retain_graph=True)
                quantile_grad_norm = sum(torch.norm(g) for g in quantile_grads).item()
                epoch_quantile_grads.append(quantile_grad_norm)

                
                epoch_vae_losses.append(vae_loss.item())
                epoch_quantile_losses.append(pinball_loss.item())
                whole_loss += vae_loss.item()
                total_loss.append(vae_loss.item() + self.lambda1 * pinball_loss.item())

            # 在epoch结束时，将平均梯度加到decoder参数上
            for name, param in self.decoder.named_parameters():
                if param.requires_grad:
                    param.grad = accumulated_grads[name]  # 设置为平均梯度
            decoder_optimizer.step()
            
            self.loss_history['vae_loss'].append(np.mean(epoch_vae_losses))
            self.loss_history['quantile_loss'].append(np.mean(epoch_quantile_losses))
            self.loss_history['quantile_gradient'].append(np.mean(epoch_quantile_grads))
            self.loss_history['total_loss'].append(np.mean(total_loss))
            self.loss_history['train_loss'].append(whole_loss)
            self.loss_history['epoch'].append(epoch)
            
            val_loss = 0
            val_vae_losses = []
            val_quantile_losses = []
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    if targetdim == 1:
                        batch = val_batch.to(device)
                        im = batch[:, -1].reshape(-1, 1).to(device)
                        im_label = batch[:, :-1].to(device)
                    else:
                        batch = val_batch.to(device)
                        im = batch[:, -targetdim:].to(device)
                        im_label = batch[:, :-targetdim].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    val_loss += val_vae_loss.item()
                    # 计算验证分位数损失
                    device = next(self.parameters()).device
                    val_generate_im = self.decode(torch.randn(im_label.shape[0], self.latent).to(device), im_label)
                    val_quantile_loss = self.quantile_loss(val_generate_im, im)
                    
                    val_vae_losses.append(val_vae_loss.item())
                    val_quantile_losses.append(val_quantile_loss.item())
            
            
            val_loss /= len(valdata_loader)
            self.loss_history['val_loss'].append(val_loss)
            if (epoch) % 20 == 0:
                print('epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, VAE Loss: {:.4f}, Quantile Loss: {:.4f}'.format(
                    epoch, whole_loss, val_loss, np.mean(epoch_vae_losses), np.mean(epoch_quantile_losses)))
                print('Avg Quantile Loss for epoch: {:.4f}'.format(accumulated_quantile_loss / len(traindata_loader)))
            
            loss_new = val_loss
            if loss_new < best_loss:
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                
                # 保存最佳模型（简化版本）
                #torch.save(self.state_dict(), best_model_path)
                
                
                print('epoch: {}, find new best loss: Val Loss: {:.4f}'.format(epoch, best_loss))
                print(f'最佳模型已保存到: {best_model_path}')
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
        pd.DataFrame(self.loss_history).to_excel(loss_csv_path, index=False)

    
    
    
    
    def trainconvae_sgd_2(self, num_epochs, targetdim, traindata_loader, valdata_loader, early_stopping, 
                     save_name=None, save_interval=50,randomnumber = None,if_test_lambda = False):
        best_loss = float('inf')
        early_stopping_counter = 0
        
        
        # 确保保存目录存在
        import os
        if if_test_lambda:
            vae_pth_dir = "lambda"
        else:
            vae_pth_dir = "MODEL"

        os.makedirs(vae_pth_dir, exist_ok=True)
        best_model_path = os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_best_model.pth")
        loss_csv_path = os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{self.lambda1}_{randomnumber}_loss_history.xlsx")
        self.loss_history = {
            'vae_loss': [],
            'quantile_loss': [],
            'val_vae_loss': [],
            'val_quantile_loss': [],
        }
        
        # 创建分离的优化器：VAE优化器（所有参数）和Decoder优化器（仅decoder参数）
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        
        whole_losslist = []
        for epoch in range(num_epochs):
            whole_loss = 0
            loss_new = 0
            epoch_vae_losses = []
            epoch_quantile_losses = []
            epoch_quantile_grads = []
            total_loss = []
            ipa_losses = [] 
            # 修复：累积梯度用于平均计算
            decoder_optimizer.zero_grad()  # 在epoch开始时清零梯度
            accumulated_quantile_loss = 0.0


            accumulated_grads = {}
            for name, param in self.decoder.named_parameters():  # 假设decoder是self.decoder
                if param.requires_grad:
                    accumulated_grads[name] = torch.zeros_like(param, device=param.device)
            
            vae_loss_list = []
            quantile_loss_list = []

            for i, batch in enumerate(traindata_loader):
                # 处理数据加载器返回的数据格式
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # 如果是tuple或list，取第一个元素
                
                batch_size = batch.shape[0]
                device = next(self.parameters()).device
                batch = batch.to(device)
                im = batch[:, -1].reshape(-1, 1).to(device)
                im_label = batch[:, :-1].to(device)
                
                # 第一步：VAE损失和梯度更新（影响所有参数）
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]

                quantile_loss = self.quantile_loss(recon_im, im)  # 计算重建图像的分位数损失
                total_loss_new = vae_loss + self.lambda1 * quantile_loss
                total_loss_new.backward()
                vae_optimizer.step()
                vae_loss_list.append(vae_loss.item())
                quantile_loss_list.append(quantile_loss.item())



            
            self.loss_history['vae_loss'].append(np.mean(vae_loss_list))
            self.loss_history['quantile_loss'].append(np.mean(quantile_loss_list))



            val_loss = 0
            val_vae_losses = []
            val_quantile_losses = []
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    if targetdim == 1:
                        batch = val_batch.to(device)
                        im = batch[:, -1].reshape(-1, 1).to(device)
                        im_label = batch[:, :-1].to(device)
                    else:
                        batch = val_batch.to(device)
                        im = batch[:, -targetdim:].to(device)
                        im_label = batch[:, :-targetdim].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    val_loss += val_vae_loss.item()
                    # 计算验证分位数损失
                    device = next(self.parameters()).device
                    val_generate_im = self.decode(torch.randn(im_label.shape[0], self.latent).to(device), im_label)
                    val_quantile_loss = self.quantile_loss(val_generate_im, im)
                    
                    val_vae_losses.append(val_vae_loss.item())
                    val_quantile_losses.append(val_quantile_loss.item())
            
            
            val_loss /= len(valdata_loader)
            self.loss_history['val_vae_loss'].append(np.mean(val_vae_losses))
            self.loss_history['val_quantile_loss'].append(np.mean(val_quantile_losses))
            if (epoch) % 20 == 0:
                print('epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, VAE Loss: {:.4f}, Quantile Loss: {:.4f}'.format(
                    epoch, whole_loss, val_loss, np.mean(epoch_vae_losses), np.mean(epoch_quantile_losses)))
                print('Avg Quantile Loss for epoch: {:.4f}'.format(accumulated_quantile_loss / len(traindata_loader)))
            
            loss_new = val_loss
            if loss_new < best_loss:
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                
                # 保存最佳模型（简化版本）
                torch.save(self.state_dict(), best_model_path)
                
                
                print('epoch: {}, find new best loss: Val Loss: {:.4f}'.format(epoch, best_loss))
                print(f'最佳模型已保存到: {best_model_path}')
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
        pd.DataFrame(self.loss_history).to_excel(loss_csv_path, index=False)
        
    

    
    def trainconvae_sgd(self, num_epochs, targetdim, traindata_loader, valdata_loader, early_stopping, 
                     save_name=None, save_interval=50,randomnumber = None,if_test_lambda = False, if_test_sample = False):
        best_loss = float('inf')
        early_stopping_counter = 0
        
        
        # 确保保存目录存在
        import os
        if if_test_lambda:
            vae_pth_dir = "lambda"
        elif if_test_sample:
            vae_pth_dir = "sample"
        else:
            vae_pth_dir = "MODEL"
        os.makedirs(vae_pth_dir, exist_ok=True)
        best_model_path = os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_best_model.pth")
        loss_csv_path = os.path.join("lossrecord", f"{save_name}_{self.targetdim}_{self.labeldim}_{self.lambda1}_{self.samplingnumber}_{randomnumber}_loss_history.xlsx")
        self.loss_history = {
            'train_loss': [],
            'val_loss': [],
            'vae_loss': [],
            'quantile_loss': [],
            'val_vae_loss': [],
            'val_quantile_loss': [],
            'ipa_loss': [],
            'total_loss': [],
            'quantile_gradient': [],  # 当 ipa_update_mode == 'batch' 时记录当轮平均梯度范数
            'epoch': [],
            'best_loss_epoch': -1
        }
        
        # 创建分离的优化器：VAE优化器（所有参数）和Decoder优化器（仅decoder参数）
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=1e-3)
        
        whole_losslist = []
        start_time_list = []
        gradient_time_list = []
        for epoch in range(num_epochs):
            whole_loss = 0
            loss_new = 0
            epoch_vae_losses = []
            epoch_quantile_losses = []
            epoch_quantile_grads = []
            total_loss = []
            ipa_losses = []
            # 修复：累积梯度用于平均计算
            decoder_optimizer.zero_grad()  # 在epoch开始时清零梯度
            accumulated_quantile_loss = 0.0


            accumulated_grads = {}
            for name, param in self.decoder.named_parameters():  # 假设decoder是self.decoder
                if param.requires_grad:
                    accumulated_grads[name] = torch.zeros_like(param, device=param.device)



            for i, batch in enumerate(traindata_loader):
                # 处理数据加载器返回的数据格式
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # 如果是tuple或list，取第一个元素
                
                batch_size = batch.shape[0]
                device = next(self.parameters()).device
                batch = batch.to(device)
                im = batch[:, -1].reshape(-1, 1).to(device)
                im_label = batch[:, :-1].to(device)
                
                # 第一步：VAE损失和梯度更新（影响所有参数）
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                vae_loss.backward()
                vae_optimizer.step()
                
                
                # for p in self.encoder.parameters():
                #     p.requires_grad = False
                decoder_optimizer.zero_grad()
                
                # Efficiently sample multiple z for the entire batch to compute quantiles
                # Generate samplingnumber samples per batch item
                z_samples = torch.randn(batch_size * self.samplingnumber, self.latent).to(device)
                im_label_repeated = im_label.repeat(self.samplingnumber, 1)
                
                # Generate images for all samples in one forward pass
                generate_ims = self.decode(z_samples, im_label_repeated)
                
                # Reshape to (batch_size, samplingnumber, ...)
                generate_ims = generate_ims.view(batch_size, self.samplingnumber, *generate_ims.shape[1:])
                
                # Compute quantiles across the sampling dimension (dim=1)
                generate_quantiles = torch.quantile(generate_ims, self.quantiles, dim=1)
                

                quantile_loss = self.quantile_loss(generate_quantiles, im)* self.lambda1  # Pinball loss or similar
                # Backward and step only on decoder
                quantile_loss.backward()
                decoder_optimizer.step()
                start_time = time.time()
                with torch.no_grad():
                    z_samples = torch.randn(batch_size * self.samplingnumber, self.latent).to(device)
                    im_label_repeated = im_label.repeat(self.samplingnumber, 1)
                    
                    # Generate images for all samples in one forward pass
                    generate_ims = self.decode(z_samples, im_label_repeated)
                    
                    # Reshape to (batch_size, samplingnumber, ...)
                    generate_ims = generate_ims.view(batch_size, self.samplingnumber, *generate_ims.shape[1:])
                    
                    # Compute quantiles across the sampling dimension (dim=1)
                    generate_quantiles = torch.quantile(generate_ims, self.quantiles, dim=1)
                    
                    # Assuming generate_im is the mean or a single sample for VAE loss, but since only training decoder with quantile loss,
                    # we might not need VAE loss. If needed, compute it separately.
                    # For now, assuming quantile_loss is the primary loss, and vae_loss is optional or zero.
                    # If vae_loss is required, add it here (e.g., using a single sample or mean).
                    
                    # Compute quantile loss (assuming self.quantile_loss takes predicted quantiles and target)
                    # Adjust if it takes mean or something else; based on code, it was using generate_im (single sample?) but now using quantiles.
                    ipa_loss = self.quantile_loss(generate_quantiles, im)  # Pinball loss or similar
                    ipa_losses.append(ipa_loss.item())


                    z_sample_new = torch.randn(im_label.shape[0], self.latent).to(device)
                    z_sample_new.requires_grad_(True)
                    generate_im_new = self.decode(z_sample_new, im_label)
                    pinball_loss = self.quantile_loss(generate_im_new, im)  # 假设quantile_loss是pinball loss
                    epoch_quantile_losses.append(pinball_loss.item())
                end_time = time.time()
                start_time_list.append(end_time - start_time)


                gradient_start_time = time.time()
                z_sample_new = torch.randn(im_label.shape[0], self.latent).to(device)
                z_sample_new.requires_grad_(True)
                generate_im_new = self.decode(z_sample_new, im_label)
                gen_quantile_new = torch.quantile(generate_im_new, self.quantiles, dim=0)
                decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
                quantile_grads = torch.autograd.grad(gen_quantile_new, decoder_params, grad_outputs=torch.ones_like(gen_quantile_new), retain_graph=True)
                gradient_time_list.append(time.time() - gradient_start_time)
                quantile_grad_norm = sum(torch.norm(g) for g in quantile_grads).item()
                epoch_quantile_grads.append(quantile_grad_norm)
                
                
                epoch_vae_losses.append(vae_loss.item())
                whole_loss += vae_loss.item()
                total_loss.append(vae_loss.item() + self.lambda1 * quantile_loss.item())


            
            self.loss_history['vae_loss'].append(np.mean(epoch_vae_losses))
            self.loss_history['quantile_loss'].append(np.mean(epoch_quantile_losses))
            self.loss_history['quantile_gradient'].append(np.mean(epoch_quantile_grads))
            self.loss_history['total_loss'].append(np.mean(total_loss))
            self.loss_history['train_loss'].append(whole_loss)
            self.loss_history['epoch'].append(epoch)
            self.loss_history['ipa_loss'].append(np.mean(ipa_losses))
            val_loss = 0
            val_vae_losses = []
            val_quantile_losses = []
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    if targetdim == 1:
                        batch = val_batch.to(device)
                        im = batch[:, -1].reshape(-1, 1).to(device)
                        im_label = batch[:, :-1].to(device)
                    else:
                        batch = val_batch.to(device)
                        im = batch[:, -targetdim:].to(device)
                        im_label = batch[:, :-targetdim].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    val_loss += val_vae_loss.item()
                    # 计算验证分位数损失
                    device = next(self.parameters()).device
                    val_generate_im = self.decode(torch.randn(im_label.shape[0], self.latent).to(device), im_label)
                    val_quantile_loss = self.quantile_loss(val_generate_im, im)
                    
                    val_vae_losses.append(val_vae_loss.item())
                    val_quantile_losses.append(val_quantile_loss.item())
            
            self.loss_history['val_vae_loss'].append(np.mean(val_vae_losses))
            self.loss_history['val_quantile_loss'].append(np.mean(val_quantile_losses))
            val_loss /= len(valdata_loader)
            self.loss_history['val_loss'].append(val_loss)
            if (epoch) % 20 == 0:
                print('epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, VAE Loss: {:.4f}, Quantile Loss: {:.4f}'.format(
                    epoch, whole_loss, val_loss, np.mean(epoch_vae_losses), np.mean(epoch_quantile_losses)))
                print('Avg Quantile Loss for epoch: {:.4f}'.format(accumulated_quantile_loss / len(traindata_loader)))
            
            loss_new = val_loss
            if loss_new < best_loss:
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                
                # 保存最佳模型（简化版本）
                torch.save(self.state_dict(), best_model_path)
                
                
                print('epoch: {}, find new best loss: Val Loss: {:.4f}'.format(epoch, best_loss))
                print(f'最佳模型已保存到: {best_model_path}')
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
        self.loss_history['avg_innerloop_time'] = np.mean(start_time_list) if start_time_list else 0.0
        self.loss_history['avg_gradient_time'] = np.mean(gradient_time_list) if gradient_time_list else 0.0
        os.makedirs("lossrecord", exist_ok=True)
        pd.DataFrame(self.loss_history).to_excel(loss_csv_path, index=False)
        return self.loss_history, best_model_path, np.mean(start_time_list)



    def trainconvae_sgd_franke_wolfe(self, num_epochs, targetdim, traindata_loader, valdata_loader, early_stopping, 
                     save_name=None, save_interval=50, randomnumber=None, initial_lambda_fw=1/2, 
                     lambda_fw_decay=0.1, lambda_fw_update_mode='standard',if_test_lambda = False):
        best_loss = float('inf')
        early_stopping_counter = 0
        
        
        # 确保保存目录存在
        import os
        if if_test_lambda:
            vae_pth_dir = "lambda"
        else:
            vae_pth_dir = "lossrecord"
        
        # 初始化lambda_fw参数
        self.lambda_fw = initial_lambda_fw
        self.lambda_fw_history = []
        
        os.makedirs(vae_pth_dir, exist_ok=True)
        best_model_path = os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_best_model.pth")
        loss_csv_path = os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_loss_history.xlsx")
        
        self.loss_history = {
            'train_loss': [],
            'val_loss': [],
            'vae_loss': [],
            'quantile_loss': [],
            'val_vae_loss': [],
            'val_quantile_loss': [],
            'ipa_loss': [],
            'total_loss': [],
            'quantile_gradient': [],
            'lambda_fw': [],  # 记录lambda_fw的变化
            'epoch': [],
            'best_loss_epoch': -1
        }
        
        # 创建优化器
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        
        # 用于自适应lambda_fw更新的变量
        prev_quantile_loss = None
        quantile_loss_momentum = 0.0
        
        for epoch in range(num_epochs):
            whole_loss = 0
            epoch_vae_losses = []
            epoch_quantile_losses = []
            epoch_quantile_grads = []
            total_loss = []
            ipa_losses = []

            self.lambda_fw = torch.tensor(self.lambda_fw, dtype=torch.float32)
            self.lambda_fw_history.append(self.lambda_fw)
            
            for i, batch in enumerate(traindata_loader):
                # 处理数据格式
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                batch_size = batch.shape[0]
                device = next(self.parameters()).device
                batch = batch.to(device)
                im = batch[:, -1].reshape(-1, 1).to(device)
                im_label = batch[:, :-1].to(device)
                
                # VAE训练步骤
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                
                # 计算分位数损失
                z_samples = torch.randn(batch_size * self.samplingnumber, self.latent).to(device)
                im_label_repeated = im_label.repeat(self.samplingnumber, 1)
                
                # 生成样本
                generate_ims = self.decode(z_samples, im_label_repeated)
                generate_ims = generate_ims.view(batch_size, self.samplingnumber, *generate_ims.shape[1:])
                
                # 计算分位数
                generate_quantiles = torch.quantile(generate_ims, self.quantiles, dim=1)
                quantile_loss = self.quantile_loss(generate_quantiles, im)
                


                lambda_fw = self.lambda_fw.clone().detach().requires_grad_(True)
                total_loss_c = (1-lambda_fw)*vae_loss + lambda_fw*quantile_loss
                grad_lambda = torch.autograd.grad(total_loss_c, lambda_fw)[0].item()

                # 3. 线性最小化得到极点 s
                s = 1.0 if grad_lambda < 0 else 0.0

                # 4. 步长
                # ---- 4.a 经典 FW 步长（你已经在用）
                k = len(self.lambda_fw_history) - 1          # 当前已经记录的次数

                # ---- 4.b 或者线搜索（取消注释使用）
                gamma = self.line_search(vae_loss.item(), quantile_loss.item(),
                                     self.lambda_fw, s)

                # 5. 凸组合更新
                self.lambda_fw = (1 - gamma) * self.lambda_fw + gamma * s
                self.lambda_fw = torch.clamp(self.lambda_fw, 0.0, 1.0)   # 防止数值漂移

                # 6. 记录历史（你已有）
                self.lambda_fw_history.append(self.lambda_fw.item())
            




                # Frank-Wolfe组合损失
                total_loss_batch = (1 - self.lambda_fw) * vae_loss + self.lambda_fw * quantile_loss                
                # 反向传播和更新
                total_loss_batch.backward()
                vae_optimizer.step()
                


                with torch.no_grad():
                    z_samples = torch.randn(batch_size * self.samplingnumber, self.latent).to(device)
                    im_label_repeated = im_label.repeat(self.samplingnumber, 1)
                    
                    # Generate images for all samples in one forward pass
                    generate_ims = self.decode(z_samples, im_label_repeated)
                    
                    # Reshape to (batch_size, samplingnumber, ...)
                    generate_ims = generate_ims.view(batch_size, self.samplingnumber, *generate_ims.shape[1:])
                    
                    # Compute quantiles across the sampling dimension (dim=1)
                    generate_quantiles = torch.quantile(generate_ims, self.quantiles, dim=1)
                    
                    # Assuming generate_im is the mean or a single sample for VAE loss, but since only training decoder with quantile loss,
                    # we might not need VAE loss. If needed, compute it separately.
                    # For now, assuming quantile_loss is the primary loss, and vae_loss is optional or zero.
                    # If vae_loss is required, add it here (e.g., using a single sample or mean).
                    
                    # Compute quantile loss (assuming self.quantile_loss takes predicted quantiles and target)
                    # Adjust if it takes mean or something else; based on code, it was using generate_im (single sample?) but now using quantiles.
                    ipa_loss = self.quantile_loss(generate_quantiles, im)  # Pinball loss or similar
                    ipa_losses.append(ipa_loss.item())


                    z_sample_new = torch.randn(im_label.shape[0], self.latent).to(device)
                    z_sample_new.requires_grad_(True)
                    generate_im_new = self.decode(z_sample_new, im_label)
                    pinball_loss = self.quantile_loss(generate_im_new, im)  # 假设quantile_loss是pinball loss
                    epoch_quantile_losses.append(pinball_loss.item())


                    
                # 计算梯度范数用于记录
                z_sample_grad = torch.randn(im_label.shape[0], self.latent).to(device)
                z_sample_grad.requires_grad_(True)
                generate_im_grad = self.decode(z_sample_grad, im_label)
                gen_quantile_grad = torch.quantile(generate_im_grad, self.quantiles, dim=0)
                
                decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
                try:
                    quantile_grads = torch.autograd.grad(gen_quantile_grad, decoder_params, 
                                                       grad_outputs=torch.ones_like(gen_quantile_grad), 
                                                       retain_graph=True, allow_unused=True)
                    quantile_grad_norm = sum(torch.norm(g) for g in quantile_grads if g is not None).item()
                except:
                    quantile_grad_norm = 0.0
                
                # 记录损失
                epoch_vae_losses.append(vae_loss.item())
                epoch_quantile_grads.append(quantile_grad_norm)
                whole_loss += total_loss_batch.item()
                total_loss.append(total_loss_batch.item())
            
            # 更新用于自适应lambda_fw的变量
            current_avg_quantile_loss = np.mean(epoch_quantile_losses)
            prev_quantile_loss = current_avg_quantile_loss
            
            # 记录历史
            self.loss_history['vae_loss'].append(np.mean(epoch_vae_losses))
            self.loss_history['quantile_loss'].append(current_avg_quantile_loss)
            self.loss_history['ipa_loss'].append(np.mean(ipa_losses))
            self.loss_history['quantile_gradient'].append(np.mean(epoch_quantile_grads))
            self.loss_history['total_loss'].append(np.mean(total_loss))
            self.loss_history['train_loss'].append(whole_loss)
            self.loss_history['lambda_fw'].append(self.lambda_fw)
            self.loss_history['epoch'].append(epoch)
            
            val_loss = 0
            val_vae_losses = []
            val_quantile_losses = []
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    if targetdim == 1:
                        batch = val_batch.to(device)
                        im = batch[:, -1].reshape(-1, 1).to(device)
                        im_label = batch[:, :-1].to(device)
                    else:
                        batch = val_batch.to(device)
                        im = batch[:, -targetdim:].to(device)
                        im_label = batch[:, :-targetdim].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    val_loss += val_vae_loss.item()
                    # 计算验证分位数损失
                    device = next(self.parameters()).device
                    val_generate_im = self.decode(torch.randn(im_label.shape[0], self.latent).to(device), im_label)
                    val_quantile_loss = self.quantile_loss(val_generate_im, im)
                    
                    val_vae_losses.append(val_vae_loss.item())
                    val_quantile_losses.append(val_quantile_loss.item())
            
            
            val_loss /= len(valdata_loader)
            self.loss_history['val_vae_loss'].append(np.mean(val_vae_losses))
            self.loss_history['val_quantile_loss'].append(np.mean(val_quantile_losses))
            self.loss_history['val_loss'].append(val_loss)
            if (epoch) % 20 == 0:
                print('epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, VAE Loss: {:.4f}, Quantile Loss: {:.4f}'.format(
                    epoch, whole_loss, val_loss, np.mean(epoch_vae_losses), np.mean(epoch_quantile_losses)))

            
            loss_new = val_loss
            if loss_new < best_loss:
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                
                # 保存最佳模型（简化版本）
                #torch.save(self.state_dict(), best_model_path)
                
                
                print('epoch: {}, find new best loss: Val Loss: {:.4f}'.format(epoch, best_loss))
                print(f'最佳模型已保存到: {best_model_path}')
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
        pd.DataFrame(self.loss_history).to_excel(loss_csv_path, index=False)


    
    def trainconvae_sgd_withpretrain(self, num_epochs, targetdim, traindata_loader, valdata_loader, early_stopping, 
                     save_name=None, save_interval=50, randomnumber=None, if_test_lambda=False,
                     if_test_sample=False, pretrain_save_name=None, fine_tune_mode="all",
                     custom_trainable_prefixes=None, decoder_lr=1e-3):
        best_loss = float('inf')
        early_stopping_counter = 0
        
        # 确保保存目录存在
        import os
        if if_test_lambda:
            vae_pth_dir = "lambda"
        elif if_test_sample:
            vae_pth_dir = "sample"
        else:
            vae_pth_dir = "MODEL"
        os.makedirs(vae_pth_dir, exist_ok=True)
        best_model_path = os.path.join(vae_pth_dir, f"{save_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_best_model.pth")
        os.makedirs("lossrecord", exist_ok=True)
        loss_csv_path = os.path.join("lossrecord", f"{save_name}_{self.targetdim}_{self.labeldim}_{self.lambda1}_{self.samplingnumber}_{randomnumber}_loss_history.xlsx")
        self.loss_history = {
            'train_loss': [],
            'val_loss': [],
            'vae_loss': [],
            'quantile_loss': [],
            'val_vae_loss': [],
            'val_quantile_loss': [],
            'ipa_loss': [],
            'total_loss': [],
            'quantile_gradient': [],  # 当 ipa_update_mode == 'batch' 时记录当轮平均梯度范数
            'epoch': [],
            'best_loss_epoch': -1
        }
        
        # 从 trainconvae 保存的最优模型加载参数
        load_name = pretrain_save_name if pretrain_save_name is not None else save_name
        pretrain_model_path = os.path.join("MODEL", f"{load_name}_{self.targetdim}_{self.labeldim}_{randomnumber}_best_model.pth")
        
        if os.path.exists(pretrain_model_path):
            print(f"加载预训练最优模型: {pretrain_model_path}")
            self.load_state_dict(torch.load(pretrain_model_path))
        else:
            print(f"警告: 预训练模型不存在: {pretrain_model_path}，将从随机初始化开始训练")
        
        # 冻结 encoder 参数，只训练 decoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder 已冻结，仅训练 Decoder")

        # 控制 decoder 的可训练参数，实现“只微调关键层”
        trainable_decoder_params = []
        trainable_names = []
        for name, param in self.decoder.named_parameters():
            should_train = True
            if fine_tune_mode == "last_layer":
                should_train = name.startswith("linear3")
            elif fine_tune_mode == "bias_only":
                should_train = name.endswith("bias")
            elif fine_tune_mode == "custom":
                prefixes = custom_trainable_prefixes if custom_trainable_prefixes is not None else []
                should_train = any(name.startswith(pfx) for pfx in prefixes)
            elif fine_tune_mode == "all":
                should_train = True
            else:
                raise ValueError(f"Unsupported fine_tune_mode: {fine_tune_mode}")

            param.requires_grad = should_train
            if should_train:
                trainable_decoder_params.append(param)
                trainable_names.append(name)

        if len(trainable_decoder_params) == 0:
            raise ValueError("No trainable decoder parameters selected. Please adjust fine_tune_mode/custom_trainable_prefixes.")

        print(f"Decoder fine-tune mode: {fine_tune_mode}, lr={decoder_lr}")
        print(f"Trainable decoder params: {trainable_names}")
        
        # 创建仅更新 decoder 的优化器
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        decoder_optimizer = torch.optim.SGD(trainable_decoder_params, lr=decoder_lr)
        
        whole_losslist = []
        for epoch in range(num_epochs):
            whole_loss = 0
            loss_new = 0
            epoch_vae_losses = []
            epoch_quantile_losses = []
            epoch_quantile_grads = []
            total_loss = []
            ipa_losses = []
            # 修复：累积梯度用于平均计算
            decoder_optimizer.zero_grad()  # 在epoch开始时清零梯度
            accumulated_quantile_loss = 0.0

            accumulated_grads = {}
            for name, param in self.decoder.named_parameters():  # 假设decoder是self.decoder
                if param.requires_grad:
                    accumulated_grads[name] = torch.zeros_like(param, device=param.device)
        

            for i, batch in enumerate(traindata_loader):
                # 处理数据加载器返回的数据格式
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # 如果是tuple或list，取第一个元素
                
                batch_size = batch.shape[0]
                device = next(self.parameters()).device
                batch = batch.to(device)
                im = batch[:, -1].reshape(-1, 1).to(device)
                im_label = batch[:, :-1].to(device)
                
                # 第一步：VAE损失和梯度更新（影响所有参数）
                
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]

                
                
                # for p in self.encoder.parameters():
                #     p.requires_grad = False
                decoder_optimizer.zero_grad()
                
                # Efficiently sample multiple z for the entire batch to compute quantiles
                # Generate samplingnumber samples per batch item
                z_samples = torch.randn(batch_size * self.samplingnumber, self.latent).to(device)
                im_label_repeated = im_label.repeat(self.samplingnumber, 1)
                
                # Generate images for all samples in one forward pass
                generate_ims = self.decode(z_samples, im_label_repeated)
                
                # Reshape to (batch_size, samplingnumber, ...)
                generate_ims = generate_ims.view(batch_size, self.samplingnumber, *generate_ims.shape[1:])
                
                # Compute quantiles across the sampling dimension (dim=1)
                generate_quantiles = torch.quantile(generate_ims, self.quantiles, dim=1)
                

                quantile_loss = self.quantile_loss(generate_quantiles, im)* self.lambda1  # Pinball loss or similar
                # Backward and step only on decoder
                quantile_loss.backward()
                decoder_optimizer.step()

                with torch.no_grad():
                    z_samples = torch.randn(batch_size * self.samplingnumber, self.latent).to(device)
                    im_label_repeated = im_label.repeat(self.samplingnumber, 1)
                    
                    # Generate images for all samples in one forward pass
                    generate_ims = self.decode(z_samples, im_label_repeated)
                    
                    # Reshape to (batch_size, samplingnumber, ...)
                    generate_ims = generate_ims.view(batch_size, self.samplingnumber, *generate_ims.shape[1:])
                    
                    # Compute quantiles across the sampling dimension (dim=1)
                    generate_quantiles = torch.quantile(generate_ims, self.quantiles, dim=1)
                    
                    # Assuming generate_im is the mean or a single sample for VAE loss, but since only training decoder with quantile loss,
                    # we might not need VAE loss. If needed, compute it separately.
                    # For now, assuming quantile_loss is the primary loss, and vae_loss is optional or zero.
                    # If vae_loss is required, add it here (e.g., using a single sample or mean).
                    
                    # Compute quantile loss (assuming self.quantile_loss takes predicted quantiles and target)
                    # Adjust if it takes mean or something else; based on code, it was using generate_im (single sample?) but now using quantiles.
                    ipa_loss = self.quantile_loss(generate_quantiles, im)  # Pinball loss or similar
                    ipa_losses.append(ipa_loss.item())


                    z_sample_new = torch.randn(im_label.shape[0], self.latent).to(device)
                    z_sample_new.requires_grad_(True)
                    generate_im_new = self.decode(z_sample_new, im_label)
                    pinball_loss = self.quantile_loss(generate_im_new, im)  # 假设quantile_loss是pinball loss
                    epoch_quantile_losses.append(pinball_loss.item())



                z_sample_new = torch.randn(im_label.shape[0], self.latent).to(device)
                z_sample_new.requires_grad_(True)
                generate_im_new = self.decode(z_sample_new, im_label)
                gen_quantile_new = torch.quantile(generate_im_new, self.quantiles, dim=0)
                quantile_grads = torch.autograd.grad(gen_quantile_new, trainable_decoder_params, grad_outputs=torch.ones_like(gen_quantile_new), retain_graph=True)
                quantile_grad_norm = sum(torch.norm(g) for g in quantile_grads).item()
                epoch_quantile_grads.append(quantile_grad_norm)
                
                
                epoch_vae_losses.append(vae_loss.item())
                whole_loss += vae_loss.item()
                total_loss.append(vae_loss.item() + self.lambda1 * quantile_loss.item())


            
            self.loss_history['vae_loss'].append(np.mean(epoch_vae_losses))
            self.loss_history['quantile_loss'].append(np.mean(epoch_quantile_losses))
            self.loss_history['quantile_gradient'].append(np.mean(epoch_quantile_grads))
            self.loss_history['total_loss'].append(np.mean(total_loss))
            self.loss_history['train_loss'].append(whole_loss)
            self.loss_history['epoch'].append(epoch)
            self.loss_history['ipa_loss'].append(np.mean(ipa_losses))
            val_loss = 0
            val_vae_losses = []
            val_quantile_losses = []
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    if targetdim == 1:
                        batch = val_batch.to(device)
                        im = batch[:, -1].reshape(-1, 1).to(device)
                        im_label = batch[:, :-1].to(device)
                    else:
                        batch = val_batch.to(device)
                        im = batch[:, -targetdim:].to(device)
                        im_label = batch[:, :-targetdim].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    val_loss += val_vae_loss.item()
                    # 计算验证分位数损失
                    device = next(self.parameters()).device
                    val_generate_im = self.decode(torch.randn(im_label.shape[0], self.latent).to(device), im_label)
                    val_quantile_loss = self.quantile_loss(val_generate_im, im)
                    
                    val_vae_losses.append(val_vae_loss.item())
                    val_quantile_losses.append(val_quantile_loss.item())
            
            self.loss_history['val_vae_loss'].append(np.mean(val_vae_losses))
            self.loss_history['val_quantile_loss'].append(np.mean(val_quantile_losses))
            val_loss /= len(valdata_loader)
            self.loss_history['val_loss'].append(val_loss)
            if (epoch) % 20 == 0:
                print('epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, VAE Loss: {:.4f}, Quantile Loss: {:.4f}'.format(
                    epoch, whole_loss, val_loss, np.mean(epoch_vae_losses), np.mean(epoch_quantile_losses)))
                print('Avg Quantile Loss for epoch: {:.4f}'.format(accumulated_quantile_loss / len(traindata_loader)))
            
            loss_new = val_loss
            if loss_new < best_loss:
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                
                # 保存最佳模型（简化版本）
                torch.save(self.state_dict(), best_model_path)
                
                
                print('epoch: {}, find new best loss: Val Loss: {:.4f}'.format(epoch, best_loss))
                print(f'最佳模型已保存到: {best_model_path}')
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
        pd.DataFrame(self.loss_history).to_excel(loss_csv_path, index=False)
