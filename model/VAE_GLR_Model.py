import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import pandas as pd
import os
from collections import OrderedDict
import time
if hasattr(torch, "func"):
    functional_call = torch.func.functional_call
    torch_func_grad = torch.func.grad
    vmap = torch.func.vmap
else:
    from torch.nn.utils.stateless import functional_call
    from functorch import grad as torch_func_grad, vmap

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
        self.silu = nn.SiLU()
    def forward(self, x,con_x):
        x = torch.cat((x, con_x), dim=1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x1 = self.linear3(x)
        x2 = self.linear4(x)
        return x1,x2

class Decoder(nn.Module):
    def __init__(self, output_dim, con_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim+con_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x,con_x):
        x = torch.cat((x, con_x), dim=1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    

class VAE_GLR_Model(nn.Module):
    def __init__(self, targetdim, labeldim, latent,data_len,epoch,  quantiles=0.5,lambda_gradient=0.5, 
                 samplingnumber=10, target_quantile=0.95,
                 cost_under=10.0, cost_over=5.0,random_seed = 0,innerloop=1,singleepoch = 10, save_xlsx = None): # 新增成本参数
        super(VAE_GLR_Model, self).__init__()
        self.fc1 = nn.Linear(targetdim + labeldim, 32)
        self.fc11 = nn.Linear(32, 64)
        self.fc21 = nn.Linear(64, latent)  # mean
        self.fc22 = nn.Linear(64, latent)  # var
        self.fc3 = nn.Linear(latent + labeldim, 32)
        self.fc31 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, targetdim)
        self.encoder = Encoder(targetdim, labeldim, 64, latent)
        self.decoder = Decoder(targetdim, labeldim, 64, latent)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=5e-4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4) # redundant, already defined above
        self.lambda_gradient = lambda_gradient
        self.epoch = epoch
        self.samplingnumber = samplingnumber
        self.save_loss = "glr_pth"
        self.save_xlsx = save_xlsx if save_xlsx is not None else "glr_xlsx"
        # 损失记录
        self.loss_history = {
            'train_loss': [],
            'val_loss': [],
            'vae_loss': [],
            'val_quantile_loss': [],
            'quantile_loss': [],
            'total_loss': [],
            'quantile_gradient': []
        }
        self.targetdim = targetdim
        self.labeldim = labeldim
        self.latent = latent
        self.target_quantile = target_quantile
        self.samplingnumber = samplingnumber
        # Newsvendor 成本系数
        self.cu = cost_under
        self.co = cost_over
        self.quantiles = quantiles
        # 不使用 register_buffer，直接作为普通属性
        self.data_len  = data_len
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        self.k_step = 1.0
        self.D_hat = []
        self.random_seed = random_seed
        self.D_hat_avg = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_D = [
    torch.zeros_like(param, requires_grad=False).to(self.device) 
    for param in self.decoder.parameters()
]
        self.q_hat = []
        self.q_hat_list = []
        self.innerloop = innerloop
        self.singleepoch = singleepoch
        for i in range(self.data_len):
            self.q_hat.append([torch.tensor(0.0).to(torch.float32)])
            self.q_hat_list.append([torch.tensor(0.0).to(torch.float32)])
        
        # 只针对 decoder 的参数创建 D_hat 和 D_hat_avg
        for i in range(self.data_len):
            self.D_hat.append([])
            for p in self.decoder.parameters():
                self.D_hat[i].append(torch.zeros_like(p))

    def _sync_auxiliary_state_device(self):
        """将 D_hat / global_D / q_hat 等列表状态迁移到当前参数所在设备，
        解决 model.to(cuda) 后这些 CPU tensor 与 GPU 梯度混用的报错。"""
        device = next(self.parameters()).device
        self.device = device
        self.global_D = [
            t.to(device) if isinstance(t, torch.Tensor) else t
            for t in self.global_D
        ]
        self.D_hat = [
            [t.to(device) if isinstance(t, torch.Tensor) else t for t in row]
            for row in self.D_hat
        ]
        self.q_hat = [
            [v.to(device) if isinstance(v, torch.Tensor) else v for v in row]
            for row in self.q_hat
        ]
        self.q_hat_list = [
            [v.to(device) if isinstance(v, torch.Tensor) else v for v in row]
            for row in self.q_hat_list
        ]

    def _build_q_tensor_for_batch(self, global_indices, q_local, device, dtype):
        q_values = torch.zeros(len(global_indices), 1, device=device, dtype=dtype)
        for idx, global_idx in enumerate(global_indices):
            if global_idx < len(self.q_hat) and self.q_hat[global_idx][0] != 0.0:
                q_values[idx, 0] = float(self.q_hat[global_idx][0])
            else:
                q_values[idx, 0] = q_local[idx, 0].to(device=device, dtype=dtype)
        return q_values

    def _vectorized_globalsingle_innerloop(self, x_label, y_true, q_values, latent_dim):
        batch_size = x_label.shape[0]
        device = x_label.device
        decoder_params = OrderedDict(self.decoder.named_parameters())
        neg_weight = y_true.new_tensor(-self.cu / (self.cu + self.co))
        pos_weight = y_true.new_tensor(self.co / (self.cu + self.co))
        gradient_start_time = time.time()

        def single_decoder_output(params, z_i, x_i):
            return functional_call(
                self.decoder,
                params,
                (z_i.unsqueeze(0), x_i.unsqueeze(0)),
            ).reshape(())

        def single_sample_surrogate(params, z_i, x_i, y_i, q_i, dim_mask):

            def output_fn(latent):
                return single_decoder_output(params, latent, x_i)

            y_pred_i = output_fn(z_i)
            grads_z_i = torch_func_grad(output_fn)(z_i)
            h_prime_i = torch.sum(grads_z_i * dim_mask)

            def h_prime_fn(latent):
                return torch.sum(torch_func_grad(output_fn)(latent) * dim_mask)

            h_double_prime_vec_i = torch_func_grad(h_prime_fn)(z_i)
            h_double_prime_i = torch.sum(h_double_prime_vec_i * dim_mask)
            score_i = -torch.sum(z_i * dim_mask)

            epsilon = y_pred_i.new_tensor(1e-6)
            h_prime_inv_i = 1.0 / (h_prime_i + epsilon * torch.sign(h_prime_i))
            psi_2_i = torch.clamp(
                h_prime_inv_i * (score_i - h_double_prime_i * h_prime_inv_i),
                -100.0,
                100.0,
            )
            h_prime_inv_i = torch.clamp(h_prime_inv_i, -100.0, 100.0)

            indicator_i = (y_pred_i <= q_i).to(y_pred_i.dtype)
            diff_i = q_i - y_i
            nv_w_i = torch.where(diff_i < 0, neg_weight, pos_weight)
            final_w_i = indicator_i * nv_w_i
            surrogate_loss_i = (y_pred_i * psi_2_i + h_prime_i * h_prime_inv_i) * final_w_i
            g2_i = torch.clamp(torch.abs(psi_2_i * indicator_i), min=1e-4)
            return surrogate_loss_i, (y_pred_i, g2_i)

        grad_fn = torch_func_grad(single_sample_surrogate, argnums=0, has_aux=True)
        chunk_size = int(os.environ.get("GLR_GLOBALSINGLE_CHUNK_SIZE", "8"))
        chunk_size = max(1, min(chunk_size, batch_size))
        grad_chunks = {param_name: [] for param_name in decoder_params.keys()}
        y_pred_chunks = []
        g2_chunks = []

        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            current_chunk_size = end_idx - start_idx
            z_chunk = torch.randn(current_chunk_size, latent_dim, device=device)
            dim_chunk = torch.randint(0, latent_dim, (current_chunk_size,), device=device)
            dim_mask_chunk = torch.zeros(current_chunk_size, latent_dim, device=device, dtype=z_chunk.dtype)
            dim_mask_chunk.scatter_(1, dim_chunk.unsqueeze(1), 1.0)

            per_sample_grads_chunk, (y_pred_chunk, g2_chunk) = vmap(
                grad_fn,
                in_dims=(None, 0, 0, 0, 0, 0),
            )(
                decoder_params,
                z_chunk,
                x_label[start_idx:end_idx],
                y_true[start_idx:end_idx].squeeze(1),
                q_values[start_idx:end_idx].squeeze(1),
                dim_mask_chunk,
            )

            for param_name in decoder_params.keys():
                grad_chunks[param_name].append(per_sample_grads_chunk[param_name].detach())
            y_pred_chunks.append(y_pred_chunk.detach())
            g2_chunks.append(g2_chunk.detach())
            del z_chunk, dim_chunk, dim_mask_chunk, per_sample_grads_chunk, y_pred_chunk, g2_chunk

        per_sample_grads = OrderedDict(
            (param_name, torch.cat(chunks, dim=0))
            for param_name, chunks in grad_chunks.items()
        )
        y_pred_batch = torch.cat(y_pred_chunks, dim=0)
        g2_batch = torch.cat(g2_chunks, dim=0)
        gradient_compute_time = time.time() - gradient_start_time
        return per_sample_grads, y_pred_batch.unsqueeze(1).detach(), g2_batch.detach().unsqueeze(1), gradient_compute_time

    def _sequential_globalsingle_innerloop(self, x_label, y_true, q_values, latent_dim):
        batch_size = x_label.shape[0]
        device = x_label.device
        decoder_named_params = OrderedDict(self.decoder.named_parameters())
        decoder_params = list(decoder_named_params.values())
        neg_weight = y_true.new_tensor(-self.cu / (self.cu + self.co))
        pos_weight = y_true.new_tensor(self.co / (self.cu + self.co))
        epsilon = y_true.new_tensor(1e-6)
        gradient_start_time = time.time()

        grad_lists = OrderedDict((param_name, []) for param_name in decoder_named_params.keys())
        y_pred_list = []
        g2_list = []

        for sample_idx in range(batch_size):
            x_i = x_label[sample_idx:sample_idx + 1]
            y_i = y_true[sample_idx:sample_idx + 1]
            q_i = q_values[sample_idx:sample_idx + 1]
            z_i = torch.randn(1, latent_dim, device=device, dtype=x_label.dtype, requires_grad=True)
            dim_i = torch.randint(0, latent_dim, (1,), device=device).item()

            y_pred_i = self.decoder(z_i, x_i)
            grads_z_i = torch.autograd.grad(
                y_pred_i,
                z_i,
                grad_outputs=torch.ones_like(y_pred_i),
                create_graph=True,
                retain_graph=True,
            )[0]
            h_prime_i = grads_z_i[:, dim_i].view(1, 1)
            grad_h_prime_i = torch.autograd.grad(
                h_prime_i,
                z_i,
                grad_outputs=torch.ones_like(h_prime_i),
                create_graph=True,
                retain_graph=True,
            )[0]
            h_double_prime_i = grad_h_prime_i[:, dim_i].view(1, 1)
            score_i = -z_i[:, dim_i].view(1, 1)

            safe_h_prime_i = torch.where(
                torch.abs(h_prime_i) < epsilon,
                torch.where(h_prime_i >= 0, epsilon, -epsilon),
                h_prime_i,
            )
            h_prime_inv_i = torch.clamp(1.0 / safe_h_prime_i, -100.0, 100.0)
            psi_core_i = score_i - h_double_prime_i * h_prime_inv_i
            psi_2_i = torch.clamp(h_prime_inv_i * psi_core_i, -100.0, 100.0)

            indicator_i = (y_pred_i.detach() <= q_i).to(y_pred_i.dtype)
            diff_i = q_i - y_i
            nv_w_i = torch.where(diff_i < 0, neg_weight, pos_weight)
            final_w_i = (indicator_i * nv_w_i).detach()

            grad_theta_i = torch.autograd.grad(
                y_pred_i,
                decoder_params,
                grad_outputs=torch.ones_like(y_pred_i),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            grad_h_theta_i = torch.autograd.grad(
                h_prime_i,
                decoder_params,
                grad_outputs=torch.ones_like(h_prime_i),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            scale_i = (h_prime_inv_i.detach() * final_w_i).reshape(())
            psi_core_scalar_i = psi_core_i.detach().reshape(())
            for param_name, param, grad_theta_val, grad_h_theta_val in zip(
                decoder_named_params.keys(),
                decoder_params,
                grad_theta_i,
                grad_h_theta_i,
            ):
                if grad_theta_val is None:
                    grad_theta_val = torch.zeros_like(param)
                if grad_h_theta_val is None:
                    grad_h_theta_val = torch.zeros_like(param)
                g1_i = scale_i * (
                    grad_theta_val.detach() * psi_core_scalar_i
                    + grad_h_theta_val.detach()
                )
                grad_lists[param_name].append(g1_i.detach())

            g2_i = torch.clamp(torch.abs((psi_2_i * indicator_i).detach()), min=1e-4, max=10.0)
            y_pred_list.append(y_pred_i.detach().reshape(1))
            g2_list.append(g2_i.reshape(1))

        per_sample_grads = OrderedDict(
            (param_name, torch.stack(grads, dim=0))
            for param_name, grads in grad_lists.items()
        )
        y_pred_batch = torch.stack(y_pred_list, dim=0).view(batch_size, 1)
        g2_batch = torch.stack(g2_list, dim=0).view(batch_size, 1)
        gradient_compute_time = time.time() - gradient_start_time
        return per_sample_grads, y_pred_batch, g2_batch, gradient_compute_time

    
    
    def get_save_path(self, save_tag=None):
        """
        获取模型保存路径。
        save_tag 用于区分不同训练配置对应的模型文件。
        """
        file_name = f"glr_loss_{self.labeldim}_{self.epoch}_{self.innerloop}_{self.random_seed}"
        if save_tag:
            file_name = f"{file_name}_{save_tag}"
        return os.path.join(self.save_loss, f"{file_name}.pth")
    
    def get_save_xlsx_path(self, save_tag=None):
        """
        获取模型保存的 Excel 路径。
        save_tag 用于区分不同训练配置对应的记录文件。
        """
        file_name = f"glr_loss_{self.labeldim}_{self.epoch}_{self.innerloop}_{self.random_seed}"
        if save_tag:
            file_name = f"{file_name}_{save_tag}"
        return os.path.join(self.save_xlsx, f"{file_name}.xlsx")
    
    
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
    
    
        
    def encode(self, x, condition):  # 编码层
        return self.encoder(x,condition)
    
    
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
    
    
    
    
    
    
    def calculate_newsvendor_loss(self, y, q):
        """
        计算 Newsvendor Cost: 
        Cost = Cu * (y - q)^+ + Co * (q - y)^+
        注意：这里 y 是需求(模型输出), q 是决策(分位数)
        通常 Newsvendor 是 q 决策量。
        如果 y > q (需求 > 库存): 缺货 (Underage), 损失 = Cu * (y - q)
        如果 y < q (需求 < 库存): 积压 (Overage), 损失 = Co * (q - y)
        """
        diff = y - q
        loss = torch.where(diff > 0, self.cu * diff, self.co * (-diff))
        return loss
    
    
    
    
    
    
    
    def train_step_sqo_vectorized_SGD(self, data_loader,valdata_loader, early_stopping,  batch_size,ifdecoderonly = False, save_tag=None):
        """
        SGD 版本：使用 DataLoader 进行随机梯度下降
        data_loader: PyTorch DataLoader，每个batch返回 (data, indices)
                    其中 indices 是数据在原始数据集中的全局索引
        batch_size: batch 大小
        """
        
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=1e-4)
        latent_dim = self.latent
        n_samples = self.samplingnumber
        vae_loss_list = []
        val_vae_loss_list = []
        val_quantile_loss_list = []
        quantile_loss_list = []
        optimizer = self.optimizer
        best_loss = float('inf')
        early_stopping_counter = 0
        stop_training_due_to_nan = False
        save_pth = self.get_save_path(save_tag)
        if not os.path.exists(self.save_loss):
            os.makedirs(self.save_loss)
        for epoch in range(self.epoch):
            sum_dk = 0.0
            sum_vae_loss  = 0
            sum_q_loss = 0            
            for batch_idx, batch_data in enumerate(data_loader):
                # 如果 DataLoader 返回 (data, indices)
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, global_indices = batch_data
                    global_indices = global_indices.cpu().numpy()
                else:
                    # 如果没有索引，根据 batch_idx 计算全局索引
                    data = batch_data
                    global_indices = np.arange(batch_idx * batch_size, 
                                               min((batch_idx + 1) * batch_size, self.data_len))
                
                current_batch_size = data.shape[0]
                device = self.device
                im = data[:, -1].reshape(-1, 1).to(device)
                im_label = data[:, :-1].to(device)
                
                
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                if not ifdecoderonly:
                    if not torch.isfinite(vae_loss):
                        print(f"Non-finite VAE loss detected at epoch {epoch+1}, batch {batch_idx}: {vae_loss.item()}")
                        stop_training_due_to_nan = True
                        break

                    vae_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    vae_optimizer.step()
                    vae_loss_val = vae_loss.item()
                else:
                    vae_loss_val = vae_loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx}, VAE Loss: {vae_loss_val:.4f}")
                sum_vae_loss += vae_loss_val
                sum_q_loss += self.quantile_loss(recon_im, im).item()
                # 清理VAE阶段的显存
                del recon_im, mu, logvar, vae_loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                for loop in range(self.innerloop):
                    x_label = data[:, :-1].to(device)
                    y_true = data[:, -1].unsqueeze(1).to(device)
                    # 采样单个 z（每个数据点一个预测）
                    z = torch.randn(current_batch_size, latent_dim).to(device)
                    z.requires_grad_(True)
                    
                    # 随机选择维度
                    dim_i = np.random.randint(0, latent_dim)

                    y_pred = self.decoder(z, x_label)
                    with torch.no_grad():
                        z_weight = torch.randn_like(z).to(device)  # 随机权重，形状与 z 相同
                        y_pred_weight = self.decoder(z_weight, x_label)  # 使用随机权重计算预测值
                    # 初始化 q_hat（仅第一次需要采样获取初值）
                    if self.q_hat[global_indices[0]][0] == 0.0:
                        with torch.no_grad():
                            x_expanded_init = data[:, :-1].unsqueeze(1).repeat(1, self.samplingnumber, 1).view(-1, self.labeldim).to(device)
                            z_init = torch.randn(current_batch_size * self.samplingnumber, latent_dim).to(device)
                            y_init = self.decoder(z_init, x_expanded_init)
                            y_reshaped_init = y_init.view(current_batch_size, n_samples)
                            q_local = torch.quantile(y_reshaped_init, self.target_quantile, dim=1, keepdim=True)
                            del x_expanded_init, z_init, y_init, y_reshaped_init

                    with torch.no_grad():
                        # 构建 q_hat tensor
                        q_for_indicator = torch.zeros(current_batch_size, 1, device=device)
                        for i, global_idx in enumerate(global_indices):
                            if global_idx < len(self.q_hat) and self.q_hat[global_idx][0] != 0.0:
                                q_for_indicator[i, 0] = self.q_hat[global_idx][0]
                            else:
                                q_for_indicator[i, 0] = q_local[i, 0]
                        
                        # 直接用预测值和 quantile value 比较计算 indicator
                        indicator = (y_pred <= q_for_indicator).float()
                        
                        # 计算 Newsvendor 权重
                        diff = q_for_indicator- y_true
                        nv_weights = torch.where(diff < 0, 
                                                torch.tensor(-self.cu/(self.cu+self.co)).to(device), 
                                                torch.tensor(self.co/(self.co+self.cu)).to(device))
                        final_weights = indicator * nv_weights
                   
                    # 计算梯度
                    grad_outputs = torch.ones_like(y_pred)
                    grads_z = torch.autograd.grad(y_pred, z, grad_outputs=grad_outputs, 
                                                create_graph=True, retain_graph=True)[0]
                    h_prime = grads_z[:, dim_i].view(-1, 1)

                    grad_h_prime = torch.autograd.grad(h_prime, z, grad_outputs=torch.ones_like(h_prime),
                                                    create_graph=True, retain_graph=True)[0]
                    h_double_prime = grad_h_prime[:, dim_i].view(-1, 1)
                    score = -z[:, dim_i].view(-1, 1)

                    # 计算 Psi
                    epsilon = 1e-6
                    h_prime_inv = 1.0 / (h_prime + epsilon * torch.sign(h_prime))
                    psi_2 = h_prime_inv * (score - h_double_prime * h_prime_inv)
                    psi_2 = torch.clamp(psi_2, -100, 100).detach()
                    h_prime_inv = torch.clamp(h_prime_inv, -100, 100).detach()

                    # 计算 Surrogate Loss（每个点一个预测值，无需 reshape）
                    surrogate_loss = (y_pred * psi_2 + h_prime * h_prime_inv) * final_weights
                    global_loss = surrogate_loss.mean()

                    if not torch.isfinite(global_loss):
                        print(f"Non-finite GLR loss detected at epoch {epoch+1}, batch {batch_idx}: {global_loss.item()}")
                        stop_training_due_to_nan = True
                        break

                    decoder_optimizer.zero_grad()
                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch+1}, Batch {batch_idx}, GLR Loss: {global_loss.item():.4f}")
                    global_loss.backward()
                    
                    # 提取 G1 梯度
                    g1_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                            for p in self.decoder.parameters()]
                    decoder_optimizer.zero_grad()

                    # 计算 G2
                    with torch.no_grad():
                        g2_vals = psi_2*indicator 
                        g2_per_point = g2_vals.view(current_batch_size)
                        g2_per_point = torch.clamp(g2_per_point, min=1e-4, max=10.0)
                        global_g2 = g2_per_point.mean()

                    # 更新参数
                    k = self.k_step
                    gamma_k = 1 / (k ** 0.55)
                    beta_k = 1 / ((k ) ** 0.6)  # q_hat的更新步长
                
                    # 使用全局索引保存 Q_hat
                    with torch.no_grad():
                        for i, global_idx in enumerate(global_indices):
                            if global_idx >= len(self.q_hat):
                                continue
                            
                            # 第一次迭代：使用采样分位数 q_local 作为初始值
                            if self.q_hat[global_idx][0] == 0.0:
                                self.q_hat[global_idx] = [q_local[i].item()]
                                self.q_hat_list[global_idx].append(q_local[i].item())
                            else:
                                # 直接用预测值与 q_hat 比较计算 indicator
                                q_hat_current = self.q_hat[global_idx][0]
                                indicator_val = float(y_pred[i, 0].item() <= q_hat_current)
                                
                                # q_hat 更新公式: q_hat_{k+1} = q_hat_k + beta_k * (target_quantile - indicator)
                                q_hat_new = q_hat_current + beta_k * (self.target_quantile - indicator_val)
                                
                                # 更新q_hat
                                self.q_hat[global_idx] = [q_hat_new]
                                self.q_hat_list[global_idx].append(q_hat_new)

                    # 使用全局索引更新 D_hat
                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.D_hat):
                            continue
                        
                        g2_i = g2_per_point[i].item()
                        
                        new_D_i = []
                        for d_val, g1_val in zip(self.D_hat[global_idx], g1_grads):
                            d_val_device = d_val.to(g1_val.device) if d_val.device != g1_val.device else d_val
                            update = g1_val - g2_i * d_val_device
                            d_new = d_val_device + gamma_k * update
                            d_new = torch.clamp(d_new, -1.0, 1.0).detach()
                            new_D_i.append(d_new)
                        self.D_hat[global_idx] = new_D_i

                # 计算当前 batch 的平均 D
                avg_D = [torch.zeros_like(p) for p in self.decoder.parameters()]
                valid_count = 0
                for i, global_idx in enumerate(global_indices):
                    if global_idx < len(self.D_hat):
                        for j, d_val in enumerate(self.D_hat[global_idx]):
                            avg_D[j] = avg_D[j] + d_val.to(avg_D[j].device)
                        valid_count += 1
                
                if valid_count > 0:
                    for j in range(len(avg_D)):
                        avg_D[j] = avg_D[j] / valid_count
                avg_D = [d_val.detach() for d_val in avg_D]
                
                # 计算每个数据点的 D_hat 范数的平均值（每个batch一个标量）
                D_norms_per_point = []
                for i, global_idx in enumerate(global_indices):
                    if global_idx < len(self.D_hat):
                        # 计算该数据点所有参数的范数的平均值
                        point_d_norms = [torch.norm(d_val).item() for d_val in self.D_hat[global_idx]]
                        point_d_norm_mean = np.mean(point_d_norms) if point_d_norms else 0.0
                        D_norms_per_point.append(point_d_norm_mean)
                
                # 对batch中所有数据点的D范数求平均，得到该batch的一个标量
                batch_D_norm_mean = np.mean(D_norms_per_point) if D_norms_per_point else 0.0
                sum_dk += batch_D_norm_mean
                
                for param, d_val in zip(self.decoder.parameters(), avg_D):
                    if param.grad is None:
                        param.grad = d_val.clone()*self.lambda_gradient
                    else:
                        param.grad.copy_(d_val)

                
                
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                decoder_optimizer.step()
                self.k_step += 1
                
                
                try:
                    del z, y_pred, surrogate_loss
                    del global_loss, g1_grads, g2_vals, g2_per_point, global_g2
                    del h_prime, h_double_prime, score, psi_2, h_prime_inv, final_weights
                    del q_for_indicator, indicator
                    del diff, nv_weights, grad_outputs, grads_z, grad_h_prime, x_label, y_true
                    del avg_D, im, im_label
                except:
                    pass
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            if stop_training_due_to_nan:
                print("Stopping training because non-finite loss was detected.")
                break
                    
                
            sum_dk /= len(data_loader)
            sum_vae_loss /= len(data_loader)
            sum_q_loss /= len(data_loader)
            self.loss_history['vae_loss'].append(sum_vae_loss)
            self.loss_history['quantile_loss'].append(sum_q_loss)
            vae_loss_list.append(sum_vae_loss)
            quantile_loss_list.append(sum_q_loss)
            self.D_hat_avg.append(sum_dk)
            val_loss = 0.0
            quantile_val = 0.0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    batch = val_batch.to(device)
                    im = batch[:, -1].reshape(-1, 1).to(device)
                    im_label = batch[:, :-1].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    quantile_val += self.quantile_loss(val_recon_im, im).item()
                    val_loss += val_vae_loss.item()
            val_loss /= len(valdata_loader)
            quantile_val /= len(valdata_loader)
            val_vae_loss_list.append(val_loss)
            val_quantile_loss_list.append(quantile_val)
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_quantile_loss'].append(quantile_val)
            loss_new = val_loss
            if loss_new < best_loss and not np.isnan(vae_loss_list[-1]):
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                #save model
                torch.save(self.state_dict(), save_pth)
                
                # 保存最佳模型（简化版本）
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
            if np.isnan(vae_loss_list[-1]):
                print("NaN detected in VAE loss, stopping training.")
                break
            # if np.abs(global_loss.item()) < 1e-3:
            #     print(f"Abnormal G2 value detected: {global_g2.item()}, stopping training.")
            #     break
            
            
        return {
            "vae_loss": vae_loss_list,
            "quantile_loss": quantile_loss_list,
            "val_vae_loss": val_vae_loss_list,
            "val_quantile_loss": val_quantile_loss_list,
            "message": f"SGD training completed: {self.epoch} epochs, {len(data_loader)} batches per epoch"
        }




    def train_step_sqo_vectorized_SGD_single(self, data_loader,valdata_loader, early_stopping,  batch_size,ifdecoderonly = False, save_tag=None):
        """
        SGD 版本：使用 DataLoader 进行随机梯度下降
        data_loader: PyTorch DataLoader，每个batch返回 (data, indices)
                    其中 indices 是数据在原始数据集中的全局索引
        batch_size: batch 大小
        """
        self._sync_auxiliary_state_device()
        print(f"Start train_step_sqo_vectorized_SGD_single: innerloop={self.innerloop}, device={self.device}, epochs={self.epoch}")
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=1e-4)
        latent_dim = self.latent
        n_samples = self.samplingnumber
        vae_loss_list = []
        val_vae_loss_list = []
        val_quantile_loss_list = []
        quantile_loss_list = []
        best_loss = float('inf')
        early_stopping_counter = 0
        stop_training_due_to_nan = False
        save_pth = self.get_save_path(save_tag)
        if not os.path.exists(self.save_loss):
            os.makedirs(self.save_loss)
        for epoch in range(self.epoch):
            sum_dk = 0.0
            sum_vae_loss  = 0
            sum_q_loss = 0            
            for batch_idx, batch_data in enumerate(data_loader):
                # 如果 DataLoader 返回 (data, indices)
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, global_indices = batch_data
                    global_indices = global_indices.cpu().numpy()
                else:
                    # 如果没有索引，根据 batch_idx 计算全局索引
                    data = batch_data
                    global_indices = np.arange(batch_idx * batch_size, 
                                               min((batch_idx + 1) * batch_size, self.data_len))
                
                current_batch_size = data.shape[0]
                device = self.device
                im = data[:, -1].reshape(-1, 1).to(device)
                im_label = data[:, :-1].to(device)
                
                
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                if not ifdecoderonly:
                    if not torch.isfinite(vae_loss):
                        print(f"Non-finite VAE loss detected at epoch {epoch+1}, batch {batch_idx}: {vae_loss.item()}")
                        stop_training_due_to_nan = True
                        break

                    vae_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    vae_optimizer.step()
                    vae_loss_val = vae_loss.item()
                else:
                    vae_loss_val = vae_loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx}, VAE Loss: {vae_loss_val:.4f}")
                sum_vae_loss += vae_loss_val
                sum_q_loss += self.quantile_loss(recon_im, im).item()
                # 清理VAE阶段的显存
                del recon_im, mu, logvar, vae_loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            
                x_label = data[:, :-1].to(device)
                y_true = data[:, -1].unsqueeze(1).to(device)
                with torch.no_grad():
                    x_expanded_init = data[:, :-1].unsqueeze(1).repeat(1, self.samplingnumber, 1).view(-1, self.labeldim).to(device)
                    z_init = torch.randn(current_batch_size * self.samplingnumber, latent_dim).to(device)
                    y_init = self.decoder(z_init, x_expanded_init)
                    y_reshaped_init = y_init.view(current_batch_size, n_samples)
                    q_local = torch.quantile(y_reshaped_init, self.target_quantile, dim=1, keepdim=True)
                    del x_expanded_init, z_init, y_init, y_reshaped_init

                for _ in range(self.innerloop):
                    for i in range(current_batch_size):
                        global_idx = global_indices[i]
                        if global_idx >= len(self.D_hat):
                            continue

                        x_i = x_label[i:i+1]
                        y_i = y_true[i:i+1]

                        z_i = torch.randn(1, latent_dim, device=device, requires_grad=True)
                        dim_i = np.random.randint(0, latent_dim)

                        y_pred_i = self.decoder(z_i, x_i)

                        with torch.no_grad():
                            if global_idx < len(self.q_hat) and self.q_hat[global_idx][0] != 0.0:
                                q_hat_val = self.q_hat[global_idx][0]
                            else:
                                q_hat_val = q_local[i, 0].item()
                            q_i = torch.tensor([[q_hat_val]], device=device)

                            indicator_i = (y_pred_i <= q_i).float()
                            diff_i = q_i - y_i
                            nv_w_i = torch.where(diff_i < 0,
                                                    torch.tensor(-self.cu/(self.cu+self.co), device=device),
                                                    torch.tensor(self.co/(self.co+self.cu), device=device))
                            final_w_i = indicator_i * nv_w_i

                        grad_outputs_i = torch.ones_like(y_pred_i)
                        grads_z_i = torch.autograd.grad(y_pred_i, z_i, grad_outputs=grad_outputs_i,
                                                            create_graph=True, retain_graph=True)[0]
                        h_prime_i = grads_z_i[:, dim_i].view(-1, 1)

                        grad_h_prime_i = torch.autograd.grad(h_prime_i, z_i, grad_outputs=torch.ones_like(h_prime_i),
                                                                create_graph=True, retain_graph=True)[0]
                        h_double_prime_i = grad_h_prime_i[:, dim_i].view(-1, 1)
                        score_i = -z_i[:, dim_i].view(-1, 1)

                        epsilon = 1e-6
                        h_prime_inv_i = 1.0 / (h_prime_i + epsilon * torch.sign(h_prime_i))
                        psi_2_i = h_prime_inv_i * (score_i - h_double_prime_i * h_prime_inv_i)
                        psi_2_i = torch.clamp(psi_2_i, -100, 100).detach()
                        h_prime_inv_i = torch.clamp(h_prime_inv_i, -100, 100).detach()

                        # g1*g3: surrogate_loss 反向传播得到 decoder 梯度
                        surrogate_loss_i = (y_pred_i * psi_2_i + h_prime_i * h_prime_inv_i) * final_w_i

                        decoder_optimizer.zero_grad()
                        surrogate_loss_i.backward()
                        bar_G1_i = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                                    for p in self.decoder.parameters()]
                        decoder_optimizer.zero_grad()

                        # g2: 密度标量
                        with torch.no_grad():
                            g2_i = (psi_2_i * indicator_i).item()
                            g2_i = max(abs(g2_i), 1e-4)

                        # 更新 per-point D_k (每个数据点维护自己的 D_hat)
                        alpha_k = 1 / (self.k_step ** 0.55)
                        for d_val, g1_val in zip(self.D_hat[global_idx], bar_G1_i):
                            update = g1_val / g2_i - d_val
                            d_val.add_(alpha_k * update)
                            d_val.clamp_(-1.0, 1.0)

                        # 更新 q_hat
                        beta_k = 1 / ((self.k_step) ** 0.6)
                        with torch.no_grad():
                            if self.q_hat[global_idx][0] == 0.0:
                                self.q_hat[global_idx] = [q_local[i].item()]
                                self.q_hat_list[global_idx].append(q_local[i].item())
                            else:
                                q_hat_current = self.q_hat[global_idx][0]
                                indicator_val = float(y_pred_i[0, 0].item() <= q_hat_current)
                                q_hat_new = q_hat_current + beta_k * (self.target_quantile - indicator_val)
                                self.q_hat[global_idx] = [q_hat_new]
                                self.q_hat_list[global_idx].append(q_hat_new)

                        del z_i, y_pred_i, surrogate_loss_i
                        del h_prime_i, h_double_prime_i, score_i, psi_2_i, h_prime_inv_i
                
                # --- 逐点循环结束后，累加 batch 内各点的 D_hat，统一更新参数 ---
                avg_D = [torch.zeros_like(p) for p in self.decoder.parameters()]
                valid_count = 0
                for i in range(current_batch_size):
                    global_idx = global_indices[i]
                    if global_idx < len(self.D_hat):
                        for j in range(len(avg_D)):
                            avg_D[j] += self.D_hat[global_idx][j]
                        valid_count += 1
                if valid_count > 0:
                    for j in range(len(avg_D)):
                        avg_D[j] /= valid_count
                decoder_optimizer.zero_grad()
                for param, d_val in zip(self.decoder.parameters(), avg_D):
                    if param.grad is None:
                        param.grad = d_val.clone() * self.lambda_gradient
                    else:
                        param.grad += d_val.clone() * self.lambda_gradient
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                decoder_optimizer.step()
                self.k_step+=1
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            if stop_training_due_to_nan:
                print("Stopping training because non-finite loss was detected.")
                break
            
            sum_dk /= len(data_loader)
            sum_vae_loss /= len(data_loader)
            sum_q_loss /= len(data_loader)
            self.loss_history['vae_loss'].append(sum_vae_loss)
            self.loss_history['quantile_loss'].append(sum_q_loss)
            vae_loss_list.append(sum_vae_loss)
            quantile_loss_list.append(sum_q_loss)
            self.D_hat_avg.append(sum_dk)
            val_loss = 0.0
            quantile_val = 0.0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    batch = val_batch.to(device)
                    im = batch[:, -1].reshape(-1, 1).to(device)
                    im_label = batch[:, :-1].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    quantile_val += self.quantile_loss(val_recon_im, im).item()
                    val_loss += val_vae_loss.item()
            val_loss /= len(valdata_loader)
            quantile_val /= len(valdata_loader)
            val_vae_loss_list.append(val_loss)
            val_quantile_loss_list.append(quantile_val)
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_quantile_loss'].append(quantile_val)
            loss_new = val_loss
            if loss_new < best_loss and not np.isnan(vae_loss_list[-1]):
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                #save model
                torch.save(self.state_dict(), save_pth)
                
                # 保存最佳模型（简化版本）
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
            if np.isnan(vae_loss_list[-1]):
                print("NaN detected in VAE loss, stopping training.")
                break
            # if np.abs(global_loss.item()) < 1e-3:
            #     print(f"Abnormal G2 value detected: {global_g2.item()}, stopping training.")
            #     break
            
            
        return {
            "vae_loss": vae_loss_list,
            "quantile_loss": quantile_loss_list,
            "val_vae_loss": val_vae_loss_list,
            "val_quantile_loss": val_quantile_loss_list,
            "message": f"SGD training completed: {self.epoch} epochs, {len(data_loader)} batches per epoch"
        }


    
    
    
    
    def train_step_sqo_vectorized_SGD_global(self, data_loader,valdata_loader, early_stopping,  batch_size,ifdecoderonly = False, save_tag=None):
        """
        SGD 版本：使用 DataLoader 进行随机梯度下降
        data_loader: PyTorch DataLoader，每个batch返回 (data, indices)
                    其中 indices 是数据在原始数据集中的全局索引
        batch_size: batch 大小
        """
        
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=1e-4)
        latent_dim = self.latent
        n_samples = self.samplingnumber
        vae_loss_list = []
        val_vae_loss_list = []
        val_quantile_loss_list = []
        quantile_loss_list = []
        optimizer = self.optimizer
        best_loss = float('inf')
        early_stopping_counter = 0
        stop_training_due_to_nan = False
        save_pth = self.get_save_path(save_tag)
        if not os.path.exists(self.save_loss):
            os.makedirs(self.save_loss)
        for epoch in range(self.epoch):
            sum_dk = 0.0
            sum_vae_loss  = 0
            sum_q_loss = 0            
            for batch_idx, batch_data in enumerate(data_loader):
                # 如果 DataLoader 返回 (data, indices)
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, global_indices = batch_data
                    global_indices = global_indices.cpu().numpy()
                else:
                    # 如果没有索引，根据 batch_idx 计算全局索引
                    data = batch_data
                    global_indices = np.arange(batch_idx * batch_size, 
                                               min((batch_idx + 1) * batch_size, self.data_len))
                
                current_batch_size = data.shape[0]
                device = self.device
                im = data[:, -1].reshape(-1, 1).to(device)
                im_label = data[:, :-1].to(device)
                
                
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                if not ifdecoderonly:
                    if not torch.isfinite(vae_loss):
                        print(f"Non-finite VAE loss detected at epoch {epoch+1}, batch {batch_idx}: {vae_loss.item()}")
                        stop_training_due_to_nan = True
                        break

                    vae_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    vae_optimizer.step()
                    vae_loss_val = vae_loss.item()
                else:
                    vae_loss_val = vae_loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx}, VAE Loss: {vae_loss_val:.4f}")
                sum_vae_loss += vae_loss_val
                sum_q_loss += self.quantile_loss(recon_im, im).item()
                # 清理VAE阶段的显存
                del recon_im, mu, logvar, vae_loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                x_label = data[:, :-1].to(device)
                y_true = data[:, -1].unsqueeze(1).to(device)
                # 采样单个 z（每个数据点一个预测）
                z = torch.randn(current_batch_size, latent_dim).to(device)
                z.requires_grad_(True)
                
                # 随机选择维度
                dim_i = np.random.randint(0, latent_dim)

                y_pred = self.decoder(z, x_label)
                with torch.no_grad():
                    z_weight = torch.randn_like(z).to(device)  # 随机权重，形状与 z 相同
                    y_pred_weight = self.decoder(z_weight, x_label)  # 使用随机权重计算预测值
                # 初始化 q_hat（仅第一次需要采样获取初值）
                if self.q_hat[global_indices[0]][0] == 0.0:
                    with torch.no_grad():
                        x_expanded_init = data[:, :-1].unsqueeze(1).repeat(1, self.samplingnumber, 1).view(-1, self.labeldim).to(device)
                        z_init = torch.randn(current_batch_size * self.samplingnumber, latent_dim).to(device)
                        y_init = self.decoder(z_init, x_expanded_init)
                        y_reshaped_init = y_init.view(current_batch_size, n_samples)
                        q_local = torch.quantile(y_reshaped_init, self.target_quantile, dim=1, keepdim=True)
                        del x_expanded_init, z_init, y_init, y_reshaped_init

                with torch.no_grad():
                    # 构建 q_hat tensor
                    q_for_indicator = torch.zeros(current_batch_size, 1, device=device)
                    for i, global_idx in enumerate(global_indices):
                        if global_idx < len(self.q_hat) and self.q_hat[global_idx][0] != 0.0:
                            q_for_indicator[i, 0] = self.q_hat[global_idx][0]
                        else:
                            q_for_indicator[i, 0] = q_local[i, 0]
                    
                    # 直接用预测值和 quantile value 比较计算 indicator
                    indicator = (y_pred <= q_for_indicator).float()
                    
                    # 计算 Newsvendor 权重
                    diff = q_for_indicator- y_true
                    nv_weights = torch.where(diff < 0, 
                                            torch.tensor(-self.cu/(self.cu+self.co)).to(device), 
                                            torch.tensor(self.co/(self.co+self.cu)).to(device))
                    final_weights = indicator * nv_weights

                # 计算梯度
                grad_outputs = torch.ones_like(y_pred)
                grads_z = torch.autograd.grad(y_pred, z, grad_outputs=grad_outputs, 
                                            create_graph=True, retain_graph=True)[0]
                h_prime = grads_z[:, dim_i].view(-1, 1)

                grad_h_prime = torch.autograd.grad(h_prime, z, grad_outputs=torch.ones_like(h_prime),
                                                create_graph=True, retain_graph=True)[0]
                h_double_prime = grad_h_prime[:, dim_i].view(-1, 1)
                score = -z[:, dim_i].view(-1, 1)

                # 计算 Psi
                epsilon = 1e-6
                h_prime_inv = 1.0 / (h_prime + epsilon * torch.sign(h_prime))
                psi_2 = h_prime_inv * (score - h_double_prime * h_prime_inv)
                psi_2 = torch.clamp(psi_2, -100, 100).detach()
                h_prime_inv = torch.clamp(h_prime_inv, -100, 100).detach()

                # 计算 Surrogate Loss（每个点一个预测值，无需 reshape）
                surrogate_loss = (y_pred * psi_2 + h_prime * h_prime_inv) * final_weights

                global_loss = surrogate_loss.mean()
                decoder_optimizer.zero_grad()
                global_loss.backward()

                # 提取 Batch 平均梯度 \bar{G}_1
                bar_G1 = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                        for p in self.decoder.parameters()]
                decoder_optimizer.zero_grad()

                # 2. 算全局平均密度 \bar{G}_2 (标量)
                with torch.no_grad():
                    g2_vals = psi_2 * indicator
                    bar_G2 = g2_vals.mean().item()  # 只有一个数字！极其省显存！
                    bar_G2 = max(bar_G2, 1e-4)      # 简单兜底

                # 3. 更新唯一的全局 D_k
                # 假设 self.global_D 已经初始化为全 0 向量
                alpha_k = 1 / (self.k_step ** 0.55)
                for d_val, g1_val in zip(self.global_D, bar_G1):
                    update = g1_val - bar_G2 * d_val
                    d_val.add_(alpha_k * update)    # In-place 更新全局 D
                    d_val.clamp_(-1.0, 1.0)         # 防御性截断

                # 4. 外层慢尺度更新网络参数
                gamma_k = 1 / (self.k_step ** 0.9)
                for param, d_val in zip(self.decoder.parameters(), self.global_D):
                    if param.grad is None:
                        param.grad = d_val.clone() * self.lambda_gradient
                    else:
                        param.grad += d_val.clone() * self.lambda_gradient


                
                
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                decoder_optimizer.step()
                self.k_step += 1



                beta_k = 1 / ((self.k_step ) ** 0.6)  # q_hat的更新步长
            
                # 使用全局索引保存 Q_hat
                with torch.no_grad():
                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.q_hat):
                            continue
                        
                        # 第一次迭代：使用采样分位数 q_local 作为初始值
                        if self.q_hat[global_idx][0] == 0.0:
                            self.q_hat[global_idx] = [q_local[i].item()]
                            self.q_hat_list[global_idx].append(q_local[i].item())
                        else:
                            # 直接用预测值与 q_hat 比较计算 indicator
                            q_hat_current = self.q_hat[global_idx][0]
                            indicator_val = float(y_pred[i, 0].item() <= q_hat_current)
                            
                            # q_hat 更新公式: q_hat_{k+1} = q_hat_k + beta_k * (target_quantile - indicator)
                            q_hat_new = q_hat_current + beta_k * (self.target_quantile - indicator_val)
                            
                            # 更新q_hat
                            self.q_hat[global_idx] = [q_hat_new]
                            self.q_hat_list[global_idx].append(q_hat_new)

                self.k_step += 1
                
                
                try:
                    del z, y_pred, surrogate_loss
                    del global_loss,g2_vals
                    del h_prime, h_double_prime, score, psi_2, h_prime_inv, final_weights
                    del q_for_indicator, indicator
                    del diff, nv_weights, grad_outputs, grads_z, grad_h_prime, x_label, y_true
                    del im, im_label
                except:
                    pass
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            if stop_training_due_to_nan:
                print("Stopping training because non-finite loss was detected.")
                break
                    
                
            sum_dk /= len(data_loader)
            sum_vae_loss /= len(data_loader)
            sum_q_loss /= len(data_loader)
            self.loss_history['vae_loss'].append(sum_vae_loss)
            self.loss_history['quantile_loss'].append(sum_q_loss)
            vae_loss_list.append(sum_vae_loss)
            quantile_loss_list.append(sum_q_loss)
            self.D_hat_avg.append(sum_dk)
            val_loss = 0.0
            quantile_val = 0.0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    batch = val_batch.to(device)
                    im = batch[:, -1].reshape(-1, 1).to(device)
                    im_label = batch[:, :-1].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    quantile_val += self.quantile_loss(val_recon_im, im).item()
                    val_loss += val_vae_loss.item()
            val_loss /= len(valdata_loader)
            quantile_val /= len(valdata_loader)
            val_vae_loss_list.append(val_loss)
            val_quantile_loss_list.append(quantile_val)
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_quantile_loss'].append(quantile_val)
            loss_new = val_loss
            if loss_new < best_loss and not np.isnan(vae_loss_list[-1]):
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                #save model
                torch.save(self.state_dict(), save_pth)
                
                # 保存最佳模型（简化版本）
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
            if np.isnan(vae_loss_list[-1]):
                print("NaN detected in VAE loss, stopping training.")
                break
            # if np.abs(global_loss.item()) < 1e-3:
            #     print(f"Abnormal G2 value detected: {global_g2.item()}, stopping training.")
            #     break
            
            
        return {
            "vae_loss": vae_loss_list,
            "quantile_loss": quantile_loss_list,
            "val_vae_loss": val_vae_loss_list,
            "val_quantile_loss": val_quantile_loss_list,
            "message": f"SGD training completed: {self.epoch} epochs, {len(data_loader)} batches per epoch"
        }


    
    
    def train_step_sqo_vectorized_SGD_global2(self, data_loader,valdata_loader, early_stopping,  batch_size,ifdecoderonly = False, save_tag=None):
        """
        SGD 版本：使用 DataLoader 进行随机梯度下降
        data_loader: PyTorch DataLoader，每个batch返回 (data, indices)
                    其中 indices 是数据在原始数据集中的全局索引
        batch_size: batch 大小
        """
        
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=1e-4)
        latent_dim = self.latent
        n_samples = self.samplingnumber
        vae_loss_list = []
        val_vae_loss_list = []
        val_quantile_loss_list = []
        quantile_loss_list = []
        optimizer = self.optimizer
        best_loss = float('inf')
        early_stopping_counter = 0
        stop_training_due_to_nan = False
        save_pth = self.get_save_path(save_tag)
        if not os.path.exists(self.save_loss):
            os.makedirs(self.save_loss)
        for epoch in range(self.epoch):
            sum_dk = 0.0
            sum_vae_loss  = 0
            sum_q_loss = 0            
            for batch_idx, batch_data in enumerate(data_loader):
                # 如果 DataLoader 返回 (data, indices)
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, global_indices = batch_data
                    global_indices = global_indices.cpu().numpy()
                else:
                    # 如果没有索引，根据 batch_idx 计算全局索引
                    data = batch_data
                    global_indices = np.arange(batch_idx * batch_size, 
                                               min((batch_idx + 1) * batch_size, self.data_len))
                
                current_batch_size = data.shape[0]
                device = self.device
                im = data[:, -1].reshape(-1, 1).to(device)
                im_label = data[:, :-1].to(device)
                
                
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                if not ifdecoderonly:
                    if not torch.isfinite(vae_loss):
                        print(f"Non-finite VAE loss detected at epoch {epoch+1}, batch {batch_idx}: {vae_loss.item()}")
                        stop_training_due_to_nan = True
                        break

                    vae_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    vae_optimizer.step()
                    vae_loss_val = vae_loss.item()
                else:
                    vae_loss_val = vae_loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx}, VAE Loss: {vae_loss_val:.4f}")
                sum_vae_loss += vae_loss_val
                sum_q_loss += self.quantile_loss(recon_im, im).item()
                # 清理VAE阶段的显存
                del recon_im, mu, logvar, vae_loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                x_label = data[:, :-1].to(device)
                y_true = data[:, -1].unsqueeze(1).to(device)
                # 采样单个 z（每个数据点一个预测）
                z = torch.randn(current_batch_size, latent_dim).to(device)
                z.requires_grad_(True)
                
                # 随机选择维度
                dim_i = np.random.randint(0, latent_dim)

                y_pred = self.decoder(z, x_label)
                with torch.no_grad():
                    z_weight = torch.randn_like(z).to(device)  # 随机权重，形状与 z 相同
                    y_pred_weight = self.decoder(z_weight, x_label)  # 使用随机权重计算预测值
                # 初始化 q_hat（仅第一次需要采样获取初值）
                if self.q_hat[global_indices[0]][0] == 0.0:
                    with torch.no_grad():
                        x_expanded_init = data[:, :-1].unsqueeze(1).repeat(1, self.samplingnumber, 1).view(-1, self.labeldim).to(device)
                        z_init = torch.randn(current_batch_size * self.samplingnumber, latent_dim).to(device)
                        y_init = self.decoder(z_init, x_expanded_init)
                        y_reshaped_init = y_init.view(current_batch_size, n_samples)
                        q_local = torch.quantile(y_reshaped_init, self.target_quantile, dim=1, keepdim=True)
                        del x_expanded_init, z_init, y_init, y_reshaped_init

                with torch.no_grad():
                    # 构建 q_hat tensor
                    q_for_indicator = torch.zeros(current_batch_size, 1, device=device)
                    for i, global_idx in enumerate(global_indices):
                        if global_idx < len(self.q_hat) and self.q_hat[global_idx][0] != 0.0:
                            q_for_indicator[i, 0] = self.q_hat[global_idx][0]
                        else:
                            q_for_indicator[i, 0] = q_local[i, 0]
                    
                    # 直接用预测值和 quantile value 比较计算 indicator
                    indicator = (y_pred <= q_for_indicator).float()
                    
                    # 计算 Newsvendor 权重
                    diff = q_for_indicator- y_true
                    nv_weights = torch.where(diff < 0, 
                                            torch.tensor(-self.cu/(self.cu+self.co)).to(device), 
                                            torch.tensor(self.co/(self.co+self.cu)).to(device))
                    final_weights = indicator * nv_weights

                # 计算梯度
                grad_outputs = torch.ones_like(y_pred)
                grads_z = torch.autograd.grad(y_pred, z, grad_outputs=grad_outputs, 
                                            create_graph=True, retain_graph=True)[0]
                h_prime = grads_z[:, dim_i].view(-1, 1)

                grad_h_prime = torch.autograd.grad(h_prime, z, grad_outputs=torch.ones_like(h_prime),
                                                create_graph=True, retain_graph=True)[0]
                h_double_prime = grad_h_prime[:, dim_i].view(-1, 1)
                score = -z[:, dim_i].view(-1, 1)

                # 计算 Psi
                epsilon = 1e-6
                h_prime_inv = 1.0 / (h_prime + epsilon * torch.sign(h_prime))
                psi_2 = h_prime_inv * (score - h_double_prime * h_prime_inv)
                psi_2 = torch.clamp(psi_2, -100, 100).detach()
                h_prime_inv = torch.clamp(h_prime_inv, -100, 100).detach()


                surrogate_loss = (y_pred * psi_2 + h_prime * h_prime_inv) * final_weights


                with torch.no_grad():
                    # 警告：必须加绝对值和强力 clamp！
                    # 如果不 clamp 到 1e-3，网络大概率在第 1 个 epoch 就会因为极小的 psi_2 产生 NaN 梯度爆炸
                    safe_psi2 = torch.clamp(torch.abs(psi_2), min=1e-3)
                    
                    # 构建逐点逆密度权重 (Point-wise Inverse G2)
                    # 因为 G2_i = psi_2_i * indicator_i
                    # 当 indicator 为 1 时，我们需要除以 psi_2_i；当 indicator 为 0 时，分子本就为 0，保持 0 即可
                    pointwise_inv_g2 = (1.0 / safe_psi2) * indicator 

                # 将逆密度权重乘到单点 Loss 上，求均值
                # 此时 loss_to_backward 的求导数学期望精精确确就是： 1/B * \sum (G1_i * G3_i / G2_i)
                pointwise_ratio_loss = (surrogate_loss * pointwise_inv_g2).mean()

                decoder_optimizer.zero_grad()
                pointwise_ratio_loss.backward()

                # 提取算好的目标均值梯度
                grad_ratio_target = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                                    for p in self.decoder.parameters()]
                decoder_optimizer.zero_grad()


                alpha_k = 1 / (self.k_step ** 0.55)
                for d_val, target_val in zip(self.global_D, grad_ratio_target):
                    # 完全对应你的公式 (9') 的括号内部逻辑
                    update = target_val - d_val
                    d_val.add_(alpha_k * update)
                    # 这里的截断比 Batch-Smoothed 方法更重要，因为单点除法极不稳定！
                    d_val.clamp_(-1.0, 1.0) 

                # 4. 外层慢尺度更新网络参数
                gamma_k = 1 / (self.k_step ** 0.9)
                for param, d_val in zip(self.decoder.parameters(), self.global_D):
                    if param.grad is None:
                        param.grad = d_val.clone() * self.lambda_gradient
                    else:
                        param.grad += d_val.clone() * self.lambda_gradient


                
                
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                decoder_optimizer.step()
                self.k_step += 1



                beta_k = 1 / ((self.k_step ) ** 0.6)  # q_hat的更新步长
            
                # 使用全局索引保存 Q_hat
                with torch.no_grad():
                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.q_hat):
                            continue
                        
                        # 第一次迭代：使用采样分位数 q_local 作为初始值
                        if self.q_hat[global_idx][0] == 0.0:
                            self.q_hat[global_idx] = [q_local[i].item()]
                            self.q_hat_list[global_idx].append(q_local[i].item())
                        else:
                            # 直接用预测值与 q_hat 比较计算 indicator
                            q_hat_current = self.q_hat[global_idx][0]
                            indicator_val = float(y_pred[i, 0].item() <= q_hat_current)
                            
                            # q_hat 更新公式: q_hat_{k+1} = q_hat_k + beta_k * (target_quantile - indicator)
                            q_hat_new = q_hat_current + beta_k * (self.target_quantile - indicator_val)
                            
                            # 更新q_hat
                            self.q_hat[global_idx] = [q_hat_new]
                            self.q_hat_list[global_idx].append(q_hat_new)

                self.k_step += 1
                
                
                try:
                    del z, y_pred, surrogate_loss
                    del global_loss,g2_vals
                    del h_prime, h_double_prime, score, psi_2, h_prime_inv, final_weights
                    del q_for_indicator, indicator
                    del diff, nv_weights, grad_outputs, grads_z, grad_h_prime, x_label, y_true
                    del im, im_label
                except:
                    pass
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            if stop_training_due_to_nan:
                print("Stopping training because non-finite loss was detected.")
                break
                    
                
            sum_dk /= len(data_loader)
            sum_vae_loss /= len(data_loader)
            sum_q_loss /= len(data_loader)
            self.loss_history['vae_loss'].append(sum_vae_loss)
            self.loss_history['quantile_loss'].append(sum_q_loss)
            vae_loss_list.append(sum_vae_loss)
            quantile_loss_list.append(sum_q_loss)
            self.D_hat_avg.append(sum_dk)
            val_loss = 0.0
            quantile_val = 0.0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    batch = val_batch.to(device)
                    im = batch[:, -1].reshape(-1, 1).to(device)
                    im_label = batch[:, :-1].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    quantile_val += self.quantile_loss(val_recon_im, im).item()
                    val_loss += val_vae_loss.item()
            val_loss /= len(valdata_loader)
            quantile_val /= len(valdata_loader)
            val_vae_loss_list.append(val_loss)
            val_quantile_loss_list.append(quantile_val)
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_quantile_loss'].append(quantile_val)
            loss_new = val_loss
            if loss_new < best_loss and not np.isnan(vae_loss_list[-1]):
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                #save model
                torch.save(self.state_dict(), save_pth)
                
                # 保存最佳模型（简化版本）
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
            if np.isnan(vae_loss_list[-1]):
                print("NaN detected in VAE loss, stopping training.")
                break
            # if np.abs(global_loss.item()) < 1e-3:
            #     print(f"Abnormal G2 value detected: {global_g2.item()}, stopping training.")
            #     break
            
            
        return {
            "vae_loss": vae_loss_list,
            "quantile_loss": quantile_loss_list,
            "val_vae_loss": val_vae_loss_list,
            "val_quantile_loss": val_quantile_loss_list,
            "message": f"SGD training completed: {self.epoch} epochs, {len(data_loader)} batches per epoch"
        }

        
    
    
    
    
    def train_step_sqo_vectorized_SGD_LR(self, data_loader,valdata_loader, early_stopping,  batch_size,ifdecoderonly = False,ifsave = False, save_tag=None):
        """
        SGD 版本：使用 DataLoader 进行随机梯度下降
        data_loader: PyTorch DataLoader，每个batch返回 (data, indices)
                    其中 indices 是数据在原始数据集中的全局索引
        batch_size: batch 大小
        """
        
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=1e-4)
        latent_dim = self.latent
        n_samples = self.samplingnumber
        vae_loss_list = []
        val_vae_loss_list = []
        val_quantile_loss_list = []
        quantile_loss_list = []
        optimizer = self.optimizer
        best_loss = float('inf')
        early_stopping_counter = 0
        stop_training_due_to_nan = False
        xlsx_save_pth = self.get_save_xlsx_path(save_tag)
        os.makedirs(self.save_xlsx, exist_ok=True)
        save_pth = self.get_save_path(save_tag)
        if not os.path.exists(self.save_loss):
            os.makedirs(self.save_loss)
        for epoch in range(self.epoch):
            sum_dk = 0.0
            sum_vae_loss  = 0
            sum_q_loss = 0            
            for batch_idx, batch_data in enumerate(data_loader):
                # 如果 DataLoader 返回 (data, indices)
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, global_indices = batch_data
                    global_indices = global_indices.cpu().numpy()
                else:
                    # 如果没有索引，根据 batch_idx 计算全局索引
                    data = batch_data
                    global_indices = np.arange(batch_idx * batch_size, 
                                               min((batch_idx + 1) * batch_size, self.data_len))
                
                current_batch_size = data.shape[0]
                device = self.device
                im = data[:, -1].reshape(-1, 1).to(device)
                im_label = data[:, :-1].to(device)
                
                
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                
                

                vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
                grad_vae_raw = torch.autograd.grad(
                    vae_loss,
                    vae_params,
                    create_graph=True,
                    allow_unused=True,
                )
                grad_vae = [
                    g if g is not None else torch.zeros_like(p)
                    for p, g in zip(vae_params, grad_vae_raw)
                ]
                
                if not ifdecoderonly:
                    vae_loss_val = vae_loss.item()
                else:
                    vae_loss_val = 0.0  # 如果只训练 decoder，VAE loss 不计算
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx}, VAE Loss: {vae_loss_val:.4f}")
                sum_vae_loss += vae_loss_val
                sum_q_loss += self.quantile_loss(recon_im, im).item()
                # 清理VAE阶段的显存
                del recon_im, mu, logvar, vae_loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                for loop in range(self.innerloop):
                    x_label = data[:, :-1].to(device)
                    y_true = data[:, -1].unsqueeze(1).to(device)
                    # 采样单个 z（每个数据点一个预测）
                    z = torch.randn(current_batch_size, latent_dim).to(device)
                    z.requires_grad_(True)
                    
                    # 随机选择维度
                    dim_i = np.random.randint(0, latent_dim)

                    y_pred = self.decoder(z, x_label)
                    with torch.no_grad():
                        z_weight = torch.randn_like(z).to(device)  # 随机权重，形状与 z 相同
                        y_pred_weight = self.decoder(z_weight, x_label)  # 使用随机权重计算预测值
                    # 初始化 q_hat（仅第一次需要采样获取初值）
                    if self.q_hat[global_indices[0]][0] == 0.0:
                        with torch.no_grad():
                            x_expanded_init = data[:, :-1].unsqueeze(1).repeat(1, self.samplingnumber, 1).view(-1, self.labeldim).to(device)
                            z_init = torch.randn(current_batch_size * self.samplingnumber, latent_dim).to(device)
                            y_init = self.decoder(z_init, x_expanded_init)
                            y_reshaped_init = y_init.view(current_batch_size, n_samples)
                            q_local = torch.quantile(y_reshaped_init, self.target_quantile, dim=1, keepdim=True)
                            del x_expanded_init, z_init, y_init, y_reshaped_init

                    with torch.no_grad():
                        # 构建 q_hat tensor
                        q_for_indicator = torch.zeros(current_batch_size, 1, device=device)
                        for i, global_idx in enumerate(global_indices):
                            if global_idx < len(self.q_hat) and self.q_hat[global_idx][0] != 0.0:
                                q_for_indicator[i, 0] = self.q_hat[global_idx][0]
                            else:
                                q_for_indicator[i, 0] = q_local[i, 0]
                        
                        # 直接用预测值和 quantile value 比较计算 indicator
                        indicator = (y_pred <= q_for_indicator).float()
                        
                        # 计算 Newsvendor 权重
                        diff = q_for_indicator- y_true
                        nv_weights = torch.where(diff < 0, 
                                                torch.tensor(-self.cu/(self.cu+self.co)).to(device), 
                                                torch.tensor(self.co/(self.co+self.cu)).to(device))
                        final_weights = indicator * nv_weights
                   
                    # 计算梯度
                    grad_outputs = torch.ones_like(y_pred)
                    grads_z = torch.autograd.grad(y_pred, z, grad_outputs=grad_outputs, 
                                                create_graph=True, retain_graph=True)[0]
                    h_prime = grads_z[:, dim_i].view(-1, 1)

                    grad_h_prime = torch.autograd.grad(h_prime, z, grad_outputs=torch.ones_like(h_prime),
                                                    create_graph=True, retain_graph=True)[0]
                    h_double_prime = grad_h_prime[:, dim_i].view(-1, 1)
                    score = -z[:, dim_i].view(-1, 1)

                    # 计算 Psi
                    epsilon = 1e-6
                    h_prime_inv = 1.0 / (h_prime + epsilon * torch.sign(h_prime))
                    psi_2 = h_prime_inv * (score - h_double_prime * h_prime_inv)
                    psi_2 = torch.clamp(psi_2, -100, 100).detach()
                    h_prime_inv = torch.clamp(h_prime_inv, -100, 100).detach()

                    # 计算 Surrogate Loss（每个点一个预测值，无需 reshape）
                    surrogate_loss = (y_pred * psi_2 + h_prime * h_prime_inv) * final_weights
                    global_loss = surrogate_loss.mean()

                    if not torch.isfinite(global_loss):
                        print(f"Non-finite GLR loss detected at epoch {epoch+1}, batch {batch_idx}: {global_loss.item()}")
                        stop_training_due_to_nan = True
                        break

                    decoder_optimizer.zero_grad()
                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch+1}, Batch {batch_idx}, GLR Loss: {global_loss.item():.4f}")
                    global_loss.backward()
                    
                    # 提取 G1 梯度
                    g1_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                            for p in self.decoder.parameters()]
                    decoder_optimizer.zero_grad()

                    # 计算 G2
                    with torch.no_grad():
                        g2_vals = psi_2*indicator 
                        g2_per_point = g2_vals.view(current_batch_size)
                        g2_per_point = torch.clamp(g2_per_point, min=1e-4, max=10.0)
                        global_g2 = g2_per_point.mean()

                    # 更新参数
                    k = self.k_step
                    gamma_k = 1 / (k ** 0.55)
                    beta_k = 1 / ((k ) ** 0.6)  # q_hat的更新步长
                
                    # 使用全局索引保存 Q_hat


                    # 使用全局索引更新 D_hat
                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.D_hat):
                            continue
                        
                        g2_i = g2_per_point[i].item()
                        
                        new_D_i = []
                        for d_val, g1_val in zip(self.D_hat[global_idx], g1_grads):
                            d_val_device = d_val.to(g1_val.device) if d_val.device != g1_val.device else d_val
                            update = g1_val - g2_i * d_val_device
                            d_new = d_val_device + gamma_k * update
                            d_new = torch.clamp(d_new, -1.0, 1.0).detach()
                            new_D_i.append(d_new)
                        self.D_hat[global_idx] = new_D_i


                with torch.no_grad():
                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.q_hat):
                            continue
                        
                        # 第一次迭代：使用采样分位数 q_local 作为初始值
                        if self.q_hat[global_idx][0] == 0.0:
                            self.q_hat[global_idx] = [q_local[i].item()]
                            self.q_hat_list[global_idx].append(q_local[i].item())
                        else:
                            # 直接用预测值与 q_hat 比较计算 indicator
                            q_hat_current = self.q_hat[global_idx][0]
                            indicator_val = float(y_pred[i, 0].item() <= q_hat_current)
                            
                            # q_hat 更新公式: q_hat_{k+1} = q_hat_k + beta_k * (target_quantile - indicator)
                            q_hat_new = q_hat_current + beta_k * (self.target_quantile - indicator_val)
                            
                            # 更新q_hat
                            self.q_hat[global_idx] = [q_hat_new]
                            self.q_hat_list[global_idx].append(q_hat_new)


                # 计算当前 batch 的平均 D
                avg_D = [torch.zeros_like(p) for p in self.decoder.parameters()]
                valid_count = 0
                for i, global_idx in enumerate(global_indices):
                    if global_idx < len(self.D_hat):
                        for j, d_val in enumerate(self.D_hat[global_idx]):
                            avg_D[j] = avg_D[j] + d_val.to(avg_D[j].device)
                        valid_count += 1
                
                if valid_count > 0:
                    for j in range(len(avg_D)):
                        avg_D[j] = avg_D[j] / valid_count
                avg_D = [d_val.detach() for d_val in avg_D]
                
                # 计算每个数据点的 D_hat 范数的平均值（每个batch一个标量）
                D_norms_per_point = []
                for i, global_idx in enumerate(global_indices):
                    if global_idx < len(self.D_hat):
                        # 计算该数据点所有参数的范数的平均值
                        point_d_norms = [torch.norm(d_val).item() for d_val in self.D_hat[global_idx]]
                        point_d_norm_mean = np.mean(point_d_norms) if point_d_norms else 0.0
                        D_norms_per_point.append(point_d_norm_mean)
                
                # 对batch中所有数据点的D范数求平均，得到该batch的一个标量
                batch_D_norm_mean = np.mean(D_norms_per_point) if D_norms_per_point else 0.0
                sum_dk += batch_D_norm_mean
                
                for param, d_val in zip(self.decoder.parameters(), avg_D):
                    if param.grad is None:
                        param.grad = d_val.clone()*self.lambda_gradient
                    else:
                        param.grad.copy_(d_val)
                for param, grad in zip(vae_params, grad_vae):
                    grad_to_add = grad.detach()
                    if param.grad is None:
                        param.grad = grad_to_add.clone()
                    else:
                        param.grad += grad_to_add
                
                
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                vae_optimizer.step()
                self.k_step += 1
                
                
                try:
                    del z, y_pred, surrogate_loss
                    del global_loss, g1_grads, g2_vals, g2_per_point, global_g2
                    del h_prime, h_double_prime, score, psi_2, h_prime_inv, final_weights
                    del q_for_indicator, indicator
                    del diff, nv_weights, grad_outputs, grads_z, grad_h_prime, x_label, y_true
                    del avg_D, im, im_label
                except:
                    pass
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            if stop_training_due_to_nan:
                print("Stopping training because non-finite loss was detected.")
                break
                    
                
            sum_dk /= len(data_loader)
            sum_vae_loss /= len(data_loader)
            sum_q_loss /= len(data_loader)
            self.loss_history['vae_loss'].append(sum_vae_loss)
            self.loss_history['quantile_loss'].append(sum_q_loss)
            vae_loss_list.append(sum_vae_loss)
            quantile_loss_list.append(sum_q_loss)
            self.D_hat_avg.append(sum_dk)
            val_loss = 0.0
            quantile_val = 0.0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    batch = val_batch.to(device)
                    im = batch[:, -1].reshape(-1, 1).to(device)
                    im_label = batch[:, :-1].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    quantile_val += self.quantile_loss(val_recon_im, im).item()
                    val_loss += val_vae_loss.item()
            val_loss /= len(valdata_loader)
            quantile_val /= len(valdata_loader)
            val_vae_loss_list.append(val_loss)
            val_quantile_loss_list.append(quantile_val)
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_quantile_loss'].append(quantile_val)
            loss_new = val_loss
            if loss_new < best_loss and not np.isnan(vae_loss_list[-1]):
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                #save model
                if not ifsave:
                    torch.save(self.state_dict(), save_pth)
                
                # 保存最佳模型（简化版本）
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
            if np.isnan(vae_loss_list[-1]):
                print("NaN detected in VAE loss, stopping training.")
                break
            # if np.abs(global_loss.item()) < 1e3:
            #     print("GLR loss too small, stopping training.")
            #     break
            
        if ifsave:
            datareturn = {
                "vae_loss": vae_loss_list,
                "quantile_loss": quantile_loss_list,
                "val_vae_loss": val_vae_loss_list,
                "val_quantile_loss": val_quantile_loss_list,
            }
            df = pd.DataFrame(datareturn)
            df.to_excel(xlsx_save_pth, index=False)


        return {
            "vae_loss": vae_loss_list,
            "quantile_loss": quantile_loss_list,
            "val_vae_loss": val_vae_loss_list,
            "val_quantile_loss": val_quantile_loss_list,
            "message": f"SGD training completed: {self.epoch} epochs, {len(data_loader)} batches per epoch"
        }



    
    def train_step_sqo_vectorized_SGD_LR_global(self, data_loader,valdata_loader, early_stopping,  batch_size,ifdecoderonly = False,ifsave = False, save_tag=None):
        """
        SGD 版本：使用 DataLoader 进行随机梯度下降
        data_loader: PyTorch DataLoader，每个batch返回 (data, indices)
                    其中 indices 是数据在原始数据集中的全局索引
        batch_size: batch 大小
        """
        
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=1e-4)
        latent_dim = self.latent
        n_samples = self.samplingnumber
        vae_loss_list = []
        val_vae_loss_list = []
        val_quantile_loss_list = []
        quantile_loss_list = []
        optimizer = self.optimizer
        best_loss = float('inf')
        early_stopping_counter = 0
        stop_training_due_to_nan = False
        xlsx_save_pth = self.get_save_xlsx_path(save_tag)
        os.makedirs(self.save_xlsx, exist_ok=True)
        save_pth = self.get_save_path(save_tag)
        if not os.path.exists(self.save_loss):
            os.makedirs(self.save_loss)
        for epoch in range(self.epoch):
            sum_dk = 0.0
            sum_vae_loss  = 0
            sum_q_loss = 0            
            for batch_idx, batch_data in enumerate(data_loader):
                # 如果 DataLoader 返回 (data, indices)
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, global_indices = batch_data
                    global_indices = global_indices.cpu().numpy()
                else:
                    # 如果没有索引，根据 batch_idx 计算全局索引
                    data = batch_data
                    global_indices = np.arange(batch_idx * batch_size, 
                                               min((batch_idx + 1) * batch_size, self.data_len))
                
                current_batch_size = data.shape[0]
                device = self.device
                im = data[:, -1].reshape(-1, 1).to(device)
                im_label = data[:, :-1].to(device)
                
                
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                
                

                vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
                grad_vae_raw = torch.autograd.grad(
                    vae_loss,
                    vae_params,
                    create_graph=True,
                    allow_unused=True,
                )
                grad_vae = [
                    g if g is not None else torch.zeros_like(p)
                    for p, g in zip(vae_params, grad_vae_raw)
                ]
                
                if not ifdecoderonly:
                    vae_loss_val = vae_loss.item()
                else:
                    vae_loss_val = 0.0  # 如果只训练 decoder，VAE loss 不计算
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx}, VAE Loss: {vae_loss_val:.4f}")
                sum_vae_loss += vae_loss_val
                sum_q_loss += self.quantile_loss(recon_im, im).item()
                # 清理VAE阶段的显存
                del recon_im, mu, logvar, vae_loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                x_label = data[:, :-1].to(device)
                y_true = data[:, -1].unsqueeze(1).to(device)
                # 采样单个 z（每个数据点一个预测）
                z = torch.randn(current_batch_size, latent_dim).to(device)
                z.requires_grad_(True)
                
                # 随机选择维度
                dim_i = np.random.randint(0, latent_dim)

                y_pred = self.decoder(z, x_label)
                with torch.no_grad():
                    z_weight = torch.randn_like(z).to(device)  # 随机权重，形状与 z 相同
                    y_pred_weight = self.decoder(z_weight, x_label)  # 使用随机权重计算预测值
                # 初始化 q_hat（仅第一次需要采样获取初值）
                if self.q_hat[global_indices[0]][0] == 0.0:
                    with torch.no_grad():
                        x_expanded_init = data[:, :-1].unsqueeze(1).repeat(1, self.samplingnumber, 1).view(-1, self.labeldim).to(device)
                        z_init = torch.randn(current_batch_size * self.samplingnumber, latent_dim).to(device)
                        y_init = self.decoder(z_init, x_expanded_init)
                        y_reshaped_init = y_init.view(current_batch_size, n_samples)
                        q_local = torch.quantile(y_reshaped_init, self.target_quantile, dim=1, keepdim=True)
                        del x_expanded_init, z_init, y_init, y_reshaped_init

                with torch.no_grad():
                    # 构建 q_hat tensor
                    q_for_indicator = torch.zeros(current_batch_size, 1, device=device)
                    for i, global_idx in enumerate(global_indices):
                        if global_idx < len(self.q_hat) and self.q_hat[global_idx][0] != 0.0:
                            q_for_indicator[i, 0] = self.q_hat[global_idx][0]
                        else:
                            q_for_indicator[i, 0] = q_local[i, 0]
                    
                    # 直接用预测值和 quantile value 比较计算 indicator
                    indicator = (y_pred <= q_for_indicator).float()
                    
                    # 计算 Newsvendor 权重
                    diff = q_for_indicator- y_true
                    nv_weights = torch.where(diff < 0, 
                                            torch.tensor(-self.cu/(self.cu+self.co)).to(device), 
                                            torch.tensor(self.co/(self.co+self.cu)).to(device))
                    final_weights = indicator * nv_weights

                # 计算梯度
                grad_outputs = torch.ones_like(y_pred)
                grads_z = torch.autograd.grad(y_pred, z, grad_outputs=grad_outputs, 
                                            create_graph=True, retain_graph=True)[0]
                h_prime = grads_z[:, dim_i].view(-1, 1)

                grad_h_prime = torch.autograd.grad(h_prime, z, grad_outputs=torch.ones_like(h_prime),
                                                create_graph=True, retain_graph=True)[0]
                h_double_prime = grad_h_prime[:, dim_i].view(-1, 1)
                score = -z[:, dim_i].view(-1, 1)

                # 计算 Psi
                epsilon = 1e-6
                h_prime_inv = 1.0 / (h_prime + epsilon * torch.sign(h_prime))
                psi_2 = h_prime_inv * (score - h_double_prime * h_prime_inv)
                psi_2 = torch.clamp(psi_2, -100, 100).detach()
                h_prime_inv = torch.clamp(h_prime_inv, -100, 100).detach()

                # 计算 Surrogate Loss（每个点一个预测值，无需 reshape）
                surrogate_loss = (y_pred * psi_2 + h_prime * h_prime_inv) * final_weights

                global_loss = surrogate_loss.mean()
                decoder_optimizer.zero_grad()
                global_loss.backward()

                # 提取 Batch 平均梯度 \bar{G}_1
                bar_G1 = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                        for p in self.decoder.parameters()]
                decoder_optimizer.zero_grad()

                # 2. 算全局平均密度 \bar{G}_2 (标量)
                with torch.no_grad():
                    g2_vals = psi_2 * indicator
                    bar_G2 = g2_vals.mean().item()  # 只有一个数字！极其省显存！
                    bar_G2 = max(bar_G2, 1e-4)      # 简单兜底

                # 3. 更新唯一的全局 D_k
                # 假设 self.global_D 已经初始化为全 0 向量
                alpha_k = 1 / (self.k_step ** 0.55)
                for d_val, g1_val in zip(self.global_D, bar_G1):
                    update = g1_val - bar_G2 * d_val
                    d_val.add_(alpha_k * update)    # In-place 更新全局 D
                    d_val.clamp_(-1.0, 1.0)         # 防御性截断

                # 4. 外层慢尺度更新网络参数
                gamma_k = 1 / (self.k_step ** 0.9)
                for param, d_val in zip(self.decoder.parameters(), self.global_D):
                    if param.grad is None:
                        param.grad = d_val.clone() * self.lambda_gradient
                    else:
                        param.grad += d_val.clone() * self.lambda_gradient

                for param, grad in zip(vae_params, grad_vae):
                    grad_to_add = grad.detach()
                    if param.grad is None:
                        param.grad = grad_to_add.clone()
                    else:
                        param.grad += grad_to_add
                
                
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                vae_optimizer.step()
                self.k_step += 1



                beta_k = 1 / ((self.k_step ) ** 0.6)  # q_hat的更新步长
            
                # 使用全局索引保存 Q_hat
                with torch.no_grad():
                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.q_hat):
                            continue
                        
                        # 第一次迭代：使用采样分位数 q_local 作为初始值
                        if self.q_hat[global_idx][0] == 0.0:
                            self.q_hat[global_idx] = [q_local[i].item()]
                            self.q_hat_list[global_idx].append(q_local[i].item())
                        else:
                            # 直接用预测值与 q_hat 比较计算 indicator
                            q_hat_current = self.q_hat[global_idx][0]
                            indicator_val = float(y_pred[i, 0].item() <= q_hat_current)
                            
                            # q_hat 更新公式: q_hat_{k+1} = q_hat_k + beta_k * (target_quantile - indicator)
                            q_hat_new = q_hat_current + beta_k * (self.target_quantile - indicator_val)
                            
                            # 更新q_hat
                            self.q_hat[global_idx] = [q_hat_new]
                            self.q_hat_list[global_idx].append(q_hat_new)

                self.k_step += 1
                try:
                    del z, y_pred, surrogate_loss
                    del global_loss,  g2_vals
                    del h_prime, h_double_prime, score, psi_2, h_prime_inv, final_weights
                    del q_for_indicator, indicator
                    del diff, nv_weights, grad_outputs, grads_z, grad_h_prime, x_label, y_true
                    del im, im_label
                except:
                    pass
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            if stop_training_due_to_nan:
                print("Stopping training because non-finite loss was detected.")
                break
                    
                
            sum_dk /= len(data_loader)
            sum_vae_loss /= len(data_loader)
            sum_q_loss /= len(data_loader)
            self.loss_history['vae_loss'].append(sum_vae_loss)
            self.loss_history['quantile_loss'].append(sum_q_loss)
            vae_loss_list.append(sum_vae_loss)
            quantile_loss_list.append(sum_q_loss)
            self.D_hat_avg.append(sum_dk)
            val_loss = 0.0
            quantile_val = 0.0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    batch = val_batch.to(device)
                    im = batch[:, -1].reshape(-1, 1).to(device)
                    im_label = batch[:, :-1].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    quantile_val += self.quantile_loss(val_recon_im, im).item()
                    val_loss += val_vae_loss.item()
            val_loss /= len(valdata_loader)
            quantile_val /= len(valdata_loader)
            val_vae_loss_list.append(val_loss)
            val_quantile_loss_list.append(quantile_val)
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_quantile_loss'].append(quantile_val)
            loss_new = val_loss
            if loss_new < best_loss and not np.isnan(vae_loss_list[-1]):
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                #save model
                if not ifsave:
                    torch.save(self.state_dict(), save_pth)
                
                # 保存最佳模型（简化版本）
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
            if np.isnan(vae_loss_list[-1]):
                print("NaN detected in VAE loss, stopping training.")
                break
            # if np.abs(global_loss.item()) < 1e3:
            #     print("GLR loss too small, stopping training.")
            #     break
            
        if ifsave:
            datareturn = {
                "vae_loss": vae_loss_list,
                "quantile_loss": quantile_loss_list,
                "val_vae_loss": val_vae_loss_list,
                "val_quantile_loss": val_quantile_loss_list,
            }
            df = pd.DataFrame(datareturn)
            df.to_excel(xlsx_save_pth, index=False)


        return {
            "vae_loss": vae_loss_list,
            "quantile_loss": quantile_loss_list,
            "val_vae_loss": val_vae_loss_list,
            "val_quantile_loss": val_quantile_loss_list,
            "message": f"SGD training completed: {self.epoch} epochs, {len(data_loader)} batches per epoch"
        }

    def train_step_sqo_vectorized_SGD_LR_globalsingleloop(self, data_loader,valdata_loader, early_stopping,  batch_size,ifdecoderonly = False,ifsave = False, save_tag=None,
                                                      ifonlyglr = False, iftwoupdate = False):
        """
        Sequential per-sample baseline for globalsingle.

        This path intentionally computes G1_i and G2_i one sample at a time so it
        can be compared against the vmap-based parallel implementation and IPA.
        """
        self._sync_auxiliary_state_device()
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        latent_dim = self.latent
        n_samples = self.samplingnumber
        vae_loss_list = []
        val_vae_loss_list = []
        val_quantile_loss_list = []
        quantile_loss_list = []
        decoder_param_names = list(OrderedDict(self.decoder.named_parameters()).keys())
        best_loss = float('inf')
        early_stopping_counter = 0
        stop_training_due_to_nan = False
        xlsx_save_pth = self.get_save_xlsx_path(save_tag)
        os.makedirs(self.save_xlsx, exist_ok=True)
        save_pth = self.get_save_path(save_tag)
        start_time_list = []
        gradient_time_list = []
        if not os.path.exists(self.save_loss):
            os.makedirs(self.save_loss)
        for epoch in range(self.epoch):
            sum_dk = 0.0
            sum_vae_loss  = 0
            sum_q_loss = 0
            for batch_idx, batch_data in enumerate(data_loader):
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, global_indices = batch_data
                    global_indices = global_indices.cpu().numpy()
                else:
                    data = batch_data
                    global_indices = np.arange(batch_idx * batch_size, 
                                            min((batch_idx + 1) * batch_size, self.data_len))
                
                current_batch_size = data.shape[0]
                device = self.device
                im = data[:, -1].reshape(-1, 1).to(device)
                im_label = data[:, :-1].to(device)
                
                # --- VAE 前向 + 梯度（对整个 batch） ---
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
                grad_vae_raw = torch.autograd.grad(
                    vae_loss,
                    vae_params,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_vae = [
                    g.detach().clone() if g is not None else torch.zeros_like(p)
                    for p, g in zip(vae_params, grad_vae_raw)
                ]
                if iftwoupdate and not ifonlyglr:
                    vae_optimizer.zero_grad()
                    for param, grad in zip(vae_params, grad_vae):
                        param.grad = grad.clone()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    vae_optimizer.step()

                if not ifdecoderonly:
                    vae_loss_val = vae_loss.item()
                else:
                    vae_loss_val = 0.0

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx}, VAE Loss: {vae_loss_val:.4f}")
                sum_vae_loss += vae_loss_val
                sum_q_loss += self.quantile_loss(recon_im, im).item()
                del recon_im, mu, logvar, vae_loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                x_label = data[:, :-1].to(device)
                y_true = data[:, -1].unsqueeze(1).to(device)

                needs_q_init = any(
                    global_idx < len(self.q_hat) and self.q_hat[global_idx][0] == 0.0
                    for global_idx in global_indices
                )
                q_local = None
                if needs_q_init:
                    with torch.no_grad():
                        x_expanded_init = data[:, :-1].unsqueeze(1).repeat(1, self.samplingnumber, 1).view(-1, self.labeldim).to(device)
                        z_init = torch.randn(current_batch_size * self.samplingnumber, latent_dim).to(device)
                        y_init = self.decoder(z_init, x_expanded_init)
                        y_reshaped_init = y_init.view(current_batch_size, n_samples)
                        q_local = torch.quantile(y_reshaped_init, self.target_quantile, dim=1, keepdim=True)
                        del x_expanded_init, z_init, y_init, y_reshaped_init

                q_values = None
                y_pred_batch = torch.zeros(current_batch_size, 1, device=device, dtype=y_true.dtype)
                beta_k = 1 / ((self.k_step) ** 0.6)
                start_time = time.time()
                for _ in range(self.innerloop):
                    q_values = self._build_q_tensor_for_batch(global_indices, q_local, device, y_true.dtype)
                    per_sample_grads, y_pred_batch, g2_batch, gradient_compute_time = self._sequential_globalsingle_innerloop(
                        x_label,
                        y_true,
                        q_values,
                        latent_dim,
                    )
                    gradient_time_list.append(gradient_compute_time)
                    alpha_k = 1 / (self.k_step ** 0.55)
                    beta_k = 1 / ((self.k_step) ** 0.6)

                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.D_hat):
                            continue

                        g2_i = g2_batch[i, 0].item()
                        for param_name, d_val in zip(decoder_param_names, self.D_hat[global_idx]):
                            g1_val = per_sample_grads[param_name][i]
                            update = g1_val / g2_i - d_val
                            d_val.add_(alpha_k * update)
                            d_val.clamp_(-1.0, 1.0)
                end_time = time.time()
                start_time_list.append(end_time - start_time)
                if q_values is None:
                    q_values = self._build_q_tensor_for_batch(global_indices, q_local, device, y_true.dtype)

                with torch.no_grad():
                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.q_hat):
                            continue

                        if self.q_hat[global_idx][0] == 0.0:
                            q_init = q_values[i, 0].item()
                            self.q_hat[global_idx] = [q_init]
                            self.q_hat_list[global_idx].append(q_init)
                        else:
                            q_hat_current = self.q_hat[global_idx][0]
                            indicator_val = float(y_pred_batch[i, 0].item() <= q_hat_current)
                            q_hat_new = q_hat_current + beta_k * (self.target_quantile - indicator_val)
                            self.q_hat[global_idx] = [q_hat_new]
                            self.q_hat_list[global_idx].append(q_hat_new)

                avg_D = [torch.zeros_like(p) for p in self.decoder.parameters()]
                valid_count = 0
                D_norms_per_point = []
                for i, global_idx in enumerate(global_indices):
                    if global_idx < len(self.D_hat):
                        point_d_norms = []
                        for j, d_val in enumerate(self.D_hat[global_idx]):
                            d_val_device = d_val.to(avg_D[j].device)
                            avg_D[j] += d_val_device
                            point_d_norms.append(torch.norm(d_val_device).item())
                        valid_count += 1
                        D_norms_per_point.append(np.mean(point_d_norms) if point_d_norms else 0.0)

                if valid_count > 0:
                    for j in range(len(avg_D)):
                        avg_D[j] = (avg_D[j] / valid_count).detach()
                batch_D_norm_mean = np.mean(D_norms_per_point) if D_norms_per_point else 0.0
                sum_dk += batch_D_norm_mean

                vae_optimizer.zero_grad()
                for param, d_val in zip(self.decoder.parameters(), avg_D):
                    param.grad = d_val.clone() * self.lambda_gradient
                if not ifonlyglr and not iftwoupdate:
                    for param, grad in zip(vae_params, grad_vae):
                        if param.grad is None:
                            param.grad = grad.clone()
                        else:
                            param.grad += grad

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                vae_optimizer.step()
                self.k_step += 1
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            if stop_training_due_to_nan:
                print("Stopping training because non-finite loss was detected.")
                break

            sum_dk /= len(data_loader)
            sum_vae_loss /= len(data_loader)
            sum_q_loss /= len(data_loader)
            self.loss_history['vae_loss'].append(sum_vae_loss)
            self.loss_history['quantile_loss'].append(sum_q_loss)
            vae_loss_list.append(sum_vae_loss)
            quantile_loss_list.append(sum_q_loss)
            self.D_hat_avg.append(sum_dk)
            val_loss = 0.0
            quantile_val = 0.0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]

                    device = next(self.parameters()).device
                    batch = val_batch.to(device)
                    im = batch[:, -1].reshape(-1, 1).to(device)
                    im_label = batch[:, :-1].to(device)

                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    quantile_val += self.quantile_loss(val_recon_im, im).item()
                    val_loss += val_vae_loss.item()
            val_loss /= len(valdata_loader)
            quantile_val /= len(valdata_loader)
            val_vae_loss_list.append(val_loss)
            val_quantile_loss_list.append(quantile_val)
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_quantile_loss'].append(quantile_val)
            loss_new = val_loss
            if loss_new < best_loss and not np.isnan(vae_loss_list[-1]):
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                if not ifsave:
                    torch.save(self.state_dict(), save_pth)
                print('-' * 10)
            else:
                early_stopping_counter += 1

            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
            if np.isnan(vae_loss_list[-1]):
                print("NaN detected in VAE loss, stopping training.")
                break

        if ifsave:
            datareturn = {
                "vae_loss": vae_loss_list,
                "quantile_loss": quantile_loss_list,
                "val_vae_loss": val_vae_loss_list,
                "val_quantile_loss": val_quantile_loss_list,
            }
            df = pd.DataFrame(datareturn)
            df.to_excel(xlsx_save_pth, index=False)

        return {
            "vae_loss": vae_loss_list,
            "quantile_loss": quantile_loss_list,
            "val_vae_loss": val_vae_loss_list,
            "val_quantile_loss": val_quantile_loss_list,
            "message": f"Sequential globalsingleloop training completed: {self.epoch} epochs, {len(data_loader)} batches per epoch",
            "avg_innerloop_time": np.mean(start_time_list) if start_time_list else 0.0,
            "avg_gradient_time": np.mean(gradient_time_list) if gradient_time_list else 0.0
        }

    def train_step_sqo_vectorized_SGD_LR_globalsingle(self, data_loader,valdata_loader, early_stopping,  batch_size,ifdecoderonly = False,ifsave = False, save_tag=None,
                                                      ifonlyglr = False, iftwoupdate = False):
        """
        SGD 版本：使用 DataLoader 进行随机梯度下降
        data_loader: PyTorch DataLoader，每个batch返回 (data, indices)
                    其中 indices 是数据在原始数据集中的全局索引
        batch_size: batch 大小
        """
        self._sync_auxiliary_state_device()
        
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=1e-4)
        latent_dim = self.latent
        n_samples = self.samplingnumber
        vae_loss_list = []
        val_vae_loss_list = []
        val_quantile_loss_list = []
        quantile_loss_list = []
        optimizer = self.optimizer
        decoder_param_names = list(OrderedDict(self.decoder.named_parameters()).keys())
        best_loss = float('inf')
        early_stopping_counter = 0
        stop_training_due_to_nan = False
        xlsx_save_pth = self.get_save_xlsx_path(save_tag)
        os.makedirs(self.save_xlsx, exist_ok=True)
        save_pth = self.get_save_path(save_tag)
        if not os.path.exists(self.save_loss):
            os.makedirs(self.save_loss)
        start_time_list = []
        gradient_time_list = []
        for epoch in range(self.epoch):
            sum_dk = 0.0
            sum_vae_loss  = 0
            sum_q_loss = 0
            for batch_idx, batch_data in enumerate(data_loader):
                # 如果 DataLoader 返回 (data, indices)
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, global_indices = batch_data
                    global_indices = global_indices.cpu().numpy()
                else:
                    # 如果没有索引，根据 batch_idx 计算全局索引
                    data = batch_data
                    global_indices = np.arange(batch_idx * batch_size, 
                                            min((batch_idx + 1) * batch_size, self.data_len))
                
                current_batch_size = data.shape[0]
                device = self.device
                im = data[:, -1].reshape(-1, 1).to(device)
                im_label = data[:, :-1].to(device)
                
                # --- VAE 前向 + 梯度（对整个 batch） ---
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]

                vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
                grad_vae_raw = torch.autograd.grad(
                    vae_loss,
                    vae_params,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_vae = [
                    g.detach().clone() if g is not None else torch.zeros_like(p)
                    for p, g in zip(vae_params, grad_vae_raw)
                ]
                if iftwoupdate and not ifonlyglr:
                    for param, grad in zip(vae_params, grad_vae):
                        if param.grad is None:
                            param.grad = grad.clone()
                        else:
                            param.grad += grad.clone()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    vae_optimizer.step()
                if not ifdecoderonly:
                    vae_loss_val = vae_loss.item()
                else:
                    vae_loss_val = 0.0
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx}, VAE Loss: {vae_loss_val:.4f}")
                sum_vae_loss += vae_loss_val
                sum_q_loss += self.quantile_loss(recon_im, im).item()
                del recon_im, mu, logvar, vae_loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                x_label = data[:, :-1].to(device)
                y_true = data[:, -1].unsqueeze(1).to(device)

                needs_q_init = any(
                    global_idx < len(self.q_hat) and self.q_hat[global_idx][0] == 0.0
                    for global_idx in global_indices
                )
                q_local = None
                if needs_q_init:
                    with torch.no_grad():
                        x_expanded_init = data[:, :-1].unsqueeze(1).repeat(1, self.samplingnumber, 1).view(-1, self.labeldim).to(device)
                        z_init = torch.randn(current_batch_size * self.samplingnumber, latent_dim).to(device)
                        y_init = self.decoder(z_init, x_expanded_init)
                        y_reshaped_init = y_init.view(current_batch_size, n_samples)
                        q_local = torch.quantile(y_reshaped_init, self.target_quantile, dim=1, keepdim=True)
                        del x_expanded_init, z_init, y_init, y_reshaped_init
                start_time = time.time()
                for _ in range(self.innerloop):
                    q_values = self._build_q_tensor_for_batch(global_indices, q_local, device, y_true.dtype)
                    per_sample_grads, y_pred_batch, g2_batch, gradient_compute_time = self._vectorized_globalsingle_innerloop(
                        x_label,
                        y_true,
                        q_values,
                        latent_dim,
                    )
                    gradient_time_list.append(gradient_compute_time)
                    alpha_k = 1 / (self.k_step ** 0.55)
                    beta_k = 1 / ((self.k_step) ** 0.6)

                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.D_hat):
                            continue

                        g2_i = g2_batch[i, 0].item()
                        for param_name, d_val in zip(decoder_param_names, self.D_hat[global_idx]):
                            g1_val = per_sample_grads[param_name][i]
                            update = g1_val / g2_i - d_val
                            d_val.add_(alpha_k * update)
                            d_val.clamp_(-1.0, 1.0)

                with torch.no_grad():
                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.q_hat):
                            continue

                        if self.q_hat[global_idx][0] == 0.0:
                            q_init = q_values[i, 0].item()
                            self.q_hat[global_idx] = [q_init]
                            self.q_hat_list[global_idx].append(q_init)
                        else:
                            q_hat_current = self.q_hat[global_idx][0]
                            indicator_val = float(y_pred_batch[i, 0].item() <= q_hat_current)
                            q_hat_new = q_hat_current + beta_k * (self.target_quantile - indicator_val)
                            self.q_hat[global_idx] = [q_hat_new]
                            self.q_hat_list[global_idx].append(q_hat_new)
                end_time = time.time()
                start_time_list.append(end_time - start_time)
                # --- 逐点循环结束后，累加 batch 内各点的 D_hat，统一更新参数 ---
                avg_D = [torch.zeros_like(p) for p in self.decoder.parameters()]
                valid_count = 0
                for i in range(current_batch_size):
                    global_idx = global_indices[i]
                    if global_idx < len(self.D_hat):
                        for j in range(len(avg_D)):
                            avg_D[j] += self.D_hat[global_idx][j]
                        valid_count += 1
                if valid_count > 0:
                    for j in range(len(avg_D)):
                        avg_D[j] /= valid_count

                vae_optimizer.zero_grad()
                for param, d_val in zip(self.decoder.parameters(), avg_D):
                    if param.grad is None:
                        param.grad = d_val.clone() * self.lambda_gradient
                    else:
                        param.grad += d_val.clone() * self.lambda_gradient
                if not ifonlyglr and not iftwoupdate:
                    for param, grad in zip(vae_params, grad_vae):
                        if param.grad is None:
                            param.grad = grad.clone()
                        else:
                            param.grad += grad
                else:
                    for param, grad in zip(vae_params, grad_vae):
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        else:
                            param.grad += torch.zeros_like(param)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                vae_optimizer.step()
                self.k_step+=1
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            if stop_training_due_to_nan:
                print("Stopping training because non-finite loss was detected.")
                break
            
                
            sum_dk /= len(data_loader)
            sum_vae_loss /= len(data_loader)
            sum_q_loss /= len(data_loader)
            self.loss_history['vae_loss'].append(sum_vae_loss)
            self.loss_history['quantile_loss'].append(sum_q_loss)
            vae_loss_list.append(sum_vae_loss)
            quantile_loss_list.append(sum_q_loss)
            self.D_hat_avg.append(sum_dk)
            val_loss = 0.0
            quantile_val = 0.0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    batch = val_batch.to(device)
                    im = batch[:, -1].reshape(-1, 1).to(device)
                    im_label = batch[:, :-1].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    quantile_val += self.quantile_loss(val_recon_im, im).item()
                    val_loss += val_vae_loss.item()
            val_loss /= len(valdata_loader)
            quantile_val /= len(valdata_loader)
            val_vae_loss_list.append(val_loss)
            val_quantile_loss_list.append(quantile_val)
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_quantile_loss'].append(quantile_val)
            loss_new = val_loss
            if loss_new < best_loss and not np.isnan(vae_loss_list[-1]):
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                #save model
                if not ifsave:
                    torch.save(self.state_dict(), save_pth)
                
                # 保存最佳模型（简化版本）
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
            if np.isnan(vae_loss_list[-1]):
                print("NaN detected in VAE loss, stopping training.")
                break
            # if np.abs(global_loss.item()) < 1e3:
            #     print("GLR loss too small, stopping training.")
            #     break
            
        if ifsave:
            datareturn = {
                "vae_loss": vae_loss_list,
                "quantile_loss": quantile_loss_list,
                "val_vae_loss": val_vae_loss_list,
                "val_quantile_loss": val_quantile_loss_list,
            }
            df = pd.DataFrame(datareturn)
            df.to_excel(xlsx_save_pth, index=False)


        return {
            "vae_loss": vae_loss_list,
            "quantile_loss": quantile_loss_list,
            "val_vae_loss": val_vae_loss_list,
            "val_quantile_loss": val_quantile_loss_list,
            "message": f"SGD training completed: {self.epoch} epochs, {len(data_loader)} batches per epoch",
            "avg_innerloop_time": np.mean(start_time_list) if start_time_list else 0.0,
            "avg_gradient_time": np.mean(gradient_time_list) if gradient_time_list else 0.0
        }

    
    


    
    def train_step_sqo_vectorized_SGD_LR_global2(self, data_loader,valdata_loader, early_stopping,  batch_size,ifdecoderonly = False,ifsave = False, save_tag=None):
        """
        SGD 版本：使用 DataLoader 进行随机梯度下降
        data_loader: PyTorch DataLoader，每个batch返回 (data, indices)
                    其中 indices 是数据在原始数据集中的全局索引
        batch_size: batch 大小
        """
        
        vae_optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=1e-4)
        latent_dim = self.latent
        n_samples = self.samplingnumber
        vae_loss_list = []
        val_vae_loss_list = []
        val_quantile_loss_list = []
        quantile_loss_list = []
        optimizer = self.optimizer
        best_loss = float('inf')
        early_stopping_counter = 0
        stop_training_due_to_nan = False
        xlsx_save_pth = self.get_save_xlsx_path(save_tag)
        os.makedirs(self.save_xlsx, exist_ok=True)
        save_pth = self.get_save_path(save_tag)
        if not os.path.exists(self.save_loss):
            os.makedirs(self.save_loss)
        for epoch in range(self.epoch):
            sum_dk = 0.0
            sum_vae_loss  = 0
            sum_q_loss = 0            
            for batch_idx, batch_data in enumerate(data_loader):
                # 如果 DataLoader 返回 (data, indices)
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, global_indices = batch_data
                    global_indices = global_indices.cpu().numpy()
                else:
                    # 如果没有索引，根据 batch_idx 计算全局索引
                    data = batch_data
                    global_indices = np.arange(batch_idx * batch_size, 
                                               min((batch_idx + 1) * batch_size, self.data_len))
                
                current_batch_size = data.shape[0]
                device = self.device
                im = data[:, -1].reshape(-1, 1).to(device)
                im_label = data[:, :-1].to(device)
                
                
                vae_optimizer.zero_grad()
                recon_im, mu, logvar = self.forward(im, im_label)
                vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                
                

                vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
                grad_vae_raw = torch.autograd.grad(
                    vae_loss,
                    vae_params,
                    create_graph=True,
                    allow_unused=True,
                )
                grad_vae = [
                    g if g is not None else torch.zeros_like(p)
                    for p, g in zip(vae_params, grad_vae_raw)
                ]
                
                if not ifdecoderonly:
                    vae_loss_val = vae_loss.item()
                else:
                    vae_loss_val = 0.0  # 如果只训练 decoder，VAE loss 不计算
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx}, VAE Loss: {vae_loss_val:.4f}")
                sum_vae_loss += vae_loss_val
                sum_q_loss += self.quantile_loss(recon_im, im).item()
                # 清理VAE阶段的显存
                del recon_im, mu, logvar, vae_loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                x_label = data[:, :-1].to(device)
                y_true = data[:, -1].unsqueeze(1).to(device)
                # 采样单个 z（每个数据点一个预测）
                z = torch.randn(current_batch_size, latent_dim).to(device)
                z.requires_grad_(True)
                
                # 随机选择维度
                dim_i = np.random.randint(0, latent_dim)

                y_pred = self.decoder(z, x_label)
                with torch.no_grad():
                    z_weight = torch.randn_like(z).to(device)  # 随机权重，形状与 z 相同
                    y_pred_weight = self.decoder(z_weight, x_label)  # 使用随机权重计算预测值
                # 初始化 q_hat（仅第一次需要采样获取初值）
                if self.q_hat[global_indices[0]][0] == 0.0:
                    with torch.no_grad():
                        x_expanded_init = data[:, :-1].unsqueeze(1).repeat(1, self.samplingnumber, 1).view(-1, self.labeldim).to(device)
                        z_init = torch.randn(current_batch_size * self.samplingnumber, latent_dim).to(device)
                        y_init = self.decoder(z_init, x_expanded_init)
                        y_reshaped_init = y_init.view(current_batch_size, n_samples)
                        q_local = torch.quantile(y_reshaped_init, self.target_quantile, dim=1, keepdim=True)
                        del x_expanded_init, z_init, y_init, y_reshaped_init

                with torch.no_grad():
                    # 构建 q_hat tensor
                    q_for_indicator = torch.zeros(current_batch_size, 1, device=device)
                    for i, global_idx in enumerate(global_indices):
                        if global_idx < len(self.q_hat) and self.q_hat[global_idx][0] != 0.0:
                            q_for_indicator[i, 0] = self.q_hat[global_idx][0]
                        else:
                            q_for_indicator[i, 0] = q_local[i, 0]
                    
                    # 直接用预测值和 quantile value 比较计算 indicator
                    indicator = (y_pred <= q_for_indicator).float()
                    
                    # 计算 Newsvendor 权重
                    diff = q_for_indicator- y_true
                    nv_weights = torch.where(diff < 0, 
                                            torch.tensor(-self.cu/(self.cu+self.co)).to(device), 
                                            torch.tensor(self.co/(self.co+self.cu)).to(device))
                    final_weights = indicator * nv_weights

                # 计算梯度
                grad_outputs = torch.ones_like(y_pred)
                grads_z = torch.autograd.grad(y_pred, z, grad_outputs=grad_outputs, 
                                            create_graph=True, retain_graph=True)[0]
                h_prime = grads_z[:, dim_i].view(-1, 1)

                grad_h_prime = torch.autograd.grad(h_prime, z, grad_outputs=torch.ones_like(h_prime),
                                                create_graph=True, retain_graph=True)[0]
                h_double_prime = grad_h_prime[:, dim_i].view(-1, 1)
                score = -z[:, dim_i].view(-1, 1)


                epsilon = 1e-6
                h_prime_inv = 1.0 / (h_prime + epsilon * torch.sign(h_prime))
                psi_2 = h_prime_inv * (score - h_double_prime * h_prime_inv)
                psi_2 = torch.clamp(psi_2, -100, 100).detach()
                h_prime_inv = torch.clamp(h_prime_inv, -100, 100).detach()


                surrogate_loss = (y_pred * psi_2 + h_prime * h_prime_inv) * final_weights


                with torch.no_grad():
                    # 警告：必须加绝对值和强力 clamp！
                    # 如果不 clamp 到 1e-3，网络大概率在第 1 个 epoch 就会因为极小的 psi_2 产生 NaN 梯度爆炸
                    safe_psi2 = torch.clamp(torch.abs(psi_2), min=1e-3)
                    
                    # 构建逐点逆密度权重 (Point-wise Inverse G2)
                    # 因为 G2_i = psi_2_i * indicator_i
                    # 当 indicator 为 1 时，我们需要除以 psi_2_i；当 indicator 为 0 时，分子本就为 0，保持 0 即可
                    pointwise_inv_g2 = (1.0 / safe_psi2) * indicator 

                # 将逆密度权重乘到单点 Loss 上，求均值
                # 此时 loss_to_backward 的求导数学期望精精确确就是： 1/B * \sum (G1_i * G3_i / G2_i)
                pointwise_ratio_loss = (surrogate_loss * pointwise_inv_g2).mean()

                decoder_optimizer.zero_grad()
                pointwise_ratio_loss.backward()

                # 提取算好的目标均值梯度
                grad_ratio_target = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                                    for p in self.decoder.parameters()]
                decoder_optimizer.zero_grad()


                alpha_k = 1 / (self.k_step ** 0.55)
                for d_val, target_val in zip(self.global_D, grad_ratio_target):
                    # 完全对应你的公式 (9') 的括号内部逻辑
                    update = target_val - d_val
                    d_val.add_(alpha_k * update)
                    # 这里的截断比 Batch-Smoothed 方法更重要，因为单点除法极不稳定！
                    d_val.clamp_(-1.0, 1.0) 


                gamma_k = 1 / (self.k_step ** 0.9)
                for param, d_val in zip(self.decoder.parameters(), self.global_D):
                    if param.grad is None:
                        param.grad = d_val.clone() * self.lambda_gradient
                    else:
                        param.grad += d_val.clone() * self.lambda_gradient

                for param, grad in zip(vae_params, grad_vae):
                    grad_to_add = grad.detach()
                    if param.grad is None:
                        param.grad = grad_to_add.clone()
                    else:
                        param.grad += grad_to_add
                
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                vae_optimizer.step()
                self.k_step += 1



                beta_k = 1 / ((self.k_step ) ** 0.6)  # q_hat的更新步长
            
                # 使用全局索引保存 Q_hat
                with torch.no_grad():
                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.q_hat):
                            continue
                        
                        # 第一次迭代：使用采样分位数 q_local 作为初始值
                        if self.q_hat[global_idx][0] == 0.0:
                            self.q_hat[global_idx] = [q_local[i].item()]
                            self.q_hat_list[global_idx].append(q_local[i].item())
                        else:
                            # 直接用预测值与 q_hat 比较计算 indicator
                            q_hat_current = self.q_hat[global_idx][0]
                            indicator_val = float(y_pred[i, 0].item() <= q_hat_current)
                            
                            # q_hat 更新公式: q_hat_{k+1} = q_hat_k + beta_k * (target_quantile - indicator)
                            q_hat_new = q_hat_current + beta_k * (self.target_quantile - indicator_val)
                            
                            # 更新q_hat
                            self.q_hat[global_idx] = [q_hat_new]
                            self.q_hat_list[global_idx].append(q_hat_new)

                self.k_step += 1
                try:
                    del per_sample_grads, y_pred_batch, g2_batch, q_values
                    del x_label, y_true
                    del im, im_label
                except:
                    pass
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            if stop_training_due_to_nan:
                print("Stopping training because non-finite loss was detected.")
                break
                    
                
            sum_dk /= len(data_loader)
            sum_vae_loss /= len(data_loader)
            sum_q_loss /= len(data_loader)
            self.loss_history['vae_loss'].append(sum_vae_loss)
            self.loss_history['quantile_loss'].append(sum_q_loss)
            vae_loss_list.append(sum_vae_loss)
            quantile_loss_list.append(sum_q_loss)
            self.D_hat_avg.append(sum_dk)
            val_loss = 0.0
            quantile_val = 0.0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    batch = val_batch.to(device)
                    im = batch[:, -1].reshape(-1, 1).to(device)
                    im_label = batch[:, :-1].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    quantile_val += self.quantile_loss(val_recon_im, im).item()
                    val_loss += val_vae_loss.item()
            val_loss /= len(valdata_loader)
            quantile_val /= len(valdata_loader)
            val_vae_loss_list.append(val_loss)
            val_quantile_loss_list.append(quantile_val)
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_quantile_loss'].append(quantile_val)
            loss_new = val_loss
            if loss_new < best_loss and not np.isnan(vae_loss_list[-1]):
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                #save model
                if not ifsave:
                    torch.save(self.state_dict(), save_pth)
                
                # 保存最佳模型（简化版本）
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
            if np.isnan(vae_loss_list[-1]):
                print("NaN detected in VAE loss, stopping training.")
                break
            # if np.abs(global_loss.item()) < 1e3:
            #     print("GLR loss too small, stopping training.")
            #     break
            
        if ifsave:
            datareturn = {
                "vae_loss": vae_loss_list,
                "quantile_loss": quantile_loss_list,
                "val_vae_loss": val_vae_loss_list,
                "val_quantile_loss": val_quantile_loss_list,
            }
            df = pd.DataFrame(datareturn)
            df.to_excel(xlsx_save_pth, index=False)


        return {
            "vae_loss": vae_loss_list,
            "quantile_loss": quantile_loss_list,
            "val_vae_loss": val_vae_loss_list,
            "val_quantile_loss": val_quantile_loss_list,
            "message": f"SGD training completed: {self.epoch} epochs, {len(data_loader)} batches per epoch"
        }

    
    
    
    
    def train_step_sqo_vectorized_SGD_pretrain(self, data_loader, valdata_loader, early_stopping, batch_size,
                                               pretrain_save_name=None, fine_tune_mode="all",
                                               custom_trainable_prefixes=None, decoder_lr=5e-4,
                                               save_tag=None):
        """
        SGD 版本（带预训练）：先从 trainconvae 保存的最优模型加载参数，冻结 encoder，只训练 decoder。
        data_loader: PyTorch DataLoader，每个batch返回 (data, indices)
                    其中 indices 是数据在原始数据集中的全局索引
        batch_size: batch 大小
        pretrain_save_name: trainconvae 保存最优模型时使用的 save_name（如 'VAEpure_exp'）
        fine_tune_mode: decoder 微调模式
            - "all": 训练全部 decoder 参数
            - "last_layer": 仅训练 decoder.linear3
            - "bias_only": 仅训练 decoder 内所有 bias
            - "custom": 仅训练名字前缀在 custom_trainable_prefixes 里的参数
        custom_trainable_prefixes: fine_tune_mode="custom" 时生效，如 ["linear3", "linear2.bias"]
        decoder_lr: decoder 优化器学习率（建议微调时使用更小学习率）
        """
        # 加载预训练最优模型（由 trainconvae 保存到 MODEL/ 目录）
        if pretrain_save_name is not None:
            pretrain_model_path = os.path.join(
                "MODEL",
                f"{pretrain_save_name}_{self.targetdim}_{self.labeldim}_{self.random_seed}_best_model.pth"
            )
            if os.path.exists(pretrain_model_path):
                print(f"加载预训练最优模型: {pretrain_model_path}")
                self.load_state_dict(torch.load(pretrain_model_path))
            else:
                print(f"警告: 预训练模型不存在: {pretrain_model_path}，将从随机初始化开始训练")

        # 冻结 encoder，只训练 decoder
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
        decoder_optimizer = torch.optim.Adam(trainable_decoder_params, lr=decoder_lr)
        latent_dim = self.latent
        n_samples = self.samplingnumber
        vae_loss_list = []
        val_vae_loss_list = []
        val_quantile_loss_list = []
        quantile_loss_list = []
        best_loss = float('inf')
        early_stopping_counter = 0
        stop_training_due_to_nan = False
        save_pth = self.get_save_path(save_tag)
        if not os.path.exists(self.save_loss):
            os.makedirs(self.save_loss)
        for epoch in range(self.epoch):
            sum_dk = 0.0
            sum_vae_loss  = 0
            sum_q_loss = 0            
            for batch_idx, batch_data in enumerate(data_loader):
                # 如果 DataLoader 返回 (data, indices)
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, global_indices = batch_data
                    global_indices = global_indices.cpu().numpy()
                else:
                    # 如果没有索引，根据 batch_idx 计算全局索引
                    data = batch_data
                    global_indices = np.arange(batch_idx * batch_size, 
                                               min((batch_idx + 1) * batch_size, self.data_len))
                
                current_batch_size = data.shape[0]
                device = self.device
                im = data[:, -1].reshape(-1, 1).to(device)
                im_label = data[:, :-1].to(device)

                # 预训练模式：跳过 VAE 反向传播，只更新 decoder
                with torch.no_grad():
                    recon_im, mu, logvar = self.forward(im, im_label)
                    vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                vae_loss_val = vae_loss.item()

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx}, VAE Loss: {vae_loss_val:.4f}")
                sum_vae_loss += vae_loss_val
                sum_q_loss += self.quantile_loss(recon_im, im).item()
                # 清理VAE阶段的显存
                del recon_im, mu, logvar, vae_loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                for loop in range(self.innerloop):
                    x_label = data[:, :-1].to(device)
                    y_true = data[:, -1].unsqueeze(1).to(device)
                    # 采样单个 z（每个数据点一个预测）
                    z = torch.randn(current_batch_size, latent_dim).to(device)
                    z.requires_grad_(True)
                    
                    # 随机选择维度
                    dim_i = np.random.randint(0, latent_dim)

                    y_pred = self.decoder(z, x_label)
                    with torch.no_grad():
                        z_weight = torch.randn_like(z).to(device)  # 随机权重，形状与 z 相同
                        y_pred_weight = self.decoder(z_weight, x_label)  # 使用随机权重计算预测值
                    # 初始化 q_hat（仅第一次需要采样获取初值）
                    if self.q_hat[global_indices[0]][0] == 0.0:
                        with torch.no_grad():
                            x_expanded_init = data[:, :-1].unsqueeze(1).repeat(1, self.samplingnumber, 1).view(-1, self.labeldim).to(device)
                            z_init = torch.randn(current_batch_size * self.samplingnumber, latent_dim).to(device)
                            y_init = self.decoder(z_init, x_expanded_init)
                            y_reshaped_init = y_init.view(current_batch_size, n_samples)
                            q_local = torch.quantile(y_reshaped_init, self.target_quantile, dim=1, keepdim=True)
                            del x_expanded_init, z_init, y_init, y_reshaped_init

                    with torch.no_grad():
                        # 构建 q_hat tensor
                        q_for_indicator = torch.zeros(current_batch_size, 1, device=device)
                        for i, global_idx in enumerate(global_indices):
                            if global_idx < len(self.q_hat) and self.q_hat[global_idx][0] != 0.0:
                                q_for_indicator[i, 0] = self.q_hat[global_idx][0]
                            else:
                                q_for_indicator[i, 0] = q_local[i, 0]
                        
                        # 直接用预测值和 quantile value 比较计算 indicator
                        indicator = (y_pred <= q_for_indicator).float()
                        
                        # 计算 Newsvendor 权重
                        diff = q_for_indicator- y_true
                        nv_weights = torch.where(diff < 0, 
                                                torch.tensor(-self.cu/(self.cu+self.co)).to(device), 
                                                torch.tensor(self.co/(self.co+self.cu)).to(device))
                        final_weights = indicator * nv_weights
                   
                    # 计算梯度
                    grad_outputs = torch.ones_like(y_pred)
                    grads_z = torch.autograd.grad(y_pred, z, grad_outputs=grad_outputs, 
                                                create_graph=True, retain_graph=True)[0]
                    h_prime = grads_z[:, dim_i].view(-1, 1)

                    grad_h_prime = torch.autograd.grad(h_prime, z, grad_outputs=torch.ones_like(h_prime),
                                                    create_graph=True, retain_graph=True)[0]
                    h_double_prime = grad_h_prime[:, dim_i].view(-1, 1)
                    score = -z[:, dim_i].view(-1, 1)

                    # 计算 Psi
                    epsilon = 1e-6
                    h_prime_inv = 1.0 / (h_prime + epsilon * torch.sign(h_prime))
                    psi_2 = h_prime_inv * (score - h_double_prime * h_prime_inv)
                    psi_2 = torch.clamp(psi_2, -100, 100).detach()
                    h_prime_inv = torch.clamp(h_prime_inv, -100, 100).detach()

                    # 计算 Surrogate Loss（每个点一个预测值，无需 reshape）
                    surrogate_loss = (y_pred * psi_2 + h_prime * h_prime_inv) * final_weights
                    global_loss = surrogate_loss.mean()

                    if not torch.isfinite(global_loss):
                        print(f"Non-finite GLR loss detected at epoch {epoch+1}, batch {batch_idx}: {global_loss.item()}")
                        stop_training_due_to_nan = True
                        break

                    decoder_optimizer.zero_grad()
                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch+1}, Batch {batch_idx}, GLR Loss: {global_loss.item():.4f}")
                    global_loss.backward()
                    
                    # 提取 G1 梯度
                    g1_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                            for p in self.decoder.parameters()]
                    decoder_optimizer.zero_grad()

                    # 计算 G2
                    with torch.no_grad():
                        g2_vals = psi_2*indicator 
                        g2_per_point = g2_vals.view(current_batch_size)
                        g2_per_point = torch.clamp(g2_per_point, min=1e-4, max=10.0)
                        global_g2 = g2_per_point.mean()

                    # 更新参数
                    k = self.k_step
                    gamma_k = 1 / (k ** 0.55)
                    beta_k = 1 / ((k ) ** 0.6)  # q_hat的更新步长
                
                    # 使用全局索引保存 Q_hat
                    with torch.no_grad():
                        for i, global_idx in enumerate(global_indices):
                            if global_idx >= len(self.q_hat):
                                continue
                            
                            # 第一次迭代：使用采样分位数 q_local 作为初始值
                            if self.q_hat[global_idx][0] == 0.0:
                                self.q_hat[global_idx] = [q_local[i].item()]
                                self.q_hat_list[global_idx].append(q_local[i].item())
                            else:
                                # 直接用预测值与 q_hat 比较计算 indicator
                                q_hat_current = self.q_hat[global_idx][0]
                                indicator_val = float(y_pred[i, 0].item() <= q_hat_current)
                                
                                # q_hat 更新公式: q_hat_{k+1} = q_hat_k + beta_k * (target_quantile - indicator)
                                q_hat_new = q_hat_current + beta_k * (self.target_quantile - indicator_val)
                                
                                # 更新q_hat
                                self.q_hat[global_idx] = [q_hat_new]
                                self.q_hat_list[global_idx].append(q_hat_new)

                    # 使用全局索引更新 D_hat
                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.D_hat):
                            continue
                        
                        g2_i = g2_per_point[i].item()
                        
                        new_D_i = []
                        for d_val, g1_val in zip(self.D_hat[global_idx], g1_grads):
                            d_val_device = d_val.to(g1_val.device) if d_val.device != g1_val.device else d_val
                            update = g1_val - g2_i * d_val_device
                            d_new = d_val_device + gamma_k * update
                            d_new = torch.clamp(d_new, -1.0, 1.0).detach()
                            new_D_i.append(d_new)
                        self.D_hat[global_idx] = new_D_i

                # 计算当前 batch 的平均 D
                avg_D = [torch.zeros_like(p) for p in self.decoder.parameters()]
                valid_count = 0
                for i, global_idx in enumerate(global_indices):
                    if global_idx < len(self.D_hat):
                        for j, d_val in enumerate(self.D_hat[global_idx]):
                            avg_D[j] = avg_D[j] + d_val.to(avg_D[j].device)
                        valid_count += 1
                
                if valid_count > 0:
                    for j in range(len(avg_D)):
                        avg_D[j] = avg_D[j] / valid_count
                avg_D = [d_val.detach() for d_val in avg_D]
                
                # 计算每个数据点的 D_hat 范数的平均值（每个batch一个标量）
                D_norms_per_point = []
                for i, global_idx in enumerate(global_indices):
                    if global_idx < len(self.D_hat):
                        # 计算该数据点所有参数的范数的平均值
                        point_d_norms = [torch.norm(d_val).item() for d_val in self.D_hat[global_idx]]
                        point_d_norm_mean = np.mean(point_d_norms) if point_d_norms else 0.0
                        D_norms_per_point.append(point_d_norm_mean)
                
                # 对batch中所有数据点的D范数求平均，得到该batch的一个标量
                batch_D_norm_mean = np.mean(D_norms_per_point) if D_norms_per_point else 0.0
                sum_dk += batch_D_norm_mean
                
                for param, d_val in zip(self.decoder.parameters(), avg_D):
                    if param.grad is None:
                        param.grad = d_val.clone()*self.lambda_gradient
                    else:
                        param.grad.copy_(d_val)
                decoder_optimizer.step()
                self.k_step += 1
                
                
                try:
                    del z, y_pred, surrogate_loss
                    del global_loss, g1_grads, g2_vals, g2_per_point, global_g2
                    del h_prime, h_double_prime, score, psi_2, h_prime_inv, final_weights
                    del q_for_indicator, indicator
                    del diff, nv_weights, grad_outputs, grads_z, grad_h_prime, x_label, y_true
                    del avg_D, im, im_label
                except:
                    pass
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            if stop_training_due_to_nan:
                print("Stopping training because non-finite loss was detected.")
                break
                    
                
            sum_dk /= len(data_loader)
            sum_vae_loss /= len(data_loader)
            sum_q_loss /= len(data_loader)
            self.loss_history['vae_loss'].append(sum_vae_loss)
            self.loss_history['quantile_loss'].append(sum_q_loss)
            vae_loss_list.append(sum_vae_loss)
            quantile_loss_list.append(sum_q_loss)
            self.D_hat_avg.append(sum_dk)
            val_loss = 0.0
            quantile_val = 0.0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    batch = val_batch.to(device)
                    im = batch[:, -1].reshape(-1, 1).to(device)
                    im_label = batch[:, :-1].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    quantile_val += self.quantile_loss(val_recon_im, im).item()
                    val_loss += val_vae_loss.item()
            val_loss /= len(valdata_loader)
            quantile_val /= len(valdata_loader)
            val_vae_loss_list.append(val_loss)
            val_quantile_loss_list.append(quantile_val)
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_quantile_loss'].append(quantile_val)
            loss_new = val_loss
            if loss_new < best_loss and not np.isnan(vae_loss_list[-1]):
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                #save model
                torch.save(self.state_dict(), save_pth)
                
                # 保存最佳模型（简化版本）
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
            if np.isnan(vae_loss_list[-1]):
                print("NaN detected in VAE loss, stopping training.")
                break
                    
        return {
            "vae_loss": vae_loss_list,
            "quantile_loss": quantile_loss_list,
            "val_vae_loss": val_vae_loss_list,
            "val_quantile_loss": val_quantile_loss_list,
            "message": f"SGD training completed: {self.epoch} epochs, {len(data_loader)} batches per epoch"
        }




    def train_step_sqo_vectorized_SGD_pretrain_global(self, data_loader, valdata_loader, early_stopping, batch_size,
                                               pretrain_save_name=None, fine_tune_mode="all",
                                               custom_trainable_prefixes=None, decoder_lr=5e-4,
                                               save_tag=None):
        """
        SGD 版本（带预训练）：先从 trainconvae 保存的最优模型加载参数，冻结 encoder，只训练 decoder。
        data_loader: PyTorch DataLoader，每个batch返回 (data, indices)
                    其中 indices 是数据在原始数据集中的全局索引
        batch_size: batch 大小
        pretrain_save_name: trainconvae 保存最优模型时使用的 save_name（如 'VAEpure_exp'）
        fine_tune_mode: decoder 微调模式
            - "all": 训练全部 decoder 参数
            - "last_layer": 仅训练 decoder.linear3
            - "bias_only": 仅训练 decoder 内所有 bias
            - "custom": 仅训练名字前缀在 custom_trainable_prefixes 里的参数
        custom_trainable_prefixes: fine_tune_mode="custom" 时生效，如 ["linear3", "linear2.bias"]
        decoder_lr: decoder 优化器学习率（建议微调时使用更小学习率）
        """
        # 加载预训练最优模型（由 trainconvae 保存到 MODEL/ 目录）
        if pretrain_save_name is not None:
            pretrain_model_path = os.path.join(
                "MODEL",
                f"{pretrain_save_name}_{self.targetdim}_{self.labeldim}_{self.random_seed}_best_model.pth"
            )
            if os.path.exists(pretrain_model_path):
                print(f"加载预训练最优模型: {pretrain_model_path}")
                self.load_state_dict(torch.load(pretrain_model_path))
            else:
                print(f"警告: 预训练模型不存在: {pretrain_model_path}，将从随机初始化开始训练")

        # 冻结 encoder，只训练 decoder
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
        decoder_optimizer = torch.optim.Adam(trainable_decoder_params, lr=decoder_lr)
        latent_dim = self.latent
        n_samples = self.samplingnumber
        vae_loss_list = []
        val_vae_loss_list = []
        val_quantile_loss_list = []
        quantile_loss_list = []
        best_loss = float('inf')
        early_stopping_counter = 0
        stop_training_due_to_nan = False
        save_pth = self.get_save_path(save_tag)
        if not os.path.exists(self.save_loss):
            os.makedirs(self.save_loss)
        for epoch in range(self.epoch):
            sum_dk = 0.0
            sum_vae_loss  = 0
            sum_q_loss = 0            
            for batch_idx, batch_data in enumerate(data_loader):
                # 如果 DataLoader 返回 (data, indices)
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    data, global_indices = batch_data
                    global_indices = global_indices.cpu().numpy()
                else:
                    # 如果没有索引，根据 batch_idx 计算全局索引
                    data = batch_data
                    global_indices = np.arange(batch_idx * batch_size, 
                                               min((batch_idx + 1) * batch_size, self.data_len))
                
                current_batch_size = data.shape[0]
                device = self.device
                im = data[:, -1].reshape(-1, 1).to(device)
                im_label = data[:, :-1].to(device)

                # 预训练模式：跳过 VAE 反向传播，只更新 decoder
                with torch.no_grad():
                    recon_im, mu, logvar = self.forward(im, im_label)
                    vae_loss = loss_function(recon_im, im, mu, logvar) / im.shape[0]
                vae_loss_val = vae_loss.item()

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx}, VAE Loss: {vae_loss_val:.4f}")
                sum_vae_loss += vae_loss_val
                sum_q_loss += self.quantile_loss(recon_im, im).item()
                # 清理VAE阶段的显存
                del recon_im, mu, logvar, vae_loss
                x_label = data[:, :-1].to(device)
                y_true = data[:, -1].unsqueeze(1).to(device)
                # 采样单个 z（每个数据点一个预测）
                z = torch.randn(current_batch_size, latent_dim).to(device)
                z.requires_grad_(True)
                
                # 随机选择维度
                dim_i = np.random.randint(0, latent_dim)

                y_pred = self.decoder(z, x_label)
                with torch.no_grad():
                    z_weight = torch.randn_like(z).to(device)  # 随机权重，形状与 z 相同
                    y_pred_weight = self.decoder(z_weight, x_label)  # 使用随机权重计算预测值
                # 初始化 q_hat（仅第一次需要采样获取初值）
                if self.q_hat[global_indices[0]][0] == 0.0:
                    with torch.no_grad():
                        x_expanded_init = data[:, :-1].unsqueeze(1).repeat(1, self.samplingnumber, 1).view(-1, self.labeldim).to(device)
                        z_init = torch.randn(current_batch_size * self.samplingnumber, latent_dim).to(device)
                        y_init = self.decoder(z_init, x_expanded_init)
                        y_reshaped_init = y_init.view(current_batch_size, n_samples)
                        q_local = torch.quantile(y_reshaped_init, self.target_quantile, dim=1, keepdim=True)
                        del x_expanded_init, z_init, y_init, y_reshaped_init

                with torch.no_grad():
                    # 构建 q_hat tensor
                    q_for_indicator = torch.zeros(current_batch_size, 1, device=device)
                    for i, global_idx in enumerate(global_indices):
                        if global_idx < len(self.q_hat) and self.q_hat[global_idx][0] != 0.0:
                            q_for_indicator[i, 0] = self.q_hat[global_idx][0]
                        else:
                            q_for_indicator[i, 0] = q_local[i, 0]
                    
                    # 直接用预测值和 quantile value 比较计算 indicator
                    indicator = (y_pred <= q_for_indicator).float()
                    
                    # 计算 Newsvendor 权重
                    diff = q_for_indicator- y_true
                    nv_weights = torch.where(diff < 0, 
                                            torch.tensor(-self.cu/(self.cu+self.co)).to(device), 
                                            torch.tensor(self.co/(self.co+self.cu)).to(device))
                    final_weights = indicator * nv_weights

                # 计算梯度
                grad_outputs = torch.ones_like(y_pred)
                grads_z = torch.autograd.grad(y_pred, z, grad_outputs=grad_outputs, 
                                            create_graph=True, retain_graph=True)[0]
                h_prime = grads_z[:, dim_i].view(-1, 1)

                grad_h_prime = torch.autograd.grad(h_prime, z, grad_outputs=torch.ones_like(h_prime),
                                                create_graph=True, retain_graph=True)[0]
                h_double_prime = grad_h_prime[:, dim_i].view(-1, 1)
                score = -z[:, dim_i].view(-1, 1)

                # 计算 Psi
                epsilon = 1e-6
                h_prime_inv = 1.0 / (h_prime + epsilon * torch.sign(h_prime))
                psi_2 = h_prime_inv * (score - h_double_prime * h_prime_inv)
                psi_2 = torch.clamp(psi_2, -100, 100).detach()
                h_prime_inv = torch.clamp(h_prime_inv, -100, 100).detach()

                # 计算 Surrogate Loss（每个点一个预测值，无需 reshape）
                surrogate_loss = (y_pred * psi_2 + h_prime * h_prime_inv) * final_weights

                global_loss = surrogate_loss.mean()
                decoder_optimizer.zero_grad()
                global_loss.backward()

                # 提取 Batch 平均梯度 \bar{G}_1
                bar_G1 = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                        for p in self.decoder.parameters()]
                decoder_optimizer.zero_grad()

                # 2. 算全局平均密度 \bar{G}_2 (标量)
                with torch.no_grad():
                    g2_vals = psi_2 * indicator
                    bar_G2 = g2_vals.mean().item()  # 只有一个数字！极其省显存！
                    bar_G2 = max(bar_G2, 1e-4)      # 简单兜底

                # 3. 更新唯一的全局 D_k
                # 假设 self.global_D 已经初始化为全 0 向量
                alpha_k = 1 / (self.k_step ** 0.55)
                for d_val, g1_val in zip(self.global_D, bar_G1):
                    update = g1_val - bar_G2 * d_val
                    d_val.add_(alpha_k * update)    # In-place 更新全局 D
                    d_val.clamp_(-1.0, 1.0)         # 防御性截断

                # 4. 外层慢尺度更新网络参数
                gamma_k = 1 / (self.k_step ** 0.9)
                for param, d_val in zip(self.decoder.parameters(), self.global_D):
                    if param.grad is None:
                        param.grad = d_val.clone() * self.lambda_gradient
                    else:
                        param.grad += d_val.clone() * self.lambda_gradient


                
                
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                decoder_optimizer.step()
                self.k_step += 1



                beta_k = 1 / ((self.k_step ) ** 0.6)  # q_hat的更新步长
            
                # 使用全局索引保存 Q_hat
                with torch.no_grad():
                    for i, global_idx in enumerate(global_indices):
                        if global_idx >= len(self.q_hat):
                            continue
                        
                        # 第一次迭代：使用采样分位数 q_local 作为初始值
                        if self.q_hat[global_idx][0] == 0.0:
                            self.q_hat[global_idx] = [q_local[i].item()]
                            self.q_hat_list[global_idx].append(q_local[i].item())
                        else:
                            # 直接用预测值与 q_hat 比较计算 indicator
                            q_hat_current = self.q_hat[global_idx][0]
                            indicator_val = float(y_pred[i, 0].item() <= q_hat_current)
                            
                            # q_hat 更新公式: q_hat_{k+1} = q_hat_k + beta_k * (target_quantile - indicator)
                            q_hat_new = q_hat_current + beta_k * (self.target_quantile - indicator_val)
                            
                            # 更新q_hat
                            self.q_hat[global_idx] = [q_hat_new]
                            self.q_hat_list[global_idx].append(q_hat_new)

                self.k_step += 1
                
                
                try:
                    del z, y_pred, surrogate_loss
                    del global_loss,  g2_vals
                    del h_prime, h_double_prime, score, psi_2, h_prime_inv, final_weights
                    del q_for_indicator, indicator
                    del diff, nv_weights, grad_outputs, grads_z, grad_h_prime, x_label, y_true
                    del im, im_label
                except:
                    pass
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            if stop_training_due_to_nan:
                print("Stopping training because non-finite loss was detected.")
                break
                    
                
            sum_dk /= len(data_loader)
            sum_vae_loss /= len(data_loader)
            sum_q_loss /= len(data_loader)
            self.loss_history['vae_loss'].append(sum_vae_loss)
            self.loss_history['quantile_loss'].append(sum_q_loss)
            vae_loss_list.append(sum_vae_loss)
            quantile_loss_list.append(sum_q_loss)
            self.D_hat_avg.append(sum_dk)
            val_loss = 0.0
            quantile_val = 0.0
            with torch.no_grad():
                for val_batch in valdata_loader:
                    # 处理数据加载器返回的数据格式
                    if isinstance(val_batch, (list, tuple)):
                        val_batch = val_batch[0]  # 如果是tuple或list，取第一个元素
                    
                    device = next(self.parameters()).device
                    batch = val_batch.to(device)
                    im = batch[:, -1].reshape(-1, 1).to(device)
                    im_label = batch[:, :-1].to(device)
                    
                    val_recon_im, val_mu, val_logvar = self.forward(im, im_label)
                    val_vae_loss = loss_function(val_recon_im, im, val_mu, val_logvar) / im.shape[0]
                    quantile_val += self.quantile_loss(val_recon_im, im).item()
                    val_loss += val_vae_loss.item()
            val_loss /= len(valdata_loader)
            quantile_val /= len(valdata_loader)
            val_vae_loss_list.append(val_loss)
            val_quantile_loss_list.append(quantile_val)
            self.loss_history['val_loss'].append(val_loss)
            self.loss_history['val_quantile_loss'].append(quantile_val)
            loss_new = val_loss
            if loss_new < best_loss and not np.isnan(vae_loss_list[-1]):
                best_loss = loss_new
                early_stopping_counter = 0
                self.loss_history['best_loss_epoch'] = epoch
                #save model
                torch.save(self.state_dict(), save_pth)
                
                # 保存最佳模型（简化版本）
                print('-' * 10)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
            if np.isnan(vae_loss_list[-1]):
                print("NaN detected in VAE loss, stopping training.")
                break
                    
        return {
            "vae_loss": vae_loss_list,
            "quantile_loss": quantile_loss_list,
            "val_vae_loss": val_vae_loss_list,
            "val_quantile_loss": val_quantile_loss_list,
            "message": f"SGD training completed: {self.epoch} epochs, {len(data_loader)} batches per epoch"
        }


