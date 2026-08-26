import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd


class con_Backbone(nn.Module):
    def __init__(self, alpha, input_dim = 1,output_dim = 1):
        super().__init__()
        self.linear_model1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU()
        )
        self.alpha = alpha
        # Condition time t
        
        self.linear_model2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, output_dim),
        )
    def forward(self, x):   
        self.linear_model1 = self.linear_model1.to(x.device)
        self.linear_model2 = self.linear_model2.to(x.device)
        x = self.linear_model1(x)
        # alpha = torch.full((x.size(0), 1), self.alpha, device=x.device)
        # x= torch.cat((x,alpha),dim=1) 
        x = self.linear_model2(x)
        return x
    
    
    
    
class Benchmark(nn.Module):
    def __init__(self, alpha, input_size,con_size,randnumber):
        super(Benchmark, self).__init__()
        self.alpha = alpha
        self.con_size = con_size
        self.input_size = input_size
        self.randnumber = randnumber
        self.backbonne = con_Backbone(alpha,con_size,input_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.losshistory = {
            'train_loss': [],
            'quantile_loss': [],
            'epoch': [],
            'best_loss_epoch': []
        }
        self.save_pth = "bench_pth"
        self.save_loss = "lossrecord"
        os.makedirs(self.save_pth, exist_ok=True)
    def forward(self, x):
        x = self.backbonne(x)
        return x
        
    





    def loss_fn(self, target, inputx):
        output = self.forward(inputx)
        return torch.mean(torch.maximum(self.alpha * (target - output), (self.alpha - 1) * (target - output)))

    
    def train(self, num_epochs, targetdim, traindata_loader, valdata_loader, early_stopping, model_save_path=None):
        if model_save_path is None:
            model_save_path = os.path.join(self.save_pth, f'benchmark_{self.con_size}_alpha_{self.alpha}_rand_{self.randnumber}.pth')
        
        best_loss = float('inf')
        early_stopping_counter = 0
        
        print(f"开始训练Benchmark模型 (alpha={self.alpha})")
        print(f"模型将保存到: {model_save_path}")
        best_epoch = 0
        for epoch in range(num_epochs):
            # 训练阶段
            whole_loss = 0
            epoch_train_losses = []
            
            for i, batch in enumerate(traindata_loader):
                batch_size = batch.shape[0]
                if targetdim == 1:
                    batch = batch
                    y1 = batch[:, -1].reshape(-1, 1)
                    x1 = batch[:, :-1]
                else:
                    batch = batch
                    y1 = torch.Tensor(batch[:, -targetdim:])
                    x1 = batch[:, :-targetdim]
                
                loss = self.loss_fn(y1, x1)
                whole_loss += loss
                epoch_train_losses.append(loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # 计算平均训练损失
            avg_train_loss = whole_loss / len(traindata_loader)
            
            # 验证阶段
            val_loss = 0
            epoch_val_losses = []
            with torch.no_grad():
                for val_batch in valdata_loader:
                    if targetdim == 1:
                        batch = val_batch
                        y1 = batch[:, -1].reshape(-1, 1)
                        x1 = batch[:, :-1]
                    else:
                        batch = val_batch
                        y1 = torch.Tensor(batch[:, -targetdim:])
                        x1 = batch[:, :-targetdim]
                    
                    batch_val_loss = self.loss_fn(y1, x1)
                    val_loss += batch_val_loss
                    epoch_val_losses.append(batch_val_loss.item())
                
                avg_val_loss = val_loss / len(valdata_loader)
            
            # 记录损失历史
            self.losshistory['train_loss'].append(avg_train_loss.item())
            self.losshistory['quantile_loss'].append(avg_train_loss.item())
            self.losshistory['epoch'].append(epoch)
            # 打印进度
            if (epoch) % 20 == 0:
                print('epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
                    epoch, avg_train_loss.item(), avg_val_loss.item()))
            
            # 检查是否为最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                early_stopping_counter = 0
                best_epoch = epoch
                # 保存最佳模型
                torch.save(self.state_dict(), model_save_path)
                print('epoch: {}, find new best loss: Val Loss: {:.4f}'.format(epoch, best_loss.item()))
                print('-' * 10)
            else:
                early_stopping_counter += 1
            self.losshistory['best_loss_epoch'].append(best_epoch)
            # 早停检查
            if early_stopping_counter == early_stopping:
                print("Early stopping after {} epochs".format(epoch))
                break
                
        # 保存损失历史
        history_path = model_save_path.replace('.pth', '_history.npy')
        # np.save(history_path, self.losshistory)
        data = pd.DataFrame(self.losshistory)
        os.makedirs(self.save_loss, exist_ok=True)
        data.to_excel(self.save_loss+'/benchmark_loss_record_{}_{}.xlsx'.format(self.con_size, self.randnumber))
        print(f"\n训练完成!")
        print(f"最佳验证损失: {best_loss.item():.6f} (Epoch {self.losshistory['best_loss_epoch']})")
        print(f"模型保存到: {model_save_path}")
        print(f"损失历史保存到: {history_path}")
        
        return best_loss.item(), self.losshistory
    
    
    def sample(self, x):
        with torch.no_grad():
            x = torch.Tensor(x).to(self.device)
            return self.forward(x).cpu().numpy()

    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            self.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"成功加载模型: {model_path}")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

    def save_model(self, save_path=None):
        """手动保存模型"""
        if save_path is None:
            save_path = os.path.join(self.save_pth, f'benchmark_{self.con_size}_alpha_{self.alpha}_manual.pth')
        
        torch.save(self.state_dict(), save_path)
        print(f"模型已保存到: {save_path}")
        return save_path




