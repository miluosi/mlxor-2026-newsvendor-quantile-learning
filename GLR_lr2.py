# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
import statsmodels.api as sm
from model.VAE_quantile_ipa import VAE_quantile_ipa
from model.VAE_end_to_end import  VAE_end_to_end
from model.VAE_GLR_Model import VAE_GLR_Model
print("所有库导入成功！")





# 定义损失函数
def pandas_loss(alpha, beta):
    def loss(y_true, y_pred):
        # Convert pandas Series to NumPy arrays
        y_true_np = y_true.to_numpy()
        y_pred_np = y_pred.to_numpy()
        # Calculate the error
        error = y_true_np - y_pred_np
        # Calculate the custom loss using pandas operations
        loss_values = np.where(error > 0, alpha * error, beta * error)
        # Calculate the mean loss
        mean_loss = pd.Series(loss_values).mean()
        return mean_loss
    return loss



# 定义损失函数
def pandas_loss_series(alphalist, betalist):
    def loss(y_true, y_pred):
        # Convert pandas Series to NumPy arrays
        y_true_np = y_true.to_numpy()
        y_pred_np = y_pred.to_numpy()
        # Calculate the error
        error = y_true_np - y_pred_np
        # Calculate the custom loss using pandas operations
        loss_values = []
        for alpha, beta in zip(alphalist, betalist):
            loss_values.append(np.where(error > 0, alpha * error, beta * error))
        loss_values = np.mean(loss_values, axis=0)
        # Calculate the mean loss
        mean_loss = pd.Series(loss_values).mean()
        return mean_loss
    return loss


# 定义ECDF函数
from scipy import interpolate
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF

def getdataecdf(x1,a,b):
    if isinstance(x1, pd.Series):
        x1 = x1.to_numpy()
    elif not isinstance(x1, np.ndarray):
        raise ValueError("Input must be a Pandas Series or NumPy array.")

    ecdf = ECDF(x1)
    y = ecdf(x1)
    x1.sort()
    y.sort()
    target_y = a / (a + np.abs(b))

    # Find the index in the sorted y array where target_y is located
    index = np.searchsorted(y, target_y, side='right') - 1
    # Get the corresponding x value from the sorted x1 array using the index
    x_value = x1[index]
    return x_value



# 读取基础数据并设置参数
data = pd.read_csv(r'Walmart.csv')
data = data.drop(['Date'], axis=1)
data2 = data.copy()
data2 = data2.drop(['Weekly_Sales'], axis=1)
data2 = data2.drop(['Store'], axis=1)
data2 = data2.drop(['Holiday_Flag'], axis=1)
data2['Store'] = data['Store']
data2['Holiday_Flag'] = data['Holiday_Flag']
data2['Weekly_Sales'] = data['Weekly_Sales']

print(f"数据形状: {data2.shape}")

import random
random.seed(42)
random_state_array = random.sample(range(1, 101), 10)

random.seed(64)
sarray = [
   [random.randint(1, 10), random.randint(-10, -1)]
   for _ in range(10)
]

# 实验参数
batch_size = 64
input_dim = 1
generate_size = 1000
num_epochs_pretrain = 500
num_epochs_ipa = 500

print(f"随机状态数组: {random_state_array}")
print(f"损失参数数组: {sarray}")
print(f"实验参数设置完成")


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



# 数据生成函数 - 线性版本
from sklearn.datasets import make_regression

def makettoy(num_samples, column, random_state):
    """生成线性数据"""
    features, target = make_regression(n_samples=num_samples,
                                        n_features=column,
                                        n_informative=column,
                                        n_targets=1,
                                        random_state=random_state)

    # Stack features and target
    feature_target = np.column_stack((features, target))

    # Filter rows where the last column is greater than 0
    filtered_rows = feature_target[feature_target[:, -1] > 0]
    # Ensure the resulting matrix has at least num_samples rows
    while filtered_rows.shape[0] < num_samples:
        # Add more samples to meet the requirement
        additional_samples, additional_target = make_regression(n_samples=1000,
                                                n_features=column,
                                                n_informative=column,
                                                n_targets=1,
                                                random_state=random_state)
        additional_rows = np.column_stack((additional_samples, additional_target))
        additional_rows = additional_rows[additional_rows[:, -1] > 0]
        filtered_rows = np.vstack((filtered_rows, additional_rows))

    return filtered_rows[:num_samples]

import numpy as np
from sklearn.datasets import make_regression

def makettoy_multi_exp(num_samples, num_features, random_state, num_exps=5):
    """生成具有多峰复杂分布的目标变量y的数据，X服从正态分布，y由随机w^T生成"""
    samples_per_exp = num_samples // num_exps
    remaining_samples = num_samples % num_exps
    
    all_data = []
    np.random.seed(random_state)
    meanx = []
    for i in range(num_features):
        np.random.seed(random_state + i)
        meanx.append(np.random.uniform(-50, 50))
    meanx = np.array(meanx)
    peakmean = np.random.uniform(0, 250, size=num_exps)  # 用于峰的偏移
    label_list = []
    w_np = np.zeros((num_exps, num_features))
    for i in range(num_exps):
        # 为每个峰生成样本
        peak_samples = samples_per_exp + (1 if i < remaining_samples else 0)
        label_list.extend([i] * peak_samples)
        # 生成正态分布的X
        np.random.seed(random_state + i)
        X = np.random.normal(loc=meanx, scale=1, size=(peak_samples, num_features))
        
        # 随机生成权重w
        w = np.random.normal(loc=0, scale=1, size=num_features)
        w_np[i,:] = w
        # 计算目标变量y = w^T * X + peak_offset + noise
        y = X @ w + peakmean[i] + np.random.normal(0, 10, size=peak_samples)
        
        # 堆叠特征和目标
        feature_target = np.column_stack((X, y))
        
        all_data.append(feature_target)
    
    # 组合所有峰的数据
    combined = np.vstack(all_data)
    combined_labels = np.array(label_list).reshape(-1, 1)
    combined = np.hstack((combined, combined_labels))  # 添加标签列作为最后
    # 随机打乱以混合数据
    np.random.seed(random_state)
    np.random.shuffle(combined)
    return combined, w_np


# 设置随机种子和参数
random.seed(42)
random_state_array = random.sample(range(1, 101), 10)

random.seed(128)
sarray = [
   [random.randint(1, 10), random.randint(-10, -1)]
   for _ in range(10)
]

# 实验参数
batch_size = 64
input_dim = 1
generate_size = 1000
num_epochs_pretrain = 500
num_epochs_ipa = 500

print(f"随机状态数组: {random_state_array}")
print(f"损失参数数组: {sarray}")
print(f"实验参数设置完成")


import numpy as np
test_sarray=[[]for i in range(10)]
for i in range(10):
    for j in range(10):
        np.random.seed(j*10 + i)
        test_quantile = np.random.uniform(0.1, 0.9)
        test_pos = test_quantile * 10
        test_neg = -(10 - test_pos)
        test_sarray[i].append([test_pos, test_neg])
        


sarray_alphalist = [s[0] for s in sarray]
sarray_betalist = [s[1] for s in sarray]




def evaluate_glr_model(model, sca_X_test, scaler_y, y_true, single_cost, series_costs, generate_size):
    device = next(model.parameters()).device
    test_series = pd.Series(y_true)

    def predict_for_cost(alpha, beta):
        predictions = []
        for test_idx in range(sca_X_test.shape[0]):
            row = sca_X_test[test_idx, :-1]
            matrix = np.tile(row, (generate_size, 1))
            condition_tensor = torch.tensor(matrix, dtype=torch.float32, device=device)
            z_sample = torch.randn(generate_size, model.latent, device=device)
            decoded = model.decode(z_sample, condition_tensor)
            generated = scaler_y.inverse_transform(decoded.detach().cpu().numpy()).reshape(-1)
            predictions.append(getdataecdf(generated, alpha, beta))
        return predictions

    single_predictions = predict_for_cost(single_cost[0], single_cost[1])
    single_loss = pandas_loss(single_cost[0], single_cost[1])(
        test_series, pd.Series(single_predictions)
    )

    series_loss = 0.0
    for alpha, beta in series_costs:
        series_predictions = predict_for_cost(alpha, beta)
        series_loss += pandas_loss(alpha, beta)(
            test_series, pd.Series(series_predictions)
        )

    return single_loss, series_loss / len(series_costs)


num_epochs = 100
early_stopping = 20
dimlist = [4, 9, 14, 19, 24]  # 只运行维度4, 9, 14, 19, 24


for dim_idx, dim in enumerate(dimlist):
    print(f"\n=== 处理维度 {dim} (非线性) ===")
    glr_global_series_list = []
    glr_global_single_list = []
    glrlr_series_list = []
    glrlr_single_list = []

    for fold, random_state in enumerate(random_state_array):
        resultdata = pd.DataFrame()
        print(f"\n{fold + 1}.for random_state = {random_state}, dim = {dim} (非线性)")

        quantile = sarray[fold][0] / (sarray[fold][0] + np.abs(sarray[fold][1]))
        print(f"量化参数: alpha={sarray[fold][0]}, beta={sarray[fold][1]}, quantile={quantile:.4f}")

        data5, _ = makettoy_multi_exp(
            num_samples=data2.shape[0] * 2,
            num_features=dim,
            random_state=random_state,
            num_exps=5,
        )
        data5 = data5[:, :-1]
        X_train_5, X_val_5 = train_test_split(data5, test_size=0.1, random_state=42)
        X_test_5, _ = makettoy_multi_exp(int(X_train_5.shape[0] / 2), dim, random_state, num_exps=5)
        X_test_5 = X_test_5[:, :-1]

        X_train_5 = np.array(X_train_5, dtype=np.float32)
        X_val_5 = np.array(X_val_5, dtype=np.float32)
        X_test_5 = np.array(X_test_5, dtype=np.float32)

        print(f"训练数据形状: {X_train_5.shape}")
        print(f"验证数据形状: {X_val_5.shape}")
        print(f"测试数据形状: {X_test_5.shape}")

        resultdata['test'] = X_test_5[:, -1]

        scay5 = StandardScaler()
        sca = StandardScaler()

        sca_X_train_5 = sca.fit_transform(X_train_5).astype(np.float32)
        sca_X_val_5 = sca.transform(X_val_5).astype(np.float32)
        sca_X_test_5 = sca.transform(X_test_5).astype(np.float32)
        scay5.fit_transform(X_train_5[:, -1].reshape(-1, 1)).astype(np.float32)

        train_len = X_train_5.shape[0]
        sca_traindata_loader_5 = DataLoader(sca_X_train_5, batch_size=batch_size, shuffle=True)
        sca_valdata_loader_5 = DataLoader(sca_X_val_5, batch_size=batch_size, shuffle=True)

        glr_global_seed = random_state
        glr_lr_seed = random_state
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vae_glr_global = VAE_GLR_Model(
            targetdim=1,
            labeldim=dim,
            latent=5,
            data_len=train_len,
            epoch=num_epochs,
            quantiles=quantile,
            samplingnumber=100,
            target_quantile=quantile,
            cost_under=sarray[fold][0],
            cost_over=np.abs(sarray[fold][1]),
            random_seed=glr_global_seed,
        )
        vae_glr_global.to(device)
        vae_glr_global.train_step_sqo_vectorized_SGD_global(
            sca_traindata_loader_5,
            sca_valdata_loader_5,
            early_stopping,
            batch_size,
            save_tag="sgd_global",
        )

        pth_glr_global = vae_glr_global.get_save_path("sgd_global")
        vae_glr_global_eval = VAE_GLR_Model(
            targetdim=1,
            labeldim=dim,
            latent=5,
            data_len=train_len,
            epoch=num_epochs,
            quantiles=quantile,
            samplingnumber=100,
            target_quantile=quantile,
            cost_under=sarray[fold][0],
            cost_over=np.abs(sarray[fold][1]),
            random_seed=glr_global_seed,
        )
        vae_glr_global_eval.load_state_dict(torch.load(pth_glr_global, map_location=device))
        vae_glr_global_eval.to(device)

        vae_glr_lr = VAE_GLR_Model(
            targetdim=1,
            labeldim=dim,
            latent=5,
            data_len=train_len,
            epoch=num_epochs,
            quantiles=quantile,
            samplingnumber=100,
            target_quantile=quantile,
            cost_under=sarray[fold][0],
            cost_over=np.abs(sarray[fold][1]),
            random_seed=glr_lr_seed,
        )
        vae_glr_lr.to(device)
        vae_glr_lr.train_step_sqo_vectorized_SGD_LR_global(
            sca_traindata_loader_5,
            sca_valdata_loader_5,
            early_stopping,
            batch_size,
            save_tag="sgd_lr_global",
        )

        pth_glr_lr = vae_glr_lr.get_save_path("sgd_lr_global")
        vae_glr_lr_eval = VAE_GLR_Model(
            targetdim=1,
            labeldim=dim,
            latent=5,
            data_len=train_len,
            epoch=num_epochs,
            quantiles=quantile,
            samplingnumber=100,
            target_quantile=quantile,
            cost_under=sarray[fold][0],
            cost_over=np.abs(sarray[fold][1]),
            random_seed=glr_lr_seed,
        )
        vae_glr_lr_eval.load_state_dict(torch.load(pth_glr_lr, map_location=device))
        vae_glr_lr_eval.to(device)

        sarray_test = test_sarray[fold]
        with torch.no_grad():
            glr_global_single_loss, glr_global_series_loss = evaluate_glr_model(
                vae_glr_global_eval,
                sca_X_test_5,
                scay5,
                resultdata['test'],
                sarray[fold],
                sarray_test,
                generate_size,
            )
            glrlr_single_loss, glrlr_series_loss = evaluate_glr_model(
                vae_glr_lr_eval,
                sca_X_test_5,
                scay5,
                resultdata['test'],
                sarray[fold],
                sarray_test,
                generate_size,
            )

        glr_global_single_list.append(glr_global_single_loss)
        glr_global_series_list.append(glr_global_series_loss)
        glrlr_single_list.append(glrlr_single_loss)
        glrlr_series_list.append(glrlr_series_loss)

    os.makedirs('npyresult_peak', exist_ok=True)
    np.save(os.path.join('npyresult_peak', f'cvaeglr_peak_dim{dim}.npy'), np.array(glr_global_single_list))
    np.save(os.path.join('npyresult_peak', f'cvaeglr_peak_dim{dim}_series.npy'), np.array(glr_global_series_list))
    np.save(os.path.join('npyresult_peak', f'glrlr_peak_dim{dim}.npy'), np.array(glrlr_single_list))
    np.save(os.path.join('npyresult_peak', f'glrlr_peak_dim{dim}_series.npy'), np.array(glrlr_series_list))
    
    
    
   
    