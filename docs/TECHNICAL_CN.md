# FG-Diff 技术文档

本文档详细介绍 FG-Diff（Frequency-Guided Diffusion Model with Perturbation Training）的技术细节，包括训练流程、网络架构和推理/评估流程。

## 目录

- [1. 概述](#1-概述)
- [2. 网络架构](#2-网络架构)
- [3. 训练流程](#3-训练流程)
- [4. 推理与评估流程](#4-推理与评估流程)
- [5. 配置参数说明](#5-配置参数说明)

---

## 1. 概述

FG-Diff 是一个基于扩散模型的骨架视频异常检测方法。主要创新点包括：

1. **条件扩散模型**：利用离散余弦变换（DCT）提取运动特征作为条件信息
2. **扰动训练策略**：通过对抗扰动训练增强模型鲁棒性
3. **信息融合**：在去噪过程中融合生成运动和观测运动的信息

### 核心思想

- **训练阶段**：学习正常运动模式的分布，同时通过扰动生成器产生对抗样本来增强模型鲁棒性
- **推理阶段**：对于输入的骨架序列，通过扩散模型重建正常运动；异常运动由于偏离正常分布，重建误差较大

---

## 2. 网络架构

### 2.1 整体架构图

```
                     输入骨架序列 x (B, C, T, V)
                                 |
                 +---------------+---------------+
                 |                               |
                 v                               v
         +---------------+               +---------------+
         |  条件编码器   |               |  扰动生成器   |
         |  (DCT/STSE)   |               |   (STSAE)     |
         +-------+-------+               +-------+-------+
                 |                               |
                 v                               v
         +---------------+               +---------------+
         |   条件嵌入    |               |   对抗扰动    |
         | (latent_dim)  |               |  (B,C,T,V)    |
         +-------+-------+               +-------+-------+
                 |                               |
                 +---------------+---------------+
                                 |
                                 v
                 +-------------------------------+
                 |       去噪 UNet 网络          |
                 |        (STSAE_Unet)           |
                 |                               |
                 |  Input: x_t + time_emb + cond |
                 |               |               |
                 |               v               |
                 |  +------------------------+   |
                 |  |   Encoder (下采样)     |   |
                 |  | [16,32,64,128,128,128] |   |
                 |  +------------------------+   |
                 |               |               |
                 |               v               |
                 |  +------------------------+   |
                 |  |   Decoder (上采样)     |   |
                 |  |   [128, 64, 32, 2]     |   |
                 |  +------------------------+   |
                 |               |               |
                 |               v               |
                 |  Output: 预测噪声 (B,C,T,V)   |
                 +-------------------------------+
                                 |
                                 v
                          重建骨架序列
```

### 2.2 去噪 UNet 网络结构

```
输入: x_t (B, C, T, V)
        |
        v
+------------------+
|  Time Embedding  |----+
|  (embedding_dim) |    |
+------------------+    |
                        |
========================|=================== Encoder ====================
                        |
+-----------------------------------------------+
| STSGCNBlock (C -> 16) + TimeEmb + CondEmb     |-----> skip_1
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| STSGCNBlock (16 -> 32) + TimeEmb              |-----> skip_2
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| STSGCNBlock (32 -> 64) + TimeEmb              |-----> skip_3
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| STSGCNBlock (64 -> 128) + TimeEmb             |-----> skip_4
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| STSGCNBlock (128 -> 128) + TimeEmb            |
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| STSGCNBlock (128 -> 128) + TimeEmb            |
+-----------------------------------------------+
        |
========================|=================== Decoder ====================
        |
        v
+-----------------------------------------------+
| STSGCNBlock (128 + skip_4 -> 128) + TimeEmb   |<----- skip_4
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| STSGCNBlock (128 + skip_3 -> 64) + TimeEmb    |<----- skip_3
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| STSGCNBlock (64 + skip_2 -> 32) + TimeEmb     |<----- skip_2
+-----------------------------------------------+
        |
        v
+-----------------------------------------------+
| STSGCNBlock (32 + skip_1 -> C_out) + TimeEmb  |<----- skip_1
+-----------------------------------------------+
        |
        v
输出: 预测噪声 (B, C, T, V)
```

### 2.3 主要组件

#### 2.3.1 去噪网络 (STSAE_Unet)

基于 UNet 的时空图卷积网络，用于预测噪声：

- **输入**：加噪的骨架序列 `(B, C, T, V)` + 时间步嵌入 + 条件嵌入
- **输出**：预测的噪声 `(B, C, T, V)`

```python
model = STSAE_Unet(
    c_in=num_coords,           # 坐标维度 (2 for 2D)
    embedding_dim=128,         # 时间嵌入维度
    unet_down_channels=[16, 32, 64, 128, 128, 128, 128],  # 下采样通道
    unet_up_channels=[128, 64, 32, 2],                     # 上采样通道
    n_frames=seg_len,          # 序列长度
    n_joints=17,               # 关节点数量
    dropout=0.4
)
```

#### 2.3.2 条件编码器

**方式一：DCT 特征提取（推荐）**

```python
def extract_dct_feature(self, tensor):
    # 1. 对输入序列进行 DCT 变换
    tensor_dct = self.transform_dct(tensor)  # (B, T, C*J)
    
    # 2. 展平并选取 top-k 分量
    tensor_flat = tensor_dct.view(tensor_dct.shape[0], -1)
    condition_data, index = torch.topk(tensor_flat, self.cond_latent_dim, dim=1)
    
    return condition_data  # (B, latent_dim)
```

**方式二：可训练编码器 (STSE)**

```python
condition_encoder = STSE(
    c_in=num_coords,
    h_dim=32,
    latent_dim=128,
    n_frames=seg_len // 2,
    n_joints=17,
    layer_channels=[16]
)
```

#### 2.3.3 扰动生成器 (STSAE)

用于生成对抗扰动，增强模型鲁棒性：

```python
perturbe_generator = STSAE(
    c_in=num_coords,
    h_dim=32,
    latent_dim=128,
    n_frames=seg_len,
    n_joints=17,
    layer_channels=[16]
)
```

### 2.4 数据格式

| 维度 | 含义 | 典型值 |
|------|------|--------|
| B | Batch Size | 4096 |
| C | 坐标维度 | 2 (x, y) |
| T | 时间帧数 | 24 (Avenue), 6 (UBnormal) |
| V | 关节点数 | 17 |

---

## 3. 训练流程

### 3.1 训练流程图

```
+-----------------------------------------------------------+
                         训练循环                            
+-----------------------------------------------------------+
                                                             
    输入: 正常骨架序列 x                                     
               ↓                                             
    +--------------------+                                   
      1. 条件信息提取                                        
       - 取前半部分帧                                        
       - DCT 特征提取                                        
    +--------------------+                                   
               ↓                                             
    +----------------------------------+                     
      2. 扩散前向过程                                        
        - 采样时间步 t                                       
        - 添加噪声                                           
        x_t = sqrt(a)*x + sqrt(1-a)*n                        
    +----------------------------------+                     
               ↓                                             
    +------------------+    +------------------+             
      3. 扰动生成       --->  4. 对抗样本生成               
        (perturbe_             x' = x_t -                    
         generator)            eps*sign(grad)                
    +------------------+    +------------------+             
                                    ↓                        
                            +------------------+             
                              5. 噪声预测                    
                                pred = UNet(                 
                                  x', t, cond)               
                            +------------------+             
                                    ↓                        
                            +------------------+             
                              6. 损失计算                    
                                L = MSE(pred,                
                                      noise)                 
                            +------------------+             
                                    ↓                        
                            +------------------+             
                              7. 双优化器更新                
                                opt1: min L                  
                                opt2: max L                  
                            +------------------+             
                                                             
+-----------------------------------------------------------+
```

### 3.2 核心代码解析

#### 3.2.1 损失计算

```python
def compute_loss(self, tensor_data, optimizer_idx):
    # 1. 提取条件信息（前半部分帧）
    condition_data, _, _ = self._mask(tensor_data, self.masked_rate)
    
    # 2. 条件编码
    if self.dct:
        condition_embedding = self.extract_dct_feature(condition_data)
    else:
        condition_embedding, _ = self._encode_condition(condition_data)
    
    # 3. 扩散前向过程：添加噪声
    t = self.noise_scheduler.sample_timesteps(tensor_data.shape[0])
    x_t, noise = self.noise_scheduler.noise_graph(tensor_data, t)
    
    # 4. 扰动训练
    if self.perturb:
        _, grad = self.perturbe_generator(x_t, t)
        x_perturbed_t = self.adversarial_example(x_t, self.weight_perturb, grad)
        predicted_noise, _ = self.model(x_perturbed_t, t=t, 
                                        condition_data=condition_embedding)
    else:
        predicted_noise, _ = self.model(x_t, t=t, 
                                        condition_data=condition_embedding)
    
    # 5. 计算损失
    loss = torch.mean(self.loss_fn(predicted_noise, noise))
    
    # 6. 返回损失（对抗训练时，扰动生成器最大化损失）
    return loss if optimizer_idx == 0 else -loss
```

#### 3.2.2 对抗样本生成

```python
def adversarial_example(self, x, epsilon, perturbation):
    """
    生成对抗样本
    
    Args:
        x: 输入数据
        epsilon: 扰动强度
        perturbation: 扰动方向（梯度）
    
    Returns:
        对抗样本 x' = x - ε * sign(perturbation)
    """
    return -epsilon * perturbation.sign() + x
```

### 3.3 训练策略

#### 双优化器策略

```python
def training_step(self, batch, batch_idx):
    optimizers = self.optimizers()
    tensor_data, _ = self._unpack_data(batch)
    
    if self.current_epoch <= self.n_epochs_tune:
        # 阶段1：同时训练去噪网络和扰动生成器
        
        # 更新去噪网络（最小化损失）
        loss_1 = self.compute_loss(tensor_data, optimizer_idx=0)
        optimizers[0].zero_grad()
        self.manual_backward(loss_1)
        optimizers[0].step()
        
        # 更新扰动生成器（最大化损失）
        loss_2 = self.compute_loss(tensor_data, optimizer_idx=1)
        optimizers[1].zero_grad()
        self.manual_backward(loss_2)
        optimizers[1].step()
    else:
        # 阶段2：仅训练去噪网络
        loss_1 = self.compute_loss(tensor_data, optimizer_idx=0)
        optimizers[0].zero_grad()
        self.manual_backward(loss_1)
        optimizers[0].step()
```

---

## 4. 推理与评估流程

### 4.1 推理流程图

```
+-----------------------------------------------------------+
                         推理流程                            
+-----------------------------------------------------------+
                                                             
    输入: 测试骨架序列 x                                     
               ↓                                             
    +------------------+                                     
      1. 条件信息提取                                        
        cond = DCT(x)                                        
    +------------------+                                     
               ↓                                             
    +------------------+                                     
      2. 初始化纯噪声                                        
        x_T ~ N(0, I)                                        
    +------------------+                                     
               ↓                                             
    +-----------------------------------------------+        
      3. 迭代去噪 (t = T, T-1, ..., 1)                       
        +---------------------------------------+            
          a. 扰动处理                                        
             x'_t = x_t - eps*sign(perturb)                  
        +---------------------------------------+            
          b. 预测噪声                                        
             pred = UNet(x'_t, t, cond)                      
        +---------------------------------------+            
          c. 去噪一步                                        
             x_{t-1} = denoise_step(x_t, pred)               
        +---------------------------------------+            
          d. 数据融合                                        
             x_{t-1} = fuse(x_{t-1}, x_hat)                  
        +---------------------------------------+            
    +-----------------------------------------------+        
               ↓                                             
    +------------------+                                     
      4. 计算重建误差                                        
        loss = MSE(                                          
          x_hat, x)                                          
    +------------------+                                     
               ↓                                             
    +------------------+                                     
      5. 异常分数计算                                        
        score = f(loss)                                      
    +------------------+                                     
                                                             
+-----------------------------------------------------------+
```

### 4.2 去噪过程

```python
def denoise(self, input_data, aggr_strategy):
    generated_xs = []
    B = input_data.shape[0]
    
    # 1. 提取条件信息
    condition_data, _, _ = self._mask(input_data, 0.5)
    condition_embedding = self.extract_dct_feature(condition_data)
    
    for i in range(self.n_generated_samples):
        # 2. 从纯噪声开始
        x = torch.randn_like(input_data, device=self.device)
        
        # 3. 迭代去噪
        for j in reversed(range(1, self.noise_steps)):
            t = torch.full((B,), j, dtype=torch.long, device=self.device)
            t_ = torch.full((B,), j-1, dtype=torch.long, device=self.device)
            
            # 获取扩散参数
            alpha = self._alpha[t][:, None, None, None]
            alpha_hat = self._alpha_hat[t][:, None, None, None]
            beta = self._beta[t][:, None, None, None]
            
            # 准备噪声
            noise = torch.randn_like(x) if j > 1 else torch.zeros_like(x)
            
            # 获取观测数据在 t-1 时刻的加噪版本
            x_t, _ = self.noise_scheduler.noise_graph(input_data, t_)
            
            # 扰动处理
            if self.perturb:
                _, grad = self.perturbe_generator(x, t)
                x_perturbed = self.adversarial_example(x, self.weight_perturb, grad)
                predicted_noise, _ = self.model(x_perturbed, t=t, 
                                                condition_data=condition_embedding)
                _, grad = self.perturbe_generator(x_t, t_)
                x_t = self.adversarial_example(x_t, self.weight_perturb, grad)
            else:
                predicted_noise, _ = self.model(x, t=t, 
                                                condition_data=condition_embedding)
            
            # 去噪一步
            x = (1 / torch.sqrt(alpha)) * \
                (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + \
                torch.sqrt(beta) * noise
            
            # 数据融合（除最后一步外）
            if j != 1:
                x = self._fuse_data(x, x_t)
        
        generated_xs.append(x)
    
    # 4. 聚合策略选择最佳结果
    selected_x, loss = self._aggregation_strategy(generated_xs, input_data, aggr_strategy)
    return selected_x, loss
```

### 4.3 异常分数计算

```python
def post_processing(self, out, gt_data, trans, meta, frames):
    """
    后处理：计算异常分数并评估 AUC
    """
    # 1. 遍历每个视频片段
    for idx in range(len(all_gts)):
        scene_idx, clip_idx = scene_clips[idx]
        gt = np.load(os.path.join(self.gt_path, all_gts[idx]))
        
        # 2. 计算每个人的重建误差
        error_per_person = []
        for fig in figs_ids:
            # 构建误差矩阵
            loss_matrix = compute_var_matrix(out_fig, frames_fig, n_frames)
            # 取每帧的最大误差
            fig_reconstruction_loss = np.nanmax(loss_matrix, axis=0)
            # 填充边界
            fig_reconstruction_loss = pad_scores(fig_reconstruction_loss, gt, 
                                                  self.anomaly_score_pad_size)
            error_per_person.append(fig_reconstruction_loss)
        
        # 3. 聚合所有人的分数
        clip_score = np.stack(error_per_person, axis=0)
        clip_score_log = np.log1p(clip_score)
        # 均值 + 范围（考虑多人场景的差异）
        clip_score = np.mean(clip_score, axis=0) + \
                     (np.amax(clip_score_log, axis=0) - np.amin(clip_score_log, axis=0))
        
        # 4. 平滑处理
        clip_score = score_process(clip_score, 
                                   self.anomaly_score_frames_shift,
                                   self.anomaly_score_filter_kernel_size)
    
    # 5. 计算 AUC
    auc = roc_auc_score(gt, pds)
    return auc
```

---

## 5. 配置参数说明

### 5.1 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `perturb` | bool | true | 是否启用扰动训练 |
| `weight_perturb` | float | 0.1 | 扰动强度 |
| `dct` | bool | true | 是否使用 DCT 提取条件特征 |
| `masked_rate_dct` | float | 0.1 | DCT 掩码率 |
| `noise_steps` | int | 20 | 扩散步数 |
| `n_epochs_tune` | int | 100 | 扰动生成器训练的 epoch 数 |

### 5.2 网络参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `embedding_dim` | int | UNet 时间嵌入维度 |
| `latent_dim` | int | 条件编码器潜在空间维度 |
| `unet_down_channels` | list | UNet 下采样通道数 |
| `unet_up_channels` | list | UNet 上采样通道数 |
| `dropout` | float | Dropout 概率 |

### 5.3 数据参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `seg_len` | int | 输入序列长度 |
| `batch_size` | int | 批次大小 |
| `num_transform` | int | 数据增强变换数量 |
| `normalization_strategy` | str | 归一化策略 ('robust'/'none') |

### 5.4 评估参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `filter_kernel_size` | int | 异常分数平滑核大小 |
| `frames_shift` | int | 帧偏移补偿 |
| `pad_size` | int | 边界填充大小 |
| `aggregation_strategy` | str | 多样本聚合策略 |

---

## 附录：文件结构

```
FG-Diff/
├── models/
│   ├── FG_DIFF.py          # 主模型定义
│   ├── stsae/
│   │   ├── stsae.py        # STSAE/STSE 编码器
│   │   └── stsae_unet.py   # UNet 网络
│   └── gcae/
│       └── stsgcn.py       # 时空图卷积层
├── utils/
│   ├── dataset.py          # 数据集加载
│   ├── diffusion_utils.py  # 扩散过程工具
│   ├── eval_utils.py       # 评估工具
│   └── get_robust_data.py  # 数据预处理
├── config/
│   ├── Avenue/             # Avenue 数据集配置
│   ├── STC/                # ShanghaiTech 数据集配置
│   └── UBnormal/           # UBnormal 数据集配置
├── train_FG-DIFF.py        # 训练脚本
└── eval_FG-DIFF.py         # 评估脚本
```
