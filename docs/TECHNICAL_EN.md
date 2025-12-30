# FG-Diff Technical Documentation

This document provides detailed technical information about FG-Diff (Frequency-Guided Diffusion Model with Perturbation Training), including training pipeline, network architecture, and inference/evaluation procedures.

## Table of Contents

- [1. Overview](#1-overview)
- [2. Network Architecture](#2-network-architecture)
- [3. Training Pipeline](#3-training-pipeline)
- [4. Inference and Evaluation](#4-inference-and-evaluation)
- [5. Configuration Parameters](#5-configuration-parameters)

---

## 1. Overview

FG-Diff is a diffusion model-based method for skeleton video anomaly detection. Key innovations include:

1. **Conditional Diffusion Model**: Extracts motion features using Discrete Cosine Transform (DCT) as conditioning information
2. **Perturbation Training Strategy**: Enhances model robustness through adversarial perturbation training
3. **Information Fusion**: Fuses generated and observed motion during the denoising process

### Core Concepts

- **Training Phase**: Learn the distribution of normal motion patterns while using a perturbation generator to create adversarial samples for enhanced robustness
- **Inference Phase**: Reconstruct normal motion from input skeleton sequences using the diffusion model; anomalous motions deviate from the normal distribution and result in larger reconstruction errors

---

## 2. Network Architecture

### 2.1 Overall Architecture

```
                     Input Skeleton Sequence x (B, C, T, V)
                                 |
                 +---------------+---------------+
                 |                               |
                 v                               v
         +---------------+               +---------------+
         |   Condition   |               | Perturbation  |
         |    Encoder    |               |   Generator   |
         |  (DCT/STSE)   |               |    (STSAE)    |
         +-------+-------+               +-------+-------+
                 |                               |
                 v                               v
         +---------------+               +---------------+
         |   Condition   |               |  Adversarial  |
         |   Embedding   |               |  Perturbation |
         | (latent_dim)  |               |  (B,C,T,V)    |
         +-------+-------+               +-------+-------+
                 |                               |
                 +---------------+---------------+
                                 |
                                 v
                 +-------------------------------+
                 |     Denoising UNet Network    |
                 |        (STSAE_Unet)           |
                 |                               |
                 |  Input: x_t + time_emb + cond |
                 |               |               |
                 |               v               |
                 |  +------------------------+   |
                 |  |  Encoder (Downsampling)|   |
                 |  | [16,32,64,128,128,128] |   |
                 |  +------------------------+   |
                 |               |               |
                 |               v               |
                 |  +------------------------+   |
                 |  |  Decoder (Upsampling)  |   |
                 |  |   [128, 64, 32, 2]     |   |
                 |  +------------------------+   |
                 |               |               |
                 |               v               |
                 |  Output: Pred Noise (B,C,T,V) |
                 +-------------------------------+
                                 |
                                 v
                       Reconstructed Skeleton
```

### 2.2 Denoising UNet Architecture

```
Input: x_t (B, C, T, V)
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
Output: Predicted Noise (B, C, T, V)
```

### 2.3 Main Components

#### 2.3.1 Denoising Network (STSAE_Unet)

A UNet-based spatiotemporal graph convolutional network for noise prediction:

- **Input**: Noisy skeleton sequence `(B, C, T, V)` + time step embedding + condition embedding
- **Output**: Predicted noise `(B, C, T, V)`

```python
model = STSAE_Unet(
    c_in=num_coords,           # Coordinate dimension (2 for 2D)
    embedding_dim=128,         # Time embedding dimension
    unet_down_channels=[16, 32, 64, 128, 128, 128, 128],  # Downsampling channels
    unet_up_channels=[128, 64, 32, 2],                     # Upsampling channels
    n_frames=seg_len,          # Sequence length
    n_joints=17,               # Number of joints
    dropout=0.4
)
```

#### 2.3.2 Condition Encoder

**Option 1: DCT Feature Extraction (Recommended)**

```python
def extract_dct_feature(self, tensor):
    # 1. Apply DCT transform to input sequence
    tensor_dct = self.transform_dct(tensor)  # (B, T, C*J)
    
    # 2. Flatten and select top-k components
    tensor_flat = tensor_dct.view(tensor_dct.shape[0], -1)
    condition_data, index = torch.topk(tensor_flat, self.cond_latent_dim, dim=1)
    
    return condition_data  # (B, latent_dim)
```

**Option 2: Trainable Encoder (STSE)**

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

#### 2.3.3 Perturbation Generator (STSAE)

Generates adversarial perturbations to enhance model robustness:

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

### 2.4 Data Format

| Dimension | Meaning | Typical Value |
|-----------|---------|---------------|
| B | Batch Size | 4096 |
| C | Coordinate Dimension | 2 (x, y) |
| T | Number of Frames | 24 (Avenue), 6 (UBnormal) |
| V | Number of Joints | 17 |

---

## 3. Training Pipeline

### 3.1 Training Flow Diagram

```
+-----------------------------------------------------------+
|                      Training Loop                        |
+-----------------------------------------------------------+
|                                                           |
|   Input: Normal skeleton sequence x                       |
|            |                                              |
|            v                                              |
|   +------------------+                                    |
|   | 1. Extract       |                                    |
|   |    Condition     |                                    |
|   |  - First half    |                                    |
|   |  - DCT features  |                                    |
|   +--------+---------+                                    |
|            |                                              |
|            v                                              |
|   +----------------------------------+                    |
|   | 2. Diffusion Forward Process     |                    |
|   |   - Sample timestep t            |                    |
|   |   - Add noise                    |                    |
|   |   x_t = sqrt(a)*x + sqrt(1-a)*n  |                    |
|   +-----------------+----------------+                    |
|                     |                                     |
|                     v                                     |
|   +------------------+    +------------------+            |
|   | 3. Perturbation  |--->| 4. Adversarial   |            |
|   |    Generation    |    |    Sample        |            |
|   |   (perturbe_     |    |   x' = x_t -     |            |
|   |    generator)    |    |   eps*sign(grad) |            |
|   +------------------+    +--------+---------+            |
|                                    |                      |
|                                    v                      |
|                           +------------------+            |
|                           | 5. Noise         |            |
|                           |    Prediction    |            |
|                           |   pred = UNet(   |            |
|                           |     x', t, cond) |            |
|                           +--------+---------+            |
|                                    |                      |
|                                    v                      |
|                           +------------------+            |
|                           | 6. Loss          |            |
|                           |    Computation   |            |
|                           |   L = MSE(pred,  |            |
|                           |         noise)   |            |
|                           +--------+---------+            |
|                                    |                      |
|                                    v                      |
|                           +------------------+            |
|                           | 7. Dual Optimizer|            |
|                           |    Update        |            |
|                           |   opt1: min L    |            |
|                           |   opt2: max L    |            |
|                           +------------------+            |
|                                                           |
+-----------------------------------------------------------+
```

### 3.2 Core Code Analysis

#### 3.2.1 Loss Computation

```python
def compute_loss(self, tensor_data, optimizer_idx):
    # 1. Extract condition information (first half of frames)
    condition_data, _, _ = self._mask(tensor_data, self.masked_rate)
    
    # 2. Condition encoding
    if self.dct:
        condition_embedding = self.extract_dct_feature(condition_data)
    else:
        condition_embedding, _ = self._encode_condition(condition_data)
    
    # 3. Diffusion forward process: add noise
    t = self.noise_scheduler.sample_timesteps(tensor_data.shape[0])
    x_t, noise = self.noise_scheduler.noise_graph(tensor_data, t)
    
    # 4. Perturbation training
    if self.perturb:
        _, grad = self.perturbe_generator(x_t, t)
        x_perturbed_t = self.adversarial_example(x_t, self.weight_perturb, grad)
        predicted_noise, _ = self.model(x_perturbed_t, t=t, 
                                        condition_data=condition_embedding)
    else:
        predicted_noise, _ = self.model(x_t, t=t, 
                                        condition_data=condition_embedding)
    
    # 5. Compute loss
    loss = torch.mean(self.loss_fn(predicted_noise, noise))
    
    # 6. Return loss (perturbation generator maximizes loss)
    return loss if optimizer_idx == 0 else -loss
```

#### 3.2.2 Adversarial Sample Generation

```python
def adversarial_example(self, x, epsilon, perturbation):
    """
    Generate adversarial sample
    
    Args:
        x: Input data
        epsilon: Perturbation strength
        perturbation: Perturbation direction (gradient)
    
    Returns:
        Adversarial sample x' = x - epsilon * sign(perturbation)
    """
    return -epsilon * perturbation.sign() + x
```

### 3.3 Training Strategy

#### Dual Optimizer Strategy

```python
def training_step(self, batch, batch_idx):
    optimizers = self.optimizers()
    tensor_data, _ = self._unpack_data(batch)
    
    if self.current_epoch <= self.n_epochs_tune:
        # Phase 1: Train both denoising network and perturbation generator
        
        # Update denoising network (minimize loss)
        loss_1 = self.compute_loss(tensor_data, optimizer_idx=0)
        optimizers[0].zero_grad()
        self.manual_backward(loss_1)
        optimizers[0].step()
        
        # Update perturbation generator (maximize loss)
        loss_2 = self.compute_loss(tensor_data, optimizer_idx=1)
        optimizers[1].zero_grad()
        self.manual_backward(loss_2)
        optimizers[1].step()
    else:
        # Phase 2: Train only denoising network
        loss_1 = self.compute_loss(tensor_data, optimizer_idx=0)
        optimizers[0].zero_grad()
        self.manual_backward(loss_1)
        optimizers[0].step()
```

---

## 4. Inference and Evaluation

### 4.1 Inference Flow Diagram

```
+-----------------------------------------------------------+
|                    Inference Pipeline                     |
+-----------------------------------------------------------+
|                                                           |
|   Input: Test skeleton sequence x                         |
|            |                                              |
|            v                                              |
|   +------------------+                                    |
|   | 1. Extract       |                                    |
|   |    Condition     |                                    |
|   |   cond = DCT(x)  |                                    |
|   +--------+---------+                                    |
|            |                                              |
|            v                                              |
|   +------------------+                                    |
|   | 2. Initialize    |                                    |
|   |    Pure Noise    |                                    |
|   |   x_T ~ N(0, I)  |                                    |
|   +--------+---------+                                    |
|            |                                              |
|            v                                              |
|   +-----------------------------------------------+       |
|   | 3. Iterative Denoising (t = T, T-1, ..., 1)   |       |
|   |   +---------------------------------------+   |       |
|   |   | a. Perturbation Processing            |   |       |
|   |   |    x'_t = x_t - eps*sign(perturb)     |   |       |
|   |   +---------------------------------------+   |       |
|   |   | b. Noise Prediction                   |   |       |
|   |   |    pred = UNet(x'_t, t, cond)         |   |       |
|   |   +---------------------------------------+   |       |
|   |   | c. Denoise One Step                   |   |       |
|   |   |    x_{t-1} = denoise_step(x_t, pred)  |   |       |
|   |   +---------------------------------------+   |       |
|   |   | d. Data Fusion                        |   |       |
|   |   |    x_{t-1} = fuse(x_{t-1}, x_hat)     |   |       |
|   |   +---------------------------------------+   |       |
|   +--------+--------------------------------------+       |
|            |                                              |
|            v                                              |
|   +------------------+                                    |
|   | 4. Compute       |                                    |
|   |    Reconstruction|                                    |
|   |    Error         |                                    |
|   |   loss = MSE(    |                                    |
|   |     x_hat, x)    |                                    |
|   +--------+---------+                                    |
|            |                                              |
|            v                                              |
|   +------------------+                                    |
|   | 5. Anomaly Score |                                    |
|   |    Computation   |                                    |
|   |   score = f(loss)|                                    |
|   +------------------+                                    |
|                                                           |
+-----------------------------------------------------------+
```

### 4.2 Denoising Process

```python
def denoise(self, input_data, aggr_strategy):
    generated_xs = []
    B = input_data.shape[0]
    
    # 1. Extract condition information
    condition_data, _, _ = self._mask(input_data, 0.5)
    condition_embedding = self.extract_dct_feature(condition_data)
    
    for i in range(self.n_generated_samples):
        # 2. Start from pure noise
        x = torch.randn_like(input_data, device=self.device)
        
        # 3. Iterative denoising
        for j in reversed(range(1, self.noise_steps)):
            t = torch.full((B,), j, dtype=torch.long, device=self.device)
            t_ = torch.full((B,), j-1, dtype=torch.long, device=self.device)
            
            # Get diffusion parameters
            alpha = self._alpha[t][:, None, None, None]
            alpha_hat = self._alpha_hat[t][:, None, None, None]
            beta = self._beta[t][:, None, None, None]
            
            # Prepare noise
            noise = torch.randn_like(x) if j > 1 else torch.zeros_like(x)
            
            # Get observed data at timestep t-1
            x_t, _ = self.noise_scheduler.noise_graph(input_data, t_)
            
            # Perturbation processing
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
            
            # Denoise one step
            x = (1 / torch.sqrt(alpha)) * \
                (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + \
                torch.sqrt(beta) * noise
            
            # Data fusion (except for the last step)
            if j != 1:
                x = self._fuse_data(x, x_t)
        
        generated_xs.append(x)
    
    # 4. Aggregation strategy to select best result
    selected_x, loss = self._aggregation_strategy(generated_xs, input_data, aggr_strategy)
    return selected_x, loss
```

### 4.3 Anomaly Score Computation

```python
def post_processing(self, out, gt_data, trans, meta, frames):
    """
    Post-processing: Compute anomaly scores and evaluate AUC
    """
    # 1. Iterate over each video clip
    for idx in range(len(all_gts)):
        scene_idx, clip_idx = scene_clips[idx]
        gt = np.load(os.path.join(self.gt_path, all_gts[idx]))
        
        # 2. Compute reconstruction error for each person
        error_per_person = []
        for fig in figs_ids:
            # Build error matrix
            loss_matrix = compute_var_matrix(out_fig, frames_fig, n_frames)
            # Take maximum error per frame
            fig_reconstruction_loss = np.nanmax(loss_matrix, axis=0)
            # Pad boundaries
            fig_reconstruction_loss = pad_scores(fig_reconstruction_loss, gt, 
                                                  self.anomaly_score_pad_size)
            error_per_person.append(fig_reconstruction_loss)
        
        # 3. Aggregate scores across all persons
        clip_score = np.stack(error_per_person, axis=0)
        clip_score_log = np.log1p(clip_score)
        # Mean + range (considering differences in multi-person scenes)
        clip_score = np.mean(clip_score, axis=0) + \
                     (np.amax(clip_score_log, axis=0) - np.amin(clip_score_log, axis=0))
        
        # 4. Smoothing
        clip_score = score_process(clip_score, 
                                   self.anomaly_score_frames_shift,
                                   self.anomaly_score_filter_kernel_size)
    
    # 5. Compute AUC
    auc = roc_auc_score(gt, pds)
    return auc
```

---

## 5. Configuration Parameters

### 5.1 Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `perturb` | bool | true | Enable perturbation training |
| `weight_perturb` | float | 0.1 | Perturbation strength |
| `dct` | bool | true | Use DCT for condition feature extraction |
| `masked_rate_dct` | float | 0.1 | DCT mask rate |
| `noise_steps` | int | 20 | Number of diffusion steps |
| `n_epochs_tune` | int | 100 | Number of epochs for perturbation generator training |

### 5.2 Network Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `embedding_dim` | int | UNet time embedding dimension |
| `latent_dim` | int | Condition encoder latent space dimension |
| `unet_down_channels` | list | UNet downsampling channel numbers |
| `unet_up_channels` | list | UNet upsampling channel numbers |
| `dropout` | float | Dropout probability |

### 5.3 Data Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `seg_len` | int | Input sequence length |
| `batch_size` | int | Batch size |
| `num_transform` | int | Number of data augmentation transforms |
| `normalization_strategy` | str | Normalization strategy ('robust'/'none') |

### 5.4 Evaluation Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `filter_kernel_size` | int | Anomaly score smoothing kernel size |
| `frames_shift` | int | Frame offset compensation |
| `pad_size` | int | Boundary padding size |
| `aggregation_strategy` | str | Multi-sample aggregation strategy |

---

## Appendix: File Structure

```
FG-Diff/
├── models/
│   ├── FG_DIFF.py          # Main model definition
│   ├── stsae/
│   │   ├── stsae.py        # STSAE/STSE encoder
│   │   └── stsae_unet.py   # UNet network
│   └── gcae/
│       └── stsgcn.py       # Spatiotemporal graph convolution layers
├── utils/
│   ├── dataset.py          # Dataset loading
│   ├── diffusion_utils.py  # Diffusion process utilities
│   ├── eval_utils.py       # Evaluation utilities
│   └── get_robust_data.py  # Data preprocessing
├── config/
│   ├── Avenue/             # Avenue dataset configuration
│   ├── STC/                # ShanghaiTech dataset configuration
│   └── UBnormal/           # UBnormal dataset configuration
├── train_FG-DIFF.py        # Training script
└── eval_FG-DIFF.py         # Evaluation script
```
