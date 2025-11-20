# Vision Transformer for Image Classification

A comprehensive implementation of Vision Transformers (ViT) from scratch using PyTorch, designed to understand transformer architecture fundamentals and advanced training techniques for computer vision tasks.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Models](#models)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Training Techniques](#training-techniques)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## 🎯 Overview

This project implements Vision Transformers for image classification on CIFAR-10, exploring the fundamental question: **"An Image is Worth 16×16 Words"**. By treating images as sequences of patches, we demonstrate how transformer architectures—originally designed for NLP—can achieve state-of-the-art performance in computer vision.

### Key Objectives

- **Educational**: Understand transformer mechanics from the ground up
- **Experimental**: Implement and compare different architectural depths
- **Practical**: Apply advanced training techniques for optimal performance

## Architecture

### Core Components

#### 1. Patch Embedding
Images are divided into fixed-size patches and linearly embedded:

```python
# Configuration for CIFAR-10 (32×32 images)
patch_size = 8×8          # Each patch is 8×8 pixels
n_patches = 16            # Total (32/8)² = 16 patches per image
d_model = 256             # Embedding dimension
```

**Implementation**: Uses a `Conv2d` layer with `kernel_size = stride = patch_size` for efficient patch extraction and projection.

#### 2. Positional Encoding
Sinusoidal positional encodings + learnable CLS token:

- **Fixed sinusoidal encodings**: `PE(pos, 2i) = sin(pos/10000^(2i/d_model))`
- **CLS token**: Learnable classification token prepended to the sequence
- Allows the model to understand spatial relationships between patches

#### 3. Multi-Head Self-Attention (MHSA)
The core of the transformer architecture:

```
Q, K, V = Linear projections of input
Attention(Q,K,V) = softmax(QK^T / √d_k) × V
```

- **Number of heads**: 8
- **Head dimension**: d_model / n_heads = 32
- Enables the model to attend to different representation subspaces

#### 4. Feed-Forward Network (MLP)
Two-layer network with GELU activation:

```
MLP(x) = GELU(Linear(x)) → Dropout → Linear → Dropout
```

- **Expansion ratio**: 4× (hidden_dim = 4 × d_model = 1024)
- Processes each position independently

#### 5. Residual Connections & Layer Normalization
Essential for training deep networks:

```python
x = x + Attention(LayerNorm(x))      # Pre-normalization
x = x + MLP(LayerNorm(x))            # Residual pathways
```

## Models

### VisionTransformer1 (Baseline)
- **Layers**: 6 encoder blocks
- **Parameters**: ~5M
- **Target Accuracy**: ~80-84% on CIFAR-10
- Focus: Understanding basic transformer mechanics

### VisionTransformer2 (Advanced)
- **Layers**: 12 encoder blocks
- **Parameters**: ~10M
- **Target Accuracy**: >85% on CIFAR-10
- **Advanced Features**:
  - Stochastic Depth (DropPath) with linear scheduling
  - Enhanced regularization strategies
  - Exponential Moving Average (EMA)
  - Mixed precision training

## Key Features

### 1. Stochastic Depth (DropPath)
Inspired by **[Huang et al., 2016](https://arxiv.org/pdf/1603.09382)** - "Deep Networks with Stochastic Depth"

```python
drop_path_rate = 0.2  # Maximum drop probability
# Linear scheduling: p_l = l/L × p_L
```

**Mechanism**: Randomly drops entire transformer blocks during training while keeping skip connections, effectively reducing network depth per mini-batch.

**Benefits**:
- Reduces overfitting in deep networks
- Acts as implicit ensemble learning
- Improves gradient flow
- Speeds up training

### 2. Advanced Data Augmentation

```python
# Training augmentations
- RandomResizedCrop(32, scale=(0.8, 1.0))
- RandomHorizontalFlip(p=0.5)
- RandAugment(num_ops=2, magnitude=9)
- RandomErasing(p=0.25)
- Mixup (α=0.2)
- CutMix (prob=0.5)
```

### 3. Regularization Techniques
- **Dropout**: 0.1 in attention and MLP layers
- **Weight Decay**: 0.05 (AdamW optimizer)
- **Label Smoothing**: 0.1
- **Gradient Clipping**: Max norm of 1.0

### 4. Optimized Training Strategy

#### Learning Rate Schedule
```python
# Warmup phase (20 epochs)
lr_warmup = base_lr × (epoch / warmup_epochs)

# Cosine annealing (remaining epochs)
lr = min_lr + 0.5 × (base_lr - min_lr) × (1 + cos(π × t/T))
```

#### Exponential Moving Average (EMA)
```python
decay = 0.9999
θ_ema = decay × θ_ema + (1 - decay) × θ_model
```

Maintains shadow parameters for more stable predictions.

#### Mixed Precision Training
- Uses `torch.amp.autocast()` for automatic mixed precision
- Reduces memory usage by ~50%
- Speeds up training by 2-3×

## Dataset

### CIFAR-10
- **Training samples**: 50,000 images
- **Test samples**: 10,000 images
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image size**: 32×32×3 (RGB)

**Normalization**:
```python
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
```

**Train/Val Split**: 90% training, 10% validation (with early stopping)

## Training Techniques

### Hyperparameters (VisionTransformer2)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| d_model | 256 | Balance between capacity and efficiency |
| n_heads | 8 | Standard for medium-sized models |
| n_layers | 12 | Deep enough for complex features |
| patch_size | 8×8 | Suitable for 32×32 images |
| batch_size | 128 | Optimal for GPU memory |
| base_lr | 3e-4 | Recommended for AdamW |
| epochs | 300 | With early stopping |
| warmup_epochs | 20 | Stabilizes early training |

### Training Pipeline

1. **Initialization**: Careful weight initialization (default PyTorch)
2. **Warmup**: Linear learning rate warmup for 20 epochs
3. **Main Training**: Cosine annealing scheduler with mixup/cutmix
4. **EMA Updates**: Shadow parameters updated at each step
5. **Validation**: Regular validation with EMA weights
6. **Early Stopping**: Stops if no improvement for 50 epochs

## Results

### Expected Performance

| Model | Parameters | Val Accuracy | Test Accuracy | Training Time* |
|-------|-----------|--------------|---------------|----------------|
| VisionTransformer1 | ~5M | ~82% | ~80-84% | ~2 hours |
| VisionTransformer2 | ~10M | ~86% | ~89%+ | ~4-5 hours |

*On google colab A100 GPU

### Visualization
Training curves show:
- Steady convergence with minimal overfitting
- Effective regularization from DropPath and augmentation
- Stable validation performance with EMA

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vision-transformer-classification.git
cd vision-transformer-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install numpy matplotlib seaborn tqdm scikit-learn
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- CUDA 11.0+ (recommended for GPU training)

## Usage

### Training VisionTransformer2

```python
# Run the notebook
jupyter notebook VisionTransformer2_with_training.ipynb

# Or execute directly in Python
python train_vit.py --model vit2 --epochs 300 --batch_size 128
```

### Configuration
Modify hyperparameters in the `config` dictionary:

```python
config = {
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 12,
    'batch_size': 128,
    'epochs': 300,
    'base_lr': 3e-4,
    # ... more parameters
}
```

### Inference

```python
# Load trained model
checkpoint = torch.load('best_vit_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
model.eval()
with torch.no_grad():
    output = model(image_tensor)
    prediction = output.argmax(dim=1)
```

## References

### Research Papers

1. **Vaswani et al., 2017** - *Attention Is All You Need*
   - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
   - The foundational transformer paper

2. **Dosovitskiy et al., 2020** - *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*
   - [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
   - Original Vision Transformer (ViT) paper
   - [Google Research GitHub](https://github.com/google-research/vision_transformer)

3. **Huang et al., 2016** - *Deep Networks with Stochastic Depth*
   - [arXiv:1603.09382](https://arxiv.org/pdf/1603.09382)
   - Introduces DropPath for training deeper networks

4. **He et al., 2016** - *Deep Residual Learning for Image Recognition*
   - Residual connections used in transformers

5. **Ba et al., 2016** - *Layer Normalization*
   - Essential for transformer training stability

### Resources

- [Hugging Face Vision Transformers Course](https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/vision-transformers-for-image-classification)
- [PyTorch Transformer Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [timm Library Implementation](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)

### Useful Articles

- [Building Vision Transformer from Scratch](https://medium.com/thedeephub/building-vision-transformer-from-scratch-using-pytorch-an-image-worth-16x16-words-24db5f159e27)
- [ViT on CIFAR-10](https://xuankunyang.github.io/blog/vit-on-cifar-10/)
- [Training ViT on Small Datasets](https://bmvc2022.mpi-inf.mpg.de/0731.pdf)
- [Understanding Inductive Bias](https://www.baeldung.com/cs/ml-inductive-bias)

**Note**: This is an educational project aimed at understanding transformer architectures. For production use, consider using pre-trained models from other more experienced plateforms.
