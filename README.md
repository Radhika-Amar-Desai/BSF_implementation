# Block Sparse Featurizer (BSF) Implementation

A PyTorch implementation of **Block Sparse Featurizers (BSFs)** and related sparse representation learning methods based on the paper *Block Sparse Featurizers*. This repository also includes implementations of **Sparse Autoencoders (SAEs)** for comparison, along with visualization and evaluation utilities.

## Features

- Vanilla Block Sparse Featurizer (BSF)
- Grassmann BSF
- Group Lasso BSF
- Improved variants of the above models
- Top-K Sparse Autoencoder (SAE)
- L1 Sparse Autoencoder
- Concept visualization utilities
- MDL (Minimum Description Length) evaluation
- DINOv2/DINOv3 feature experiments

## Repository Structure

```
.
├── models/                     # Model implementations
├── notebooks/                  # Training and evaluation notebooks
├── conceptual_understanding/   # Notes explaining BSF concepts
├── model_checkpoints/          # Pretrained model weights
├── data/                       # Sample dataset
├── train.py                    # Training script
├── viz.py                      # Visualization utilities
├── mdl_formula.py              # MDL evaluation
└── data.py                     # Dataset loading utilities
```

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- einops
- tqdm

Install dependencies using:

```bash
pip install torch numpy matplotlib scikit-learn einops tqdm
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Radhika-Amar-Desai/BSF_implementation.git
cd BSF_implementation
```

The main experiments can be run using the notebooks inside the `notebooks/` directory or by using the training scripts.

## Implemented Models

- Vanilla BSF
- Improved Vanilla BSF
- Grassmann BSF
- Improved Grassmann BSF
- Group Lasso BSF
- Improved Group Lasso BSF
- Top-K Sparse Autoencoder
- L1 Sparse Autoencoder

## Evaluation

The repository includes:

- Reconstruction loss
- R² score
- Minimum Description Length (MDL)
- Learned concept visualization

## Checkpoints

Pretrained checkpoints for several models are provided in the `model_checkpoints/` directory.
