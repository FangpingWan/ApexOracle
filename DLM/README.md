# ApexOracle

This directory contains the implementation of Masked Diffusion Language Models (MDLM) with Classifier-Based Guidance (CBG) as well as antibiotic predictors for antimicrobial molecule design.

### Train Diffusion Language Model on Molecular SELFIES

To train the DLM with the multitask descriptor regression loss on curated SELFIES data:
```bash
python main.py \
  model=small \
  data=SELFIES \
  wandb.name=dlm-mtr \
  parameterization=subs \
  model.length=1024 \
  sampling.steps=1000
```

## Directory Structure
```
DLM/
├── configs/                    # Hydra configuration files
│   ├── callbacks/             # Training callbacks (EMA, checkpointing)
│   ├── data/                  # Dataset configurations
│   ├── guidance/              # Guidance method configs 
│   ├── lr_scheduler/          # Learning rate schedulers
│   ├── model/                 # Model architectures
│   ├── noise/                 # Noise schedule configurations
│   └── strategy/              # Training strategies
├── models/                    # Neural network architectures
│   ├── dit.py                # DiT backbone + all classifier models
│   ├── antibiotic_classifier.py  # Utilities for antibiotic guidance
│   ├── autoregressive.py     # Autoregressive baselines
│   └── ema.py                # Exponential Moving Average
├── scripts/                   # Training and evaluation scripts
├── classifier.py              # Lightning wrapper for classifiers
├── diffusion.py              # Core diffusion model with guided generation
├── dataloader.py             # Data loading utilities
├── noise_schedule.py         # Noise scheduling for diffusion
├── utils.py                  # General utilities
├── apexoracle_layers.py      # Shared model components (RegressionHead, etc.)
├── predictor_utils.py        # Utilities for predictor training
├── train_predictor.py        # Training script for predictors
├── cosine_lr.py              # Cosine learning rate scheduler
├── timmscheduler.py          # Timm scheduler integration
├── main.py                   # Training entry point
└── README.md                 # This file
```

## Installation

### Dependencies for DLM
```bash
torch: 2.4.1+cu118
datasets: 2.18.0
einops: 0.7.0
fsspec: 2024.2.0
git-lfs: 1.6
h5py: 3.10.0
hydra-core: 1.3.2
ipdb: 0.13.13
lightning: 2.2.1
notebook: 7.1.1
nvitop: 1.3.2
omegaconf: 2.3.0
packaging: 23.2
pandas: 2.2.1
rich: 13.7.1
seaborn: 0.13.2
scikit-learn: 1.4.0
transformers: 4.38.2
triton: 2.1.0
wandb: 0.13.5
flash-attn: 2.6.3
```


## Acknowledgements

This repository was built off of [MDLM](https://github.com/kuleshov-group/mdlm) and [discrete-diffusion-guidance
](https://github.com/kuleshov-group/discrete-diffusion-guidance).
