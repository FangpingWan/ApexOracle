# DLM Codebase Walkthrough

This document provides a detailed walkthrough of the DLM codebase, explaining the purpose and key components of each file and directory.

## Table of Contents

1. [Overview](#overview)
2. [Core Files](#core-files)
3. [Model Architectures](#model-architectures)
4. [Configuration System](#configuration-system)
5. [Guided Generation Pipeline](#guided-generation-pipeline)
6. [Training Pipeline](#training-pipeline)

---

## Overview

The DLM codebase is organized into several key components:
- **Core diffusion logic**: `diffusion.py`, `noise_schedule.py`
- **Classifier wrapper**: `classifier.py`
- **Model architectures**: `models/dit.py`, `models/antibiotic_classifier.py`
- **Training infrastructure**: `main.py`, `dataloader.py`
- **Configuration**: `configs/` directory with Hydra configs

---

## Core Files

### main.py
**Purpose**: Main entry point for training diffusion models and classifiers

**Key Components**:
```python
@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Setup trainer with Lightning
    # Initialize model (DIT or Classifier)
    # Train with specified configs
```

**Usage**:
```bash
python main.py model=dit data=SELFIES
```

**What it does**:
1. Loads Hydra configuration from `configs/`
2. Initializes PyTorch Lightning trainer
3. Sets up model based on config (diffusion or classifier)
4. Handles distributed training, callbacks, and logging
5. Runs training loop

---

### diffusion.py (1630 lines)
**Purpose**: Core diffusion model with integrated guided generation methods

**Key Classes**:

#### `Diffusion(L.LightningModule)`
Main diffusion model class extending Lightning

**Important Methods**:

1. **`forward(x, sigma, **kwargs)`** (lines ~50-100)
   - Forward pass through diffusion model
   - Takes noisy input `x` and noise level `sigma`
   - Returns denoised predictions

2. **`sample(batch_size, **kwargs)`** (lines ~200-300)
   - Standard sampling without guidance
   - Iteratively denoises from pure noise
   - Returns generated sequences

3. **`_cbg_denoise(...)`** (lines ~1078-1194)
   ```python
   def _cbg_denoise(
       self,
       conditioning_class: int,
       gamma: float,
       classifier_model: classifier.Classifier,
       xt: torch.tensor,
       time_conditioning: torch.tensor,
       move_chance_t: torch.tensor,
       move_chance_s: torch.tensor,
       use_approx: bool = False,
       cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
   ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:
   ```

   **What it does**:
   - Implements basic Classifier-Based Guidance (CBG)
   - Combines diffusion posterior with classifier gradients
   - Formula: `p_θ(x_{t-1}|x_t) ∝ p_θ(x_{t-1}|x_t) * p_φ(y|x_{t-1})^γ`
   - Supports gradient approximation for efficiency

4. **`_cbg_denoise_antibiotic(...)`** (lines ~1195-1367)
   ```python
   def _cbg_denoise_antibiotic(
       self,
       conditioning_class: int,
       gamma: float,
       classifier_model: classifier.Classifier,
       xt: torch.tensor,
       time_conditioning: torch.tensor,
       move_chance_t: torch.tensor,
       move_chance_s: torch.tensor,
       use_approx: bool = False,
       cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
       step: int = None,
   ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:
   ```

   **What it does**:
   - Extended CBG specifically for antibiotic generation
   - Adds **nucleus sampling** for diversity
   - Filters forbidden tokens (., I, Sn, Br)
   - Supports **variable-length sequences**
   - Prints debug info for monitoring MIC predictions

5. **`_cbg_denoise_antibiotic_remdm_loop(...)`** (lines ~1368-1626)
   ```python
   def _cbg_denoise_antibiotic_remdm_loop(
       self,
       conditioning_class: int,
       gamma: float,
       classifier_model: classifier.Classifier,
       xt: torch.tensor,
       time_conditioning: torch.tensor,
       move_chance_t: torch.tensor,
       move_chance_s: torch.tensor,
       use_approx: bool = False,
       cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
       step: int = None,
       t=None,
       dt=None,
   ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:
   ```

   **What it does** (Three-phase sampling):

   **Phase 1: Early MDLM** (t > t_on = 0.7)
   - Uses MDLM sampling with strong guidance (gamma_l = 8.0)
   - Allows padding tokens to be remasked
   - Establishes initial molecular structure

   **Phase 2: Middle ReMDM** (t_off < t ≤ t_on, i.e., 0.3 < t ≤ 0.7)
   - Switches to ReMDM sampling with moderate guidance (gamma_s = 5.0)
   - **Prevents padding remask** to maintain sequence length
   - Uses reparameterization for better gradient flow
   - Refines molecular structure

   **Phase 3: Late MDLM** (t ≤ t_off = 0.3)
   - Returns to MDLM with strong guidance (gamma_l = 8.0)
   - Allows padding remask again
   - Finalizes molecular details

   **Key Features**:
   - Ensures CLS token at position 0
   - Nucleus sampling throughout
   - Forbidden token filtering
   - Time-adaptive guidance strength

**File Location References**:
- Guidance configs: `configs/guidance/cbg_antibiotic.yaml`
- Classifier import: line 15 (`import classifier`)

---

### classifier.py (504 lines)
**Purpose**: PyTorch Lightning wrapper for noise-conditioned classifiers

**Key Class**:

#### `Classifier(L.LightningModule)`
Wraps classifier models for training and inference

**Constructor** (lines ~50-150):
```python
def __init__(self, config, vocab_size, tokenizer):
    # Initialize backbone based on classifier_type
    if config.classifier_type == 'dit':
        self.backbone = dit.DITClassifier(config, vocab_size)
    elif config.classifier_type == 'dit_AMP':
        self.backbone = dit.DITClassifier_AMP(config, vocab_size)
    elif config.classifier_type == 'dit_reg_cls_AMP':
        self.backbone = dit.DIT_Reg_Cls_AMP(config, vocab_size)
    elif config.classifier_type == 'dit_synergy_cls_AMP':
        self.backbone = dit.DIT_Syn_Cls_Pep_Cls_AMP(config, vocab_size)
```

**Key Methods**:

1. **`forward(x, sigma, x_emb=None, step=None)`** (lines ~200-220)
   - Forward pass through classifier
   - `x`: noisy sequences (batch_size, seq_len)
   - `sigma`: noise levels (batch_size,)
   - `x_emb`: optional genome/text embeddings
   - Returns class logits or regression values

2. **`get_log_probs(x, sigma, x_emb=None)`** (lines ~250-270)
   ```python
   def get_log_probs(self, x, sigma, x_emb=None):
       """
       Get log probabilities for standard CBG guidance.

       Returns:
           log_probs: Log probabilities via softmax (for classification)
       """
       logits = self.forward(x, sigma, x_emb=x_emb)
       return torch.nn.functional.log_softmax(logits, dim=-1)
   ```

   **Used in**: `_cbg_denoise()` for standard CBG

3. **`get_log_probs_antibiotic_guaidance(x, sigma, x_emb=None, step=None)`** (lines ~271-300)
   ```python
   def get_log_probs_antibiotic_guaidance(self, x, sigma, x_emb=None, step=None):
       """
       Get raw regression outputs for antibiotic guidance.

       Note: Method name has typo "guaidance" instead of "guidance" for
       historical reasons. Kept for backward compatibility.

       Returns:
           raw_outputs: Regression values (NOT log probs)
       """
       return self.forward(x, sigma, x_emb=x_emb, step=step)
   ```

   **Used in**: `_cbg_denoise_antibiotic()` and `_cbg_denoise_antibiotic_remdm_loop()`

   **Why raw outputs?** For MIC regression, we want continuous values, not probabilities

4. **`training_step(batch, batch_idx)`** (lines ~350-400)
   - Processes training batch
   - Computes loss (cross-entropy or MSE)
   - Logs metrics (accuracy, precision, recall)

5. **`validation_step(batch, batch_idx)`** (lines ~400-450)
   - Evaluates on validation data
   - Computes validation metrics
   - Used for model selection

**Metrics Classes** (lines ~450-504):
```python
class CrossEntropy:
    # Cross-entropy loss for classification

class Accuracy:
    # Classification accuracy

class Precision:
    # Precision metric

class Recall:
    # Recall metric
```

---

### predictor_utils.py (839 lines)
**Purpose**: Utilities and model classes for training noise-conditioned predictors

**Key Components**:

#### 1. `mol_emb_mdlm(nn.Module)` - Noise-Conditioned MDLM Wrapper
**Purpose**: Wraps a pretrained DIT backbone to create a predictor that works with noisy inputs

**Key Methods**:
```python
def __init__(self, config, vocab_size, ckpt_path, mask_index):
    # Loads pretrained DIT backbone
    self.backbone = self.load_DIT()
    # Sets up noise schedule
    self.noise = noise_schedule.get_noise(self.config)
    # Adds regression head
    self.regression_head = RegressionHead(...)
```

```python
def q_xt(self, x, move_chance):
    """Computes noisy sample xt by randomly masking tokens.

    Args:
        x: Clean input sequence
        move_chance: Probability of masking each token

    Returns:
        xt: Noisy sequence with some tokens replaced by mask_index
    """
    move_indices = torch.rand(*x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt
```

**Why noise conditioning?**
For classifier-based guidance to work during diffusion sampling, the predictor must be able to handle noisy inputs (partially masked sequences). Training with noise conditioning enables the predictor to provide meaningful gradients at all diffusion timesteps.

#### 2. Global Tokenizer Pattern
```python
_tokenizer = None

def set_tokenizer(tokenizer):
    """Set global tokenizer for collate functions."""
    global _tokenizer
    _tokenizer = tokenizer
```

**Purpose**: Avoids circular imports by using a global tokenizer variable that's set at runtime. This allows collate functions to access the tokenizer without importing from the training script.

**Usage in train_predictor.py**:
```python
tokenizer = get_tokenizer(...)
set_tokenizer(tokenizer)  # Make available to collate functions
```

#### 3. Dataset Classes with SELFIES Support

**`SMILESDataset_with_genome_and_text(Dataset)`** (lines ~150-250):
```python
class SMILESDataset_with_genome_and_text(Dataset):
    """Dataset for molecular sequences with genome and text embeddings.

    Note: Despite the name 'SMILES', this dataset actually processes tokenized
    SELFIES (SELF-referencIng Embedded Strings) representations, not SMILES.
    The 'SMILES' column contains pre-tokenized SELFIES strings.
    """
```

**Why the confusing name?** Historical reasons. The original data columns were named "SMILES" but actually contain SELFIES strings. The name was kept for backward compatibility.

**Other Dataset Classes**:
- `SMILESDataset_with_genome_wo_text` - Without text embeddings
- `SMILESDataset_wo_genome_w_text` - Without genome embeddings
- `SMILESDataset_wo_genome_wo_text` - Only molecular sequences

#### 4. Collate Functions with Tokenization
Multiple collate functions for different embedding configurations:
- `collate_fn()` - Full: genome + text + molecular embeddings
- `collate_fn_genome_only()` - Genome + molecular embeddings
- `collate_fn_text_only()` - Text + molecular embeddings
- `collate_fn_cls()` - Simple classification (molecular only)

All handle padding to fixed length (1024) and use the global `_tokenizer`.

#### 5. Shared Components (imported from apexoracle_layers.py)
```python
from apexoracle_layers import (
    RegressionHead,              # Regression output layer
    FirstTokenAttention_genome,  # Cross-attention for genome embeddings
    load_all_genome_embeddings,  # Load bacterial genome embeddings
    load_text_wo_genome_embeddings  # Load text embeddings
)
```

**Purpose**: Avoids code duplication by importing shared model components.

#### 6. Utility Functions
- `get_embedded_genome_IDs()` - Get genome IDs with available embeddings
- `get_original_strain_name_with_genome_embedding()` - Map strain names
- `calculate_r2()` - Calculate R² metric for regression
- `cluster_species()` - Species clustering for data splits
- `exclude_wrong_species_ATCC_map()` - Data cleaning utility

---

### train_predictor.py (1111 lines)
**Purpose**: Training script for noise-conditioned molecular property predictors

**Overview**:
This script trains ensemble predictors that predict molecular properties (e.g., MIC values) from noisy SELFIES representations. The trained models can be used with `classifier.py` for classifier-based guidance during diffusion sampling.

**Key Sections** (marked with clear headers for readability):

#### 1. ARGUMENT PARSING (lines ~10-30)
```python
parser = argparse.ArgumentParser(
    description='Train noise-conditioned predictor for classifier-guided generation'
)
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('-e', '--epoch', type=int, default=100)
parser.add_argument('-w', '--weight_decay', type=float, default=0.0)
```

#### 2. CONFIGURATION & INITIALIZATION (lines ~35-70)
```python
genome_embedding_scale_factor = 1e14
text_embedding_scale_factor = 1
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
```

#### 3. LOGGING SETUP (lines ~75-110)
Sets up detailed logging to both file and console with timestamps.

#### 4. DATA LOADING & PREPROCESSING (lines ~115-350)
- Loads genome and text embeddings
- Creates train/validation/test splits
- Handles species clustering to avoid data leakage
- Creates DataLoaders with appropriate collate functions

```python
# Load embeddings
genome_embeddings_dict = load_all_genome_embeddings(embeddings_folder_path, device)

# Create datasets
train_dataset = SMILESDataset_with_genome_and_text(
    csv_file_path,
    genome_embeddings_dict,
    text_embeddings,
    tokenizer
)

# Set global tokenizer for collate functions
set_tokenizer(tokenizer)
```

#### 5. ENSEMBLE TRAINING LOOP (lines ~355-end)
**Why ensemble?** Multiple models trained with different random seeds improve robustness and provide uncertainty estimates.

```python
for ensemble in range(num_ensembles):
    # --------------------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------------------
    num_epochs = args.epoch
    batch_size = 70
    freeze_epochs = 5000  # Freeze backbone for N epochs

    # --------------------------------------------------------------------
    # Model Initialization
    # --------------------------------------------------------------------
    mdlm_model = mol_emb_mdlm(
        config,
        len(tokenizer.get_vocab()),
        DIT_ckpt_path,
        tokenizer.mask_token_id
    )

    # --------------------------------------------------------------------
    # Training Loop with Noise Conditioning
    # --------------------------------------------------------------------
    for epoch in range(num_epochs):
        # Sample noise level for this batch
        noise_level = torch.rand(1) * (1 - sampling_eps) + sampling_eps

        # Create noisy inputs
        xt = mdlm_model.q_xt(clean_inputs, noise_level)

        # Forward pass with noisy inputs
        predictions = mdlm_model(xt, noise_level, genome_emb, text_emb)

        # Compute loss
        loss = F.mse_loss(predictions, targets)
```

**Key Training Features**:
- **Noise conditioning**: Trains on inputs with random masking levels
- **Frozen backbone**: Option to freeze DIT backbone initially
- **Mixed precision**: Uses PyTorch AMP for faster training
- **Best model checkpointing**: Saves best model based on validation R²
- **Detailed logging**: Tracks R², MSE, and correlation metrics

**Checkpoint Format**:
```python
checkpoint = {
    'mdlm_model_state_dict': mdlm_model.state_dict(),
    're_head_state_dict': regression_head.state_dict(),
    'best_r2': best_r2,
    'epoch': epoch
}
```

**Usage in Classifier-Based Guidance**:
1. Train predictor: `python train_predictor.py -d 0 -e 100`
2. Save checkpoint to path specified in config
3. Load in `models/dit.py` via `load_pretrained_weight()` method (lines 625-642)
4. Use during guided generation in `diffusion.py`

---

### noise_schedule.py
**Purpose**: Defines noise schedules for diffusion process

**Key Functions**:
- `get_noise_schedule(schedule_type)`: Returns noise schedule
- `loglinear_schedule(t)`: Log-linear noise schedule
- `cosine_schedule(t)`: Cosine noise schedule

**What it does**:
- Controls how much noise is added at each timestep
- Affects generation quality and speed

---

### utils.py
**Purpose**: General utility functions

**Common Functions**:
- Tokenization utilities
- Data preprocessing
- Checkpoint loading/saving
- Evaluation metrics

---

### dataloader.py
**Purpose**: Data loading and preprocessing

**Key Components**:
- Dataset classes for molecular sequences
- Tokenization (SELFIES format)
- Batch collation
- Data augmentation

**Common Datasets**:
- `SELFIESDataset`: Basic SELFIES sequences
- `AntibioticDataset`: With MIC labels
- `SynergyDataset`: Peptide + small molecule pairs

---

### cosine_lr.py & timmscheduler.py
**Purpose**: Learning rate schedulers

- **cosine_lr.py**: Cosine annealing scheduler
- **timmscheduler.py**: Timm library scheduler integration

---

## Model Architectures

### models/dit.py (1408 lines)
**Purpose**: DiT backbone and all classifier models

**File Structure**:

#### Part 1: DiT Diffusion Model (lines 1-468)

**Class: `DIT(nn.Module)`**
- Transformer-based diffusion backbone
- Uses DiT blocks (attention + MLP)
- Supports flash-attention for speed

**Key Components**:
```python
class DIT(nn.Module):
    def __init__(self, config, vocab_size):
        # Embedding layers
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.sigma_embeddings = nn.Sequential(...)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DITBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # Output head
        self.output_head = nn.Linear(d_model, vocab_size)
```

#### Part 2: Classifier Models (lines 469-1408)

**1. DITClassifier (lines ~469-561)**
```python
class DITClassifier(nn.Module):
    """
    Basic noise-conditioned classifier.

    Features:
    - Multiple pooling methods: mean, max, cls, last, attention_mean
    - Noise-level conditioning via sigma embeddings
    - Simple classification head
    """
    def __init__(self, config, vocab_size):
        self.encoder = DIT(config, vocab_size)
        self.pooling_method = config.pooling  # 'cls', 'mean', 'max', etc.
        self.classifier_head = nn.Linear(d_model, num_classes)

    def forward(self, x, sigma):
        # Encode sequences
        h = self.encoder(x, sigma)

        # Pool features based on method
        if self.pooling_method == 'cls':
            pooled = h[:, 0, :]  # CLS token
        elif self.pooling_method == 'mean':
            pooled = h.mean(dim=1)
        elif self.pooling_method == 'max':
            pooled = h.max(dim=1)[0]
        # ... other pooling methods

        # Classify
        return self.classifier_head(pooled)
```

**When to use**: Basic classification tasks without embeddings

---

**2. DITClassifier_AMP (lines ~562-757)**
```python
class DITClassifier_AMP(nn.Module):
    """
    Antibiotic MIC prediction with cross-attention.

    Features:
    - CLS token extraction for molecular features
    - Cross-attention with genome embeddings (Evo2)
    - Cross-attention with text embeddings
    - Regression head for MIC prediction
    """
    def __init__(self, config, vocab_size):
        # Encoder
        self.encoder = DIT(config, vocab_size)

        # Cross-attention modules
        self.co_cross_attn_genome = FirstTokenAttention_genome(d_model)
        self.co_cross_attn_text = FirstTokenAttention_genome(d_model)

        # Regression head
        self.regression_head = RegressionHead(
            input_dim=d_model * 3,  # mol + genome + text
            output_dim=1
        )

    def forward(self, x, sigma, x_emb=None, step=None):
        # Encode molecular sequence
        h = self.encoder(x, sigma)

        # Extract CLS token (molecular features)
        mol_cls_embedding = h[:, 0, :]  # Shape: (batch, d_model)

        # Cross-attend with genome embeddings
        genome_emb = x_emb['genome']  # Shape: (batch, emb_dim)
        mol_cls_genome = self.co_cross_attn_genome(
            mol_cls_embedding, genome_emb
        )

        # Cross-attend with text embeddings
        text_emb = x_emb['text']
        mol_cls_text = self.co_cross_attn_text(
            mol_cls_embedding, text_emb
        )

        # Concatenate features
        final_features = torch.cat([
            mol_cls_embedding,
            mol_cls_genome,
            mol_cls_text
        ], dim=-1)

        # Predict MIC
        mic_prediction = self.regression_head(final_features)
        return mic_prediction
```

**When to use**:
- Antibiotic MIC prediction
- Cross-species generalization
- When you have genome/text embeddings

**Key Insight**:
- CLS token (`x[:, 0, :]`) captures holistic molecular properties
- Cross-attention integrates species-specific information
- Enables zero-shot prediction on new bacterial species

---

**3. DIT_Reg_Cls_AMP (lines ~758-1033)**
```python
class DIT_Reg_Cls_AMP(nn.Module):
    """
    Dual-head: Regression + Classification.

    Architecture:
    - Shared encoder (DIT)
    - Separate heads for regression and classification
    - Cross-attention for both tasks

    Use case:
    - Predict MIC (regression)
    - Predict antimicrobial activity (classification)
    """
    def __init__(self, config, vocab_size):
        self.encoder = DIT(config, vocab_size)

        # Regression components
        self.co_cross_attn_genome_reg = FirstTokenAttention_genome(d_model)
        self.co_cross_attn_text_reg = FirstTokenAttention_genome(d_model)
        self.regression_head = RegressionHead(d_model * 3, 1)

        # Classification components
        self.co_cross_attn_genome_cls = FirstTokenAttention_genome(d_model)
        self.co_cross_attn_text_cls = FirstTokenAttention_genome(d_model)
        self.classification_head = nn.Linear(d_model * 3, num_classes)

    def forward(self, x, sigma, x_emb=None):
        h = self.encoder(x, sigma)
        mol_cls = h[:, 0, :]

        # Regression branch
        mol_genome_reg = self.co_cross_attn_genome_reg(mol_cls, x_emb['genome'])
        mol_text_reg = self.co_cross_attn_text_reg(mol_cls, x_emb['text'])
        reg_features = torch.cat([mol_cls, mol_genome_reg, mol_text_reg], dim=-1)
        mic_pred = self.regression_head(reg_features)

        # Classification branch
        mol_genome_cls = self.co_cross_attn_genome_cls(mol_cls, x_emb['genome'])
        mol_text_cls = self.co_cross_attn_text_cls(mol_cls, x_emb['text'])
        cls_features = torch.cat([mol_cls, mol_genome_cls, mol_text_cls], dim=-1)
        activity_pred = self.classification_head(cls_features)

        return mic_pred, activity_pred
```

**When to use**: Multi-task learning for both continuous and discrete predictions

---

**4. DIT_Syn_Cls_Pep_Cls_AMP (lines ~1034-1408)**
```python
class DIT_Syn_Cls_Pep_Cls_AMP(nn.Module):
    """
    Triple-head: Synergy + Small Molecule Classification + Peptide Classification.

    Architecture:
    - Shared encoder
    - Three separate prediction heads
    - Complex cross-attention for synergy modeling

    Use case:
    - Predict synergistic effects of molecule pairs
    - Classify individual molecule properties
    """
```

**When to use**:
- Studying drug combinations
- Predicting synergistic antimicrobial effects
- Peptide-small molecule interactions

---

### models/antibiotic_classifier.py (1838 lines)
**Purpose**: Utilities and helpers for antibiotic-specific guided generation

**Key Components**:

**1. RegressionHead (lines ~50-100)**
```python
class RegressionHead(nn.Module):
    """
    Neural network for MIC regression.

    Architecture:
    - Input: Concatenated features (molecular + genome + text)
    - Hidden layers with dropout
    - Output: Single MIC value
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256]):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], output_dim)
        )
```

**2. FirstTokenAttention_genome (lines ~100-200)**
```python
class FirstTokenAttention_genome(nn.Module):
    """
    Cross-attention module for integrating embeddings.

    Mechanism:
    - Query: Molecular CLS token
    - Key/Value: Genome or text embeddings
    - Output: Attended molecular representation
    """
    def __init__(self, d_model):
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(embed_dim, d_model)
        self.value_proj = nn.Linear(embed_dim, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads=8)

    def forward(self, mol_cls, genome_emb):
        # Project to same dimension
        query = self.query_proj(mol_cls)
        key = self.key_proj(genome_emb)
        value = self.value_proj(genome_emb)

        # Cross-attention
        attended, _ = self.attention(query, key, value)
        return attended
```

**3. Embedding Loading Functions (lines ~200-500)**

```python
def load_all_genome_embeddings(species_list, embedding_path):
    """
    Load precomputed Evo2 genome embeddings.

    Args:
        species_list: List of bacterial species
        embedding_path: Path to embedding files

    Returns:
        embeddings: Dict mapping species -> embedding tensor
    """
    embeddings = {}
    for species in species_list:
        emb_file = f"{embedding_path}/{species}_evo2.pt"
        embeddings[species] = torch.load(emb_file)
    return embeddings

def load_text_wo_genome_embeddings(description_list, model_name="bert-base"):
    """
    Generate text embeddings using transformer models.

    Args:
        description_list: List of text descriptions
        model_name: HuggingFace model name

    Returns:
        embeddings: Tensor of text embeddings
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    for text in description_list:
        tokens = tokenizer(text, return_tensors='pt', padding=True)
        output = model(**tokens)
        # Use CLS token from text model
        embeddings.append(output.last_hidden_state[:, 0, :])

    return torch.stack(embeddings)
```

**4. Dataset Classes (lines ~500-1000)**

```python
class SMILESDataset_with_genome_and_text(Dataset):
    """
    Dataset for antibiotic training with multi-modal data.

    Includes:
    - SMILES sequences
    - MIC labels
    - Bacterial species
    - Genome embeddings
    - Text descriptions
    """
    def __init__(self, smiles_list, mic_list, species_list, ...):
        self.smiles = smiles_list
        self.mic = mic_list
        self.species = species_list
        self.genome_embeddings = load_all_genome_embeddings(species_list)
        self.text_embeddings = load_text_wo_genome_embeddings(descriptions)

    def __getitem__(self, idx):
        return {
            'smiles': self.smiles[idx],
            'mic': self.mic[idx],
            'genome_emb': self.genome_embeddings[self.species[idx]],
            'text_emb': self.text_embeddings[idx]
        }
```

**5. Collate Functions (lines ~1000-1200)**

```python
def collate_fn(batch):
    """
    Batch collation with padding and embedding stacking.

    Handles:
    - Variable-length sequences
    - Padding to max length in batch
    - Stacking genome/text embeddings
    """
    smiles = [item['smiles'] for item in batch]
    mics = torch.tensor([item['mic'] for item in batch])
    genome_embs = torch.stack([item['genome_emb'] for item in batch])
    text_embs = torch.stack([item['text_emb'] for item in batch])

    # Pad sequences
    max_len = max(len(s) for s in smiles)
    padded_smiles = [s + [PAD_TOKEN] * (max_len - len(s)) for s in smiles]

    return {
        'smiles': torch.tensor(padded_smiles),
        'mic': mics,
        'x_emb': {
            'genome': genome_embs,
            'text': text_embs
        }
    }
```

**6. ATCC Strain Mapping (lines ~1200-1500)**
```python
# Maps ATCC strain IDs to species names
ATCC_TO_SPECIES = {
    'ATCC_25922': 'Escherichia_coli',
    'ATCC_29213': 'Staphylococcus_aureus',
    'ATCC_27853': 'Pseudomonas_aeruginosa',
    # ... more mappings
}

def get_species_from_atcc(atcc_id):
    return ATCC_TO_SPECIES.get(atcc_id, 'Unknown')
```

**7. Species Clustering (lines ~1500-1838)**
```python
def cluster_species_by_phylogeny(species_list, n_clusters=5):
    """
    Cluster bacterial species by phylogenetic relationships.

    Used for:
    - Creating train/test splits by species
    - Evaluating cross-species generalization
    """
    # Implementation uses phylogenetic distance matrices
```

---

### models/autoregressive.py
**Purpose**: Autoregressive baseline models

**Key Models**:
- GPT-style autoregressive models
- Used as baselines to compare with diffusion

---

### models/ema.py
**Purpose**: Exponential Moving Average for model weights

**Usage**:
```python
ema = EMA(model, decay=0.9999)

# During training
ema.update(model)

# For inference
with ema.average_parameters():
    predictions = model(data)
```

---

## Configuration System

### configs/ Directory Structure

```
configs/
├── config.yaml           # Main config file
├── callbacks/
│   ├── ema.yaml         # EMA settings
│   └── checkpoint.yaml  # Checkpointing settings
├── data/
│   ├── SELFIES.yaml     # SELFIES dataset config
│   └── antibiotic_mic.yaml
├── guidance/
│   ├── cbg.yaml                # Standard CBG
│   ├── cbg_antibiotic.yaml     # Antibiotic CBG
│   ├── cbg_synergy.yaml        # Synergy prediction
│   ├── cfg.yaml                # Classifier-Free Guidance
│   ├── fudge.yaml              # FUDGE
│   ├── nos.yaml                # Noise-based sampling
│   ├── nos_antibiotic.yaml     # Antibiotic NOS
│   └── pplm.yaml               # PPLM
├── lr_scheduler/
│   ├── cosine.yaml
│   └── linear.yaml
├── model/
│   ├── dit.yaml         # DiT diffusion model
│   ├── small.yaml       # Small DiT variant
│   └── dit_classifier.yaml
├── noise/
│   ├── loglinear.yaml
│   └── cosine.yaml
└── strategy/
    ├── ddp.yaml         # Distributed Data Parallel
    └── fsdp.yaml        # Fully Sharded Data Parallel
```

### Key Config Files

#### configs/guidance/cbg_antibiotic.yaml
```yaml
# Classifier-Based Guidance for Antibiotic Generation
guidance_type: cbg_antibiotic_remdm

# Guidance strengths
gamma_l: 8.0      # MDLM phases (early/late)
gamma_s: 5.0      # ReMDM phase (middle)

# ReMDM phase boundaries
t_on: 0.7         # Start ReMDM at t=0.7
t_off: 0.3        # End ReMDM at t=0.3

# Sampling parameters
nucleus_p: 0.9    # Top-p sampling threshold
temperature: 1.0  # Sampling temperature

# Target
conditioning_class: 0  # 0 = low MIC (more potent)

# Options
use_approx: false      # Use exact gradients
forbid_tokens: ['.', 'I', 'Sn', 'Br']  # Forbidden SELFIES tokens
```

#### configs/model/dit.yaml
```yaml
# DiT Model Configuration
_target_: models.dit.DIT

# Architecture
n_layers: 12
d_model: 768
n_heads: 12
d_ff: 3072

# Dropout
dropout: 0.1
attention_dropout: 0.1

# Sequence length
max_length: 1024

# Flash attention
use_flash_attn: true
```

#### configs/model/dit_classifier.yaml
```yaml
# Classifier Configuration
_target_: classifier.Classifier

# Classifier type
classifier_type: dit_AMP  # Options: dit, dit_AMP, dit_reg_cls_AMP, dit_synergy_cls_AMP

# Architecture (same as diffusion model)
n_layers: 12
d_model: 768
n_heads: 12

# Pooling method (for basic DITClassifier)
pooling: cls  # Options: cls, mean, max, attention_mean

# Task-specific
num_classes: 10  # For classification tasks
regression: true  # For MIC regression
```

---

## Guided Generation Pipeline

### Step-by-Step: Generating Antibiotics with Low MIC

**1. Train Diffusion Model**
```bash
python main.py \
    model=dit \
    data=SELFIES \
    training.batch_size=128 \
    training.epochs=100
```

**2. Train Noise-Conditioned Classifier**
```bash
python main.py \
    model=dit_classifier \
    model.classifier_type=dit_AMP \
    data=antibiotic_mic \
    training.batch_size=64 \
    training.epochs=50
```

**3. Generate with Guidance**
```python
from DLM.diffusion import Diffusion
from DLM.classifier import Classifier

# Load models
diffusion = Diffusion.load_from_checkpoint("diffusion.ckpt")
classifier = Classifier.load_from_checkpoint("classifier.ckpt")

# Set to eval mode
diffusion.eval()
classifier.eval()

# Generate
with torch.no_grad():
    samples = diffusion.sample_with_guidance(
        batch_size=100,
        classifier_model=classifier,
        guidance_method='cbg_antibiotic_remdm',
        gamma_l=8.0,
        gamma_s=5.0,
        t_on=0.7,
        t_off=0.3,
        conditioning_class=0,  # Low MIC
        nucleus_p=0.9
    )
```

### What Happens Internally?

**Iteration 1 (t=1.0, fully noisy)**:
1. `xt` is pure noise (random tokens)
2. Diffusion model predicts denoised version
3. Classifier predicts MIC from noisy sequence
4. Guidance adjusts predictions toward low MIC
5. Sample next state `x_{t-dt}` using adjusted probs

**Iteration 500 (t=0.5, middle of ReMDM phase)**:
1. `xt` is partially denoised, molecular structure emerging
2. **ReMDM reparameterization** improves gradient flow
3. **Padding remask disabled** to maintain sequence length
4. Classifier guidance with `gamma_s=5.0` (moderate)
5. **Nucleus sampling** (p=0.9) maintains diversity

**Iteration 900 (t=0.1, nearly denoised)**:
1. `xt` is almost final molecule
2. Back to MDLM with strong guidance (`gamma_l=8.0`)
3. Refining final molecular details
4. Classifier confidence high, guidance precise
5. Final sampling produces valid SELFIES

---

## Training Pipeline

### Training Flow Diagram

```
main.py
  ├─> Load Hydra config
  ├─> Initialize Lightning Trainer
  │     ├─> DDP/FSDP strategy
  │     ├─> Mixed precision (automatic)
  │     └─> Callbacks (EMA, checkpointing)
  ├─> Initialize Model
  │     ├─> Diffusion (from diffusion.py)
  │     └─> or Classifier (from classifier.py)
  ├─> Setup DataLoader (from dataloader.py)
  │     ├─> Load dataset
  │     ├─> Tokenization
  │     └─> Batching with collate_fn
  └─> Train
        ├─> Training step
        │     ├─> Forward pass
        │     ├─> Compute loss
        │     ├─> Backprop
        │     └─> Update EMA
        ├─> Validation step
        │     └─> Evaluate metrics
        └─> Checkpointing
```

### Loss Functions

**Diffusion Model**:
```python
# Cross-entropy loss between predicted and target tokens
loss = F.cross_entropy(
    model_output.view(-1, vocab_size),
    target.view(-1)
)
```

**Classifier (MIC Regression)**:
```python
# MSE loss for regression
loss = F.mse_loss(predictions, targets)
```

**Classifier (Classification)**:
```python
# Cross-entropy for classification
loss = F.cross_entropy(logits, labels)
```

---

## Key Design Patterns

### 1. Noise Conditioning
All classifiers are conditioned on noise level `sigma`:
```python
h = encoder(x, sigma)  # sigma tells model how noisy x is
```

**Why?** Classifier needs to understand noise level to make accurate predictions from noisy inputs.

### 2. CLS Token Pooling
Extracting molecular features via CLS token:
```python
mol_cls_embedding = h[:, 0, :]  # First token
```

**Why?** Transformers learn to encode global information in CLS token, perfect for classification.

### 3. Cross-Attention for Embeddings
Integrating external knowledge:
```python
mol_features = encoder(molecule)
mol_with_genome = cross_attention(mol_features, genome_emb)
```

**Why?** Allows model to incorporate species-specific information for better generalization.

### 4. Three-Phase Sampling
MDLM → ReMDM → MDLM:
```python
if t > t_on:
    use_mdlm(gamma_l)
elif t > t_off:
    use_remdm(gamma_s)
else:
    use_mdlm(gamma_l)
```

**Why?** Combines strengths of both methods: MDLM for structure, ReMDM for refinement.

---

## Common Debugging Tips

### Issue: Guidance Not Working
**Check**:
1. Classifier is in eval mode: `classifier.eval()`
2. Gradients enabled: `torch.set_grad_enabled(True)`
3. Gamma not too high/low: `gamma ∈ [1.0, 10.0]`

### Issue: Mode Collapse
**Solution**:
1. Enable nucleus sampling: `nucleus_p=0.9`
2. Reduce gamma: `gamma=3.0`
3. Increase temperature: `temperature=1.2`

### Issue: Invalid SELFIES
**Solution**:
1. Enable forbidden token filtering
2. Check tokenizer vocab
3. Verify padding handling

---

## Summary

The DLM codebase provides a complete framework for:
1. **Training** discrete diffusion models on molecular sequences
2. **Training** noise-conditioned classifiers for guidance
3. **Generating** targeted molecules with classifier-based guidance
4. **Evaluating** on antimicrobial property prediction tasks

**Key Files to Start With**:
1. `main.py` - Training entry point
2. `diffusion.py` - Core diffusion + guidance
3. `classifier.py` - Classifier wrapper
4. `models/dit.py` - All model architectures
5. `configs/guidance/cbg_antibiotic.yaml` - Guidance config

**Next Steps**:
1. Read through `diffusion.py` to understand sampling loop
2. Study `_cbg_denoise_antibiotic_remdm_loop()` for guidance logic
3. Examine `DITClassifier_AMP` for feature extraction
4. Experiment with configs in `configs/guidance/`

Good luck with your ML interview preparation! 🚀
