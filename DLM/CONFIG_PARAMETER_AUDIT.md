# Config Parameter Cross-Reference Audit Report
**DLM Directory Analysis**
**Date**: 2025-11-25
**Scope**: All Python files in `/home/user/ApexOracle/DLM/`

---

## Executive Summary

This audit identifies config parameters referenced in Python code that are missing from YAML configuration files. The analysis found **4 critical missing parameter categories** and **10 experiment-specific missing parameters** that need to be addressed.

**Overall Assessment**: NEEDS ATTENTION
**Priority**: Address critical missing parameters before production use

---

## 1. CRITICAL MISSING PARAMETERS

These parameters are accessed in code WITHOUT defaults and are NOT found in any config files:

### 1.1 `classifier_backbone`
- **Severity**: HIGH
- **Files**: `/home/user/ApexOracle/DLM/classifier.py` (line 184, 187, 190, 193, 196)
- **Usage**: `config.classifier_backbone == 'dit'`
- **Purpose**: Selects classifier architecture (dit, dit_AMP, dit_reg_cls_AMP, dit_synergy_cls_AMP, hyenadna)
- **Impact**: Will cause AttributeError if not provided
- **Fix**: Add to main config.yaml or create separate classifier config

### 1.2 `classifier_model.*` (Complete Section Missing)
- **Severity**: HIGH
- **Files**: `/home/user/ApexOracle/DLM/classifier.py`, `/home/user/ApexOracle/DLM/models/dit.py`
- **Missing Subparameters**:
  - `classifier_model.hidden_size` - Model hidden dimension
  - `classifier_model.cond_dim` - Conditioning dimension
  - `classifier_model.n_blocks` - Number of transformer blocks
  - `classifier_model.n_heads` - Number of attention heads
  - `classifier_model.dropout` - Dropout rate
  - `classifier_model.scale_by_sigma` - Whether to scale by sigma
  - `classifier_model.num_classes` - Number of output classes
  - `classifier_model.hyena_model_name_or_path` - HyenaDNA model path
  - `classifier_model.n_layer` - Number of layers (for HyenaDNA)
- **Purpose**: Required when using classifier-based guidance (CBG)
- **Impact**: Will cause AttributeError when initializing classifier models
- **Fix**: Create `configs/classifier_model/` directory with architecture configs (similar to `configs/model/`)

### 1.3 `data.num_classes`
- **Severity**: HIGH
- **Files**: `/home/user/ApexOracle/DLM/classifier.py` (lines 224, 225), `/home/user/ApexOracle/DLM/models/dit.py`
- **Usage**: Used for multi-class classification metrics and model initialization
- **Impact**: Will cause AttributeError in classifier training/inference
- **Fix**: Add `num_classes` field to data config files for classification tasks

### 1.4 `data.label_col`
- **Severity**: MEDIUM
- **Files**: `/home/user/ApexOracle/DLM/classifier.py` (lines 505-509)
- **Usage**: `hasattr(self.config.data, 'label_col')` - checked before use
- **Purpose**: Specifies which column contains labels in the dataset
- **Impact**: Code has hasattr check, but should be defined for classification datasets
- **Fix**: Add `label_col` to data config files for supervised learning tasks

---

## 2. EXPERIMENT-SPECIFIC MISSING PARAMETERS

These parameters are used in specialized model classes for antibiotic and synergy experiments. They are only needed when using specific classifier backbones (dit_reg_cls_AMP, dit_synergy_cls_AMP):

### 2.1 Antibiotic MIC Optimization Parameters
Used in `DIT_Reg_Cls_AMP` class (`/home/user/ApexOracle/DLM/models/dit.py`, line 670-671):

- **`sampling.target_MIC`**
  - Purpose: Target MIC (Minimum Inhibitory Concentration) value for optimization
  - Used in: comp_reg_log() method

- **`sampling.target_MIC_max`**
  - Purpose: Maximum MIC value for scaling during generation
  - Used in: comp_reg_log() method to compute target schedule

- **`sampling.gaussian_sigma_min`**
  - Purpose: Minimum Gaussian sigma for noisy regression guidance
  - Used in: comp_reg_log() method for noise scheduling

- **`sampling.gaussian_sigma_max`**
  - Purpose: Maximum Gaussian sigma for noisy regression guidance
  - Used in: comp_reg_log() method for noise scheduling

### 2.2 Genome/Strain Conditioning Parameters
Used in `DIT_Reg_Cls_AMP.load_genome_test_embedding()` (`/home/user/ApexOracle/DLM/models/dit.py`, lines 646-658):

- **`sampling.genome_test_emb_dir_path.ATCC_genome_emb`**
  - Purpose: Path to ATCC bacterial genome embeddings
  - Used in: load_genome_test_embedding()

- **`sampling.genome_test_emb_dir_path.ATCC_text_emb`**
  - Purpose: Path to ATCC text description embeddings
  - Used in: load_genome_test_embedding()

- **`sampling.genome_test_emb_dir_path.only_text_emb`**
  - Purpose: Path to text-only embeddings (no genome)
  - Used in: load_genome_test_embedding()

- **`sampling.strain`**
  - Purpose: Bacterial strain identifier for conditional generation
  - Used in: load_genome_test_embedding()

### 2.3 Synergy Experiment Parameters
Used in `DIT_Syn_Cls_Pep_Cls_AMP` class:

- **`sampling.synergy_mol_emb_dict_path`**
  - Purpose: Path to synergy molecule embedding dictionary
  - Used in: Synergy prediction model

- **`sampling.synergy_mol_name`**
  - Purpose: Name of synergy molecule for conditional generation
  - Used in: Synergy prediction model

**Recommendation**: Create experiment-specific config files:
- `configs/experiments/antibiotic_mic.yaml`
- `configs/experiments/synergy_prediction.yaml`

---

## 3. PARAMETERS WITH DEFAULTS (OK)

These parameters are checked with `getattr()` or `hasattr()` and have default values, so they're safe to be missing:

| Parameter | Default Value | Location |
|-----------|---------------|----------|
| `config.sampling.batch_size` | `config.loader.eval_batch_size` | diffusion.py:1143 |
| `config.sampling.use_cache` | `False` | diffusion.py:1258 |
| `config.sampling.use_float64` | `False` | diffusion.py:1297 |
| `config.training.use_label_smoothing` | `False` | classifier.py:513 |
| `config.is_fudge_classifier` | `False` | classifier.py:519 |
| `config.classifier_model.pooling` | `"mean"` | models/dit.py:516 |

---

## 4. CONFIG-RELATED ISSUES

### 4.1 Runtime Config Modification (Anti-Pattern)
**File**: `/home/user/ApexOracle/DLM/diffusion.py` (lines 1765-1784)

**Issue**: Code directly modifies config object at runtime:
```python
self.config.guidance.reg_guide_weight = 1.0
self.config.guidance.cls_guide_weight = 0.0
```

**Impact**:
- Config values are overwritten during sampling
- Makes it hard to track actual values used
- Violates immutability principle
- Can cause confusion when debugging

**Recommendation**: Use local variables instead:
```python
# Instead of modifying config
reg_guide_weight = 1.0
cls_guide_weight = 0.0
# Use these local variables throughout the method
```

### 4.2 Missing `loader.persistent_workers`
**File**: `/home/user/ApexOracle/DLM/classifier.py` (line 327)

**Issue**: Code accesses `self.config.loader.persistent_workers` but it's not defined in `config.yaml`

**Current Workaround**: Other files (diffusion.py, dataloader.py) hardcode `persistent_workers=True`

**Fix**: Add to `config.yaml`:
```yaml
loader:
  persistent_workers: true
```

---

## 5. RECOMMENDATIONS

### Immediate Actions (High Priority)

1. **Create `classifier_model` config directory**
   ```bash
   mkdir -p configs/classifier_model
   ```
   Create architecture configs similar to `configs/model/`:
   - `configs/classifier_model/dit.yaml`
   - `configs/classifier_model/dit_amp.yaml`
   - `configs/classifier_model/hyenadna.yaml`

2. **Add `classifier_backbone` to main config**
   Add to `configs/config.yaml`:
   ```yaml
   classifier_backbone: dit  # dit / dit_AMP / dit_reg_cls_AMP / dit_synergy_cls_AMP / hyenadna
   ```

3. **Add `data.num_classes` to classification datasets**
   Update relevant data configs (e.g., for binary/multi-class classification):
   ```yaml
   # configs/data/classification_dataset.yaml
   num_classes: 2  # or appropriate number
   label_col: "MIC_threshold"  # or appropriate column name
   ```

4. **Add `loader.persistent_workers`**
   Add to `configs/config.yaml`:
   ```yaml
   loader:
     persistent_workers: true
   ```

### Medium Priority Actions

5. **Create experiment-specific configs**
   - `configs/experiments/antibiotic_mic.yaml` - with all MIC optimization parameters
   - `configs/experiments/synergy.yaml` - with synergy-specific parameters

6. **Refactor runtime config modification**
   In `diffusion.py`, replace config modifications with local variables

7. **Add config validation**
   Implement validation that checks for required parameters based on:
   - `config.mode` (train / ppl_eval / sample_eval)
   - `config.backbone` (dit / ar / etc.)
   - `config.classifier_backbone` (when using classifier guidance)

### Example Config Structure

Suggested new config structure:
```
configs/
├── config.yaml                    # Main config (add classifier_backbone, loader.persistent_workers)
├── classifier_model/              # NEW: Classifier architecture configs
│   ├── dit.yaml
│   ├── dit_amp.yaml
│   └── hyenadna.yaml
├── experiments/                   # NEW: Experiment-specific configs
│   ├── antibiotic_mic.yaml        # Antibiotic MIC optimization
│   └── synergy.yaml               # Synergy prediction
├── data/
│   ├── SELFIES.yaml              # UPDATE: Add num_classes, label_col if needed
│   └── ...
└── ...
```

---

## 6. DETAILED PARAMETER MAPPING

### Parameters by File

#### `/home/user/ApexOracle/DLM/classifier.py`
- MISSING: `classifier_backbone` (line 184, 187, 190, 193, 196)
- MISSING: `classifier_model.*` (lines 198-199, 205)
- MISSING: `data.num_classes` (lines 224-225)
- MISSING: `data.label_col` (lines 505-509, has hasattr check)
- MISSING: `loader.persistent_workers` (line 327)

#### `/home/user/ApexOracle/DLM/models/dit.py`
- MISSING: `classifier_model.*` (multiple locations)
- MISSING: `sampling.target_MIC_max` (line 670)
- MISSING: `sampling.gaussian_sigma_min` (line 671)
- MISSING: `sampling.gaussian_sigma_max` (line 671)
- MISSING: `sampling.genome_test_emb_dir_path.*` (lines 646-654)
- MISSING: `sampling.strain` (line 658)
- MISSING: `sampling.synergy_mol_emb_dict_path`
- MISSING: `sampling.synergy_mol_name`

#### `/home/user/ApexOracle/DLM/diffusion.py`
- ISSUE: Runtime modification of `config.guidance.reg_guide_weight` (lines 1765, 1774, 1783)
- ISSUE: Runtime modification of `config.guidance.cls_guide_weight` (lines 1766, 1775, 1784)

---

## 7. TESTING RECOMMENDATIONS

To verify config completeness:

1. **Test with classifier mode**
   ```bash
   python main.py classifier_backbone=dit
   ```
   Should fail if classifier_model config is missing.

2. **Test with antibiotic guidance**
   ```bash
   python main.py +experiments=antibiotic_mic
   ```
   Should fail if MIC optimization parameters are missing.

3. **Add config validation function**
   ```python
   def validate_config(config):
       if config.mode == 'train' and hasattr(config, 'classifier_backbone'):
           assert hasattr(config, 'classifier_model'), "classifier_model config required"
       # Add more validations
   ```

---

## 8. SUMMARY STATISTICS

- **Total Python files analyzed**: 15
- **Total config files analyzed**: 42 YAML files
- **Unique config parameter accesses**: 113
- **Parameters defined in configs**: 241
- **Critical missing parameters**: 4 main categories (11 individual params)
- **Experiment-specific missing**: 10 parameters
- **Parameters with defaults (OK)**: 6 parameters
- **Config-related issues**: 2 issues

---

## Appendix: Search Commands Used

To reproduce this analysis:
```bash
# Find all config accesses in Python files
grep -r "config\." DLM/*.py DLM/*/*.py

# Check if parameter exists in configs
grep -r "classifier_model" configs/

# Find getattr usage (parameters with defaults)
grep -r "getattr.*config" DLM/
```
