# RPCA Integration for Multiverse World Model

This document describes the **Robust Principal Component Analysis (RPCA)** enhancement to the Multiverse world model, implementing the techniques described in our RPCA research paper.

## 🎯 Overview

The RPCA integration addresses three key challenges in the Multiverse world model:

1. **Cross-view consistency**: Ensuring both camera views show consistent physics/crashes
2. **Long-range context efficiency**: Reducing VRAM usage for extended temporal horizons  
3. **Dataset quality**: Cleaning mis-synchronized or noisy training data

### Key Features

- ✅ **Multiple RPCA algorithms** (Inexact ALM, PCP, Non-convex)
- ✅ **Dual-branch architecture** for separate L/S processing
- ✅ **Memory optimization** with compressed storage
- ✅ **Preprocessing pipeline** integration
- ✅ **Comprehensive loss engineering**
- ✅ **Experiment comparison framework**
- ✅ **Configuration validation**

## 🚀 Quick Start

### 1. Installation

Install additional dependencies:
```bash
pip install -r requirements.txt  # Updated with RPCA dependencies
```

### 2. Enable RPCA

Update your configuration in `config/env/racing.yaml`:
```yaml
rpca:
  enabled: true                    # Enable RPCA preprocessing
  method: "inexact_alm"           # RPCA algorithm
  lambda_coeff: null              # Auto-compute regularization
  cache_dir: "rpca_cache"         # Cache decompositions
```

### 3. Run Training

Use the enhanced training script:
```bash
# Standard training with RPCA
python src/main_rpca.py

# Or use the original main with automatic RPCA detection
python src/main.py
```

### 4. Run Experiments

Compare RPCA vs baseline performance:
```bash
# Quick comparison
python run_rpca_experiments.py --data /path/to/processed/data --experiments baseline rpca_default

# Full experiment suite  
python experiments/rpca_comparison.py --data /path/to/processed/data
```

## 📁 Project Structure

```
multiverse/
├── src/
│   ├── rpca/                    # RPCA core implementation
│   │   ├── core.py             # RPCA algorithms (ALM, PCP, etc.)
│   │   ├── decomposition.py    # Frame processing & loss functions
│   │   ├── losses.py           # Advanced loss engineering
│   │   ├── memory_optimization.py  # Memory-efficient processing
│   │   └── utils.py            # Utility functions
│   ├── data/
│   │   ├── rpca_processor.py   # Data pipeline integration
│   │   ├── rpca_dataset.py     # Enhanced datasets
│   │   └── ...
│   ├── models/
│   │   ├── diffusion/
│   │   │   └── rpca_denoiser.py    # Dual-branch denoiser
│   │   └── rpca_agent.py       # RPCA-enhanced agent
│   ├── rpca_trainer.py         # Enhanced trainer
│   └── main_rpca.py            # RPCA-aware main script
├── experiments/
│   ├── rpca_comparison.py      # Experiment framework
│   └── rpca_comparison_config.yaml
├── tests/
│   └── test_rpca_integration.py    # Comprehensive tests
├── run_rpca_experiments.py     # Experiment runner
├── run_tests.py               # Test runner
└── validate_rpca_config.py    # Config validation
```

## 🔧 Configuration

### Basic RPCA Configuration

```yaml
# config/env/racing.yaml
rpca:
  enabled: true
  method: "inexact_alm"          # Algorithm: inexact_alm, pcp, nonconvex
  lambda_coeff: null             # Auto-compute: 1/sqrt(max(m,n))
  temporal_mode: true            # (pixels,time) vs (time,pixels) matrix
  fusion_method: "concat"        # Dual-branch fusion: concat, add, attention
  
  # Memory optimization
  cache_dir: "rpca_cache"
  max_cache_size: 1000
  enable_parallel: true
  
  # Loss function weights
  loss_weights:
    lambda_lowrank: 1.0          # Low-rank component weight
    lambda_sparse: 1.0           # Sparse component weight  
    lambda_consistency: 0.1      # L + S = D consistency
    beta_nuclear: 0.01           # Nuclear norm regularization
```

### Advanced Configuration

```yaml
rpca:
  # Cross-view consistency (for dual-camera)
  cross_view_mode: "stack_channels"    # stack_channels, separate
  
  # Memory optimization  
  compression_threshold: 0.01
  use_compressed_storage: true
  
  # Streaming/online processing
  streaming:
    enabled: false
    rank: 10
    forgetting_factor: 0.95
    
  # Progressive training
  progressive:
    enabled: false
    warmup_epochs: 10
    schedule: "linear"             # linear, cosine
```

## 🧪 RPCA Algorithms

### 1. Inexact ALM (Default)
Fast and practical implementation suitable for most use cases.
```python
rpca = InexactALM(lambda_coeff=0.1, max_iter=1000, tol=1e-7)
L, S = rpca.fit_transform(D)
```

### 2. Principal Component Pursuit (PCP)
Exact convex optimization with theoretical guarantees.
```python
rpca = PrincipalComponentPursuit(lambda_coeff=0.1, max_iter=500)
L, S = rpca.fit_transform(D)
```

### 3. Non-convex Factorization
Faster for large matrices with known/estimated rank.
```python
rpca = NonconvexRPCA(rank=10, lambda_coeff=0.1, max_iter=1000)
L, S = rpca.fit_transform(D)
```

### 4. Streaming RPCA
Online processing for real-time applications.
```python
streaming = StreamingRPCAProcessor(rank=10, ambient_dim=1024)
for observation in data_stream:
    L_part, S_part = streaming.update(observation)
```

## 🏗️ Model Architecture

### Dual-Branch Processing

The RPCA-enhanced denoiser uses a dual-branch architecture:

```
Input Frames
     ↓
  RPCA Decomposition
   /        \
  L          S
  ↓          ↓
Shared     Sparse
Backbone   Head
  ↓          ↓
   \        /
    Fusion Module
        ↓
   Output Frames
```

### Key Components

1. **Shared Backbone**: Processes low-rank component (global structure)
2. **Sparse Head**: Lightweight network for sparse component (events)  
3. **Fusion Module**: Combines L and S features (concat/add/attention)

## 📊 Loss Engineering

### RPCA Loss Components

```python
loss = (λ_lr * MSE(pred_L, target_L) +           # Low-rank reconstruction
        λ_sp * L1(pred_S, target_S) +            # Sparse reconstruction
        λ_cons * MSE(pred_L + pred_S, target) +  # Consistency
        β * nuclear_norm(pred_L) +               # Rank regularization
        γ * sparsity_penalty(pred_S))           # Sparsity promotion
```

### Progressive Training

Gradually transitions from standard to RPCA loss:
```python
total_loss = (1-α) * standard_loss + α * rpca_loss
# where α increases from 0 to 1 over warmup_epochs
```

## 💾 Memory Optimization

### Compressed Storage

RPCA decompositions are stored efficiently:
- **Low-rank**: Store only U, Σ, V^T factors
- **Sparse**: Store only non-zero indices and values
- **Typical compression**: 5-20x memory reduction

### Memory-Efficient Processing

```python
# Automatic chunking for large batches
processor = MemoryEfficientRPCABatch(max_memory_mb=1000)
L_batch, S_batch = processor.process_batch_frames(frames, rpca_processor)
```

## 🔬 Experiments & Benchmarking

### Running Comparisons

```bash
# Quick test with key experiments
python run_rpca_experiments.py \
    --data /path/to/processed/data \
    --experiments baseline rpca_default rpca_lambda_0p1 \
    --quick

# Full experiment suite
python experiments/rpca_comparison.py \
    --data /path/to/processed/data \
    --output ./experiment_results
```

### Available Experiments

- `baseline`: Original Multiverse (no RPCA)
- `rpca_default`: RPCA with default settings
- `rpca_fusion_add`: Additive fusion instead of concatenation
- `rpca_lambda_X`: Different λ values (0.01, 0.1, 0.5)
- `rpca_weighted_*`: Different loss weight configurations

### Metrics Collected

- **Performance**: PSNR, SSIM, final loss, convergence time
- **Efficiency**: Memory usage, inference FPS, compression ratio
- **Quality**: Decomposition quality, nuclear norm, sparsity

## 🧪 Testing

### Run Test Suite

```bash
# Fast tests only
python run_tests.py

# All tests including slow integration
python run_tests.py --all

# Specific test categories
python run_tests.py --unit
python run_tests.py --integration

# Performance benchmarks
python run_tests.py --benchmark

# With coverage reporting
python run_tests.py --coverage --html-report
```

### Test Categories

- **Unit Tests**: RPCA algorithms, loss functions, utilities
- **Integration Tests**: Data pipeline, model integration, end-to-end
- **Performance Tests**: Memory usage, speed benchmarks
- **Configuration Tests**: Config validation, error handling

## 🔧 Configuration Validation

Validate your RPCA configuration:
```bash
# Validate current config
python validate_rpca_config.py

# Validate specific file
python validate_rpca_config.py --config my_config.yaml

# Auto-fix common issues
python validate_rpca_config.py --fix-common-issues

# Create example config
python validate_rpca_config.py --create-example rpca_example.yaml
```

## 📈 Performance Tips

### Memory Optimization
- Use `compression_enabled=True` for large datasets
- Set appropriate `max_cache_size` based on available RAM
- Consider `temporal_mode=True` for video sequences

### Speed Optimization  
- Use `method="inexact_alm"` for best speed/quality tradeoff
- Enable `enable_parallel=True` for multi-core processing
- Use smaller `batch_size` if memory-constrained

### Quality Optimization
- Tune `lambda_coeff` based on your data's sparsity
- Use `fusion_method="attention"` for best quality (slower)
- Adjust `loss_weights` based on your priorities

## 🐛 Troubleshooting

### Common Issues

**Issue**: Out of memory during RPCA processing
```yaml
# Solution: Reduce batch size or enable memory optimization
rpca:
  max_cache_size: 100
  compression_threshold: 0.05
denoiser:
  training:
    batch_size: 8
```

**Issue**: Poor decomposition quality
```yaml  
# Solution: Tune lambda coefficient
rpca:
  lambda_coeff: 0.01  # Lower for more low-rank emphasis
  # or
  lambda_coeff: 0.5   # Higher for more sparsity emphasis
```

**Issue**: Slow convergence
```yaml
# Solution: Use different algorithm or adjust tolerance
rpca:
  method: "nonconvex"  # Faster for large matrices
  # or adjust ALM parameters in code
```

### Debug Mode

Enable verbose logging:
```bash
RPCA_DEBUG=1 python src/main_rpca.py
```

## 📚 References

- **RPCA Theory**: Candès et al. "Robust Principal Component Analysis?" JACM 2011
- **Inexact ALM**: Lin et al. "The Augmented Lagrange Multiplier Method" 2010  
- **Applications**: [Our RPCA Multiverse Paper](rpca_multiverse.md)

## 🤝 Contributing

1. Follow the existing code structure in `src/rpca/`
2. Add comprehensive tests for new features
3. Update configuration schemas as needed
4. Run the full test suite before submitting changes

## 📄 License

This RPCA integration maintains the same license as the base Multiverse project.