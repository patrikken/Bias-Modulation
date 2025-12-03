# How to Run train.py

## Overview
`train.py` is the main training script for the Bias Modulation project. It uses PyTorch Lightning's CLI interface to train models on fairness datasets with various debiasing methods.

## Prerequisites

### 1. Install Dependencies
Ensure you have all required packages installed:
```bash
pip install -r requirement.txt
```

Key dependencies include:
- `lightning[pytorch-extra]` - PyTorch Lightning framework
- `torch` - PyTorch for model training
- `scikit-learn` - For data preprocessing and metrics
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `fairlearn` - Fairness metrics
- `optuna` - Hyperparameter optimization

### 2. Prepare Data
Ensure your datasets are in the `data/` directory. Supported datasets include: 
- `heart` - Heart disease dataset
- `mimic` - MIMIC-III dataset 

## Basic Usage

### Minimal Example
```bash
python train.py --config configs/adult/1hot.yaml
```

### With Custom Parameters
```bash
python train.py \
  --config configs/bias_modulation.yaml \
  --data_name adult \
  --method_name bias_modulation \
  --output_dir ./results \
  --save_weights true
```

## Configuration Files

The project uses YAML configuration files located in the `configs/` directory. Each config specifies:

### Key Configuration Parameters

**Model Parameters:**
- `input_dim` - Input feature dimension (set automatically from datamodule)
- `output_dim` - Number of output classes
- `hidden_layers` - List of hidden layer sizes (e.g., `[32, 32]`)
- `dropout` - Dropout rate for regularization
- `activation` - Activation function (e.g., `relu`)
- `learning_rate` - Learning rate for optimization
- `weight_decay` - L2 regularization weight
- `nb_groups` - Number of sensitive attribute groups

**Data Parameters:**
- `batch_size` - Batch size for training
- `num_workers` - Number of parallel data loading workers
- `sensitive_attr` - Sensitive attributes for fairness (can be intersectional)
- `is_1hot` - Whether to use one-hot encoding for sensitive attributes
- `is_intersectional` - Whether to use intersectional fairness approach
- `is_decoupled` - Whether to use decoupled training

**Trainer Parameters:**
- `max_epochs` - Maximum number of training epochs
- `accelerator` - `cpu` or `gpu` 
- `devices` - Number of devices to use
- `check_val_every_n_epoch` - Validation frequency
- `log_every_n_steps` - Logging frequency
- `callbacks` - List of training callbacks (e.g., ModelCheckpoint, EarlyStopping)

### Example Configuration Structure
```yaml
seed_everything: 42
method_name: bias_modulation
output_dir: results/
model:
  class_path: models.bias_modulation.BiasModulated
  init_args:
    output_dim: 2
    hidden_layers: [32, 32]
    dropout: 0.5
    learning_rate: 0.001
    weight_decay: 0.0001
data:
  class_path: dataloaders.adult.Adult
  init_args:
    batch_size: 32
    num_workers: 4
    sensitive_attr: ['sex']
trainer:
  max_epochs: 500
  accelerator: cpu
```

## Available Methods

The following debiasing methods are available via configuration:

1. **`1hot`** - One-hot encoding baseline
2. **`bias_modulation`** - Bias Modulation method
3. **`dcp`** - Decoupled Training approach
4. **`intersec`** - Intersectional fairness
5. **`multitask`** - Multi-task learning approach

## Command-Line Arguments

You can override configuration settings via command-line arguments:

```bash
python train.py \
  --config <config_path> \
  --data_name <dataset_name> \
  --method_name <method_name> \
  --output_dir <output_directory> \
  --save_weights <true|false>
```

### Arguments
- `--config` - Path to YAML configuration file (required)
- `--data_name` - Dataset name (default: `adult`)
- `--method_name` - Debiasing method (default: `1hot`)
- `--output_dir` - Directory to save results (default: `./results`)
- `--save_weights` - Whether to save model weights (default: `false`)

## Special Cases

### Bias Modulation Model
When using the `BiasModulated` model, the script automatically:
1. Trains a demographic predictor first (if not already trained)
2. Caches the demographic predictor weights
3. Uses these weights to initialize the bias modulation model

The demographic predictor checkpoint is saved to:
```
<output_dir>/<sensitive_attrs>/demographic_predictor/results/models/best_<seed>.ckpt
```

## Output Structure

Training outputs are saved to the directory specified in the configuration:

```
<output_dir>/
├── <sensitive_attrs>/
│   ├── models/
│   │   └── best_<seed>.ckpt          # Trained model weights
│   └── results/
│       ├── metrics_<seed>.csv        # Test metrics
│       └── results_<seed>.csv        # Detailed results
```

## Logging

The script logs training information to the console with the following format:
```
YYYY-MM-DD HH:MM:SS,XXX - LEVEL - Message
```

Logs include:
- Training progress
- Validation metrics
- Model checkpoints
- Test results
- Demographic predictor training (if applicable)

## GPU Training

To enable GPU training, modify the configuration:
```yaml
trainer:
  accelerator: gpu
  devices: 1  # or number of GPUs to use
```

Or pass it as a command-line argument:
```bash
python train.py --config configs/bias_modulation.yaml \
  --trainer.accelerator gpu \
  --trainer.devices 1
```

## Reproducibility

All training runs use a seed for reproducibility. The seed is set in the configuration:
```yaml
seed_everything: 42
```

This ensures:
- Random number generator reproducibility
- Same data splits across runs
- Deterministic model initialization

## Troubleshooting

**Issue: "Input dimension mismatch"**
- The input dimension is automatically determined from the datamodule. Remove `input_dim` from manual config, or ensure it matches your data.

**Issue: "Demographic predictor not found"**
- For BiasModulated models, the demographic predictor is trained automatically. Ensure write permissions to the output directory.

**Issue: Out of memory (OOM)**
- Reduce `batch_size` in the data configuration
- Reduce `hidden_layers` sizes
- Use gradient accumulation in the trainer configuration

**Issue: Slow training on CPU**
- Consider using `trainer.accelerator: gpu` if CUDA is available
- Reduce `num_workers` in data configuration

## Examples

### Train Bias Modulation on Adult Dataset
```bash
python train.py --config configs/bias_modulation.yaml \
  --data_name adult \
  --output_dir ./results/adult_bm
```

### Train with Intersectional Fairness
```bash
python train.py --config configs/intersec.yaml \
  --data_name adult \
  --method_name intersec
```

### Train on Heart Dataset
```bash
python train.py --config configs/heart/bias_modulation.yaml \
  --output_dir ./results/heart_bm \
  --save_weights true
```

### Train with Custom Hyperparameters
```bash
python train.py --config configs/bias_modulation.yaml \
  --model.init_args.hidden_layers [64,32] \
  --model.init_args.learning_rate 0.0001 \
  --trainer.max_epochs 300
```

## Advanced Features

### Multi-GPU Training
```bash
python train.py --config configs/bias_modulation.yaml \
  --trainer.accelerator gpu \
  --trainer.devices 2,3  # Use GPUs 2 and 3
```

### Custom Sensitive Attributes
Modify the configuration file or use command-line arguments:
```bash
python train.py --config configs/bias_modulation.yaml \
  --data.init_args.sensitive_attr '[sex, race]'
```

## Related Scripts

- `test.py` - Testing and evaluation script
- `hyperparameter_search.py` - Hyperparameter optimization using Optuna
- `train_decoupled.py` - Specialized script for decoupled training
- `aggregate_results.py` - Aggregate results from multiple runs
- `run.sh` - Bash script to run full experiments

## More Information

For detailed model documentation, see:
- `models/mlp.py` - Base MLP model
- `models/bias_modulation.py` - Bias Modulation implementation
- `dataloaders/` - Dataset loaders for different datasets
