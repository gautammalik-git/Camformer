# Camformer Unified Command-Line Interface for Pre-training

This unified interface makes it easy to train and test both basic and large Camformer models using simple commands.

## Quick Start

```bash
# Train basic model
python camformer.py model_basic --train_data data.txt

# Train large model with residual connections
python camformer.py model_large --train_data data.txt

# Train with verbose output
python camformer.py model_basic --train_data data.txt -v
```

## Available Models

### `model_basic`
- **Architecture**: Basic Camformer (no residual connections)
- **Encoding**: `onehotWithP` (5 input channels)
- **Layers**: 6 convolutional layers with varying channel sizes
- **Parameters**: ~1.4M parameters
- **Script**: `base_camformer.py`

### `model_large`
- **Architecture**: Large Camformer with residual connections
- **Encoding**: `onehot` (4 input channels)
- **Layers**: 6 convolutional layers with residual blocks
- **Parameters**: ~16.6M parameters
- **Script**: `large_camformer.py`

## Usage Examples

### Basic Usage
```bash
# Train basic model with default settings
python camformer.py model_basic --train_data train_data.txt

# Train large model with test data
python camformer.py model_large --train_data train_data.txt --test_data test_data.txt
```

### Advanced Usage
```bash
# Train with custom parameters
python camformer.py model_basic --train_data data.txt --epochs 100 --batch_size 128 --lr 0.0001

# Train with custom output directory
python camformer.py model_large --train_data data.txt --output_dir my_results

# Train with verbose output
python camformer.py model_basic --train_data data.txt -v
```

### Getting Help
```bash
# Show general help
python camformer.py

# Show help for specific model
python camformer.py model_basic --help
python camformer.py model_large --help
```

## Command Line Arguments

Both models support the same arguments:

- `--train_data`: Path to training data file (required)
- `--test_data`: Path to test data file (optional)
- `--output_dir`: Output directory for results (default: model-specific)
- `--epochs`: Number of training epochs (default: 50 for basic, 50 for large)
- `--batch_size`: Batch size (default: 256 for both)
- `--lr`: Learning rate (default: 0.001 for both)
- `--seed`: Random seed (default: 42)
- `-v, --verbose`: Enable verbose output

## Data Format

Training and test data should be tab-separated files with two columns:
- Column 1: DNA sequence (string)
- Column 2: Expression value (float)

Example:
```
ATCGATCGATCG	1.234
GCTAGCTAGCTA	0.567
```

## Output

Both models create the same output structure:
- `config.json`: Model configuration
- `metrics.json`: Performance metrics
- `summary.json`: Training summary
- `predictions.csv`: Model predictions
- `train_data.csv`, `val_data.csv`, `test_data.csv`: Data splits
- `models/best_model.pt`: Best trained model

## Performance Metrics

Both models report:
- **Pearson correlation (r)**: Linear correlation between predicted and actual values
- **Pearson R²**: Variance explained by the model

## Requirements

- Python 3.7+
- PyTorch
- pandas
- scikit-learn
- scipy
- numpy

## Troubleshooting

1. **Script not found**: Make sure you're running from the Camformer directory
2. **Data format error**: Ensure your data is tab-separated with sequence and expression columns
3. **CUDA errors**: The script automatically falls back to CPU if CUDA is not available
4. **Memory issues**: Try reducing batch size with `--batch_size 64`

