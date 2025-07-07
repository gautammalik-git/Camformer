#!/usr/bin/env python3
"""
Camformer Training and Testing Script (Original Model with Residual Connections)

This script trains the original Camformer model (from base/model.py) on training data and tests it on test data.
The original model includes residual connections between convolutional blocks.

Performance Metrics:
- Primary: Pearson correlation (r) and Pearson R² (variance explained)

Usage:
    python large_camformer.py --train_data train_data.txt --test_data test_data.txt --output_dir results/

"""

import argparse
import os
import time
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# Import Camformer utilities
from base.utils import (
    createDataLoaders
)
from base.model_utils import train, predict, getNumParams
from base.run_utils import setSeeds
from base.model import CNN  # Import the Camformer model with residual connections


def get_default_config():
    """Get default configuration for Camformer training with residual connections."""
    return {
        # Data parameters
        "seq_enc": "onehot",
        "target_len": 110,
        "N_tolerance": 3,
        "margin": 3,
        
        # Model parameters (Camformer architecture with residual connections)
        "model_args": {
            "feature_height": 110,
            "feature_width": 1,
            "batch_size": 256,
            "input_channels": 4,
            "out_channels": [512, 512, 512, 512, 512, 512],  # 6 layers = 3 residual blocks
            "kernels": [(10, 1), (10, 1), (10, 1), (10, 1), (10, 1), (10, 1)],  
            "pool_kernels": [(1, 1), (1, 1), (1, 1), (1, 1), (10, 1), (1, 1)],  
            "paddings": "same",  
            "strides": [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],  # Must match out_channels length (6)
            "pool_strides": [(1, 1), (1, 1), (1, 1), (1, 1), (4, 1), (1, 1)],  
            "dropouts": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],  
            "linear_output": [256, 256],  
            "linear_dropouts": [0.0, 0.0]  # Must match linear_output length
        },
        
        # Training parameters
        "epochs": 50,
        "patience": 10,
        "lr": 0.001,
        "weight_decay": 0.001,
        "batch_size": 256,
        "loss": "L1",
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        
        # Data split parameters (using stratified splits)
        "train_prop": 0.9,  # 90% training 
        "val_prop": 0.05,   # 5% validation
        "test_prop": 0.05,  # 5% test
        
        # Random seed
        "seed": 42,
        
        # Output parameters
        "output_dir": "camformer_large_results",
        "save_model": True,
        "save_predictions": True
    }


def validate_model_config(config):
    """Validate that the model configuration is compatible with residual connections."""
    model_args = config["model_args"]
    
    # Check that out_channels has even length for residual blocks
    if len(model_args["out_channels"]) % 2 != 0:
        raise ValueError(f"out_channels must have even length for residual connections. Got {len(model_args['out_channels'])}")
    
    # Check that all parameter lists have the same length
    param_lists = [
        ("out_channels", model_args["out_channels"]),
        ("kernels", model_args["kernels"]),
        ("pool_kernels", model_args["pool_kernels"]),
        ("strides", model_args["strides"]),
        ("pool_strides", model_args["pool_strides"]),
        ("dropouts", model_args["dropouts"])
    ]
    
    lengths = [len(param_list) for _, param_list in param_lists]
    if len(set(lengths)) > 1:
        raise ValueError(f"All model parameter lists must have the same length. Got lengths: {dict(param_lists)}")
    
    # Check that paddings is "same" for residual connections
    if model_args["paddings"] != "same":
        raise ValueError(f"Original Camformer model requires 'same' padding for residual connections. Got {model_args['paddings']}")
    
    print(f"Model configuration validated: {len(model_args['out_channels'])} layers in {len(model_args['out_channels'])//2} residual blocks")


def load_data(train_file, test_file=None, verbose=False):
    """Load training and test data."""
    if verbose:
        print(f"Loading training data from: {train_file}")
    
    train_df = pd.read_csv(train_file, delimiter='\t', header=None, names=['seq', 'expr'])
    if verbose:
        print(f"Training data shape: {train_df.shape}")
    
    test_df = None
    if test_file and os.path.exists(test_file):
        if verbose:
            print(f"Loading test data from: {test_file}")
        test_df = pd.read_csv(test_file, delimiter='\t', header=None, names=['seq', 'expr'])
        if verbose:
            print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df


def prepare_data_splits(train_df, test_df, config, verbose=False):
    """Prepare data splits for training, validation, and testing using stratified splits."""
    if test_df is not None:
        if verbose:
            print("Using provided test data")
        train_prop = config["train_prop"] / (config["train_prop"] + config["val_prop"])
        
        # Try stratified split first, fallback to random split if it fails
        try:
            from base.utils import make_stratified_splits
            X_temp, y_temp, X_val, y_val, X_test_temp, y_test_temp, bins = make_stratified_splits(
                X=train_df[["seq"]], y=train_df["expr"], 
                trProp=train_prop, seed=config["seed"]
            )
            if verbose:
                print("Using stratified splits for train/val")
        except ValueError as e:
            if verbose:
                print(f"Stratified split failed: {e}")
                print("Falling back to random splits")
            # Fallback to random split
            from sklearn.model_selection import train_test_split
            X_temp, X_val, y_temp, y_val = train_test_split(
                train_df[["seq"]], train_df["expr"], 
                test_size=(1-train_prop), random_state=config["seed"], shuffle=True
            )
        
        train_data = pd.DataFrame({'seq': X_temp['seq'], 'expr': y_temp})
        val_data = pd.DataFrame({'seq': X_val['seq'], 'expr': y_val})
        test_data = test_df
    else:
        if verbose:
            print("Splitting training data into train/val/test using stratified splits")
        
        # Try stratified splits first, fallback to random split if it fails
        try:
            from base.utils import make_stratified_splits
            X_train, y_train, X_val, y_val, X_test, y_test, bins = make_stratified_splits(
                X=train_df[["seq"]], y=train_df["expr"], 
                trProp=config["train_prop"], seed=config["seed"]
            )
            if verbose:
                print("Using stratified splits for train/val/test")
        except ValueError as e:
            if verbose:
                print(f"Stratified split failed: {e}")
                print("Falling back to random splits")
            # Fallback to random split
            from sklearn.model_selection import train_test_split
            # First split: train vs val+test
            X_temp, X_val_test, y_temp, y_val_test = train_test_split(
                train_df[["seq"]], train_df["expr"], 
                test_size=(1-config["train_prop"]), random_state=config["seed"], shuffle=True
            )
            # Second split: val vs test
            X_val, X_test, y_val, y_test = train_test_split(
                X_val_test, y_val_test, 
                test_size=0.5, random_state=config["seed"], shuffle=True
            )
            X_train, y_train = X_temp, y_temp
        
        train_data = pd.DataFrame({'seq': X_train['seq'], 'expr': y_train})
        val_data = pd.DataFrame({'seq': X_val['seq'], 'expr': y_val})
        test_data = pd.DataFrame({'seq': X_test['seq'], 'expr': y_test})
    
    if verbose:
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data


def create_data_loaders(train_data, val_data, test_data, config, verbose=False):
    """Create PyTorch DataLoaders with drop_last=False for val and test to avoid empty loaders."""
    if verbose:
        print("Creating data loaders...")
    
    config["input_file"] = "temp_data.txt"
    
    # Use the existing createDataLoaders function - this already encodes the data once
    TrainLoader, ValLoader, TestLoader = createDataLoaders(
        seq_tr=train_data[["seq"]], expr_tr=train_data["expr"],
        seq_va=val_data[["seq"]], expr_va=val_data["expr"],
        seq_te=test_data[["seq"]], expr_te=test_data["expr"],
        config=config
    )
    
    # Modify the val and test loaders to use drop_last=False without re-encoding
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Get the tensors from the existing loaders and create new ones with drop_last=False
    val_dataset = ValLoader.dataset
    test_dataset = TestLoader.dataset
    
    ValLoader = DataLoader(dataset=val_dataset, 
                          num_workers=0, 
                          batch_size=config["batch_size"], 
                          shuffle=True, 
                          drop_last=False)
    
    TestLoader = DataLoader(dataset=test_dataset, 
                           num_workers=0, 
                           batch_size=config["batch_size"], 
                           shuffle=True, 
                           drop_last=False)
    
    return TrainLoader, ValLoader, TestLoader


def create_model(config, verbose=False):
    """Create and initialize the Camformer model with residual connections."""
    if verbose:
        print("Creating Camformer model with residual connections...")
    
    # Validate model configuration
    validate_model_config(config)
    
    model = CNN(**config["model_args"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    

    from base.model import initialize_weights_he
    model.apply(initialize_weights_he)
    
    num_params = getNumParams(model, print=False)
    config["model_size"] = f"{num_params/1e6:.1f}M"
    if verbose:
        print(f"Model parameters: {config['model_size']}")
        print(f"Residual blocks: {len(config['model_args']['out_channels'])//2}")
    
    return model, device


def train_model(model, TrainLoader, ValLoader, config, device, verbose=False):
    """Train the original Camformer model."""
    print("Starting model training...")
    
    model_dir = Path(config["output_dir"]) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    config["best_model_path"] = str(model_dir / "best_model.pt")
    config["device"] = device
    
    start_time = time.time()
    trained_model = train(TrainLoader=TrainLoader, model=model, 
                         config=config, ValLoader=ValLoader)
    training_time = time.time() - start_time
    
    config["training_time"] = training_time
    if verbose:
        print(f"Training completed in {training_time:.1f} seconds")
    
    return trained_model


def test_model(model, TestLoader, config, device, verbose=False):
    """Test the trained model on test data."""
    if verbose:
        print("Testing model...")
    
    # Fix for PyTorch 2.6 compatibility, add weights_only=False
    best_model = torch.load(config["best_model_path"], weights_only=False)
    best_model.to(device)
    best_model.eval()

    start_time = time.time()
    
    # Use the original expr values from test_data instead of y_true from DataLoader
    # since the DataLoader shuffles the data and causes order mismatch
    from base.utils import createDataLoader
    test_loader_no_shuffle = createDataLoader(
        seq=config["test_data"]["seq"].to_frame(), 
        expr=config["test_data"]["expr"], 
        config=config, 
        isTest=True, 
        drop=False
    )
    
    # Get predictions in the correct order (matching original test data)
    y_pred_ordered, y_true_ordered = predict(model=best_model, DataLoader=test_loader_no_shuffle, config=config)
    
    # Use the y_true_ordered from the DataLoader to ensure same length
    y_true_original = y_true_ordered
    
    test_time = time.time() - start_time
    
    # Calculate Pearson correlation and R² as primary performance metrics
    pearson_result = pearsonr(y_true_original, y_pred_ordered)
    pearson_r = pearson_result[0]  # Extract correlation coefficient
    pearson_r2 = pearson_r ** 2
    
    # Convert numpy values to Python native types for JSON serialization
    metrics = {
        "pearson_r": float(pearson_r),
        "pearson_r2": float(pearson_r2),
        "test_time": test_time
    }
    
    if verbose:
        print(f"Test completed in {test_time:.1f} seconds")
    
    # Always print performance metrics (these are essential)
    print(f"Primary Performance Metrics:")
    print(f"  Pearson correlation (r): {pearson_r:.4f}")
    print(f"  Pearson R² (variance explained): {pearson_r2:.4f}")
    
    return y_pred_ordered, y_true_original, metrics


def save_results(train_data, val_data, test_data, predictions, true_values, metrics, config, verbose=False):
    """Save training results, predictions, and metrics."""
    if verbose:
        print("Saving results...")
    
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a JSON-serializable version of the config
    config_serializable = {}
    for key, value in config.items():
        if isinstance(value, torch.device):
            config_serializable[key] = str(value)
        elif isinstance(value, (int, float, str, bool, list, dict)):
            config_serializable[key] = value
        else:
            # Skip non-serializable objects
            config_serializable[key] = str(type(value))
    
    # Save configuration
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_serializable, f, indent=2)
    
    # Save predictions using original expr values
    if config["save_predictions"]:
        if verbose:
            print(f"Debug: test_data length: {len(test_data)}, predictions length: {len(predictions)}")
        
        # Create predictions DataFrame - predictions and true_values are already aligned
        pred_df = pd.DataFrame({
            'seq': test_data["seq"].iloc[:len(predictions)],
            'expr': test_data["expr"].iloc[:len(predictions)],
            'predicted': predictions,
            'true': true_values
        })
        
        pred_df.to_csv(output_dir / "predictions.csv", index=False)
        
        if verbose:
            print(f"Predictions saved with {len(pred_df)} samples")
    
    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save data splits
    train_data.to_csv(output_dir / "train_data.csv", index=False)
    val_data.to_csv(output_dir / "val_data.csv", index=False)
    test_data.to_csv(output_dir / "test_data.csv", index=False)
    
    # Save summary with primary performance metrics
    summary = {
        "model_size": config["model_size"],
        "training_time": config["training_time"],
        "test_time": metrics["test_time"],
        "primary_metrics": {
            "pearson_r": metrics["pearson_r"],
            "pearson_r2": metrics["pearson_r2"],
        },
        "data_info": {
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "test_samples_processed": len(predictions)
        },
        "model_info": {
            "residual_blocks": len(config["model_args"]["out_channels"])//2,
            "total_layers": len(config["model_args"]["out_channels"])
        }
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"Results saved to: {output_dir}")


def run_training_pipeline(train_file, test_file=None, config=None, verbose=False):
    """Run the complete training and testing pipeline."""
    if verbose:
        print("=" * 60)
        print("ORIGINAL CAMFORMER TRAINING AND TESTING PIPELINE (WITH RESIDUAL CONNECTIONS)")
        print("=" * 60)
    
    # Initialize configuration
    if config is None:
        config = get_default_config()
    
    # Set device and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    setSeeds(config["seed"])
    
    if verbose:
        print(f"Using device: {device}")
        print(f"Random seed: {config['seed']}")
    
    # Load data
    train_df, test_df = load_data(train_file, test_file, verbose)
    
    # Prepare data splits
    train_data, val_data, test_data = prepare_data_splits(train_df, test_df, config, verbose)
    
    # Store test_data in config for access in test_model
    config["test_data"] = test_data
    
    # Create data loaders
    TrainLoader, ValLoader, TestLoader = create_data_loaders(train_data, val_data, test_data, config, verbose)
    
    # Create model
    model, device = create_model(config, verbose)
    
    # Train model
    trained_model = train_model(model, TrainLoader, ValLoader, config, device, verbose)
    
    # Test model
    predictions, true_values, metrics = test_model(trained_model, TestLoader, config, device, verbose)
    
    # Save results
    save_results(train_data, val_data, test_data, predictions, true_values, metrics, config, verbose)
    
    if verbose:
        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description="Train and test large Camformer model with residual connections (16.6M parameters)")
    
    parser.add_argument("--train_data", required=True, 
                       help="Path to training data file (tab-separated: sequence, expression)")
    parser.add_argument("--test_data", 
                       help="Path to test data file (optional, if not provided will split from train data)")
    parser.add_argument("--output_dir", default="camformer_large_results", 
                       help="Output directory for results")
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose output (show all details)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Update config with command line arguments
    config["output_dir"] = args.output_dir
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    config["seed"] = args.seed
    
    # Run training pipeline
    run_training_pipeline(args.train_data, args.test_data, config, args.verbose)


if __name__ == "__main__":
    main() 
