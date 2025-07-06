#!/usr/bin/env python3
"""
Camformer Unified Command-Line Interface

This script provides a unified interface to train and test both basic and large Camformer models.
Users can easily choose between models using simple commands.

Usage:
    python camformer.py model_basic --train_data train_data.txt --test_data test_data.txt
    python camformer.py model_large --train_data train_data.txt --test_data test_data.txt

Available Models:
    model_basic  - Basic Camformer model (no residual connections - 1.4M parameters)
    model_large  - Large Camformer model with residual connections (16.6M parameters)

Examples:
    # Train basic model with verbose output
    python camformer.py model_basic --train_data data.txt -v
    
    # Train large model with custom parameters
    python camformer.py model_large --train_data data.txt --epochs 100 --batch_size 128
    
    # Train basic model with custom output directory
    python camformer.py model_basic --train_data data.txt --output_dir my_results
"""

import sys
import argparse
import subprocess
from pathlib import Path


def get_model_script(model_type):
    """Get the appropriate training script based on model type."""
    if model_type == "model_basic":
        return "base_camformer.py"
    elif model_type == "model_large":
        return "large_camformer.py"
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: model_basic, model_large")


def get_model_description(model_type):
    """Get description for the model type."""
    descriptions = {
        "model_basic": "Basic Camformer model (no residual connections) - 6 conv layers, onehotWithP encoding",
        "model_large": "Large Camformer model with residual connections (16.6M parameters) - 6 conv layers, onehot encoding"
    }
    return descriptions.get(model_type, "Unknown model")


def validate_model_type(model_type):
    """Validate that the model type is supported."""
    valid_models = ["model_basic", "model_large"]
    if model_type not in valid_models:
        print(f"Error: Invalid model type '{model_type}'")
        print(f"Available models: {', '.join(valid_models)}")
        print("\nUsage examples:")
        print("  python camformer.py model_basic --train_data data.txt")
        print("  python camformer.py model_large --train_data data.txt")
        sys.exit(1)


def check_script_exists(script_path):
    """Check if the training script exists."""
    if not Path(script_path).exists():
        print(f"Error: Training script '{script_path}' not found!")
        print("Make sure you're running this from the Camformer directory.")
        sys.exit(1)


def run_training_script(script_path, args):
    """Run the training script with the provided arguments."""
    try:
        # Build the command
        cmd = [sys.executable, script_path] + args
        
        print(f"Running: {' '.join(cmd)}")
        print(f"Model: {get_model_description(script_path.replace('_camformer.py', '').replace('base', 'model_basic').replace('large', 'model_large'))}")
        print("-" * 60)
        
        # Run the script
        result = subprocess.run(cmd, check=True)
        
        print("-" * 60)
        print("Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Training script failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: Could not find Python executable or training script")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)


def main():
    """Main function for the unified Camformer interface."""
    
    # Check if model type is provided
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python camformer.py <model_type> [arguments]")
        print("\nAvailable Models:")
        print("  model_basic  - Basic Camformer model (no residual connections)")
        print("  model_large  - Large Camformer model with residual connections")
        print("\nExamples:")
        print("  python camformer.py model_basic --train_data data.txt")
        print("  python camformer.py model_large --train_data data.txt -v")
        print("\nFor help with specific model arguments:")
        print("  python camformer.py model_basic --help")
        print("  python camformer.py model_large --help")
        sys.exit(1)
    
    # Extract model type and remaining arguments
    model_type = sys.argv[1]
    remaining_args = sys.argv[2:]
    
    # Validate model type
    validate_model_type(model_type)
    
    # Get the appropriate training script
    script_path = get_model_script(model_type)
    
    # Check if script exists
    check_script_exists(script_path)
    
    # If user wants help, show help for the specific model
    if "--help" in remaining_args or "-h" in remaining_args:
        print(f"Help for {model_type}:")
        print(f"Description: {get_model_description(model_type)}")
        print("-" * 60)
        subprocess.run([sys.executable, script_path, "--help"])
        sys.exit(0)
    
    # Run the training script
    run_training_script(script_path, remaining_args)


if __name__ == "__main__":
    main() 