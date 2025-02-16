"""
Date: Jan 25, 2025
This directory contains Camformer models: Original (Model ID: 9) and Mini (Model ID: 219).

Step 0: Unzip the files: `tar -xvzf Camformer_ID.tar.gz`
Step 1: Clone the Camformer repo from: https://github.com/Bornelov-lab/Camformer
Step 2: Move `trainining_results_tr0.9` and `demo_test.*` directory to the main repo path.
Step 3: Run this file. It should. If it doesn't check your file paths.
"""

# Load libraries
import time
import pandas as pd
import torch

from base.model_utils import predict, getNumParams
from base.utils import createDataLoader, my_pearsonr, my_spearmanr, loadConfig
from base.run_utils import setSeeds

from base.model import CNN # For ID: 9
#from base.model_basic import CNN # For ID: 219


# Load data
df = pd.read_csv("demo_test.txt", header=None, delimiter="\t", names=["seq","expr"])
seq, expr = df[["seq"]], df["expr"]

# Load model
config_path = f"./training_results_tr0.9/9_onehot_L1_AdamW_ReduceLROnPlateau/42/config.json"
config = loadConfig(file_path=config_path)
setSeeds(config["seed"])
TestLoader = createDataLoader(seq=seq, expr=expr, config=config, drop=False)

model_args = config["model_args"]
loaded_model = CNN(**model_args)
loaded_model = torch.load(config["best_model_path"])
config["model_size"] = f"{getNumParams(loaded_model, print=False)/1e6}M"
loaded_model.to(config["device"])
start = time.time()

# Make prediction
y_pred, y_true = predict(model=loaded_model, DataLoader=TestLoader, config=config)
print(f"\nPrediction completed. Duration: {time.time() - start:.0f}s.")

config["test_r"] = my_pearsonr(y_true, y_pred)
config["test_rho"] = my_spearmanr(y_true, y_pred)
print(f"Test scores: (r = {config['test_r']:.4f}, rho = {config['test_rho']:.4f})\n")

