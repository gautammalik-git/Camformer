"""
Train a Camformer model with some known configuration with different seeds (replicates)

Date: 21 January 2024
"""

#imports
import time
import os

import pandas as pd
import torch

from base.utils import make_stratified_splits, createDir, createDataLoaders, draw_histogram
from base.utils import my_pearsonr, my_spearmanr, writeToCSV, loadConfig
from base.model_utils import train, predict, getNumParams
from base.run_utils import setSeeds, saveDict

#from base.model import CNN #for Large/Small/Original models (Residual-CNN)
from base.model_basic import CNN #for Mini model (Simple-CNN)


#Array of random seeds (42 is the seed used, when evaluating a single model)
SEEDS = [42, 23, 47, 56, 89, 12, 34, 78, 67, 98]

#Load the config variable (based on top-k performers from hyperparameter tuning)
#config_dir = "tuning_results/9_onehot_L1_AdamW_ReduceLROnPlateau" #The original submission model for the DREAM 2022 challenge
#config_dir = "tuning_results/14_onehot_Huber_AdamW_OneCycleLR" #Best Large configuration (Rank: 1st; Picked)
#config_dir = "tuning_results/141_onehotWithP_Huber_AdamW_ReduceLROnPlateau" #Best Small configuration (Rank: 3rd; Picked)
#config_dir = "tuning_results/219_onehotWithP_MSE_AdamW_ReduceLROnPlateau" #Best Mini configuration (Rank: Lower; Picked)
config_dir = "tuning_results/189_onehot_L1_AdamW_ReduceLROnPlateau" #Mini configuration similar config to original model

train_res_basepath = "training_results_tr0.9" #Training results (models, configs, results, etc.) base directory

#Loop through all seeds
for seed in SEEDS:
    print(f"Training Camformer with seed: {seed}.")

    config_path = f"{config_dir}/config.json"

    config = loadConfig(file_path=config_path)
    model_args = config["model_args"]

    #Set all random seeds
    config["seed"] = seed
    setSeeds(config["seed"])

    #Load data and prepare [train, val, test] splits for hyperparameter tuning
    df = pd.read_csv(config["input_file"], delimiter='\t', header=None)
    df.columns = ['seq','expr']

    seq_tr, expr_tr, seq_va, expr_va, seq_te, expr_te, bins = make_stratified_splits(X=df[["seq"]], y=df["expr"], 
                                                                                     trProp=0.9,
                                                                                     seed=config["seed"])

    #Draw histograms of the expression in each split (train, val, test); It will only work on Jupyter
    #draw_histogram(y_train=expr_tr, y_val=expr_va, y_test=expr_te, bins=bins)

    #Where to save the training model and results?
    dir_name = os.path.basename(config_dir)
    res_dir = f"./{train_res_basepath}/{dir_name}/{config['seed']}"
    createDir(directory_path=res_dir, remove_existing=False)

    #Create a summary file (delete the existing score_board.csv)
    score_board = f"{res_dir}/score_board.csv"
    if os.path.exists(score_board):
        os.remove(score_board)

    #Create the DataLoaders (uses the chosen seq_encoding)
    TrainLoader, ValLoader, TestLoader = createDataLoaders(seq_tr, expr_tr, seq_va, expr_va, seq_te, expr_te, config)

    config["best_model_path"] = f"{res_dir}/model.pt"
                    
    #Initialise the model
    model = CNN(**model_args)
    model.to(config["device"]) # Move model to device
    config["model_size"] = f"{getNumParams(model, print=False)/1e6}M"
    print(f"Number of Parameters: {config['model_size']}")

    #Train model (uses TrainLoader, ValLoader)
    print("Model Training started.")
                    
    start = time.time()
    model = train(TrainLoader=TrainLoader, model=model, config=config, ValLoader=ValLoader)
    config["time_taken"] = time.time() - start
    print(f"\nTraining completed. Duration: {config['time_taken']:.0f}s.")

    #Test the trained model on the test split (TestLoader)
    best_model = CNN(**model_args)
    best_model = torch.load(config["best_model_path"])
    best_model.to(config["device"])
    start = time.time()
    y_pred, y_true = predict(model=best_model, DataLoader=TestLoader, config=config)
    print(f"\nPrediction completed. Duration: {time.time() - start:.0f}s.")

    config["test_r"] = my_pearsonr(y_true, y_pred)
    config["test_rho"] = my_spearmanr(y_true, y_pred)
    print(f"Test scores: (r = {config['test_r']:.4f}, rho = {config['test_rho']:.4f})")

    #Save the test score on the config file (for future use)
    saveDict(config=config, path=res_dir)

    #Write the results to a csv file
    writeToCSV(file_path=f"{res_dir}/score_board.csv", config=config)
                    
    print(f"Training completed. Model and config saved.\n")
