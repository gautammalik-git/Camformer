"""
Test and score all the trained Camformer models and replicates (models constructing with training data) 
using public and private test set from the challenge

Date: 13 Feb 2024
"""

import os
import time
import torch
import pandas as pd

from base.utils import createDataLoader, loadConfig, createDir, my_pearsonr, my_spearmanr, writeScoresToCSV
from base.run_utils import setSeeds, saveDict
from base.model_utils import predict, getNumParams
from base.eval_utils import calculate_eval_metrics

from base.model import CNN #for Large/Small/Original models (Residual-CNN)
#from base.model_basic import CNN #for Mini models (Simple-CNN)


train_res_basepath = "training_results_tr0.9" #Training results (models, configs, results, etc.) base directory
test_res_basepath = "testing_results_tr0.9" #Testing results base directory

#List all the trained model directories
parent_dir = f"./{train_res_basepath}/9_onehot_L1_AdamW_ReduceLROnPlateau" #Camformer Original (submission model)
#parent_dir = f"./{train_res_basepath}/14_onehot_Huber_AdamW_OneCycleLR" #Camformer Large
#parent_dir = f"./{train_res_basepath}/141_onehotWithP_Huber_AdamW_ReduceLROnPlateau" #Camformer Small
#parent_dir = f"./{train_res_basepath}/219_onehotWithP_MSE_AdamW_ReduceLROnPlateau" #Camformer Mini
#parent_dir = f"./{train_res_basepath}/189_onehot_L1_AdamW_ReduceLROnPlateau" #Mini configuration similar to original model

model_dirs = [f"{parent_dir}/{dir}" for dir in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, dir))]

#Parent directory for result files
res_dir = f"./{test_res_basepath}/challenge_testset_final_models/{os.path.basename(parent_dir)}"

#Loop through all the models in tuning_results directory
for dir in model_dirs:
    if os.path.isfile(f"{dir}/model.pt"):
        config_dir = dir
        config_path = f"{config_dir}/config.json"
        config = loadConfig(file_path=config_path)

        config["input_file"] = "./data/DREAM2022/test_sequences.txt"

        #Set all random seeds
        setSeeds(config["seed"])
        
        #This is just to get an identifier associated with the score file (RunID has a different meaning; tuning model ID)
        config["RunID"] = config["seed"]

        #Load data and extract sequence and expression
        df = pd.read_csv(config["input_file"], delimiter='\t', header=None)
        df.columns = ["seq","expr"]

        #Extract sequences and expression
        seq = df[["seq"]]
        expr = df["expr"]

        #Create a DataLoader object for the test sequences
        TestLoader = createDataLoader(seq=seq, expr=expr, config=config)

        #Load the specified model
        model_args = config["model_args"]
        loaded_model = CNN(**model_args)
        loaded_model = torch.load(config["best_model_path"])
        config["model_size"] = f"{getNumParams(loaded_model, print=False)/1e6}M"
        loaded_model.to(config["device"])
        start = time.time()
        y_pred, y_true = predict(model=loaded_model, DataLoader=TestLoader, config=config)
        print(f"\nPrediction completed. Duration: {time.time() - start:.0f}s.")

        config["test_r"] = my_pearsonr(y_true, y_pred)
        config["test_rho"] = my_spearmanr(y_true, y_pred)
        print(f"Test scores: (r = {config['test_r']:.4f}, rho = {config['test_rho']:.4f})")

        #Save the predictions per sequence
        df.drop(columns=["expr"], inplace=True)
        df["pred"] = y_pred
        dirname = f"{res_dir}/{os.path.basename(dir)}"
        createDir(directory_path=dirname)
        df.to_csv(f"{dirname}/seq_predexpr.csv", index=False, header=False, sep="\t")
        print(f"Predictions saved inside: {dirname}.")

        #Evaluate the public leaderboard standing
        config["econfig"], config["econfig_f"] = calculate_eval_metrics(pred_file=f"{dirname}/seq_predexpr.csv", private_eval=True)
        saveDict(config, dirname)

        #Save to CSV
        writeScoresToCSV(file_path=f"{res_dir}/score_board.csv", config=config)
        writeScoresToCSV(file_path=f"{res_dir}/score_board_f.csv", config=config, is_private=True)
