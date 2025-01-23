import pandas as pd
import numpy as np
from base.utils import my_pearsonr, my_spearmanr


def read_scores(directory):
    """
    Function to read scores from all the replicates of the trained model (10 random seeds)
    """
    scores = []
    pearsons = []
    spearmans = []
    SEEDS = [42, 23, 47, 56, 89, 12, 34, 78, 67, 98]
    for seed in SEEDS:
        file_path = os.path.join(directory, str(seed), "score_board.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)
            df.columns = ["RunID", "Seq_Enc", "LossFn", "Optimizer", "LRScheduler", "ModelSize", "Training duration", "r", "rho"]
            r = df["r"].values[0]
            rho = df["rho"].values[0]
            pearsons.append(r)
            spearmans.append(rho)
            scores.append(r + rho)
    return pearsons, spearmans, scores

def getFinalScores(base_path="testing_results_tr0.9"):
    """
    Function to collate all the results from all models (Camformer: O, L, S, Mi; LegNet)
    Return for both public and private test sets
    """
    df_O = pd.read_csv(f"{base_path}/challenge_testset_final_models/9_onehot_L1_AdamW_ReduceLROnPlateau/score_board.csv")
    df_L = pd.read_csv(f"{base_path}/challenge_testset_final_models/14_onehot_Huber_AdamW_OneCycleLR/score_board.csv")
    df_S = pd.read_csv(f"{base_path}/challenge_testset_final_models/141_onehotWithP_Huber_AdamW_ReduceLROnPlateau/score_board.csv")
    df_Mi = pd.read_csv(f"{base_path}/challenge_testset_final_models/219_onehotWithP_MSE_AdamW_ReduceLROnPlateau/score_board.csv")
    df_Leg = pd.read_csv("/home/dash01/SBLab/LegNet/scripts/dream2022/score_board.csv")

    df_f_O = pd.read_csv(f"{base_path}/challenge_testset_final_models/9_onehot_L1_AdamW_ReduceLROnPlateau/score_board_f.csv")
    df_f_L = pd.read_csv(f"{base_path}/challenge_testset_final_models/14_onehot_Huber_AdamW_OneCycleLR/score_board_f.csv")
    df_f_S = pd.read_csv(f"{base_path}/challenge_testset_final_models/141_onehotWithP_Huber_AdamW_ReduceLROnPlateau/score_board_f.csv")
    df_f_Mi = pd.read_csv(f"{base_path}/challenge_testset_final_models/219_onehotWithP_MSE_AdamW_ReduceLROnPlateau/score_board_f.csv")
    df_f_Leg = pd.read_csv("/home/dash01/SBLab/LegNet/scripts/dream2022/score_board_f.csv")

    df_O["Model"] = "Orig"
    df_L["Model"] = "Large"
    df_S["Model"] = "Small"
    df_Mi["Model"] = "Mini"
    df_Leg["Model"] = "Leg"

    df_f_O["Model"] = "Orig"
    df_f_L["Model"] = "Large"
    df_f_S["Model"] = "Small"
    df_f_Mi["Model"] = "Mini"
    df_f_Leg["Model"] = "Leg"

    df = pd.concat([df_O, df_L, df_S, df_Mi, df_Leg]).reset_index(drop=True)
    df_f = pd.concat([df_f_O, df_f_L, df_f_S, df_f_Mi, df_f_Leg]).reset_index(drop=True)

    df.sort_values(["Model", "Seed"], ignore_index=False, inplace=True)
    df_f.sort_values(["Model", "Seed"], ignore_index=False, inplace=True)
    
    return df, df_f