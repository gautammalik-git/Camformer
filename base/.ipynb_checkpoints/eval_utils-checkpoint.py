"""
Official evaluation of the Camformer models

Source: Most parts of this code is taken from Zenodo repository.
See the README.md file for more details.
"""

#Libraries
import os
import numpy as np
import pandas as pd
import random
import json
from collections import OrderedDict
import csv
from scipy.stats import pearsonr, spearmanr
import json
from base.utils import printConfig


#Some utility functions
def calculate_correlations(index_list, expressions, GROUND_TRUTH_EXPR):
    PRED_DATA = OrderedDict()
    GROUND_TRUTH = OrderedDict()

    for j in index_list:
        PRED_DATA[str(j)] = float(expressions[j])
        GROUND_TRUTH[str(j)] = float(GROUND_TRUTH_EXPR[j])

    pearson = pearsonr(list(GROUND_TRUTH.values()), list(PRED_DATA.values()))[0]
    spearman = spearmanr(list(GROUND_TRUTH.values()), list(PRED_DATA.values()))[0]

    return pearson, spearman


def calculate_diff_correlations(pair_list, expressions, GROUND_TRUTH_EXPR):
    Y_pred_selected = []
    expressions_selected = []

    for pair in pair_list:
        ref, alt = pair[0], pair[1]
        Y_pred_selected.append(expressions[alt] - expressions[ref])
        expressions_selected.append(GROUND_TRUTH_EXPR[alt] - GROUND_TRUTH_EXPR[ref])

    Y_pred_selected = np.array(Y_pred_selected)
    expressions_selected = np.array(expressions_selected)

    pearson = pearsonr(expressions_selected, Y_pred_selected)[0]
    spearman = spearmanr(expressions_selected, Y_pred_selected)[0]

    return pearson, spearman


def calculate_eval_metrics(target_file="OfficialEval/filtered_test_data_with_MAUDE_expression.txt",
                           pred_file=None,
                           private_eval=False,
                           verbose=False):
    """
    Take a predicted expression file (seq, expr) per row, and compute the evaluation metrics.

    target_file : The target expression file (e.g. experimentally obtained)
    pred_file   : The predicted expression file
    private_eval: Is it for private leaderboard? Final test!
    """
    if not pred_file or not os.path.exists(pred_file):
        print(f"Error: {pred_file} doesn't exist!")
        return "nothing"

    with open(target_file) as f:
        reader = csv.reader(f, delimiter="\t")
        lines = list(reader)

    filtered_tagged_sequences = [line[0] for line in lines]
    expressions = [line[1] for line in lines]

    GROUND_TRUTH_EXPR = np.array([float(expressions[i]) for i in range(len(filtered_tagged_sequences))])
    
    if verbose:
        print(f"No. of ground truth expressions: {len(GROUND_TRUTH_EXPR)}")
        print(f"Max value of ground truth expr : {max(GROUND_TRUTH_EXPR)}")
        print(f"Max value of ground truth expr : {min(GROUND_TRUTH_EXPR)}")

    #Different promoter classes IDs (private/test set)
    df = pd.read_csv("OfficialEval/test_subset_ids/high_exp_seqs.csv")
    high = list(df["pos"])
    high = np.unique(np.array(high))

    df = pd.read_csv("OfficialEval/test_subset_ids/low_exp_seqs.csv")
    low = list(df["pos"])
    low = np.unique(np.array(low))

    df = pd.read_csv("OfficialEval/test_subset_ids/yeast_seqs.csv")
    yeast = list(df["pos"])
    yeast = np.unique(np.array(yeast))

    df = pd.read_csv("OfficialEval/test_subset_ids/all_random_seqs.csv")
    random = list(df["pos"])
    random = np.unique(np.array(random))

    df = pd.read_csv("OfficialEval/test_subset_ids/challenging_seqs.csv")
    challenging = list(df["pos"])
    challenging = np.unique(np.array(challenging))

    df = pd.read_csv("OfficialEval/test_subset_ids/all_SNVs_seqs.csv")
    SNVs_alt = list(df["alt_pos"])
    SNVs_ref = list(df["ref_pos"])
    SNVs = list(set(list(zip(SNVs_alt, SNVs_ref))))

    df = pd.read_csv("OfficialEval/test_subset_ids/motif_perturbation.csv")
    motif_perturbation_alt = list(df["alt_pos"])
    motif_perturbation_ref = list(df["ref_pos"])
    motif_perturbation = list(set(list(zip(motif_perturbation_alt, motif_perturbation_ref))))

    df = pd.read_csv("OfficialEval/test_subset_ids/motif_tiling_seqs.csv")
    motif_tiling_alt = list(df["alt_pos"])
    motif_tiling_ref = list(df["ref_pos"])
    motif_tiling = list(set(list(zip(motif_tiling_alt, motif_tiling_ref))))

    if verbose:
        print(len(high))
        print(len(low))
        print(len(yeast))
        print(len(random))
        print(len(challenging))
        print(len(motif_perturbation))
        print(len(motif_tiling))
        print(len(SNVs))


    #Public leader board (promoter class IDs)
    with open("OfficialEval/public_leaderboard_ids/high_exp_indices.json", "r") as f:
        public_high = [int(index) for index in list(json.load(f).keys())]

    with open("OfficialEval/public_leaderboard_ids/low_exp_indices.json", "r") as f:
        public_low = [int(index) for index in list(json.load(f).keys())]

    with open("OfficialEval/public_leaderboard_ids/yeast_exp_indices.json", "r") as f:
        public_yeast = [int(index) for index in list(json.load(f).keys())]

    with open("OfficialEval/public_leaderboard_ids/random_exp_indices.json", "r") as f:
        public_random = [int(index) for index in list(json.load(f).keys())]

    with open("OfficialEval/public_leaderboard_ids/challenging_exp_indices.json", "r") as f:
        public_challenging = [int(index) for index in list(json.load(f).keys())]
        
    with open("OfficialEval/public_leaderboard_ids/SNVs_exp_indices.json", "r") as f:
        public_SNVs = [(int(index.split(",")[0]), int(index.split(",")[1])) for index in list(json.load(f).keys())]

    with open("OfficialEval/public_leaderboard_ids/motif_perturbation_exp_indices.json", "r") as f:
        public_motif_perturbation = [(int(index.split(",")[0]), int(index.split(",")[1])) for index in list(json.load(f).keys())]

    with open("OfficialEval/public_leaderboard_ids/motif_tiling_exp_indices.json", "r") as f:
        public_motif_tiling = [(int(index.split(",")[0]), int(index.split(",")[1])) for index in list(json.load(f).keys())]

    public_ids = {}
    public_ids["high"] = public_high
    public_ids["low"] = public_low
    public_ids["yeast"] = public_yeast
    public_ids["random"] = public_random
    public_ids["challenging"] = public_challenging
    public_ids["SNVs"] = public_SNVs
    public_ids["motif_perturbation"] = public_motif_perturbation
    public_ids["motif_tiling"] = public_motif_tiling

    #Private leader board (promoter class IDs)
    public_single_indices = public_high + public_low + public_yeast + public_random + public_challenging
    public_double_indices = public_SNVs + public_motif_perturbation + public_motif_tiling

    public_indices = []

    for index in public_double_indices:
        public_indices.append(index[0])
        public_indices.append(index[1])

    for index in public_single_indices:
        public_indices.append(index)

    public_indices = list(set(public_indices))

    public_ids["all"] = public_indices

    #Public leaderboard evaluation
    expressions = []
    with open(pred_file) as f:
        lines = f.readlines()

    for j in range(len(lines)):
        exp = lines[j].split("\t")[1].split("\n")[0]
        expressions.append(float(exp))

    expressions = np.array(expressions)

    def get_eval_scores(ids):
        """
        A local function to compute and return the evaluation metrics
        """
        # Calculate correlations
        pearson, spearman = calculate_correlations(ids["all"], expressions, GROUND_TRUTH_EXPR)
        high_pearson, high_spearman = calculate_correlations(ids["high"], expressions, GROUND_TRUTH_EXPR)
        low_pearson, low_spearman = calculate_correlations(ids["low"], expressions, GROUND_TRUTH_EXPR)
        yeast_pearson, yeast_spearman = calculate_correlations(ids["yeast"], expressions, GROUND_TRUTH_EXPR)
        random_pearson, random_spearman = calculate_correlations(ids["random"], expressions, GROUND_TRUTH_EXPR)
        challenging_pearson, challenging_spearman = calculate_correlations(ids["challenging"], expressions, GROUND_TRUTH_EXPR)

        # Calculate difference correlations
        SNVs_pearson, SNVs_spearman = calculate_diff_correlations(ids["SNVs"], expressions, GROUND_TRUTH_EXPR)
        motif_perturbation_pearson, motif_perturbation_spearman = calculate_diff_correlations(ids["motif_perturbation"], expressions, GROUND_TRUTH_EXPR)
        motif_tiling_pearson, motif_tiling_spearman = calculate_diff_correlations(ids["motif_tiling"], expressions, GROUND_TRUTH_EXPR)

        # Calculate scores
        pearsons_score = (pearson**2 + 0.3 * high_pearson**2 + 0.3 * low_pearson**2 + 0.3 * yeast_pearson**2 + 
                          0.3 * random_pearson**2 + 0.5 * challenging_pearson**2 + 1.25 * SNVs_pearson**2 + 
                          0.3 * motif_perturbation_pearson**2 + 0.4 * motif_tiling_pearson**2) / 4.65


        spearmans_score = (spearman + 0.3 * high_spearman + 0.3 * low_spearman + 0.3 * yeast_spearman 
                           + 0.3 * random_spearman + 0.5 * challenging_spearman + 1.25 * SNVs_spearman
                           + 0.3 * motif_perturbation_spearman + 0.4 * motif_tiling_spearman) / 4.65
        
        # Update eval_config
        variable_names = ["pearson", "spearman", "high_pearson", "low_pearson", "yeast_pearson", "random_pearson", "challenging_pearson", 
                          "SNVs_pearson", "motif_perturbation_pearson", "motif_tiling_pearson",
                          "high_spearman", "low_spearman", "yeast_spearman", "random_spearman", "challenging_spearman",
                          "SNVs_spearman", "motif_perturbation_spearman", "motif_tiling_spearman", "pearsons_score", "spearmans_score"]

        econfig = {}
        for variable_name in variable_names:
            econfig[variable_name] = locals()[variable_name]
        econfig["all_pearson_r2"] = pearson**2

        return econfig
        
    econfig = get_eval_scores(ids=public_ids)

    if verbose:
        print("="*50)
        print(f"\t\tPUBLIC EVALUATION")
        print("="*50)
        print(f"No. of high expression sequences: {len(public_high)}")
        print(f"No. of low expression sequences: {len(public_low)}")
        print(f"No. of yeast expression sequences: {len(public_yeast)}")
        print(f"No. of random expression sequences: {len(public_random)}")
        print(f"No. of challenging expression sequences: {len(public_challenging)}")
        print(f"No. of SNV expression sequences: {len(public_SNVs)}")
        print(f"No. of motif perturbation expression sequences: {len(public_motif_perturbation)}")
        print(f"No. of motif tiling expression sequences: {len(public_motif_tiling)}")
        print(f"No. of all expression sequences: {len(public_ids['all'])}")
        print("-"*50)
        for key, value in econfig.items():
            print(f"{key}: {value:.5f}")
        print("="*50)
    
    econfig_f = {}

    if private_eval:
        """
        Runs only if private evaluation flag is set to TRUE
        """
        final_high = [exp for exp in high if exp not in public_indices]
        final_low = [exp for exp in low if exp not in public_indices]
        final_yeast = [exp for exp in yeast if exp not in public_indices]
        final_random = [exp for exp in random if exp not in public_indices]
        final_challenging = [exp for exp in challenging if exp not in public_indices]
        final_SNVs = [exp for exp in SNVs if exp not in public_double_indices]
        final_motif_perturbation = [exp for exp in motif_perturbation if exp not in public_double_indices]
        final_motif_tiling = [exp for exp in motif_tiling if exp not in public_double_indices]
        final_all = [exp for exp in list(range(len(GROUND_TRUTH_EXPR))) if exp not in public_indices]

        final_ids = {}
        final_ids["all"] = final_all
        final_ids["high"] = final_high
        final_ids["low"] = final_low
        final_ids["yeast"] = final_yeast
        final_ids["random"] = final_random
        final_ids["challenging"] = final_challenging
        final_ids["SNVs"] = final_SNVs
        final_ids["motif_perturbation"] = final_motif_perturbation
        final_ids["motif_tiling"] = final_motif_tiling

        econfig_f = get_eval_scores(ids=final_ids)

        if verbose:
            print("\n")
            print("="*50)
            print(f"\tPRIVATE (FINAL) EVALUATION")
            print("="*50)
            print(f"No. of high expression sequences: {len(final_high)}")
            print(f"No. of low expression sequences: {len(final_low)}")
            print(f"No. of yeast expression sequences: {len(final_yeast)}")
            print(f"No. of random expression sequences: {len(final_random)}")
            print(f"No. of challenging expression sequences: {len(final_challenging)}")
            print(f"No. of SNV expression sequences: {len(final_SNVs)}")
            print(f"No. of motif perturbation expression sequences: {len(final_motif_perturbation)}")
            print(f"No. of motif tiling expression sequences: {len(final_motif_tiling)}")
            print(f"No. of all expression sequences: {len(final_all)}")
            print("-"*50)
            for key, value in econfig_f.items():
                print(f"{key}: {value:.5f}")
            print("="*50)


    return econfig, econfig_f

        
if __name__ == "__main__":
    calculate_eval_metrics(pred_file="./testing_results/data_DREAM2022_test_sequences/seq_predexpr.csv",
                           private_eval=True,
                           verbose=True)