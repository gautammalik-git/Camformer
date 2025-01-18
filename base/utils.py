import warnings
import csv
import json
import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt


def writeToCSV(file_path, config):
    """
    Save a row with few variables to a csv file
    """
    with open(file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header if it's the first iteration
        if config["RunID"] == 1:
            csv_writer.writerow(["RunID", "Seq_Enc", "LossFn", "Optimizer", "LRScheduler", "ModelSize", "Training duration", "r", "rho"])

        csv_writer.writerow([
            config["RunID"], 
            config["seq_enc"], 
            config["loss"], 
            config["optimizer"], 
            config["scheduler"], 
            config["model_size"], 
            config["time_taken"], 
            config["test_r"], 
            config["test_rho"]
        ])

    print(f"Result file {file_path} is updated.\n")


def writeScoresToCSV(file_path, config, is_private=False):
    """
    Save a row with few variables to a csv file: Use this to store evaluation scores
    """
    # Write the header if it's the first iteration
    if config["RunID"] == 1:
        with open(file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            content = list(reader)

        header = ["RunID", "Seq_Enc", "LossFn", "Optimizer", "LRScheduler", "ModelSize", "pearson", "spearman", "high_pearson", 
                  "low_pearson", "yeast_pearson", "random_pearson", "challenging_pearson", "SNVs_pearson", 
                  "motif_perturbation_pearson", "motif_tiling_pearson", "high_spearman", "low_spearman", "yeast_spearman", 
                  "random_spearman", "challenging_spearman", "SNVs_spearman", "motif_perturbation_spearman", 
                  "motif_tiling_spearman", "pearsons_score", "spearmans_score", "all_pearson_r2"]

        content.insert(0, header)

        # Write the updated content back to the CSV file
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(content)

    with open(file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        #If saving the private score? (private test set)
        if is_private:
            econfig = config["econfig_f"]
        else:
            econfig = config["econfig"]

        csv_writer.writerow([
            config["RunID"], 
            config["seq_enc"], 
            config["loss"], 
            config["optimizer"], 
            config["scheduler"], 
            config["model_size"],
            econfig["pearson"],
            econfig["spearman"],
            econfig["high_pearson"],
            econfig["low_pearson"],
            econfig["yeast_pearson"],
            econfig["random_pearson"],
            econfig["challenging_pearson"],
            econfig["SNVs_pearson"],
            econfig["motif_perturbation_pearson"],
            econfig["motif_tiling_pearson"],
            econfig["high_spearman"],
            econfig["low_spearman"],
            econfig["yeast_spearman"],
            econfig["random_spearman"],
            econfig["challenging_spearman"],
            econfig["SNVs_spearman"],
            econfig["motif_perturbation_spearman"],
            econfig["motif_tiling_spearman"],
            econfig["pearsons_score"],
            econfig["spearmans_score"],
            econfig["all_pearson_r2"]
        ])  

    print(f"Result file {file_path} is updated.\n")


def loadConfig(file_path):
    """
    Load a config file (a json file passed to file_path) and return the object
    """
    with open(file_path, 'r') as json_file:
        json_object = json.load(json_file)

    return json_object


def printConfig(file_path):
    """
    Print a config file (a json file passed to file_path); each key in a separate line
    """
    with open(file_path, 'r') as json_file:
        for line in json_file:
            json_object = json.loads(line)
            for key, value in json_object.items():
               print(f"{key}: {value}")


def countDirs(path):
    """
    Count number of model directories present in a path
    """
    entries = os.listdir(path)
    dirs = (entry for entry in entries if os.path.isdir(os.path.join(path, entry)))
    num_dirs = sum(1 for _ in dirs)
    return num_dirs


def createDir(directory_path, remove_existing=True):
    """
    Simple function to create a directory
    """
    if os.path.exists(directory_path) and remove_existing:
        shutil.rmtree(directory_path)
        print(f"Removed existing directory: {directory_path}")

    os.makedirs(directory_path)
    print(f"Created new directory: {directory_path}")


def draw_histogram(y_train, y_val, y_test, bins):
    """
    Draw histogram of distributions of the expressions in each split
    """
    def draw_hist(ax, data, bins, title):
        ax.hist(data, bins=bins, edgecolor='black')
        ax.set_xlabel('bins')
        ax.set_ylabel('Freq')
        ax.set_title(title)

    fig, axes = plt.subplots(1, 3, figsize=(6, 1.5)) 

    draw_hist(axes[0], y_train, bins, 'train')
    draw_hist(axes[1], y_val, bins, 'val')
    draw_hist(axes[2], y_test, bins, 'test')

    plt.tight_layout()
    plt.show()


def my_spearmanr(series1, series2):
    """
    Calculate Spearman correlation. If return nan instead return 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            rho = stats.spearmanr(series1, series2)[0]
            if np.isnan(rho):
                rho = 0
        except:
            rho=0
    return rho


def my_pearsonr(series1, series2):
    """
    Calculate Pearson correlation. If return nan instead return 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = stats.pearsonr(series1, series2)[0]
            if np.isnan(r):
                r = 0
        except:
            r = 0
    return r


def onehote(seq):
    """
    One hot encode DNA sequence.
    If not ACGT, then fill with 0.
    """
    seq_enc = [] # empty list to store the endoded seq.
    mapping = {"A":[1., 0., 0., 0.], 
               "C":[0., 1., 0., 0.], 
               "G":[0., 0., 1., 0.], 
               "T":[0., 0., 0., 1.],
               "N":[0., 0., 0., 0.]}
    for i in seq:
        seq_enc.append(mapping[i])
    return seq_enc


def multihote(seq, isTest=False, onlyN=True, onlyP=False, onlyInt=False, isR=True, both=False):
    """
    Multihot encoding of a sequence with relevant properties.
    seq     : Sequence "ACGTTTAAANN..."
    isTest  : If the sequence (seq) is a part of "test" sequences, then encoding will be done differenyly
    onlyN   : One hot encoding, plus the unknown nucleotide 'N' is encoded as a probability vector
    onlyP   : If not to use integer seq information, but only include purine/pyrimidine information
    onlyInt : If not to use P information; only include information on integer sequence (requires isR to be passed adequately)
    isR     : Include information that a nucleotide belongs to a sequence with integer expression (0) or real-valued expression (1)
    both    : Both P and Int information to be included in the encoding

    Ways to call:
    multihote(seq)
    multihote(seq, onlyN=False, onlyP=True)
    multihote(seq, onlyN=False, onlyInt=True, isR=True/False)
    multihote(seq, onlyN=False, onlyInt=False, both=True, isR=True/False)
    multihote(seq, onlyN=False): This is same as onehote(seq)

    If the sequence is a "test" sequence; only two encoding calls are affected
    multihote(seq, isTest=True, onlyN=False, onlyInt=True, isR=True/False)
    multihote(seq, isTest=True, onlyN=False, onlyInt=False, both=True, isR=True/False)
    """
    seq_enc = [] 
    
    if onlyN: #simple one-hot of 4 nucleotides; N: 1/4 due to uncertainty
        mapping = {"A":[1.,   0.,   0.,   0.], 
                   "C":[0.,   1.,   0.,   0.], 
                   "G":[0.,   0.,   1.,   0.], 
                   "T":[0.,   0.,   0.,   1.],
                   "N":[0.25, 0.25, 0.25, 0.25]}
        for i in seq:
            seq_enc.append(mapping[i])
        return seq_enc
    
    if isTest and both: #5th bit 1
        mapping = {"A":[1.,   0.,   0.,   0.,   1., 1.], 
                   "C":[0.,   1.,   0.,   0.,   1., 0.], 
                   "G":[0.,   0.,   1.,   0.,   1., 1.],
                   "T":[0.,   0.,   0.,   1.,   1., 0.],
                   "N":[0.25, 0.25, 0.25, 0.25, 1., 0.5]}
        for i in seq:
            seq_enc.append(mapping[i])
        return seq_enc
    
    if not isTest and both: #5th and 6th bits
        if not isR: 
            mapping = {"A":[1.,   0.,   0.,   0.,   0., 1.], 
                       "C":[0.,   1.,   0.,   0.,   0., 0.], 
                       "G":[0.,   0.,   1.,   0.,   0., 1.],
                       "T":[0.,   0.,   0.,   1.,   0., 0.],
                       "N":[0.25, 0.25, 0.25, 0.25, 0., 0.5]}
        else: 
            mapping = {"A":[1.,   0.,   0.,   0.,   1., 1.], 
                       "C":[0.,   1.,   0.,   0.,   1., 0.], 
                       "G":[0.,   0.,   1.,   0.,   1., 1.],
                       "T":[0.,   0.,   0.,   1.,   1., 0.],
                       "N":[0.25, 0.25, 0.25, 0.25, 1., 0.5]}
        for i in seq:
            seq_enc.append(mapping[i])
        return seq_enc
    
    if isTest and onlyInt: #5th bit: 1
        mapping = {"A":[1.,   0.,   0.,   0.,   1.], 
                   "C":[0.,   1.,   0.,   0.,   1.], 
                   "G":[0.,   0.,   1.,   0.,   1.],
                   "T":[0.,   0.,   0.,   1.,   1.],
                   "N":[0.25, 0.25, 0.25, 0.25, 1.]}
        for i in seq:
            seq_enc.append(mapping[i])
        return seq_enc
    
    if not isTest and onlyInt: #5th bit: 0/1
        if not isR: 
            mapping = {"A":[1.,   0.,   0.,   0.,   0.], 
                       "C":[0.,   1.,   0.,   0.,   0.], 
                       "G":[0.,   0.,   1.,   0.,   0.],
                       "T":[0.,   0.,   0.,   1.,   0.],
                       "N":[0.25, 0.25, 0.25, 0.25, 0.]}
        else: 
            mapping = {"A":[1.,   0.,   0.,   0.,   1.], 
                       "C":[0.,   1.,   0.,   0.,   1.], 
                       "G":[0.,   0.,   1.,   0.,   1.],
                       "T":[0.,   0.,   0.,   1.,   1.],
                       "N":[0.25, 0.25, 0.25, 0.25, 1.]}
        for i in seq:
            seq_enc.append(mapping[i])
        return seq_enc
    
    if onlyP: #5th bit: 1/0/0.5
        mapping = {"A":[1.,   0.,   0.,   0.,   1.], 
                   "C":[0.,   1.,   0.,   0.,   0.], 
                   "G":[0.,   0.,   1.,   0.,   1.],
                   "T":[0.,   0.,   0.,   1.,   0.],
                   "N":[0.25, 0.25, 0.25, 0.25, 0.5]}
        for i in seq:
            seq_enc.append(mapping[i])
        return seq_enc
    
    if not onlyN:
        mapping = {"A":[1., 0., 0., 0.], 
                   "C":[0., 1., 0., 0.], 
                   "G":[0., 0., 1., 0.], 
                   "T":[0., 0., 0., 1.],
                   "N":[0., 0., 0., 0.]}
        for i in seq:
            seq_enc.append(mapping[i])
        return seq_enc

def onehote_reverse(seq):
    """
    Get back sequence from ohe.
    """
    seq2=[] # empty list to store the endoded seq.
    mapping = {"[1.0, 0.0, 0.0, 0.0]":"A", 
               "[0.0, 1.0, 0.0, 0.0]":"C", 
               "[0.0, 0.0, 1.0, 0.0]":"G", 
               "[0.0, 0.0, 0.0, 1.0]":"T"}
    for i in seq:
        i = str(i)
        seq2.append(mapping[i] if i in mapping.keys() else "N") # If not in the above map, use N
    return seq2


def getSeqFromOnehot(X):
    """
    It takes a tensor X (usually, numpy X) of dimension [1,4,l,1] 
    and returns a sequence of length l
    """
    X = np.squeeze(X)

    nucleotide_mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    seq = ""
    for i in range(X.shape[1]):  
        nucleotide_index = np.argmax(X[:, i])
        nucleotide = nucleotide_mapping[nucleotide_index]
        seq += nucleotide
    return seq


def OHE(input_file, N_tolerance=3, target_len=110, margin=3, drop=True):
    """
    One Hot Encode input and return X, y
    """
    # Read the input file and save as X and y values
    X = []
    y = []
    target_len_data = 0 # Store the max length of input sequence.
    path, ext = os.path.splitext(input_file)
    n_drop = 0 # Number of seq dropped because of N count
    line_count = 0

    with open(input_file) as f:
        for line in f:
            line_count += 1
            line = line.strip()
            line_list = line.split()
            seq = line_list[0]
            y_val = float(line_list[1])
            if seq.count("N") <= N_tolerance:
                X.append(line_list[0])
                if len(line_list[0]) > target_len_data: # Store if longer than seen before
                    target_len_data = len(seq)
                y.append(float(y_val))
            else:
                n_drop+=1
    
    print("Total lines in infile: ", line_count)
    print("Dropped due to more N count: ", n_drop)
    
    data = []
    y_out = []
    not_on_target = 0 # number of instances outside the target length
    dropped_seq = 0
    for n, item in enumerate(X):
        # Add padding/truncate to make every instance the same length.
        if len(item) != target_len:
            # If drop is set, the instance is discarded, else it is trucated/padded to the set length
            if abs(len(item)-target_len) > margin:
                not_on_target+=1
                if drop:
                    dropped_seq+=1
                    continue # move to next item
            while len(item) < target_len_data and len(item) < target_len:
                item+="N"
            if len(item) > target_len:
                item = item[:target_len]
        
        # OHE and save
        ohe_seq = onehote(item)
        data.append(ohe_seq)
        y_out.append(y[n])
    
    print("Outside length spec: ", dropped_seq)
    return data, y_out


def make_random_splits(X, y, trProp=0.3, seed=42):
    """
    Random splitting of a regression dataset
    trProp  : Proportion of the whole dataset to be used as train set; rest 1-trProp, equally split into val, test sets
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-trProp, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    print(f'Dataset with {len(X)} instances was split into 3 splits.\n')
    print(f'Train\t: {len(X_train)}\nVal\t: {len(X_val)}\nTest\t: {len(X_test)}')

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_stratified_splits(X, y, nQ=10, trProp=0.3, seed=42):
    """
    Stratified splitting of a regression dataset
    trProp  : Proportion of the whole dataset to be used as train set; rest 1-trProp, equally split into val, test sets
    nQ      : Number of quantiles
    """
    quantiles, bins = pd.qcut(y, q=nQ, labels=False, retbins=True, duplicates='drop')

    n_splits = 1

    stratified_splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=1-trProp, random_state=seed)

    for train_index, val_test_index in stratified_splitter.split(X, quantiles):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val_test, y_val_test = X.iloc[val_test_index], y.iloc[val_test_index]

    stratified_splitter_train_val = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=seed)

    for val_index, test_index in stratified_splitter_train_val.split(X_val_test, quantiles.iloc[val_test_index]):
        X_val, y_val = X_val_test.iloc[val_index], y_val_test.iloc[val_index]
        X_test, y_test = X_val_test.iloc[test_index], y_val_test.iloc[test_index]

    print(f'Dataset with {len(X)} instances was split into 3 splits.\n')
    print(f'Train\t: {len(X_train)}\nVal\t: {len(X_val)}\nTest\t: {len(X_test)}')

    return X_train, y_train, X_val, y_val, X_test, y_test, bins


def make_stratified_splits_W(X, y, w, nQ=10, trProp=0.3, seed=42):
    """
    Stratified splitting of a regression dataset (with weights of input instances)
    trProp  : Proportion of the whole dataset to be used as train set; rest 1-trProp, equally split into val, test sets
    nQ      : Number of quantiles
    """
    quantiles, bins = pd.qcut(y, q=nQ, labels=False, retbins=True, duplicates='drop')

    n_splits = 1

    stratified_splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=1-trProp, random_state=seed)

    for train_index, val_test_index in stratified_splitter.split(X, quantiles):
        X_train, y_train, w_train = X.iloc[train_index], y.iloc[train_index], w.iloc[train_index]
        X_val_test, y_val_test, w_val_test = X.iloc[val_test_index], y.iloc[val_test_index], w.iloc[val_test_index]

    stratified_splitter_train_val = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=seed)

    for val_index, test_index in stratified_splitter_train_val.split(X_val_test, quantiles.iloc[val_test_index]):
        X_val, y_val, w_val = X_val_test.iloc[val_index], y_val_test.iloc[val_index], w_val_test.iloc[val_index]
        X_test, y_test, w_test = X_val_test.iloc[test_index], y_val_test.iloc[test_index], w_val_test.iloc[test_index]

    print(f'Dataset with {len(X)} instances was split into 3 splits.\n')
    print(f'Train\t: {len(X_train)}\nVal\t: {len(X_val)}\nTest\t: {len(X_test)}')

    return X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test, bins


def encodeData(sequences, exprs, N_tolerance=3, target_len=110, margin=3, drop=True, choice="onehot", isTest=False):
    """
    Encode a set of sequences using the choice of representation provided (choice)
    isTest: If the passed sequences are intended to be used for testing, we don't want data leakage.
            However, this affects encoding in (integer sequence: "Int") and (isInt and includeP: "Both") encoding choices.

    Note (18/12/2024): I noticed that the models were trained using this version of encodeData(), now called encodeData_old()
                       The version below handles sequences with 'N' differently in test sets. But is more adaptable.
    """
    target_len_data = 0 # Store the max length of input sequence.
    n_drop = 0 # Number of seq dropped because of N count

    X = []
    y = []

    for i in range(len(sequences)):
        seq = sequences.values[i][0]
        expr = exprs.values[i]
        if seq.count("N") <= N_tolerance:
            X.append(seq)
            if len(seq) > target_len_data: # Store if longer than seen before
                target_len_data = len(seq)
            y.append(expr)
        else:
            n_drop += 1
    
    print(f"Number of sequences passed to encode\t: {len(sequences)}")
    print(f"Dropped due to more N count (> {N_tolerance})\t: {n_drop}")
    
    X_enc = []
    y_out = []

    not_on_target = 0 # number of instances outside the target length
    dropped_seq = 0
    for n, item in enumerate(X):
        # Add padding/truncate to make every instance the same length.
        if len(item) != target_len:
            # If drop is set, the instance is discarded, else it is trucated/padded to the set length
            if abs(len(item)-target_len) > margin:
                not_on_target+=1
                if drop:
                    dropped_seq+=1
                    continue # move to next item
            while len(item) < target_len_data and len(item) < target_len:
                item+="N"
            if len(item) > target_len:
                item = item[:target_len]

        #call the sequence multihot encode function
        if choice == "onehot":
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False)
        if choice == "onehotWithN":
            seq_enc = multihote(seq=item, isTest=isTest)
        if choice == "onehotWithP":
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False, onlyP=True)
        if choice == "onehotWithInt":
            isR = False if y[n].is_integer() else True
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False, onlyInt=True, isR=isR)
        if choice == "onehotWithBoth":
            isR = False if y[n].is_integer() else True
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False, both=True, isR=isR)

        X_enc.append(seq_enc)
        y_out.append(y[n])
    
    print(f"Dropped due to outside length spec\t: {dropped_seq}\n")
    return X_enc, y_out


def encodeData_v1(sequences, exprs, N_tolerance=3, target_len=110, margin=3, drop=True, choice="onehot", isTest=False):
    """
    Encode a set of sequences using the choice of representation provided (choice)
    isTest: If the passed sequences are intended to be used for testing, we don't want data leakage.
            However, this affects encoding in (integer sequence: "Int") and (isInt and includeP: "Both") encoding choices.

    18/12/2024: Just keeping a backup of the old version (encodeData). This version does not throw away sequences with more N.
                I don't remember when was this version used where; probably wasn't.
                This version throws an error, because sequences may not have consistent length, after adding that isTest condition
                within the first for loop. This is fixed in the encodeData_v1_corrected() as below; and tested to work.
    """
    target_len_data = 0 # Store the max length of input sequence.
    n_drop = 0 # Number of seq dropped because of N count

    X = []
    y = []

    for i in range(len(sequences)):
        seq = sequences.values[i][0]
        expr = exprs.values[i]

        if not isTest:
            if seq.count("N") <= N_tolerance:
                X.append(seq)
                if len(seq) > target_len_data: # Store if longer than seen before
                    target_len_data = len(seq)
                y.append(expr)
            else:
                n_drop += 1
        else:
            X.append(seq)
            y.append(expr)
    
    print(f"Number of sequences passed to encode\t: {len(sequences)}")
    print(f"Dropped due to more N count (> {N_tolerance})\t: {n_drop}")
    
    X_enc = []
    y_out = []

    not_on_target = 0 # number of instances outside the target length
    dropped_seq = 0
    for n, item in enumerate(X):
        # Add padding/truncate to make every instance the same length.
        if len(item) != target_len:
            # If drop is set, the instance is discarded, else it is trucated/padded to the set length
            if abs(len(item)-target_len) > margin:
                not_on_target+=1
                if drop:
                    dropped_seq+=1
                    continue # move to next item
            while len(item) < target_len_data and len(item) < target_len:
                item+="N"
            if len(item) > target_len:
                item = item[:target_len]

        #call the sequence multihot encode function
        if choice == "onehot":
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False)
        if choice == "onehotWithN":
            seq_enc = multihote(seq=item, isTest=isTest)
        if choice == "onehotWithP":
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False, onlyP=True)
        if choice == "onehotWithInt":
            isR = False if y[n].is_integer() else True
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False, onlyInt=True, isR=isR)
        if choice == "onehotWithBoth":
            isR = False if y[n].is_integer() else True
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False, both=True, isR=isR)

        X_enc.append(seq_enc)
        y_out.append(y[n])
    
    print(f"Dropped due to outside length spec\t: {dropped_seq}\n")
    return X_enc, y_out


def encodeData_v1_corrected(sequences, exprs, N_tolerance=3, target_len=110, margin=3, drop=True, choice="onehot", isTest=False):
    """
    Encode a set of sequences using the choice of representation provided (choice)
    isTest: If the passed sequences are intended to be used for testing, we don't want data leakage.
            However, this affects encoding in (integer sequence: "Int") and (isInt and includeP: "Both") encoding choices.

    18/12/2024: Fix to the encodeData_v1() above. Handles sequences (in test set) with more N values well; doesn't throw them away.
    """
    target_len_data = 0 # Store the max length of input sequence.
    n_drop = 0 # Number of seq dropped because of N count

    X = []
    y = []

    for i in range(len(sequences)):
        seq = sequences.values[i][0]
        expr = exprs.values[i]

        if not isTest:
            if seq.count("N") <= N_tolerance:
                X.append(seq)
                if len(seq) > target_len_data: # Store if longer than seen before
                    target_len_data = len(seq)
                y.append(expr)
            else:
                n_drop += 1
        else:
            X.append(seq)
            y.append(expr)

    print(f"Number of sequences passed to encode\t: {len(sequences)}")
    print(f"Dropped due to more N count (> {N_tolerance})\t: {n_drop}")

    # Ensure all sequences are padded or truncated to target_len
    X_processed = []
    for seq in X:
        if len(seq) < target_len:
            seq += "N" * (target_len - len(seq))  # Pad with 'N' to target length
        elif len(seq) > target_len:
            seq = seq[:target_len]  # Truncate to target length
        X_processed.append(seq)

    X = X_processed

    X_enc = []
    y_out = []

    not_on_target = 0 # number of instances outside the target length
    dropped_seq = 0
    for n, item in enumerate(X):
        #call the sequence multihot encode function
        if choice == "onehot":
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False)
        if choice == "onehotWithN":
            seq_enc = multihote(seq=item, isTest=isTest)
        if choice == "onehotWithP":
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False, onlyP=True)
        if choice == "onehotWithInt":
            isR = False if y[n].is_integer() else True
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False, onlyInt=True, isR=isR)
        if choice == "onehotWithBoth":
            isR = False if y[n].is_integer() else True
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False, both=True, isR=isR)

        X_enc.append(seq_enc)
        y_out.append(y[n])

    print(f"Dropped due to outside length spec\t: {dropped_seq}\n")
    return X_enc, y_out



def encodeData_W(sequences, exprs, weights, N_tolerance=3, target_len=110, margin=3, drop=True, choice="onehot", isTest=False):
    """
    Encode a set of sequences using the choice of representation provided (choice)
    isTest: If the passed sequences are intended to be used for testing, we don't want data leakage.
            However, this affects encoding in (integer sequence: "Int") and (isInt and includeP: "Both") encoding choices.
    """
    target_len_data = 0 # Store the max length of input sequence.
    n_drop = 0 # Number of seq dropped because of N count

    X = []
    y = []
    w = []

    for i in range(len(sequences)):
        seq = sequences.values[i][0]
        expr = exprs.values[i]
        weight = weights.values[i]
        if seq.count("N") <= N_tolerance:
            X.append(seq)
            if len(seq) > target_len_data: # Store if longer than seen before
                target_len_data = len(seq)
            y.append(expr)
            w.append(weight)
        else:
            n_drop += 1
    
    print(f"Number of sequences passed to encode\t: {len(sequences)}")
    print(f"Dropped due to more N count (> {N_tolerance})\t: {n_drop}")
    
    X_enc = []
    y_out = []
    w_out = []

    not_on_target = 0 # number of instances outside the target length
    dropped_seq = 0
    for n, item in enumerate(X):
        # Add padding/truncate to make every instance the same length.
        if len(item) != target_len:
            # If drop is set, the instance is discarded, else it is trucated/padded to the set length
            if abs(len(item)-target_len) > margin:
                not_on_target+=1
                if drop:
                    dropped_seq+=1
                    continue # move to next item
            while len(item) < target_len_data and len(item) < target_len:
                item+="N"
            if len(item) > target_len:
                item = item[:target_len]

        #call the sequence multihot encode function
        if choice == "onehot":
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False)
        if choice == "onehotWithN":
            seq_enc = multihote(seq=item, isTest=isTest)
        if choice == "onehotWithP":
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False, onlyP=True)
        if choice == "onehotWithInt":
            isR = False if y[n].is_integer() else True
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False, onlyInt=True, isR=isR)
        if choice == "onehotWithBoth":
            isR = False if y[n].is_integer() else True
            seq_enc = multihote(seq=item, isTest=isTest, onlyN=False, both=True, isR=isR)

        X_enc.append(seq_enc)
        y_out.append(y[n])
        w_out.append(w[n])
    
    print(f"Dropped due to outside length spec\t: {dropped_seq}\n")
    return X_enc, y_out, w_out


def createDataLoaders(seq_tr, expr_tr, seq_va, expr_va, seq_te, expr_te, config):
    """
    Create Train, Val and Test DataLoader objects from train, val and test splits (X, y)s
    """
    #encode the data
    X_train, y_train = encodeData(sequences=seq_tr, exprs=expr_tr, choice=config["seq_enc"], isTest=False)
    X_val, y_val     = encodeData(sequences=seq_va, exprs=expr_va, choice=config["seq_enc"], isTest=True)
    X_test, y_test   = encodeData(sequences=seq_te, exprs=expr_te, choice=config["seq_enc"], isTest=True)

    #create DataLoaders (train, val, test)
    TrainX_Tensor = torch.tensor(X_train).unsqueeze(1)
    TrainX_Tensor = torch.transpose(TrainX_Tensor,1,3)
    TrainY_Tensor = Tensor(y_train)

    g = torch.Generator()
    g.manual_seed(config["seed"])

    TrainLoader = DataLoader(dataset=TensorDataset(TrainX_Tensor, TrainY_Tensor), 
                             num_workers=0, 
                             batch_size=config["batch_size"], 
                             shuffle=True, 
                             drop_last=True, 
                             worker_init_fn=np.random.seed(config["seed"]), 
                             generator=g)

    ValX_Tensor = torch.tensor(X_val).unsqueeze(1)
    ValX_Tensor = torch.transpose(ValX_Tensor,1,3)
    ValY_Tensor = Tensor(y_val)

    ValLoader = DataLoader(dataset=TensorDataset(ValX_Tensor, ValY_Tensor), 
                           num_workers=0, 
                           batch_size=config["batch_size"], 
                           shuffle=True, 
                           drop_last=True)

    TestX_Tensor = torch.tensor(X_test).unsqueeze(1)
    TestX_Tensor = torch.transpose(TestX_Tensor,1,3)
    TestY_Tensor = Tensor(y_test)

    TestLoader = DataLoader(dataset=TensorDataset(TestX_Tensor, TestY_Tensor), 
                            num_workers=0, 
                            batch_size=config["batch_size"], 
                            shuffle=True, 
                            drop_last=True)
    
    return TrainLoader, ValLoader, TestLoader


def createDataLoaders_W(seq_tr, expr_tr, weight_tr, seq_va, expr_va, weight_va, seq_te, expr_te, weight_te, config):
    """
    Create Train, Val and Test DataLoader objects from train, val and test splits (X, y, w)s
    """
    #encode the data
    X_train, y_train, w_train = encodeData_W(sequences=seq_tr, exprs=expr_tr, weights=weight_tr, choice=config["seq_enc"], isTest=False)
    X_val, y_val, w_val       = encodeData_W(sequences=seq_va, exprs=expr_va, weights=weight_va, choice=config["seq_enc"], isTest=True)
    X_test, y_test, w_test    = encodeData_W(sequences=seq_te, exprs=expr_te, weights=weight_te, choice=config["seq_enc"], isTest=True)

    #create DataLoaders (train, val, test)
    TrainX_Tensor = torch.tensor(X_train).unsqueeze(1)
    TrainX_Tensor = torch.transpose(TrainX_Tensor,1,3)
    TrainY_Tensor = Tensor(y_train)
    TrainW_Tensor = Tensor(w_train)

    g = torch.Generator()
    g.manual_seed(config["seed"])

    TrainLoader = DataLoader(dataset=TensorDataset(TrainX_Tensor, TrainY_Tensor, TrainW_Tensor), 
                             num_workers=0, 
                             batch_size=config["batch_size"], 
                             shuffle=True, 
                             drop_last=True, 
                             worker_init_fn=np.random.seed(config["seed"]), 
                             generator=g)

    ValX_Tensor = torch.tensor(X_val).unsqueeze(1)
    ValX_Tensor = torch.transpose(ValX_Tensor,1,3)
    ValY_Tensor = Tensor(y_val)
    ValW_Tensor = Tensor(w_val)

    ValLoader = DataLoader(dataset=TensorDataset(ValX_Tensor, ValY_Tensor, ValW_Tensor), 
                           num_workers=0, 
                           batch_size=config["batch_size"], 
                           shuffle=True, 
                           drop_last=True)

    TestX_Tensor = torch.tensor(X_test).unsqueeze(1)
    TestX_Tensor = torch.transpose(TestX_Tensor,1,3)
    TestY_Tensor = Tensor(y_test)
    TestW_Tensor = Tensor(w_test)

    TestLoader = DataLoader(dataset=TensorDataset(TestX_Tensor, TestY_Tensor, TestW_Tensor), 
                            num_workers=0, 
                            batch_size=config["batch_size"], 
                            shuffle=True, 
                            drop_last=True)
    
    return TrainLoader, ValLoader, TestLoader


def createDataLoader(seq, expr, config, isTest=True, drop=True):
    """
    Create a DataLoader object from input sequnces and their expressions (for prediction)
    isTest: Whether it is a test dataset? Default: True
    """
    X, y = encodeData(sequences=seq, exprs=expr, choice=config["seq_enc"], isTest=isTest, drop=drop)
    
    if not len(X):
        return None

    X_Tensor = torch.tensor(X).unsqueeze(1)
    X_Tensor = torch.transpose(X_Tensor,1,3)
    Y_Tensor = Tensor(y)

    TestLoader = DataLoader(dataset=TensorDataset(X_Tensor, Y_Tensor), 
                            num_workers=0, 
                            batch_size=config["batch_size"], 
                            shuffle=False, 
                            drop_last=False)
    
    return TestLoader


def createDataLoader_v1(seq, expr, config, isTest=True, drop=False):
    """
    Create a DataLoader object from input sequnces and their expressions (for prediction);
    This version is useful to prepare data for the traininig set.
    
    isTest: Whether it is a test dataset? Default: True
    """
    X, y = encodeData_v1_corrected(sequences=seq, exprs=expr, choice=config["seq_enc"], isTest=isTest, drop=drop)
    
    if not len(X):
        return None

    X_Tensor = torch.tensor(X).unsqueeze(1)
    X_Tensor = torch.transpose(X_Tensor,1,3)
    Y_Tensor = Tensor(y)

    TestLoader = DataLoader(dataset=TensorDataset(X_Tensor, Y_Tensor), 
                            num_workers=0, 
                            batch_size=config["batch_size"], 
                            shuffle=False, 
                            drop_last=False)
    
    return TestLoader
    

def createDataLoaders_in_chunks(seq_tr, expr_tr, seq_va, expr_va, seq_te, expr_te, config, chunk_size=1000000):
    """
    Create Train, Val and Test DataLoader objects from train, val and test splits (X, y)s.
    This function is useful for loading and dealing with very large datasets (roughly; > 10M instances)
    """
    import tempfile

    # Save to temporary files
    def save_to_temp_csv(data, prefix):
        temp_csv = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix=".csv")
        data.to_csv(temp_csv.name, index=False)
        temp_csv.close()
        return temp_csv.name

    seq_tr_file = save_to_temp_csv(seq_tr, "/tmp/seq_tr_")
    seq_va_file = save_to_temp_csv(seq_va, "/tmp/seq_va_")
    seq_te_file = save_to_temp_csv(seq_te, "/tmp/seq_te_")
    expr_tr_file = save_to_temp_csv(expr_tr, "/tmp/expr_tr_")
    expr_va_file = save_to_temp_csv(expr_va, "/tmp/expr_va_")
    expr_te_file = save_to_temp_csv(expr_te, "/tmp/expr_te_")

    def process_and_collect_data(sequences_file, exprs_file, config, isTest):
        X_enc_list = []
        y_out_list = []
        for seq_chunk, expr_chunk in zip(pd.read_csv(sequences_file, chunksize=chunk_size), pd.read_csv(exprs_file, chunksize=chunk_size)):
            X_chunk, y_chunk = encodeData(sequences=seq_chunk, exprs=expr_chunk, choice=config["seq_enc"], isTest=isTest)
            X_enc_list.extend(X_chunk)
            y_out_list.extend(y_chunk)
        return X_enc_list, y_out_list

    # Encode the data
    X_train, y_train = process_and_collect_data(seq_tr_file, expr_tr_file, config, isTest=False)
    X_val, y_val = process_and_collect_data(seq_va_file, expr_va_file, config, isTest=True)
    X_test, y_test = process_and_collect_data(seq_te_file, expr_te_file, config, isTest=True)

    # Clean up temporary files
    os.remove(seq_tr_file)
    os.remove(seq_va_file)
    os.remove(seq_te_file)
    os.remove(expr_tr_file)
    os.remove(expr_va_file)
    os.remove(expr_te_file)

    # Create DataLoaders (train, val, test)
    def create_tensor_loader(X, y, batch_size, shuffle):
        X_tensor = torch.tensor(X).unsqueeze(1)
        X_tensor = torch.transpose(X_tensor, 1, 3)
        Y_tensor = Tensor(y).squeeze()
        print(X_tensor.shape)
        print(Y_tensor.shape)
        dataset = TensorDataset(X_tensor, Y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)

    g = torch.Generator()
    g.manual_seed(config["seed"])

    TrainLoader = create_tensor_loader(X_train, y_train, config["batch_size"], shuffle=True)
    ValLoader = create_tensor_loader(X_val, y_val, config["batch_size"], shuffle=True)
    TestLoader = create_tensor_loader(X_test, y_test, config["batch_size"], shuffle=True)

    return TrainLoader, ValLoader, TestLoader
