import pandas as pd
import random
import gc
import glob
import numpy as np
from scipy.stats import entropy
from base.utils import createDataLoader
from base.model_utils import predict

# Global variables (if needed for mapping expressions); can be computed once and set (but better to avoid re-computing)
#df_trn = pd.read_csv("/home/dash01/SBLab/Camformer/data/DREAM2022/train_sequences.txt", header=None, delimiter="\t", names=["seq","expr"])
#min_trn_expr, max_trn_expr = df_trn["expr"].min(), df_trn["expr"].max()
#print(min_trn_expr, max_trn_expr)

#df_tst = pd.read_csv("/home/dash01/SBLab/Camformer/OfficialEval/filtered_test_data_with_MAUDE_expression.txt", delimiter='\t', header=None, names=["sequence","expr"])
#min_tst_expr, max_tst_expr = df_tst["expr"].min(), df_tst["expr"].max()
#print(min_tst_expr, max_tst_expr)
#del df_trn, df_tst
#gc.collect()

min_trn_expr, max_trn_expr, min_tst_expr, max_tst_expr = 0.0, 17.0, -1.40175772469441, 1.65983547290171
print(min_trn_expr, max_trn_expr)
print(min_tst_expr, max_tst_expr)

def map_to_train_expr(expr):
    """
    Map an expression value to the range of expression values in the training set.
    """
    global max_trn_expr, min_trn_expr, max_tst_expr, min_tst_expr
    expr_std = (expr - min_tst_expr) / (max_tst_expr - min_tst_expr)
    return expr_std * (max_trn_expr - min_trn_expr) + min_trn_expr

def predict_expression(model, config, sequences, exprs):
    """
    Expects a list of sequences. Returns a list of predicted expressions.
    """
    df = pd.DataFrame(columns=["seq", "expr"])
    df["seq"] = sequences
    df["expr"] = exprs
    config["batch_size"] = len(sequences)  # max is 80*3 = 240
    TestLoader = createDataLoader(df[["seq"]], df["expr"], config, isTest=True)
    if TestLoader is None:
        print("Value error: createDataLoader returned None. No valid sequences to process.")
        return None
    y_pred, _ = predict(model, TestLoader, config)
    return y_pred

def generate_mutations(sequence):
    """
    Generate all possible single-nucleotide mutations for a given sequence.
    """
    if len(sequence) == 110:
        prefix_len, suffix_len = 17, 13
        prefix = sequence[:prefix_len]
        suffix = sequence[-suffix_len:]
        middle_seq = sequence[prefix_len:-suffix_len]
    else:
        raise ValueError("Sequence length should be 110.")

    mutations = []
    nucleotides = ['A', 'C', 'G', 'T']
    for i, original_nucleotide in enumerate(middle_seq):
        for nucleotide in nucleotides:
            if nucleotide != original_nucleotide:
                mutated_sequence = (prefix + middle_seq[:i] + nucleotide + middle_seq[i + 1:] + suffix)
                mutations.append((i + prefix_len, original_nucleotide, nucleotide, mutated_sequence))
    return mutations

def mutate_sequence(sequence, position, base):
    """
    Mutate a sequence at a specific position with a specific base.
    """
    mutated_sequence = list(sequence)
    mutated_sequence[position] = base
    return "".join(mutated_sequence)

def sample_sequences_from_file(file_path, num_samples=10, sequence_length=110):
    """
    Sample a specific number of sequences from a file.
    """
    sequences = []
    with open(file_path, "r") as file:
        for line in file:
            sequence, expression = line.strip().split("\t")
            if len(sequence) == sequence_length:
                sequences.append((sequence, float(expression)))
    num_samples = min(num_samples, len(sequences))
    return random.sample(sequences, num_samples)

def generate_logo_w_flanking(model, config, original_sequence, original_expression, entropy_thres=5.0, save_dir=None, print_expr=False):
    """
    Generate Logo Plots for any given sequence of length 110.
    """
    # If the sequence is already processed, no need to run ISM again 
    if save_dir:
        file_pattern = f"{save_dir}/{original_sequence}_*.pdf"
        matching_files = glob.glob(file_pattern)
        if matching_files:
            return False
    
    import matplotlib.pyplot as plt
    import logomaker
    import os

    print(f"Generating ISM logos for: {original_sequence}")
    
    assert len(original_sequence) == 110, f"Error: Expected sequence of length 110, but got length {len(original_sequence)}."

    mapped_raw_expression = map_to_train_expr(original_expression)
    predicted_expression = predict_expression(model, config, [original_sequence], [original_expression])

    mutations = generate_mutations(original_sequence)
    mutated_sequences = [mutated_sequence for (_, _, _, mutated_sequence) in mutations]
    predicted_expressions = predict_expression(model, config, mutated_sequences, [original_expression] * len(mutated_sequences))

    expression_changes = [
        (i, original, mutated, pred_expr - predicted_expression[0])
        for (i, original, mutated, _), pred_expr in zip(mutations, predicted_expressions)
    ]

    pssm_data = pd.DataFrame(0.0, index=range(0, 110), columns=['A', 'C', 'G', 'T'])

    for (i, original, mutated, change_in_expression) in expression_changes:
        if 17 <= i < 97:
            pssm_data.at[i, mutated] += change_in_expression

    # the following may not be necessary; just keepign it however
    for i in range(17, 97):
        original_nucleotide = original_sequence[i]
        pssm_data.at[i, original_nucleotide] = 0

    pssm_data /= 3

    importance_data = pssm_data.sum(axis=1)
    importance_pssm = pd.DataFrame(0.0, index=range(0, 110), columns=['A', 'C', 'G', 'T'])

    for i in range(0, 110):
        original_nucleotide = original_sequence[i]
        importance_pssm.at[i, original_nucleotide] = importance_data[i]

    importance_data_normalized = importance_data.abs() / importance_data.abs().sum()
    entropy_value = entropy(importance_data_normalized, base=2)

    if print_expr:
        print(f"Orig: {original_expression} | Mapped Orig: {mapped_raw_expression} | Pred: {predicted_expression}")

    if entropy_value < entropy_thres:
        fig, ax = plt.subplots(figsize=(10, 3))
        logo = logomaker.Logo(pssm_data, ax=ax)
        logo.style_glyphs(shade_below_zero=True, fade_below_zero=True)
        logo.style_xticks(anchor=-1, spacing=5, rotation=0)
        ax.set_xlabel("Nucleotide position")
        ax.set_ylabel(f"$\Delta E$")
        plt.tight_layout()

        pssm_data['Nucleotide'] = list(original_sequence)
        pssm_data.set_index('Nucleotide', inplace=True)

        if save_dir:
            file_path = f"{save_dir}/{original_sequence}_oe{original_expression:.2f}_me{mapped_raw_expression:.2f}"
            plt.savefig(f"{file_path}.pdf", format="pdf", bbox_inches="tight")
            pssm_data.to_csv(f"{file_path}_pssm_data.csv", index=True)
        else:
            pssm_data.to_csv("/tmp/pssm_data.csv", index=True)
            plt.show()

        color_scheme = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
        fig, ax = plt.subplots(figsize=(10, 3))
        logo = logomaker.Logo(importance_pssm, color_scheme=color_scheme, ax=ax)
        logo.style_xticks(anchor=-1, spacing=5, rotation=0)
        ax.set_xlabel("Nucleotide position")
        ax.set_ylabel(f"Average $\Delta E$")
        plt.tight_layout()

        importance_pssm['Nucleotide'] = list(original_sequence)
        importance_pssm.set_index('Nucleotide', inplace=True)

        if save_dir:
            plt.savefig(f"{file_path}.avg.pdf", format="pdf", bbox_inches="tight")
            importance_pssm.to_csv(f"{file_path}_importance_pssm.csv", index=True)
        else:
            importance_pssm.to_csv("/tmp/importance_pssm.csv", index=True)
            plt.show()

        plt.close("all")
        return True

    return False


