import numpy as np
import pandas as pd


def hamming_distance(seq1, seq2):
    """
    Compute Hamming distance between two sequences
    """
    assert len(seq1) == len(seq2), "Sequences must be of the same length."
    distance = sum(1 for x, y in zip(seq1, seq2) if x != y) / len(seq1)
    
    return distance


def makeSNVseqs(original_seq, expr):
    """
    Makes single-nt mutations from position 18 to 97 in a promoter sequence
    Returns: (seq, expr) pairs as a dataframe
    """
    prefix = original_seq[:17]
    suffix = original_seq[-13:]
    middle_seq = original_seq[17:97]

    nucleotides = ['A', 'C', 'G', 'T']

    snv_seqs = [original_seq]  
    for i in range(len(middle_seq)):
        for nucleotide in nucleotides:
            if middle_seq[i] != nucleotide:
                snv_seq = prefix + middle_seq[:i] + nucleotide + middle_seq[i + 1:] + suffix
                snv_seqs.append(snv_seq)

    print(f"{len(snv_seqs)} SNV sequences generated.")
    
    df = pd.DataFrame({
        'seq': snv_seqs,
        'expr': [expr] + [np.nan] * (len(snv_seqs)-1)
    })
    
    return df

def write_to_fasta(sequences, output_file="/tmp/seq.fa"):
    """
    Write sequences to a FASTA file.
    Prepares FASTA sequence as ">sequence_{id} followed by the sequence"
    Example Input: 
                    AAGTTGTTCCAA
    Example Output:
                    >sequence_1
                    AAGTTGTTCCAA
    """
    import os
    output_file = os.path.expanduser(output_file)
    with open(output_file, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">sequence_{i+1}\n")
            f.write(seq + "\n")