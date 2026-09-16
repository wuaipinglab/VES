import numpy as np
import pandas as pd
from collections import defaultdict
import os
import torch
import tqdm
import torch.nn as nn

class BlosumEmbedding(nn.Module):
    def __init__(self, csv_path, aa_order="ACDEFGHIKLMNPQRSTVWY"):
        super().__init__()
        df = pd.read_csv(csv_path, index_col=0)
        df = df.loc[list(aa_order), list(aa_order)]

        mat = torch.tensor(df.values, dtype=torch.float32)
        self.register_buffer("blosum_mat", mat)  # shape: (20, 20)
        self.aa2idx = {aa: i for i, aa in enumerate(aa_order)}

    def forward(self, seqs):

        if isinstance(seqs, str):
            seqs = [seqs]

        B = len(seqs)
        L = max(len(s) for s in seqs)
        V = self.blosum_mat.size(1)

        embeddings = torch.zeros((B, L, V), dtype=torch.float32)

        for i, seq in enumerate(seqs):
            for j, aa in enumerate(seq):
                if aa not in self.aa2idx:
                    embeddings[i, j] = torch.ones(V) / V
                else:
                    idx = self.aa2idx[aa]
                    embeddings[i, j] = self.blosum_mat[idx]

        return embeddings

class OneHotAAEmbedder:
    def __init__(self, aa_order="ACDEFGHIKLMNPQRSTVWY", pad_token="-"):

        self.aa_order = aa_order
        self.vocab_size = len(aa_order)
        self.aa2idx = {aa: i for i, aa in enumerate(aa_order)}
        self.pad_token = pad_token

    def encode_batch(self, seqs):

        B = len(seqs)
        L = max(len(s) for s in seqs)
        one_hot = torch.zeros((B, L, self.vocab_size), dtype=torch.float32)

        for i, seq in enumerate(seqs):
            for j, aa in enumerate(seq):
                idx = self.aa2idx.get(aa, None)
                if idx is not None:
                    one_hot[i, j, idx] = 1.0
                else:
                    pass
        return one_hot

class MSA_processing:
    def __init__(self,
        MSA_location="",
        theta=0.2,
        use_weights=True,
        weights_location="./data/weights",
        preprocess_MSA=True,
        threshold_sequence_frac_gaps=0.5,
        threshold_focus_cols_frac_gaps=0.3,
        remove_sequences_with_indeterminate_AA_in_focus_cols=True
        ):

          
        np.random.seed(2025)
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.theta = theta
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.use_weights = use_weights
        self.preprocess_MSA = preprocess_MSA
        self.threshold_sequence_frac_gaps = threshold_sequence_frac_gaps
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.remove_sequences_with_indeterminate_AA_in_focus_cols = remove_sequences_with_indeterminate_AA_in_focus_cols

        self.gen_alignment()
        self.create_all_singles()

    def gen_alignment(self):

        self.aa_dict = {}
        for i,aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        self.seq_name_to_sequence = defaultdict(str)
        name = ""
        with open(self.MSA_location, "r") as msa_data:
            for i, line in enumerate(msa_data):
                line = line.rstrip()
                if line.startswith(">"):
                    name = line
                    if i==0:
                        self.focus_seq_name = name
                else:
                    self.seq_name_to_sequence[name] += line

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s!='-'] 
        self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols]
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        start = 1
        stop = 565
        self.focus_start_loc = int(start)
        self.focus_stop_loc = int(stop)
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col+int(start):self.focus_seq[idx_col] for idx_col in self.focus_cols} 
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col+int(start):idx_col for idx_col in self.focus_cols} 

        for seq_name,sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".","-")
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]

        if self.remove_sequences_with_indeterminate_AA_in_focus_cols:
            alphabet_set = set(list(self.alphabet))
            seq_names_to_remove = []
            for seq_name,sequence in self.seq_name_to_sequence.items():
                for letter in sequence:
                    if letter not in alphabet_set and letter != "-":
                        seq_names_to_remove.append(seq_name)
                        continue
            seq_names_to_remove = list(set(seq_names_to_remove))
            for seq_name in seq_names_to_remove:
                del self.seq_name_to_sequence[seq_name]

        print ("Encoding sequences")
        blosum_df = pd.read_csv('blosum62_softmax_prob.csv', index_col=0)
        mat = blosum_df.values
        self.blosum_mat = mat # shape: (20, 20)

        self.one_hot_encoding = np.zeros((len(self.seq_name_to_sequence.keys()),len(self.focus_cols),len(self.alphabet)))
        self.blosum_encoding = np.zeros((len(self.seq_name_to_sequence.keys()),len(self.focus_cols),len(self.alphabet)))
        for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence[seq_name]
            for j,letter in enumerate(sequence):
                if letter in self.aa_dict: 
                    k = self.aa_dict[letter]
                    self.one_hot_encoding[i,j,k] = 1.0
                    self.blosum_encoding[i,j] = self.blosum_mat[k]
        self.num_sequences = self.one_hot_encoding.shape[0]
        print ("Data Shape =",self.one_hot_encoding.shape)
    
    def create_all_singles(self):
        start_idx = self.focus_start_loc
        focus_seq_index = 0
        self.mutant_to_letter_pos_idx_focus_list = {}
        list_valid_mutations = []
        alphabet_set = set(list(self.alphabet))
        for i,letter in enumerate(self.focus_seq):
            if letter in alphabet_set and letter != "-":
                for mut in self.alphabet:
                    pos = start_idx+i
                    if mut != letter:
                        mutant = letter+str(pos)+mut
                        self.mutant_to_letter_pos_idx_focus_list[mutant] = [letter, pos, focus_seq_index]
                        list_valid_mutations.append(mutant)
                focus_seq_index += 1   
        self.all_single_mutations = list_valid_mutations

    def save_all_singles(self, output_filename):
        with open(output_filename, "w") as output:
            output.write('mutations')
            for mutation in self.all_single_mutations:
                output.write('\n')
                output.write(mutation)
