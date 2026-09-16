import torch
from torch.utils.data import Dataset, DataLoader


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"
aa_to_id = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
id_to_aa = {i: aa for aa, i in aa_to_id.items()}
vocab_size = len(AMINO_ACIDS)  # 21

def tokenize_sequence(seq, max_len=565):

    ids = [aa_to_id.get(aa, aa_to_id["-"]) for aa in seq] 
    if len(ids) < max_len:
        ids += [aa_to_id["-"]] * (max_len - len(ids))
    return torch.tensor(ids[:max_len], dtype=torch.long)

class ProteinMSADataset(Dataset):
    def __init__(self, msa_file, max_len=565):
        with open(msa_file, "r") as f:
            lines = [line.strip() for line in f if not line.startswith(">")]
        self.sequences = [tokenize_sequence(seq, max_len) for seq in lines]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq[:-1] 
        y = seq[1:]  
        return x, y
    
class ProteinMSAFullSeqDataset(Dataset):
    def __init__(self, msa_file, max_len=565):
        with open(msa_file, "r") as f:
            lines = [line.strip() for line in f if not line.startswith(">")]
        self.sequences = [tokenize_sequence(seq, max_len) for seq in lines]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq

    
