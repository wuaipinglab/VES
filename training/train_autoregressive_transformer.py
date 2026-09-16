import os, sys
import argparse
import pandas as pd
import json
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import copy
import math

from AT_model import ARTransformerRoPE
from utils.data_process_msa import ProteinMSADataset

import copy

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"  ## 20 aa + gap
stoi = {aa: i for i, aa in enumerate(AA_ALPHABET)}
itos = {i: aa for aa, i in stoi.items()}
vocab_size = len(stoi)

def tokenize(seq):
    return [stoi[ch] for ch in seq]

def detokenize(ids):
    return "".join([itos[i] for i in ids])


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device="cuda"):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 21), y.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

@torch.no_grad()
def generate(model, start_seq="A", max_len=565, device="cuda"):
    model.eval()
    x = torch.tensor([stoi[ch] for ch in start_seq], dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_len - len(start_seq)):
        logits = model(x)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)
    return detokenize(x[0].tolist())


class EMA:
        def __init__(self, model: nn.Module, decay: float = 0.999):
            self.decay = decay
            self.ema = copy.deepcopy(model).eval()
            for p in self.ema.parameters():
                p.requires_grad_(False)
        @torch.no_grad()
        def update(self, model: nn.Module):
            msd, esd = model.state_dict(), self.ema.state_dict()
            for k in esd.keys():
                if esd[k].dtype.is_floating_point:
                    esd[k].mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)
                else:
                    esd[k] = msd[k]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--MSA_data_folder', default='./data/MSA', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', default='./data/mappings/example_mapping.csv', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', default=0, type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--training_logs_location', default='./logs/', type=str, help='Location of VAE model parameters')
    parser.add_argument('--batch_size', default=36, type=int, help='Batch size for dataloader')
    parser.add_argument('--vocab_size', default=21, type=int, help='protein msa sequence vocabulary size')
    args = parser.parse_args()

    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    print("Protein name: "+str(protein_name))
    print("MSA file: "+str(msa_location))


    dataset = ProteinMSADataset(msa_location)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    def train_score_based_model(model, train_loader, num_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        ema = EMA(model, 0.999)
        criterion = nn.CrossEntropyLoss() 
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_idx, (batch_data, target) in enumerate(train_loader):
                batch_data, target = batch_data.cuda(), target.cuda()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(batch_data)
                    loss = criterion(logits.view(-1, 21), target.view(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
                epoch_loss += loss.item()
                if (batch_idx + 1) % 100 == 0:
                    avg = epoch_loss / (batch_idx + 1)
                    print(f"Epoch {epoch} | iter {batch_idx+1}/{len(train_loader)} | loss {avg:.4f}")


            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ARTransformerRoPE(vocab_size=21, seq_len=565, d_model=256, nhead=8, num_layers=6).to(device)
    train_score_based_model(model, train_loader, num_epochs=15)
    torch.save(model.state_dict(), './results/Diffusion_model/Dec2_FLU_AT.pth')

    ppl = evaluate_perplexity(model, val_loader, device)
    print(f"Final Epoch, val perplexity={ppl:.4f}")
