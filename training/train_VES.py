import os, sys
import argparse
import pandas as pd
import json
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import copy

from Diffusion_Model_Unet_FLU_128_ch import UNetScoreBasedDiffusionModel
from utils import data_utils_FLU


class MSAData(torch.utils.data.Dataset):
    def __init__(self, msa_data):
        self.data = torch.tensor(msa_data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
    parser = argparse.ArgumentParser(description='VES Official Implementation')
    parser.add_argument('--MSA_data_folder', default='./data/MSA', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', default='./data/mappings/example_mapping.csv', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', default=0, type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', default= './data/weights', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--model_name_suffix', default='FLU_DM_UNet_Attention_128_ch', type=str, help='model checkpoint name will be the protein name followed by this suffix')
    parser.add_argument('--training_logs_location', default='./logs/', type=str, help='Location of model parameters')
    args = parser.parse_args()

    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    print("Protein name: "+str(protein_name))
    print("MSA file: "+str(msa_location))

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
    else:
        try:
            theta = float(mapping_file['theta'][args.protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: "+str(theta))

    data = data_utils_FLU.MSA_processing(
            MSA_location=msa_location,
            theta=theta,
            use_weights=True,
            weights_location=args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    )

    x_train = data.one_hot_encoding
    best_val_loss = None

    dataset = MSAData(x_train)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True) 

    def train_score_based_model(model, dataloader, num_epochs, timesteps):
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler(device='cuda',enabled=True)
        ema = EMA(model, 0.999)  
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_idx, batch_data in enumerate(dataloader):
                batch_data = batch_data.cuda()
                optimizer.zero_grad()
                t = torch.randint(0, timesteps, (batch_data.size(0),)).to(batch_data.device)
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    pred_x0 = model(batch_data,t)
                    loss = F.mse_loss(pred_x0,batch_data)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
                epoch_loss += loss.item()
                if (batch_idx + 1) % 100 == 0:
                    avg = epoch_loss / (batch_idx + 1)
                    print(f"Epoch {epoch} | iter {batch_idx+1}/{len(dataloader)} | loss {avg:.4f}")


            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetScoreBasedDiffusionModel(input_size=20,hidden_size=64,timesteps=1000,cond_dim=None).to(device)
    train_score_based_model(model, dataloader, num_epochs=60, timesteps=1000)
    torch.save(model.state_dict(), './results/Diffusion_model/FLU_DM_UNet_Attention_128_ch.pth')
