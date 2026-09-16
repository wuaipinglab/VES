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
from tqdm import tqdm

from ProteinGAN import Generator, Discriminator
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

def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, device, lambda_gp=10.0) -> torch.Tensor:
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    interp = alpha * real + (1 - alpha) * fake
    interp.requires_grad_(True)
    critic_out = critic(interp)
    grads = torch.autograd.grad(
        outputs=critic_out,
        inputs=interp,
        grad_outputs=torch.ones_like(critic_out, device=device),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]  # (B, L, vocab)
    grads = grads.reshape(B, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gp


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Baseline_ProteinGAN')
    parser.add_argument('--MSA_data_folder', default='./data/MSA', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', default='./data/mappings/example_mapping.csv', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', default=0, type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', default= './data/weights', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--model_name_suffix', default='Dec2_FLU_ProteinGAN', type=str, help='model checkpoint name will be the protein name followed by this suffix')
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
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True) 

    def train_score_based_model(dataloader, device, num_epochs):
        G = Generator(z_dim=128, base=64, vocab=20, init_len=16).to(device)
        D = Discriminator(vocab=20, base=64).to(device)
        opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
        opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
        fixed_z = torch.randn(16, 128, device=device)
        global_step = 0
        save_dir = './results/ProteinGAN/'
        for epoch in range(num_epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for real in pbar:
                real = real.to(device).float()  # (B, L, vocab)
                for _ in range(5):
                    z = torch.randn(real.shape[0], 128, device=device)
                    fake = G(z)  # (B, L, vocab)
                    opt_D.zero_grad()
                    real_scores = D(real)
                    fake_scores = D(fake.detach())
                    gp = gradient_penalty(D, real, fake.detach(), device=device, lambda_gp=10.0)
                    d_loss = fake_scores.mean() - real_scores.mean() + gp
                    d_loss.backward()
                    opt_D.step()
                opt_G.zero_grad()
                z = torch.randn(real.shape[0], 128, device=device)
                fake = G(z)
                g_loss = -D(fake).mean()
                g_loss.backward()
                opt_G.step()

                global_step += 1
                pbar.set_postfix({'d_loss': d_loss.item(), 'g_loss': g_loss.item()})
            torch.save({'G': G.state_dict(), 'D': D.state_dict()}, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            with torch.no_grad():
                fake_fixed = G(fixed_z).cpu().numpy()  # (16, L, vocab)
                seqs = np.argmax(fake_fixed, axis=-1)  # (16, L)
                np.save(os.path.join(save_dir, f'generated_epoch_{epoch+1}.npy'), seqs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_score_based_model(dataloader, device, num_epochs=100)
