"""
References:
- ProteinGAN official repo (TensorFlow): https://github.com/Biomatter-Designs/ProteinGAN.
- Repecka et al., Expanding functional protein sequence space using GANs (preprint / paper).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# User config / hyperparams
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_SPECTRAL_NORM = True         # original repo used spectral norm for stability (SNGAN)
N_RES_BLOCKS = 2                # number of residual blocks to stack at chosen positions
SELF_ATTENTION_POS = "mid"      # "early" | "mid" | "late" | None

def sn(module):
    if USE_SPECTRAL_NORM:
        return nn.utils.spectral_norm(module)
    return module

class SelfAttention1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, L = x.size()

        query = self.query(x).permute(0, 2, 1)   # B, L, C'
        key   = self.key(x)                      # B, C', L
        attn  = torch.bmm(query, key)            # B, L, L
        attn  = F.softmax(attn, dim=-1)

        value = self.value(x).permute(0, 2, 1)   # B, L, C

        out = torch.bmm(attn, value).permute(0, 2, 1)  # B, C, L
        out = self.gamma * out + x
        return out

# ---------------------------
# Residual blocks
# ---------------------------
class DResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Shortcut path
        self.shortcut = nn.Conv1d(in_channels, out_channels, 
                                  kernel_size=3, stride=2, padding=1)

        # Main path
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, stride=2, padding=1)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)

        return out + shortcut

class GResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Shortcut: transposed conv upsample
        self.shortcut = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # Main path
        self.act1 = nn.LeakyReLU(0.2)
        self.deconv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.act2 = nn.LeakyReLU(0.2)
        self.conv = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.act1(x)
        out = self.deconv(out)
        out = self.act2(out)
        out = self.conv(out)

        return out + shortcut

# ---------------------------
# Generator
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, vocab=20, base=64):
        super().__init__()

        self.input_proj = nn.Conv1d(vocab, base, kernel_size=3, padding=1)

        self.res1 = DResBlock(base, base)
        self.attn = SelfAttention1D(base)

        self.res2 = DResBlock(base, base)
        self.res3 = DResBlock(base, base)
        self.res4 = DResBlock(base, base)
        self.res5 = DResBlock(base, base)

        self.final_conv = nn.Conv1d(base, base, kernel_size=3, padding=1)
        self.score = nn.Linear(base, 1)

    def forward(self, x):          # x: (B, L, vocab)
        x = x.permute(0,2,1)       # to (B, vocab, L)
        x = self.input_proj(x)

        x = self.res1(x)
        x = self.attn(x)

        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = self.final_conv(x)

        # global pooling → score
        x = x.mean(dim=2)
        return self.score(x)
    
    def all_likelihood_components_wo_noise(self, x):
        x = x.permute(0,2,1)       # to (B, vocab, L)
        x = self.input_proj(x)

        x = self.res1(x)
        x = self.attn(x)

        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = self.final_conv(x)

        x = x.reshape(-1, x.size[1]*x.size[2])
        

    
# ---------------------------
# Discriminator / Critic
# ---------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=128, base=64, vocab=20, init_len=16):
        super().__init__()

        self.fc = nn.Linear(z_dim, base * init_len)
        self.init_len = init_len
        self.base = base

        self.res1 = GResBlock(base, base)
        self.res2 = GResBlock(base, base)
        self.res3 = GResBlock(base, base)
        self.res4 = GResBlock(base, base)

        self.attn = SelfAttention1D(base)

        self.res5 = GResBlock(base, base)

        self.bn = nn.BatchNorm1d(base)
        self.final = nn.Conv1d(base, vocab, kernel_size=3, padding=1)
        self.reshape_ge = nn.Linear(512,565)

    def forward(self, z):
        out = self.fc(z)
        out = out.view(z.size(0), self.base, self.init_len)

        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        out = self.attn(out)

        out = self.res5(out)

        out = self.bn(out)
        out = self.final(out)
        out = self.reshape_ge(out)
        return out.permute(0,2,1)   # → (B, L, vocab logits)
