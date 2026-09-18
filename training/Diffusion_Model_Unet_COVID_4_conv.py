import torch
import torch.nn as nn
import math
from typing import Optional
import torch.nn.functional as F


def fixed_sinusoidal_embedding(dim, max_pos=10000):

    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
    pos = torch.arange(max_pos).float().unsqueeze(1)
    ang = pos * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(max_pos, 1)], dim=-1)
    return emb


def apply_rotary_pos_emb(q, k, sinusoidal_pos):

    L = q.size(-2)
    hd = q.size(-1)
    sin = sinusoidal_pos[:L, :hd].unsqueeze(0)[:, :, :hd//2]  # (1,L,hd/2)
    cos = sinusoidal_pos[:L, hd//2:hd].unsqueeze(0)[:, :, :hd//2]

    q_ = q.view(*q.shape[:-1], hd//2, 2)
    k_ = k.view(*k.shape[:-1], hd//2, 2)

    q_rot0 = q_[..., 0] * cos - q_[..., 1] * sin
    q_rot1 = q_[..., 0] * sin + q_[..., 1] * cos
    k_rot0 = k_[..., 0] * cos - k_[..., 1] * sin
    k_rot1 = k_[..., 0] * sin + k_[..., 1] * cos

    q_out = torch.stack([q_rot0, q_rot1], dim=-1).view(*q.shape)
    k_out = torch.stack([k_rot0, k_rot1], dim=-1).view(*k.shape)
    return q_out, k_out


class FiLM(nn.Module):

    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, feature_dim * 2),
        )

    def forward(self, x, cond):

        B, L, C = x.shape
        gam_shift = self.net(cond)  # (B, 2*C)
        gamma, beta = gam_shift.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)  # (B,1,C)
        beta = beta.unsqueeze(1)
        return x * (1 + gamma) + beta


class LightweightRoPEAttention(nn.Module):
    def __init__(self, dim, num_heads=4, max_pos=8192):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.max_pos = max_pos
        sinus = fixed_sinusoidal_embedding(self.head_dim, max_pos=self.max_pos)
        self.register_buffer("sinusoidal_table", sinus, persistent=False)
        self.scale = (self.head_dim) ** -0.5

    def forward(self, x):

        B, L, D = x.shape
        q = self.to_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, hd)
        k = self.to_k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q_ = q.transpose(1, 2).reshape(B * self.num_heads, L, self.head_dim)
        k_ = k.transpose(1, 2).reshape(B * self.num_heads, L, self.head_dim)
        sinus = self.sinusoidal_table.to(x.device)  # (max_pos, head_dim)
        qr, kr = apply_rotary_pos_emb(q_, k_, sinus)
        qr = qr.view(B, self.num_heads, L, self.head_dim)
        kr = kr.view(B, self.num_heads, L, self.head_dim)

        attn = torch.matmul(qr, kr.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, L, hd)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.to_out(out)


class ConvBlock1DFiLM(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.ln = nn.LayerNorm(out_ch)
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.film = FiLM(cond_dim, out_ch)
        else:
            self.film = None

    def forward(self, x, cond=None):

        out = self.conv(x)
        out = out.transpose(1, 2)  # (B, L, C)
        out = self.ln(out)
        if self.film is not None and cond is not None:
            out = self.film(out, cond)
        out = out.transpose(1, 2)
        return out


class UpConv1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)

## flexible revision for up conv    
class UpConv1D_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(size=1273, mode='nearest'),
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)



class UNetScoreBasedDiffusionModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, timesteps=1000, cond_dim: Optional[int] = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.cond_dim = cond_dim
        self.alphabet_size = 20
        self.seq_len = 1273

        self.noise_schedule = torch.linspace(0, 1, timesteps)

        n1 = hidden_size
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        

        self.Conv1 = ConvBlock1DFiLM(input_size, filters[0], cond_dim=None)
        self.Maxpool1 = nn.MaxPool1d(2)
        self.Conv2 = ConvBlock1DFiLM(filters[0], filters[1], cond_dim=None)
        self.Maxpool2 = nn.MaxPool1d(2)
        self.Conv3 = ConvBlock1DFiLM(filters[1], filters[2], cond_dim=None)
        self.Maxpool3 = nn.MaxPool1d(2)
        self.Conv4 = ConvBlock1DFiLM(filters[2], filters[3], cond_dim=None)
        self.Maxpool4 = nn.MaxPool1d(2)

        self.bottleneck_attn = LightweightRoPEAttention(filters[3], num_heads=8, max_pos=self.seq_len//8 + 4) ## max_pos=self.seq_len//2^(num_maxpooling)+4


        self.Up4 = UpConv1D(filters[3], filters[2])
        self.Up_conv4 = ConvBlock1DFiLM(filters[3], filters[2], cond_dim=None)

        self.Up3 = UpConv1D(filters[2], filters[1])
        self.Up_conv3 = ConvBlock1DFiLM(filters[2], filters[1], cond_dim=None)

        self.Up2 = UpConv1D_2(filters[1], filters[0])
        self.Up_conv2 = ConvBlock1DFiLM(filters[1], filters[0], cond_dim=None)

        self.Conv_out = nn.Conv1d(filters[0], input_size, kernel_size=1)

    def forward(self, x, t):

        noise = torch.randn_like(x)
        t = t.long()
        sig = self.noise_schedule.to(x.device)[t].view(-1, 1, 1)
        noisy_input = x + noise * sig

        x_in = noisy_input.transpose(1, 2)  # (B, C, L)
        e1 = self.Conv1(x_in)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        b = e4.transpose(1, 2)  # (B, L_b, D)
        b = self.bottleneck_attn(b)
        b = b.transpose(1, 2)

        d4 = self.Up4(b)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_out(d2)
        return out.transpose(1, 2)
    
    def all_likelihood_components(self, x):

        noise = torch.randn_like(x)
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
        sig = self.noise_schedule.to(x.device)[t].view(-1, 1, 1)
        noisy_input = x + noise * sig

        x_in = noisy_input.transpose(1, 2)  # (B, C, L)
        e1 = self.Conv1(x_in)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        b = e4.transpose(1, 2)  # (B, L_b, D)
        b = self.bottleneck_attn(b)
        b = b.transpose(1, 2)

        d4 = self.Up4(b)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_out(d2)
        out = out.transpose(1, 2)

        out = out.reshape(-1,self.alphabet_size*self.seq_len)
        x = x.reshape(-1,self.alphabet_size*self.seq_len)
        
        BCE_batch_tensor = torch.sum(F.mse_loss(x, out, reduction='none'),dim=1)
        
        ELBO_batch_tensor = -(BCE_batch_tensor)

        return ELBO_batch_tensor

