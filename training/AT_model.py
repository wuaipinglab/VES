import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        return emb

def apply_rotary_pos_emb(q, k, rope):
    seq_len = q.size(2)
    rope = rope[:seq_len, :].to(q.device)  # [L, D]
    cos, sin = rope.cos(), rope.sin()
    q_cos, q_sin = q * cos.unsqueeze(0).unsqueeze(0), q * sin.unsqueeze(0).unsqueeze(0)
    k_cos, k_sin = k * cos.unsqueeze(0).unsqueeze(0), k * sin.unsqueeze(0).unsqueeze(0)
    q_rot = torch.cat((-q_sin[..., ::2], q_cos[..., 1::2]), dim=-1)
    k_rot = torch.cat((-k_sin[..., ::2], k_cos[..., 1::2]), dim=-1)
    return q_rot, k_rot

class MultiheadAttentionRoPE(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.o_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.d_head)

    def forward(self, x, causal_mask=True):
        B, L, D = x.size()
        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.nhead, self.d_head).transpose(1, 2)  # (B,H,L,Dh)
        k = k.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.d_head).transpose(1, 2)

        rope = self.rope(L, x.device)
        q, k = apply_rotary_pos_emb(q, k, rope)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,L,L)
        if causal_mask:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B,H,L,Dh)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff=1024):
        super().__init__()
        self.attn = MultiheadAttentionRoPE(d_model, nhead)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class ARTransformerRoPE(nn.Module):
    def __init__(self, vocab_size, seq_len=565, d_model=256, nhead=8, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, nhead) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.alphabet_size = 21

    def forward(self, x):
        x = self.embedding(x)  # (B,L,D)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.fc_out(x)  # (B,L,V)
    
    def all_likelihood_components(self, input):
        input = input.to(torch.long)
        batch_size = input.shape[0]
        target = input[:,1:]
        x = self.embedding(input[:,:-1])  # (B,L,D)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        out = self.fc_out(x)


        loss = F.cross_entropy(out.reshape(-1,self.alphabet_size), target.reshape(-1), reduction='none')
        loss = loss.view(batch_size, -1)
        
        CE_batch_tensor = torch.sum(loss,dim=1)
        ELBO_batch_tensor = -(CE_batch_tensor)

        return ELBO_batch_tensor
