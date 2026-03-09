# models/transformer.py
import math
import torch
import torch.nn as nn

def sinusoidal_time_embedding(t: torch.Tensor, dim: int):
    """
    t: (B,) int64 or float
    returns: (B, dim)
    """
    half = dim // 2
    device = t.device
    t = t.float()
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / half)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=device)], dim=-1)
    return emb

class TransformerDenoiser(nn.Module):
    """
    Input: x_t (B, 18), t (B,), y (B,)
    Output: eps_hat (B, 18)
    """
    def __init__(self, x_dim: int, num_classes: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.x_dim = x_dim
        self.d_model = d_model

        self.scalar_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, x_dim, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.class_embed = nn.Embedding(num_classes, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        B = x_t.shape[0]
        # tokenize 18 dims
        xt = x_t.unsqueeze(-1)  # (B,18,1)
        h = self.scalar_embed(xt)  # (B,18,d_model)
        h = h + self.pos_embed

        t_emb = sinusoidal_time_embedding(t, self.d_model)
        t_emb = self.time_mlp(t_emb)  # (B,d_model)
        c_emb = self.class_embed(y)    # (B,d_model)

        cond = (t_emb + c_emb).unsqueeze(1)  # (B,1,d_model)
        h = h + cond  # broadcast to all tokens

        h = self.encoder(h)  # (B,18,d_model)
        eps_hat = self.out(h).squeeze(-1)  # (B,18)
        return eps_hat
