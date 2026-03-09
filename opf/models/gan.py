import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim: int, x_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        c = self.label_emb(y)
        inp = torch.cat([z, c], dim=1)
        return self.net(inp)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, x_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        c = self.label_emb(y)
        inp = torch.cat([x, c], dim=1)
        return self.net(inp).squeeze(-1)
