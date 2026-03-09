# utils.py
import os
import json
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

class StandardScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.eps = eps

    @classmethod
    def fit(cls, x: np.ndarray):
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.maximum(std, 1e-6)
        return cls(mean, std)

    def transform(self, x: np.ndarray):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: np.ndarray):
        return x * (self.std + self.eps) + self.mean

    def to_json(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "eps": self.eps}

    @classmethod
    def from_json(cls, d):
        return cls(np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32), float(d.get("eps", 1e-8)))

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class EMA:
    """Exponential Moving Average for more stable sampling."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=(1.0 - self.decay))

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])

def save_checkpoint(path: str, payload: dict):
    torch.save(payload, path)

def load_checkpoint(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)