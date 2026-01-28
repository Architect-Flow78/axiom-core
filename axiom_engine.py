import torch
import torch.nn as nn

class AxiomEngine(nn.Module):
    def __init__(self, d_model=384, hidden=256, inv_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.GELU()
        )

        self.invariant_gate = nn.Sequential(
            nn.Linear(hidden, inv_dim),
            nn.LayerNorm(inv_dim),
            nn.Tanh()
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.invariant_gate(h)

    def hash(self, x):
        return torch.sign(self.forward(x))
      
