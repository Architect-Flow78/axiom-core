import torch
import torch.nn.functional as F

from axiom_engine import AxiomEngine
from contrastive import contrastive_loss

def noise_stability(model, x, sigma=0.01):
    noisy = x + torch.randn_like(x) * sigma
    z1 = model(x)
    z2 = model(noisy)
    return F.cosine_similarity(z1, z2).mean().item()

if __name__ == "__main__":
    torch.manual_seed(0)

    model = AxiomEngine()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = 256
    dim = 384

    data = torch.randn(batch, dim)

    print("Before training:", noise_stability(model, data))

    for step in range(800):
        x = torch.randn(batch, dim)
        x_pos = x + torch.randn_like(x) * 0.02

        z1 = model(x)
        z2 = model(x_pos)

        loss = contrastive_loss(z1, z2)

        optim.zero_grad()
        loss.backward()
        optim.step()

    print("After training:", noise_stability(model, data))
  
