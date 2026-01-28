import torch
import torch.nn.functional as F

def contrastive_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    batch = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)

    sim = torch.matmul(reps, reps.T)
    mask = torch.eye(2 * batch, device=z1.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    positives = torch.sum(z1 * z2, dim=-1)
    positives = torch.cat([positives, positives], dim=0)

    num = torch.exp(positives / temperature)
    den = torch.sum(torch.exp(sim / temperature), dim=-1)

    loss = -torch.log(num / den)
    return loss.mean()
  
