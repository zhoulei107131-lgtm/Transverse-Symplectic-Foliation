import torch

from tinngnn.models import EnergyDissipationModel


def step_split(
    model: EnergyDissipationModel,
    x: torch.Tensor,
    edges: torch.Tensor,
    dt: float,
    eps: float,
) -> torch.Tensor:
    """One splitting step with reversible and irreversible substeps."""
    q = x[:, 0:1].requires_grad_(True)
    p = x[:, 1:2]
    s = x[:, 2:3]

    u, _ = model.potential_U(q, edges)
    dU_dq = torch.autograd.grad(u, q, create_graph=True)[0]
    p_half = p - dt * dU_dq
    q_new = q + dt * p_half

    h_new = model.gnn(q_new.detach(), edges)
    gamma = model.gamma(h_new)
    lam = (eps**2) * gamma * dt
    p_new = p_half * torch.exp(-lam)
    s_new = s + 0.5 * (p_half**2) * (1.0 - torch.exp(-2.0 * lam))

    return torch.cat([q_new.detach(), p_new, s_new], dim=-1)
