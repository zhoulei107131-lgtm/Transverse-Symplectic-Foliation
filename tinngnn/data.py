import math

import torch


def simulate_ground_truth(
    N: int,
    edges: torch.Tensor,
    steps: int,
    dt: float,
    eps: float,
    k: float = 1.0,
    gamma0: float = 0.2,
    device: str = "cpu",
) -> torch.Tensor:
    """Ground truth via springs + linear friction using same splitting form."""
    q = torch.randn(N, 1, device=device)
    p = torch.randn(N, 1, device=device)
    s = torch.zeros(N, 1, device=device)

    traj = []
    for _ in range(steps):
        x = torch.cat([q, p, s], dim=-1)
        traj.append(x)

        q_req = q.requires_grad_(True)
        send, recv = edges[:, 0], edges[:, 1]
        qdiff = q_req[send] - q_req[recv]
        u_dir = 0.25 * k * (qdiff**2).sum()
        dU_dq = torch.autograd.grad(u_dir, q_req)[0]

        p_half = p - dt * dU_dq
        q_new = q + dt * p_half

        lam = (eps**2) * gamma0 * dt
        p_new = p_half * math.exp(-lam)
        s_new = s + 0.5 * (p_half**2) * (1.0 - math.exp(-2.0 * lam))

        q, p, s = q_new.detach(), p_new, s_new

    traj.append(torch.cat([q, p, s], dim=-1))
    return torch.stack(traj, dim=0)
