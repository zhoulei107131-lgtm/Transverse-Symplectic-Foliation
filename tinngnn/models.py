import torch
import torch.nn as nn
import torch.nn.functional as F

from tinngnn.graph import scatter_add


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyMPNN(nn.Module):
    """Node embedding h_i from q_i using message passing on edges."""

    def __init__(self, hidden: int = 64, mp_steps: int = 2) -> None:
        super().__init__()
        self.mp_steps = mp_steps
        self.node_enc = MLP(1, hidden, hidden)
        self.edge_mlp = MLP(2 * hidden + 1, hidden, hidden)
        self.node_upd = MLP(2 * hidden, hidden, hidden)

    def forward(self, q: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        n_nodes = q.size(0)
        h = self.node_enc(q)
        send = edges[:, 0]
        recv = edges[:, 1]
        for _ in range(self.mp_steps):
            h_send = h[send]
            h_recv = h[recv]
            qdiff = q[send] - q[recv]
            m_ij = self.edge_mlp(torch.cat([h_send, h_recv, qdiff], dim=-1))
            m_i = scatter_add(m_ij, recv, n_nodes)
            h = self.node_upd(torch.cat([h, m_i], dim=-1))
        return h


class EnergyDissipationModel(nn.Module):
    """Energy model with learned potential and nonnegative dissipation."""

    def __init__(self, hidden: int = 64, mp_steps: int = 2) -> None:
        super().__init__()
        self.gnn = TinyMPNN(hidden=hidden, mp_steps=mp_steps)
        self.node_potential = MLP(hidden, hidden, 1)
        self.edge_potential = MLP(2 * hidden + 1, hidden, 1)
        self.gamma_head = MLP(hidden, hidden, 1)

    def potential_U(self, q: torch.Tensor, edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.gnn(q, edges)
        psi = self.node_potential(h).sum()
        send, recv = edges[:, 0], edges[:, 1]
        qdiff = q[send] - q[recv]
        phi_dir = self.edge_potential(torch.cat([h[send], h[recv], qdiff], dim=-1))
        phi = 0.5 * phi_dir.sum()
        return psi + phi, h

    def gamma(self, h: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.gamma_head(h))

    def energy(self, q: torch.Tensor, p: torch.Tensor, s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        u, _ = self.potential_U(q, edges)
        kinetic = 0.5 * (p**2).sum()
        internal = s.sum()
        return kinetic + u + internal
