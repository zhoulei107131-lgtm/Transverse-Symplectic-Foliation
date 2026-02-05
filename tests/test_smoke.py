import torch

import tinngnn
from tinngnn.data import simulate_ground_truth
from tinngnn.graph import chain_edges
from tinngnn.integrator import step_split
from tinngnn.models import EnergyDissipationModel


def test_smoke_train_step() -> None:
    torch.manual_seed(0)
    device = "cpu"
    n_nodes = 4
    steps = 3
    dt = 0.05
    eps = 0.2

    edges = chain_edges(n_nodes).to(device)
    model = EnergyDissipationModel(hidden=16, mp_steps=1).to(device)
    assert tinngnn is not None

    traj = simulate_ground_truth(n_nodes, edges, steps=steps, dt=dt, eps=eps, device=device)
    x_t = traj[0]
    x_tp1 = traj[1]

    x_pred = step_split(model, x_t, edges, dt=dt, eps=eps)
    assert x_pred.shape == x_tp1.shape

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        opt.zero_grad()
        x_pred = step_split(model, x_t, edges, dt=dt, eps=eps)
        loss = torch.nn.functional.mse_loss(x_pred, x_tp1)
        loss.backward()
        opt.step()
