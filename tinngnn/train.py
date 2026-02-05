import argparse

import torch
import torch.nn.functional as F

from tinngnn.data import simulate_ground_truth
from tinngnn.graph import chain_edges
from tinngnn.integrator import step_split
from tinngnn.models import EnergyDissipationModel


def train(
    device: str = "cpu",
    N: int = 8,
    steps: int = 5,
    dt: float = 0.05,
    lr: float = 1e-3,
    hidden: int = 64,
    mp_steps: int = 2,
    epochs: int = 2,
    batch_size: int = 8,
) -> EnergyDissipationModel:
    torch.manual_seed(0)
    edges = chain_edges(N).to(device)

    model = EnergyDissipationModel(hidden=hidden, mp_steps=mp_steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    eps_schedule = [0.2]

    for stage, eps in enumerate(eps_schedule):
        for epoch in range(epochs):
            traj = simulate_ground_truth(N, edges, steps=steps, dt=dt, eps=eps, device=device)
            t_idx = torch.randint(0, steps, (batch_size,), device=device)
            loss = 0.0

            for t in t_idx:
                x_t = traj[t]
                x_tp1 = traj[t + 1]
                x_pred = step_split(model, x_t, edges, dt=dt, eps=eps)

                loss_data = F.mse_loss(x_pred, x_tp1)

                q_t, p_t, s_t = x_t[:, 0:1], x_t[:, 1:2], x_t[:, 2:3]
                q_p, p_p, s_p = x_pred[:, 0:1], x_pred[:, 1:2], x_pred[:, 2:3]
                E_t = model.energy(q_t, p_t, s_t, edges)
                E_p = model.energy(q_p, p_p, s_p, edges)
                loss_E = (E_p - E_t).pow(2)

                loss = loss + loss_data + 0.1 * loss_E

            loss = loss / t_idx.numel()
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(
                f"[stage {stage + 1}/{len(eps_schedule)} eps={eps:.2f}] "
                f"epoch {epoch + 1}/{epochs} loss={loss.item():.4e}"
            )

    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--mp_steps", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    _ = train(
        device=args.device,
        N=args.N,
        steps=args.steps,
        dt=args.dt,
        lr=args.lr,
        hidden=args.hidden,
        mp_steps=args.mp_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
