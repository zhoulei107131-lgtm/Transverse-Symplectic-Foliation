# tinngnn (Tiny Non-Equilibrium GNN MVP)

This repository restructures the provided single-file prototype into a small testable Python package.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run training (quick CPU run)

```bash
python -m tinngnn.train --device cpu --steps 5
```

Default training settings are intentionally small so the run finishes quickly in cloud environments.

## Run tests

```bash
pytest -q
```

## Minimal example

```python
import torch
from tinngnn.graph import chain_edges
from tinngnn.models import EnergyDissipationModel
from tinngnn.integrator import step_split

N = 8
edges = chain_edges(N)
model = EnergyDissipationModel(hidden=64, mp_steps=2)
x = torch.randn(N, 3)
x_next = step_split(model, x, edges, dt=0.05, eps=0.2)
print(x_next.shape)  # torch.Size([8, 3])
```
