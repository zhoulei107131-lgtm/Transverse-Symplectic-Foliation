import torch


def chain_edges(n: int) -> torch.Tensor:
    """Build directed edges for an undirected chain graph."""
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    return torch.tensor(edges, dtype=torch.long)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Aggregate src rows by index using addition on dim 0."""
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out
