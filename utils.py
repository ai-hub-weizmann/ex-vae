from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GINConv, TransformerConv

Distribution = torch.distributions.Distribution


def make_layer(
    in_dim: int,
    out_dim: int,
    add_act: bool = True,
    act: Callable[..., torch.nn.Module] = torch.nn.ReLU,
    use_norm: bool = True,
    norm: str = "batch",
) -> torch.nn.Module:
    if norm == "batch":
        norm_l = torch.nn.BatchNorm1d(out_dim)
    if norm == "layer":
        norm_l = torch.nn.LayerNorm(out_dim, elementwise_affine=False)

    layers = [torch.nn.Linear(in_dim, out_dim)]
    if use_norm:
        layers += [norm_l]
    if add_act:
        layers += [act()]
    return torch.nn.Sequential(*layers)


def dkl(q: Distribution):
    """Analytical solution for D_KL(Q || P), the KL divergence of two Gaussian random variables Q and P.
    D_KL(Q || P) = E_Q [ log(Q / P) ]
    where P is the standard normal
    Args:
        mu: mean of Q
        sigma: standard deviation of Q
    """
    mu = q.loc
    sigma = q.scale
    return 0.5 * (mu**2 + sigma**2 - 1) - sigma.log()
