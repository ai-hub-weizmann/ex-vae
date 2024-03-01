from typing import Optional

import torch


__all__ = ["dkl", "_compute_kl_weight"]


Distribution = torch.distributions.Distribution


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


def _compute_kl_weight(
    epoch: int,
    n_epochs_kl_warmup: Optional[int],
    step: Optional[int] = None,
    n_steps_kl_warmup: Optional[int] = None,
    max_kl_weight: float = 1.0,
    min_kl_weight: float = 0.0,
) -> float:
    """Computes the kl weight for the current step or epoch.
    If both `n_epochs_kl_warmup` and `n_steps_kl_warmup` are None `max_kl_weight` is returned.

    Code from scvi-tools package.

    Parameters
    ----------
    epoch
        Current epoch.
    step
        Current step.
    n_epochs_kl_warmup
        Number of training epochs to scale weight on KL divergences from
        `min_kl_weight` to `max_kl_weight`
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from
        `min_kl_weight` to `max_kl_weight`
    max_kl_weight
        Maximum scaling factor on KL divergence during training.
    min_kl_weight
        Minimum scaling factor on KL divergence during training.
    """
    if min_kl_weight > max_kl_weight:
        raise ValueError(
            f"min_kl_weight={min_kl_weight} is larger than max_kl_weight={max_kl_weight}."
        )

    slope = max_kl_weight - min_kl_weight
    if n_epochs_kl_warmup:
        if epoch < n_epochs_kl_warmup:
            return slope * (epoch / n_epochs_kl_warmup) + min_kl_weight
    elif n_steps_kl_warmup:
        if step < n_steps_kl_warmup:
            return slope * (step / n_steps_kl_warmup) + min_kl_weight
    return max_kl_weight
