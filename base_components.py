import torch
from torch import nn

from typing import Callable, Optional

_all__ = ["Encoder", "BernoulliDecoder", "PoissonDecoder"]


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


class Encoder(nn.Module):
    """
    Encoder network for the VAE.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_sizes: list = [256, 128],
    ):
        super().__init__()

        # ------------------------------WRITE YOUR CODE---------------------------------#
        # encode a data point. This should return couple of tensors (mu, logvar) representing
        # the mean and the log variance of the Gaussian q(Z | X)

        layer_sizes = [n_input, *hidden_sizes]

        self.encoder = torch.nn.Sequential(
            *[
                make_layer(
                    in_dim=in_size,
                    out_dim=out_size,
                )
                for (in_size, out_size) in zip(layer_sizes[:-1], layer_sizes[1:])
            ]
        )

        self.mu = nn.Linear(hidden_sizes[-1], n_output)
        self.log_var = nn.Linear(hidden_sizes[-1], n_output)

    def forward(self, x, eps=1e-5):
        # ------------------------------WRITE YOUR CODE---------------------------------#
        # generate mu and log_var from x, then sample z from the distribution N(mu, log_var) using rsample() method.

        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        var = torch.nn.Softplus()(log_var) + eps

        latent_dist = torch.distributions.Normal(mu, var.sqrt())

        latent_sample = latent_dist.rsample()

        return latent_dist, latent_sample


class BernoulliDecoder(nn.Module):
    """
    Decoder network for the VAE.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_sizes: list = [128, 256],
    ):
        super().__init__()

        layer_sizes = [n_input, *hidden_sizes]

        # ------------------------------WRITE YOUR CODE---------------------------------#
        # decode a latent variable. This should return a tensor representing the mean of the
        # Bernoulli distribution p(X | Z)

        self.decoder = torch.nn.Sequential(
            *[
                make_layer(
                    in_dim=in_size,
                    out_dim=out_size,
                )
                for (in_size, out_size) in zip(layer_sizes[:-1], layer_sizes[1:])
            ]
        )

        self.mean = nn.Linear(hidden_sizes[-1], n_output)

    def forward(self, z):
        # ------------------------------WRITE YOUR CODE---------------------------------#
        # generate mean from z

        z = self.decoder(z)

        mean = torch.sigmoid(self.mean(z))

        dist = torch.distributions.Bernoulli(mean)

        return dist


class PoissonDecoder(BernoulliDecoder):
    """
    Decoder network for the Poisson VAE.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_sizes: list = [128, 256],
    ):
        super().__init__(n_input, n_output, hidden_sizes)

    def forward(self, z):
        # ------------------------------WRITE YOUR CODE---------------------------------#
        # generate mean from z

        z = self.decoder(z)

        mean = torch.nn.Softplus()(self.mean(z))

        dist = torch.distributions.Poisson(mean)

        return dist
