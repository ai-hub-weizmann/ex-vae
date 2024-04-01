import torch
from torch import nn

from typing import Callable, Optional

_all__ = ["Encoder", "PoissonDecoder"]


def make_layer(
    in_dim: int,
    out_dim: int,
    add_act: bool = True,
    act: Callable[..., torch.nn.Module] = torch.nn.ReLU,
    use_norm: bool = True,
    norm: str = "batch",
) -> torch.nn.Module:
    """

    Parameters
    ----------
    in_dim : int
        Number of input features.
    out_dim : int
        Number of output features.
    add_act : bool
        Whether to add an activation function.
    act : Callable[..., torch.nn.Module]
        Activation function.
    use_norm : bool
        Whether to use normalization.
    norm : str
        Type of normalization. Either "batch" or "layer".

    Returns
    -------
    torch.nn.Module
        A layer.
    """

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


# class Encoder(nn.Module):
#     """
#     Encoder network for the VAE.
#     """

#     def __init__(
#         self,
#         n_input: int,
#         n_output: int,
#         hidden_sizes: list = [256, 128],
#     ):
#         super().__init__()

#         # BEGIN SOLUTION

#         # encode a data point. This should return couple of tensors (mu, logvar) representing
#         # the mean and the log variance of the Gaussian q(Z | X)

#         # layer_sizes = [n_input, *hidden_sizes]

#         # self.encoder =

#         # self.mu =
#         # self.log_var =

#         raise NotImplementedError

#         # END SOLUTION

#     def forward(self, x, eps=1e-5):

#         # BEGIN SOLUTION

#         # generate mu and log_var from x, then sample z from the distribution N(mu, log_var) using either
#         # rsample() method or your own reparametrization trick implementation. Return the distribution and the sample.

#         raise NotImplementedError

#         # END SOLUTION


# class PoissonDecoder(nn.Module):
#     """
#     Decoder network for the Poisson VAE (count data )
#     """

#     def __init__(
#         self,
#         n_input: int,
#         n_output: int,
#         hidden_sizes: list = [128, 256],
#     ):
#         super().__init__()

#         # BEGIN SOLUTION
#         # decode a latent variable. This should return a tensor representing the mean of the
#         # Poisson distribution p(X | Z)

#         # layer_sizes = [n_input, *hidden_sizes]

#         # self.decoder =
#         # self.mean =

#         raise NotImplementedError
#         # END SOLUTION

#     def forward(self, z):

#         # BEGIN SOLUTION
#         # This should return a tensor representing the mean (=rate) of the
#         # Poisson distribution p(X | Z)

#         z = self.decoder(z)

#         mean = torch.nn.Softplus()(self.mean(z))

#         dist = torch.distributions.Poisson(mean)

#         return dist

#         # END SOLUTION


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

        # BEGIN SOLUTION

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

        # END SOLUTION

    def forward(self, x, eps=1e-5):

        # BEGIN SOLUTION

        # generate mu and log_var from x, then sample z from the distribution N(mu, log_var) using either
        # rsample() method or your own reparametrization trick implementation. Return the distribution and the sample.

        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        var = torch.nn.Softplus()(log_var) + eps

        latent_dist = torch.distributions.Normal(mu, var.sqrt())

        latent_sample = latent_dist.rsample()

        return latent_dist, latent_sample

        # END SOLUTION


class PoissonDecoder(nn.Module):
    """
    Decoder network for the Poisson VAE (count data )
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_sizes: list = [128, 256],
    ):
        super().__init__()

        # BEGIN SOLUTION
        # decode a latent variable. This should return a tensor representing the mean of the
        # Poisson distribution p(X | Z)

        layer_sizes = [n_input, *hidden_sizes]

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

        # END SOLUTION

    def forward(self, z):

        # BEGIN SOLUTION
        # This should return a tensor representing the mean (=rate) of the
        # Poisson distribution p(X | Z)

        z = self.decoder(z)

        mean = torch.nn.Softplus()(self.mean(z))

        dist = torch.distributions.Poisson(mean)

        return dist

        # END SOLUTION
