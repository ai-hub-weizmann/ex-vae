from typing import Callable, Optional

import torch

from torch import nn

from base_components import Encoder, BernoulliDecoder, PoissonDecoder


__all__ = ["VAE"]


class VAE(nn.Module):
    """
    Variational Autoencoder.

    Parameters
    ----------

    n_input : int
        Number of input features.
    n_output : int
        Number of output features.
    hidden_sizes_encoder : list
        List of hidden layer sizes for the encoder network.
    hidden_sizes_decoder : list
        List of hidden layer sizes for the decoder network.
    latent_dim : int
        Dimensionality of the latent space.
    likelihood : str
        Likelihood of the data. Either "bernoulli" or "poisson".
    log_variational : bool
        Whether to take the log of the input data before encoding it.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_sizes_encoder: list = [256, 128],
        hidden_sizes_decoder: list = [128, 256],
        latent_dim: int = 2,
        likelihood: str = "bernoulli",
        log_variational: bool = False,
    ):
        super().__init__()

        self.likelihood = likelihood
        self.log_variational = log_variational

        self.encoder = Encoder(n_input, latent_dim, hidden_sizes_encoder)
        if likelihood == "bernoulli":
            self.decoder = BernoulliDecoder(latent_dim, n_output, hidden_sizes_decoder)
        elif likelihood == "poisson":
            self.decoder = PoissonDecoder(latent_dim, n_output, hidden_sizes_decoder)

    def forward(self, x):
        # ------------------------------WRITE YOUR CODE---------------------------------#
        # generate mu and log_var from x, then sample z from the distribution N(mu, log_var) using rsample() method.
        # generate mean from z

        if self.log_variational:
            x = torch.log(x + 1)

        latent_dist, latent_sample = self.encoder(x)
        obs_dist = self.decoder(latent_sample)

        return obs_dist, latent_dist
