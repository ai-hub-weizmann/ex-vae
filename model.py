from typing import Callable, Optional

import torch

from torch import nn

from base_components import Encoder, PoissonDecoder


__all__ = ["VAE", "get_latent_representation"]


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
        log_variational: bool = True,
        #########################################
        # Add any other argument
    ):
        super().__init__()

        # BEGIN SOLUTION

        # self.encoder =
        #
        # self.decoder =

    def forward(self, x, batch_covariate: Optional[torch.Tensor] = None):
        # ------------------------------WRITE YOUR CODE---------------------------------#
        # generate mu and log_var from x, then sample z from the distribution N(mu, log_var) using rsample() method.
        # generate Poisson mean from z
        # Return the latent distribution and the observation distribution

        # useful transformation for the input data to make it more Gaussian - so training is easier
        if self.log_variational:
            x = torch.log1p(x)

        # BEGIN SOLUTION

        # latent_dist =
        # count_dist =

        raise NotImplementedError

        # END SOLUTION


def get_latent_representation(
    model: VAE,
    loader: Optional[Callable] = None,
    device: Optional[torch.device] = None,
    count_layer: Optional[str] = "counts",
) -> torch.Tensor:
    """Get the latent representation of the input data.

    Args:
      model (VAE): The VAE model.
      x (torch.Tensor): The input data.
      n_samples (int): The number of samples to draw from the latent space.

    Returns:
      torch.Tensor: The latent representation of the input data.
    """

    # BEGIN SOLUTION

    model.eval()

    with torch.no_grad():
        for batch in loader:
            x_input = batch.layers[count_layer].to(device)

    raise NotImplementedError

    # END SOLUTION
