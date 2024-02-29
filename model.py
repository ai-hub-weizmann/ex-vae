from typing import Callable, Optional

import torch

from torch import nn

from base_components import Encoder, BernoulliDecoder, PoissonDecoder


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


class VAE_batchcorr(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_batch: int,
        concat_batch: bool = False,
    ):
        super().__init__()

        self.n_batch = n_batch
        self.n_latent = latent_dim
        self.concat_batch = concat_batch

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        self.batch_embedding = nn.Embedding(
            num_embeddings=self.n_batch, embedding_dim=self.n_latent
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim if self.concat_batch else latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus(),  # enforce positivity
        )

    def forward(self, x, batch_covariate: Optional[torch.Tensor] = None):

        x = torch.log(x + 1)

        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        var = torch.nn.Softplus()(log_var) + 1e-6

        latent_dist = torch.distributions.Normal(mu, var.sqrt())

        latent_sample = mu + var.sqrt() * torch.randn_like(mu)

        if self.concat_batch:
            # batch_embedding = torch.functional.F.one_hot(
            #     batch_covariate, num_classes=self.n_batch
            # ).float()

            # transform batch_covariate to Long tensor:
            batch_covariate = batch_covariate.long()

            batch_embedding = self.batch_embedding(batch_covariate)
            latent_sample = torch.cat([latent_sample, batch_embedding], dim=-1)

        obs_rates = self.decoder(latent_sample)

        obs_dist = torch.distributions.Poisson(obs_rates)

        return obs_dist, latent_dist


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

    model.eval()

    qz_mean = []

    with torch.no_grad():
        for batch in loader:
            x_input = batch.layers[count_layer].to(device)

            x = model.encoder(x_input)
            mu = model.mu(x)

            # obs_dist, latent_dist = model(x_input)

            # qz_mean.append(latent_dist.loc.cpu())

            qz_mean.append(mu.cpu())

        qz_mean = torch.cat(qz_mean, dim=0)

    return qz_mean.numpy()
