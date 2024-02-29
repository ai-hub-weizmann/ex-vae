import torch

from tqdm import tqdm

from utils import _compute_kl_weight

from typing import Optional

__all__ = ["train_epoch", "test_epoch", "train"]


def train_epoch(
    model,
    optimizer,
    scheduler,
    kl_weight,
    loader,
    device,
    kl_scale=1.0,
    count_layer="counts",
    batch_obs: Optional[str] = "batch",
):
    """Trains over an epoch.

    Args:
      model (nn.Module): The model.
      criterion (callable): The loss function. Should return a scalar tensor.
      optimizer (optim.Optimizer): The optimizer.
      loader (torch.utils.data.DataLoader): The test set data loader.
      device (torch.device): The device to run on.

    Returns:
      acc_metric (Metric): The accuracy metric over the epoch.
      loss_metric (Metric): The loss metric over the epoch.
    """

    model.train()

    n_cells_train = len(loader.dataset)

    train_kl_epoch, train_recon_epoch = 0, 0

    for batch in loader:

        optimizer.zero_grad()

        x_input = batch.layers[count_layer].to(device)
        batch_covariate = torch.tensor(batch.obs[batch_obs], dtype=torch.long).to(
            device
        )

        p_x, q_z = model(x_input, batch_covariate)

        # ------------------------------WRITE YOUR CODE---------------------------------#
        # compute the KL part of the ELBO loss here

        # kl_loss = dkl(q_z).sum(axis=1).mean()
        kl_loss = (
            torch.distributions.kl_divergence(q_z, torch.distributions.Normal(0, 1))
            .sum(axis=1)
            .sum()
        )

        # ------------------------------WRITE YOUR CODE---------------------------------#
        # compute the reconstruction part of the ELBO loss here

        # recon_loss = -p_x.log_prob(x_input).sum()

        recon_loss = torch.nn.PoissonNLLLoss(log_input=False, reduction="sum")(
            p_x.mean, x_input
        )

        # ------------------------------WRITE YOUR CODE---------------------------------#
        # compute the ELBO loss here

        loss = kl_scale * kl_weight * kl_loss + recon_loss
        loss.backward()
        optimizer.step()

        train_kl_epoch += kl_loss.item()
        train_recon_epoch += recon_loss.item()

    train_kl_epoch = train_kl_epoch / n_cells_train
    train_recon_epoch = train_recon_epoch / n_cells_train

    if scheduler is not None:
        scheduler.step()

    return train_kl_epoch, train_recon_epoch


def test_epoch(
    model, loader, device, count_layer="counts", batch_obs: Optional[str] = "batch"
):

    model.eval()

    n_cells_test = len(loader.dataset)

    test_kl, test_recon = 0, 0

    with torch.no_grad():
        for batch in loader:

            x_input = batch.layers[count_layer].to(device)
            batch_covariate = torch.tensor(batch.obs[batch_obs], dtype=torch.int8).to(
                device
            )

            p_x, q_z = model(x_input, batch_covariate)

            # ------------------------------WRITE YOUR CODE---------------------------------#
            # compute the KL part of the ELBO loss here

            kl_loss = (
                torch.distributions.kl_divergence(q_z, torch.distributions.Normal(0, 1))
                .sum(axis=1)
                .sum()
            )

            # ------------------------------WRITE YOUR CODE---------------------------------#
            # compute the reconstruction part of the ELBO loss here

            recon_loss = -p_x.log_prob(x_input).sum()

            test_kl += kl_loss.item()
            test_recon += recon_loss.item()

    test_kl = test_kl / n_cells_test
    test_recon = test_recon / n_cells_test

    return test_kl, test_recon


def train_loop(
    model,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    device,
    count_layer="counts",
    batch_obs: Optional[str] = "batch",
    n_epochs=100,
    test_every=1,
    n_epochs_kl_warmup=10,
    max_kl_weight=1.0,
    min_kl_weight=0.0,
    kl_scale=1.0,
):
    """Trains a model to minimize some loss function and reports the progress.

    Args:
      model (nn.Module): The model.
      criterion (callable): The loss function. Should return a scalar tensor.
      optimizer (optim.SGD): The optimizer.
      train_loader (torch.utils.data.DataLoader): The training set data loader.
      test_loader (torch.utils.data.DataLoader): The test set data loader.
      device (torch.device): The device to run on.
      epochs (int): Number of training epochs.
      test_every (int): How frequently to report progress on test data.
    """

    train_kl, train_recon = [], []
    test_kl, test_recon = [], []

    for epoch in tqdm(range(1, n_epochs + 1)):

        kl_weight = _compute_kl_weight(
            epoch,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            max_kl_weight=max_kl_weight,
            min_kl_weight=min_kl_weight,
        )

        train_kl_epoch, train_recon_epoch = train_epoch(
            model,
            optimizer,
            scheduler,
            kl_weight,
            train_loader,
            device,
            kl_scale=kl_scale,
            count_layer=count_layer,
            batch_obs=batch_obs,
        )

        if epoch % test_every == 0:
            test_kl_epoch, test_recon_epoch = test_epoch(
                model, test_loader, device, count_layer=count_layer, batch_obs=batch_obs
            )

            train_kl.append(train_kl_epoch)
            train_recon.append(train_recon_epoch)

            test_kl.append(test_kl_epoch)
            test_recon.append(test_recon_epoch)

    return train_kl, train_recon, test_kl, test_recon
