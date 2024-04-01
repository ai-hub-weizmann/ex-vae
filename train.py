import torch

from tqdm import tqdm

from utils import _compute_kl_weight

from typing import Optional, Tuple

__all__ = ["train_epoch", "test_epoch", "train"]


def train_epoch(
    model,
    optimizer,
    scheduler,
    kl_weight,
    loader,
    device,
    count_layer="counts",
    batch_obs: Optional[str] = "batch",
) -> Tuple[float, float]:
    """

    Trains a model for one epoch.

    Params:
    -------

    model: nn.Module
        The model.
    optimizer: optim.Optimizer
        The optimizer.
    scheduler: optim.lr_scheduler._LRScheduler
        The learning rate scheduler.
    kl_weight: float
        The weight to apply to the KL divergence term.
    loader: torch.utils.data.DataLoader
        The data loader.
    device: torch.device
        The device to run on.
    count_layer: str
        The name of the count layer in the adata.
    batch_obs: str
        The name of the batch observation in the adata.

    Returns:
    --------

    float
        The KL divergence loss.
    float
        The reconstruction loss.

    """

    model.train()

    n_cells_train = len(loader.dataset)

    train_kl_epoch, train_recon_epoch = 0, 0

    for batch in loader:

        optimizer.zero_grad()

        x_input = batch.layers[count_layer].to(device)

        # Optional if you implement batch correction in the model
        batch_covariate = torch.tensor(batch.obs[batch_obs], dtype=torch.long).to(
            device
        )

        # BEGIN SOLUTION

        # compute the forward pass here

        # p_x, q_z =

        # compute the KL part of the ELBO loss here

        # kl_loss =

        # compute the reconstruction part of the ELBO loss here (you can use the Pytorch PoissonNLLLoss here, or implement it yourself)

        # recon_loss =

        # compute the ELBO loss here

        # loss = kl_weight * kl_loss + recon_loss

        # loss.backward()
        # optimizer.step()

        # train_kl_epoch += kl_loss.item()
        # train_recon_epoch += recon_loss.item()

        # END SOLUTION

    train_kl_epoch = train_kl_epoch / n_cells_train
    train_recon_epoch = train_recon_epoch / n_cells_train

    if scheduler is not None:
        scheduler.step()

    return train_kl_epoch, train_recon_epoch


def test_epoch(
    model, loader, device, count_layer="counts", batch_obs: Optional[str] = "batch"
):
    """

    Tests a model for one epoch.

    Params:
    -------
    model: nn.Module
        The model.
    loader: torch.utils.data.DataLoader
        The data loader.
    device: torch.device
        The device to run on.
    count_layer: str
        The name of the count layer in the batch.
    batch_obs: str
        The name of the batch observation in the batch.

    Returns:
    --------
    float
        The KL divergence loss.
    float
        The reconstruction loss.

    """

    model.eval()

    n_cells_test = len(loader.dataset)

    test_kl, test_recon = 0, 0

    with torch.no_grad():
        for batch in loader:

            x_input = batch.layers[count_layer].to(device)
            batch_covariate = torch.tensor(batch.obs[batch_obs], dtype=torch.int8).to(
                device
            )

            # BEGIN SOLUTION

            # compute the forward pass here
            # p_x, q_z =

            # compute the KL part of the ELBO loss here

            # kl_loss =

            # compute the reconstruction part of the ELBO loss here

            # recon_loss =

            # test_kl += kl_loss.item()
            # test_recon += recon_loss.item()

            # END SOLUTION

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
    """

    Trains a model for a number of epochs.

    Params:
    -------
    model: nn.Module
        The model.
    optimizer: optim.Optimizer
        The optimizer.
    scheduler: optim.lr_scheduler._LRScheduler
        The learning rate scheduler.
    train_loader: torch.utils.data.DataLoader
        The training data loader.
    test_loader: torch.utils.data.DataLoader
        The test data loader.
    device: torch.device
        The device to run on.
    count_layer: str
        The name of the count layer in the adata
    batch_obs: str
        The name of the batch observation in the adata
    n_epochs: int
        The number of epochs to train for.
    test_every: int
        How often to test the model.
    n_epochs_kl_warmup: int
        The number of epochs to scale the KL divergence weight from min to max.
    max_kl_weight: float
        The maximum weight to apply to the KL divergence term.
    min_kl_weight: float
        The minimum weight to apply to the KL divergence term.
    kl_scale: float
        The scaling factor for the KL divergence.

    Returns:
    --------
    list
        The training KL divergence losses.
    list
        The training reconstruction losses.
    list
        The test KL divergence losses.
    list
        The test reconstruction losses.

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

        # train the model for one epoch here

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
