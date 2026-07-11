import torch
import torch.nn as nn

from tqdm import tqdm


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
):

    model.train()

    criterion = nn.MSELoss()

    running_loss = 0

    dead_block_counter = None

    for features in tqdm(dataloader):

        ###################################
        # Features
        ###################################

        features = features.to(device)

        ###################################
        # Forward
        ###################################

        output = model(features)

        reconstruction = output["reconstruction"]

        ###################################
        # Reconstruction loss
        ###################################

        loss = criterion(
            reconstruction,
            features,
        )

        ###################################
        # Backprop
        ###################################

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        ###################################
        # Statistics
        ###################################

        running_loss += loss.item()

        mask = output["active_mask"]

        if dead_block_counter is None:

            dead_block_counter = mask.sum(0)

        else:

            dead_block_counter += mask.sum(0)

    epoch_loss = running_loss / len(dataloader)

    dead_blocks = (dead_block_counter == 0).sum().item()

    return epoch_loss, dead_blocks


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
):

    model.eval()

    criterion = nn.MSELoss()

    running_loss = 0

    for features in dataloader:

        features = features.to(device)

        output = model(features)

        reconstruction = output["reconstruction"]

        loss = criterion(
            reconstruction,
            features,
        )

        running_loss += loss.item()

    return running_loss / len(dataloader)


def fit(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-4,
    device="cuda",
):

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):

        train_loss, dead_blocks = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_loss = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"Dead Blocks {dead_blocks}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "best_model.pth",
            )