import torch.nn as nn
from tqdm.auto import tqdm
from torch import device, cuda


def train_model(model: nn.Module, train_loader, criterion, optimizer, num_epochs: int):
    D = device("cuda" if cuda.is_available() else "cpu")
    model = model.to(D)
    pbar = tqdm(total=num_epochs * len(train_loader), desc="Training")
    for epoch in range(num_epochs):
        Losses = []
        for *data, target in train_loader:
            data = [d.to(D) for d in data]
            target = target.to(D)
            optimizer.zero_grad()
            output = model(*data)
            loss = criterion(output, target)
            Losses.append(loss.item())
            loss.backward()
            optimizer.step()

            pbar.update(1)
        pbar.set_postfix(
            {
                "Loss": f"{sum(Losses) / len(Losses):.4f}",
                "Epoch": f"{epoch + 1}/{num_epochs}",
            },
        )
        # break when all values of Losses are less than 1e-5
        if all([l < 1e-5 for l in Losses]):
            break
    pbar.close()

    return model, loss.item()  # type: ignore
