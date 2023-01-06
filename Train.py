import torch
import torch.nn as nn


def train(model: nn.Module, dataloader_train, optimizer, criterion, device):
    """
    This function assumes that Dataloader holds all batches inside
    should probably change
    """
    model.train()
    total_loss = 0

    for index, (data, labels) in enumerate(dataloader_train):
        # send data and labels to device
        data = data.to(device)
        labels = labels.to(device)

        # compute loss by criteria
        output = model(data)
        loss = criterion(output, labels)

        # flushing, computing gradients, applying
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def evaluate(model: nn.Module, dataloader_eval, criterion, device):
    model.eval()
    with torch.no_grad():  # don't keep track of gradients (faster)
        total_loss = 0
        for index, (data, labels) in enumerate(dataloader_eval):
            # send data and labels to device
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)

            # compute validation loss for model selection
            loss = criterion(output, labels)
            total_loss += loss.item()

        return total_loss

