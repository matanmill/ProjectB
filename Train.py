import torch
import torch.nn as nn
from torchmetrics.classification import multilabel_average_precision


def train(model: nn.Module, dataloader_train, optimizer, criterion, device):
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
    num_bathces = len(dataloader_eval)
    with torch.no_grad():  # don't keep track of gradients (faster)
        total_loss = 0
        accuracy = 0
        for index, (data, labels) in enumerate(dataloader_eval):
            # send data and labels to device
            data = data.to(device)
            labels = labels.to(device)
            labels_size = len(labels)
            output = model(data)

            # compute validation loss for model selection
            loss = criterion(output, labels)
            total_loss += loss.item()
            accuracy += multilabel_average_precision(output, labels, num_labels=labels_size, average='macro')

        accuracy = accuracy/num_bathces

        return total_loss, accuracy


def test(model, dataloader_test, criterion, device):
    score = 0
    batch_num = len(dataloader_test)
    for index, (data, labels) in enumerate(dataloader_test):
        # send data and labels to device, compute mAP for this batch
        data = data.to(device)
        labels = labels.to(device)
        labels_size = len(labels)
        predictions = model(data)
        score += multilabel_average_precision(predictions, labels, num_labels=labels_size, average='macro')

    # total score/ number of batches
    score = score/batch_num
    return score



