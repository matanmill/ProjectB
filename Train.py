import torch
import torch.nn as nn


def train(model: nn.Module, dataloader_train, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for index, (data, labels) in enumerate(dataloader_train):
        # send data and labels to device
        data = torch.reshape(data, (-1, 40, 400))
        data = data.to(device)
        labels = torch.cat(labels, dim=1)
        labels = torch.transpose(labels, 0, 1)
        print(labels.size())
        labels = labels.to(device)

        # compute loss by criteria
        output = model(data)
        output = torch.squeeze(output, dim=1)
        print("label size is" + str(labels.size()))
        print("output size is " + str(output.size()))
        loss = criterion(output, labels)

        # flushing, computing gradients, applying
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def evaluate(model: nn.Module, dataloader_eval, criterion, device, metric):
    model.eval()
    num_bathces = len(dataloader_eval)
    with torch.no_grad():  # don't keep track of gradients (faster)
        total_loss = 0
        accuracy = 0
        for index, (data, labels) in enumerate(dataloader_eval):
            # send data and labels to device
            data = torch.reshape(data, (-1, 40, 400))
            data = data.to(device)
            labels = torch.cat(labels, dim=1)
            labels = torch.transpose(labels, 0, 1)
            labels = labels.to(device)
            labels_size = len(labels)
            output = model(data)
            output = torch.squeeze(output, dim=1)

            # compute validation loss for model selection
            loss = criterion(output, labels)
            total_loss += loss.item()
            accuracy += metric(output, labels)

        accuracy = accuracy/num_bathces

        return total_loss, accuracy


def test(model, dataloader_test, criterion, device, metric):
    score = 0
    batch_num = len(dataloader_test)
    for index, (data, labels) in enumerate(dataloader_test):
        # send data and labels to device, compute mAP for this batch
        data = torch.reshape(data, (-1, 40, 400))
        data = data.to(device)
        labels = torch.cat(labels, dim=1)
        labels = torch.transpose(labels, 0, 1)
        labels = labels.to(device)
        predictions = model(data)
        predictions = torch.squeeze(predictions, dim=1)
        score += metric(predictions, labels)

    # total score/ number of batches
    score = score/batch_num
    return score



