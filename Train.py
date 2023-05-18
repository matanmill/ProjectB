import torch
import torch.nn as nn


def train(model: nn.Module, dataloader_train, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for index, (data, labels, audio_wav) in enumerate(dataloader_train):
        # send data and labels to device
        # data = torch.reshape(data, (-1, 40, 400))
        data = data.to(device)
        # labels = torch.cat(labels, dim=1)
        # labels = torch.transpose(labels, 0, 1)
        labels = labels.to(device)

        # compute loss by criteria
        optimizer.zero_grad()
        output = model(data)
        output = torch.squeeze(output, dim=1)
        loss = criterion(output, labels)

        # flushing, computing gradients, applying
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def evaluate(model: nn.Module, dataloader_eval, criterion, device, metric):
    model.eval()
    total_loss = 0
    with torch.no_grad():  # don't keep track of gradients (faster)
        for index, (data, labels, audio_wav) in enumerate(dataloader_eval):

            # send data and labels to device
            data = data.to(device)
            labels = labels.to(device)
            labels = labels.to(torch.long)
            output = model(data)
            output = torch.squeeze(output, dim=1)

            # compute validation loss for model selection
            loss = criterion(output, labels)
            total_loss += loss.item()
            metric(output, labels)

        score = metric.compute()
        metric.reset()

        return total_loss, score


def test(model, dataloader_test, device, metric):
    model.eval()
    with torch.no_grad():
        for index, (data, labels, audio_wav) in enumerate(dataloader_test):
            # send data and labels to device, compute mAP for this batch
            data = data.to(device)
            labels = labels.to(device)
            labels = labels.to(torch.long)
            predictions = model(data)
            predictions = torch.squeeze(predictions, dim=1)

            metric(predictions, labels)
            # predictions = predictions.detach()

        score = metric.compute()
        metric.reset()
        return score

# making mAP calculation based on clips and not on 1 sec's sample
def avg_test(model, dataloader_test, device, metric):
    model.eval()
    with torch.no_grad():
        clip_pred_dict = {} #dictionary of clips and their predicted labels. key:clip , value:list of one-secs labels
        clip_GT_dict = {} # dictionary of clips and their GT labels.

        # first, build the dictionary
        for index, (data, labels, audio_wav) in enumerate(dataloader_test):
            data = data.to(device)
            labels = labels.to(device)
            labels = labels.to(torch.long)
            predictions = model(data)
            predictions = torch.squeeze(predictions, dim=1)

            # we need to iterate over the samples in the current batch
            for i in range(data.size(0)):
                single_prediction = predictions[i]  # Prediction for a single sample
                single_label = labels[i]  # Label for a single sample
                single_audio_wav = audio_wav[i]
                clip_pred_dict[single_audio_wav].append(single_prediction)
                clip_GT_dict[single_audio_wav].append(single_label)

        # now, each key's value is a new sample.
        # question is what will be the label of this new sample?
        # let's make an averaged softmax
        for audio_wav, preds in clip_pred_dict.items():
            avg_softmax = torch.mean(preds, dim=1, keepdim=True)
            metric(avg_softmax, (clip_GT_dict[audio_wav])[:,0])
            # predictions = predictions.detach()

        score = metric.compute()
        metric.reset()
        return score


"""
def test(model, dataloader_test, criterion, device, metric):
    score = 0
    batch_num = len(dataloader_test)
    for index, (data, labels) in enumerate(dataloader_test):
        # send data and labels to device, compute mAP for this batch
        data = data.to(device)
        labels = labels.to(device)
        labels = labels.to(torch.long)
        predictions = model(data)
        predictions = torch.squeeze(predictions, dim=1)
        metric(predictions, labels)

    score = metric.compute()
    metric.reset()
    return score
"""


