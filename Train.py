import torch
import torch.nn as nn


def train(model: nn.Module, dataloader_train, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for index, (data, labels, audio_wav) in enumerate(dataloader_train):
        data = data.to(device)
        labels = labels.to(device)

        # compute loss by criteria
        optimizer.zero_grad()
        output = model(data)
        output = torch.squeeze(output, dim=1)
        loss = criterion(output, labels)

        # flushing, computing gradients, applying
        loss.backward()
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
    ##### this function is extremely inefficient - needs further work
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
                if single_audio_wav in clip_pred_dict:
                    clip_pred_dict[single_audio_wav].append(single_prediction)
                    clip_GT_dict[single_audio_wav].append(single_label)
                # initialize
                else:
                    clip_pred_dict[single_audio_wav] = []
                    clip_GT_dict[single_audio_wav] = []
                    clip_pred_dict[single_audio_wav].append(single_prediction)
                    clip_GT_dict[single_audio_wav].append(single_label)

            # caculating stuff for the entire test
        for audio_wav, preds in clip_pred_dict.items():
            preds = torch.mean(torch.stack(preds), dim=0, keepdim=True)
            labels = torch.unsqueeze(clip_GT_dict[audio_wav][0], dim=0)
            metric(preds, labels)

        mean_score = metric.compute()
        metric.reset()
        return mean_score



