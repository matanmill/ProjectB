import torch
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import numpy as np
from Paths import FSD50K_paths as paths

plt.style.use('ggplot')
time = time.time()


# TODO: add parser arguments inside the saving function
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
            self, best_valid_loss=float('inf'), best_map=0, method="map"
    ):
        self.best_valid_loss = best_valid_loss
        self.best_map = best_map
        self.method = method

    def __call__(
            self, current_val_loss, current_map_score,
            epoch, model, optimizer, criterion, path, args
    ):
        """
        :param current_val_loss: validation loss for current epoch
        :param current_val_loss: mAP for current epoch
        :param epoch: epoch number
        :param model: trined model
        :param optimizer: optimizer
        :param criterion: loss function
        :param path: saving path
        :return: NA
        """
        # create path for saving current training instance
        # TODO: add capability to save all arg parser inputs
        if not os.path.exists(path):
            os.makedirs(path)

        # if decided on validation method of saving stuff:
        if self.method == "validation" and current_val_loss < self.best_valid_loss:
            self.best_valid_loss = current_val_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'hyper_parameters': args
            }, os.path.join(path, 'model.pth'))  # add epoch num

        if self.method == "map" and current_map_score > self.best_map:
            self.best_map = current_map_score
            print(f"\nBest mAP score: {self.best_map}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'hyper_parameters': args
            }, os.path.join(path, 'model.pth'))  # add epoch num

        return self.best_map, self.best_valid_loss


def save_model(epochs, model, optimizer, criterion, path):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, 'model.pth')
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path)


def save_plots(valid_acc, train_loss, valid_loss, path):
    """
    Function to save the loss and accuracy plots to disk.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Move tensors to the CPU
    valid_acc = torch.tensor(valid_acc)
    valid_acc = valid_acc.cpu().numpy()
    train_loss = torch.tensor(train_loss)
    train_loss = train_loss.cpu().numpy()
    valid_loss = torch.tensor(valid_loss)
    valid_loss = valid_loss.cpu().numpy()

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(path, 'accuracy'))

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss'))


def vocab_to_list(path):
    vocab = pd.read_csv(path, header=None)
    label_names = vocab.iloc[:, 1]
    label_names = list(map(str, label_names))
    return label_names


def visualize_confusion_matrices(confusion_tensor, mlap_list, indices=None):
    # If indices are provided, select the corresponding confusion matrices
    if indices is not None:
        confusion_tensor = confusion_tensor[indices]
        mlap_list = mlap_list[indices]

    # Create a grid of subplots with one row and N columns
    n_plots = confusion_tensor.shape[0] // 2
    fig, axs = plt.subplots(nrows=2, ncols=n_plots, figsize=(2,2))
    axs = axs.ravel().tolist()

    # Loop over the confusion matrices and plot each one
    for i in range(confusion_tensor.shape[0]):
        # Get the confusion matrix and normalize its values
        confusion_matrix = confusion_tensor[i]
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = replace_nan_with_zero(confusion_matrix)

        # Plot the confusion matrix as an image
        im = axs[i].matshow(confusion_matrix, cmap='Blues')
        for j in range(confusion_matrix.shape[0]):
            for p in range(confusion_matrix.shape[1]):
                axs[i].text(x=p, y=j, s=round(float(confusion_matrix[j, p]), 3), va='center', ha='center', size='small')

        # Add axis labels and a colorbar to the plot
        axs[i].set_xlabel('Predicted label', fontsize=8)
        axs[i].set_ylabel('True label', fontsize=8)
        axs[i].set_title(mlap_list[i], fontsize=8)
        axs[i].grid(False)
        fig.colorbar(im, ax=axs[i])

    # Adjust the spacing between subplots and show the plot
    plt.subplots_adjust(wspace=1, hspace=1)  # add this line
    plt.show()


def replace_nan_with_zero(tensor):
    # Replace NaN values with 0
    tensor[torch.isnan(tensor)] = 0

    # Return the updated tensor
    return tensor


def Confusion_Matrix(model, dataset, device, confusion_matrix, Visualize=True, visualize_indices=None, vocabulary_path=paths['vocabulary'], top_k=True, k=20, metric_list=None):
    """
    :param model: model you want to create a confusion matrix for
    :param dataset: test dataset
    :param device: device to preform calculations on
    :param confusion_matrix: confusion matrix calculation instance
    :param Visualize: bool, True if you want to preform visualization of confusion matrix, False to return it
    :param visualize_indices: indices
    :param vocabulary_path: path for vocabulary of dataset
    :param top_k: bool, create a confusion matrix only for top-k classes (metric-wise)
    :param k: k for top k
    :param metric_list: list of labels with AP score (could be anything)
    :return: if visualize is false, return the confusion matrix itself
    """
    model.eval()
    mlap_list = vocab_to_list(vocabulary_path)
    with torch.no_grad():
        for index, (data, labels) in enumerate(dataset):
            # send data and labels to device, compute mAP for this batch
            data = data.to(device)
            labels = labels.to(device)
            labels = labels.to(torch.long)
            predictions = model(data)
            predictions = torch.squeeze(predictions, dim=1)
            confusion_matrix.update(predictions, labels)

        ConfusionMatrix = confusion_matrix.compute()
        confusion_matrix.reset()

        if top_k:
            metric_list = replace_nan_with_zero(metric_list)
            scores, indices = torch.topk(metric_list, k=k, sorted=False)
            ConfusionMatrix = ConfusionMatrix[indices.tolist()]
            print(ConfusionMatrix)
            mlap_list = [mlap_list[i] for i in indices]
            visualize_indices = None

        if Visualize:
            visualize_confusion_matrices(confusion_tensor=ConfusionMatrix, mlap_list=mlap_list, indices=visualize_indices)
        else:
            print("No visualization of confusion matrix, returning the matrix itself")
            return ConfusionMatrix
