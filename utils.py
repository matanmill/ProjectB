import torch
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import numpy as np
from Paths import FSD50K_paths as paths
import json
from collections import defaultdict

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
        # change name validation to loss
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


def visualize_confusion_matrices(confusion_tensor, mlap_list, indices=None, saving_path=None, name="Confusion_matrix_top_12", show_plot=False):
    """
    :param confusion_tensor: tensor containing all confusion matrices to be plotted
    :param mlap_list: list of mAP for confusion matrices
    :param indices: indices for compatability
    :param saving_path: path to save the plot
    :param name: name of figure to save
    :param show_plot:
    :return:
    """

    if saving_path is None:
        print("No path was given, not saving anything!")
        exit(1)
    else:
        saving_path = os.path.join(saving_path, name)

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
    plt.savefig(saving_path, dpi=800)
    if show_plot:
        plt.show()


def replace_nan_with_zero(tensor):
    # Replace NaN values with 0
    tensor[torch.isnan(tensor)] = 0

    # Return the updated tensor
    return tensor


def Confusion_Matrix(model, dataset, device, confusion_matrix, saving_path, Visualize=True, visualize_indices=None,
                     vocabulary_path=paths['vocabulary'], top_k=True, k=20, metric_list=None):
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
    :param saving_path: path to save the confusion matrix generated
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
            visualize_confusion_matrices(confusion_tensor=ConfusionMatrix, mlap_list=mlap_list, indices=visualize_indices, saving_path=saving_path)
        else:
            print("No visualization of confusion matrix, returning the matrix itself")
            return ConfusionMatrix


def create_label_dictionary(vocab_path):
    vocab = pd.read_csv(vocab_path, header=None)
    l_dict = {}
    for index in vocab[0]:
        l_dict.update({vocab[2][index]: vocab[1][index]})

    return l_dict


def plot_histogram(vocabulary_path, json_path, name, shareY=True, save=False, show=True, path=r'C:\Users\matan\Desktop\ProjectB\outputs'):
    # Load the vocabulary and label dictionary
    label_dict = create_label_dictionary(vocabulary_path)

    # Load the training data
    with open(json_path, 'r') as f:
        training_json = json.load(f)

    # Count the labels
    label_counts = defaultdict(int)
    for row in training_json['data']:
        labels = row['labels'].split(',')
        for label in labels:
            label_counts[label_dict[label]] += 1

    # Create the subplots
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 10), sharey=shareY)
    keys, values = [], []
    for key, value in label_counts.items():
        keys.append(key)
        values.append(value)

    print(sum(label_counts.values()))

    # Plot the data in each subplot
    for i in range(8):
        start = i * 25
        end = (i+1) * 25
        axs[i // 4, i % 4].bar(list(label_counts.keys())[start:end], list(label_counts.values())[start:end])
        # axs[i//4, i%4].set_title('25 labels: {}'.format(i+1))

    # Set the x-axis labels for each subplot
    for ax in axs.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Set the overall title for the figure
    fig.suptitle(name)

    if save:
        path = os.path.join(path, "Analyze Results")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, name.replace(" ", "_")), dpi=800)

    # Show the plot
    if show:
        plt.show()


def histogram_mlap(mlap_list, show=False, saving_path=None, name="AveragePrecision_hist"):
    """
    :param mlap_list: list of AP scores for calculation of histogram
    :param show: should you show the plot or only save
    :param saving_path: path for saving
    :param name: name of the figure
    :return: nothing
    """
    if saving_path is None:
        print("No path was given, not saving anything!")
        exit(1)
    else:
        saving_path = os.path.join(saving_path, name)
    # Convert the PyTorch tensor to a NumPy array
    mlap_array = mlap_list.numpy()
    mean_mlap = np.mean(mlap_array)

    # Plot the histogram using Matplotlib
    fig, ax = plt.subplots()
    ax.hist(mlap_array, bins=20)
    ax.set_title(f"Mean Average Precision Score: {mean_mlap:.2f}")
    ax.set_xlabel("Average Precision")
    ax.set_ylabel("Frequency")

    if show:
        plt.show()
