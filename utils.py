import torch
import matplotlib.pyplot as plt
import os
import time

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

