from BaseArchitecture import BaseTransformer
import torch
from torch import nn
import LOADER
import time
import torch.optim as opt
from Train import train, evaluate, test
from utils import SaveBestModel, save_plots, save_model


###########################################################################
# part 1 - Load data, seperate, batchify + define training device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# trainloader = LOADER.train_dataloader
# validloader = LOADER.val_dataloader
# testloader = LOADER.eval_dataloader

label_vocabulary_path = r'C:\FSD50K\FSD50K.ground_truth\vocabulary.csv'
train_path = './datafiles/fsd50k_tr_full.json'
eval_path = './datafiles/fsd50k_eval_full.json'
val_path = './datafiles/fsd50k_val_full.json'

train_dataset = LOADER.AudioDataset(train_path, label_vocabulary_path,run_small_data=True)
val_dataset = LOADER.AudioDataset(eval_path, label_vocabulary_path,run_small_data=True)
eval_dataset = LOADER.AudioDataset(val_path, label_vocabulary_path,run_small_data=True)

# Create the dataloader  #######!!add num_workers if we have GPU!!!!!!!##########
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=LOADER.audio_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=LOADER.audio_collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=LOADER.audio_collate_fn)


###########################################################################
# part 2 - define the model and hyperparameters
# all hyperparameters are already implemented inside the class

saving_path = "model.pt"
labels_size = 200  # 200 final categories
base_model = BaseTransformer()
save_best_model = SaveBestModel()

# split the training between all GPU's available
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    base_model = nn.DataParallel(base_model)

base_model.to(device)

###########################################################################
# part 3 - define the training process and train
# will transport to a different module, this is here for comfortability for now

criterion = nn.HuberLoss()
lr = 5.0
optimizer = opt.Adam(base_model.parameters(), lr=lr)
schedualer = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
# can try to implement schedualer from attention is all you need

# parameters for finding the best model and gathering statistics
best_val_loss = float('inf')
best_model = None
epoch_num = 50
total_batches = len(trainloader)
start_time = time.time()

train_loss, val_loss = [], []
val_acc = []

# training process over epochs
for epoch in range(epoch_num):
    print(f"[INFO]: Epoch {epoch + 1} of {epoch_num}")
    # train phase
    train_loss_epoch = train(model=base_model, dataloader_train=trainloader, criterion=criterion,
                             optimizer=optimizer, device=device)

    # evaluation phase
    val_loss_epoch, val_epoch_acc = evaluate(base_model, validloader, criterion=criterion, device=device)

    # keep track of progress and save the best model
    print(f"Training loss: {train_loss_epoch:.3f}")
    print(f"Validation loss: {val_loss_epoch:.3f}, validation acc: {val_epoch_acc:.3f}")

    train_loss.append(train_loss_epoch)
    val_loss.append(val_loss_epoch)
    val_acc.append(val_epoch_acc)

    save_best_model(val_loss_epoch, epoch, base_model, optimizer, criterion)

# save the trained model weights for a final time
save_model(epoch_num, base_model, optimizer, criterion)

# save the loss and accuracy plots
save_plots(val_acc, train_loss, val_loss)

print('TRAINING COMPLETE')
###########################################################################
# part 4 - test

Final_Accuracy = test(model=base_model, dataloader_test=testloader, criterion=criterion, device=device)
print("Final accuracy for the model, based on mAP metric is " + str(Final_Accuracy))
