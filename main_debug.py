from BaseArchitecture import BaseTransformer
import torch
from torch import nn
from torch.utils.data import DataLoader
import LOADER
import time
import torch.optim as opt
from Train import train, evaluate, test
from utils import SaveBestModel, save_plots, save_model
from torchmetrics.classification import MultilabelAveragePrecision
from torch.utils.data import WeightedRandomSampler
import argparse

#########################################################################
###           Don't run here - use normal main function               ###
#########################################################################
# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_labels', default=200, type=int, help='number of labels')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--learning_rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout')
parser.add_argument("--epochs", type=int, default=50, help="number of maximum training epochs")
parser.add_argument("--saving_path", type=str, default=r'C:\Users\matan\OneDrive\Desktop\technion\semester 8\Project B\outputs',
                    help="path for saving results")
parser.add_argument("--label_vocabulary_path", type=str, default=r'C:\FSD50K\FSD50K.ground_truth\vocabulary.csv',
                    help="path for decoding the labels from provided vocabulary")
parser.add_argument("--train_path", type=str, default='./datafiles/fsd50k_tr_full.json',
                    help="path for training set")
parser.add_argument("--eval_path", type=str, default='./datafiles/fsd50k_eval_full.json',
                    help="path for test set")
parser.add_argument("--val_path", type=str, default='./datafiles/fsd50k_val_full.json',
                    help="path for validation set")
parser.add_argument('--balanced_set', default=False, help='if use balance sampling', type=str)
args = parser.parse_args()

# Save the arguments
torch.save(args, 'model_args.pt') # where does it save stuff?

# for summary of your requested parameters, uncomment this or see model_args.pt
# args_user = torch.load('model_args.pt')
# print("our args",args_user)


###########################################################################
# part 1 - Load data, seperate, batchify + define training device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# trainloader = LOADER.train_dataloader
# validloader = LOADER.val_dataloader
# testloader = LOADER.eval_dataloader

label_vocabulary_path = args.label_vocabulary_path
train_path = args.train_path
eval_path = args.eval_path
val_path = args.val_path

# Set the seed - add it to the args
torch.manual_seed(args.seed)

train_dataset = LOADER.AudioDataset(train_path, args.num_labels, label_vocabulary_path, run_small_data=True)
val_dataset = LOADER.AudioDataset(eval_path, args.num_labels, label_vocabulary_path, run_small_data=True)
eval_dataset = LOADER.AudioDataset(val_path, args.num_labels, label_vocabulary_path, run_small_data=True)

if args.balanced_set:
    print('balanced sampler is being used')
    samples_weight = np.loadtxt(args.data_train[:-5] + '_weight.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=LOADER.audio_collate_fn, sampler=sampler)
else:
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=LOADER.audio_collate_fn)

# Create the dataloader  #######!!add num_workers if we have GPU!!!!!!!##########
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=LOADER.audio_collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=LOADER.audio_collate_fn)


###########################################################################
# part 2 - define the model and hyperparameters
# all hyperparameters are already implemented inside the class

num_labels = args.num_labels  # change to parameter recieved, also you added it twice
saving_path = os.path.join(args.saving_path, time.strftime("%Y%m%d-%H%M%S")) # add to parameters
labels_size = 200  # 200 final categories
base_model = BaseTransformer(dropout=args.dropout)
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
optimizer = opt.Adam(base_model.parameters(), lr=args.learning_rate)
schedualer = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
metric = MultilabelAveragePrecision(num_labels=num_labels, average='macro', thresholds=None)
# can try to implement schedualer from attention is all you need

# parameters for finding the best model and gathering statistics
best_val_loss = float('inf')
best_model = None
epoch_num = args.epochs
total_batches = len(train_dataloader)
start_time = time.time()

train_loss, val_loss = [], []
val_acc = []

# training process over epochs
for epoch in range(epoch_num):
    print(f"[INFO]: Epoch {epoch + 1} of {epoch_num}")
    # train phase
    train_loss_epoch = train(model=base_model, dataloader_train=train_dataloader, criterion=criterion,
                             optimizer=optimizer, device=device)

    # evaluation phase
    val_loss_epoch, val_epoch_acc = evaluate(base_model, val_dataloader, criterion=criterion,
                                             device=device, metric=metric)

    # keep track of progress and save the best model
    print(f"Training loss: {train_loss_epoch:.3f}")
    print(f"Validation loss: {val_loss_epoch:.3f}, validation acc: {val_epoch_acc:.3f}")

    train_loss.append(train_loss_epoch)
    val_loss.append(val_loss_epoch)
    val_acc.append(val_epoch_acc)

    save_best_model(val_loss_epoch, epoch, base_model, optimizer, criterion, path=saving_path)
    metric.reset()

# save the trained model weights for a final time - don't need to save at the end
# save_model(epoch_num, base_model, optimizer, criterion, path=saving_path)

# save the loss and accuracy plots
save_plots(val_acc, train_loss, val_loss, path=saving_path)

print('TRAINING COMPLETE')
###########################################################################
# part 4 - test

Final_Accuracy = test(model=base_model, dataloader_test=eval_dataloader, criterion=criterion,
                      device=device, metric=metric)
print("Final accuracy for the model, based on mAP metric is " + str(Final_Accuracy))
