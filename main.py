import Train
from BaseArchitecture import BaseTransformer
import torch
from torch import nn
from torch.utils.data import DataLoader
import LOADER
import time
import torch.optim as opt
from Train import train, evaluate
from utils import SaveBestModel, save_plots, histogram_mlap, Confusion_Matrix
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelConfusionMatrix
from torch.utils.data import WeightedRandomSampler
import argparse
import os
import numpy as np

#########################################################################
# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_labels', default=200, type=int, help='number of labels')
parser.add_argument('--num_labels_no_music', default=198, type=int, help='number of labels')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout probability for the transformer architecture')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--learning_rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--epochs", type=int, default=100, help="number of maximum training epochs")
parser.add_argument("--saving_path", type=str, default=r'./outputs',
                    help="path for saving results")
parser.add_argument("--confMat_path", type=str, default=r'./outputs/confMats',
                    help="path for saving results")
parser.add_argument("--label_vocabulary_path", type=str, default=r'./FSD50K/FSD50K.ground_truth/vocabulary.csv',
                    help="path for decoding the labels from provided vocabulary")
parser.add_argument("--train_path", type=str, default='./datafiles/fsd50k_tr_full.json',
                    help="path for training set")
parser.add_argument("--train_path_enhanced", type=str, default='./datafiles/fsd50k_tr_full_type1_2_5.json',
                    help="path for enhanced training set")
parser.add_argument("--test_path", type=str, default='./datafiles/fsd50k_eval_full.json',
                    help="path for test set")
parser.add_argument("--val_path", type=str, default='./datafiles/fsd50k_val_full.json',
                    help="path for validation set")
parser.add_argument("--label_vocabulary_no_music_path", type=str,
                    default=r'./FSD50K/FSD50K.ground_truth/vocabulary_wo_music.csv',
                    help="path for decoding the labels from provided vocabulary")
parser.add_argument("--train_no_music_path", type=str, default='./datafiles/fsd50k_tr_full_no_music.json',
                    help="path for training set")
parser.add_argument("--test_no_music_path", type=str, default='./datafiles/fsd50k_eval_full_no_music.json',
                    help="path for test set")
parser.add_argument("--val_no_music_path", type=str, default='./datafiles/fsd50k_val_full_no_music.json',
                    help="path for validation set")
parser.add_argument("--mAP_epsilon", type=float, default=0.001, help=" epsilon for stopping condition based on mAP")
parser.add_argument("--epoch_plateua", type=int, default=10, help="after <num> epochs without change according to "
                                                                  "stopping criteria we want to stop training")
parser.add_argument('--balanced_set', default=False, help='if use balance sampling', type=str)
parser.add_argument('--enhanced_set', default=False, help='if use enhanced sampling', type=str)
parser.add_argument('--seed', default=42, type=int, help='seed for randomizing the batch')
parser.add_argument('--name', default="No_Music_lr_0_0001_epochs_100", type=str, help='name for saving the model')
parser.add_argument('--small_data', default=False, type=bool, help='for debugging, maybe you want to use a small dataset')
parser.add_argument('--no_music', default=False, type=bool, help='to run training without music label')
args = parser.parse_args()

###########################################################################
# part 1 - Load data, seperate, batchify + define training device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.no_music:
    label_vocabulary_path = args.label_vocabulary_no_music_path
    train_path = args.train_no_music_path
    test_path = args.test_no_music_path
    val_path = args.val_no_music_path
    num_labels = args.num_labels_no_music
else:
    label_vocabulary_path = args.label_vocabulary_path
    num_labels = args.num_labels
    test_path = args.test_path
    val_path = args.val_path

if args.enhanced_set:
    train_path = args.train_path_enhanced
else:
    train_path = args.train_path

torch.manual_seed(args.seed)

train_dataset = LOADER.AudioDataset(train_path, args.num_labels, label_vocabulary_path, run_small_data=args.small_data)
val_dataset = LOADER.AudioDataset(val_path, args.num_labels, label_vocabulary_path, run_small_data=args.small_data)
test_dataset = LOADER.AudioDataset(test_path, num_labels, label_vocabulary_path, run_small_data=args.small_data)

# Create the dataloader  #######!!add num_workers if we have GPU!!!!!!!##########
if args.balanced_set:
    print('balanced sampler is being used')
    samples_weight = np.loadtxt(args.train_path[:-5] + '_weight.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=LOADER.audio_collate_fn, sampler=sampler)
else:
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=LOADER.audio_collate_fn)

val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=LOADER.audio_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=LOADER.audio_collate_fn)

###########################################################################
# part 2 - define the model and hyperparameters
# all hyperparameters are already implemented inside the class

name = time.strftime("%Y%m%d-%H%M%S") + args.name
saving_path = os.path.join(args.saving_path, name)  # add to parameters
base_model = BaseTransformer(dropout=args.dropout, label_number=args.num_labels)
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

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
metric = MultilabelAveragePrecision(num_labels=num_labels, average='macro', thresholds=None)
mlap_metric = MultilabelAveragePrecision(num_labels=args.num_labels, average=None, thresholds=None)
# can try to implement schedualer from attention is all you need

# parameters for finding the best model and gathering statistics
best_val_loss = float('inf')
best_model = None
epoch_num = args.epochs
start_time = time.time()

train_loss = []
val_loss = []
val_acc = []
no_improvement_counter = 0
# add schedualer and schedualer step
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

    best_mAP, best_val_loss = save_best_model(current_val_loss=val_loss_epoch, current_map_score=val_epoch_acc,
                                              epoch=epoch, model=base_model, optimizer=optimizer, criterion=criterion,
                                              path=saving_path, args=args)

    # need to enhance stopping condition
    if epoch > 5:
        if val_acc[-1] - val_acc[-2] < args.mAP_epsilon:
            no_improvement_counter += 1
        else:
            no_improvement_counter = 0
        if no_improvement_counter == args.epoch_plateua:  # change to 10
            print("Stopping training phase, mAP score doesn't improve")
            break

    # step for schedualer
    # scheduler.step()

# save the loss and accuracy plots - consider moving them inside loop
# neptune/weight and biases (wandb) - loggers
save_plots(val_acc, train_loss, val_loss, path=saving_path)

print('TRAINING COMPLETE, Testing model and running analysis now')
###########################################################################

test_model_path = os.path.join(args.svaing_path, 'model.pth')

# defining - model testsed
checkpoint = torch.load(test_model_path, map_location=device)  # add to parser - testing folder wanted
model_test = BaseTransformer(dropout=args.dropout, label_number=num_labels)
model_test.load_state_dict(checkpoint['model_state_dict'])
model_test.to(device)

# whole clip computation
mlap_list_whole_clip = Train.avg_test(model=model_test, dataloader_test=test_dataloader, device=device, metric=mlap_metric)

# computing mAP accuracy
Final_mAP = torch.nanmean(mlap_list_whole_clip)
print("Final mAP score is: " + str(Final_mAP))

###### Step 1 - confusion matrix and such
ConfusionMatrix = MultilabelConfusionMatrix(num_labels=args.num_labels)
calc_confusion_matrix = Confusion_Matrix(model=model_test, dataset=test_dataloader, device=device, vocabulary_path=label_vocabulary_path,
                                         confusion_matrix=ConfusionMatrix, metric_list=mlap_list_whole_clip, k=12, saving_path=saving_path)

histogram_mlap(mlap_list_whole_clip)
print("Finished Training and Analysis for model: " + args.name)
