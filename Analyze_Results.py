import torch
from torch.utils.data import DataLoader
import LOADER
from torchmetrics.classification import MultilabelAveragePrecision
import argparse
from Paths import FSD50K_paths as paths
from Train import test, avg_test
from BaseArchitecture import BaseTransformer
from torchmetrics.classification import MultilabelConfusionMatrix
import utils
import pandas as pd

##################################################################
## Python script to analyze inidividual model rsults more in depth
##################################################################

#########################################################################
# parsing arguments
parser = argparse.ArgumentParser()

parser.add_argument("--test_model_path", type=str,
                    default=r'./outputs/20230520-095937lr_0001_epochs100_no_music/model.pth',
                    help="path for model you want to test")
parser.add_argument('--num_labels', default=198, type=int, help='number of labels')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout probability for the transformer architecture')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--learning_rate', default=0.0005, type=float, metavar='LR', help='initial learning rate')
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
parser.add_argument('--small_data', default=False, type=bool,
                    help='for debugging, maybe you want to use a small dataset')
parser.add_argument('--samples_num', default=100, type=int, help='number of samples in small dataset')
parser.add_argument('--no_music', default=0, type=int, help='1 : to run test without music label')


if __name__ == '__main__':
    args = parser.parse_args()
    print("wwwwiii",args)

print("THE TEST MODEL PATH: ",args.test_model_path)
# plot the normal training - no music to see
# utils.plot_histogram(args.label_vocabulary_path, args.training_path,
#                     name="Training Dataset without 'Music' & 'Musical Instruments'")
# utils.plot_histogram(args.label_vocabulary_path, args.val_path, name="Validation Dataset without 'Music' & 'Musical Instruments'")
# utils.plot_histogram(args.label_vocabulary_path, args.test_path, name="Testing Dataset without 'Music' & 'Musical Instruments'")


# defining - device, test dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metric = MultilabelAveragePrecision(num_labels=args.num_labels, average='macro', thresholds=None)
mlap = MultilabelAveragePrecision(num_labels=args.num_labels, average=None, thresholds=None)
if args.no_music:
    print("wii", args.no_music)
    test_dataset = LOADER.AudioDataset(args.test_no_music_path, args.num_labels, args.label_vocabulary_no_music_path,
                                       small_data_num=args.samples_num, run_small_data=args.small_data)
else:
    test_dataset = LOADER.AudioDataset(args.test_path, args.num_labels, args.label_vocabulary_path,
                                       small_data_num=args.samples_num, run_small_data=args.small_data)

test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=LOADER.audio_collate_fn)

# defining - model testsed
checkpoint = torch.load(args.test_model_path, map_location=device)  # add to parser - testing folder wanted
model = BaseTransformer(dropout=args.dropout, label_number=args.num_labels)
model.load_state_dict(checkpoint['model_state_dict'])
check = checkpoint['hyper_parameters']  # this is for validation
print(check)
model.to(device)

# computing AP list
#mlap_list = test(model=model, dataloader_test=test_dataloader, device=device, metric=mlap)

# whole clip computation
mlap_list_whole_clip = avg_test(model=model, dataloader_test=test_dataloader, device=device, metric=mlap)

# print("mAP score list is: " + str(mlap_list))
print("mAP with clip samples:" + str(mlap_list_whole_clip))

# computing mAP accuracy
#Final_mAP = test(model=model, dataloader_test=test_dataloader, device=device, metric=metric)
Final_mAP = torch.nanmean(mlap_list_whole_clip)
print("mAP score is: " + str(Final_mAP))

# more in depth analysis of results
###### Step 1 - confusion matrix and such
ConfusionMatrix = MultilabelConfusionMatrix(num_labels=args.num_labels)
calc_confusion_matrix = utils.Confusion_Matrix(model=model, dataset=test_dataloader, device=device,
                                              confusion_matrix=ConfusionMatrix, metric_list=mlap_list_whole_clip, k=12, ConfMat=args.ConfMat)

utils.histogram_mlap(mlap_list_whole_clip)


# plot the normal training to see
# utils.plot_histogram(args.label_vocabulary_path, args.training_path)

# plot the enhanced training to see
# utils.plot_histogram(args.label_vocabulary_path, args.enhanced_training_path)

