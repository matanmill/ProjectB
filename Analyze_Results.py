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
# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_labels', default=200, type=int, help='number of labels')
parser.add_argument('--samples_num', default=100, type=int, help='number of samples to use in case of small dataset')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument("--label_vocabulary_path", type=str, default=paths['vocabulary'],
                    help="path for decoding the labels from provided vocabulary")
parser.add_argument("--test_path", type=str, default='./datafiles/fsd50k_eval_full.json',
                    help="path for test set")
parser.add_argument("--test_model_path", type=str, default=r'./outputs/20230513-091155baseline_lr_00001/model.pth',
                    help="path for model you want to test")
parser.add_argument("--training_path", type=str, default='./datafiles/fsd50k_tr_full.json',
                    help="path for test set")
parser.add_argument("--enhanced_training_path", type=str, default='./datafiles/fsd50k_tr_full_type1_2_5.json',
                    help="path for test set")
parser.add_argument("--run_small_data", type=bool, default=False, help="should you run on a small dataset")


args = parser.parse_args()


# defining - device, test dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metric = MultilabelAveragePrecision(num_labels=args.num_labels, average='macro', thresholds=None)
mlap = MultilabelAveragePrecision(num_labels=args.num_labels, average=None, thresholds=None)
test_dataset = LOADER.AudioDataset(args.test_path, args.num_labels, args.label_vocabulary_path,
                                   small_data_num=args.samples_num, run_small_data=args.run_small_data)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=LOADER.audio_collate_fn)


# defining - model testsed
checkpoint = torch.load(args.test_model_path, map_location=device)  # add to parser - testing folder wanted
model = BaseTransformer()
model.load_state_dict(checkpoint['model_state_dict'])
check = checkpoint['hyper_parameters'] # this is for validation
print(check)
model.to(device)

# computing AP list
mlap_list = test(model=model, dataloader_test=test_dataloader, device=device, metric=mlap)

mlap_list_whole_clip = avg_test(model=model, dataloader_test=test_dataloader, device=device, metric=mlap)

# test with whole clip samples
mlap_list = test
print("mAP score list is: " + str(mlap_list))

# computing mAP accuracy
# Final_mAP = test(model=model, dataloader_test=test_dataloader, device=device, metric=metric)
Final_mAP = torch.sum(mlap_list) / args.num_labels
print("mAP score is: " + str(Final_mAP))

# more in depth analysis of results
###### Step 1 - confusion matrix and such
ConfusionMatrix = MultilabelConfusionMatrix(num_labels=args.num_labels)
calc_confusion_matrix = utils.Confusion_Matrix(model=model, dataset=test_dataloader, device=device,
                                               confusion_matrix=ConfusionMatrix, metric_list=mlap_list, k=12)

utils.histogram_mlap(mlap_list)


# plot the normal training to see
# utils.plot_histogram(args.label_vocabulary_path, args.training_path)

# plot the enhanced training to see
# utils.plot_histogram(args.label_vocabulary_path, args.enhanced_training_path)




