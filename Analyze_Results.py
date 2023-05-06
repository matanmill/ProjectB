import torch
from torch.utils.data import DataLoader
import LOADER
from torchmetrics.classification import MultilabelAveragePrecision
import argparse
from Paths import FSD50K_paths as paths
from Train import test
from BaseArchitecture import BaseTransformer
from torchmetrics.classification import MultilabelConfusionMatrix
import utils

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
parser.add_argument("--test_model_path", type=str, default=r'C:\Users\matan\OneDrive\Desktop\technion\semester 9\Project B\outputs\20230503-180540\model.pth',
                    help="path for model you want to test")

args = parser.parse_args()

# defining - device, test dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metric = MultilabelAveragePrecision(num_labels=args.num_labels, average='macro', thresholds=None)
mlap = MultilabelAveragePrecision(num_labels=args.num_labels, average=None, thresholds=None)
test_dataset = LOADER.AudioDataset(args.test_path, args.num_labels, args.label_vocabulary_path, small_data_num=args.samples_num, run_small_data=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=LOADER.audio_collate_fn)


# defining - model testsed
checkpoint = torch.load(args.test_model_path,map_location=device)  # add to parser - testing folder wanted
model = BaseTransformer()
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# computing AP list
mlap_list = test(model=model, dataloader_test=test_dataloader, device=device, metric=mlap)
print("mAP score list is: " + str(mlap_list))

# computing mAP accuracy
Final_mAP = test(model=model, dataloader_test=test_dataloader, device=device, metric=metric)
print("mAP score is: " + str(Final_mAP))

# more in depth analysis of results
###### Step 1 - confusion matrix and such
ConfusionMatrix = MultilabelConfusionMatrix(num_labels=args.num_labels)
calc_confusion_matrix = utils.Confusion_Matrix(model=model, dataset=test_dataloader,device=device,confusion_matrix=ConfusionMatrix, metric_list=mlap_list, k=8)
