from Paths import FSD50K_paths
from BaseArchitecture import BaseTransformer
import torch
from torch import nn

###########################################################################
# part 1 - Load data, seperate, batchify + define training device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###########################################################################
# part 2 - define the model and hyperparameters
# all hyperparameters are already implemented inside the class

labels_size = 200  # 200 final categories
model = BaseTransformer().to(device)

###########################################################################
# part 3 - define the training process and train

criterion = nn.HuberLoss()

###########################################################################
# part 4 - evaluate mAP of the model

###########################################################################
# part 5 - save everything
