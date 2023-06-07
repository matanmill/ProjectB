####### based on code from PSLA article #######

import numpy as np
import json
import os
import utils
import argparse

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--small_vocabulary_path", type=str, default=r'./datafiles/small_FSD50K_vocabulary.csv',
                    help="path for decoding the labels from provided vocabulary")
parser.add_argument("--training_data_path", type=str, default='./FSD50K/FSDK50.dev_audio/Resampled_dev',
                    help="path for training set")
parser.add_argument("--test_data_path", type=str, default='./FSD50K/FSDK50.eval_audio/Resampled_eval',
                    help="path for test set")
parser.add_argument("--small_train_path", type=str, default='./datafiles/Small_FSD50K_training.json',
                    help="path for json of training data")
parser.add_argument("--small_validation_path", type=str, default='./datafiles/Small_FSD50K_validation.json',
                    help="path for json of validation data")
parser.add_argument("--small_test_path", type=str, default='./datafiles/Small_FSD50K_test.csv',
                    help="path for json of test data")
parser.add_argument("--training_data_csv", type=str, default='./FSD50K/FSD50K.ground_truth/dev.csv',
                    help="path for training ground truth csv for FSD50K")
parser.add_argument("--test_data_csv", type=str, default='./FSD50K/FSD50K.ground_truth/eval.csv',
                    help="path for test ground truth csv for FSD50K")
args = parser.parse_args()

test_data_path = args.test_data_path
train_data_path = args.training_data_path

# create json datafiles for training, validation, and evaluation set
train_ground_truth_csv = args.training_data_csv
train_ground_truth = np.loadtxt(train_ground_truth_csv, skiprows=1, dtype=str)
tr_cnt, val_cnt = 0, 0

# loading new vocabulary for the smaller dataset
small_vocab_list = utils.vocab_to_list(args.small_vocabulary_path, id_or_name='id')

# only apply to the vocal sound data
fsd_tr_data = []
fsd_val_data = []
for i in range(len(train_ground_truth)):
    try:
        fileid = train_ground_truth[i].split(',"')[0]
        labels = train_ground_truth[i].split(',"')[2][0:-1]
        set_info = labels.split('",')[1]
    except:
        fileid = train_ground_truth[i].split(',')[0]
        labels = train_ground_truth[i].split(',')[2]
        set_info = train_ground_truth[i].split(',')[3][0:-1]

    labels = labels.split('",')[0]
    label_list = labels.split(',')
    new_label_list = []
    for label in label_list:
        if label in small_vocab_list:
            new_label_list.append(label)
    new_label_list = ','.join(new_label_list)
    # note, all recording we use are 16kHZ.
    cur_dict = {"wav": train_data_path + '//' + fileid + '.wav', "labels": new_label_list}

    if set_info == 'trai':
        fsd_tr_data.append(cur_dict)
        tr_cnt += 1
    elif set_info == 'va':
        fsd_val_data.append(cur_dict)
        val_cnt += 1
    else:
        raise ValueError('unrecognized set')

if not os.path.exists('datafiles'):
    os.mkdir('datafiles')

with open(args.small_train_path, 'w') as f:
    json.dump({'data': fsd_tr_data}, f, indent=1)
print('Processed {:d} samples for the FSD50K training set.'.format(tr_cnt))

with open(args.small_validation_path, 'w') as f:
    json.dump({'data': fsd_val_data}, f, indent=1)
print('Processed {:d} samples for the FSD50K validation set.'.format(val_cnt))

## process the evaluation set
test_ground_truth_csv = args.test_data_csv
test_ground_truth = np.loadtxt(test_ground_truth_csv, skiprows=1, dtype=str)

cnt = 0

# only apply to the vocal sound data
vc_data = []
for i in range(len(test_ground_truth)):
    try:
        fileid = test_ground_truth[i].split(',"')[0]
        labels = test_ground_truth[i].split(',"')[2][0:-1]
    except:
        fileid = test_ground_truth[i].split(',')[0]
        labels = test_ground_truth[i].split(',')[2]

    label_list = labels.split(',')
    new_label_list = []
    for label in label_list:
        if label in small_vocab_list:
            new_label_list.append(label)

    if len(new_label_list) != 0:
        new_label_list = ','.join(new_label_list)
        cur_dict = {"wav": test_data_path + '//'+ fileid + '.wav', "labels": new_label_list}
        vc_data.append(cur_dict)
        cnt += 1

with open(args.small_test_path, 'w') as f:
    json.dump({'data': vc_data}, f, indent=1)
print('Processed {:d} samples for the FSD50K evaluation set.'.format(cnt))


