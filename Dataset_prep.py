####### based on code from PSLA article #######

import numpy as np
import json
import os
from Paths import FSD50K_paths
import librosa
import soundfile as sf

# dataset downloaded from https://zenodo.org/record/4060432#.YXXR0tnMLfs
# please change it to your FSD50K dataset path
# the data organization might change with versioning, the code is tested early 2021

fsd_path = FSD50K_paths['code_exploring_eval']
un_resampled_path = r'C:\FSD50K\FSD50K.dev_audio'

# create json datafiles for training, validation, and evaluation set
fsd_dev_csv = FSD50K_paths['ground_truth_dev']
fsdeval = np.loadtxt(fsd_dev_csv, skiprows=1, dtype=str)
tr_cnt, val_cnt = 0, 0


# only apply to the vocal sound data
fsd_tr_data = []
fsd_val_data = []
for i in range(len(fsdeval)):
    try:
        fileid = fsdeval[i].split(',"')[0]
        labels = fsdeval[i].split(',"')[2][0:-1]
        set_info = labels.split('",')[1]
    except:
        fileid = fsdeval[i].split(',')[0]
        labels = fsdeval[i].split(',')[2]
        set_info = fsdeval[i].split(',')[3][0:-1]

    labels = labels.split('",')[0]
    label_list = labels.split(',')
    new_label_list = []
    for label in label_list:
        new_label_list.append(label)
    new_label_list = ','.join(new_label_list)
    # note, all recording we use are 16kHZ.
    cur_dict = {"wav": un_resampled_path + '\\'+ fileid + '.wav', "labels": new_label_list}

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

with open('./datafiles/fsd50k_tr_full.json', 'w') as f:
    json.dump({'data': fsd_tr_data}, f, indent=1)
print('Processed {:d} samples for the FSD50K training set.'.format(tr_cnt))

with open('./datafiles/fsd50k_val_full.json', 'w') as f:
    json.dump({'data': fsd_val_data}, f, indent=1)
print('Processed {:d} samples for the FSD50K validation set.'.format(val_cnt))

## process the evaluation set
fsd_eval_csv = FSD50K_paths['ground_truth_eval']
fsdeval = np.loadtxt(fsd_eval_csv, skiprows=1, dtype=str)

cnt = 0

# only apply to the vocal sound data
vc_data = []
for i in range(len(fsdeval)):
    try:
        fileid = fsdeval[i].split(',"')[0]
        labels = fsdeval[i].split(',"')[2][0:-1]
    except:
        fileid = fsdeval[i].split(',')[0]
        labels = fsdeval[i].split(',')[2]

    label_list = labels.split(',')
    new_label_list = []
    for label in label_list:
        new_label_list.append(label)

    if len(new_label_list) != 0:
        new_label_list = ','.join(new_label_list)
        cur_dict = {"wav": fsd_path + '\\'+ fileid + '.wav', "labels": new_label_list}
        vc_data.append(cur_dict)
        cnt += 1

with open('./datafiles/fsd50k_eval_full.json', 'w') as f:
    json.dump({'data': vc_data}, f, indent=1)
print('Processed {:d} samples for the FSD50K evaluation set.'.format(cnt))


# generate balanced sampling weight file
os.system('python ../../src/gen_weight_file.py --dataset fsd50k --label_indices_path {:s} --datafile_path {:s}'.format(
    './class_labels_indices.csv', './datafiles/fsd50k_tr_full.json'))

# (optional) create label enhanced set.
# Go to /src/label_enhancement/
