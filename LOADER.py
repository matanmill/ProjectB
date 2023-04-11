import librosa
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import json
from IPython.display import Audio
from tqdm import tqdm
import resampy
import pandas as pd

# Open the JSON file and read the contents
####### this is the input from the user... all other things : make_paths.. in the init
with open('./Hirerachy.json', 'r') as f:
    eval_data = json.load(f)


#### The JSONs are list of dictionaries... each one of the has path and label
##### this is delivered to the class , train val as parameter!!!!

# with open('./datafiles/fsd50k_eval_full.json', 'r') as f:
#     eval_data = json.load(f)
#
# with open('./datafiles/fsd50k_tr_full.json', 'r') as f:
#     train_data = json.load(f)
#
# with open('./datafiles/fsd50k_val_full.json', 'r') as f:
#     val_data = json.load(f)
#
# label_vocabulary = pd.read_csv(r'C:\FSD50K\FSD50K.ground_truth\vocabulary.csv',header=None)
# print(label_vocabulary.iloc[:,2])

# extract paths and labels . labels as Multi-Hot vector
# @ data: wavs and label encoding. see json's.
# @ vocabulary: csv that maps label encoding to numerical value between 0-199
# @ default flag for running on small_data
def make_paths(data, labels_num, vocabulary, run_small_data=False, num_small_samples=10):
    one_hot_vec = torch.zeros(labels_num, 1)
    internal_dict = data['data']
    audio_paths = []
    labels = []
    for cnt, path in enumerate(internal_dict):
        # Access a value in the internal dictionary
        if cnt > num_small_samples and run_small_data:
            break
        one_hot_vec = torch.zeros(labels_num, 1)
        wav = (internal_dict[cnt]).get('wav')
        label = (internal_dict[cnt]).get('labels')
        label = label.split(",")
        label_indices = vocabulary.iloc[:, 2].isin(label)
        one_hot_vec[label_indices] = 1
        audio_paths.append(wav)
        labels.append(one_hot_vec)
    print(type(labels))
    return audio_paths, labels


######### this is delivered to __init__
'''
audio_paths_eval,labels_eval = make_paths(eval_data,label_vocabulary)
print(torch.nonzero(labels_eval[0]))
audio_paths_train,labels_train = make_paths(train_data,label_vocabulary)
audio_paths_val,labels_val = make_paths(val_data,label_vocabulary)
'''


# import torchaudio

# dataset = torchaudio.datasets.LIBRISPEECH('./Dataset', 'train-clean-100', download=True)


class AudioDataset(Dataset):
    def __init__(self, json_path, labels_num, vacbulary_path, run_small_data=False):
        with open(json_path, 'r') as f:
            audio_data = json.load(f)

        label_vocabulary = pd.read_csv(vacbulary_path, header=None)
        audio_paths_list, labels_list = make_paths(audio_data, labels_num, label_vocabulary, run_small_data=run_small_data)

        self.audio_paths = audio_paths_list
        self.labels = labels_list
        self.segments = []
        for audiopath, label in zip(audio_paths_list, labels_list):
            # Get the info of the audio file
            info = sf.info(audiopath)
            sr = info.samplerate
            len_y = info.frames
            label = label
            segments = [(i, i + sr, label, audiopath) for i in range(0, len_y, sr)]

            # Add the segments to the list
            self.segments.extend(segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        # Load the audio data from the file
        i, i_plus_sr, label, audio_wav = self.segments[index]

        audio, sr = sf.read(audio_wav)
        audio = audio.astype('float32')
        # print("index is:",index)

        audio = audio[i:i_plus_sr]
        # Resample the audio data to a sample rate of 16000 Hz
        audio = torch.from_numpy(audio)
        # If the audio data is less than 1 second, repeat it to make the duration 1 second
        if len(audio) < sr:
            audio = torch.cat([audio, torch.zeros(sr - len(audio))])
        return audio, label


def audio_collate_fn(batch):
    # Get the audio samples and labels from the batch
    audio_samples = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    labels = torch.cat(labels, dim=1)
    labels = torch.transpose(labels, 0, 1)
    # Stack the audio samples along the first dimension to form a mini-batch
    audio_batch = torch.stack(audio_samples, dim=0)
    # Creating the 40X400 input matrix
    audio_batch = torch.reshape(audio_batch, (-1, 40, 400))
    #print("what", audio_batch.size())
    return audio_batch, labels


#deliver to main!!!!!!
#Select a subset of the dataset to use
# label_vocabulary_path = r'C:\FSD50K\FSD50K.ground_truth\vocabulary.csv'
# train_path = './datafiles/fsd50k_tr_full.json'
# eval_path = './datafiles/fsd50k_eval_full.json'
# val_path = './datafiles/fsd50k_val_full.json'
#
# train_dataset = AudioDataset(train_path, label_vocabulary_path,run_small_data=True)
# val_dataset = AudioDataset(eval_path, label_vocabulary_path,run_small_data=True)
# eval_dataset = AudioDataset(val_path, label_vocabulary_path,run_small_data=True)
#
# # Create the dataloader  #######!!add num_workers if we have GPU!!!!!!!##########
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=audio_collate_fn)
# val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=audio_collate_fn)
# eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=audio_collate_fn)
# #####delinver to main
#
#
# ## test the execution:
# num_samples = len(train_dataset)
# print(num_samples)
# train_dataloader = tqdm(train_dataloader, total=num_samples, desc="Processing audio data")
# print("wii")
# #Example , Iterate through the dataloader to yield batches of data
#
# for audio_data, labels in train_dataloader:
#     #Train your model on the batch of data
#     pass
#
#     print(f"Audio shape: {labels.size()}")
#     #print(labels.size())
#     #print("dataloader",train_dataloader)
#     #Audio(audio_data[25].numpy(), rate=16000)
