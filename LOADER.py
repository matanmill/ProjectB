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
with open('./Hirerachy.json', 'r') as f:
    eval_data = json.load(f)
#### The JSONs are list of dictionaries... each one of the has path and label
with open('./datafiles/fsd50k_eval_full.json', 'r') as f:
    eval_data = json.load(f)

with open('./datafiles/fsd50k_tr_full.json', 'r') as f:
    train_data = json.load(f)

with open('./datafiles/fsd50k_val_full.json', 'r') as f:
    val_data = json.load(f)

label_vocabulary = pd.read_csv(r'C:\FSD50K\FSD50K.ground_truth\vocabulary.csv',header=None)
print(label_vocabulary.iloc[:,2])

# Access an internal dictionary and extract paths and labels for Audiodataset __init__
def make_paths(data,vocabulary):
    one_hot_vec = torch.zeros(200,1)
    internal_dict = data['data']
    audio_paths=[]
    labels=[]
    for cnt,path in enumerate(internal_dict):
    # Access a value in the internal dictionary
        one_hot_vec = torch.zeros(200, 1)
        wav = (internal_dict[cnt]).get('wav')
        label = (internal_dict[cnt]).get('labels')
        label = label.split(",")
        label_indices = vocabulary.iloc[:, 2].isin(label)
        one_hot_vec[label_indices] = 1
        audio_paths.append(wav)
        labels.append(one_hot_vec)
    return audio_paths,labels

audio_paths_eval,labels_eval = make_paths(eval_data,label_vocabulary)
print(torch.nonzero(labels_eval[0]))
audio_paths_train,labels_train = make_paths(train_data,label_vocabulary)
audio_paths_val,labels_val = make_paths(val_data,label_vocabulary)


#import torchaudio

#dataset = torchaudio.datasets.LIBRISPEECH('./Dataset', 'train-clean-100', download=True)

class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels
        self.segments = []
        for audiopath, label in zip(audio_paths, labels):
            # Get the info of the audio file
            info = sf.info(audiopath)
            sr = info.samplerate
            len_y = info.frames
            label = label.nonzero().flatten()
            segments = [(i, i+sr, label) for i in range(0, len_y, sr)]

            # Add the segments to the list
            self.segments.extend(segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        # Load the audio data from the file
        audio, sr = sf.read(self.audio_paths[index])
        print("index is:",index)
        i, i_plus_sr, label = self.segments[index]
        audio = audio[i:i_plus_sr]
        # Resample the audio data to a sample rate of 16000 Hz
        audio = resampy.resample(audio, sr, 16000)
        print(f"Resampled audio data shape: {audio.shape}")

        # If the audio data is less than 1 second, repeat it to make the duration 1 second
        if len(audio) < sr:
            audio = np.tile(audio, int(16000 / len(audio)))
        return audio, label



def audio_collate_fn(batch):
    # Get the audio samples and labels from the batch
    audio_samples = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # Stack the audio samples along the first dimension to form a mini-batch
    audio_batch = torch.stack(audio_samples, dim=0)
    return audio_batch, labels




# Select a subset of the dataset to use

train_dataset = AudioDataset(audio_paths_train, labels_train)
val_dataset = AudioDataset(audio_paths_val, labels_val)
eval_dataset = AudioDataset(audio_paths_eval, labels_eval)

# Create the dataloader  #######!!add num_workers if we have GPU!!!!!!!##########
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=audio_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=audio_collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=audio_collate_fn)

### test the execution:
num_samples = len(train_dataset)
print(num_samples)
train_dataloader = tqdm(train_dataloader, total=num_samples, desc="Processing audio data")
print("wii")
'''
# Example , Iterate through the dataloader to yield batches of data
if __name__ == '__main__':
    for audio_data, labels in train_dataloader:
        #Train your model on the batch of data
        pass
        #print(f"Audio shape: {audio_data.shape}")
        #print("labels")
        #print("dataloader",train_dataloader)
        #Audio(audio_data[25].numpy(), rate=16000)
'''
print("wiiiiiiii")

