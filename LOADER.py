import librosa
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import json
from IPython.display import Audio
from tqdm import tqdm


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

# Access an internal dictionary and extract paths and labels for Audiodataset __init__
def make_paths(data):
    internal_dict = data['data']
    audio_paths=[]
    labels=[]
    for cnt,path in enumerate(internal_dict):
    # Access a value in the internal dictionary
        wav = (internal_dict[cnt]).get('wav')
        label = (internal_dict[cnt]).get('labels')
        audio_paths.append(wav)
        labels.append(label)
    return audio_paths,labels

audio_paths_eval,labels_eval = make_paths(eval_data)
audio_paths_train,labels_train = make_paths(train_data)
audio_paths_val,labels_val = make_paths(val_data)


#import torchaudio

#dataset = torchaudio.datasets.LIBRISPEECH('./Dataset', 'train-clean-100', download=True)


class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        # Load the audio file and apply any desired preprocessing
        audio, _ = sf.read(audio_path)
        audio = audio.astype(np.float32)
        audio = audio / np.abs(audio).max()  # Normalize the audio data

        return audio, label

def collate_fn(samples):
    # Find the longest audio signal in the batch
    max_length = max([len(s[0]) for s in samples])

    # Create tensors to hold the padded audio signals and labels/metadata
    audio_tensor = torch.zeros(len(samples), max_length)
    labels = []


    # Pad and stack the audio signals, and collect the labels and metadata
    for i, s in enumerate(samples):
        audio_tensor[i, :len(s[0])] = torch.tensor(s[0])
        labels.append(s[1])


    return audio_tensor, labels


# Select a subset of the dataset to use

train_dataset = AudioDataset(audio_paths_train, labels_train)
val_dataset = AudioDataset(audio_paths_val, labels_val)
eval_dataset = AudioDataset(audio_paths_eval, labels_eval)

# Create the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4,)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

### test the execution:
num_samples = len(train_dataset)
train_dataloader = tqdm(train_dataloader, total=num_samples, desc="Processing audio data")
print("wii")
# Example , Iterate through the dataloader to yield batches of data
for audio_data, labels in train_dataloader:
    #Train your model on the batch of data
    pass
    #print(f"Audio shape: {audio_data.shape}")
    #print("labels")
    #print("dataloader",train_dataloader)
    #Audio(audio_data[25].numpy(), rate=16000)

print("wiiiiiiii")

