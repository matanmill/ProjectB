import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import resampy

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
    audio_paths = []
    labels_b = []

    for cnt, path in enumerate(internal_dict):
        # Access a value in the internal dictionary
        wav = (internal_dict[cnt]).get('wav')
        label = (internal_dict[cnt]).get('labels')
        audio_paths.append(wav)
        labels_b.append(label)
    return audio_paths, labels_b


audio_paths_eval, labels_eval = make_paths(eval_data)
audio_paths_train, labels_train = make_paths(train_data)
audio_paths_val, labels_val = make_paths(val_data)


# import torchaudio

# dataset = torchaudio.datasets.LIBRISPEECH('./Dataset', 'train-clean-100', download=True)


class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels
        self.segments = []
        for audiopath, label in zip(audio_paths, labels):

            # Load the audio data from the file
            audio, sr = sf.read(audiopath)
            # Resample the audio data to a sample rate of 16000 Hz
            audio = resampy.resample(audio, sr, 16000)
            print(f"Resampled audio data shape: {audio.shape}")

            # If the audio data is less than 1 second, repeat it to make the duration 1 second
            if len(audio) < sr:
                audio = np.tile(audio, int(16000 / len(audio)))

            # Split the audio data into non-overlapping segments of 1 second
            segments = [(i, audio[i:i+sr], label) for i in range(0, len(audio), 16000)]

            # Add the segments to the list, extend is for multiple elements to be added
            self.segments.extend(segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        # Return the segments, labels
        return self.segments[index]


def collate_fn(samples):
    # Find the longest audio signal in the batch
    max_length = max([len(s[0]) for s in samples])

    # Create tensors to hold the padded audio signals and labels/metadata
    audio_tensor = torch.zeros(len(samples), max_length)
    labels_b = []

    # Pad and stack the audio signals, and collect the labels and metadata
    for i, s in enumerate(samples):
        audio_tensor[i, :len(s[0])] = torch.tensor(s[0])
        labels.append(s[1])

    return audio_tensor, labels_b


# Select a subset of the dataset to use

train_dataset = AudioDataset(audio_paths_train, labels_train)
val_dataset = AudioDataset(audio_paths_val, labels_val)
eval_dataset = AudioDataset(audio_paths_eval, labels_eval)

# Create the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=1)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

### test the execution:
num_samples = len(train_dataset)
train_dataloader = tqdm(train_dataloader, total=num_samples, desc="Processing audio data")
print("wii")
# Example , Iterate through the dataloader to yield batches of data
if __name__ == '__main__':
    for audio_data, labels in train_dataloader:
        #Train your model on the batch of data
        pass
        #print(f"Audio shape: {audio_data.shape}")
        #print("labels")
        #print("dataloader",train_dataloader)
        #Audio(audio_data[25].numpy(), rate=16000)

print("wiiiiiiii")