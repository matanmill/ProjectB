import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from audioset import AudioSet

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
        audio, _ = librosa.load(audio_path, sr=44100)
        audio = audio.astype(np.float32)
        audio = audio / np.abs(audio).max()  # Normalize the audio data

        return audio, label

def batching_function(batch):
    # Unpack the batch
    audio_data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Convert audio data to tensors
    audio_data = torch.from_numpy(np.stack(audio_data))

    # Convert labels to tensors
    labels = torch.from_numpy(np.stack(labels))

    return audio_data, labels

# Load the AudioSet dataset
dataset = AudioSet()

# Select a subset of the dataset to use
audio_paths = dataset.paths[:1000]  # Select the first 1000 audio files
labels = dataset.labels[:1000]  # Select the labels for the first 1000 audio files
print("before:",labels)

# Create the dataset
audio_dataset = AudioDataset(audio_paths, labels)

# Create the dataloader
dataloader = DataLoader(audio_dataset, batch_size=32, shuffle=True, collate_fn=batching_function)

# Iterate through the dataloader to yield batches of data
for audio_data, labels in dataloader:
    # Train your model on the batch of data
    print(audio_data)
    print(labels)
