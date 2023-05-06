import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import librosa.feature as feat
import librosa.effects as effects
import numpy as np

# Open the JSON file and read the contents
####### this is the input from the user... all other things : make_paths.. in the init
with open('./Hirerachy.json', 'r') as f:
    eval_data = json.load(f)


def make_paths(data, labels_num, vocabulary,num_small_samples, run_small_data=False):
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


# add more feature control, like talked about two weeks back
class AudioDataset(Dataset):
    def __init__(self, json_path, labels_num, vacbulary_path, run_small_data=False, small_data_num=10, mode='audio',
                 feature_type='mfcc', window_size=25, window_type="hamming", num_coeff=20):
        with open(json_path, 'r') as f:
            audio_data = json.load(f)

        label_vocabulary = pd.read_csv(vacbulary_path, header=None)
        audio_paths_list, labels_list = make_paths(audio_data, labels_num, label_vocabulary,
                                                   run_small_data=run_small_data, num_small_samples=small_data_num)

        self.audio_paths = audio_paths_list
        self.labels = labels_list
        self.segments = []
        self.mode = mode
        self.feature_type = feature_type
        self.window_size = window_size
        self.window_type = window_type
        self.num_coeff = num_coeff

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
        if self.mode == 'audio':
            return audio, label
        elif self.mode == 'features':
            feat_matrix = self.extract_features(audio, sr)
            return feat_matrix, label
        else:
            raise ValueError("Invalid mode specified. Must be 'audio' or 'features'.")

    def extract_features(self, y, sr):
        y = effects.preemphasis(y)
        if self.feature_type == 'mfcc':
            mfcc_feat = feat.mfcc(y=y, sr=sr, win_length=self.window_size,
                                  n_mfcc=self.num_coeff, window=self.window_type)
            mfcc_delta = feat.delta(mfcc_feat)
            mfcc_delta2 = feat.delta(mfcc_delta)
            feature_matrix = np.concatenate([mfcc_feat, mfcc_delta, mfcc_delta2])
        elif self.feature_type == 'stft':
            feature_matrix = feat.chroma_stft(y=y, sr=sr, win_length=self.window_size,
                                              window=self.window_type, n_chroma=self.num_coeff)
        else:
            raise ValueError(" Invalid feature type specified. Must be 'mfcc' or 'stft'")
        return feature_matrix


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
    # print("what", audio_batch.size())
    return audio_batch, labels


def features_collate_fn(batch):
    feat_matrix = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    labels = torch.cat(labels, dim=1)
    labels = torch.transpose(labels, 0, 1)
    audio_batch = torch.stack(feat_matrix, dim=0)
    return audio_batch, labels


