import librosa.feature as feat
import scipy.io.wavfile as ww
import soundfile
import numpy as np
import speechpy
import spafe.features.mfcc as mfcc
import spafe.features.gfcc as gfcc

y, sr = soundfile.read(r'C:\FSD50K\Resampled_dev\699.wav')
if len(y) < 16000:
    y = np.concatenate([y, np.zeros(sr - len(y))]).shape()
y = y[:sr-1]
print(y)

mfcc_feat = feat.mfcc(y=y, sr=sr)
# mfcc_feat_norm = speechpy.processing.cmvn(mfcc_feat, True)
# print(np.shape(mfcc_feat_norm))
# print(mfcc_feat_norm[:, 0])
# print(mfcc_feat[:, 0])

mfcc_delta_feat = feat.delta(mfcc_feat)
mfcc_delta_second_feat = feat.delta(mfcc_delta_feat)
print(np.shape(mfcc_feat))
print(np.shape(mfcc_delta_feat))
print(np.shape(mfcc_delta_second_feat))
stft_feat = feat.chroma_stft(y=y, sr=sr, n_chroma=17)
print(np.shape(stft_feat))

feat_matrix = np.concatenate([mfcc_feat, mfcc_delta_feat, mfcc_delta_second_feat, stft_feat])
print(np.shape(feat_matrix))
tryss = np.concatenate([mfcc_feat])
print(np.shape(tryss))
"""
sr, y2 = ww.read(r'C:\FSD50K\Resampled_dev\136.wav')
print(type(y2.astype('float16')))
mfcc_feats_spafe = mfcc.imfcc(y2)
"""