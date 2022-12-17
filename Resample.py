# python code regarding the resample of audio files
import os
import soundfile as sf
import librosa
from Paths import FSD50K_paths

# paths
fsd_path_dev = FSD50K_paths['code_exploring_dev']
resampled_path_dev = r'C:\FSD50K\Code_Exploring\dev_resampled'
fsd_path_eval = FSD50K_paths['code_exploring_eval']
resampled_path_eval = r'C:\FSD50K\Code_Exploring\eval_resampled'


def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


def resample_audio(base_path, target_path):
    resample_cnt = 0
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    files = get_immediate_files(base_path)
    for audiofile in files:
        source_file = os.path.join(base_path, audiofile)
        audiofile = os.path.splitext(audiofile)[0]
        file_name = audiofile + "_resampled.wav"
        output_file = os.path.join(target_path, file_name)
        output_str = f"ffmpeg -i {source_file} -ac 1 -ar 16000 {output_file}"
        os.system(output_str)
    print('Resampling finished.')
    print('--------------------------------------------')


print("NOW resampling FSD50K to 16 KHz")
resample_audio(fsd_path_dev, resampled_path_dev)
resample_audio(fsd_path_eval, resampled_path_eval)
