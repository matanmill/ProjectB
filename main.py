import os
import json
from Paths import FSD50K_paths


def fs_tree(root):
    results = {}
    for (dirpath, dirnames, filenames) in os.walk(root):
        parts = dirpath.split(os.sep)
        curr = results
        for p in parts:
            curr = curr.setdefault(p, {})
    return results


def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


def resample_audio(base_path, target_path):
    resample_cnt = 0
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    files = get_immediate_files(base_path)
    for audiofile in files:
        os.system('sox ' + base_path + audiofile + ' -r 16000 ' + target_path + audiofile + '> /dev/null 2>&1')
        resample_cnt += 1
        if resample_cnt % 50 == 0:
            print('Resampled {:d} samples.'.format(resample_cnt))
    print('Resampling finished.')
    print('--------------------------------------------')


# checking what the resample audio does
dev_small_path = FSD50K_paths["code_exploring_dev"]
resampled_path = r'C:\FSD50K\Code_Exploring\resampled_dev'

resample_audio(dev_small_path, resampled_path)
