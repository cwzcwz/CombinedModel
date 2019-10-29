"""
数据增强
"""
import numpy as np
import os
import pandas as pd

data_dir = 'E:/data'


# rgb的数据增强方法
def rgb_samplerate_3():
    rgbdata_dir_name = 'rgbimage_list'
    aug_rgbdata_name = 'aug_rgbimage_list'
    sample_rate = 3
    rgbdata_path = os.path.join(data_dir, rgbdata_dir_name)
    aug_rgbdata_path = os.path.join(data_dir, aug_rgbdata_name)
    if not os.path.exists(aug_rgbdata_path):
        os.makedirs(aug_rgbdata_path)

    for i in os.listdir(rgbdata_path):
        txt_path = os.path.join(rgbdata_path, i)
        aug_txt_path = os.path.join(aug_rgbdata_path,
                                    i[:-5] + 'aug-samplerate-' + str(sample_rate) + '-' + i[-5] + '.txt')
        x = pd.read_csv(txt_path, header=None)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            image_name = x[j][0]
            sample_data.append(image_name)
        with open(aug_txt_path, 'w') as f:
            for k in sample_data:
                f.write(k + '\n')


def rgb_samplerate_4():
    rgbdata_dir_name = 'rgbimage_list'
    aug_rgbdata_name = 'aug_rgbimage_list'
    sample_rate = 4
    rgbdata_path = os.path.join(data_dir, rgbdata_dir_name)
    aug_rgbdata_path = os.path.join(data_dir, aug_rgbdata_name)
    if not os.path.exists(aug_rgbdata_path):
        os.makedirs(aug_rgbdata_path)

    for i in os.listdir(rgbdata_path):
        txt_path = os.path.join(rgbdata_path, i)
        aug_txt_path = os.path.join(aug_rgbdata_path,
                                    i[:-5] + 'aug-samplerate-' + str(sample_rate) + '-' + i[-5] + '.txt')
        x = pd.read_csv(txt_path, header=None)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            image_name = x[j][0]
            sample_data.append(image_name)
        with open(aug_txt_path, 'w') as f:
            for k in sample_data:
                f.write(k + '\n')


def rgb_samplerate_5():
    rgbdata_dir_name = 'rgbimage_list'
    aug_rgbdata_name = 'aug_rgbimage_list'
    sample_rate = 5
    rgbdata_path = os.path.join(data_dir, rgbdata_dir_name)
    aug_rgbdata_path = os.path.join(data_dir, aug_rgbdata_name)
    if not os.path.exists(aug_rgbdata_path):
        os.makedirs(aug_rgbdata_path)

    for i in os.listdir(rgbdata_path):
        txt_path = os.path.join(rgbdata_path, i)
        aug_txt_path = os.path.join(aug_rgbdata_path,
                                    i[:-5] + 'aug-samplerate-' + str(sample_rate) + '-' + i[-5] + '.txt')
        x = pd.read_csv(txt_path, header=None)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            image_name = x[j][0]
            sample_data.append(image_name)
        with open(aug_txt_path, 'w') as f:
            for k in sample_data:
                f.write(k + '\n')


def rgb_samplerate_6():
    rgbdata_dir_name = 'rgbimage_list'
    aug_rgbdata_name = 'aug_rgbimage_list'
    sample_rate = 6
    rgbdata_path = os.path.join(data_dir, rgbdata_dir_name)
    aug_rgbdata_path = os.path.join(data_dir, aug_rgbdata_name)
    if not os.path.exists(aug_rgbdata_path):
        os.makedirs(aug_rgbdata_path)

    for i in os.listdir(rgbdata_path):
        txt_path = os.path.join(rgbdata_path, i)
        aug_txt_path = os.path.join(aug_rgbdata_path,
                                    i[:-5] + 'aug-samplerate-' + str(sample_rate) + '-' + i[-5] + '.txt')
        x = pd.read_csv(txt_path, header=None)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            image_name = x[j][0]
            sample_data.append(image_name)
        with open(aug_txt_path, 'w') as f:
            for k in sample_data:
                f.write(k + '\n')


# 深度图的增强方法
def depth_samplerate_3():
    depthdata_dir_name = 'depthimage_list'
    aug_depthdata_name = 'aug_depthimage_list'
    sample_rate = 3
    depthdata_path = os.path.join(data_dir, depthdata_dir_name)
    aug_depthdata_path = os.path.join(data_dir, aug_depthdata_name)
    if not os.path.exists(aug_depthdata_path):
        os.makedirs(aug_depthdata_path)

    for i in os.listdir(depthdata_path):
        txt_path = os.path.join(depthdata_path, i)
        aug_txt_path = os.path.join(aug_depthdata_path,
                                    i[:-5] + 'aug-samplerate-' + str(sample_rate) + '-' + i[-5] + '.txt')
        x = pd.read_csv(txt_path, header=None)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            image_name = x[j][0]
            sample_data.append(image_name)
        with open(aug_txt_path, 'w') as f:
            for k in sample_data:
                f.write(k + '\n')


def depth_samplerate_4():
    depthdata_dir_name = 'depthimage_list'
    aug_depthdata_name = 'aug_depthimage_list'
    sample_rate = 4
    depthdata_path = os.path.join(data_dir, depthdata_dir_name)
    aug_depthdata_path = os.path.join(data_dir, aug_depthdata_name)
    if not os.path.exists(aug_depthdata_path):
        os.makedirs(aug_depthdata_path)

    for i in os.listdir(depthdata_path):
        txt_path = os.path.join(depthdata_path, i)
        aug_txt_path = os.path.join(aug_depthdata_path,
                                    i[:-5] + 'aug-samplerate-' + str(sample_rate) + '-' + i[-5] + '.txt')
        x = pd.read_csv(txt_path, header=None)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            image_name = x[j][0]
            sample_data.append(image_name)
        with open(aug_txt_path, 'w') as f:
            for k in sample_data:
                f.write(k + '\n')


def depth_samplerate_5():
    depthdata_dir_name = 'depthimage_list'
    aug_depthdata_name = 'aug_depthimage_list'
    sample_rate = 5
    depthdata_path = os.path.join(data_dir, depthdata_dir_name)
    aug_depthdata_path = os.path.join(data_dir, aug_depthdata_name)
    if not os.path.exists(aug_depthdata_path):
        os.makedirs(aug_depthdata_path)

    for i in os.listdir(depthdata_path):
        txt_path = os.path.join(depthdata_path, i)
        aug_txt_path = os.path.join(aug_depthdata_path,
                                    i[:-5] + 'aug-samplerate-' + str(sample_rate) + '-' + i[-5] + '.txt')
        x = pd.read_csv(txt_path, header=None)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            image_name = x[j][0]
            sample_data.append(image_name)
        with open(aug_txt_path, 'w') as f:
            for k in sample_data:
                f.write(k + '\n')


def depth_samplerate_6():
    depthdata_dir_name = 'depthimage_list'
    aug_depthdata_name = 'aug_depthimage_list'
    sample_rate = 6
    depthdata_path = os.path.join(data_dir, depthdata_dir_name)
    aug_depthdata_path = os.path.join(data_dir, aug_depthdata_name)
    if not os.path.exists(aug_depthdata_path):
        os.makedirs(aug_depthdata_path)

    for i in os.listdir(depthdata_path):
        txt_path = os.path.join(depthdata_path, i)
        aug_txt_path = os.path.join(aug_depthdata_path,
                                    i[:-5] + 'aug-samplerate-' + str(sample_rate) + '-' + i[-5] + '.txt')
        x = pd.read_csv(txt_path, header=None)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            image_name = x[j][0]
            sample_data.append(image_name)
        with open(aug_txt_path, 'w') as f:
            for k in sample_data:
                f.write(k + '\n')


def skeleton_samplerate_3():
    skeleton_dir_name = 'skeletontxt'
    aug_skeleton_dir_name = 'aug_skeletontxt'
    sample_rate = 3
    skeleton_path = os.path.join(data_dir, skeleton_dir_name)
    aug_skeleton_path = os.path.join(data_dir, aug_skeleton_dir_name)
    if not os.path.exists(aug_skeleton_path):
        os.makedirs(aug_skeleton_path)
    for i in os.listdir(skeleton_path):
        dir_path = os.path.join(skeleton_path, i)
        aug_dir_path = os.path.join(aug_skeleton_path, i[:-1] + 'aug-samplerate-' + str(sample_rate) + '-' + i[-1])
        if not os.path.exists(aug_dir_path):
            os.makedirs(aug_dir_path)
        for j in os.listdir(dir_path):
            txt_path = os.path.join(dir_path, j)
            aug_txt_path = os.path.join(aug_dir_path, j)
            x = pd.read_csv(txt_path, header=None)
            x = x.values
            sample_data = []
            total_sample = x.shape[0]
            rate_sample = int(total_sample / sample_rate)
            for k in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
                line_data = x[k][0]
                sample_data.append(line_data)
            with open(aug_txt_path, 'w') as f:
                for l in sample_data:
                    f.write(l + '\n')


def skeleton_samplerate_4():
    skeleton_dir_name = 'skeletontxt'
    aug_skeleton_dir_name = 'aug_skeletontxt'
    sample_rate = 4
    skeleton_path = os.path.join(data_dir, skeleton_dir_name)
    aug_skeleton_path = os.path.join(data_dir, aug_skeleton_dir_name)
    if not os.path.exists(aug_skeleton_path):
        os.makedirs(aug_skeleton_path)
    for i in os.listdir(skeleton_path):
        dir_path = os.path.join(skeleton_path, i)
        aug_dir_path = os.path.join(aug_skeleton_path, i[:-1] + 'aug-samplerate-' + str(sample_rate) + '-' + i[-1])
        if not os.path.exists(aug_dir_path):
            os.makedirs(aug_dir_path)
        for j in os.listdir(dir_path):
            txt_path = os.path.join(dir_path, j)
            aug_txt_path = os.path.join(aug_dir_path, j)
            x = pd.read_csv(txt_path, header=None)
            x = x.values
            sample_data = []
            total_sample = x.shape[0]
            rate_sample = int(total_sample / sample_rate)
            for k in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
                line_data = x[k][0]
                sample_data.append(line_data)
            with open(aug_txt_path, 'w') as f:
                for l in sample_data:
                    f.write(l + '\n')


def skeleton_samplerate_5():
    skeleton_dir_name = 'skeletontxt'
    aug_skeleton_dir_name = 'aug_skeletontxt'
    sample_rate = 5
    skeleton_path = os.path.join(data_dir, skeleton_dir_name)
    aug_skeleton_path = os.path.join(data_dir, aug_skeleton_dir_name)
    if not os.path.exists(aug_skeleton_path):
        os.makedirs(aug_skeleton_path)
    for i in os.listdir(skeleton_path):
        dir_path = os.path.join(skeleton_path, i)
        aug_dir_path = os.path.join(aug_skeleton_path, i[:-1] + 'aug-samplerate-' + str(sample_rate) + '-' + i[-1])
        if not os.path.exists(aug_dir_path):
            os.makedirs(aug_dir_path)
        for j in os.listdir(dir_path):
            txt_path = os.path.join(dir_path, j)
            aug_txt_path = os.path.join(aug_dir_path, j)
            x = pd.read_csv(txt_path, header=None)
            x = x.values
            sample_data = []
            total_sample = x.shape[0]
            rate_sample = int(total_sample / sample_rate)
            for k in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
                line_data = x[k][0]
                sample_data.append(line_data)
            with open(aug_txt_path, 'w') as f:
                for l in sample_data:
                    f.write(l + '\n')


def skeleton_samplerate_6():
    skeleton_dir_name = 'skeletontxt'
    aug_skeleton_dir_name = 'aug_skeletontxt'
    sample_rate = 6
    skeleton_path = os.path.join(data_dir, skeleton_dir_name)
    aug_skeleton_path = os.path.join(data_dir, aug_skeleton_dir_name)
    if not os.path.exists(aug_skeleton_path):
        os.makedirs(aug_skeleton_path)
    for i in os.listdir(skeleton_path):
        dir_path = os.path.join(skeleton_path, i)
        aug_dir_path = os.path.join(aug_skeleton_path, i[:-1] + 'aug-samplerate-' + str(sample_rate) + '-' + i[-1])
        if not os.path.exists(aug_dir_path):
            os.makedirs(aug_dir_path)
        for j in os.listdir(dir_path):
            txt_path = os.path.join(dir_path, j)
            aug_txt_path = os.path.join(aug_dir_path, j)
            x = pd.read_csv(txt_path, header=None)
            x = x.values
            sample_data = []
            total_sample = x.shape[0]
            rate_sample = int(total_sample / sample_rate)
            for k in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
                line_data = x[k][0]
                sample_data.append(line_data)
            with open(aug_txt_path, 'w') as f:
                for l in sample_data:
                    f.write(l + '\n')


def eeg_samplerate_3_50_mean():
    eeg_dir_name = 'eeg'
    aug_eeg_dir_name = 'aug_eeg'
    sample_rate = 3
    sample_timesteps = 50
    eeg_path = os.path.join(data_dir, eeg_dir_name)
    aug_eeg_path = os.path.join(data_dir, aug_eeg_dir_name)
    if not os.path.exists(aug_eeg_path):
        os.makedirs(aug_eeg_path)
    for i in os.listdir(eeg_path):
        txt_path = os.path.join(eeg_path, i)
        aug_txt_path = os.path.join(aug_eeg_path,
                                    i[:-5] + 'aug_samplerate_mean_' + str(sample_rate) + '_' + str(
                                        sample_timesteps) + '_' +
                                    i[-5] + '.txt')
        x = pd.read_csv(txt_path)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            clip_start = j - int(sample_timesteps / 2)
            clip_end = j + int(sample_timesteps / 2)
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > total_sample:
                clip_start = clip_start - (clip_end - total_sample)
                clip_end = total_sample
            clip = x[clip_start:clip_end]
            clip = np.mean(clip, axis=0)
            sample_data.append(clip)
        sample_data = np.array(sample_data)
        np.savetxt(aug_txt_path, sample_data, fmt='%7.6f', delimiter=',')


def eeg_samplerate_3_100_mean():
    eeg_dir_name = 'eeg'
    aug_eeg_dir_name = 'aug_eeg'
    sample_rate = 3
    sample_timesteps = 100
    eeg_path = os.path.join(data_dir, eeg_dir_name)
    aug_eeg_path = os.path.join(data_dir, aug_eeg_dir_name)
    if not os.path.exists(aug_eeg_path):
        os.makedirs(aug_eeg_path)
    for i in os.listdir(eeg_path):
        txt_path = os.path.join(eeg_path, i)
        aug_txt_path = os.path.join(aug_eeg_path,
                                    i[:-5] + 'aug_samplerate_mean_' + str(sample_rate) + '_' + str(
                                        sample_timesteps) + '_' +
                                    i[-5] + '.txt')
        x = pd.read_csv(txt_path)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            clip_start = j - int(sample_timesteps / 2)
            clip_end = j + int(sample_timesteps / 2)
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > total_sample:
                clip_start = clip_start - (clip_end - total_sample)
                clip_end = total_sample
            clip = x[clip_start:clip_end]
            clip = np.mean(clip, axis=0)
            sample_data.append(clip)
        sample_data = np.array(sample_data)
        np.savetxt(aug_txt_path, sample_data, fmt='%7.6f', delimiter=',')


def eeg_samplerate_5_50_mean():
    eeg_dir_name = 'eeg'
    aug_eeg_dir_name = 'aug_eeg'
    sample_rate = 5
    sample_timesteps = 50
    eeg_path = os.path.join(data_dir, eeg_dir_name)
    aug_eeg_path = os.path.join(data_dir, aug_eeg_dir_name)
    if not os.path.exists(aug_eeg_path):
        os.makedirs(aug_eeg_path)
    for i in os.listdir(eeg_path):
        txt_path = os.path.join(eeg_path, i)
        aug_txt_path = os.path.join(aug_eeg_path,
                                    i[:-5] + 'aug_samplerate_mean_' + str(sample_rate) + '_' + str(
                                        sample_timesteps) + '_' +
                                    i[-5] + '.txt')
        x = pd.read_csv(txt_path)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            clip_start = j - int(sample_timesteps / 2)
            clip_end = j + int(sample_timesteps / 2)
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > total_sample:
                clip_start = clip_start - (clip_end - total_sample)
                clip_end = total_sample
            clip = x[clip_start:clip_end]
            clip = np.mean(clip, axis=0)
            sample_data.append(clip)
        sample_data = np.array(sample_data)
        np.savetxt(aug_txt_path, sample_data, fmt='%7.6f', delimiter=',')


def eeg_samplerate_5_100_mean():
    eeg_dir_name = 'eeg'
    aug_eeg_dir_name = 'aug_eeg'
    sample_rate = 5
    sample_timesteps = 100
    eeg_path = os.path.join(data_dir, eeg_dir_name)
    aug_eeg_path = os.path.join(data_dir, aug_eeg_dir_name)
    if not os.path.exists(aug_eeg_path):
        os.makedirs(aug_eeg_path)
    for i in os.listdir(eeg_path):
        txt_path = os.path.join(eeg_path, i)
        aug_txt_path = os.path.join(aug_eeg_path,
                                    i[:-5] + 'aug_samplerate_mean_' + str(sample_rate) + '_' + str(
                                        sample_timesteps) + '_' +
                                    i[-5] + '.txt')
        x = pd.read_csv(txt_path)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            clip_start = j - int(sample_timesteps / 2)
            clip_end = j + int(sample_timesteps / 2)
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > total_sample:
                clip_start = clip_start - (clip_end - total_sample)
                clip_end = total_sample
            clip = x[clip_start:clip_end]
            clip = np.mean(clip, axis=0)
            sample_data.append(clip)
        sample_data = np.array(sample_data)
        np.savetxt(aug_txt_path, sample_data, fmt='%7.6f', delimiter=',')


def eeg_samplerate_3_50_max():
    eeg_dir_name = 'eeg'
    aug_eeg_dir_name = 'aug_eeg'
    sample_rate = 3
    sample_timesteps = 50
    eeg_path = os.path.join(data_dir, eeg_dir_name)
    aug_eeg_path = os.path.join(data_dir, aug_eeg_dir_name)
    if not os.path.exists(aug_eeg_path):
        os.makedirs(aug_eeg_path)
    for i in os.listdir(eeg_path):
        txt_path = os.path.join(eeg_path, i)
        aug_txt_path = os.path.join(aug_eeg_path,
                                    i[:-5] + 'aug_samplerate_max_' + str(sample_rate) + '_' + str(
                                        sample_timesteps) + '_' +
                                    i[-5] + '.txt')
        x = pd.read_csv(txt_path)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            clip_start = j - int(sample_timesteps / 2)
            clip_end = j + int(sample_timesteps / 2)
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > total_sample:
                clip_start = clip_start - (clip_end - total_sample)
                clip_end = total_sample
            clip = x[clip_start:clip_end]
            clip = np.max(clip, axis=0)
            sample_data.append(clip)
        sample_data = np.array(sample_data)
        np.savetxt(aug_txt_path, sample_data, fmt='%7.6f', delimiter=',')


def eeg_samplerate_3_100_max():
    eeg_dir_name = 'eeg'
    aug_eeg_dir_name = 'aug_eeg'
    sample_rate = 3
    sample_timesteps = 100
    eeg_path = os.path.join(data_dir, eeg_dir_name)
    aug_eeg_path = os.path.join(data_dir, aug_eeg_dir_name)
    if not os.path.exists(aug_eeg_path):
        os.makedirs(aug_eeg_path)
    for i in os.listdir(eeg_path):
        txt_path = os.path.join(eeg_path, i)
        aug_txt_path = os.path.join(aug_eeg_path,
                                    i[:-5] + 'aug_samplerate_max_' + str(sample_rate) + '_' + str(
                                        sample_timesteps) + '_' +
                                    i[-5] + '.txt')
        x = pd.read_csv(txt_path)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            clip_start = j - int(sample_timesteps / 2)
            clip_end = j + int(sample_timesteps / 2)
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > total_sample:
                clip_start = clip_start - (clip_end - total_sample)
                clip_end = total_sample
            clip = x[clip_start:clip_end]
            clip = np.max(clip, axis=0)
            sample_data.append(clip)
        sample_data = np.array(sample_data)
        np.savetxt(aug_txt_path, sample_data, fmt='%7.6f', delimiter=',')


def eeg_samplerate_5_50_max():
    eeg_dir_name = 'eeg'
    aug_eeg_dir_name = 'aug_eeg'
    sample_rate = 5
    sample_timesteps = 50
    eeg_path = os.path.join(data_dir, eeg_dir_name)
    aug_eeg_path = os.path.join(data_dir, aug_eeg_dir_name)
    if not os.path.exists(aug_eeg_path):
        os.makedirs(aug_eeg_path)
    for i in os.listdir(eeg_path):
        txt_path = os.path.join(eeg_path, i)
        aug_txt_path = os.path.join(aug_eeg_path,
                                    i[:-5] + 'aug_samplerate_max_' + str(sample_rate) + '_' + str(
                                        sample_timesteps) + '_' +
                                    i[-5] + '.txt')
        x = pd.read_csv(txt_path)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            clip_start = j - int(sample_timesteps / 2)
            clip_end = j + int(sample_timesteps / 2)
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > total_sample:
                clip_start = clip_start - (clip_end - total_sample)
                clip_end = total_sample
            clip = x[clip_start:clip_end]
            clip = np.max(clip, axis=0)
            sample_data.append(clip)
        sample_data = np.array(sample_data)
        np.savetxt(aug_txt_path, sample_data, fmt='%7.6f', delimiter=',')


def eeg_samplerate_5_100_max():
    eeg_dir_name = 'eeg'
    aug_eeg_dir_name = 'aug_eeg'
    sample_rate = 5
    sample_timesteps = 100
    eeg_path = os.path.join(data_dir, eeg_dir_name)
    aug_eeg_path = os.path.join(data_dir, aug_eeg_dir_name)
    if not os.path.exists(aug_eeg_path):
        os.makedirs(aug_eeg_path)
    for i in os.listdir(eeg_path):
        txt_path = os.path.join(eeg_path, i)
        aug_txt_path = os.path.join(aug_eeg_path,
                                    i[:-5] + 'aug_samplerate_max_' + str(sample_rate) + '_' + str(
                                        sample_timesteps) + '_' +
                                    i[-5] + '.txt')
        x = pd.read_csv(txt_path)
        x = x.values
        sample_data = []
        total_sample = x.shape[0]
        rate_sample = int(total_sample / sample_rate)
        for j in np.linspace(0, total_sample, rate_sample + 2, dtype=np.int)[1:rate_sample + 1]:
            clip_start = j - int(sample_timesteps / 2)
            clip_end = j + int(sample_timesteps / 2)
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > total_sample:
                clip_start = clip_start - (clip_end - total_sample)
                clip_end = total_sample
            clip = x[clip_start:clip_end]
            clip = np.max(clip, axis=0)
            sample_data.append(clip)
        sample_data = np.array(sample_data)
        np.savetxt(aug_txt_path, sample_data, fmt='%7.6f', delimiter=',')


if __name__ == "__main__":
    # rgb_samplerate_3()
    # rgb_samplerate_4()
    # rgb_samplerate_5()
    # rgb_samplerate_6()
    # depth_samplerate_3()
    # depth_samplerate_4()
    # depth_samplerate_5()
    # depth_samplerate_6()
    # skeleton_samplerate_3()
    # skeleton_samplerate_4()
    # skeleton_samplerate_5()
    # skeleton_samplerate_6()
    eeg_samplerate_3_50_mean()
    eeg_samplerate_3_100_mean()
    eeg_samplerate_5_50_mean()
    eeg_samplerate_5_100_mean()
    eeg_samplerate_3_50_max()
    eeg_samplerate_3_100_max()
    eeg_samplerate_5_50_max()
    eeg_samplerate_5_100_max()
