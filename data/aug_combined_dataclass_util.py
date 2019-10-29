"""preprocess dataset"""

import os
import random
from data import aug_sorted_dir
import numpy as np

train_percent = 0.9  # 训练数据和交叉验证数据占的比例，自己根据实际调节
val_percent = 0.1  # 训练数据占trainval的比例，即用来训练的数据
eegallfilepath = 'E:/data/aug_eeg'
rgballfilepath = 'E:/data/aug_rgbimage_list'
depthallfilepath = 'E:/data/aug_depthimage_list'
skeletonallfilepath = 'E:/data/aug_skeletontxt'
txtsavepath = './'

eeg_total_file = aug_sorted_dir.sorted_eeg_data(eegallfilepath)
eeg_total_file = np.array(eeg_total_file).repeat(2).tolist()

rgb_total_file = aug_sorted_dir.sorded_kinect_data(rgballfilepath)
depth_total_file = aug_sorted_dir.sorded_kinect_data(depthallfilepath)
skeleton_total_file = aug_sorted_dir.sorded_kinect_data(skeletonallfilepath)

num = len(rgb_total_file)
list = range(num)
tv = int(num * train_percent)
train = random.sample(list, tv)

ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')

for i in list:
    name = eeg_total_file[i]
    name = name + ' ' + rgb_total_file[i]
    name = name + ' ' + depth_total_file[i]
    name = name + ' ' + skeleton_total_file[i]
    if name[-1] == '0':
        name = name + ' 0' + '\n'
    if name[-1] == '1':
        name = name + ' 1' + '\n'

    if i in train:
        ftrain.write(name)
    else:
        fval.write(name)

ftrain.close()
fval.close()
