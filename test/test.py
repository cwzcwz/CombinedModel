import keras
from model import attention_new
from model import attention_new
import numpy as np
import pandas as pd
import os

# file_path = './08_0.69.h5'
# model = keras.models.load_model(file_path, custom_objects={'AttentionLayer': attention.AttentionLayer})
# print(model.summary())
# for layer in model.layers:
#     print(layer.name)

# sample = []
# a2 = 0
# a = np.array([[-0.065438, -0.039483], [- 0.062554, -0.070794], [-0.058572, -0.055688], [-0.054314, -0.076699]])
# a1 = np.array([[-0.065438, -0.039483], [- 0.062554, -0.070794], [-0.058572, -0.055688], [-0.054314, -0.076699]])
# a2 = a2 + a
# a2 = a2 + a1
# a2 = a2 / 2
#
# for i in np.linspace(0, 2, 2 + 2)[1:2 + 1]:
#     sample.append(a[int(i)])
# print(np.array(sample).shape)

# print(len([[-0.065438, -0.039483], [- 0.062554, -0.070794], [-0.058572, -0.055688], [-0.054314, -0.076699]]))
# order = list(range(20))
# self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
#                        range(0, len(order), self.batch_size)]


# file = './local_000000824_20190617095200_data_aug-samplerate-max-3-50-0.txt'
# x = pd.read_csv(file, header=None, ).T
# x = x[1:].T.values
# print(x)
# print(x.shape)
#
# eeg_file = open(file)
# eeg_datas = []
# for i, line in enumerate(eeg_file.readlines()):
#     a = line.strip().split(',')[1:]
#     eeg_datas.append([float(x.strip()) for x in a])
# eeg_datas = np.array(eeg_datas, dtype=float)
# print(eeg_datas)
# print(eeg_datas.shape)

# total_samples = 10
# sample_num = 10
# for i in np.linspace(0, total_samples, sample_num + 2, dtype=np.int)[1:sample_num + 1]:
#
#     print('sss:', i)
#     clip_start = i - int(6 / 2)
#     clip_end = i + int(6 / 2)
#     if clip_start < 0:
#         clip_end = clip_end - clip_start
#         clip_start = 0
#     if clip_end > total_samples:
#         clip_start = clip_start - (clip_end - total_samples)
#         clip_end = total_samples
#     print(clip_start)
#     print(clip_end)

# for a in os.walk('D:/迅雷下载'):
#     print(a[0])
#     print(a[1])
#     print(a[2])
#     print('end')

# import matplotlib.pyplot as plt
#
# eeg_path = './local_000000824_20190617095200_data_aug_samplerate_mean_3_50_0.txt'
# x = pd.read_csv(eeg_path, header=None).T
# eeg_datas = x[1:].T.values
# plt.plot(eeg_datas)
# plt.show()

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
d = np.reshape(x, (2, 4))
print(d)
d = np.resize(d, (2, 6))
print(d)
