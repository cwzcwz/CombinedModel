from preprocess.generator import Generator

import os
import pandas as pd
import numpy as np

dissaOrsa_class = {
    'satisfaction': 0,
    'dissatisfaction': 1
}


class CombinedDataGenerator(Generator):
    """ Generate data for a Pascal VOC dataset."""

    def __init__(
            self,
            data_dir,
            set_name,
            classes=dissaOrsa_class,
            **kwargs
    ):
        self.data_dir = data_dir
        self.set_name = set_name
        self.classes = classes
        combined_file = open(os.path.join('./data', set_name + '.txt')).readlines()
        self.combined_names = [l.strip().split(' ')[0:-1] for l in combined_file]
        self.combinednet_combined_classes = [l.strip().split(' ')[-1] for l in combined_file]

        # # eeg list
        # eeg_file = open(os.path.join('./data', data_name[0] + '_' + set_name + '.txt')).readlines()
        # self.eeg_names = [l.strip().split(' ')[0] for l in eeg_file]
        # self.eegnet_eeg_classes = [l.strip().split(' ')[1] for l in eeg_file]
        # # rgb list
        # rgb_file = open(os.path.join('./data', data_name[1] + '_' + set_name + '.txt')).readlines()
        # self.rgb_names = [l.strip().split(' ')[0] for l in rgb_file]
        # self.rgbnet_rgb_classes = [l.strip().split(' ')[1] for l in rgb_file]
        # # depth list
        # depth_file = open(os.path.join('./data', data_name[2] + '_' + set_name + '.txt')).readlines()
        # self.depth_names = [l.strip().split(' ')[0] for l in depth_file]
        # self.depthnet_depth_classes = [l.strip().split(' ')[1] for l in depth_file]
        # # skeleton list
        # skeleton_file = open(os.path.join('./data', data_name[3] + '_' + set_name + '.txt')).readlines()
        # self.skeleton_names = [l.strip().split(' ')[0] for l in skeleton_file]
        # self.skeletonnet_skeleton_classes = [l.strip().split(' ')[1] for l in skeleton_file]

        super(CombinedDataGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.combined_names)

    def load_combinednet_classes(self, combined_index):
        return self.combinednet_combined_classes[combined_index]

    def num_combinednet_classes(self):
        return len(self.classes)

    def load_combined(self, combined_index):
        # eeg inputs data
        eeg_path = os.path.join(self.data_dir[0], self.combined_names[combined_index][0])

        # eeg txt util (no head)
        x = pd.read_csv(eeg_path, header=None).T
        eeg_datas = x[1:].T.values

        # eeg txt util (head)
        # eeg_file = open(eeg_path)
        # eeg_datas = []
        # for i, line in enumerate(eeg_file.readlines()):
        #     if i != 0:
        #         a = line.strip().split(',')[1:]
        #         eeg_datas.append([float(x.strip()) for x in a])
        # eeg_datas = np.array(eeg_datas, dtype=float)

        # rgb inputs data
        rgb_path = os.path.join(self.data_dir[1],
                                self.combined_names[combined_index][1][:-22] + self.combined_names[combined_index][1][
                                    -5])
        txt_path = os.path.join(self.data_dir[2], self.combined_names[combined_index][1])
        rgb_datas = [rgb_path, txt_path]

        # depth inputs data
        depth_path = os.path.join(self.data_dir[3],
                                  self.combined_names[combined_index][2][:-22] + self.combined_names[combined_index][2][
                                      -5])
        txt_path = os.path.join(self.data_dir[4], self.combined_names[combined_index][2])
        depth_datas = [depth_path, txt_path]

        # skeleton inputs data
        skeleton_path = os.path.join(self.data_dir[5], self.combined_names[combined_index][3])
        path_list = os.listdir(skeleton_path)
        path_list.sort()
        skeleton_path = os.path.join(skeleton_path, path_list[0])
        x = pd.read_csv(skeleton_path, sep=' ', header=None).T
        x = x[:-1].T
        skeleton_data = x.values

        return [eeg_datas, rgb_datas, depth_datas, skeleton_data]
