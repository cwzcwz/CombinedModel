import numpy as np
import random
import keras
import pandas as pd
import os


class Generator(keras.utils.Sequence):
    """ Abstract generator class.
    """

    def __init__(
            self,
            batch_size=1,
            group_method='random',  # one of 'none', 'random'
            shuffle_groups=True,
            sample_num=1000,
            image_height=224,
            image_weight=224
    ):
        """ Initialize Generator object.

        Args
            batch_size             : The size of the batches to generate.
            shuffle_groups         : If True, shuffles the groups each epoch.
            sample_num             : sample rate
        """

        self.sample_num = int(sample_num)
        self.image_height = image_height
        self.image_weight = image_weight
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        # Define groups
        self.group_combineds()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def size(self):
        """ Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def load_combinednet_classes(self, combined_index):
        """Load combinednet_classes for an combined_index"""
        raise NotImplementedError('load_combinednet_classes methood not implemented')

    def num_combinednet_classes(self):
        raise NotImplementedError('num_combinednet_classes method not implemented')

    def load_combined(self, combined_index):
        """ （combined）Load an combined at the combined_index.
        """
        raise NotImplementedError('load_combined method not implemented')

    def load_combined_group(self, group):
        """ Load combineds for all combineds in a group."""
        return np.array([self.load_combined(combined_index) for combined_index in group])

    def group_combineds(self):
        """ Order the combineds according to self.order and makes groups of self.batch_size."""
        # determine the order of the combineds
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def compute_inputs(self, combined_group):
        """ Compute inputs for the network using an combined_group."""
        # eeg inputs
        # sample_data_group = []
        eeg_inputs = []
        rgb_inputs = []
        depth_inputs = []
        skeleton_inputs = []
        skeleton_motion_inputs = []
        for k in combined_group:
            sample_data = []
            total_samples = len(k[0])
            for i in np.linspace(0, total_samples, self.sample_num + 2, dtype=np.int)[1:self.sample_num + 1]:
                clip_start = i - int(6 / 2)
                clip_end = i + int(6 / 2)
                if clip_start < 0:
                    clip_end = clip_end - clip_start
                    clip_start = 0
                if clip_end > total_samples:
                    clip_start = clip_start - (clip_end - total_samples)
                    clip_end = total_samples
                clip = k[0][clip_start:clip_end]
                clip = np.mean(clip, axis=0)
                # frame_data_local = 0
                # for i1 in range(int(i), int(i) + 100):
                #     if i1 >= total_samples:
                #         break
                #     frame_data_local += k[0][int(i1)]
                # frame_data = frame_data_local / 100
                sample_data.append(clip)
            eeg_inputs.append(sample_data)

            # rgb inputs
            sample_data = []
            x = pd.read_csv(k[1][1], sep=' ', header=None)
            x = x.values
            total_sample = x.shape[0]
            for j in np.linspace(0, total_sample, self.sample_num + 2, dtype=np.int)[1:self.sample_num + 1]:
                image_name = x[int(j)][0]
                image_path = os.path.join(str(k[1][0]), str(image_name))
                png_image = keras.preprocessing.image.load_img(image_path,
                                                               target_size=(self.image_height, self.image_weight))
                y = keras.preprocessing.image.img_to_array(png_image)
                sample_data.append(y)
            rgb_inputs.append(sample_data)

            # depth inputs
            sample_data = []
            x = pd.read_csv(k[2][1], sep=' ', header=None)
            x = x.values
            total_sample = x.shape[0]
            for j in np.linspace(0, total_sample, self.sample_num + 2, dtype=np.int)[1:self.sample_num + 1]:
                image_name = x[int(j)][0]
                image_path = os.path.join(str(k[2][0]), str(image_name))
                png_image = keras.preprocessing.image.load_img(image_path,
                                                               target_size=(self.image_height, self.image_weight))
                y = keras.preprocessing.image.img_to_array(png_image)
                sample_data.append(y)
            depth_inputs.append(sample_data)

            # skeleton inputs
            for i in k[3]:
                sample_data = []
                i = i.reshape([-1, 25, 3])
                total_sample = i.shape[0]
                for j in np.linspace(0, total_sample, self.sample_num + 2, dtype=np.int)[1:self.sample_num + 1]:
                    sample_data.append(i[int(j)])
                p = np.array(sample_data)
                p_diff = p[1:, :, :] - p[:-1, :, :]
                p_diff = np.concatenate((p_diff, np.expand_dims(p_diff[-1, :, :], axis=0)))
            skeleton_inputs.append(p)
            skeleton_motion_inputs.append(p_diff)
        return [np.array(eeg_inputs), np.array(rgb_inputs), np.array(depth_inputs), np.array(skeleton_inputs),
                np.array(skeleton_motion_inputs)]

    def compute_combinednet_targets(self, group):
        combinednet_labels_batch = np.zeros((len(group), 1), dtype=keras.backend.floatx())
        for index, combined_index in enumerate(group):
            combinednet_labels_batch[index] = int(self.load_combinednet_classes(combined_index))
        return combinednet_labels_batch

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
            Returns:
                inputs:ndarray(batch_size,sample_num,electrode_num)
                targets:combinednet_labels_batch:ndarray(batch_size,num_combinednet_classes)
        """
        # load combineds and annotations
        combined_group = self.load_combined_group(group)

        # compute network inputs
        inputs = self.compute_inputs(combined_group)

        # compute network targets
        targets = self.compute_combinednet_targets(group)

        return inputs, targets

    def __len__(self):
        """Number of batches for generator.
        """
        return len(self.groups)

    def __getitem__(self, index):
        """Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)
        return inputs, targets
