from model import eegnet
from model import rgbmotionnet
from model import depthmotionnet
from model import skeletonmotionnet
import keras


def CombinedModel(
        class_num=1,
        sample_num=1000,
        image_height=224,
        image_weight=224,
        image_channel=3,
        skeleton_joints=25,
        skeleton_channel=3,
        eeg_channel=8
):
    inputs_eeg = keras.layers.Input(name='eeg_inputs', shape=(sample_num, eeg_channel))
    inputs_rgb = keras.layers.Input(name='rgb_inputs', shape=(sample_num, image_height, image_weight, image_channel))
    inputs_depth = keras.layers.Input(name='depth_inputs',
                                      shape=(sample_num, image_height, image_weight, image_channel))
    inputs_skeleton = keras.layers.Input(name='skeleton_inputs', shape=(sample_num, skeleton_joints, skeleton_channel))
    inputs_skeleton_motion = keras.layers.Input(name='skeleton_motion_inputs',
                                                shape=(sample_num, skeleton_joints, skeleton_channel))
    # eeg model
    eeg_model = eegnet.eegnet(inputs=inputs_eeg)
    # rgb model
    rgb_model = rgbmotionnet.RGBMotionNet(inputs=inputs_rgb)
    # depth model
    depth_model = depthmotionnet.DepthMotionNet(inputs=inputs_depth)
    # skeleton model
    skeleton_model = skeletonmotionnet.SkeletonMotionNet(inputs=inputs_skeleton, inputs_motion=inputs_skeleton_motion)

    x = keras.layers.Concatenate()(

        [eeg_model.output, rgb_model.output, depth_model.output, skeleton_model.output])

    x = keras.layers.Dense(32)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(class_num, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[inputs_eeg, inputs_rgb, inputs_depth, inputs_skeleton, inputs_skeleton_motion],
                               outputs=x)
    return model
