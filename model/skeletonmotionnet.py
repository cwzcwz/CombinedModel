import keras


def SkeletonMotionNet(
        inputs=None,
        inputs_motion=None,
        sample_frames=1000,
        joints=25,
        channel=3
):
    if inputs is None:
        inputs = keras.layers.Input(shape=(sample_frames, joints, channel))
    if inputs_motion is None:
        inputs_motion = keras.layers.Input(shape=(sample_frames, joints, channel))

    # branch1
    x = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(inputs)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(filters=16, kernel_size=(3, 1), padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Permute((1, 3, 2))(x)

    x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.5)(x)

    # branch2
    x_2 = keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(inputs_motion)
    # x_2 = keras.layers.BatchNormalization()(x_2)
    x_2 = keras.layers.ReLU()(x_2)

    x_2 = keras.layers.Conv2D(filters=16, kernel_size=(3, 1), padding='same')(x_2)
    # x_2 = keras.layers.BatchNormalization()(x_2)

    x_2 = keras.layers.Permute((1, 3, 2))(x_2)

    x_2 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same')(x_2)
    # x_2 = keras.layers.BatchNormalization()(x_2)

    x_2 = keras.layers.MaxPool2D(pool_size=(2, 2))(x_2)
    x_2 = keras.layers.Dropout(0.5)(x_2)

    # concat
    x = keras.layers.concatenate([x, x_2], axis=-1)

    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Dense(64, activation='relu')(x)

    model = keras.models.Model(inputs=[inputs, inputs_motion], outputs=x)
    return model
