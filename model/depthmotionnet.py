import keras
from model import attention_new


def ShareNet():
    base_model = keras.applications.MobileNetV2(include_top=False, pooling='avg')
    return keras.models.Model(inputs=base_model.input, outputs=base_model.output)


def DepthMotionNet(
        inputs=None,
        image_height=224,
        image_weight=224,
        image_channel=3,
        sample_num=1000
):
    """construct a DepthMotionNet model
        Returns
        A keras.models.Model which takes an image as input and outputs the classification on the image.
        The outputs is defined as follow:
            depthmotionnet_classification
    """
    if inputs is None:
        inputs = keras.layers.Input(shape=(sample_num, image_height, image_weight, image_channel))
    sharenet = ShareNet()
    x = keras.layers.TimeDistributed(sharenet)(inputs)
    x = keras.layers.Bidirectional(keras.layers.CuDNNGRU(units=16, return_sequences=True))(x)
    # x = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(units=16, return_sequences=True))(x)
    x = attention_new.AttentionLayer(attention_dim=sample_num)(x)
    x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model
