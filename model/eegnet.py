import keras
from model import attention_new


def eegnet(
        inputs=None,
        sample_num=1000,
        eeg_channel=8
):
    if inputs is None:
        inputs = keras.layers.Input(shape=(sample_num, eeg_channel))
    x = keras.layers.Bidirectional(keras.layers.CuDNNGRU(units=16, return_sequences=True))(inputs)
    # x = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(units=16, return_sequences=True))(x)
    x = attention_new.AttentionLayer(attention_dim=sample_num)(x)
    x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    # x = keras.layers.Dense(num_eegnet_classes, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)

    return model
