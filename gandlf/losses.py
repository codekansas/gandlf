"""Loss functions that might work well for training GANs."""

import keras.backend as K


def negative_binary_crossentropy(y_true, y_pred):
    return -K.mean(K.binary_crossentropy(y_pred, 1 - y_true), axis=-1)
