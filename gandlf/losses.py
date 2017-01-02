"""Loss functions that might work well for training GANs."""

import keras.backend as K


def negative_binary_crossentropy(y_true, y_pred):
    """Instead of minimizing log(1-D), maximize log(D).

    Note that when using this loss function, you should not change the target.
    For example, if you want G -> 0 and D -> 1, then you should replace your
    binary_crossentropy loss with negative_binary_crossentropy loss without
    changing to G -> 1.
    """

    return -K.mean(K.binary_crossentropy(y_pred, 1 - y_true), axis=-1)
