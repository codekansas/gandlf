import warnings

import keras.backend as K


def negative_binary_crossentropy(y_true, y_pred):
    """Instead of minimizing log(1-D), maximize log(D).

    Note that when using this loss function, you should not change the target.
    For example, if you want G -> 0 and D -> 1, then you should replace your
    binary_crossentropy loss with negative_binary_crossentropy loss without
    changing to G -> 1.
    """

    return -K.mean(K.binary_crossentropy(y_pred, 1 - y_true), axis=-1)


def maximize(_, y_pred):
    """Maximizes y_pred, regardless of y_true."""

    return -K.mean(y_pred)


def minimize(_, y_pred):
    """Minimizes y_pred, regardless of y_true."""

    return K.mean(y_pred)


def rbf_moment_matching(y_true, y_pred, sigmas=[2, 5, 10, 20, 40, 80]):
    """Generative moment matching loss with RBF kernel.

    Reference: https://arxiv.org/abs/1502.02761
    """

    warnings.warn('Moment matching loss is still in development.')

    if len(K.int_shape(y_pred)) != 2 or len(K.int_shape(y_true)) != 2:
        raise ValueError('RBF Moment Matching function currently only works '
                         'for outputs with shape (batch_size, num_features).'
                         'Got y_true="%s" and y_pred="%s".' %
                         (str(K.int_shape(y_pred)), str(K.int_shape(y_true))))

    sigmas = list(sigmas) if isinstance(sigmas, (list, tuple)) else [sigmas]

    x = K.concatenate([y_pred, y_true], 0)

    # Performs dot product between all combinations of rows in X.
    xx = K.dot(x, K.transpose(x))  # (batch_size, batch_size)

    # Performs dot product of all rows with themselves.
    x2 = K.sum(x * x, 1, keepdims=True)  # (batch_size, None)

    # Gets exponent entries of the RBF kernel (without sigmas).
    exponent = xx - 0.5 * x2 - 0.5 * K.transpose(x2)

    # Applies all the sigmas.
    total_loss = None
    for sigma in sigmas:
        kernel_val = K.exp(exponent / sigma)
        loss = K.sum(kernel_val)
        total_loss = loss if total_loss is None else loss + total_loss

    return total_loss

negative_xent = negative_binary_crossentropy
gmmn = GMMN = rbf_moment_matching
