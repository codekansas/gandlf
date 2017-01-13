import keras
import keras.backend as K
from keras.utils.generic_utils import get_from_module


def exp_l1(a, b):
    """Exponential of L1 similarity. Maximum is 1 (a == b), minimum is 0."""

    return K.exp(l1(a, b))


def exp_l2(a, b):
    """Exponential of L2 similarity. Maximum is 1 (a == b), minimum is 0."""

    return K.exp(l2(a, b))


def l1(a, b):
    """L1 similarity. Maximum is 0 (a == b), minimum is -inf."""

    return -K.sum(K.abs(a - b), axis=-1)


def l2(a, b):
    """L2 similarity. Maximum is 0 (a == b), minimum is -inf."""

    return -K.sum(K.square(a - b), axis=-1)


def cosine(a, b):
    """Cosine similarity. Maximum is 1 (a == b), minimum is -1 (a == -b)."""

    a = K.l2_normalize(a)
    b = K.l2_normalize(b)
    return 1 - K.mean(a * b, axis=-1)


def sigmoid(a, b):
    """Sigmoid similarity. Maximum is 1 (a == b), minimum is 0."""

    return K.sigmoid(K.sum(a * b, axis=-1))


def euclidean(a, b):
    """Euclidian similarity. Maximum is 1 (a == b), minimum is 0 (a == -b)."""

    x = K.sum(K.square(a - b), axis=-1)
    return 1. / (1. + x)


def geometric(a, b):
    """Geometric mean of sigmoid and euclidian similarity."""

    return sigmoid(a, b) * euclidean(a, b)


def arithmetic(a, b):
    """Arithmetic mean of sigmoid and euclidean similarity."""

    return (sigmoid(a, b) + euclidean(a, b)) * 0.5


rbf = RBF = exp_l2  # Radial basis function.
gesd = GESD = geometric
aest = AESD = arithmetic


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'similarity',
                           instantiate=False, kwargs=kwargs)
