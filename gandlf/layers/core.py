import keras
import keras.backend as K


class PermanentDropout(keras.layers.Dropout):
    """Applies dropout to the input that isn't turned off during testing.

    This is one possible improvement for your generator models.

    Args:
        p: float between 0 and 1. Fraction of the input units to drop.
    """

    def call(self, x, mask=None):
        if 0. < self.p < 1.:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.p, noise_shape)
        return x
