from __future__ import absolute_import

import keras
import keras.backend as K

from .. import similarities


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


class BatchSimilarity(keras.layers.Layer):
    """Calculates intrabatch similarity, for minibatch discrimination.

    The minibatch similarities can be added as features for the existing
    layer by using a Merge layer. The layer outputs a Tensor with shape
    (batch_size, num_similarities) for 2D tensors, (batch_size, None,
    num_similarities) for 3D tensors, and so on.

    In order to make this layer linear time with respect to the batch size,
    instead of doing a pairwise comparison between each pair of samples in
    the batch size, for each sample a random sample is uniformly selected
    with which to do pairwise comparison.

    Args:
        similarity: str, the similarity type. See gandlf.similarities for a
            possible types. Alternatively, it can be a function which takes
            two tensors as inputs and returns their similarity. A list or
            tuple of similarities will apply all the similarities.

    Reference: "Improved Techniques for Training GANs"
        https://arxiv.org/abs/1606.03498
    """

    def __init__(self, similarity='exp_l1', **kwargs):
        if isinstance(similarity, (list, tuple)):
            self.similarities = [similarities.get(s) for s in similarity]
        else:
            self.similarities = [similarities.get(similarity)]
        super(BatchSimilarity, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError('The input to a BatchSimilarity layer must be '
                             'at least 2D. Got %d dims.' % len(input_shape))

    def call(self, x, mask=None):
        sims = []
        for sim in self.similarities:
            batch_size = K.shape(x)[0]
            idx = K.random_uniform((batch_size,), low=0, high=batch_size,
                                   dtype='int32')
            x_shuffled = K.gather(x, idx)
            sims.append(sim(x, x_shuffled))

        return K.concatenate(sims, axis=-1)

    def get_output_shape_for(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError('The input to a BatchSimilarity layer must be '
                             'at least 2D. Got %d dims.' % len(input_shape))
        output_shape = list(input_shape)
        output_shape[-1] = len(self.similarities)
        return tuple(output_shape)

    def get_config(self):
        config = {'similarity': [s.__name__ for s in self.similarities]}
        base_config = super(BatchSimilarity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
