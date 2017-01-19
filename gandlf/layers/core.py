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
    layer by using a Merge layer. The layer only works for inputs with shape
    (batch_size, num_features). Inputs with more dimensions can be flattened.

    In order to make this layer linear time with respect to the batch size,
    instead of doing a pairwise comparison between each pair of samples in
    the batch, for each sample a random sample is uniformly selected with
    which to do pairwise comparison.

    Args:
        similarity: str, the similarity type. See gandlf.similarities for a
            possible types. Alternatively, it can be a function which takes
            two tensors as inputs and returns their similarity. A list or
            tuple of similarities will apply all the similarities.
        n: int or list of ints (one for each similarity), number of times to
            repeat each similarity, using a different sample to calculate the
            other similarity.

    Reference: "Improved Techniques for Training GANs"
        https://arxiv.org/abs/1606.03498
    """

    def __init__(self, similarity='exp_l1', n=1, **kwargs):
        if not isinstance(similarity, (list, tuple)):
            similarity = [similarity]
        if not isinstance(n, (list, tuple)):
            n = [n for _ in similarity]

        self.similarities = [similarities.get(s) for s in similarity]
        self.n = n

        super(BatchSimilarity, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('The input to a BatchSimilarity layer must be '
                             '2D. Got %d dims.' % len(input_shape))

    def call(self, x, mask=None):
        sims = []
        for n, sim in zip(self.n, self.similarities):
            for _ in range(n):
                batch_size = K.shape(x)[0]
                idx = K.random_uniform((batch_size,), low=0, high=batch_size,
                                       dtype='int32')
                x_shuffled = K.gather(x, idx)
                pair_sim = sim(x, x_shuffled)
                for _ in range(K.ndim(x) - 1):
                    pair_sim = K.expand_dims(pair_sim, dim=1)
                sims.append(pair_sim)

        return K.concatenate(sims, axis=-1)

    def get_output_shape_for(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('The input to a BatchSimilarity layer must be '
                             '2D. Got %d dims.' % len(input_shape))
        output_shape = list(input_shape)
        output_shape[-1] = sum(self.n)
        return tuple(output_shape)

    def get_config(self):
        config = {'similarity': [s.__name__ for s in self.similarities]}
        base_config = super(BatchSimilarity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
