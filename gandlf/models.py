from __future__ import absolute_import

from keras import backend as K
import keras


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


class Model(keras.models.Model):

    def __init__(self, generator, discriminator, name=None):

        if not isinstance(generator, keras.models.Model) or generator.input:
            raise ValueError('"generator" must both be a Keras model with no '
                             'input tensors.')

        if not isinstance(discriminator, keras.models.Model):
            raise ValueError('"discriminator" must both be a Keras model.')

        if len(generator.output) != len(discriminator.input):
            raise ValueError('The discriminator should have one input per '
                             'output of the generator. Got %d discriminator '
                             'inputs and %d generator outputs.' %
                             (len(discriminator.input), len(generator.output)))

        self.generator = generator
        self.discriminator = discriminator

        # TODO: Need to pass the generator through the discriminator to get
        # its output tensors.

    def _make_sample_function(self):
        if not hasattr(self, 'sample_function'):
            self.sample_function = None

        if self.sample_function is None:
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs = [K.learning_phase()]
            else:
                inputs = []

            kwargs = getattr(self, '_function_kwargs', {})

            self.sample_function = K.function(inputs,
                                              self.generator.outputs,
                                              updates=self.state_updates,
                                              **kwargs)

    def sample(self, batch_size=32, verbose=0):
        """Samples from the generator.

        Args:
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        Returns:
            A Numpy array of generated samples.
        """

        if self.uses_learning_phase and not isinstance(K.learning_phase, int):
            ins = [0.]
        else:
            ins = []

        self._make_predict_function()
        f = self.predict_function

        return self._predict_loop(f, ins,
                                  batch_size=batch_size, verbose=verbose)
