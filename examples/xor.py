"""TODO: Docstring."""

from __future__ import print_function

import argparse
import keras

import gandlf
import numpy as np

# For repeatability.
np.random.seed(1667)


def get_training_data(num_samples):
    """Generates some training data."""

    x = np.random.randint(0, 2, size=(num_samples, 2))

    y = np.logical_xor(x[:, 0], x[:, 1])
    y = np.cast['int32'](y)
    y = np.eye(2)[y]  # Makes it a choice between two items.

    x = np.cast['float32'](x) * 2 - 1  # Scales to [-1, 1].
    x += np.random.normal(scale=1.)  # Adds some random noise.

    return x, y


def build_generator(latent_size):
    latent_layer = keras.layers.Input(shape=(latent_size,), name='latent')
    class_input = keras.layers.Input(shape=(2,), name='class')

    input_layer = keras.layers.merge([latent_layer, class_input],
                                     mode='concat')
    hidden_layer = keras.layers.Dense(16, activation='tanh')(input_layer)
    output_layer = keras.layers.Dense(2)(hidden_layer)

    return keras.models.Model([latent_layer, class_input],
                              output_layer, name='generator')


def build_discriminator():
    input_layer = keras.layers.Input(shape=(2,), name='data_input')

    normalized = keras.layers.BatchNormalization()(input_layer)
    hidden_layer = keras.layers.Dense(16, activation='tanh')(normalized)

    real_fake_pred = keras.layers.Dense(1, activation='sigmoid')(hidden_layer)
    class_pred = keras.layers.Dense(2, activation='sigmoid',
                                    name='class')(hidden_layer)

    return keras.models.Model(input_layer, [real_fake_pred, class_pred],
                              name='discriminator')


if __name__ == '__main__':
    latent_size = 10
    nb_epoch = 100
    num_samples = 5

    model = gandlf.Model(build_generator(latent_size),
                         build_discriminator())
    model.compile('adam', metrics=['accuracy'])

    x, y = get_training_data(1000)

    model.fit({'latent': 'normal', 'data_input': x, 'class': y}, {'class': y},
              nb_epoch=nb_epoch, batch_size=100)

    # Samples from the model.
    p = model.sample({
        'latent': 'normal',
        'class': y[:num_samples],
    }, num_samples=num_samples)
    print(p)
    print(x[:num_samples])
    print(y[:num_samples])
