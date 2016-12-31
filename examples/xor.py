"""Illustrates the basic usage of this package."""

import keras

import gandlf
import numpy as np


def get_training_data(num_samples):
    """Generates some random training data."""

    x = np.random.randint(0, 2, size=(num_samples, 2))
    y = np.logical_xor(x[:, 0], x[:, 1], dtype=np.float32)

    x = np.cast['float32'](x)
    x += np.random.normal(scale=0.1)

    return x, y


def build_generator(latent_size):
    input_layer = keras.layers.Input(shape=(latent_size,), name='latent_vec')
    hidden_layer = keras.layers.Dense(16, activation='relu')(input_layer)
    output_layer = keras.layers.Dense(2, activation='tanh')(hidden_layer)

    return keras.models.Model(input_layer, output_layer, name='generator')


def build_discriminator():
    input_layer = keras.layers.Input(shape=(2,), name='data_input')
    hidden_layer = keras.layers.Dense(16, activation='tanh')(input_layer)
    output_layer = keras.layers.Dense(1, activation='sigmoid')(hidden_layer)

    return keras.models.Model(input_layer, output_layer, name='discriminator')


if __name__ == '__main__':
    latent_size = 10

    generator = build_generator(latent_size)
    discriminator = build_discriminator()

    x, y = get_training_data(1000)

    model = gandlf.Model(generator, discriminator)
    model.compile(keras.optimizers.SGD(lr=0.00001), metrics=['accuracy'])

    model.fit({
        'latent_vec': 'normal',
        'data_input': x,
    }, nb_epoch=20)

    p = model.sample({
        'latent_vec': 'normal'
    }, num_samples=10)
    print(p)
