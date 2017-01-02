#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hello gandlf!

This example trains the GAN to approximate four normal distributions centered
around (-1, -1), (-1, 1), (1, -1) and (1, 1). It can be trained as a vanilla
GAN or as an Auxiliary Classifier GAN, where it learns to classify the
distributions according to the XOR.

The model doesn't work super well (it is hard to get the generator to equally
distribute among the four distributions) but it is a good proof-of-concept
that can be run quickly on a single CPU.

To show all command line options:

    ./examples/xor.py --help

The model runs in unsupervised mode by default. To run as an ACGAN:

    ./examples/xor.py --supervised
"""

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

    x = np.cast['float32'](x) * 2 - 1  # Scales to [-1, 1].
    x += np.random.normal(scale=0.3, size=x.shape)  # Adds some random noise.

    y_ohe = np.eye(2)[y]
    y = np.expand_dims(y, -1)

    return x, y, y_ohe


def build_generator(latent_size):
    """Builds a simple two-layer generator network."""

    latent_layer = keras.layers.Input(shape=(latent_size,), name='latent')
    class_input = keras.layers.Input(shape=(1,), name='class')
    embedded = keras.layers.Embedding(2, latent_size,
                                      init='glorot_normal')(class_input)
    flat_embedded = keras.layers.Flatten()(embedded)

    input_layer = keras.layers.merge([latent_layer, flat_embedded],
                                     mode='mul')
    hidden_layer = keras.layers.Dense(16, activation='tanh')(input_layer)
    output_layer = keras.layers.Dense(2)(hidden_layer)

    return keras.models.Model([latent_layer, class_input],
                              output_layer, name='generator')


def build_discriminator():
    """Builds a simple two-layer discriminator network."""

    input_layer = keras.layers.Input(shape=(2,), name='real')

    normalized = keras.layers.BatchNormalization()(input_layer)
    hidden_layer = keras.layers.Dense(16, activation='tanh')(normalized)

    real_fake_pred = keras.layers.Dense(1, activation='sigmoid',
                                        name='real_fake')(hidden_layer)
    class_pred = keras.layers.Dense(2, activation='sigmoid',
                                    name='class')(hidden_layer)

    # The first output of this model (real_fake_pred) is treated as
    # the "real / fake" predictor.
    return keras.models.Model(input_layer, [real_fake_pred, class_pred],
                              name='discriminator')


def train_model(args, x, y, y_ohe):
    """Returns a trained model."""

    model = gandlf.Model(build_generator(args.nb_latent),
                         build_discriminator())
    model.compile(optimizer='adam', loss={
        'class': 'categorical_crossentropy',
        'fake': gandlf.losses.negative_binary_crossentropy,
        'real': 'binary_crossentropy',
    }, metrics=['accuracy'])

    model.fit({'latent': 'normal', 'real': x, 'class': y},
              {'fake': 'ones', 'real': 'zeros', 'class': y_ohe},
              train_auxiliary=args.supervised, nb_epoch=args.nb_epoch,
              batch_size=args.nb_batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Basic XOR example using a GAN.')

    training_params = parser.add_argument_group('training params')
    training_params.add_argument('--nb_epoch', type=int, default=10,
                                 metavar='INT',
                                 help='number of training epochs')
    training_params.add_argument('--nb_batch', type=int, default=100,
                                 metavar='INT',
                                 help='number of samples per batch')
    training_params.add_argument('--supervised', default=False,
                                 action='store_true',
                                 help='if set, train as an ACGAN')

    model_params = parser.add_argument_group('model params')
    model_params.add_argument('--nb_latent', type=int, default=10,
                              metavar='INT',
                              help='dimensions in the latent vector')
    model_params.add_argument('--nb_samples', type=int, default=10000,
                              metavar='INT',
                              help='total number of training samples')

    args = parser.parse_args()

    # Get the training data.
    x, y, y_ohe = get_training_data(args.nb_samples)

    # Trains the model.
    model = train_model(args, x, y, y_ohe)

    ##### Evaluates the trained model and prints a bunch of stuff. #####

    print('\n:: Input Data ::')
    print(x[:10])

    print('\n:: Target Data ::')
    print(y[:10])

    if args.supervised:
        print('\n:: Predictions for Real Data ::')
        print(np.argmax(model.predict({'real': x[:10]})[1], -1)
              .reshape((-1, 1)))

    print('\n:: Generated Input Data (Knowing Target Data) ::')
    p = model.sample({'latent': 'normal', 'class': y[:10]})
    print(p)

    if args.supervised:
        print('\n:: Predictions for Generated Data ::')
        print(np.argmax(model.predict({'real': p})[1], -1).reshape((-1, 1)))
