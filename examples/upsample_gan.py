#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Using generative adversarial networks to super-resolve pictures.
"""

from __future__ import print_function

import argparse
import os

import gandlf
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import keras
from keras.datasets import mnist
from keras.datasets import cifar10

# Requires Pillow: pip install Pillow
from PIL import Image

# For consistency.
keras.backend.set_image_dim_ordering('tf')


def build_generator(small_shape, up_factor):
    """Builds the generator model."""

    num_channels = small_shape[-1]

    # Model inputs.
    latent = keras.layers.Input(shape=small_shape, name='latent')
    low_dim = keras.layers.Input(shape=small_shape, name='low_dim_gen')

    # Merge latent with base image.
    hidden = keras.layers.merge([low_dim, latent], mode='concat')
    hidden = keras.layers.UpSampling2D((up_factor, up_factor))(hidden)
    hidden = keras.layers.Convolution2D(64, 5, 5, border_mode='same')(hidden)
    hidden = keras.layers.Activation('tanh')(hidden)
    hidden = keras.layers.Convolution2D(64, 5, 5, border_mode='same')(hidden)
    hidden = keras.layers.Activation('tanh')(hidden)

    # activation on last output.
    hidden = keras.layers.Convolution2D(128, 1, 1)(hidden)
    hidden = keras.layers.Activation('tanh')(hidden)
    hidden = keras.layers.Convolution2D(num_channels, 1, 1)(hidden)
    output = keras.layers.Activation('sigmoid')(hidden)

    return keras.models.Model(input=[latent, low_dim],
                              output=output,
                              name='generator')


def build_discriminator(small_shape, up_factor):
    """Builds the discriminator model."""

    image_shape = (small_shape[0] * up_factor,
                   small_shape[1] * up_factor,
                   small_shape[2])

    image = keras.layers.Input(shape=image_shape, name='real_image')
    low_dim = keras.layers.Input(shape=small_shape, name='low_dim_dis')
    low_exp = keras.layers.UpSampling2D((up_factor, up_factor))(low_dim)

    # Merge generated image with real image.
    hidden = keras.layers.merge([image, low_exp],
                                mode='concat', concat_axis=-1)

    hidden = keras.layers.Convolution2D(64, 5, 5)(hidden)
    hidden = keras.layers.MaxPooling2D((2, 2))(hidden)
    hidden = keras.layers.LeakyReLU()(hidden)

    hidden = keras.layers.Convolution2D(64, 5, 5)(hidden)
    hidden = keras.layers.MaxPooling2D((2, 2))(hidden)
    hidden = keras.layers.LeakyReLU()(hidden)

    hidden = keras.layers.Convolution2D(128, 1, 1)(hidden)
    hidden = keras.layers.Activation('tanh')(hidden)

    # Pooling for classification layer.
    hidden = keras.layers.GlobalAveragePooling2D()(hidden)
    fake = keras.layers.Dense(1, W_regularizer='l2',
                              activation='sigmoid', name='src')(hidden)

    return keras.models.Model(input=[image, low_dim],
                              output=fake,
                              name='discriminator')


def mean_bins(X_input):
    return (X_input[:, ::2, ::2] + X_input[:, 1::2, ::2] +
            X_input[:, ::2, 1::2] + X_input[:, 1::2, 1::2]) / 4.


def generate_training_data(data='mnist'):
    if data == 'mnist':
        (X_train, _), (_, _) = mnist.load_data()
        X_train = np.expand_dims(X_train, -1) / 255.
    elif data == 'cifar':
        (X_train, _), (_, _) = cifar10.load_data()
        X_train = X_train / 255.
    else:
        raise ValueError('data should be "mnist" or "cifar", got '
                         '"%s".' % data)

    # Downsamples by averaging adjacent pixels.
    X_low_dim = mean_bins(X_train)

    return X_low_dim, X_train


def upsample(X_input, weights_path, up_factor):
    """Uses the trained model to upsample an image."""

    generator = build_generator(X_input.shape[1:], up_factor)
    discriminator = build_discriminator(X_input.shape[1:], up_factor)
    model = gandlf.Model(generator=generator, discriminator=discriminator)
    model.generator.load_weights(weights_path)
    X_output = model.sample(['normal', X_input])
    return X_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generative adversarial network self-portrait script.')

    parser.add_argument(
        '--weights_path',
        metavar='PATH',
        default='/tmp/super_resolve.h5',
        type=str,
        help='where to save the model weights '
             '(default is /tmp/super_resolve.h5)'
    )
    parser.add_argument(
        '--generator_path',
        metavar='PATH',
        default='/tmp/generator_resolve.h5',
        type=str,
        help='where to store the generator weights '
             '(default is /tmp/generator_resolve.h5)'
    )
    parser.add_argument(
        '--reset_weights',
        default=False,
        action='store_true',
        help='if set, remove old weights'
    )
    parser.add_argument(
        '--dataset',
        metavar='DATASET',
        default='mnist',
        choices=['mnist', 'cifar'],
        help='the dataset to use ("mnist" or "cifar", default "mnist")'
    )
    parser.add_argument(
        '--nb_epoch',
        metavar='N',
        default=10,
        type=int,
        help='number of training epochs'
    )
    args = parser.parse_args()

    # Generates the training data.
    X_low_dim, X_data = generate_training_data(args.dataset)
    up_factor = X_data.shape[1] / X_low_dim.shape[1]

    # Builds the model.
    generator = build_generator(X_low_dim.shape[1:], up_factor)
    discriminator = build_discriminator(X_low_dim.shape[1:], up_factor)
    model = gandlf.Model(generator=generator, discriminator=discriminator)
    optimizer = keras.optimizers.Adam(1e-4)
    loss = {'dis': 'binary_crossentropy',
            'gen': 'binary_crossentropy'}
    model.compile(loss=loss, optimizer=optimizer)

    # Loads existing weights, if they exist.
    if os.path.exists(args.weights_path) and not args.reset_weights:
        model.load_weights(args.weights_path)

    # Fit the training data.
    model.fit(['normal', X_low_dim, X_data, X_low_dim],
              {'gen_real': '1', 'fake': '0'},
              batch_size=100, nb_epoch=args.nb_epoch)

    # Save the model weights.
    model.save_weights(args.weights_path)
    print('Saved weights to "%s"' % args.weights_path)

    # Save the generator weights.
    model.generator.save_weights(args.generator_path)
    print('Saved generator weights to "%s"' % args.generator_path)

    # Samples from the model.
    X_inputs = [X_low_dim[:3], X_data[:3]]
    for _ in range(4):
        X_inputs.append(upsample(X_inputs[-1], args.generator_path, up_factor))
        print('New shape:', X_inputs[-1].shape)

    for j in range(3):
        plt.figure()

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(-np.squeeze(X_inputs[i][j]), cmap='gray')
            plt.axis('off')

    plt.show()
