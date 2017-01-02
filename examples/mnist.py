#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more detals.

This example is based on the Keras example, which uses the same generator and
discriminator model. It can be found at:

https://github.com/fchollet/keras/blob/master/examples/mnist_acgan.py

Consult https://github.com/lukedeo/keras-acgan for more information about the
model.

To show all command line options:

    ./examples/xor.py --help

The model runs in unsupervised mode by default. To run as an ACGAN:

    ./examples/xor.py --supervised
"""

from __future__ import print_function

import argparse

import keras
from keras.datasets import mnist

import gandlf
import numpy as np


# For repeatability.
np.random.seed(1337)

# To make the images work correctly.
keras.backend.set_image_dim_ordering('th')


def build_generator(latent_size):
    """Builds the generator model."""

    cnn = keras.models.Sequential()

    cnn.add(keras.layers.Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(keras.layers.Dense(128 * 7 * 7, activation='relu'))
    cnn.add(keras.layers.Reshape((128, 7, 7)))

    cnn.add(keras.layers.UpSampling2D(size=(2, 2)))
    cnn.add(keras.layers.Convolution2D(256, 5, 5, border_mode='same',
                                       activation='relu', init='glorot_normal'))

    cnn.add(keras.layers.UpSampling2D(size=(2, 2)))
    cnn.add(keras.layers.Convolution2D(128, 5, 5, border_mode='same',
                                       activation='relu', init='glorot_normal'))

    cnn.add(keras.layers.Convolution2D(1, 2, 2, border_mode='same',
                                       activation='tanh', init='glorot_normal'))

    latent = keras.layers.Input(shape=(latent_size,), name='latent')

    image_class = keras.layers.Input(shape=(1,), dtype='int32', name='class')

    embedded = keras.layers.Embedding(10, latent_size,
                                      init='glorot_normal')(image_class)
    cls = keras.layers.Flatten()(embedded)
    h = keras.layers.merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return keras.models.Model(input=[latent, image_class], output=fake_image,
                              name='generator')


def build_discriminator():
    """Builds the discriminator model."""

    cnn = keras.models.Sequential()

    cnn.add(keras.layers.Convolution2D(32, 3, 3, border_mode='same',
                                       subsample=(2, 2),
                                       input_shape=(1, 28, 28)))
    cnn.add(keras.layers.LeakyReLU())
    cnn.add(keras.layers.Dropout(0.3))

    cnn.add(keras.layers.Convolution2D(64, 3, 3, border_mode='same',
                                       subsample=(1, 1)))
    cnn.add(keras.layers.LeakyReLU())
    cnn.add(keras.layers.Dropout(0.3))

    cnn.add(keras.layers.Convolution2D(128, 3, 3, border_mode='same',
                                       subsample=(2, 2)))
    cnn.add(keras.layers.LeakyReLU())
    cnn.add(keras.layers.Dropout(0.3))

    cnn.add(keras.layers.Convolution2D(256, 3, 3, border_mode='same',
                                       subsample=(1, 1)))
    cnn.add(keras.layers.LeakyReLU())
    cnn.add(keras.layers.Dropout(0.3))

    cnn.add(keras.layers.Flatten())

    image = keras.layers.Input(shape=(1, 28, 28), name='real')

    features = cnn(image)

    fake = keras.layers.Dense(1, activation='sigmoid',
                              name='generation')(features)
    aux = keras.layers.Dense(10, activation='softmax',
                             name='class')(features)

    return keras.models.Model(input=image, output=[fake, aux],
                              name='discriminator')


def get_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=1)

    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Auxiliary Classifier GAN for MNIST digits.')

    training_params = parser.add_argument_group('training params')
    training_params.add_argument('--nb_epoch', type=int, default=50,
                                 metavar='INT',
                                 help='number of epochs to train')
    training_params.add_argument('--nb_batch', type=int, default=32,
                                 metavar='INT',
                                 help='number of samples per batch')
    training_params.add_argument('--supervised', default=False,
                                 action='store_true',
                                 help='if set, train as an ACGAN')

    model_params = parser.add_argument_group('model params')
    model_params.add_argument('--nb_latent', type=int, default=100,
                              metavar='INT',
                              help='dimensions in the latent vector')
    model_params.add_argument('--save_path', type=str, metavar='STR',
                              default='/tmp/mnist_gan.keras_model',
                              help='Where to save the model after training')

    optimizer_params = parser.add_argument_group('optimizer params')
    optimizer_params.add_argument('--lr', type=float, default=0.0002,
                                  metavar='FLOAT',
                                  help='learning rate for Adam optimizer')
    optimizer_params.add_argument('--beta', type=float, default=0.5,
                                  metavar='FLOAT',
                                  help='beta 1 for Adam optimizer')

    args = parser.parse_args()

    # Builds the model itself.
    optimizer = keras.optimizers.Adam(lr=args.lr, beta_1=args.beta)
    model = gandlf.Model(build_generator(args.nb_latent),
                         build_discriminator())
    model.compile(optimizer=optimizer, loss={
        'class': 'categorical_crossentropy',
        'real': gandlf.losses.negative_binary_crossentropy,
        'fake': 'binary_crossentropy',
    }, metrics=['accuracy'])

    # Gets training and testing data.
    (X_train, y_train), (_, _) = get_mnist_data()
    y_train_ohe = np.eye(10)[np.squeeze(y_train)]

    model.fit({'latent': 'normal', 'class': y_train, 'real': X_train},
              {'fake': 'ones', 'real': 'zeros', 'class': y_train_ohe},
              train_auxiliary=args.supervised, nb_epoch=args.nb_epoch,
              batch_size=args.nb_batch)

    model.save(args.save_path)
    print('Saved model:', args.save_path)
