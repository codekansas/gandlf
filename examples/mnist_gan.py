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

In addition to the ACGAN implementation from the Keras examples folder, there
is a lite version of the model, which is faster on CPUs.

To show all command line options:

    ./examples/xor.py --help

To train the lite version of the model:

    ./examples/xor.py --lite

Samples from the model can be plotted using Matplotlib:

    ./examples/xor.py --plot
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
    """Builds the big generator model."""

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
    image_class = keras.layers.Input(shape=(1,), dtype='int32',
                                     name='image_class')

    embed = keras.layers.Embedding(10, latent_size, init='glorot_normal')
    cls = keras.layers.Flatten()(embed(image_class))
    h = keras.layers.merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return keras.models.Model(input=[latent, image_class], output=fake_image,
                              name='generator')


def build_discriminator():
    """Builds the big discriminator model."""

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

    image = keras.layers.Input(shape=(1, 28, 28), name='real_data')

    features = cnn(image)

    fake = keras.layers.Dense(1, activation='sigmoid', name='src')(features)
    aux = keras.layers.Dense(10, activation='softmax', name='class')(features)

    return keras.models.Model(input=image, output=[fake, aux],
                              name='discriminator')


def build_generator_lite(latent_size):
    """Builds the much smaller generative model."""

    latent = keras.layers.Input((latent_size,), name='latent')
    image_class = keras.layers.Input((1,), dtype='int32', name='image_class')

    embed = keras.layers.Embedding(10, latent_size, init='glorot_normal')
    cls = keras.layers.Flatten()(embed(image_class))
    input_vec = keras.layers.merge([latent, cls], mode='mul')

    hidden = keras.layers.Dense(512,
        W_regularizer=keras.regularizers.l2(0.001),
        activity_regularizer=keras.regularizers.activity_l2(0.001))(input_vec)
    hidden = keras.layers.LeakyReLU()(hidden)
    hidden = gandlf.layers.PermanentDropout(0.2)(hidden)

    hidden = keras.layers.Dense(512,
        W_regularizer=keras.regularizers.l2(0.001),
        activity_regularizer=keras.regularizers.activity_l2(0.001))(hidden)
    hidden = keras.layers.LeakyReLU()(hidden)
    hidden = gandlf.layers.PermanentDropout(0.2)(hidden)

    output_layer = keras.layers.Dense(28 * 28,
        activity_regularizer=keras.regularizers.activity_l2(0.0001))(hidden)
    fake_image = keras.layers.Reshape((1, 28, 28))(output_layer)

    return keras.models.Model([latent, image_class], fake_image)


def build_discriminator_lite():
    """Builds the much smaller discriminator model."""

    image = keras.layers.Input((1, 28, 28), name='real_data')
    reshaped = keras.layers.Flatten()(image)
    # reshaped = keras.layers.BatchNormalization()(reshaped)

    # First hidden layer.
    hidden = keras.layers.Dense(512)(reshaped)
    hidden = keras.layers.Dropout(0.2)(hidden)
    hidden = keras.layers.LeakyReLU()(hidden)

    # Second hidden layer.
    hidden = keras.layers.Dense(512)(hidden)
    hidden = keras.layers.Dropout(0.2)(hidden)
    hidden = keras.layers.LeakyReLU()(hidden)

    # Output layer.
    fake = keras.layers.Dense(1, activation='sigmoid', name='src')(hidden)
    aux = keras.layers.Dense(10, activation='softmax', name='class')(hidden)

    return keras.models.Model(input=image, output=[fake, aux])


def get_mnist_data():
    """Puts the MNIST data in the right format."""

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=1)

    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    return (X_train, y_train), (X_test, y_test)


def train_model(args, X_train, y_train, y_train_ohe):
    """This is the core part whre the model is trained."""

    adam_optimizer = keras.optimizers.Adam(lr=args.lr, beta_1=args.beta)

    if args.lite:
        generator = build_generator_lite(args.nb_latent)
        discriminator = build_discriminator_lite()
    else:
        generator = build_generator(args.nb_latent)
        discriminator = build_discriminator()

    model = gandlf.Model(generator, discriminator)

    # This turns supervised learning on and off.
    loss_weights = {
        'src': 1.,
        'src_real': 3.,
        'class': 0. if args.unsupervised else 1.,
        'class_gen': 0. if args.unsupervised else 0.1,  # Weight much less.
        'class_fake': 0.,  # Should ignore this completely.
    }

    model.compile(optimizer=['sgd', adam_optimizer], loss={
        'class': 'categorical_crossentropy',
        'src': 'binary_crossentropy',
    }, metrics=['accuracy'], loss_weights=loss_weights)

    model.fit(['normal', y_train, X_train],
              {'class': y_train_ohe,
               'src_gen': '1', 'src_fake': '0', 'src_real': '1'},
              nb_epoch=args.nb_epoch, batch_size=args.nb_batch)

    return model


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
    training_params.add_argument('--plot', default=False, action='store_true',
                                 help='If set, plot samples from generator.')

    model_params = parser.add_argument_group('model params')
    model_params.add_argument('--nb_latent', type=int, default=100,
                              metavar='INT',
                              help='dimensions in the latent vector')
    model_params.add_argument('--save_path', type=str, metavar='STR',
                              default='/tmp/mnist_gan.keras_model',
                              help='Where to save the model after training')
    model_params.add_argument('--lite', default=False, action='store_true',
                              help='If set, trains the lite version instead')
    model_params.add_argument('--unsupervised', default=False,
                              action='store_true',
                              help='If set, model doesn\'t use class labels')

    optimizer_params = parser.add_argument_group('optimizer params')
    optimizer_params.add_argument('--lr', type=float, default=0.0002,
                                  metavar='FLOAT',
                                  help='learning rate for Adam optimizer')
    optimizer_params.add_argument('--beta', type=float, default=0.5,
                                  metavar='FLOAT',
                                  help='beta 1 for Adam optimizer')

    args = parser.parse_args()

    # Gets training and testing data.
    (X_train, y_train), (_, _) = get_mnist_data()

    # Turns digit labels into one-hot encoded labels.
    y_train_ohe = np.eye(10)[np.squeeze(y_train)]

    model = train_model(args, X_train, y_train, y_train_ohe)

    if args.plot:
        nb_samples = 3
        labels = y_train[:nb_samples]
        samples = model.sample(['normal', labels])

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('To plot samples from the generator, you must '
                              'install Matplotlib (not found in path).')

        for i, (sample, digit) in enumerate(zip(samples, labels)):
            sample = sample.reshape((28, 28))
            plt.figure()
            plt.imshow(sample, cmap='gray')
            plt.axis('off')
        plt.show()

    model.save(args.save_path)
    print('Saved model:', args.save_path)
