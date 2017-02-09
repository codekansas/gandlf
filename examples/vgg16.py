#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Uses pre-trained VGG16 weights to generate images.

While this is not strictly a GAN (since no real data is being fed for the
model to discriminate against), it demonstrates how Gandlf can be combined
with pre-trained weights to do creative things.
"""

from __future__ import print_function

import argparse

import keras
from keras.applications import vgg16

import gandlf
import numpy as np

# For repeatability.
np.random.seed(1337)

# To make the images work correctly.
keras.backend.set_image_dim_ordering('tf')


def build_generator(latent_size, nb_classes):
    """Builds the generator model."""

    cnn = keras.models.Sequential()

    cnn.add(keras.layers.Dense(14 * 14 * 16, input_dim=latent_size))
    cnn.add(keras.layers.LeakyReLU())

    cnn.add(keras.layers.Reshape((14, 14, 16)))

    cnn.add(keras.layers.UpSampling2D(size=(4, 4)))
    cnn.add(keras.layers.Convolution2D(256, 5, 5, border_mode='same',
                                       init='glorot_normal'))
    cnn.add(keras.layers.LeakyReLU())

    cnn.add(keras.layers.UpSampling2D(size=(4, 4)))
    cnn.add(keras.layers.Convolution2D(256, 5, 5, border_mode='same',
                                       init='glorot_normal'))
    cnn.add(keras.layers.LeakyReLU())

    cnn.add(keras.layers.Convolution2D(3, 2, 2, border_mode='same',
                                       activation='tanh', init='glorot_normal'))

    latent = keras.layers.Input(shape=(latent_size,), name='latent')
    image_class = keras.layers.Input(shape=(1,), dtype='int32',
                                     name='image_class')

    # Shapes the latent input to make it easier to generate the right classes.
    embed = keras.layers.Embedding(nb_classes, latent_size,
                                   init='glorot_normal')
    cls = keras.layers.Flatten()(embed(image_class))
    h = keras.layers.merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    generator_weighted = keras.models.Model(input=[latent, image_class],
                                            output=fake_image,
                                            name='generator_weighted')

    return generator_weighted


def train_model(args):
    """This is the core part, where the model is trained."""

    # This is the VGG16 input shape.
    input_shape = (244, 244, 3)

    # Number of ImageNet classes.
    nb_classes = 1000

    # Builds the Adam optimizer according to our specifications.
    adam_optimizer = keras.optimizers.Adam(lr=args.lr, beta_1=args.beta)

    generator = build_generator(args.nb_latent, nb_classes)
    discriminator = vgg16.VGG16(weights='imagenet', include_top=True)

    model = gandlf.Model(generator, discriminator)
    model.discriminator.trainable = False

    # Compiles the model using the Adam optimizer and categorical crossentropy.
    model.compile(optimizer=adam_optimizer, loss=['categorical_crossentropy'],
                  metrics=['acc'])

    # Randomly selects some classes to generate.
    classes = np.random.randint(0, nb_classes, (args.nb_batch,))
    eye = np.eye(nb_classes, dtype='int16')
    classes_ohe = eye[classes]
    classes = np.expand_dims(classes, -1)

    # Trains the model, ignoring discriminator.
    model.fit({'latent': 'normal', 'image_class': classes, 'input_1': 'ones'},
              [classes_ohe], nb_epoch=args.nb_epoch, batch_size=args.nb_batch)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Using pre-trained VGG16 to generate images.')

    training_params = parser.add_argument_group('training params')
    training_params.add_argument('--nb_epoch', type=int, default=50,
                                 metavar='INT',
                                 help='number of epochs to train')
    training_params.add_argument('--nb_batch', type=int, default=32,
                                 metavar='INT',
                                 help='number of samples per batch')
    training_params.add_argument('--plot', type=int, default=0,
                                 metavar='INT',
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

    optimizer_params = parser.add_argument_group('optimizer params')
    optimizer_params.add_argument('--lr', type=float, default=0.0002,
                                  metavar='FLOAT',
                                  help='learning rate for Adam optimizer')
    optimizer_params.add_argument('--beta', type=float, default=0.5,
                                  metavar='FLOAT',
                                  help='beta 1 for Adam optimizer')

    args = parser.parse_args()

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('To plot samples from the generator, you must '
                              'install Matplotlib (not found in path).')

    model = train_model(args)

    if args.plot:
        samples = model.sample(['normal', 4], num_samples=args.plot)

        for sample in samples:
            sample = sample.reshape((224, 224, 3))
            plt.figure()
            plt.imshow(sample, cmap='gray')
            plt.axis('off')
        plt.show()

    model.save(args.save_path)
    print('Saved model:', args.save_path)
