#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Uses pre-trained VGG16 weights to generate images.

While this is not strictly a GAN (since no real data is being fed for the
model to discriminate against), it demonstrates how Gandlf can be combined
with pre-trained weights to do creative things.
"""

from __future__ import print_function

import argparse
import os

import keras
from keras.applications import vgg16

import gandlf
import numpy as np
import matplotlib.pyplot as plt

# For repeatability.
np.random.seed(1337)

# To make the images work correctly.
keras.backend.set_image_dim_ordering('tf')


def build_generator(latent_size, nb_classes):
    """Builds the generator model."""

    latent = keras.layers.Input(shape=(latent_size,))
    img_class = keras.layers.Input(shape=(nb_classes,))

    x = keras.layers.Dense(latent_size)(img_class)
    x = keras.layers.merge([latent, x], mode='mul')

    # First Dense layer.
    x = keras.layers.Dense(512, init='glorot_normal')(latent)
    x = keras.layers.Activation('tanh')(x)

    # Second Dense layer.
    x = keras.layers.Dense(7 * 7 * 10, init='glorot_normal')(x)
    x = keras.layers.Activation('tanh')(x)
    x = keras.layers.Reshape((7, 7, 10))(x)

    # Upsample to get the final image.
    for _ in range(5):
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.GaussianNoise(0.01)(x)
        x = keras.layers.Convolution2D(64, 5, 5, border_mode='same')(x)
        x = keras.layers.Activation('tanh')(x)
        x = keras.layers.Convolution2D(64, 5, 5, border_mode='same')(x)
        x = keras.layers.Activation('tanh')(x)

    x = keras.layers.Convolution2D(128, 1, 1)(x)
    x = keras.layers.Activation('tanh')(x)
    x = keras.layers.Convolution2D(3, 1, 1)(x)
    output = keras.layers.Activation('sigmoid')(x)

    return keras.models.Model(input=[latent, img_class],
                              output=output,
                              name='generator')


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
    model.discriminator.trainable = False  # Turn off discriminator updates.

    # Randomly selects some classes to generate.
    nb_points = args.nb_batch * args.batch_size
    classes = np.random.randint(0, nb_classes, (nb_points,))
    classes = np.eye(nb_classes, dtype='int16')[classes]

    # Compiles the model using the Adam optimizer and categorical crossentropy.
    model.compile(optimizer=adam_optimizer, loss=['categorical_crossentropy'])

    # Loads the weights, if they exist.
    if os.path.exists(args.save_path) and not args.reset_weights:
        model.generator.load_weights(args.save_path)
        print('Loaded weights from "%s"' % args.save_path)

    # Trains the model, ignoring discriminator.
    model.fit(['normal', classes, 'ones'], [classes],
              nb_epoch=args.nb_epoch, batch_size=args.batch_size)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Using pre-trained VGG16 to generate images.')

    training_params = parser.add_argument_group('training params')
    training_params.add_argument('--nb_epoch', type=int, default=50,
                                 metavar='INT',
                                 help='number of epochs to train')
    training_params.add_argument('--nb_batch', type=int, default=1000,
                                 metavar='INT',
                                 help='number of training batches')
    training_params.add_argument('--batch_size', type=int, default=16,
                                 metavar='INT',
                                 help='number of samples per batch')
    training_params.add_argument('--plot', type=int, default=0,
                                 metavar='INT',
                                 help='if set, plot samples from generator.')

    model_params = parser.add_argument_group('model params')
    model_params.add_argument('--nb_latent', type=int, default=100,
                              metavar='INT',
                              help='dimensions in the latent vector')
    model_params.add_argument('--save_path', type=str, metavar='STR',
                              default='/tmp/mnist_gan.keras_model',
                              help='where to save the model after training')
    model_params.add_argument('--lite', default=False, action='store_true',
                              help='if set, trains the lite version instead')
    model_params.add_argument('--reset_weights', default=False,
                              action='store_true',
                              help='if set, the weights are reset')

    optimizer_params = parser.add_argument_group('optimizer params')
    optimizer_params.add_argument('--lr', type=float, default=0.001,
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

    model.generator.save_weights(args.save_path)
    print('Saved weights:', args.save_path)

    # Plots samples from the model.
    classes = np.eye(model.output_shape[0][-1], dtype='int16')[range(3)]
    x = model.sample(['normal', classes])
    for sample in x:
        plt.figure()
        plt.imshow(sample)
    plt.show()
