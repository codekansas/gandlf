#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a GAN for producing high-resolution MNIST digits, in the style of
Hardmaru's blog post: "Generating Large Images from Latent Vectors"

The discriminator model is trained to take the x and y coordinates and
the pixel value and determine if the pixel value is realistic for that
coordinate. The generator mdoel is trained to take as inputs (x, y)
coordinates and predict the value of the pixel.
"""

from __future__ import print_function

import argparse

from keras import backend as K
from keras import optimizers

from keras.datasets import mnist

from keras.engine import InputSpec

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import Lambda
from keras.layers import merge
from keras.layers import Reshape
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from keras.models import Model

import gandlf
import numpy as np

# For repeatability.
np.random.seed(1337)


def wide_normal(shape, name=None):
    value = np.random.normal(scale=2., size=shape)
    return K.variable(value, name=name)

class WideDense(Dense):
    """Hack to enable saving / loading with custom function."""

    def __init__(self, output_dim, **kwargs):
        kwargs['init'] = wide_normal
        super(WideDense, self).__init__(output_dim, **kwargs)

def build_generator(latent_size):
    """Builds the generator model."""

    # The x, y inputs.
    x = Input((28 * 28, 1), name='gen_x')
    y = Input((28 * 28, 1), name='gen_y')
    r = merge([x, y],
          mode=lambda x: K.sqrt(x[0] * x[1]),
          output_shape=lambda x: x[0])

    # The latent vector and image class (constant across points).
    latent = Input((latent_size,), name='latent')
    image_cls = Input((10,), name='gen_image_cls')

    # These inputs are constant over all points.
    time_const = merge([latent, image_cls], mode='concat')
    time_const = RepeatVector(28 * 28)(time_const)

    # Merges the inputs to a single input.
    hidden = merge([x, y, r, time_const], mode='concat')
    hidden = TimeDistributed(Dense(256, activation='tanh'))(hidden)
    hidden = gandlf.layers.PermanentDropout(0.3)(hidden)
    hidden = TimeDistributed(Dense(256, activation='tanh'))(hidden)
    hidden = gandlf.layers.PermanentDropout(0.3)(hidden)

    # Output layer.
    output = TimeDistributed(Dense(1, activation='tanh'))(hidden)

    return Model(input=[x, y, latent, image_cls], output=output)


def build_discriminator():
    """Builds the discriminator model."""

    pixel = Input((28 * 28, 1), name='pixel')
    flat = Flatten()(pixel)
    sim = gandlf.layers.BatchSimilarity('rbf', n=5)(flat)

    # This input is constant over all points.
    image_cls = Input((10,), name='dis_image_cls')

    # The input values.
    hidden = merge([flat, sim, image_cls], mode='concat')
    hidden = Dense(256, activation='tanh')(hidden)
    hidden = Dense(256, activation='tanh')(hidden)

    # Output layer.
    output = Dense(1, activation='sigmoid', name='src')(hidden)

    return Model(input=[pixel, image_cls], output=output)


def get_mnist_data(binarize=False):
    """Puts the MNIST data in the right format."""

    (image_data, image_labels), _ = mnist.load_data()

    if binarize:
        image_data = np.where(image_data >= 10, 1, -1)
    else:
        image_data = (image_data.astype(np.float32) - 127.5) / 127.5

    image_data = np.reshape(image_data, (image_data.shape[0], -1))

    return image_data, image_labels


def train_model(args, image_data, image_labels, x_idx, y_idx):
    """This is the core part where the model is trained."""

    adam_optimizer = optimizers.Adam(lr=args.lr)

    # Builds the model.
    if args.load_existing:
        custom_objects = {'WideDense': WideDense}
        model = gandlf.models.load_model(args.save_path, custom_objects)
    else:
        generator = build_generator(args.nb_latent)
        discriminator = build_discriminator()
        model = gandlf.Model(generator, discriminator)
    model.compile(optimizer=adam_optimizer,
                  loss='binary_crossentropy')

    # Builds the inputs.
    gen_inputs = [x_idx, y_idx, args.latent_type, image_labels]
    dis_inputs = [image_data, image_labels]
    inputs = gen_inputs + dis_inputs

    # Builds the targets (gen and real -> 1, fake -> 0).
    targets = {'gen_real': 1, 'fake': 0}

    # Train the model.
    model.fit(inputs, targets,
              nb_epoch=args.nb_epoch,
              batch_size=args.nb_batch)

    return model


def plot_upsampled_digit(model, upsample_factor, nb_latent, output=None):

    # Creates the upsampled indices.
    num = 28 * upsample_factor

    # Samples from the model.
    if output is None:
        x_up = np.tile(np.linspace(-1, 1, num=num), (num, 1))
        y_up = np.transpose(x_up, (1, 0))
        x_up = np.reshape(x_up, (-1, 28 * 28, 1))
        y_up = np.reshape(y_up, (-1, 28 * 28, 1))

        # Gets a random label.
        label = np.random.randint(0, 10)
        print(label)  # Lets the user know the target label.

        # Converts to one-hot encoding and tiles across all the batches.
        label = np.tile([np.eye(10)[label]], (x_up.shape[0], 1))

        # Gets a random latent vector.
        latent_vec = np.tile(np.random.uniform(size=(nb_latent,)),
                             (x_up.shape[0], 1))
        output = model.sample([x_up, y_up, latent_vec, label])

    # Reshapes the output to get an image.
    output = np.reshape(output, (num, num))

    # Plot the output (negative to make 1 -> black, 0 -> white).
    plt.figure()
    plt.imshow(-output, cmap='gray', interpolation='bicubic')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GAN for generating upsampled MNIST digits.')

    training_params = parser.add_argument_group('training params')
    training_params.add_argument('--nb_epoch', type=int, default=10,
                                 metavar='INT',
                                 help='number of epochs to train')
    training_params.add_argument('--nb_batch', type=int, default=32,
                                 metavar='INT',
                                 help='number of samples per batch')
    training_params.add_argument('--plot', type=int, default=0,
                                 metavar='INT',
                                 help='number of generator samples to plot')
    training_params.add_argument('--binarize', default=False,
                                 action='store_true',
                                 help='if set, make mnist data binary')

    model_params = parser.add_argument_group('model params')
    model_params.add_argument('--nb_latent', type=int, default=10,
                              metavar='INT',
                              help='dimensions in the latent vector')
    model_params.add_argument('--save_path', type=str, metavar='STR',
                              default='/tmp/mnist_upsample.keras_model',
                              help='where to save the model after training')
    model_params.add_argument('--latent_type', type=str, default='normal',
                              metavar='STR',
                              help='"normal" or "uniform"')
    model_params.add_argument('--factor', type=int, default=2,
                              metavar='INT',
                              help='factor to upsample')
    model_params.add_argument('--load_existing', default=False,
                              action='store_true',
                              help='if set, load an existing model')

    optimizer_params = parser.add_argument_group('optimizer params')
    optimizer_params.add_argument('--lr', type=float, default=0.001,
                                  metavar='FLOAT',
                                  help='learning rate for Adam optimizer')

    args = parser.parse_args()

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('To plot samples from the generator, you must '
                              'install Matplotlib (not found in path).')

    if args.latent_type.lower() not in ['normal', 'uniform']:
        raise ValueError('Latent vector must be either "normal" or "uniform", '
                         'got "%s".' % args.latent_type)

    # Gets the training data.
    image_data, image_labels = get_mnist_data(args.binarize)

    # x_idx: [[-1] * 28, [-26/28] * 28, ..., [26/28] * 28, [1] * 28]
    # Maps the range [-28, 28] to [-1, 1]. x_idx increases in the x direction,
    # y_idx increases in the y direction.
    x_idx = np.tile(np.linspace(-1, 1, num=28), (image_data.shape[0], 28, 1))
    y_idx = np.transpose(x_idx, (0, 2, 1))

    # Sets to correct shapes.
    image_data = np.reshape(image_data, (-1, 28 * 28, 1))
    x_idx = np.reshape(x_idx, (-1, 28 * 28, 1))
    y_idx = np.reshape(y_idx, (-1, 28 * 28, 1))
    image_labels = np.eye(10)[image_labels]

    # Train the model.
    model = train_model(args, image_data, image_labels, x_idx, y_idx)

    # Plots samples from the model.
    for i in range(args.plot):
        plot_upsampled_digit(model, args.factor, args.nb_latent)

    # Saves the model.
    model.save(args.save_path)
    print('Saved model:', args.save_path)
