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


class BiasedDense(Dense):
    """Like a normal Dense layer, but without setting the initial bias to 0."""

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.b = self.add_weight((self.output_dim,),
                                 initializer=self.init,
                                 name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer,
                                 constraint=self.b_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True


def wide_uniform(shape, name=None):
    value = np.random.uniform(low=-4, high=4, size=shape)
    return K.variable(value, name=name)


def build_generator(latent_size):
    """Builds the generator model."""

    # The x, y inputs.
    x = Input((1,), name='gen_x')
    y = Input((1,), name='gen_y')
    r = merge([x, y],
              mode=lambda x: K.sqrt(x[0] * x[1]),
              output_shape=lambda x: x[0])

    # The latent vector and image class (constant across points).
    latent = Input((latent_size,), name='latent')
    image_cls = Input((1,), dtype='int32', name='gen_image_cls')
    image_emb = Embedding(10, 1, wide_uniform)(image_cls)
    image_emb = Flatten()(image_emb)

    # Merges the inputs to a single input.
    hidden = merge([x, y, r, latent, image_emb], mode='concat')
    hidden = BiasedDense(64, wide_uniform, activation='tanh')(hidden)
    hidden = BiasedDense(64, wide_uniform, activation='tanh')(hidden)
    hidden = BiasedDense(64, wide_uniform, activation='tanh')(hidden)

    # Output layer.
    output = Dense(1, activation='tanh')(hidden)

    return Model(input=[x, y, latent, image_cls], output=output)


def build_discriminator():
    """Builds the discriminator model."""

    # The x, y, pixel value inputs.
    x = Input((1,), name='dis_x')
    y = Input((1,), name='dis_y')
    r = merge([x, y],
              mode=lambda x: K.sqrt(x[0] * x[1]),
              output_shape=lambda x: x[0])
    pixel = Input((1,), name='pixel')

    image_cls = Input((1,), dtype='int32', name='dis_image_cls')
    image_emb = Embedding(10, 10, 'normal')(image_cls)
    image_emb = Flatten()(image_emb)

    # The input values.
    hidden = merge([x, y, r, pixel, image_emb], mode='concat')
    hidden = BiasedDense(64, wide_uniform, activation='tanh')(hidden)
    hidden = BiasedDense(64, wide_uniform, activation='tanh')(hidden)
    hidden = BiasedDense(64, wide_uniform, activation='tanh')(hidden)

    # Output layer.
    output = BiasedDense(1, activation='sigmoid', name='src')(hidden)

    return Model(input=[pixel, x, y, image_cls], output=output)


def get_mnist_data(binarize=False):
    """Puts the MNIST data in the right format."""

    (image_data, image_labels), _ = mnist.load_data()

    if binarize:
        image_data = np.where(image_data >= 10, 1, -1)
    else:
        image_data = (image_data.astype(np.float32) - 127.5) / 127.5

    image_data = np.reshape(image_data, (image_data.shape[0], -1))
    image_labels = np.expand_dims(image_labels, -1)

    return image_data, image_labels


def train_model(args, image_data, image_labels, x_idx, y_idx):
    """This is the core part where the model is trained."""

    adam_optimizer = optimizers.Adam(lr=args.lr)

    # Builds the model.
    generator = build_generator(args.nb_latent)
    discriminator = build_discriminator()
    model = gandlf.Model(generator, discriminator)
    model.compile(optimizer=adam_optimizer,
                  loss={'fake_real': 'binary_crossentropy',
                        'gen': gandlf.losses.negative_binary_crossentropy})

    # Builds the inputs.
    gen_inputs = [x_idx, y_idx, args.latent_type, image_labels]
    dis_inputs = [image_data, x_idx, y_idx, image_labels]
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
    x_up = np.tile(np.linspace(-1, 1, num=num), (num, 1))
    y_up = np.transpose(x_up, (1, 0))
    x_up = np.reshape(x_up, (-1, 1))
    y_up = np.reshape(y_up, (-1, 1))

    # Gets a random label.
    label = np.tile([np.random.randint(0, 10)], (x_up.shape[0], 1))

    # Gets a random latent vector.
    latent_vec = np.tile(np.random.uniform(size=(nb_latent,)),
                         (x_up.shape[0], 1))

    # Samples from the model.
    if output is None:
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
    model_params.add_argument('--latent_type', type=str, default='uniform',
                              metavar='STR',
                              help='"normal" or "uniform"')
    model_params.add_argument('--factor', type=int, default=2,
                              metavar='INT',
                              help='factor to upsample')

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

    # Flattens MNIST digits.
    image_data = np.reshape(image_data, (-1, 1))
    image_labels = np.repeat(image_labels, 28 * 28, axis=0)

    # Flatten x and y indices.
    x_idx = np.reshape(x_idx, (-1, 1))
    y_idx = np.reshape(y_idx, (-1, 1))

    # Train the model.
    model = train_model(args, image_data, image_labels, x_idx, y_idx)

    # Plots samples from the model.
    for i in range(args.plot):
        plot_upsampled_digit(model, args.factor, args.nb_latent)

    # Saves the model.
    model.save(args.save_path)
    print('Saved model:', args.save_path)
