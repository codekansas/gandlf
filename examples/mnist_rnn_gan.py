#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train an Conditional Generative Adversarial Network (CGAN) on the
MNIST dataset.

This example is similar to the mnist_gan.py example, except that the
generator and discriminator consist of Recurrent Neural Networks, with
attention components for including information about the image itself.
Also, instead of predicting the output class, the output class is used
as an attention parameter for the generator and discriminator RNNs.

To show all command line options:

    ./examples/xor.py --help

Samples from the model can be plotted using Matplotlib:

    ./examples/xor.py --plot 3

By default, the model doesn't use any attention. To use attention, specify
the --gen_mode and --dis_mode parameters:

    ./examples/xor.py --gen_mode <mode> --dis_mode <mode>

The options for "mode" are:
 - None: No attention
 - 1D: RNN pays attention to the number label
 - 2D: RNN pays attention to the input image
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
keras.backend.set_image_dim_ordering('tf')


def build_generator(latent_size, mode):
    """Builds the generator model."""

    latent = keras.layers.Input((latent_size,), name='latent')
    rnn_input = keras.layers.RepeatVector(28)(latent)

    rnn_1 = keras.layers.LSTM(128, return_sequences=True)
    rnn_2 = keras.layers.LSTM(28, return_sequences=True)
    expand = keras.layers.Reshape((28, 28, 1), name='gen_image')

    if mode == '1d':  # Pay attention to class labels.
        input_class = keras.layers.Input((1,), dtype='int32',
                                         name='image_class_gen')
        embed = keras.layers.Embedding(10, 64, init='glorot_normal')
        embedded = keras.layers.Flatten()(embed(input_class))
        rnn_1 = gandlf.layers.RecurrentAttention1D(rnn_1, embedded)
        rnn_2 = gandlf.layers.RecurrentAttention1D(rnn_2, embedded)
        gen_image = expand(rnn_2(rnn_1(rnn_input)))
        return keras.models.Model(input=[latent, input_class],
                                  output=gen_image)

    elif mode == '2d':  # Pay attention to whole image.
        ref_image = keras.layers.Input((28, 28, 1), name='ref_image_gen')
        flat = keras.layers.Reshape((28, 28))(ref_image)
        rnn_1 = gandlf.layers.RecurrentAttention2D(rnn_1, flat)
        rnn_2 = gandlf.layers.RecurrentAttention2D(rnn_2, flat)
        gen_image = expand(rnn_2(rnn_1(rnn_input)))
        return keras.models.Model(input=[latent, ref_image],
                                  output=gen_image)

    else:  # No attention component.
        gen_image = expand(rnn_2(rnn_1(rnn_input)))
        return keras.models.Model(input=latent,
                                  output=gen_image)


def build_discriminator(mode):
    """Builds the discriminator model."""

    image = keras.layers.Input((28, 28, 1), name='real_data')
    rnn_input = keras.layers.Reshape((28, 28))(image)

    rnn_1 = keras.layers.LSTM(128, return_sequences=True)
    rnn_2 = keras.layers.LSTM(1, return_sequences=False, name='pred_fake')

    if mode == '1d':  # Pay attention to class labels.
        input_class = keras.layers.Input((1,), dtype='int32',
                                         name='image_class_dis')
        embed = keras.layers.Embedding(10, 64, init='glorot_normal')
        embedded = keras.layers.Flatten()(embed(input_class))
        rnn_1_attn = gandlf.layers.RecurrentAttention1D(rnn_1, embedded)
        rnn_2_attn = gandlf.layers.RecurrentAttention1D(rnn_2, embedded,
                                                        name='pred_fake')
        pred_fake = rnn_2_attn(rnn_1_attn(rnn_input))
        return keras.models.Model(input=[image, input_class],
                                  output=pred_fake)

    elif mode == '2d':  # Pay attention to whole image.
        ref_image = keras.layers.Input((28, 28, 1), name='ref_image_dis')
        attn_reshaped = keras.layers.Reshape((28, 28))(ref_image)
        rnn_1_attn = gandlf.layers.RecurrentAttention2D(rnn_1, attn_reshaped)
        rnn_2_attn = gandlf.layers.RecurrentAttention2D(rnn_2, attn_reshaped,
                                                        name='pred_fake')
        pred_fake = rnn_2_attn(rnn_1_attn(rnn_input))
        return keras.models.Model(input=[image, ref_image],
                                  output=pred_fake)

    else:
        pred_fake = rnn_2(rnn_1(rnn_input))
        return keras.models.Model(input=image,
                                  output=pred_fake)


def get_mnist_data():
    """Puts the MNIST data in the right format."""

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=-1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=-1)

    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    return (X_train, y_train), (X_test, y_test)

def train_model(args, X_train, y_train):
    """This is the core part where the model is trained."""

    adam_optimizer = keras.optimizers.Adam(lr=args.lr, beta_1=args.beta)

    gen_mode, dis_mode = args.gen_mode.lower(), args.dis_mode.lower()
    generator = build_generator(args.nb_latent, gen_mode)
    discriminator = build_discriminator(dis_mode)

    # Builds the model with the right parameters.
    model = gandlf.Model(generator, discriminator)
    loss_weights = {'pred_fake': 1.}
    loss = {'pred_fake': 'binary_crossentropy'}
    model.compile(optimizer=adam_optimizer, loss=loss,
                  loss_weights=loss_weights)

    # Model inputs.
    inputs = {'latent': 'normal', 'real_data': X_train}

    # Adds extra inputs for generator attention part.
    if gen_mode == '1d':
        inputs['image_class_gen'] = y_train
    elif gen_mode == '2d':
        inputs['ref_image_gen'] = X_train

    # Adds extra inputs for discriminator attention part.
    if dis_mode == '1d':
        inputs['image_class_dis'] = y_train
    elif dis_mode == '2d':
        inputs['ref_image_dis'] = X_train

    # Model outputs.
    outputs = {'pred_fake': '1', 'pred_fake_fake': '0'}

    model.fit(inputs, outputs, nb_epoch=args.nb_epoch, batch_size=args.nb_batch)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Recurrent GAN for MNIST digits.')

    training_params = parser.add_argument_group('training params')
    training_params.add_argument('--nb_epoch', type=int, default=50,
                                 metavar='INT',
                                 help='number of epochs to train')
    training_params.add_argument('--nb_batch', type=int, default=32,
                                 metavar='INT',
                                 help='number of samples per batch')
    training_params.add_argument('--plot', type=int, default=0,
                                 metavar='INT',
                                 help='number of generator samples to plot')

    model_params = parser.add_argument_group('model params')
    model_params.add_argument('--nb_latent', type=int, default=100,
                              metavar='INT',
                              help='dimensions in the latent vector')
    model_params.add_argument('--save_path', type=str, metavar='STR',
                              default='/tmp/mnist_rnn_gan.keras_model',
                              help='where to save the model after training')
    model_params.add_argument('--gen_mode', type=str, metavar='STR',
                              default='none',
                              help='generator attn: "none", "1d", or "2d"')
    model_params.add_argument('--dis_mode', type=str, metavar='STR',
                              default='none',
                              help='discriminator attn: "none", "1d", or "2d"')

    optimizer_params = parser.add_argument_group('optimizer params')
    optimizer_params.add_argument('--lr', type=float, default=0.0002,
                                  metavar='FLOAT',
                                  help='learning rate for Adam optimizer')
    optimizer_params.add_argument('--beta', type=float, default=0.5,
                                  metavar='FLOAT',
                                  help='beta 1 for Adam optimizer')

    args = parser.parse_args()

    if args.gen_mode.lower() not in ['none', '1d', '2d']:
        raise ValueError('"gen_mode" should be "none" (no attention), "1d" '
                         '(pay attention to labels), or "2d" (pay attention '
                         'to image). Got: %s' % args.gen_mode)

    if args.dis_mode.lower() not in ['none', '1d', '2d']:
        raise ValueError('"dis_mode" should be "none" (no attention), "1d" '
                         '(pay attention to labels), or "2d" (pay attention '
                         'to image). Got: %s' % args.dis_mode)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('To plot samples from the generator, you must '
                              'install Matplotlib (not found in path).')

    # Gets training and testing data.
    (X_train, y_train), (_, _) = get_mnist_data()

    model = train_model(args, X_train, y_train)
