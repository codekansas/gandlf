#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
After training a GAN, use gradient descent to map images back to their original
position in the latent vector.
"""

from __future__ import print_function

import argparse

import keras
from keras.datasets import mnist
import keras.backend as K

import gandlf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# For repeatability.
np.random.seed(1337)

# To make the images work correctly.
keras.backend.set_image_dim_ordering('tf')


def plot_as_gif(x, x_ref, title,
                interval=50):
    """Plots data as a gif.

    Args:
        x: numpy array with shape (batch_size, 28, 28, 1).
        interval: int, the time between frames in milliseconds.
    """

    filename = title.lower().replace(' ', '_')
    save_path = '/tmp/%s.gif' % filename

    # Plots the first sample.
    fig, (ax, ax2) = plt.subplots(1, 2)
    im = ax.imshow(x[0].reshape((28, 28)),
            interpolation='none',
            aspect='auto',
            cmap='gray',
            animated=True)

    # Fixes the dimensions.
    ax.axis('off')

    def updatefig(i, *args):
        im.set_array(x[i].reshape((28, 28)))
        return im,

    anim = FuncAnimation(fig, updatefig,
            frames=np.arange(0, x.shape[0]),
            interval=interval)

    ax2.imshow(x_ref.reshape((28, 28)),
            interpolation='none',
            aspect='auto',
            cmap='gray')
    ax2.axis('off')

    plt.suptitle(title)

    anim.save(save_path, dpi=80, writer='imagemagick')
    print('Saved gif to "%s".' % save_path)

    plt.show()


def build_generator():
    """Builds the big generator model."""

    latent = keras.layers.Input((100,), name='latent')

    image_class = keras.layers.Input((10,), dtype='float32',
                                     name='image_class')
    d = keras.layers.Dense(100)(image_class)
    merged = keras.layers.merge([latent, d], mode='sum')

    hidden = keras.layers.Dense(512)(merged)
    hidden = keras.layers.LeakyReLU()(hidden)

    hidden = keras.layers.Dense(512)(hidden)
    hidden = keras.layers.LeakyReLU()(hidden)

    output_layer = keras.layers.Dense(28 * 28,
        activation='tanh')(hidden)
    fake_image = keras.layers.Reshape((28, 28, 1))(output_layer)

    return keras.models.Model(input=[latent, image_class],
                              output=fake_image)


def build_discriminator():
    """Builds the big discriminator model."""

    image = keras.layers.Input((28, 28, 1), name='real_data')
    hidden = keras.layers.Flatten()(image)

    # First hidden layer.
    hidden = keras.layers.Dense(512)(hidden)
    hidden = keras.layers.Dropout(0.3)(hidden)
    hidden = keras.layers.LeakyReLU()(hidden)

    # Second hidden layer.
    hidden = keras.layers.Dense(512)(hidden)
    hidden = keras.layers.Dropout(0.3)(hidden)
    hidden = keras.layers.LeakyReLU()(hidden)

    # Output layer.
    fake = keras.layers.Dense(1, activation='sigmoid', name='s')(hidden)
    aux = keras.layers.Dense(10, activation='softmax', name='c')(hidden)

    return keras.models.Model(input=image, output=[fake, aux])


def reverse_generator(generator, X_sample, y_sample, title):
    """Gradient descent to map images back to their latent vectors."""

    latent_vec = np.random.normal(size=(1, 100))

    # Function for figuring out how to bump the input.
    target = K.placeholder()
    loss = K.sum(K.square(generator.outputs[0] - target))
    grad = K.gradients(loss, generator.inputs[0])[0]
    update_fn = K.function(generator.inputs + [target], [grad])

    # Repeatedly apply the update rule.
    xs = []
    for i in range(60):
        print('%d: latent_vec mean=%f, std=%f'
              % (i, np.mean(latent_vec), np.std(latent_vec)))
        xs.append(generator.predict_on_batch([latent_vec, y_sample]))
        for _ in range(10):
            update_vec = update_fn([latent_vec, y_sample, X_sample])[0]
            latent_vec -= update_vec * update_rate

    # Plots the samples.
    xs = np.concatenate(xs, axis=0)
    plot_as_gif(xs, X_sample, title)



def get_mnist_data(binarize=False):
    """Puts the MNIST data in the right format."""

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if binarize:
        X_test = np.where(X_test >= 10, 1, -1)
        X_train = np.where(X_train >= 10, 1, -1)
    else:
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    update_rate = 0.1

    generator = build_generator()

    (X_train, y_train), (X_test, y_test) = get_mnist_data(binarize=False)
    X_sample = X_test[5:6]
    y_sample = y_test[5:6]

    # Plot samples before training.
    reverse_generator(
            generator,
            X_sample,
            y_sample,
            'Before training generator')

    # Trains GAN.
    discriminator = build_discriminator()
    model = gandlf.Model(generator, discriminator)
    loss_weights = {'s': 1., 'c': 1., 'c_fake': 0.}
    model.compile(optimizer=['adam', 'sgd'],
                  loss=['binary_crossentropy', 'categorical_crossentropy'],
                  loss_weights=loss_weights)
    model.fit(['normal', y_train, X_train],
              {'s': 'ones', 's_fake': 'zeros', 'c': y_train},
              nb_epoch=1,
              batch_size=32)

    # Plot samples after training.
    reverse_generator(
            generator,
            X_sample,
            y_sample,
            'After training generator')
