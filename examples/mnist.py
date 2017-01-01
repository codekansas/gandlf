"""TODO: Docstring."""

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
    """Builds generator model.

    TODO: Improve docstring.
    """

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
    """Builds the discriminator model.

    TODO: Improve docstring.
    """

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
                             name='auxiliary')(features)

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
    training_params.add_argument('--nb_unsup_epoch', type=int, default=50,
                                 metavar='INT',
                                 help='number of unsupervised epochs.')
    training_params.add_argument('--nb_sup_epoch', type=int, default=10,
                                 metavar='INT',
                                 help='number of supervised epochs.')
    training_params.add_argument('--nb_batch', type=int, default=32,
                                 metavar='INT',
                                 help='number of samples per batch.')

    model_params = parser.add_argument_group('model params')
    model_params.add_argument('--nb_latent', type=int, default=100,
                              metavar='INT',
                              help='dimensions in the latent vector.')

    optimizer_param = parser.add_argument_group('optimizer params')
    optimizer_param.add_argument('--lr', type=float, default=0.0002,
                                 metavar='FLOAT',
                                 help='learning rate for Adam optimizer.')
    optimizer_param.add_argument('--beta', type=float, default=0.5,
                                 metavar='FLOAT',
                                 help='beta 1 for Adam optimizer.')

    args = parser.parse_args()

    # Builds the model itself.
    optimizer = keras.optimizers.Adam(lr=args.lr, beta_1=args.beta)
    model = gandlf.Model(build_generator(args.nb_latent),
                         build_discriminator())
    model.compile(optimizer=optimizer, metrics=['accuracy'])

    # Gets data.
    (X_train, y_train), (_, _) = get_mnist_data()

    unsup_inputs = {
        'latent': 'normal',
        'real': X_train,
        'class': y_train
    }

    sup_inputs = {
        'latent': 'normal',
        'real': X_train,
        'class': y_train,
    }
    sup_outputs = {
        'auxiliary': np.eye(10)[np.squeeze(y_train)]  # One-hot encoded.
    }

    # print(model.sample({'latent': 'normal', 'class': y_train}, 10))

    # Fits data to unsupervised GAN.
    print('Starting unsupervised GAN training.')
    model.fit(unsup_inputs,
              nb_epoch=args.nb_unsup_epoch,
              batch_size=args.nb_batch)

    # Fits data to supervised GAN.
    print('Starting supervised GAN training.')
    model.fit(sup_inputs, sup_outputs,
              nb_epoch=args.nb_sup_epoch,
              batch_size=args.nb_batch)
