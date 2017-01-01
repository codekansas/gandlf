"""TODO: Docstring."""

from __future__ import print_function

import argparse
import keras

import gandlf
import numpy as np

# For repeatability.
np.random.seed(1667)


def get_training_data(num_samples):
    """Generates some training data."""

    x = np.random.randint(0, 2, size=(num_samples, 2))

    y = np.logical_xor(x[:, 0], x[:, 1])
    y = np.cast['int32'](y)

    x = np.cast['float32'](x) * 2 - 1  # Scales to [-1, 1].
    x += np.random.normal(scale=0.3, size=x.shape)  # Adds some random noise.

    return x, y


def build_generator(latent_size):
    latent_layer = keras.layers.Input(shape=(latent_size,), name='latent')
    class_input = keras.layers.Input(shape=(1,), name='class')
    embedded = keras.layers.Embedding(2, latent_size,
                                      init='glorot_normal')(class_input)
    flat_embedded = keras.layers.Flatten()(embedded)

    input_layer = keras.layers.merge([latent_layer, flat_embedded],
                                     mode='mul')
    hidden_layer = keras.layers.Dense(16, activation='tanh')(input_layer)
    output_layer = keras.layers.Dense(2)(hidden_layer)

    return keras.models.Model([latent_layer, class_input],
                              output_layer, name='generator')


def build_discriminator():
    input_layer = keras.layers.Input(shape=(2,), name='real')

    normalized = keras.layers.BatchNormalization()(input_layer)
    hidden_layer = keras.layers.Dense(16, activation='tanh')(normalized)

    real_fake_pred = keras.layers.Dense(1, activation='sigmoid',
                                        name='real_fake')(hidden_layer)
    class_pred = keras.layers.Dense(2, activation='sigmoid',
                                    name='class')(hidden_layer)

    # The first output of this model (real_fake_pred) is treated as
    # the "real / fake" predictor.
    return keras.models.Model(input_layer, [real_fake_pred, class_pred],
                              name='discriminator')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Basic XOR example using a GAN.')

    training_params = parser.add_argument_group('training params')
    training_params.add_argument('--nb_epoch', type=int, default=10,
                                 metavar='INT',
                                 help='number of training epochs')
    training_params.add_argument('--nb_batch', type=int, default=100,
                                 metavar='INT',
                                 help='number of samples per batch')
    training_params.add_argument('--supervised', default=False,
                                 action='store_true',
                                 help='if set, train as an ACGAN')

    model_params = parser.add_argument_group('model params')
    model_params.add_argument('--nb_latent', type=int, default=10,
                              metavar='INT',
                              help='dimensions in the latent vector')
    model_params.add_argument('--nb_samples', type=int, default=10000,
                              metavar='INT',
                              help='total number of training samples')

    args = parser.parse_args()

    model = gandlf.Model(build_generator(args.nb_latent),
                         build_discriminator())
    model.compile(optimizer='adam', loss={
        'class': 'categorical_crossentropy',
        'generator': gandlf.losses.negative_binary_crossentropy,
        'discriminator': 'binary_crossentropy',
    }, metrics=['accuracy'])

    x, y = get_training_data(args.nb_samples)
    y_ohe = np.eye(2)[y]
    y = np.expand_dims(y, -1)

    model.fit({'latent': 'normal', 'real': x, 'class': y},
              {'generator': 'ones', 'discriminator': 'zeros', 'class': y_ohe},
              train_auxiliary=args.supervised, nb_epoch=args.nb_epoch,
              batch_size=args.nb_batch)

    print('\n:: Input Data ::')
    print(x[:10])

    print('\n:: Target Data ::')
    print(y[:10])

    if args.supervised:
        print('\n:: Predictions for Real Data ::')
        print(np.argmax(model.predict({'real': x[:10]})[1], -1)
              .reshape((-1, 1)))

    print('\n:: Generated Input Data (Knowing Target Data) ::')
    p = model.sample({'latent': 'normal', 'class': y[:10]})
    print(p)

    if args.supervised:
        print('\n:: Predictions for Generated Data ::')
        print(np.argmax(model.predict({'real': p})[1], -1).reshape((-1, 1)))
