import keras
import gandlf


def build_generator(latent_size):
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

    latent = keras.layers.Input(shape=(latent_size, ))

    image_class = keras.layers.Input(shape=(1,), dtype='int32')

    embedded = keras.layers.Embedding(10, latent_size,
                                      init='glorot_normal')(image_class)
    cls = keras.layers.Flatten()(embedded)
    h = keras.layers.merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return keras.models.Model(input=[latent, image_class], output=fake_image)


def build_discriminator():
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

    image = keras.layers.Input(shape=(1, 28, 28))

    features = cnn(image)

    fake = keras.layers.Dense(1, activation='sigmoid',
                              name='generation')(features)
    aux = keras.layers.Dense(10, activation='softmax',
                             name='auxiliary')(features)

    return keras.models.Model(input=image, output=[fake, aux])


print ('Hello world!')
