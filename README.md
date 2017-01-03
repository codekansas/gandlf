## Generative Adversarial Network Deep Learning Framework

>He that breaks a thing to find out what it is has left the path of wisdom.
>(Council of Elrond Style Guide)

This is a framework built on top of [Keras](https://github.com/fchollet/keras) for training [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).

Because it's built on top of Keras, it has the benefits of being *modular*, *minimal* and *extensible*, running on both CPU and GPU using either Tensorflow or Theano.

## Installation

Using Pip:

````sh
pip install git+https://github.com/codekansas/gandlf
pip install h5py  # To save and load Keras models
````

Installing from source:

````sh
git clone https://github.com/codekansas/gandlf
cd gandlf
pip install -r requirements.txt
python setup.py install
````

## Quick Start

Below demonstrates how a Gandlf model works.

````python
import keras
import gandlf

def build_generator():
    latent_vec = keras.layers.Input(shape=..., name='latent_vec')
    output_layer = keras.layers.Dense(...)(latent_vec)
    return keras.models.Model(input=[latent_vec], output=[output_layer])

def build_discriminator():
    data_input = keras.layers.Input(shape=..., name='data_input')
    output_layer = keras.layers.Dense(..., name='src')(data_input)
    return keras.models.Model(input=[data_input], output=[output_layer])

model = gandlf.Model(generator=build_generator(),
                     discriminator=build_discriminator())

model.compile(optimizer='adam', loss='binary_crossentropy')

# Latent vector is fed data from a random normal distribution.
# <input_data> represents the real data.
# 'zeros' and 'ones' represent 'fake' and 'real', respectively.
# In this case, the discriminator learns 'real data' -> 1
# and fake 'generated data' -> 0, while the generator learns
# 'generated data' -> 1.
model.fit(['normal', <input_data>], ['ones', 'zeros'])

# There are many ways to do the same thing, depending on the level
# of specificity you need (especially when training with auxiliary parts).
# The above function could be written as any of the following:
model.fit(['normal', <input_data>], {'gen_real': 'ones', 'fake': 'zeros'})
model.fit({'latent_vec': 'normal', 'data_input': <input_data>},
          {'src': 'ones', 'src_fake': 'zeros'})
model.fit({'latent_vec': 'normal', 'data_input': <input_data>},
          {'src_gen': '1', 'src_real': '1', 'src_fake': '0'})

# The model provides a function for predicting the discriminator's
# output on some input data, which is useful for auxiliary classification.
model_predictions = model.predict([<input_data>])

# The model also provides a function for sampling from the generator.
generated_data = model.sample(['normal'], num_samples=10)

# Under the hood, other functions work like their Keras counterparts.
model.save('/save/path')
model.generator.save('/generator/save/path')
model.discriminator.save('/discriminator/save/path')
````

## Guiding Principles

In no particular order:

 - *Keras-esque*: The APIs should feel familiar for Keras users, with some minor changes.
 - *Powerful*: Models should support a wide variety of GAN architectures.
 - *Extensible*: Models should be easy to modify for different experiments.

## Issues Etiquette

More examples would be awesome! If you use this for something, create a stand-alone script that can be run and I'll put it in the `examples` directory. Just create a pull request for it.

Contribute code too! Anything that might be interesting and relevant for building GANs. Since this is more task-specific than Keras, there is more room for more experimental layers and ideas (notice that "dependability" isn't one of the guiding principles, although it would be good to not have a huge nest of bugs).

If you encounter an error, I would really like to hear about it! But please raise an issue before creating a pull request, to discuss the error. Even better, look around the code to see if you can spot what's going wrong. Try to practice good etiquette, not just for this project, but for open source projects in general; this means making an honest attempt at solving the problem before asking for help.

