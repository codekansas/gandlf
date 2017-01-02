## Generative Adversarial Network Deep Learning Framework

>He that breaks a thing to find out what it is has left the path of wisdom.
>(Tim Peters, Council of Elrond Style Guide)

This is a framework built on top of [Keras](https://github.com/fchollet/keras) for training [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).

## Installation

Using Pip:

    pip install git+https://github.com/codekansas/gandlf
    pip install h5py  # To save and load Keras models

Installing from source:

    git clone https://github.com/codekansas/gandlf
    cd gandlf
    pip install -r requirements.txt
    python setup.py install

## Guiding Principles

In no particular order:

 - *Keras-esque*: The APIs should feel familiar for Keras users, with some minor changes.
 - *Powerful*: Models should support a wide variety of GAN architectures.
 - *Extensible*: Models should be easy to modify for different experiments.

## Issues Etiquette

More examples would be awesome! If you use this for something, create a stand-alone script that can be run and I'll put it in the `examples` directory. Just create a pull request for it.

If you encounter an error, I would really like to hear about it! But please raise an issue before creating a pull request, to discuss the error. There a number of resources available for discussion; here is a list, again in no particular order:

 - [/r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
 - [Adversarial Training Facebook Group](https://www.facebook.com/groups/675606912596390/)
