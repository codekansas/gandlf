[vgg16.py](https://github.com/codekansas/gandlf/blob/master/examples/vgg16.py)

This example illustrates how a pre-trained model can be used to generate images. The discriminator is the VGG16 model which is available with Keras, and the discriminator weights are frozen. The generator learns to generate samples of a desired class. This isn't a pure GAN, because the discriminator model doesn't learn to discriminate adversarial examples, but the generator does learn to map a random normal distribution to generate image classes in the ImageNet dataset.

The trained samples are a work in progress (waiting on access to a GPU). If you run it and get interesting results, send them my way! Otherwise, hopefully I'll have these updated in a month or so.
