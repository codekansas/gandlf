[vgg16.py](https://github.com/codekansas/gandlf/blob/master/examples/vgg16.py)

This example illustrates how a pre-trained model can be used to generate images. The discriminator is the VGG16 model which is available with Keras, and the discriminator weights are frozen. The generator learns to generate samples of a desired class. This isn't a pure GAN, because the discriminator model doesn't learn to discriminate adversarial examples, but the generator does learn to map a random normal distribution to generate image classes in the ImageNet dataset.

TODO: Add images sampled from this model (it is really big and takes a while to train, so they will be added whenever I have access to a good enough computer).

