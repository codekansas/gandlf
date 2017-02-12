<a href="https://github.com/codekansas/gandlf/blob/master/examples/mnist_gan.py" class="icon icon-github"> mnist_gan.py</a>

This example illustrates how to use a Gandlf model to generate MNIST digits. The model can be run in supervised mode (where the discriminator and generator know the desired class labels) or unsupervised mode (which is a more pure GAN implementation).

This example is a Gandlf implementation of the Keras MNIST ACGAN example, which can be found [here](https://github.com/fchollet/keras/blob/master/examples/mnist_acgan.py). One important distinction is that Gandlf runs the generator and discriminator updates in parallel rather than sequentially; this can't be done in Keras normally.

The GIF below illustrates samples from different latent vectors for each number. As the latent vector is smoothly interpolated, the image smoothly changes.

[![MNIST GAN](/resources/same_digit.gif)](/resources/same_digit.gif)

The next GIF illustrates smoothly interpolating between different digits.

[![MNIST GAN Different Digits](/resources/cycling_digits.gif)](/resources/cycling_digits.gif)

In addition to the convolutional model, a simple feed-forward model can be used for the discriminator and generator which can be trained much more quickly (feasible to run it on a laptop). The samples below show randomly sampled generated images from the lite model, which took about 100 seconds per epoch on a Macbook Pro.

[![MNIST GAN Lite](/resources/mnist_gan_lite.png)](/resources/mnist_gan_lite.png)

