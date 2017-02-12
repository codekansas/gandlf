<a href="https://github.com/codekansas/gandlf/blob/master/examples/upsample_gan.py" class="icon icon-github"> upsample_gan.py</a>

This is an implementation of the [neural-enhance](https://github.com/alexjc/neural-enhance) project. The model can be trained on MNIST or CIFAR data. The MNIST digits below illustrate the model. The first two digits the "real" data, where the first is a downsampled version of the second. The remaining digits are the results of iteratively applying the model, upsampling the original digit.

[![MNIST GAN Upsampled](/resources/upsampling/resolved_five.png)](/resources/upsampling/resolved_five.png)

The same process can be done on CIFAR images:

[![CIFAR GAN Upsampled](/resources/upsampling/resolved_bear.png)](/resources/upsampling/resolved_bear.png)

# Model architecture

The generator and discriminator are both convolutional neural networks. I played around with different sized models. The MNIST and CIFAR models were trained for 10 epochs, which took about 10 minutes on a Titan X GPU. The generator architecture first had an upsampling layer, then several convolutional layers.

The upsampling part took advantage of the fact that the filters in the convolutional layer are dimension-invariant, so the same filters can be applied at double the resolution. To do sampling, the model is reconstructed at twice the resolution, then the trained generator weights are loaded in order to produce a new image.

# Related links

 - [neural-enhance](https://github.com/alexjc/neural-enhance): A Lasagne implementation of this upsampling idea
 - [Generating Large Images from Latent Vectors](http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/): Train a model to take (X, Y) coordinates and output a pixel intensity, then interpolate between points to get high-resolution images
 - [Pixel Recursive Super Resolution](https://arxiv.org/pdf/1702.00783.pdf): Train a PixelCNN model to un-pixelate faces

