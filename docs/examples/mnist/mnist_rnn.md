<a href="https://github.com/codekansas/gandlf/blob/master/examples/mnist_rnn_gan.py" class="icon icon-github"> mnist_rnn_gan.py</a>

This example illustrates:

 - Using RNNs in your GAN architecture
 - Using attention components to direct GAN learning (with the recurrent attention wrappers, `gandlf.layers.RecurrentAttention1D` and `gandlf.layers.RecurrentAttention2D`)

Samples from this model are a work in progress (waiting on access to a GPU to run it for a longer period). This example is more of a proof-of-concept for how to use RNNs in GANs (in particular, using the attention components to make them supervised).
