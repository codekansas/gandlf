<a href="https://github.com/codekansas/gandlf/blob/master/examples/reversing_gan.py" class="icon icon-github"> upsample_gan.py</a>

This example shows how to use gradient descent to map an image back to the latent vector that it corresponds to. This approach can be done on an untrained model or a trained model. The GIF on the left shows the gradient descent process, as the latent vector moves in the direction that causes the generator to produce the target image on the right.

[![Reversing before training](/resources/reversing/before_training_generator.gif)](/resources/reversing/before_training_generator.gif)

[![Reversing after training](/resources/reversing/after_training_generator.gif)](/resources/reversing/before_training_generator.gif)

This could be an interesting way to benchmark the GAN performance; given a real image, map it backwards to its latent vector, and see how far the latent vector is from a normal vector.

