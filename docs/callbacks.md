## AdaptiveLearningRate

````python
gandlf.callbacks.AdaptiveLearningRate(discriminator_lr, generator_lr)
````

Adapts the learning rate on each batch according to the model's loss, where `discriminator_lr` and `generator_lr` are the maximum learning rates for the discriminator and generator models.

On each batch, the learning rate for the generator and discriminator are adapted according to the losses of each:

````python
batch_dis_lr = discriminator_lr * (discriminator_loss / total_loss)
batch_gen_lr = generator_lr * (generator_loss / total_loss)
````

