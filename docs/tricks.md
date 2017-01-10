## General

A list of useful GAN hacks is available [here](https://github.com/soumith/ganhacks#5-avoid-sparse-gradients-relu-maxpool). Most of the tricks listed below are taken from that list, with references for how to implement them in Gandlf.

## Data

  - Normalize the data to `[-1, 1]`
    - This data can then be approximated using the `tanh` function
  - Use Batch Normalization (implemented in Keras as [BatchNormalization](https://keras.io/layers/normalization/#batchnormalization))

## Model

  - Use `keras.layers.LeakyReLU` instead of the ReLU activation function
  - If available, use labeled data (Auxiliary Classifier GAN or Conditional GAN). These can be integrated as inputs and outputs or as [attention](layers/attention.md) components.
  - Use dropout in both the train and test phase: This is implemented as [`gandlf.layers.PermanentDropout`](layers/core.md#permanentdropout)
  - Use minibatch discrimination: Compare the intrabatch similarities and add add it as a feature for the discriminator. This helps the model avoid generating only type of output. This is implemented as [`gandlf.layers.BatchSimilarity`](layers/core.md#batchsimilarity).

## Training

  - Modified loss function: Instead of training the Generator to minimize `log(1-D)`, train it to minimize `-log(D)`. This is implemented as [`gandlf.losses.negative_binary_crossentropy`](losses.md#negative-binary-crossentropy)
  - For the latent vector(s), sample from a normal distribution instead of a uniform distribution. This is implemented as `model.fit(inputs=['normal', ...)`
  - Reinforcement learning stability tricks: Not yet implemented
  - Use the Adam optimizer for the discriminator, and the SGD optimizer for the generator. This can be done as `model = gandlf.Model(optimizer=['adam', 'sgd'], ...)`
  - Don't let the discriminator saturate!
  - Adapt the generator and discriminator updates so that when the generator loss is high relative to the discriminator, its learning rate is also higher. This is implemented as [`gandlf.callbacks.AdaptiveLearningRate`](callbacks.md#adaptivelearningrate)

