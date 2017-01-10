[core.py](https://github.com/codekansas/gandlf/blob/master/gandlf/layers/core.py)

Core layers, tricks which can help easily improve GANs in many cases.

## PermanentDropout

````python
gandlf.layers.PermanentDropout()
````

An alternative to Keras [Dropout](https://keras.io/layers/core/#dropout) which stays active during both training and testing.

## BatchSimilarity

````python
gandlf.layers.BatchSimilarity(similarity='exp_l1')
````

Calculates the minibatch similarities, a trick introduced in [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498). These similarities can be added as features for the existing layer by using a Merge layer. The layer outputs a Tensor with shape `(batch_size, num_similarities)` for 2D tensors, `(batch_size, None, num_similarities)` for 3D Tensors, and so on.

In order to make this layer linear time with respect to the batch size, instead of doing a pairwise comparison between each pair of samples in the batch, for each sample a random sample is uniformly selected with which to do pairwise comparison.

The `similarity` argument can be one of:

 - `exp_l1`: `exp(sum(abs(a - b)))`
 - `exp_l2` or `rbf`: `exp(sum(square(a - b)))`
 - `l1`: `sum(abs(a - b))`
 - `l2`: `sum(square(a - b))`
 - `cosine`: `dot(a, b) / (|| a || * || b ||)`
 - `sigmoid`: `sigmoid(dot(a, b))`
 - `euclidean`: `1 / (1 + sum(square(a - b)))`
 - `geometric`: `sigmoid(a, b) * euclidean(a, b)`
 - `arithmetic`: `(sigmoid(a, b) + euclidean(a, b)) / 2`

These implementations can be found in [`similarities.py`](https://github.com/codekansas/gandlf/blob/master/gandlf/similarities.py).

Alternatively, a function can be provided which take two tensors and returns their similarity. Multiple similarity arguments can be provided as a list or tuple. 

The following example illustrates how to merge the similarity features with the layer output:

````python
sims = gandlf.layers.BatchSimilarity(['sim1', 'sim2', etc.])(input_layer)
output_layer = keras.layers.merge([input_layer, sims], mode='concat')
````

