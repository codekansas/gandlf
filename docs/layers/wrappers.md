[wrappers.py](https://github.com/codekansas/gandlf/blob/master/gandlf/layers/wrappers.py)

## Residual

````python
gandlf.layers.Residual(layer, merge_mode='sum')
````

Applies a residual to any Keras layer or model, so long as it's inputs are the same dimension as its outputs. Useful for implementing residual architectures.

The provided `layer` has to have the same input and output dimensions. Given an input `x`, the output is:

````python
output = merge_mode(x, layer(x))
````

`merge_mode` can be a string like for [Merge](https://keras.io/layers/core/#merge) or a `Merge` layer itself.

