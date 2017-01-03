[core.py](https://github.com/codekansas/gandlf/blob/master/gandlf/layers/core.py)

Core layers, tricks which can help easily improve GANs in many cases.

## PermanentDropout

````python
gandlf.layers.PermanentDropout()
````

An alternative to Keras [Dropout](https://keras.io/layers/core/#dropout) which stays active during both training and testing.

