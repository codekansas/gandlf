import keras
import keras.backend as K


class MaxValue(keras.constraints.Constraint):
    """MaxValue weight constraint.
    Constrains the weights incident to each hidden unit
    to have a value less than or equal to a desired value.
    Useful for implementing Wasserstein GANs.

    Args:
        m: the maximum norm for the incoming weights.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Convolution2D` layer with `dim_ordering="tf"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.

    Reference: https://arxiv.org/abs/1701.07875
    """

    def __init__(self, c=0.01, axis=0):
        self.c = c
        self.axis = axis

    def __call__(self, p):
        return K.tanh(p) * self.c

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c,
                'axis': self.axis}


# aliases
maxvalue = MaxValue
