import keras
import keras.backend as K


class RecurrentAttention1D(keras.layers.Wrapper):
    """Makes a recurrent layer pay attention to an attention tensor.

    This implementation takes an attention tensor with shape (batch_size,
    num_features). On each recurrent step, the hidden state is weighted by the
    a vector `s`, which is given by:

        m = attn_activation(dot(h, U_m) + dot(attention, U_a) + b_m)
        s = attn_gate(dot(m, U_s) + b_s)
        h_new = s * h

    Args:
        layer: Keras Recurrent layer, the layer to apply the attention to.
        attention: Keras tensor with shape (batch_size, num_features). For
            example, this could the output of a Dense or GlobalMaxPooling1D
            layer.
        attn_activation: activation function. Can be the name of an existing
            function (str) or another function. See Keras
            [activations](https://keras.io/activations/) and the above
            equation for `m`.
        attn_gate_func: activation function. Can be the name of an existing
            function (str) or another function. See Keras
            [activations](https://keras.io/activations/) and the above
            equation for `s`.
        W_regularizer: instance of Keras WeightRegularizer. See Keras
            [regularizers](https://keras.io/regularizers/). Applied to all of
            the weight matrices.
        b_regularizer: instance of Keras WeightRegularizer. See Keras
            [regularizers](https://keras.io/regularizers/). Applied to all of
            the bias vectors.
    """

    def __init__(self, layer, attention, attn_activation='tanh',
                 attn_gate_func='sigmoid', W_regularizer=None,
                 b_regularizer=None, **kwargs):

        if not isinstance(layer, keras.layers.Recurrent):
            raise ValueError('The RecurrentAttention wrapper only works on '
                             'recurrent layers.')

        # Should know this so that we can handle multiple hidden states.
        self._wraps_lstm = isinstance(layer, keras.layers.LSTM)

        if not hasattr(attention, '_keras_shape'):
            raise ValueError('Attention should be a Keras tensor.')

        if len(K.int_shape(attention)) != 2:
            raise ValueError('The attention input for RecurrentAttention2D '
                             'should be a tensor with shape (batch_size, '
                             'num_features). Got shape=%s.' %
                             str(K.int_shape(attention)))

        self.supports_masking = True
        self.attention = attention
        self.attn_activation = keras.activations.get(attn_activation)
        self.attn_gate_func = keras.activations.get(attn_gate_func)

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        super(RecurrentAttention1D, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert input_shape >= 3
        self.input_spec = [keras.engine.InputSpec(shape=input_shape)]

        # Builds the wrapped layer.
        if not self.layer.built:
            self.layer.build(input_shape)

        super(RecurrentAttention1D, self).build()

        attention_dim = self.attention._keras_shape[1]
        output_dim = self.layer.output_dim

        self.attn_U_a = self.add_weight((attention_dim, output_dim),
                                   initializer=self.layer.inner_init,
                                   name='{}_attn_U_a'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.attn_U_m = self.add_weight((output_dim, output_dim),
                                   initializer=self.layer.inner_init,
                                   name='{}_attn_U_m'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.attn_b_m = self.add_weight((output_dim,),
                                   initializer='zero',
                                   name='{}_attn_b_m'.format(self.name),
                                   regularizer=self.b_regularizer)

        self.attn_U_s = self.add_weight((output_dim, output_dim),
                                   initializer=self.layer.inner_init,
                                   name='{}_attn_U_s'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.attn_b_s = self.add_weight((output_dim,),
                                   initializer='zero',
                                   name='{}_attn_b_s'.format(self.name),
                                   regularizer=self.b_regularizer)

        self.trainable_weights = [self.attn_U_a, self.attn_U_m, self.attn_b_m,
                                  self.attn_U_s, self.attn_b_s]
        self.trainable_weights += self.layer.trainable_weights

    def reset_states(self):
        self.layer.reset_states()

    def get_constants(self, x):
        constants = self.layer.get_constants(x)
        constants.append(K.dot(self.attention, self.attn_U_a))
        return constants

    def _compute_attention(self, h, attention):
        m = self.attn_activation(K.dot(h, self.attn_U_m) + attention +
                                 self.attn_b_m)
        s = self.attn_gate_func(K.dot(m, self.attn_U_s) + self.attn_b_s)
        return s

    def step(self, x, states):
        if self._wraps_lstm:  # If the recurrent layer is an LSTM.
            h, [_, c] = self.layer.step(x, states)
            h *= self._compute_attention(h, states[4])
            return h, [h, c]

        else:  # All other RNN types.
            h, [h] = self.layer.step(x, states)
            h *= self._compute_attention(h, states[3])
            return h, [h, c]

    def get_output_shape_for(self, input_shape):
        if self.layer.return_sequences:
            return (input_shape[0], input_shape[1], self.layer.output_dim)
        else:
            return (input_shape[0], self.layer.output_dim)

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        if self.layer.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')

        initial_states = (self.layer.states if self.layer.stateful else
                          self.layer.get_initial_states(x))
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)

        last_output, outputs, states = K.rnn(
            self.step, preprocessed_input, initial_states,
            go_backwards=self.layer.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.layer.unroll,
            input_length=input_shape[1])

        if self.layer.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.layer.states[i], states[i]))
            self.add_update(updates, x)

        return outputs if self.layer.return_sequences else last_output

    def get_config(self):
        raise NotImplementedError('Saving attention components is not '
                                  'supported yet.')


class RecurrentAttention2D(keras.layers.Wrapper):
    """Makes a recurrent layer pay attention to an attention tensor.

    This implementation takes an attention tensor with shape (batch_size,
    num_input_timesteps, num_features). On each recurrent step, the hidden
    state is weighted by the a vector `s`, which is computed as a weighted sum
    of the input vectors as follows:

        t = time_dist_activation(dot(h, U_t) + b_t)
        w = sum(t * attention)
        s = attn_gate_func(dot(w, U_a) + b_a)
        h_new = s * h

    Generally, on each timestep, the hidden state is used to compute a weight
    distribution over each timestep in the attention tensor. This is used to
    get a weighted sum, which has shape (batch_size, num_attn_feats). This is
    linearly transformed to get `s`, which weights the hidden state.

    Args:
        layer: Keras Recurrent layer, the layer to apply the attention to.
        attention: Keras tensor with shape (batch_size, num_timesteps,
            num_features). For example, this could the output of a Dense or
            GlobalMaxPooling1D layer.
        time_dist_activation: activation function. Can be the name of an
            existing function (str) or another function. See Keras
            [activations](https://keras.io/activations/). A softmax function
            intuitively means "determine how important each time input is".
        attn_gate_func: activation function. Can be the name of an existing
            function (str) or another function. See Keras
            [activations](https://keras.io/activations/) and the equations.
        W_regularizer: instance of Keras WeightRegularizer. See Keras
            [regularizers](https://keras.io/regularizers/). Applied to all of
            the weight matrices.
        b_regularizer: instance of Keras WeightRegularizer. See Keras
            [regularizers](https://keras.io/regularizers/). Applied to all of
            the bias vectors.
    """

    def __init__(self, layer, attention, time_dist_activation='softmax',
                 attn_gate_func='sigmoid', W_regularizer=None,
                 b_regularizer=None, **kwargs):

        if not isinstance(layer, keras.layers.Recurrent):
            raise ValueError('The RecurrentAttention wrapper only works on '
                             'recurrent layers.')

        # Should know this so that we can handle multiple hidden states.
        self._wraps_lstm = isinstance(layer, keras.layers.LSTM)

        if not hasattr(attention, '_keras_shape'):
            raise ValueError('Attention should be a Keras tensor.')

        if len(K.int_shape(attention)) != 3:
            raise ValueError('The attention input for RecurrentAttention2D '
                             'should be a tensor with shape (batch_size, '
                             'num_timesteps, num_features). Got shape=%s.' %
                             str(K.int_shape(attention)))

        self.supports_masking = True
        self.attention = attention

        self.time_dist_activation = keras.activations.get(time_dist_activation)
        self.attn_gate_func = keras.activations.get(attn_gate_func)

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        super(RecurrentAttention2D, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert input_shape >= 3
        self.input_spec = [keras.engine.InputSpec(shape=input_shape)]

        # Builds the wrapped layer.
        if not self.layer.built:
            self.layer.build(input_shape)

        super(RecurrentAttention2D, self).build()

        num_attn_timesteps, num_attn_feats = K.int_shape(self.attention)[1:]
        output_dim = self.layer.output_dim

        self.attn_U_t = self.add_weight((output_dim, num_attn_timesteps),
                                   initializer=self.layer.inner_init,
                                   name='{}_attn_U_t'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.attn_b_t = self.add_weight((num_attn_timesteps,),
                                   initializer='zero',
                                   name='{}_attn_b_t'.format(self.name),
                                   regularizer=self.b_regularizer)

        self.attn_U_a = self.add_weight((num_attn_feats, output_dim),
                                   initializer=self.layer.inner_init,
                                   name='{}_attn_U_a'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.attn_b_a = self.add_weight((output_dim,),
                                   initializer='zero',
                                   name='{}_attn_b_a'.format(self.name),
                                   regularizer=self.b_regularizer)

        self.trainable_weights = [self.attn_U_t, self.attn_b_t,
                                  self.attn_U_a, self.attn_b_a]

    def reset_states(self):
        self.layer.reset_states()

    def _compute_attention(self, h):
        time_weights = K.expand_dims(K.dot(h, self.attn_U_t) + self.attn_b_t,
                                     dim=-1)
        time_weights = self.time_dist_activation(time_weights)
        weighted_sum = K.sum(time_weights * self.attention, axis=1)
        attn_vec = K.dot(weighted_sum, self.attn_U_a) + self.attn_b_a
        return self.attn_gate_func(attn_vec)

    def step(self, x, states):
        if self._wraps_lstm:  # If the recurrent layer is an LSTM.
            h, [_, c] = self.layer.step(x, states)
            h *= self._compute_attention(h)
            return h, [h, c]

        else:  # All other RNN types.
            h, [h] = self.layer.step(x, states)
            h *= self._compute_attention(h)
            return h, [h, c]

    def get_output_shape_for(self, input_shape):
        if self.layer.return_sequences:
            return (input_shape[0], input_shape[1], self.layer.output_dim)
        else:
            return (input_shape[0], self.layer.output_dim)

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        if self.layer.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')

        initial_states = (self.layer.states if self.layer.stateful else
                          self.layer.get_initial_states(x))
        constants = self.layer.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)

        last_output, outputs, states = K.rnn(
            self.step, preprocessed_input, initial_states,
            go_backwards=self.layer.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.layer.unroll,
            input_length=input_shape[1])

        if self.layer.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.layer.states[i], states[i]))
            self.add_update(updates, x)

        return outputs if self.layer.return_sequences else last_output

    def get_config(self):
        raise NotImplementedError('Saving attention components is not '
                                  'supported yet.')
