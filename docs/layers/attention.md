[attention.py](https://github.com/codekansas/gandlf/blob/master/gandlf/layers/attention.py)

## RecurrentAttention1D

````python
gandlf.layers.RecurrentAttention1D(layer, attention, attn_activation='tanh', attn_gate_func='sigmoid', W_regularizer=None, b_regularizer, **kwargs)
````

Makes the wrapped Keras RNN `layer` pay attention to the `attention` tensor, which has shape `(batch_size, num_attn_features)`.

The updated hidden state is computed after each timestep as:

````python
trainable_params = [U_m, b_m, U_a, U_s, b_s]

# Given hidden output "h" at each timestep:
m = attn_activation(dot(h, U_m) + dot(attention, U_a) + b_m)
s = attn_gate(dot(m, U_s) + b_s)
h_new = s * h  # Element-wise weighting.
````

## RecurrentAttention2D

````python
gandlf.layers.RecurrentAttention2D(layer, attention, time_dist_activation='softmax', attn_gate_func='sigmoid', W_regularizer=None, b_regularizer=None, **kwargs)
````

Makes the wrapped Keras RNN `layer` pay attention to the `attention` tensor, which has shape `(batch_size, num_attn_timesteps, num_attn_features)`.

The updated hidden state is computed after each timestep as:

````python
trainable_params = [U_t, b_t, U_a, b_a]

# Given hidden output "h" at each timestep:
t = time_dist_activation(dot(h, U_t), b_t)
w = sum(t * attention)  # Weights each timestep by `t`.
s = attn_gate_func(dot(w, U_a) + b_a)
h_new = s * h  # Element-wise weighting.
````

