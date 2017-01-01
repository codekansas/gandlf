from __future__ import absolute_import
from __future__ import print_function

import copy

from keras import callbacks as keras_callbacks
from keras import models as keras_models
from keras.engine import training
import six

from gandlf import losses
import keras.backend as K
import numpy as np


def _as_list(x):
    if x is None:
        return []
    elif isinstance(x, (list, tuple, set)):
        return list(x)
    else:
        return [x]


def _as_set(x):
    if x is None:
        return set()
    elif isinstance(x, (list, tuple, set)):
        return set(x)
    else:
        return set([x])


def is_numpy_array(x):
    return type(x).__module__ == np.__name__


def get_batch(X, start=None, stop=None):
    """Like keras.engine.training.slice_X, but supports latent vectors.

    Args:
        X: Numpy array or list of Numpy arrays.
        start: integer, the start of the batch, or a list of integers, the
            indices of each sample in to use in this batch.
        stop: integer, the end of the batch (only needed if start is an
            integer).

    Returns:
        X[start:stop] if X is array-like, or [x[start:stop] for x in X]
        if X is a list. Latent vector functions will be called as appropriate.
    """

    if isinstance(X, list):
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] if is_numpy_array(x)
                    else x(len(start)) for x in X]
        else:
            return [x[start:stop] if is_numpy_array(x)
                    else x(stop - start) for x in X]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return (X[start] if is_numpy_array(x)
                    else x(len(start)))
        else:
            return (X[start:stop] if is_numpy_array(x)
                    else x(stop - start))


class Model(keras_models.Model):
    """TODO: Docstring."""

    def __init__(self, generator, discriminator, name=None):
        self._check_generator_and_discriminator(generator, discriminator)

        self.generator = generator
        self.discriminator = discriminator

        generator_discriminator = keras_models.Model(
            input=generator.inputs,
            output=discriminator(generator.outputs),
            name='generator_around_discriminator')

        # Copies the outputs of the combined model, with two of the first
        # output (since the first output should be the true / false
        # prediction).
        inputs = generator_discriminator.inputs
        outputs = (generator_discriminator.outputs[:1] +
                   generator_discriminator.outputs)

        self.num_outputs = (len(generator.outputs), len(discriminator.outputs))

        # The model is treated as the generator by Keras.
        super(Model, self).__init__(inputs, outputs, name)

        # Copies the output names from the discriminator.
        self.output_names = (['generator', 'discriminator'] +
                             self.discriminator.output_names[1:])

    def _check_generator_and_discriminator(self, generator, discriminator):
        """Validates the provided models in a user-friendly way."""

        # Checks that both are Keras models.
        if not (isinstance(generator, keras_models.Model) and
                isinstance(discriminator, keras_models.Model)):
            raise ValueError('The generator and discriminator should both '
                             'be Keras models. Got discriminator=%s, '
                             'generator=%s' % (type(discriminator),
                                               type(generator)))

        # Checks that the discriminator has at least one output.
        if len(discriminator.outputs) == 0:
            raise ValueError('The discriminator model must have at least '
                             'one output (the True / False output).')

        # Checks that only the first discriminator output is binary.
        discriminator_outputs = discriminator.inbound_nodes[0].output_shapes
        if discriminator_outputs[0][-1] != 1:
            raise ValueError('The first output of the discriminator model '
                             'should be binary (decides if the input sample '
                             'is real or fake). It actually has %d outputs.' %
                             discriminator_outputs[0][-1])

        if len(generator.outputs) != len(discriminator.inputs):
            raise ValueError('The discriminator model should have one input '
                             'per output of the generator model.')

    def _sort_weights_by_name(self, weights):
        """Sorts weights by name and returns them."""

        if not weights:
            return []

        if K.backend() == 'theano':
            key = lambda x: x.name if x.name else x.auto_name
        else:
            key = lambda x: x.name

        weights.sort(key=key)
        return weights

    def _compute_loss(self, index, masks, loss_weights):
        """Computes loss for a single output."""

        y_true = self.targets[index]
        y_pred = self.outputs[index]
        weighted_loss = training.weighted_objective(self.loss_functions[index])
        sample_weight = self.sample_weights[index]
        mask = masks[index]
        loss_weight = loss_weights[index]
        output_loss = weighted_loss(y_true, y_pred, sample_weight, mask)
        return loss_weight * output_loss

    def _compute_losses(self):
        """Computes generator and discriminator losses."""

        # Recomputes the masks (done in the parent compile method).
        masks = self.compute_mask(self.inputs, mask=None)
        if masks is None:
            masks = [None for _ in self.outputs]
        elif not isinstance(masks, list):
            masks = [masks]

        # Recomputes the loss weights list (done in the parent compile method).
        if self.loss_weights is None:
            loss_weights = [1. for _ in self.outputs]
        elif isinstance(loss_weights, dict):
            loss_weights = [loss_weights.get(name, 1.)
                            for name in self.output_names]
        else:
            loss_weights = list(loss_weights)

        # The generator loss is the first index.
        self.generator_loss = self._compute_loss(0, masks, loss_weights)

        # The discriminator loss is the second index.
        self.discriminator_loss = self._compute_loss(1, masks, loss_weights)

        # Auxiliary loss is all the other losses.
        auxiliary_loss = None
        for index in range(2, len(self.outputs)):
            index_loss = self._compute_loss(index, masks, loss_weights)
            auxiliary_loss = (index_loss if auxiliary_loss is None
                              else auxiliary_loss + index_loss)
        self.auxiliary_loss = auxiliary_loss

    def compile(self, optimizer, loss, metrics=None, loss_weights=None,
                sample_weight_mode=None, **kwargs):
        """TODO: Docstring."""

        # Call the "parent" compile method.
        super(Model, self).compile(optimizer=optimizer,
                                   loss=loss,
                                   metrics=metrics,
                                   loss_weights=loss_weights,
                                   sample_weight_mode=sample_weight_mode,
                                   **kwargs)

        # This lets the model know that it has been compiled.
        self.non_auxiliary_train_function = None
        self.auxiliary_train_function = None

        # Computes the generator and discriminator losses.
        self._compute_losses()

        # Separates the generator and discriminator weights.
        self._collected_trainable_weights = (
            self._sort_weights_by_name(self.generator.trainable_weights),
            self._sort_weights_by_name(self.discriminator.trainable_weights),
        )

    def _get_learning_phase(self):
        if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
            return [K.learning_phase()]
        else:
            return []

    def _get_clean_updates(self, params, loss):
        """Uses optimizer to get weight updates, removing unconnected parts."""

        grads = self.optimizer.get_gradients(loss, params)
        new_params = [param
                      for param, grad in zip(params, grads)
                      if grad is not None]

        return self.optimizer.get_updates(new_params, self.constraints, loss)

    def _make_non_auxiliary_train_function(self):
        """Builds the non-auxiliary train function."""

        if not hasattr(self, 'non_auxiliary_train_function'):
            raise RuntimeError('You must compile your model before using it.')

        if self.non_auxiliary_train_function is None:

            # Collects inputs to the function.
            inputs = (self.inputs +
                      self.discriminator.inputs +
                      self.targets +
                      self.sample_weights +
                      self._get_learning_phase())

            # Gets the generator updates.
            generator_updates = self._get_clean_updates(
                self._collected_trainable_weights[0],
                self.generator_loss)

            # Gets the discriminator updates.
            discriminator_updates = self._get_clean_updates(
                self._collected_trainable_weights[1],
                self.discriminator_loss)

            updates = generator_updates + discriminator_updates + self.updates

            # Returns loss and metrics. Updates weights at each call.
            self.non_auxiliary_train_function = K.function(
                inputs,
                [self.total_loss] + self.metrics_tensors,
                updates=updates,
                **self._function_kwargs)

    def _make_auxiliary_train_function(self):
        """Builds the auxiliary train function."""

        if not hasattr(self, 'auxiliary_train_function'):
            raise RuntimeError('You must compile your model before using it.')

        if self.auxiliary_train_function is None:

            # Collects inputs to the function.
            inputs = (self.inputs +
                      self.discriminator.inputs +
                      self.targets +
                      self.sample_weights +
                      self._get_learning_phase())

            # Gets the generator updates.
            generator_updates = self._get_clean_updates(
                self._collected_trainable_weights[0],
                self.generator_loss)

            # Gets the discriminator updates.
            discriminator_updates = self._get_clean_updates(
                self._collected_trainable_weights[1],
                self.discriminator_loss + self.auxiliary_loss)

            updates = (generator_updates + discriminator_updates +
                       self.updates)

            self.auxiliary_train_function = K.function(
                inputs,
                [self.total_loss] + self.metrics_tensors,
                updates=updates,
                **self._function_kwargs)

    def _standardize_input_data(self, data, names, shapes, nb_train_samples,
                                check_batch_dim=True, exception_prefix=''):
        """Standardizes the provided input data."""

        fixed_data = []

        # Make arrays at least 2D
        for array, shape, name in zip(data, shapes, names):
            if is_numpy_array(array):

                dims_equal = [a == b for a, b in zip(array.shape, shape)]
                if not check_batch_dim:
                    dims_equal = dims_equal[1:]

                if len(array.shape) != len(shape) or not all(dims_equal):
                    raise ValueError('Error when checking %s: expected %s '
                                     'to have shape %s, but got an array '
                                     'with shape %s' %
                                     (exception_prefix, name, str(shape),
                                      str(array.shape)))

            elif isinstance(array, six.string_types):
                array = array.lower()
                shape_no_b = shape[1:]

                if array == 'normal':
                    array = lambda b: np.random.uniform(size=(b,) + shape_no_b)
                elif array == 'uniform':
                    array = lambda b: np.random.uniform(size=(b,) + shape_no_b)
                elif array == 'ones' or array == 'one' or array == '1':
                    array = np.ones(shape=(nb_train_samples,) + shape[1:])
                elif array == 'zeros' or array == 'zero' or array == '0':
                    array = np.zeros(shape=(nb_train_samples,) + shape[1:])
                else:
                    raise ValueError('Error when checking %s:'
                                     'Invalid name of random type: %s'
                                     'Choices are "normal" or "uniform".' %
                                     (exception_prefix, array))

            elif hasattr(array, '__call__'):
                called = array(1)
                dims_equal = [a == b for a, b in zip(called.shape,
                                                     [1] + _as_list(shape)[1:])]

                if len(called.shape) != len(shape) and not all(dims_equal):
                    raise ValueError('Error when checking %s: The provided '
                                     'function returned an invalid shape '
                                     '(expected %s, got %s).' %
                                     (exception_prefix, str(shape),
                                      str(called.shape)))

            else:
                raise ValueError('The argument should either be a string '
                                 'or a callable function, got %s' %
                                 str(array))

            fixed_data.append(array)

        return fixed_data

    def _convert_input_to_list(self, data, names):

        if isinstance(data, dict):
            for key in data.keys():
                if key not in names:
                    raise ValueError('Invalid key provided for input data: '
                                     '"%s". Should be one of %s' %
                                     (str(key), str(names)))
            data = [data.get(name) for name in names]

        data = _as_list(data)

        if len(data) != len(names):
            raise ValueError('Incorrect number of inputs to the model '
                             '(expected %d, got %d)' %
                             (len(names), len(data)))

        return data

    def _standardize_user_data(self, x, y, sample_weight, class_weight,
                               input_names, input_shapes, output_names,
                               output_shapes, check_batch_dim, batch_size):
        """Standardizes given user data."""

        if not hasattr(self, 'optimizer'):
            raise RuntimeError('You must compile a model before training or '
                               'testing. Use `model.compile`.')

        x = self._convert_input_to_list(x, input_names)
        y = self._convert_input_to_list(y, output_names)

        # Calculates the number of training samples.
        nb_train_samples = None
        for arr in x + y:
            if is_numpy_array(arr):
                nb_train_samples = arr.shape[0]
                break
        else:
            raise ValueError('At least one of the fed inputs must be a Numpy '
                             'array (usually the real training data).')

        x = self._standardize_input_data(
            x, input_names, input_shapes, nb_train_samples,
            check_batch_dim=False, exception_prefix='model input')
        y = self._standardize_input_data(
            y, output_names, output_shapes, nb_train_samples,
            check_batch_dim=False, exception_prefix='model output')
        sample_weights = training.standardize_sample_weights(
            sample_weight, output_names)
        class_weights = training.standardize_class_weights(
            class_weight, output_names)
        sample_weights = [training.standardize_weights(ref, sw, cw, mode)
                          for ref, sw, cw, mode
                          in zip(y, sample_weights, class_weights,
                                 self.sample_weight_modes)]

        training.check_loss_and_target_compatibility(
            y, self.loss_functions, output_shapes)

        return x, y, sample_weights, nb_train_samples

    def _get_out_labels(self):
        """Gets deduplicated output labels."""

        out_labels = []

        # Fixes a bug where accuracy names weren't matching loss names.
        names = (name[:-5] for name in self.metrics_names
                 if name.endswith('_loss'))
        for i, name in enumerate(self.metrics_names):
            if name.endswith('_acc'):
                name = next(names, name[:-4]) + '_acc'
            out_labels.append(name)

        # Deduplicates the labels.
        deduped_out_labels = []
        for i, label in enumerate(out_labels):
            new_label = label
            if out_labels.count(label) > 1:
                dup_idx = out_labels[:i].count(label) + 1
                new_label += '_%d' % dup_idx
            deduped_out_labels.append(new_label)

        return deduped_out_labels

    def _make_sample_function(self):
        """Instantiates the sample function."""

        if not hasattr(self, 'sample_function'):
            self.sample_function = None

        if self.sample_function is None:
            inputs = self.inputs + self._get_learning_phase()
            outputs = self.generator.outputs
            kwargs = getattr(self, '_function_kwargs', {})
            self.sample_function = K.function(inputs, outputs,
                                              updates=self.state_updates,
                                              **kwargs)

    def sample(self, x, num_samples=None, batch_size=32, verbose=0):
        """TODO: Docstring."""

        input_names = self.input_names
        input_shapes = self.internal_input_shapes

        x = self._convert_input_to_list(x, input_names)

        # Calculates the number of training samples.
        if num_samples is None:
            for arr in x:
                if is_numpy_array(arr):
                    num_samples = arr.shape[0]
                    break
            else:
                raise ValueError('None of the model inputs have an explicit '
                                 'batch size, so it must be specified in '
                                 'num_samples.')

        x = self._standardize_input_data(
            x, input_names, input_shapes, num_samples,
            check_batch_dim=False, exception_prefix='model input')

        # Updates callable parts.
        x = [i(num_samples) if hasattr(i, '__call__') else i for i in x]

        if self._get_learning_phase():
            ins = x + [0.]
        else:
            ins = x

        self._make_sample_function()
        f = self.sample_function
        return self._predict_loop(f, ins, batch_size=batch_size,
                                  verbose=verbose)

    def _make_predict_function(self):
        """Instantiates the predict function."""

        if not hasattr(self, 'predict_function'):
            self.predict_function = None

        if self.predict_function is None:
            inputs = self.discriminator.inputs + self._get_learning_phase()
            outputs = self.discriminator.outputs
            kwargs = getattr(self, '_function_kwargs', {})
            self.predict_function = K.function(inputs, outputs,
                                               updates=self.state_updates,
                                               **kwargs)

    def predict(self, x, num_samples=None, batch_size=32, verbose=0):
        """TODO: Docstring."""

        input_names = self.discriminator.input_names
        input_shapes = self.discriminator.internal_input_shapes

        x = self._convert_input_to_list(x, input_names)

        # Calculates the number of training samples.
        if num_samples is None:
            for arr in x:
                if is_numpy_array(arr):
                    num_samples = arr.shape[0]
                    break
            else:
                raise ValueError('None of the model inputs have an explicit '
                                 'batch size, so it must be specified in '
                                 'num_samples.')

        x = self._standardize_input_data(
            x, input_names, input_shapes, num_samples,
            check_batch_dim=False, exception_prefix='model input')

        # Updates callable parts.
        x = [i(num_samples) if hasattr(i, '__call__') else i for i in x]

        if self._get_learning_phase():
            ins = x + [0.]
        else:
            ins = x

        self._make_predict_function()
        f = self.predict_function
        return self._predict_loop(f, ins, batch_size=batch_size,
                                  verbose=verbose)

    def fit(self, x, y, train_auxiliary=False, batch_size=32, nb_epoch=10,
            verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None,
            sample_weight=None, initial_epoch=0):
        """TODO: Docstring."""

        if validation_split or validation_data:
            raise NotImplementedError('Validation sets are not yet '
                                      'implemented for gandlf models.')

        input_names = (self.input_names +
                       self.discriminator.input_names)
        input_shapes = (self.internal_input_shapes +
                        self.discriminator.internal_input_shapes)
        output_names = self.output_names
        output_shapes = self.internal_output_shapes

        x, y, sample_weights, nb_train_samples = self._standardize_user_data(
            x, y, sample_weight, class_weight, input_names, input_shapes,
            output_names, output_shapes, False, batch_size)
        if train_auxiliary:
            self._make_auxiliary_train_function()
            train_fn = self.auxiliary_train_function
        else:
            self._make_non_auxiliary_train_function()
            train_fn = self.non_auxiliary_train_function

        if self._get_learning_phase():
            ins = x + y + sample_weights + [1.]
        else:
            ins = x + y + sample_weights

        out_labels = self._get_out_labels()
        callback_metrics = copy.copy(out_labels)

        return self._fit_loop(train_fn, ins, nb_train_samples,
                              out_labels=out_labels,
                              batch_size=batch_size, nb_epoch=nb_epoch,
                              verbose=verbose, callbacks=callbacks,
                              shuffle=shuffle,
                              callback_metrics=callback_metrics,
                              initial_epoch=initial_epoch)

    def _fit_loop(self, f, ins, nb_train_samples, out_labels=None,
                  batch_size=32, nb_epoch=100, verbose=1, callbacks=None,
                  shuffle=True, callback_metrics=None, initial_epoch=0):
        """TODO: Docstring."""

        index_array = np.arange(nb_train_samples)

        self.history = keras_callbacks.History()
        callbacks = [keras_callbacks.BaseLogger()] + (callbacks or [])
        callbacks += [self.history]
        if verbose:
            callbacks += [keras_callbacks.ProgbarLogger()]
        callbacks = keras_callbacks.CallbackList(callbacks)

        out_labels = out_labels or []

        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self

        callbacks._set_model(callback_model)
        callbacks._set_params({
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'nb_sample': nb_train_samples,
            'verbose': verbose,
            'do_validation': False,
            'metrics': callback_metrics or [],
        })

        callbacks.on_train_begin()
        callback_model.stop_training = False
        self.validation_data = None

        for epoch in range(initial_epoch, nb_epoch):
            callbacks.on_epoch_begin(epoch)
            if shuffle == 'batch':
                index_array = training.batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = training.make_batches(nb_train_samples, batch_size)
            epoch_logs = {}
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]

                try:
                    if isinstance(ins[-1], float):
                        ins_batch = get_batch(ins[:-1], batch_ids)
                        ins_batch += [ins[-1]]
                    else:
                        ins_batch = get_batch(ins, batch_ids)
                except TypeError:
                    raise TypeError('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')

                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = f(ins_batch)
                if not isinstance(outs, list):
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if callback_model.stop_training:
                break
        callbacks.on_train_end()
        return self.history

    def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
        raise NotImplementedError()  # TODO: Implement this properly.

    def _make_test_function(self):
        raise NotImplementedError()  # TODO: Implement this properly.

    def _make_train_function(self):
        raise NotImplementedError()  # TODO: Fix this (some dependencies).