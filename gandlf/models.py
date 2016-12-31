from __future__ import absolute_import
from __future__ import print_function

import IN
import copy

from keras import callbacks as keras_callbacks
from keras import layers as keras_layers
from keras import metrics as keras_metrics
from keras import models as keras_models
from keras import objectives
from keras import optimizers
from keras.engine import training
import six

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


def get_random_func(random_type, shape):
    if isinstance(random_type, six.string_types):
        random_type = random_type.lower()

        if random_type == 'normal':
            return lambda bsize: np.random.uniform(size=(bsize,) + shape[1:])
        elif random_type == 'uniform':
            return lambda bsize: np.random.uniform(size=(bsize,) + shape[1:])
        else:
            raise ValueError('Invalid name of random type: %s'
                             'Choices are "normal" or "uniform".' %
                             random_type)

    elif hasattr(random_type, '__call__'):
        return random_type

    else:
        raise ValueError('The random_type should either be a string '
                         'or a callable function, got %s' %
                         str(random_type))


def slice_X(X, start=None, stop=None):
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
    """TODO: Docstring.

    Should change the model so that the first output is treated as the "decider"
    and the other inputs are treated as "auxiliary". The model inputs should
    be special classes that are treated as random distributions when the model
    is run.
    """

    def __init__(self, generator, discriminator, name=None):
        self._check_generator_and_discriminator(generator, discriminator)

        self.generator = generator
        self.discriminator = discriminator
        self.generator_discriminator = keras_models.Model(
            input=generator.inputs,
            output=discriminator(generator.outputs),
            name='generator_around_discriminator')

        self.num_outputs = (len(generator.outputs), len(discriminator.outputs))

        # The model is treated as the generator by Keras.
        super(Model, self).__init__(generator.inputs, generator.outputs, name)

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

    def _prepare_loss_weights(self, loss_weights):
        """Performs checks on loss weights and returns a list of weights."""

        if loss_weights is None:
            return [1. for _ in range(len(self.outputs))]

        elif isinstance(loss_weights, dict):
            self._check_input_dictionary_keys(loss_weights.keys(), False)
            return [loss_weights.get(name, 1.) for name in self.output_names]

        elif isinstance(loss_weights, (list, tuple)):
            if len(loss_weights) != len(self.outputs):
                raise ValueError('When passing a list as loss_weights, '
                                 'it should have one entry per model output.'
                                 'The model has %d outputs, but you passed '
                                 'loss_weights=%s' %
                                 (len(self.outputs), str(loss_weights)))
            return loss_weights

        else:
            raise TypeError('Could not interpret loss_weights argument: %s' %
                            str(loss_weights))

    def _prepare_masks(self):
        """Computes the masks all the way from start to end."""

        masks = self.generator_discriminator.compute_mask(
            self.generator_discriminator.inputs, mask=None)

        if masks is None:
            return [None for _ in self.generator_discriminator.outputs]

        return _as_list(masks)

    def _prepare_sample_weights(self, sample_weight_mode):
        """Prepares the sample weights."""

        output_names = self.discriminator.output_names

        if isinstance(sample_weight_mode, dict):
            self._check_input_dictionary_keys(sample_weight_mode.keys(), True)

            sample_weights = []
            sample_weight_modes = []
            for name in output_names:
                if sample_weight_mode.get(name) == 'temporal':
                    sample_weights.append(K.placeholder(
                        ndim=2, name=name + '_sample_weights'))
                    sample_weight_modes.append('temporal')
                else:
                    sample_weights.append(K.placeholder(
                        ndim=1, name=name + '_sample_weights'))
                    sample_weight_modes.append(None)

        elif isinstance(sample_weight_mode, list):
            if len(sample_weight_mode) != len(output_names):
                raise ValueError('When passing a list as sample_weight_mode, '
                                 'it should have one entry per discriminator '
                                 'model output. The model has %d outputs, but '
                                 'you passed sample_weight_mode=%s' %
                                 (len(output_names)),
                                 str(sample_weight_mode))

            sample_weights = []
            sample_weight_modes = []
            for mode, name in zip(sample_weight_mode, output_names):
                if mode == 'temporal':
                    sample_weights.append(K.placeholder(
                        ndim=2, name=name + '_sample_weights'))
                    sample_weight_modes.append('temporal')
                else:
                    sample_weights.append(K.placeholder(
                        ndim=1, name=name + '_sample_weights'))
                    sample_weight_modes.append(None)

        else:
            if sample_weight_mode == 'temporal':
                sample_weights = [
                    K.placeholder(ndim=2, name=name + '_sample_weights')
                    for name in output_names
                ]
                sample_weight_modes = ['temporal' for _ in output_names]
            else:
                sample_weights = [
                    K.placeholder(ndim=1, name=name + '_sample_weights')
                    for name in output_names
                ]
                sample_weight_modes = [None for _ in output_names]

        return sample_weights, sample_weight_modes

    def _compute_loss(self, y_true, y_pred, weighted_loss, loss_weight,
                      sample_weight, mask, output_name, is_binary):
        """Computes loss for a single pair (y_true, y_pred)."""

        output_loss = weighted_loss(y_true, y_pred, sample_weight, mask)
        output_loss *= loss_weight

        # Adds the computed loss metrics.
        self.metrics_tensors.append(output_loss)
        self.metrics_names.append(output_name + '_loss')

        # Adds other metrics.
        for metric in self.metrics:
            if metric == 'accuracy' or metric == 'acc':
                if is_binary:
                    acc_fn = keras_metrics.binary_accuracy
                elif K.is_sparse(y_pred):
                    acc_fn = keras_metrics.sparse_categorical_accuracy
                else:
                    acc_fn = keras_metrics.categorical_accuracy

                self.metrics_tensors.append(acc_fn(y_true, y_pred))
                self.metrics_names.append(output_name + '_acc')

            else:
                metric_fn = keras_metrics.get(metric)
                metric_result = metric_fn(y_true, y_pred)

                if not isinstance(metric_result, dict):
                    metric_result = {
                        metric_fn.__name__: metric_result
                    }

                for name, tensor in six.iteritems(metric_result):
                    self.metrics_tensors.append(tensor)
                    self.metrics_names.append(output_name + '_' + name)

        return output_loss

    def _compute_all_losses(self, weighted_losses, loss_weights_list,
                            sample_weights, masks):
        """Computes the total loss and returns loss and placeholder tensors."""

        # Adds placeholders for the discriminator auxiliary outputs.
        auxiliary_placeholders = list()
        for i, output in enumerate(self.discriminator.outputs[1:]):
            shape = self.discriminator.internal_output_shapes[i + 1]
            name = self.discriminator.output_names[i + 1]
            auxiliary_placeholders.append(K.placeholder(
                ndim=len(shape),
                name=name + '_target',
                sparse=K.is_sparse(output),
                dtype=K.dtype(output)))

        # Add losses for the binary decision part.
        discriminator_loss = self._compute_loss(
            y_true=K.ones_like(self.discriminator.outputs[0]),
            y_pred=self.discriminator.outputs[0],
            weighted_loss=weighted_losses[0],
            loss_weight=loss_weights_list[0],
            sample_weight=sample_weights[0],
            mask=masks[0],
            output_name='discriminator',
            is_binary=True)

        generator_loss = self._compute_loss(
            y_true=K.zeros_like(self.generator_discriminator.outputs[0]),
            y_pred=self.generator_discriminator.outputs[0],
            weighted_loss=weighted_losses[0],
            loss_weight=loss_weights_list[0],
            sample_weight=sample_weights[0],
            mask=masks[0],
            output_name='generator',
            is_binary=True)

        # Adds losses for the auxiliary parts, ignoring the first output.
        auxiliary_loss = None
        for i in range(1, len(self.discriminator.outputs)):
            loss = self._compute_loss(
                y_true=auxiliary_placeholders[i - 1],
                y_pred=self.discriminator.outputs[i],
                weighted_loss=weighted_losses[i],
                loss_weight=loss_weights_list[i],
                sample_weight=sample_weights[i],
                mask=masks[i],
                output_name=self.discriminator.output_names[i],
                is_binary=False)

            if auxiliary_loss is None:
                auxiliary_loss = loss
            else:
                auxiliary_loss += loss

        # Adds generator regularization penalties / misc losses.
        for loss in self.generator.losses:
            generator_loss += loss

        # Adds discriminator regularization penalities / misc losses.
        for loss in self.discriminator.losses:
            auxiliary_loss += loss
            discriminator_loss += loss

        all_losses = (generator_loss, discriminator_loss, auxiliary_loss)
        return all_losses, auxiliary_placeholders[1:]

    def _check_input_dictionary_keys(self, names, check_all=False):
        """Checks to make sure that all the names are valid output_names.
        If check_all is True, it makes sure that all the output_names are
        included in names. The output_names are the outputs of the
        discriminator (since those are the relevant ones for training).
        """

        for name in names:
            if name not in self.discriminator.output_names:
                raise ValueError('Unknown entry in provided dictionary: "%s". '
                                 'Only expected the following keys: %s' %
                                 (name, str(self.discriminator.output_names)))

        if check_all:
            for name in self.discriminator.output_names:
                if name not in names:
                    raise ValueError('Output "%s" missing from the provided '
                                     'dictionary (only has %s)' %
                                     (name, names))

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

    def compile(self, optimizer, metrics=None, loss_weights=None,
                sample_weight_mode=None, **kwargs):
        """TODO: Docstring."""

        self.optimizer = optimizers.get(optimizer)
        self.sample_weight_mode = sample_weight_mode
        self.loss_weights = loss_weights

        loss_weights_list = self._prepare_loss_weights(loss_weights)

        # Only the first output is binary; the others are categorical.
        loss_functions = [K.binary_crossentropy]
        loss_functions += [K.categorical_crossentropy
                           for _ in range(self.num_outputs[1] - 1)]
        self.loss_functions = loss_functions

        weighted_losses = [training.weighted_objective(fn)
                           for fn in loss_functions]

        masks = self._prepare_masks()

        (sample_weights, sample_weight_modes) = self._prepare_sample_weights(
            sample_weight_mode=sample_weight_mode)
        self.sample_weights = sample_weights
        self.sample_weight_modes = sample_weight_modes

        # Prepares metrics.
        self.metrics = _as_list(metrics)
        self.metrics_names = ['loss']
        self.metrics_tensors = []

        self.all_losses, self.targets = self._compute_all_losses(
            weighted_losses, loss_weights_list, sample_weights,
            masks)

        # Gets the total loss tensor.
        self.total_loss = self.all_losses[0] + self.all_losses[1]
        if self.all_losses[2] is not None:
            self.total_loss += self.all_losses[2]

        # Functions for train, test and predict are compiled lazily when
        # required, so this passes on the keyword arguments.
        self._function_kwargs = kwargs

        # This lets the model know that it has been compiled.
        self.non_auxiliary_train_function = None
        self.auxiliary_train_function = None

        self._collected_trainable_weights = (
            self._sort_weights_by_name(self.generator.trainable_weights),
            self._sort_weights_by_name(self.discriminator.trainable_weights),
        )

    def _get_learning_phase(self):
        if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
            return [K.learning_phase()]
        else:
            return []

    def _make_non_auxiliary_train_function(self):
        """Builds the non-auxiliary train function."""

        if not hasattr(self, 'non_auxiliary_train_function'):
            raise RuntimeError('You must compile your model before using it.')

        if self.non_auxiliary_train_function is None:

            # Collects inputs to the function.
            inputs = (self.generator_discriminator.inputs +
                      self.discriminator.inputs +
                      self.sample_weights +
                      self._get_learning_phase())

            # Gets the generator updates.
            generator_updates = self.optimizer.get_updates(
                self._collected_trainable_weights[0],
                self.constraints,
                self.all_losses[0])

            # Gets the discriminator updates.
            discriminator_updates = self.optimizer.get_updates(
                self._collected_trainable_weights[1],
                self.constraints,
                self.all_losses[1])

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
            inputs = (self.generator_discriminator.inputs +
                      self.discriminator.inputs +
                      self.targets +
                      self.sample_weights +
                      self._get_learning_phase())

            # Gets the generator updates.
            generator_updates = self.optimizer.get_updates(
                self._collected_trainable_weights[0],
                self.constraints,
                self.all_losses[0])

            # Gets the discriminator updates.
            discriminator_updates = self.optimizer.get_updates(
                self._collected_trainable_weights[1],
                self.contraints,
                self.all_losses[1])

            # Gets the auxiliary updates.
            auxiliary_updates = self.optimizer.get_updates(
                self._collected_trainable_weights[1],
                self.constraints,
                self.all_losses[2])

            updates = (generator_updates + discriminator_updates +
                       auxiliary_updates + self.updates)

            self.auxiliary_train_function = K.function(
                inputs,
                [self.total_loss] + self.metrics_tensors,
                updates=updates,
                **self._function_kwargs)

    def _make_sample_function(self):
        """Builds the predict function."""

        if not hasattr(self, 'sample_function'):
            self.sample_function = None

        if self.sample_function is None:
            inputs = self.generator.inputs + self._get_learning_phase()

            kwargs = getattr(self, '_function_kwargs', {})
            self.sample_function = K.function(
                inputs,
                self.generator.outputs,
                updates=self.state_updates,
                **kwargs)

    def _standardize_input_data(self, data, names, shapes=None,
                                check_batch_dim=True,
                                exception_prefix=''):
        """Standardizes the provided input data."""

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
                             (len(data), len(names)))

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
                fixed_data.append(array)

            else:
                fixed_data.append(get_random_func(array, shape))

        return fixed_data

    def _standardize_user_data(self, x, y, sample_weight, input_names,
                               input_shapes, output_names, output_shapes,
                               class_weight=None,
                               check_batch_dim=True,
                               batch_size=None):
        """Standardizes given user data."""

        if not hasattr(self, 'optimizer'):
            raise RuntimeError('You must compile a model before training or '
                               'testing. Use `model.compile`.')

        x = self._standardize_input_data(
            x, input_names, input_shapes,
            check_batch_dim=False, exception_prefix='model input')
        y = self._standardize_input_data(
            y, output_names, output_shapes,
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
            y, self.loss_functions[1:], output_shapes)

        return x, y, sample_weights

    def _get_out_labels(self):
        """Gets deduplicated output labels."""

        out_labels = self.metrics_names
        deduped_out_labels = []
        for i, label in enumerate(out_labels):
            new_label = label
            if out_labels.count(label) > 1:
                dup_idx = out_labels[:i].count(label) + 1
                new_label += '_%d' % dup_idx
            deduped_out_labels.append(new_label)

        return deduped_out_labels

    def sample(self, x, num_samples=None, batch_size=32, verbose=0):
        """TODO: Docstring."""

        x = self._standardize_input_data(
            x, self.generator.input_names,
            self.generator.internal_input_shapes,
            check_batch_dim=False, exception_prefix='model input')

        if any(not is_numpy_array(arr) for arr in x):
            if num_samples is None:
                raise ValueError('To sample using latent vectors, '
                                 'num_samples must be specified.')
            x = [arr if is_numpy_array(arr) else arr(num_samples) for arr in x]

        ins = x + self._get_learning_phase()

        self._make_sample_function()
        f = self.sample_function
        return self._predict_loop(f, ins,
                                  batch_size=batch_size, verbose=verbose)

    def fit(self, x, y=None, batch_size=32, nb_epoch=10, verbose=1,
            callbacks=None, validation_split=0., validation_data=None,
            shuffle=True, class_weight=None, sample_weight=None,
            initial_epoch=0):
        """TODO: Docstring."""

        if validation_split or validation_data:
            raise NotImplementedError('Validation sets are not yet '
                                      'implemented for gandlf models.')

        input_names = (self.generator_discriminator.input_names +
                       self.discriminator.input_names)
        input_shapes = (self.generator_discriminator.internal_input_shapes +
                        self.discriminator.internal_input_shapes)

        if y:
            self._make_auxiliary_train_function()
            train_fn = self.auxiliary_train_function
            output_shapes = []
            output_names = []
        else:
            self._make_non_auxiliary_train_function()
            train_fn = self.non_auxiliary_train_function
            output_shapes = (self.generator_discriminator
                                 .internal_output_shapes[1:])
            output_names = (self.generator_discriminator
                                .output_names[1:])

        x, y, sample_weights = self._standardize_user_data(
            x, y, sample_weight, input_names, input_shapes, output_names,
            output_shapes, class_weight=class_weight, check_batch_dim=False,
            batch_size=batch_size)

        # Calculates the number of training samples.
        nb_train_samples = None
        for arr in x:
            if hasattr(arr, 'shape'):
                nb_train_samples = arr.shape[0]
                break

        # Adds default weights to the head.
        sample_weights = [np.ones(shape=(nb_train_samples,))] + sample_weights

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
                        ins_batch = slice_X(ins[:-1], batch_ids)
                        ins_batch += [ins[-1]]
                    else:
                        ins_batch = slice_X(ins, batch_ids)
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

    def predict(self, x, batch_size=32, verbose=0):
        raise NotImplementedError()

    def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
        raise NotImplementedError()

    def _make_test_function(self):
        raise NotImplementedError()  # TODO: Implement this properly.

    def _make_train_function(self):
        raise NotImplementedError()

    def _make_predict_function(self):
        raise NotImplementedError()
