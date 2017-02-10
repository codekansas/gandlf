from __future__ import absolute_import
from __future__ import print_function

from . import losses as gandlf_losses

import copy
import inspect
import itertools
import json
import sys

import keras
from keras.engine import training as keras_training

import six

import keras.backend as K
import numpy as np


def _as_list(x):
    """Converts an object to a list."""

    if x is None:
        return []
    elif isinstance(x, (list, tuple, set)):
        return list(x)
    else:
        return [x]


def is_numpy_array(x):
    """Returns True if an object is a Numpy array."""

    return type(x).__module__ == np.__name__


def _get_callable(callable_type, shape_no_b):
    """Gets callable function."""

    if callable_type == 'normal':
        return lambda b: np.random.normal(size=(b,) + shape_no_b)
    elif callable_type == 'uniform':
        return lambda b: np.random.uniform(size=(b,) + shape_no_b)
    elif callable_type == 'ones' or callable_type == '1':
        return lambda b: np.ones(shape=(b,) + shape_no_b)
    elif callable_type == 'zeros' or callable_type == '0':
        return lambda b: np.zeros(shape=(b,) + shape_no_b)
    elif callable_type == 'ohe' or callable_type == 'onehot':
        def _get_ohe(b):
            nb_classes = shape_no_b[-1]
            idx = np.random.randint(0, nb_classes, tuple(b) + shape_no_b[:-1])
            return np.eye(nb_classes)[idx]

        return _get_ohe
    elif isinstance(callable_type, float):
        return lambda b: np.ones(shape=(b,) + shape_no_b) * callable_type
    else:
        raise ValueError('Error when checking %s:'
                         'Invalid data type string: %s'
                         'Choices are "normal", "uniform",'
                         '"ones" or "zeros".' %
                         (exception_prefix, array))


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
            return (X[start] if is_numpy_array(X)
                    else X(len(start)))
        else:
            return (X[start:stop] if is_numpy_array(X)
                    else X(stop - start))


def save_model(model, filepath, overwrite=True):

    def get_json_type(obj):
        if hasattr(obj, 'get_config'):
            return {'class_name': obj.__class__.__name__,
                    'config': obj.get_config()}

        if type(obj).__module__ == np.__name__:
            return obj.item()

        if callable(obj) or type(obj).__name__ == type.__name__:
            return obj.__name__

        raise TypeError('Not JSON Serializable:', obj)

    import h5py
    from keras import __version__ as keras_version

    if not overwrite and os.path.isfile(filepath):
        proceed = keras.models.ask_to_proceed_with_overwrite(filepath)
        if not proceed:
            return

    f = h5py.File(filepath, 'w')
    f.attrs['keras_version'] = str(keras_version).encode('utf8')
    f.attrs['generator_config'] = json.dumps({
        'class_name': model.discriminator.__class__.__name__,
        'config': model.generator.get_config(),
    }, default=get_json_type).encode('utf8')
    f.attrs['discriminator_config'] = json.dumps({
        'class_name': model.discriminator.__class__.__name__,
        'config': model.discriminator.get_config(),
    }, default=get_json_type).encode('utf8')

    generator_weights_group = f.create_group('generator_weights')
    discriminator_weights_group = f.create_group('discriminator_weights')
    model.generator.save_weights_to_hdf5_group(generator_weights_group)
    model.discriminator.save_weights_to_hdf5_group(discriminator_weights_group)

    f.flush()
    f.close()


def load_model(filepath, custom_objects=None):
    """Loads a Gandlf model, which includes Gandlf custom layers.

    This is done the same way as it is normally done in Keras, except that
    the optimizer is screwy, because Gandlf uses two optimizers. The model
    should be recompiled.
    """

    if not custom_objects:
        custom_objects = {}

    # Adds Gandlf layers as custom objects.
    name_cls_pairs = inspect.getmembers(sys.modules['gandlf.layers'],
                                        inspect.isclass)
    gandlf_layers = dict((name, cls) for name, cls in name_cls_pairs
                         if name and name[0] != '_')
    custom_objects.update(gandlf_layers)

    def deserialize(obj):
        if isinstance(obj, list):
            return [custom_objects[v] if v in custom_objects else v
                    for v in obj]

        if isinstance(obj, dict):
            return dict((k, custom_objects[v] if v in custom_objects else v)
                     for k, v in obj.items())

        if obj in custom_objects:
            return custom_objects[obj]

        return obj

    import h5py
    f = h5py.File(filepath, mode='r')

    # Gets the correct config parts.
    generator_config = f.attrs.get('generator_config')
    if generator_config is None:
        raise ValueError('No generator found in config file.')
    generator_config = json.loads(generator_config.decode('utf-8'))

    discriminator_config = f.attrs.get('discriminator_config')
    if discriminator_config is None:
        raise ValueError('No discriminator found in config file.')
    discriminator_config = json.loads(discriminator_config.decode('utf-8'))

    # Instantiates the models.
    generator = keras.models.model_from_config(generator_config,
                                               custom_objects)
    discriminator = keras.models.model_from_config(discriminator_config,
                                                   custom_objects)

    # Sets the weights.
    generator.load_weights_from_hdf5_group(f['generator_weights'])
    discriminator.load_weights_from_hdf5_group(f['discriminator_weights'])

    return Model(generator=generator, discriminator=discriminator)

class Model(keras.models.Model):
    """The core model for training GANs.

    Both a generator and a discriminator are needed to build the model.

    Updates are calculated and applied to the generator and discriminator
    simultaneously. If the discriminator has auxiliary outputs, it can also
    be trained to minimize the auxiliary loss, by setting `train_auxiliary`
    in the `fit` function. This makes it easy to train the model in both
    a supervised and unsupervised fashion.

    Like how a cup of tea is needed to calculate improbability factors,
    a source of randomness is usually necessary to train GANs. To make it
    easier to add random inputs, you can specify the type of random input
    rather than explicitly providing them. At runtime, the model will
    generate new normal random variables with the right input shape and
    feed them to the correct model input. This can be done with any
    normal Keras input.
    """

    def __init__(self, generator, discriminator, name=None):
        """Initializer for the GANDLF model.

        Args:
            generator: a Keras model that generates outputs.
            discriminator: a Keras model that has at least one input for every
                output of the generator model, and has at least one output.
            name: str (optional), the name for this model.
        """

        self._check_generator_and_discriminator(generator, discriminator)

        self.generator = generator
        self.discriminator = discriminator

        num_generated = len(generator.outputs)
        gen_dis_outputs = discriminator(generator.outputs +
                                        discriminator.inputs[num_generated:])

        generator_discriminator = keras.models.Model(
            input=generator.inputs + discriminator.inputs[num_generated:],
            output=gen_dis_outputs,
            name='generator_around_discriminator')

        inputs = (generator_discriminator.inputs[:len(generator.inputs)] +
                  discriminator.inputs[:num_generated] +
                  generator_discriminator.inputs[len(generator.inputs):])
        outputs = generator_discriminator.outputs * 2 + discriminator.outputs

        # Adds gen, fake and real outputs.
        dis_output_names = self.discriminator.output_names
        output_names = ['%s_gen' % name for name in dis_output_names]
        output_names += ['%s_fake' % name for name in dis_output_names]
        output_names += ['%s_real' % name for name in dis_output_names]

        # Fixes the output layers to have the right layer names.
        fixed_outputs = []
        for output_name, output in zip(output_names, outputs):
            fixed_output = copy.copy(output)
            new_layer = copy.copy(output._keras_history[0])
            new_keras_history = (new_layer,) + output._keras_history[1:]
            fixed_output._keras_history = new_keras_history
            fixed_output._keras_history[0].name = output_name
            fixed_outputs.append(fixed_output)

        # The model is treated as the generator by Keras.
        super(Model, self).__init__(inputs, fixed_outputs, name)

    def _check_generator_and_discriminator(self, generator, discriminator):
        """Validates the provided models in a user-friendly way."""

        # Checks that both are Keras models.
        if not (isinstance(generator, keras.models.Model) and
                isinstance(discriminator, keras.models.Model)):
            raise ValueError('The generator and discriminator should both '
                             'be Keras models. Got discriminator=%s, '
                             'generator=%s' % (type(discriminator),
                                               type(generator)))

        if len(generator.outputs) > len(discriminator.inputs):
            raise ValueError('The discriminator model should have at least one '
                             'input per output of the generator model.')

        # Checks that the input and output shapes line up.
        generator_shapes = generator.inbound_nodes[0].output_shapes
        discriminator_shapes = discriminator.inbound_nodes[0].input_shapes

        if any(g != d for g, d in zip(generator_shapes, discriminator_shapes)):
            raise ValueError('The discriminator input shapes should be the '
                             'same as the generator output shapes. The '
                             'discriminator has inputs %s while the '
                             'generator has outputs %s.' %
                             (str(discriminator_shapes),
                              str(generator_shapes)))

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

    def _compute_losses(self):
        """Computes generator and discriminator losses."""

        # Recomputes the masks (done in the parent compile method).
        masks = self.compute_mask(self.inputs, mask=None)
        if masks is None:
            masks = [None for _ in self.outputs]
        elif not isinstance(masks, list):
            masks = [masks]

        # Re-converts loss weights to a list.
        if self.loss_weights is None:
            loss_weights = [1. for _ in self.outputs]
        elif isinstance(self.loss_weights, dict):
            loss_weights = [self.loss_weights[name]
                            for name in self.output_names]
        else:
            loss_weights = list(self.loss_weights)

        def _compute_loss(index):
            """Computes loss for a single output."""

            y_true = self.targets[index]
            y_pred = self.outputs[index]
            weighted_loss = keras_training.weighted_objective(
                self.loss_functions[index])
            sample_weight = self.sample_weights[index]
            mask = masks[index]
            loss_weight = loss_weights[index]
            output_loss = weighted_loss(y_true, y_pred, sample_weight, mask)

            # Update the metrics tensors.
            self.metrics_tensors[index] = output_loss

            return loss_weight * output_loss

        num_outputs = len(self.discriminator.outputs)
        self.generator_loss = sum(_compute_loss(i) for i in range(num_outputs))
        self.discriminator_loss = sum(_compute_loss(i) for i in
                                      range(num_outputs, 3 * num_outputs))

        # Adds regularization losses.
        self.generator_loss += sum(self.generator.losses)
        self.discriminator_loss += sum(self.discriminator.losses)

    def _update_metrics_names(self):
        """This is a small hack to fix the metric names."""

        i = 1
        for i, name in enumerate(self.output_names, i):
            self.metrics_names[i] = name + '_loss'

        nested_metrics = keras_training.collect_metrics(self.metrics,
                                                  self.output_names)

        for name, output_metrics in zip(self.output_names, nested_metrics):
            for metric in output_metrics:
                i += 1

                if metric == 'accuracy' or metric == 'acc':
                    self.metrics_names[i] = name + '_acc'
                else:
                    metric_fn = keras.metrics.get(metric)
                    self.metrics_names[i] = name + '_' + metric_fn.__name__

    def _cast_outputs_to_all_modes(self, obj, module=None):
        output_names = self.discriminator.output_names

        if isinstance(obj, dict):
            for name in output_names:

                # A name by itself is interpretted as (gen, fake, real).
                if name in obj:
                    val = obj.pop(name)
                    for suffix in ['_gen', '_fake', '_real']:
                        if name + suffix not in obj:
                            obj[name + suffix] = val

                # Adds discriminator -> (fake, real).
                if name + '_dis' in obj:
                    val = obj.pop(name + '_dis')
                    for suffix in ['_fake', '_real']:
                        if name + suffix not in obj:
                            obj[name + suffix] = val

                # Adds all ways to combine two ore more.
                for a, b, _ in itertools.permutations(['gen', 'real', 'fake']):
                    composite = name + '_' + a + '_' + b
                    if composite in obj:
                        val = obj.pop(composite)
                        for suffix in [a, b]:
                            if name + '_' + suffix not in obj:
                                obj[name + '_' + suffix] = val

            # Each by itself is interpretted as meaning all of that type.
            for name in ['gen', 'fake', 'real']:
                if name in obj:
                    val = obj.pop(name)
                    for prefix in output_names:
                        if prefix + '_' + name not in obj:
                            obj[prefix + '_' + name] = val

            # Discriminator -> (fake, real).
            if 'dis' in obj:
                val = obj.pop('dis')
                for suffix in ['_fake', '_real']:
                    for prefix in output_names:
                        if prefix + suffix not in obj:
                            obj[prefix + suffix] = val

            # Adds all ways to combine two.
            for a, b, _ in itertools.permutations(['gen', 'real', 'fake']):
                composite = a + '_' + b
                if composite in obj:
                    val = obj.pop(composite)
                    for prefix in output_names:
                        for suffix in [a, b]:
                            if prefix + '_' + suffix not in obj:
                                obj[prefix + '_' + suffix] = val

            if module is not None:
                for k in obj.keys():
                    if (isinstance(obj[k], six.string_types) and
                        hasattr(module, obj[k])):
                        obj[k] = getattr(module, obj[k])

        elif isinstance(obj, (list, tuple)):
            if len(obj) == len(output_names):
                obj = list(obj) * 3

            elif len(obj) == 2:  # (gen and real), fake
                obj = ([obj[0]] * len(output_names) +
                       [obj[1]] * len(output_names) +
                       [obj[0]] * len(output_names))

            elif len(obj) == 3:  # gen, fake and real
                obj = ([obj[0]] * len(output_names) +
                       [obj[1]] * len(output_names) +
                       [obj[2]] * len(output_names))

            if module is not None:
                obj_fixed = []
                for k in obj:
                    if (isinstance(k, six.string_types) and
                        hasattr(module, k)):
                        obj_fixed.append(getattr(module, k))
                    else:
                        obj_fixed.append(k)
                obj = obj_fixed

        return obj

    def compile(self, loss, optimizer, metrics=None,
                loss_weights=None, sample_weight_mode=None, **kwargs):
        """Configures the model for training.

        # Args:
            optimizer: str (name of optimizer) or optimizer object.
                See Keras [optimizers](https://keras.io/optimizers/).
                Alternateively, a pair of optimizers (discriminator_optimizer,
                generator_optimizer) can be passed to use separate
                optimizers for the two models.
            loss: str (name of objective function) or objective function.
                More objective functions can be found under gandlf.losses.
                See Keras [objectives](https://keras.io/objectives/).
                It is almost always a good idea to use binary crossentropy
                loss for the real / fake prediction. To use a different
                loss for the auxiliary outputs, provide them as a list or
                dictionary.
            metrics: list of metrics to be evaluated by the model during
                training and testing. Typically you will use `metrics=['acc']`
                to calculate accuracy. To specify different metrics for
                different outputs, you could also pass a dictionary, such as
                `metrics={'real': 'accuracy', 'fake': 'accuracy', ...}`.
            sample_weight_mode: if you need to do timestep-wise sample
                weighting (2D weights), set this to "temporal". "None" defaults
                to sample-wise weights (1D). If the model has multiple outputs,
                you can use a different `sample_Weight_mode` on each output by
                passing a dictionary or list of modes.
            kwargs: extra arguments that are passed to the Theano backend (not
                used by Tensorflow).
        """

        # Preprocess the losses and loss weights, so that one value can be
        # specified for all three training modes  (generator, discriminator
        # fake, and discriminator real).
        loss = self._cast_outputs_to_all_modes(loss, gandlf_losses)
        loss_weights = self._cast_outputs_to_all_modes(loss_weights)

        self.gen_optimizer = None

        # Checks to see if the user passed separate optimizers for the
        # generator and the discriminator.
        if isinstance(optimizer, (list, tuple)):
            if len(optimizer) != 2:
                raise ValueError('If you pass a list for the optimizer, the '
                                 'list should be [discriminator_optimizer, '
                                 'generator_optimizer]. Got: %s' %
                                 str(optimizer))
            optimizer, gen_optimizer = optimizer
            self.gen_optimizer = keras.optimizers.get(gen_optimizer)

        # Call the "parent" compile method.
        super(Model, self).compile(optimizer=optimizer,
                                   loss=loss,
                                   metrics=metrics,
                                   loss_weights=loss_weights,
                                   sample_weight_mode=sample_weight_mode,
                                   **kwargs)

        # Aliases the discriminator optimizer to the regular optimizer.
        self.dis_optimizer = self.optimizer
        if self.gen_optimizer is None:
            self.gen_optimizer = self.optimizer

        # This lets the model know that it has been compiled.
        self.non_auxiliary_train_function = None
        self.auxiliary_train_function = None

        # Computes the generator and discriminator losses.
        self._compute_losses()
        self._update_metrics_names()

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

    def _make_train_function(self):
        """Builds the auxiliary train function."""

        if not hasattr(self, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        if self.train_function is None:

            # Collects inputs to the function.
            inputs = (self.inputs +
                      self.targets +
                      self.sample_weights +
                      self._get_learning_phase())

            # Gets the generator updates.
            generator_updates = self.gen_optimizer.get_updates(
                self._collected_trainable_weights[0],
                self.constraints,
                self.generator_loss)

            # Gets the discriminator updates.
            discriminator_updates = self.dis_optimizer.get_updates(
                self._collected_trainable_weights[1],
                self.constraints,
                self.discriminator_loss)

            updates = (generator_updates + discriminator_updates +
                       self.updates)

            self.train_function = K.function(
                inputs,
                [self.total_loss] +
                self.metrics_tensors,
                updates=updates,
                **self._function_kwargs)

    def _standardize_input_data(self, data, names, shapes,
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

            elif isinstance(array, six.string_types + (int, float)):
                if isinstance(array, six.string_types):
                    callable_type = array.lower()
                else:
                    callable_type = float(array)

                array = _get_callable(callable_type, shape[1:])

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
            x, input_names, input_shapes,
            check_batch_dim=False, exception_prefix='model input')
        y = self._standardize_input_data(
            y, output_names, output_shapes,
            check_batch_dim=False, exception_prefix='model output')
        y_exp = [y_inst(nb_train_samples) if hasattr(y_inst, '__call__')
                 else y_inst for y_inst in y]
        sample_weights = keras_training.standardize_sample_weights(
            sample_weight, output_names)
        class_weights = keras_training.standardize_class_weights(
            class_weight, output_names)
        sample_weights = [keras_training.standardize_weights(ref, sw, cw, mode)
                          for ref, sw, cw, mode
                          in zip(y_exp, sample_weights, class_weights,
                                 self.sample_weight_modes)]

        keras_training.check_loss_and_target_compatibility(
            y_exp, self.loss_functions, output_shapes)

        return x, y, sample_weights, nb_train_samples

    def _make_sample_function(self):
        """Instantiates the sample function."""

        if not hasattr(self, 'sample_function'):
            self.sample_function = None

        if self.sample_function is None:
            inputs = self.generator.inputs + self._get_learning_phase()
            outputs = self.generator.outputs
            kwargs = getattr(self, '_function_kwargs', {})
            self.sample_function = K.function(inputs, outputs,
                                              updates=self.state_updates,
                                              **kwargs)

    def sample(self, x, num_samples=None, batch_size=32, verbose=0):
        """Samples from the generator using the given inputs in batches.

        Args:
            x: single input, list of inputs, or dictionary of (str, input)
                pairs where the key is the input name. The input can either be
                a Numpy array, a string specifying a type (one of "normal",
                "uniform", "zeros", or "ones"), or a function that takes a
                batch size and returns an array with that batch size.
            num_samples: if none of the inputs are Numpy arrays with explicit
                sample sizes, this should be set to determine the number of
                samples to return. It overrides the sample size of any arrays.
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        Returns:
            A Numpy array of samples from the generator.
        """

        input_names = self.generator.input_names
        input_shapes = self.generator.internal_input_shapes

        x = self._convert_input_to_list(x, input_names)

        # Calculates the number of training samples.
        for arr in x:
            if is_numpy_array(arr):
                if num_samples is None:
                    num_samples = arr.shape[0]
                elif num_samples != arr.shape[0]:
                    raise ValueError('Multiple arrays were found with '
                                     'conflicting sample sizes.')

        if num_samples is None:
            raise ValueError('None of the model inputs have an explicit '
                             'sample size, so it must be specified in '
                             'num_samples.')

        x = self._standardize_input_data(
            x, input_names, input_shapes,
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
        """Runs the discriminator on a sample and returns all its outputs.

        Args:
            x: single input, list of inputs, or dictionary of (str, input)
                pairs where the key is the input name. The input can either be
                a Numpy array, a string specifying a type (one of "normal",
                "uniform", "zeros", or "ones"), or a function that takes a
                batch size and returns an array with that batch size.
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        Returns:
            A Numpy array of predictions, where the first is the real / fake
            prediction, and all others are auxiliary outputs.
        """

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
            x, input_names, input_shapes,
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

    def fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0):
        """Trains the model for a fixed number of epochs (iterations on data).

        Args:
            x: single input, list of inputs, or dictionary of (str, input)
                pairs where the key is the input name. The input can either be
                a Numpy array, a string specifying a type (one of "normal",
                "uniform", "zeros", or "ones"), or a function that takes a
                batch size and returns an array with that batch size.
                These values are the inputs of the model.
            y: same types as x, the targets of the model.
            batch_size: integer, the number of samples per gradient update.
            nb_epoch: integer, the number of times to iterate over the
                training data arrays.
            verbose: 0, 1, or 2, the verbosity mode.
                0 = silent, 1 = verbose, 2 = one log line per epoch.
            callbacks: list of callbacks to be called during training.
                See Keras [callbacks](https://keras.io/callbacks/).
            validation_split: float between 0 and 1: fraction of training data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate the loss and any model
                metrics on this data at the end of each epoch.
            validation_data: data on which to evaluate the los and any model
                metrics at the end of each epoch. The model will not be trained
                on this data. This could be a tuple (x_val, y_val) or a tuple
                (x_val, y_val, val_sample_weights).
            shuffle: boolean, whether to shuffle the training data before
                each epoch.
            class_weight: optional dictionary mapping class indices (integers)
                a weight (float) to apply to the model's loss for the samples
                from this class during training. This can be useful to tell
                the model to "pay more attention" to samples from an
                under-represented class.
            sample_weight: optional array of the same length as x, containing
                weights to apply to the model's loss for each sample. In the
                case of temporal data, you can pass a 2D array with shape
                (sample, sequence_length), to apply a different weight to
                every timestep of each sample. In this case you should make
                sure to specify sample_weight_mode="temporal" in compile().
            initial_epoch: epoch at which to start training (useful for
                resuming a previous training run).

        Returns:
            A "History" instance. Its `history` attribute contains all
            information collected during training.
        """

        # Allows passing data per output.
        y = self._cast_outputs_to_all_modes(y)

        if validation_split or validation_data:
            raise NotImplementedError('Validation sets are not yet '
                                      'implemented for gandlf models.')

        input_names = self.input_names
        input_shapes = self.internal_input_shapes
        output_names = self.output_names
        output_shapes = self.internal_output_shapes

        x, y, sample_weights, nb_train_samples = self._standardize_user_data(
            x, y, sample_weight, class_weight, input_names, input_shapes,
            output_names, output_shapes, False, batch_size)

        self._make_train_function()
        train_fn = self.train_function

        if self._get_learning_phase():
            ins = x + y + sample_weights + [1.]
        else:
            ins = x + y + sample_weights

        # Deduplicates output labels.
        out_labels = []
        for label in self.metrics_names:
            if out_labels.count(label) > 1:
                label += '_' + str(out_labels.count(label) + 1)
            out_labels.append(label)

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
        """The core loop that fits the data."""

        index_array = np.arange(nb_train_samples)

        self.history = keras.callbacks.History()
        callbacks = [keras.callbacks.BaseLogger()] + (callbacks or [])
        callbacks += [self.history]
        if verbose:
            callbacks += [keras.callbacks.ProgbarLogger()]
        callbacks = keras.callbacks.CallbackList(callbacks)

        out_labels = out_labels or []

        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self

        callbacks.set_model(callback_model)
        callbacks.set_params({
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
                index_array = keras_training.batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = keras_training.make_batches(nb_train_samples, batch_size)
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

    def save(self, filepath, overwrite=True):
        save_model(self, filepath, overwrite)

    def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
        raise NotImplementedError()  # TODO: Implement this properly.

    def _make_test_function(self):
        raise NotImplementedError()  # TODO: Implement this properly.
