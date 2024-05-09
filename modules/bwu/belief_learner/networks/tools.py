import itertools
from typing import List, Union, Optional

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.bijectors as tfb
from tensorflow_probability.python import bijectors as tfb
from keras.engine.keras_tensor import KerasTensor


def _get_elem(
        list_of_models: Union[List[Optional[tfk.Model]], Optional[tfk.Model]],
        i: int,
        default=tfk.Sequential([tfkl.Lambda(tf.identity)])
):
    list_of_models = tf.nest.flatten(list_of_models)
    model = list_of_models[min(i, len(list_of_models) - 1)]
    if model is None:
        return default
    else:
        return model


def get_activation_fn(activation: str):
    if hasattr(tf.nn, activation):
        return getattr(tf.nn, activation)
    elif hasattr(tfb, activation):
        return getattr(tfb, activation)()
    else:
        # custom activations
        def smooth_elu(x):
            return tf.nn.softplus(2. * x + 2.) / 2. - 1.

        other_activations = {
            'smooth_elu': smooth_elu,
            'SmoothELU': tfb.Chain([tfb.Shift(-1.), tfb.Scale(.5), tfb.Softplus(), tfb.Shift(2.), tfb.Scale(2.)],
                                   name='SmoothELU'),
            'linear': None,
        }
        return other_activations.get(
            activation,
            ValueError("activation {} unknown".format(activation)))


def scan_model(model: tfk.Model):
    hidden_units = []
    activation = None
    if model is None:
        return [128, 128], tf.nn.relu
    for layer in model.layers:
        if hasattr(layer, 'units'):
            hidden_units.append(layer.units)
        if hasattr(layer, 'activation') and activation != layer.activation:
            activation = layer.activation
    return hidden_units, activation


def _pass_in_layers(layers: List[tfkl.Layer], input_):
    output = input_
    for layer in layers:
        output = layer(output)
    return output


def layer_rep(layer):
    if isinstance(layer, tfkl.Conv2D):
        rep = layer.__class__.__name__
        rep += f"(filters={layer.filters}, kernel_size={layer.kernel_size}, padding={layer.padding}, strides={layer.strides})"
    elif isinstance(layer, tfkl.Activation):
        rep = f"Activation({layer.activation.__name__})"
    elif isinstance(layer, tfkl.Flatten):
        rep = "Flatten()"
    elif isinstance(layer, tfkl.Reshape):
        rep = f"Reshape({layer.target_shape})"
    elif isinstance(layer, tfkl.Dense):
        rep = f"Dense(units={layer.units}"
        activation = getattr(layer.activation, '__name__',
                             getattr(layer.activation, 'name', layer.activation))
        if activation != 'linear':
            rep += f", activation={activation}"
        rep += ')'
    elif isinstance(layer, tfkl.BatchNormalization):
        rep = "BatchNormalization()"
    elif isinstance(layer, tfkl.InputLayer):
        rep = f"Input({layer.input_shape})"
    elif isinstance(layer, KerasTensor):
        rep = f"KerasTensor(name={layer.name}, shape={layer.shape})"
    elif isinstance(layer, tfkl.Concatenate):
        rep = "Concatenate()"
    else:
        raise ValueError(f"Unknown layer {layer}")
    return rep


def rep_concat_layers(layers):
    layers = itertools.chain(*[[l] if not isinstance(l, tfk.Model) else l.layers for l in layers])
    repr_layers = list(map(layer_rep, layers))
    repr_layers = f"Concatenate([{' , '.join(repr_layers)}])"
    return repr_layers


def rep_layers(layers):
    layers = itertools.chain(*[[l] if not isinstance(l, tfk.Model) else l.layers for l in layers])
    repr_layers = list(map(layer_rep, layers))
    repr_layers = ' --> '.join(repr_layers)
    return repr_layers
