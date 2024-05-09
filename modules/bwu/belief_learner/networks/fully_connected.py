from typing import Tuple, Optional, List, Union, Callable

import numpy as np
import tensorflow.keras.layers as tfkl

from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.networks.tools import get_activation_fn, _pass_in_layers, rep_layers


def _fc_network_layers(hidden_units: Tuple[int, ...],
                       # F: general activation functions can now be provided by hand
                       # the set of available activation function is now larger (see get_activation_fn)
                       activation: Union[Callable, str],
                       output_dim: Optional[Tuple[int, ...]],
                       batch_norm: bool,
                       raw_last: bool,
                       **kwargs,
                       ):
    layers = [tfkl.Flatten()]
    assert output_dim is not None, "output_dim should be provided"
    units = tuple(list(hidden_units) + [np.prod(output_dim)])
    for i, unit in enumerate(units):
        # apply_ = not raw_last and (i + 1 == len(units))
        apply_ = not raw_last or (i + 1 != len(units))  
        layers.append(tfkl.Dense(unit))
        if apply_ and batch_norm:
            layers.append(tfkl.BatchNormalization())
        if apply_ and activation:
            if callable(activation):
                layers.append(tfkl.Activation(activation))
            else:
                layers.append(tfkl.Activation(get_activation_fn(activation)))
    if output_dim is not None:
        layers.append(tfkl.Reshape(output_dim))
    return layers


def _fc_network(input_: tfkl.Input,
                hidden_units: Tuple[int, ...],
                activation: str,
                output_dim: Optional[Tuple[int, ...]],
                batch_norm: bool,
                raw_last: bool,
                **kwargs,
                ):
    layers = _fc_network_layers(hidden_units, activation, output_dim, batch_norm, raw_last)
    return _pass_in_layers(layers, input_)


class FullyConnected(tfkl.Layer):
    def __init__(
            self,
            model_arch: ModelArchitecture,
            **kwargs
    ):
        self.model_arch = model_arch
        d = model_arch.short_dict()
        if 'name' not in kwargs:
            kwargs['name'] = d.pop('name', None)
        super().__init__(**kwargs)
        self._layers = _fc_network_layers(**d)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self._layers, inputs)

    def __repr__(self):
        return f'{getattr(self, "name" , self.__class__.__name__)} : {rep_layers(self._layers)}'

    def get_config(self):
        config = super(FullyConnected, self).get_config()
        config["model_arch"] = self.model_arch
        return config

class TransposeFullyConnected(tfkl.Layer):
    def __init__(
            self,
            model_arch: ModelArchitecture,
            output_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ):
        self.model_arch = model_arch
        if self.model_arch.transpose:
            d = self.model_arch.short_dict()
        else:
            invert_model_arch = model_arch.invert(output_shape)
            d = invert_model_arch.short_dict()
        name = d.pop('name')
        super().__init__(name=name, **kwargs)
        self._layers: List[tfkl.Layer] = _fc_network_layers(**d)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self._layers, inputs)



