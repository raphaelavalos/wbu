from typing import Tuple, Optional, List, Union, Callable

import tensorflow.keras.layers as tfkl

from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.networks.tools import _pass_in_layers, get_activation_fn


def _cnn_network_layers(
        filters: Tuple[int, ...],
        kernel_size: Tuple[Union[Tuple[int, ...], int], ...],
        strides: Tuple[Union[Tuple[int, ...], int], ...],
        padding: Tuple[str, ...],
        # cf. _fc_network_layers
        activation: Union[Callable, str],
        batch_norm: bool,
        raw_last: bool,
        transpose: bool = False,
        **kwargs,
):
    layers = []
    tf_layer = tfkl.Conv2D if not transpose else tfkl.Conv2DTranspose
    elements = [filters, kernel_size, strides, padding]
    # check that the number of elements is the same for all components
    n = len(elements[0])
    for element in elements[1:]:
        assert len(element) == n, "the number of filters, kernel_size, strides, padding should be the same"
    for i, (filters_, kernel_size_, stride_, padding_) in enumerate(zip(*elements)):
        apply_ = (i + 1 != len(filters)) or not raw_last
        layers.append(tf_layer(
            filters=filters_,
            kernel_size=kernel_size_,
            strides=stride_,
            padding=padding_,
        ))
        if apply_ and batch_norm:
            layers.append(tfkl.BatchNormalization())
        if apply_ and activation:
            if callable(activation):
                layers.append(tfkl.Activation(activation))
            else:
                layers.append(tfkl.Activation(get_activation_fn(activation)))
    if not transpose:
        layers.append(tfkl.Flatten())
    return layers


def _conv_network(input_: tfkl.Input,
                  filters: Tuple[int, ...],
                  kernel_size: Tuple[Union[Tuple[int, ...], int], ...],
                  strides: Tuple[Union[Tuple[int, ...], int], ...],
                  padding: Tuple[str, ...],
                  activation: str,
                  batch_norm: bool,
                  raw_last: bool,
                  transpose: bool = False,
                  **kwargs,
                  ):
    layers = []
    if 'input_dim' in kwargs:
        layers = [tfkl.Reshape(kwargs['input_dim'])]
    layers += _cnn_network_layers(filters, kernel_size, strides, padding, activation, batch_norm, raw_last, transpose)
    return _pass_in_layers(layers, input_)


class Convolutional(tfkl.Layer):
    def __init__(
            self,
            model_arch: ModelArchitecture,
            **kwargs,
    ):
        self.model_arch = model_arch
        d = model_arch.short_dict()
        name = d.pop('name', None)
        super().__init__(name=name, **kwargs)
        # kernel_size, strides, padding = [
        #     tf.nest.flatten((param if param is not None else default))
        #     for param, default in zip((kernel_size, strides, padding), (3, 1, 'valid'))
        # ]
        self._layers: List[tfkl.Layer] = _cnn_network_layers(**d)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self._layers, inputs)


class Deconvolutional(tfkl.Layer):
    def __init__(
            self,
            model_arch: ModelArchitecture,
            output_shape: Optional[Tuple[int, ...]] = None,
            **kwargs,
    ):
        self.model_arch = model_arch
        if self.model_arch.transpose:
            d = self.model_arch.short_dict()
        else:
            invert_model_arch = model_arch.invert(output_shape)
            d = invert_model_arch.short_dict()
        name = d.pop('name')
        super().__init__(name=name, **kwargs)
        self._layers: List[tfkl.Layer] = _cnn_network_layers(**d)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self._layers, inputs)
