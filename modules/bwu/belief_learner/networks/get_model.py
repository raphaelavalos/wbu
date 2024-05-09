from typing import Tuple, Optional
import tensorflow.keras as tfk
from belief_learner.networks.conv import _conv_network, Convolutional, Deconvolutional
from belief_learner.networks.fully_connected import _fc_network, FullyConnected, TransposeFullyConnected
from belief_learner.networks.model_architecture import ModelArchitecture

from belief_learner.utils import get_logger

logger = get_logger(__name__)


def get_model(model_arch: ModelArchitecture,
              invert: bool = False,
              output_dim: Optional[Tuple[int, ...]] = None,
              input_dim: Optional[Tuple[int, ...]] = None,
              as_model: bool = False,
              name: Optional[str] = None,
              ):
    if name:
        model_arch = model_arch._replace(name=name)
    if model_arch.is_cnn:
        if model_arch.hidden_units is not None:
            logger.warning(f"Model arch {model_arch} is a CNN, should not have hidden_units !")
            model_arch = model_arch._replace(hidden_units=None)
    if as_model:
        if invert:
            if model_arch.output_dim is not None:  
                assert output_dim is None
                output_dim = model_arch.output_dim
            if output_dim is None:
                # dirty output dim inference
                _net = get_model(model_arch, as_model=True)
                output_dim = _net.outputs[0].shape[1:]
                del _net
            input_ = tfk.Input(output_dim)
            model_arch = model_arch.invert(input_dim)
        else:
            if model_arch.input_dim is not None:
                assert input_dim is None
                input_dim = model_arch.input_dim
            assert input_dim is not None
            input_ = tfk.Input(input_dim)
        if model_arch.is_cnn:
            output = _conv_network(input_=input_, **model_arch.short_dict())
        else:
            output = _fc_network(input_=input_, **model_arch.short_dict())
        model = tfk.Model(inputs=input_, outputs=output, name=model_arch.name)
        return model
    if model_arch.is_cnn:
        if invert:
            layer = Deconvolutional(model_arch=model_arch, output_shape=output_dim)
        else:
            layer = Convolutional(model_arch)
    else:
        if invert:
            layer = TransposeFullyConnected(model_arch=model_arch, output_shape=output_dim)
        else:
            layer = FullyConnected(model_arch=model_arch)
    return layer
