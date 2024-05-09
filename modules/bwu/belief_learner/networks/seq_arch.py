
from typing import Iterable

import tensorflow.keras.layers as tfkl

from belief_learner.networks.get_model import get_model
from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.networks.tools import _pass_in_layers


class SeqLayer(tfkl.Layer):
    def __init__(self,
                 model_arch_seq: Iterable[ModelArchitecture],
                 **kwargs):
        name = kwargs.pop('name', None)
        if name is None:
            name = '-'.join(model_arch.name for model_arch in model_arch_seq)
        self.layers = [
            get_model(model_arch, as_model=False)
            for model_arch in model_arch_seq
        ]
        super().__init__(name=name, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return _pass_in_layers(self.layers, inputs)
