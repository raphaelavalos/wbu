from typing import List

import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from belief_learner.networks.tools import get_activation_fn
from belief_learner.networks.model_architecture import ModelArchitecture


def build_model_as_layer(model_architecture: ModelArchitecture) -> tfkl.Layer:
    pass


def build_model(model_architecture: ModelArchitecture) -> tfk.Model:
    return tfk.Sequential([build_model_as_layer(model_architecture)])


def _get_elem(l: List, i: int):
    return l[min(i, len(l) - 1)]


def generate_sequential_model(architecture: ModelArchitecture):
    return tfk.Sequential([
        tfkl.Dense(
            units,
            activation=get_activation_fn(architecture.activation),
            name="{}_layer{:d}".format(architecture.name, i) if architecture.name is not None else None,
        ) for i, units in enumerate(architecture.hidden_units)],
        name=architecture.name)
