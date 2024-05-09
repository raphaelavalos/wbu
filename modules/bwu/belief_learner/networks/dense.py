from typing import List, Tuple, Union, Callable, Optional

import tensorflow as tf
import tensorflow_probability.python.distributions.categorical
from tensorflow_probability import distributions as tfd
import tensorflow.keras as keras

from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.networks.get_model import get_model
from belief_learner.networks.fully_connected import _fc_network_layers
from belief_learner.networks.tools import _pass_in_layers

def generate_rl_model(
        model_architecture: ModelArchitecture,
        dueling: bool, actor_critic: bool,
        sub_belief_upscaler: Optional[keras.Model] = None,
        belief_embedding: Optional[keras.Model] = None,
) -> tf.keras.models.Model:
    """
    Create a model.

    :param model_architecture: The architecture model.
    :param dueling: If true the last layer uses dueling.
    :return: A keras model.
    """
    assert model_architecture.raw_last
    assert int(dueling) + int(actor_critic) != 2, "Can't be dueling dqn and actor critic."
    assert belief_embedding == sub_belief_upscaler is None or None not in [belief_embedding, sub_belief_upscaler]
    if sub_belief_upscaler is not None:
        pre_proc_net = keras.Sequential([
            sub_belief_upscaler,
            # does not update the sub-belief upscaler
            tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x), name='sub_belief_upscaler_stop_grad'),
            tf.keras.layers.Lambda(lambda x: tfd.Categorical(logits=x).probs_parameter(), name='belief_probs'),
            belief_embedding
        ], name="belief_processing_net")
    else:
        pre_proc_net = keras.Sequential([tf.keras.layers.Lambda(tf.identity)])
    if not dueling and not actor_critic:
        return keras.Sequential([
            keras.layers.Input(model_architecture.input_dim),
            pre_proc_net,
            get_model(model_arch=model_architecture, as_model=True)],
            name="policy_model")
    input_ = keras.layers.Input(model_architecture.input_dim)
    layers = _fc_network_layers(model_architecture.hidden_units,
                                model_architecture.activation,
                                model_architecture.output_dim,
                                model_architecture.batch_norm,
                                model_architecture.raw_last)
    if sub_belief_upscaler is not None:
        layers = [pre_proc_net] + layers
    hidden = _pass_in_layers(layers[:-1], input_)
    advantage = layers[-1](hidden)
    value = keras.layers.Dense(units=1, activation=None)(hidden)
    if dueling:
        advantage -= tf.reduce_mean(advantage, -1, True)
        output = advantage + value
    else:
        output = [advantage, value]
    model = tf.keras.models.Model(inputs=input_, outputs=output,
                                  name=model_architecture.name)
    return model
