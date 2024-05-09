from typing import Optional, Tuple, Union, Callable

import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl

from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.networks.nn import generate_sequential_model
from belief_learner.networks.tools import _get_elem, rep_concat_layers, rep_layers, layer_rep


def have_elu(model: tfk.Model):
    for layer in model.layers:
        activation = getattr(layer, 'activation', None)
        if activation is not None:
            if getattr(activation, '__name__', None) == 'elu':  
                return True
    return False


class SteadyStateLipschitzFunction(tfk.Model):

    def __init__(
            self,
            latent_state: tfk.Input,
            steady_state_lipschitz_network: tfk.Model,
            transition_based: bool = False,
            next_latent_state: Optional[tfk.Input] = None,
            latent_action: Optional[tfkl.Input] = None,
    ):
        inputs = [latent_state] + ([latent_action] if latent_action is not None else []) + [next_latent_state]

        network_input = tfkl.Lambda(lambda _inputs: _inputs if transition_based else _inputs[:1])(inputs)
        network_input = tfkl.Concatenate()(network_input)
        assert not have_elu(steady_state_lipschitz_network)
        _steady_state_lipschitz_network = steady_state_lipschitz_network(network_input)
        _steady_state_lipschitz_network = tfkl.Dense(
            units=1,
            activation=None,
            name='steady_state_lipschitz_network_output'
        )(_steady_state_lipschitz_network)

        super(SteadyStateLipschitzFunction, self).__init__(
            inputs=inputs,
            outputs=_steady_state_lipschitz_network,
            name='steady_state_lipschitz_network')

    def __repr__(self):
        name = getattr(self, 'name', self.__class__.__name__)
        layers = rep_concat_layers(self.inputs) + ' --> ' + rep_layers(self.layers[len(self.inputs) + 1:])
        return name + ' : ' + layers


class TransitionLossLipschitzFunction(tfk.Model):
    # Using latent_obs, latent_state would break guarantees.
    def __init__(
            self,
            state: Union[tfkl.Input, Tuple[tfkl.Input, ...]],
            action: tfkl.Input,
            latent_state: tfkl.Input,
            next_latent_state: tfkl.Input,
            transition_loss_lipschitz_network: tfk.Model,
            latent_action: Optional[tfkl.Input] = None,
            flatten_units: int = 64,
            pre_proc_net: Optional[Union[tfkl.Layer, tfk.Model, Callable]] = None,
            **kwargs,
    ):
        assert not have_elu(transition_loss_lipschitz_network)
        try:
            no_input_states = len(state)
        except TypeError:
            no_input_states = 1

        if pre_proc_net is not None:
            pre_proc_net = tf.nest.flatten(pre_proc_net)
            assert len(pre_proc_net) == no_input_states or len(pre_proc_net) == 1, \
                "the number of pre-processing networks should be one or have the same size as the number of inputs"

        if no_input_states > 1:
            components = []
            nets = []
            for i, state_component in enumerate(state):
                if pre_proc_net is not None:
                    _state = _get_elem(pre_proc_net, i)(state_component)
                else:
                    _state = state_component
                net = tfk.Sequential([
                    tfkl.Flatten(),
                    tfkl.Dense(
                        units=flatten_units,
                        activation='sigmoid'
                    )
                ])
                nets.append(net)
                x = net(_state)
                components.append(x)
            _state = tfkl.Concatenate()(components)
        else:
            _state = state

        inputs = [state, action, latent_state]

        if latent_action is not None:
            inputs.append(latent_action)
        inputs.append(next_latent_state)
        # combine multiple state-components into _state
        _transition_loss_lipschitz_network = tfkl.Concatenate()([_state] + inputs[1:])
        _transition_loss_lipschitz_network = transition_loss_lipschitz_network(_transition_loss_lipschitz_network)
        _transition_loss_lipschitz_network = tfkl.Dense(
            units=1,
            activation=None,
            name='transition_loss_lipschitz_network_output'
        )(_transition_loss_lipschitz_network)

        super(TransitionLossLipschitzFunction, self).__init__(
            inputs=inputs,
            outputs=_transition_loss_lipschitz_network,
            name='transition_loss_lipschitz_network')
        self.pre_proc_net = pre_proc_net
        self.nets = nets
        self.transition_loss_lipschitz_network = transition_loss_lipschitz_network

    def __repr__(self):
        return f"Concatenate([\n" \
               f"\t{rep_layers([self.layers[0]])} --> {self.pre_proc_net[0]} --> {rep_layers([self.nets[0]])}, \n" \
               f"\t{rep_layers([self.layers[1]])} --> {self.pre_proc_net[1]} --> {rep_layers([self.nets[1]])},\n" \
               f"\t{', '.join(map(layer_rep, self.inputs[2:]))}\n" \
               f"]) --> {rep_layers([self.transition_loss_lipschitz_network, self.layers[-1]])}"


class ObservationLipschitzFunction(tfk.Model):
    def __init__(
            self,
            mdp_state: tfkl.Input,
            observation: tfkl.Input,
            action: tfkl.Input,
            next_mdp_state: tfkl.Input,
            next_observation: tfkl.Input,
            lipschitz_arch: ModelArchitecture,
            flatten_units: int = 64,
            pre_proc_net: Optional[Union[tfkl.Layer, tfk.Model, Callable]] = None,
            **kwargs,
    ):
        state = [mdp_state, observation]
        next_state = [next_mdp_state, next_observation]
        lipschitz_net = generate_sequential_model(lipschitz_arch.replace(name="obs_lip_fn"))

        if pre_proc_net is not None:
            pre_proc_net = tf.nest.flatten(pre_proc_net)
            assert 1 <= len(pre_proc_net) <= 2, \
                "the number of pre-processing networks should be one or have the same size as the number of inputs"

        states = [state, next_state]
        for s_idx, s in enumerate(states):
            components = []
            nets = []

            for i, state_component in enumerate(s):
                if pre_proc_net is not None:
                    _state = _get_elem(pre_proc_net, i)(state_component)
                else:
                    _state = state_component
                net = tfk.Sequential([
                    tfkl.Flatten(),
                    tfkl.Dense(
                        units=flatten_units,
                        activation='sigmoid'
                    )
                ])
                nets.append(net)
                x = net(_state)
                components.append(x)
            states[s_idx] = tfkl.Concatenate()(components)

        state, next_state = states
        inputs = [mdp_state, observation, action, next_mdp_state, next_observation]

        _lipschitz_net = tfkl.Concatenate()([state, action, next_state])
        _lipschitz_net = lipschitz_net(_lipschitz_net)
        _lipschitz_net = tfkl.Dense(
            units=1,
            activation=None,
            name='obs_loss_lipschitz_network_output'
        )(_lipschitz_net)

        super(ObservationLipschitzFunction, self).__init__(
            inputs=inputs,
            outputs=_lipschitz_net,
            name='obs_loss_lipschitz_network')
