import gc
import json
import math
import os.path
from collections import namedtuple
from collections.abc import Iterable

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Callable, NamedTuple, List, Union, Dict, Collection
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.keras.utils import Progbar
from tensorflow.python.keras.metrics import Mean, MeanSquaredError
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd
from tf_agents.trajectories import time_step as ts

import tf_agents
from tf_agents.policies import TFPolicy
from tf_agents.typing.types import Float, Int
from tf_agents.environments import tf_py_environment

from belief_learner.networks.autoregressive_bernoulli import AutoRegressiveBernoulliNetwork
from belief_learner.networks.latent_policy import LatentPolicyNetwork
from belief_learner.networks.decoders import RewardNetwork, ActionReconstructionNetwork, StateReconstructionNetwork
from belief_learner.networks.encoders import StateEncoderNetwork, ActionEncoderNetwork, \
    AutoRegressiveStateEncoderNetwork, EncodingType, \
    LatentStateEncoderNetwork
from belief_learner.networks.lipschitz_functions import SteadyStateLipschitzFunction, TransitionLossLipschitzFunction
from belief_learner.networks.steady_state_network import SteadyStateNetwork
from belief_learner.utils.costs import get_cost_fn
from belief_learner.networks.nn import generate_sequential_model
from belief_learner.networks.tools import get_activation_fn
from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.networks.get_model import get_model
from belief_learner.networks.seq_arch import SeqLayer
from belief_learner.wae.mdp.variational_mdp import VariationalMarkovDecisionProcess, EvaluationCriterion, \
    debug_gradients, debug, epsilon
from belief_learner.verification.local_losses import estimate_local_losses_from_samples

from belief_learner.utils import get_logger

logger = get_logger(__name__)


class WassersteinRegularizerScaleFactor(NamedTuple):
    global_scaling: Optional[Float] = None
    global_gradient_penalty_multiplier: Optional[Float] = None
    steady_state_scaling: Optional[Float] = None
    steady_state_gradient_penalty_multiplier: Optional[Float] = None
    local_transition_loss_scaling: Optional[Float] = None
    local_transition_loss_gradient_penalty_multiplier: Optional[Float] = None
    observation_regularizer_scaling: Optional[Float] = 1.
    observation_regularizer_gradient_penalty_multiplier: Optional[Float] = 10.

    values = namedtuple('WassersteinRegularizer', ['scaling', 'gradient_penalty_multiplier'])

    def sanity_check(self):
        if self.global_scaling is None and (self.steady_state_scaling is None or
                                            self.local_transition_loss_scaling is None or
                                            self.observation_regularizer_scaling is None):
            raise ValueError("Either a global scaling value or a unique scaling value for"
                             "each Wasserstein regularizer should be provided.")

        if self.global_gradient_penalty_multiplier is None and (
                self.steady_state_gradient_penalty_multiplier is None or
                self.local_transition_loss_gradient_penalty_multiplier is None or
                self.observation_regularizer_gradient_penalty_multiplier is None):
            raise ValueError("Either a global gradient penalty multiplier or a unique multiplier for"
                             "each Wasserstein regularizer should be provided.")

    @property
    def stationary(self):
        self.sanity_check()
        return self.values(
            scaling=self.steady_state_scaling if self.steady_state_scaling is not None else self.global_scaling,
            gradient_penalty_multiplier=(self.steady_state_gradient_penalty_multiplier
                                         if self.steady_state_gradient_penalty_multiplier is not None else
                                         self.global_gradient_penalty_multiplier))

    @property
    def local_transition_loss(self):
        self.sanity_check()
        return self.values(
            scaling=(self.local_transition_loss_scaling
                     if self.local_transition_loss_scaling is not None else
                     self.global_scaling),
            gradient_penalty_multiplier=(self.local_transition_loss_gradient_penalty_multiplier
                                         if self.local_transition_loss_gradient_penalty_multiplier is not None else
                                         self.global_gradient_penalty_multiplier))

    @property
    def observation_regularizer(self):
        return self.values(
            scaling=(
                self.observation_regularizer_scaling
                if self.observation_regularizer_scaling is not None
                else self.global_scaling),
            gradient_penalty_multiplier=(
                self.observation_regularizer_gradient_penalty_multiplier
                if self.observation_regularizer_gradient_penalty_multiplier is not None
                else self.global_gradient_penalty_multiplier))

    def short_dict(self):
        return {k: v for k, v in self._asdict().items() if v is not None}


class BaseModelArchitecture(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(name="base_model")
        for key, value in kwargs.items():
            setattr(self, key, value)


# wae_mdp.training_step([z_states, z_observations, labels, z_actions, rewards, z_states_next, z_observations_next])
#    -> wae_mdp.compute_apply_gradients(batch)

# wae.obs_encoder
# wae.obs_decoder
# wae.state_encoder

# wae = WassersteinMarkovDecisionProcess(...)



# state_encoder = wae.state_encoder(state) -> embed_state
# obs_encoder = wae.state_encoder(obs) -> embed_obs
# embed_ext_state = f(state_encoder, obs_encoder) # f = concat
# phi([state, observation]) -> z (relaxed bernoulli) -- deterministic
# psi(z) -> [state^, observation^] -- deterministic

# z = phi([state, observation]) = phi_wae(encode_state(state), encode_obs(observation)])) = phi_wae(z_state, z_observation)
# decoder(z) = [state, observation] where observation = O(z) and state = decode_state(z, observation)
# O == decode_obs
# O: Z -> OMEGA, decode_state: Z, OMEGA -> S
# where Z is the latent space, S is the state space, OMEGA is the observation space

class WassersteinMarkovDecisionProcess(VariationalMarkovDecisionProcess):
    def __init__(
            self,
            state_shape: Union[Collection[Tuple[int, ...]], Tuple[int, ...]],
            action_shape: Tuple[int, ...],
            reward_shape: Tuple[int, ...],
            label_shape: Tuple[int, ...],
            discretize_action_space: bool,
            state_encoder_network: ModelArchitecture,
            action_decoder_network: ModelArchitecture,
            transition_network: ModelArchitecture,
            reward_network: ModelArchitecture,
            decoder_network: ModelArchitecture,
            latent_policy_network: ModelArchitecture,
            steady_state_lipschitz_network: ModelArchitecture,
            transition_loss_lipschitz_network: ModelArchitecture,
            latent_state_size: int,
            number_of_discrete_actions: Optional[int] = None,
            action_encoder_network: Optional[ModelArchitecture] = None,
            state_encoder_pre_processing_network: Optional[Union[Collection[ModelArchitecture], ModelArchitecture]] = \
                    None,
            state_decoder_pre_processing_network: Optional[Union[Collection[ModelArchitecture], ModelArchitecture]] = \
                    None,
            time_stacked_states: bool = False,
            state_encoder_temperature: float = 2. / 3,
            state_prior_temperature: float = 1. / 2,
            action_encoder_temperature: Optional[Float] = None,
            latent_policy_temperature: Optional[Float] = None,
            wasserstein_regularizer_scale_factor: WassersteinRegularizerScaleFactor = WassersteinRegularizerScaleFactor(
                global_scaling=1., global_gradient_penalty_multiplier=1.),
            encoder_temperature_decay_rate: float = 0.,
            prior_temperature_decay_rate: float = 0.,
            reset_state_label: bool = True,
            autoencoder_optimizer: Optional = None,
            wasserstein_regularizer_optimizer: Optional = None,
            entropy_regularizer_scale_factor: float = 0.,
            entropy_regularizer_decay_rate: float = 0.,
            entropy_regularizer_scale_factor_min_value: float = 0.,
            importance_sampling_exponent: Optional[Float] = 1.,
            importance_sampling_exponent_growth_rate: Optional[Float] = 0.,
            time_stacked_lstm_units: int = 128,
            reward_bounds: Optional[Tuple[float, float]] = None,
            latent_stationary_network: Optional[tfk.Model] = None,
            action_entropy_regularizer_scaling: float = 1.,
            enforce_upper_bound: bool = False,
            squared_wasserstein: bool = False,
            n_critic: int = 5,
            trainable_prior: bool = True,
            state_encoder_type: EncodingType = EncodingType.DETERMINISTIC,
            policy_based_decoding: bool = False,
            deterministic_state_embedding: bool = True,
            state_encoder_softclipping: bool = False,
            external_latent_policy: Optional[TFPolicy] = None,
            clip_by_global_norm: Optional[float] = None,
            cost_fn: str = 'l22',
            cost_weights: Optional[Dict[str, float]] = None,
            cost_fn_state: Optional[Callable] = None,
            recover_fn_state: Optional[Callable] = tfb.Identity(),
            recover_fn_obs: Optional[Callable] = None,
            distillation: bool = True,
            env_name: Optional[str] = None,
            *args, **kwargs
    ):
        assert state_encoder_type == EncodingType.DETERMINISTIC
        super(WassersteinMarkovDecisionProcess, self).__init__(
            state_shape=state_shape, action_shape=action_shape, reward_shape=reward_shape, label_shape=label_shape,
            encoder_network=None, transition_network=None, reward_network=None, decoder_network=None,
            time_stacked_states=time_stacked_states, latent_state_size=latent_state_size,
            encoder_temperature=state_encoder_temperature, prior_temperature=state_prior_temperature,
            encoder_temperature_decay_rate=encoder_temperature_decay_rate,
            prior_temperature_decay_rate=prior_temperature_decay_rate,
            pre_loaded_model=True, optimizer=None,
            reset_state_label=reset_state_label,
            evaluation_window_size=0,
            evaluation_criterion=EvaluationCriterion.MEAN,
            importance_sampling_exponent=importance_sampling_exponent,
            importance_sampling_exponent_growth_rate=importance_sampling_exponent_growth_rate,
            time_stacked_lstm_units=time_stacked_lstm_units,
            reward_bounds=reward_bounds,
            entropy_regularizer_scale_factor=entropy_regularizer_scale_factor,
            entropy_regularizer_scale_factor_min_value=entropy_regularizer_scale_factor_min_value,
            entropy_regularizer_decay_rate=entropy_regularizer_decay_rate,
            deterministic_state_embedding=deterministic_state_embedding)

        # for saving the model
        _params = list(locals().items())
        self._params = {key: str(value) for key, value in _params}

        self.wasserstein_regularizer_scale_factor = wasserstein_regularizer_scale_factor
        self.mixture_components = None
        self._autoencoder_optimizer = autoencoder_optimizer
        self._wasserstein_regularizer_optimizer = wasserstein_regularizer_optimizer
        self.action_discretizer = discretize_action_space
        self.policy_based_decoding = policy_based_decoding
        self.action_entropy_regularizer_scaling = action_entropy_regularizer_scaling
        self.enforce_upper_bound = enforce_upper_bound
        self.squared_wasserstein = squared_wasserstein
        self.n_critic = n_critic
        self.trainable_prior = trainable_prior
        self.include_state_encoder_entropy = not (
                entropy_regularizer_scale_factor < epsilon
                or state_encoder_type is EncodingType.DETERMINISTIC)
        self.include_action_encoder_entropy = not (action_entropy_regularizer_scaling < epsilon)
        self._state_encoder_type = state_encoder_type
        self._cost_fn = cost_fn
        self._cost_fn_state = cost_fn_state
        self._recover_fn_state = recover_fn_state
        self._recover_fn_obs = recover_fn_obs
        self._obs_variance = tf.Variable(tf.zeros(state_shape[1]))
        self._obs_variance_counter = tf.Variable(0)
        self._env_name = env_name
        self.external_latent_policy = external_latent_policy

        if self.action_discretizer:
            self.number_of_discrete_actions = number_of_discrete_actions
        else:
            assert len(action_shape) == 1
            self.number_of_discrete_actions = self.action_shape[0]

        self._action_encoder_temperature = None
        if action_encoder_temperature is None:
            self.action_encoder_temperature = 1. / (self.number_of_discrete_actions - 1)
        else:
            self.action_encoder_temperature = action_encoder_temperature

        self._latent_policy_temperature = None
        if latent_policy_temperature is None:
            self.latent_policy_temperature = self.action_encoder_temperature / 1.5
        else:
            self.latent_policy_temperature = latent_policy_temperature

        self._sample_additional_transition = False
        if state_encoder_softclipping:
            self.softclip = tf.nn.tanh
        else:
            self.softclip = tfb.Identity()

        try:
            len(state_shape[0])
            state = []
            for i, shape in enumerate(state_shape):
                state.append(tfkl.Input(shape=shape, name=f"state_{i:d}"))
            n_states = len(state)
        except TypeError:
            n_states = 1
            state = tfkl.Input(shape=state_shape, name="state")

        self._cost_weights = {
            'state': [1.] * n_states,
            'action': 1.,
            'reward': 1.,
        }
        if cost_weights is not None:
            self._cost_weights.update(cost_weights)

        for k, v in self._cost_weights.items():
            if isinstance(v, Iterable) and not math.isclose(sum(v), 1.) or v != 1.:
                logger.warning(f"The (sum) cost weight for '{k}' is not 1. This changes the relative weight of "
                               f"reconstruction over the other losses.")

        action = tfkl.Input(shape=action_shape, name="action")
        latent_state = tfkl.Input(shape=(self.latent_state_size,), name="latent_state")
        latent_action = tfkl.Input(shape=(self.number_of_discrete_actions,), name="latent_action")
        next_latent_state = tfkl.Input(shape=(self.latent_state_size,), name='next_latent_state')

        self._state_input = state
        self._action_input = action
        self._latent_state_input = latent_state
        self._latent_action_input = latent_action
        self._next_latent_state_input = next_latent_state

        # state encoder network
        def _get_pre_processing_layers(state_encoder_pre_processing_network: ModelArchitecture,
                                       prefix: Optional[str] = None):
            try:
                len(state_encoder_pre_processing_network)
            except TypeError:
                state_encoder_pre_processing_network = [state_encoder_pre_processing_network]

            if prefix:
                state_encoder_pre_processing_network = [tuple(model_arch.replace(name=f"{prefix}-{model_arch.name}")
                                                              for model_arch in model_arch_tuples)
                                                        if model_arch_tuples
                                                        else model_arch_tuples
                                                        for model_arch_tuples in state_encoder_pre_processing_network]

            pre_processing_layers = []
            for i, model_arch_tuple in enumerate(state_encoder_pre_processing_network):
                if len(model_arch_tuple) == 1:
                    pre_processing_layers.append(get_model(model_arch_tuple[0]))
                elif len(model_arch_tuple) > 1:
                    pre_processing_layers.append(SeqLayer(model_arch_tuple))
                else:
                    pre_processing_layers.append(None)
            return pre_processing_layers

        self._get_pre_processing_layers = _get_pre_processing_layers

        self.pre_processing_layers = _get_pre_processing_layers(state_encoder_pre_processing_network)
        for pre_processing_layer in self.pre_processing_layers:
            logger.debug(f"Pre-processing layer --- {pre_processing_layer}")
        if state_encoder_type is EncodingType.AUTOREGRESSIVE:
            hidden_units, activation = (state_encoder_network.hidden_units,
                                        get_activation_fn(state_encoder_network.activation))
            self.state_encoder_network = AutoRegressiveStateEncoderNetwork(
                state_shape=state_shape,
                activation=activation,
                hidden_units=hidden_units,
                latent_state_size=self.latent_state_size,
                atomic_prop_dims=self.atomic_prop_dims,
                time_stacked_states=self.time_stacked_states,
                temperature=self.state_encoder_temperature,
                time_stacked_lstm_units=self.time_stacked_lstm_units,
                pre_proc_net=self.pre_processing_layers,
                output_softclip=self.softclip)
        elif state_encoder_type is EncodingType.DETERMINISTIC:
            self.state_encoder_network = LatentStateEncoderNetwork(
                inputs=state,
                latent_state_size=latent_state_size,
                output_softclip=self.softclip,
                pre_proc_net=self.pre_processing_layers,
                atomic_prop_dims=self.atomic_prop_dims,
                latent_embedding_arch=state_encoder_network,
            )
            logger.debug(f"State Encoder Network  --- {self.state_encoder_network}")
        else:
            self.state_encoder_network = StateEncoderNetwork(
                state=state,
                activation=get_activation_fn(state_encoder_network.activation),
                hidden_units=state_encoder_network.hidden_units,
                latent_state_size=self.latent_state_size,
                atomic_prop_dims=self.atomic_prop_dims,
                time_stacked_states=self.time_stacked_states,
                time_stacked_lstm_units=self.time_stacked_lstm_units,
                pre_proc_net=self.pre_processing_layers,
                output_softclip=self.softclip,
                lstm_output=state_encoder_type is EncodingType.LSTM)
        # action encoder network
        if self.action_discretizer and not self.policy_based_decoding:
            self.action_encoder_network = ActionEncoderNetwork(
                latent_state=latent_state,
                action=action,
                number_of_discrete_actions=self.number_of_discrete_actions,
                action_encoder_network=generate_sequential_model(action_encoder_network), )
        else:
            self.action_encoder_network = None

        # transition network
        self.transition_network = AutoRegressiveBernoulliNetwork(
            event_shape=(self.latent_state_size,),
            activation=get_activation_fn(transition_network.activation),
            hidden_units=transition_network.hidden_units,
            conditional_event_shape=(self.latent_state_size + self.number_of_discrete_actions,),
            temperature=self.state_prior_temperature,
            name='autoregressive_transition_network')
        # stationary distribution over latent states
        if latent_stationary_network is not None:
            self.latent_stationary_network: AutoRegressiveBernoulliNetwork = SteadyStateNetwork(
                atomic_prop_dims=self.atomic_prop_dims,
                latent_state_size=latent_state_size,
                activation=get_activation_fn(latent_stationary_network.activation),
                hidden_units=latent_stationary_network.hidden_units,
                trainable_prior=trainable_prior,
                temperature=self.state_prior_temperature,
                name='latent_stationary_network')
        else:
            # in that case, need to be set
            self.latent_stationary_network = None

        # latent policy
        if self.external_latent_policy is None and latent_policy_network is not None:
            self.latent_policy_network = LatentPolicyNetwork(
                latent_state=latent_state,
                latent_policy_network=generate_sequential_model(latent_policy_network),
                number_of_discrete_actions=self.number_of_discrete_actions, )
            logger.debug(f"LatentPolicyNetwork --- {self.latent_policy_network}")
        else:
            self.latent_policy_network = None
        # reward function
        self.reward_network = RewardNetwork(
            latent_state=latent_state,
            latent_action=latent_action,
            next_latent_state=next_latent_state,
            reward_network=generate_sequential_model(reward_network),
            reward_shape=self.reward_shape)
        logger.debug(f"RewardNetwork --- {self.reward_network}")
        # state reconstruction function
        self._random_decoder = False
        if decoder_network is not None:
            self.reconstruction_network = StateReconstructionNetwork(
                latent_state=latent_state,
                decoder_network=generate_sequential_model(decoder_network),
                state_shape=self.state_shape,
                time_stacked_states=self.time_stacked_states,
                time_stacked_lstm_units=self.time_stacked_lstm_units)
        else:
            self.reconstruction_network = None

        # action reconstruction function
        if self.action_discretizer and not self.policy_based_decoding:
            self.action_reconstruction_network = ActionReconstructionNetwork(
                latent_state=latent_state,
                latent_action=latent_action,
                action_decoder_network=generate_sequential_model(action_decoder_network),
                action_shape=self.action_shape)
        else:
            self.action_reconstruction_network = None
        # steady state Lipschitz function
        self.steady_state_lipschitz_network = SteadyStateLipschitzFunction(
            latent_state=latent_state,
            latent_action=latent_action if not self.policy_based_decoding else None,
            next_latent_state=next_latent_state,
            steady_state_lipschitz_network=generate_sequential_model(steady_state_lipschitz_network),
            transition_based=distillation)
        logger.debug(f"Steady State Lipschitz Network --- {self.steady_state_lipschitz_network}")
        # transition loss Lipschitz function
        self._transition_loss_lipschitz_network_arch = transition_loss_lipschitz_network
        self._state_encoder_pre_processing_network = state_encoder_pre_processing_network

        self.transition_loss_lipschitz_network = self.create_transition_loss_lip_net()
        logger.debug(f"Transition Loss Lipschitz Network --- {self.transition_loss_lipschitz_network}")

        if kwargs.get('summary', True):
            self.summary()

        self.encoder_network = self.state_encoder_network

        self._base_architecture = BaseModelArchitecture(
            state_encoder_network=state_encoder_network,
            action_decoder_network=action_decoder_network,
            reward_network=reward_network,
            decoder_network=decoder_network,
            latent_policy_network=latent_policy_network,
            steady_state_lipschitz_network=steady_state_lipschitz_network,
            transition_loss_lipschitz_network=transition_loss_lipschitz_network,
            action_encoder_network=action_encoder_network,
            state_encoder_pre_processing_network=state_encoder_pre_processing_network,
            state_decoder_pre_processing_network=state_decoder_pre_processing_network)

        self.clip_by_global_norm = clip_by_global_norm

        self.loss_metrics = {
            'reconstruction_loss': Mean(name='reconstruction_loss'),
            'state_mse': MeanSquaredError(name='state_mse'),
            'action_mse': MeanSquaredError(name='action_mse'),
            'reward_mse': MeanSquaredError(name='reward_loss'),
            'transition_loss': Mean('transition_loss'),
            'latent_policy_entropy': Mean('latent_policy_entropy'),
            'steady_state_regularizer': Mean('steady_state_wasserstein_regularizer'),
            'gradient_penalty': Mean('gradient_penalty'),
            'marginal_state_encoder_entropy': Mean('marginal_state_encoder_entropy'),
            'transition_log_probs': Mean('transition_log_probs'),
            'gradients_max': Mean('gradients_max'),
            'gradients_min': Mean('gradients_min'),
            'marginal_variance': Mean('marginal_variance'),
            'score': Mean('score'),
        }
        if "RepeatPrevious" in (self._env_name or ''):
            self.loss_metrics["score"] = Mean('score')
            self._k = (int(state_shape[0][0]) - 1 - 4) // 4
            for i in range(self._k):
                self.loss_metrics[f"score_{i}"] = Mean(f"score_{i}")

        # if self.clip_by_global_norm:
        self.loss_metrics['gradients_max_grad_norm'] = Mean('gradients_max_grad_norm')
        self.loss_metrics['gradients_min_grad_norm'] = Mean('gradients_min_grad_norm')

        self._delete_tracking("loss_metrics")
        if n_states > 1:
            self.loss_metrics.pop("state_mse")
            for i in range(n_states):
                self.loss_metrics[f'state_{i:d}_mse'] = MeanSquaredError(name=f'state_{i:d}_mse')
        if self.include_state_encoder_entropy or self.include_action_encoder_entropy:
            self.loss_metrics['entropy_regularizer'] = Mean('entropy_regularizer')
        if state_encoder_type is not EncodingType.DETERMINISTIC:
            self.loss_metrics.update({
                'binary_encoding_log_probs': Mean('binary_encoding_log_probs'),
                'state_encoder_entropy': Mean('state_encoder_entropy'),
            })
        if self.policy_based_decoding or self._random_decoder:
            self.loss_metrics['marginal_variance'] = Mean(name='marginal_variance')
            if self._random_decoder:
                self.loss_metrics['variance'] = Mean(name='variance')
        elif self.action_discretizer:
            self.loss_metrics.update({
                'marginal_action_encoder_entropy': Mean('marginal_action_encoder_entropy'),
                'action_encoder_entropy': Mean('action_encoder_entropy'),
            })
            self.temperature_metrics.update({
                't_1_action': self.action_encoder_temperature,
                't_2_action': self.latent_policy_temperature,
            })

        self._score = Mean("wae_score")
        self._last_score = None

    def create_transition_loss_lip_net(self):
        transition_lip = TransitionLossLipschitzFunction(
            state=self._state_input,
            action=self._action_input,
            latent_state=self._latent_state_input,
            latent_action=self._latent_action_input if self.action_discretizer else None,
            next_latent_state=self._next_latent_state_input,
            transition_loss_lipschitz_network=generate_sequential_model(self._transition_loss_lipschitz_network_arch),
            pre_proc_net=self._get_pre_processing_layers(self._state_encoder_pre_processing_network, prefix='transition_loss'))
        return transition_lip

    @property
    def evaluation_window(self):
        return tf.expand_dims(self._score.result(), 0)

    def summary(self):
        for net in [
            self.state_encoder_network,
            self.latent_stationary_network,
            self.latent_policy_network,
            self.action_encoder_network,
            self.transition_network,
            self.reward_network,
            self.reconstruction_network,
            self.action_reconstruction_network,
            self.steady_state_lipschitz_network,
            self.transition_loss_lipschitz_network
        ]:
            if net is not None:
                net.summary()

    def anneal(self):
        super().anneal()
        for var, decay_rate in [
            (self._action_encoder_temperature, self.encoder_temperature_decay_rate),
            (self._latent_policy_temperature, self.prior_temperature_decay_rate),
        ]:
            if decay_rate.numpy().all() > 0:
                var.assign(var * (1. - decay_rate))

    def attach_optimizer(
            self,
            optimizers: Optional[Union[Tuple, List]] = None,
            autoencoder_optimizer: Optional = None,
            wasserstein_regularizer_optimizer: Optional = None
    ):
        assert optimizers is not None or (
                autoencoder_optimizer is not None and wasserstein_regularizer_optimizer is not None)
        if optimizers is not None:
            assert len(optimizers) == 2
            autoencoder_optimizer, wasserstein_regularizer_optimizer = optimizers
        self._autoencoder_optimizer = autoencoder_optimizer
        self._wasserstein_regularizer_optimizer = wasserstein_regularizer_optimizer

    def detach_optimizer(self):
        autoencoder_optimizer = self._autoencoder_optimizer
        wasserstein_regularizer_optimizer = self._wasserstein_regularizer_optimizer
        self._autoencoder_optimizer = None
        self._wasserstein_regularizer_optimizer = None
        return autoencoder_optimizer, wasserstein_regularizer_optimizer

    def binary_encode_state(self, state: Float, label: Optional[Float] = None) -> tfd.Distribution:
        return self.state_encoder_network.discrete_distribution(
            state=state, label=label)

    def relaxed_state_encoding(
            self,
            state: Float,
            temperature: Float,
            label: Optional[Float] = None,
            training: bool = False,
            *args, **kwargs
    ) -> tfd.Distribution:
        return self.state_encoder_network.relaxed_distribution(
            state=state, temperature=temperature, label=label, training=training)

    def discrete_action_encoding(
            self,
            latent_state: tf.Tensor,
            action: tf.Tensor,
    ) -> tfd.Distribution:
        if self.action_discretizer:
            return self.action_encoder_network.discrete_distribution(
                latent_state=latent_state, action=action)
        else:
            return tfd.Deterministic(loc=action)

    def relaxed_action_encoding(
            self,
            latent_state: tf.Tensor,
            action: tf.Tensor,
            temperature
    ) -> tfd.Distribution:
        if self.action_discretizer:
            return self.action_encoder_network.relaxed_distribution(
                latent_state=latent_state, action=action, temperature=temperature)
        else:
            return tfd.Deterministic(loc=action)

    def decode_state(self, latent_state: tf.Tensor, training: bool = False) -> tfd.Distribution:
        return self.reconstruction_network.distribution(latent_state=latent_state, training=training)

    def decode_action(
            self,
            latent_state: tf.Tensor,
            latent_action: tf.Tensor,
            *args, **kwargs
    ) -> tfd.Distribution:
        if self.action_discretizer:
            return self.action_reconstruction_network.distribution(
                latent_state=latent_state, latent_action=latent_action)
        else:
            return tfd.Deterministic(loc=latent_action)

    def action_generator(
            self,
            latent_state: Float
    ) -> tfd.Distribution:
        if self.action_discretizer:
            batch_size = tf.shape(latent_state)[0]
            loc = self.action_reconstruction_network([
                tf.repeat(latent_state, self.number_of_discrete_actions, axis=0),
                tf.tile(tf.eye(self.number_of_discrete_actions), [batch_size, 1])
            ])
            loc = tf.reshape(
                loc,
                tf.concat([[batch_size], [self.number_of_discrete_actions], self.action_shape], axis=-1))
            return tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(
                    logits=self.discrete_latent_policy(latent_state).logits_parameter()),
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=loc,
                    scale_diag=tf.ones(tf.shape(loc)) * 1e-6))
        else:
            return self.discrete_latent_policy(latent_state)

    def relaxed_latent_transition(
            self,
            latent_state: Float,
            latent_action: Float,
            temperature: Optional[Float] = None,
            *args, **kwargs
    ) -> tfd.Distribution:
        return self.transition_network.relaxed_distribution(
            conditional_input=tf.concat([latent_state, latent_action], axis=-1))

    def discrete_latent_transition(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor, *args, **kwargs
    ) -> tfd.Distribution:
        return self.transition_network.discrete_distribution(
            conditional_input=tf.concat([latent_state, latent_action], axis=-1))

    def relaxed_markov_chain_latent_transition(
            self, latent_state: tf.Tensor, temperature: float = 1e-5, reparamaterize: bool = True
    ) -> tfd.Distribution:
        return NotImplemented

    def discrete_markov_chain_latent_transition(
            self, latent_state: tf.Tensor
    ) -> tfd.Distribution:
        return NotImplemented

    def relaxed_latent_policy(
            self,
            latent_state: tf.Tensor,
            temperature: Float = 1e-5,
    ) -> tfd.Distribution:
        if self.external_latent_policy is not None:
            return self.external_latent_policy.distribution(
                ts.TimeStep(
                    observation=latent_state,
                    reward=0.,
                    discount=1.,
                    step_type=ts.StepType.MID
                )
            ).action
        else:
            return self.latent_policy_network.relaxed_distribution(
                latent_state=latent_state, temperature=temperature)

    def discrete_latent_policy(self, latent_state: tf.Tensor):
        if self.external_latent_policy is not None:
            return self.external_latent_policy.distribution(
                ts.TimeStep(
                    observation=latent_state,
                    reward=0.,
                    discount=1.,
                    step_type=ts.StepType.MID
                )
            ).action
        else:
            return self.latent_policy_network.discrete_distribution(latent_state=latent_state)

    def reward_distribution(
            self,
            latent_state: Float,
            latent_action: Float,
            next_latent_state: Float,
            training: bool = False,
            *args, **kwargs
    ) -> tfd.Distribution:
        return self.reward_network.distribution(
            latent_state=latent_state,
            latent_action=latent_action,
            next_latent_state=next_latent_state,
            training=training,
        )

    def markov_chain_reward_distribution(
            self,
            latent_state: Float,
            next_latent_state: Float,
    ) -> tfd.Distribution:
        batch_size = tf.shape(latent_state)[0]
        loc = self.reward_network([
            tf.repeat(latent_state, self.number_of_discrete_actions, axis=0),
            tf.tile(tf.eye(self.number_of_discrete_actions), [batch_size, 1]),
            tf.repeat(next_latent_state, self.number_of_discrete_actions, axis=0),
        ])
        loc = tf.reshape(loc, tf.concat([[batch_size], [self.number_of_discrete_actions], self.reward_shape], axis=-1))
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=self.discrete_latent_policy(latent_state).logits_parameter()),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=loc,
                scale_diag=tf.ones(tf.shape(loc)) * 1e-6))

    def discrete_latent_steady_state_distribution(
            self,
            batch_size: Optional[int] = None,
            *args, **kwargs) -> tfd.Distribution:
        if batch_size is None:
            return self.latent_stationary_network.discrete_distribution(*args, **kwargs)
        else:
            return tfd.BatchBroadcast(
                self.latent_stationary_network.discrete_distribution(*args, **kwargs),
                with_shape=[batch_size])

    def relaxed_latent_steady_state_distribution(
            self,
            batch_size: Optional[int] = None,
            *args, **kwargs
    ) -> tfd.Distribution:
        if batch_size is None:
            return self.latent_stationary_network.relaxed_distribution(*args, **kwargs)
        else:
            return tfd.BatchBroadcast(
                self.latent_stationary_network.relaxed_distribution(*args, **kwargs),
                with_shape=[batch_size])

    def _joint_relaxed_steady_state_distribution(self, batch_size: int) -> tfd.Distribution:
        """
        Retrieves the joint stationary distribution over latent states, actions, and successor states.
        """
        return tfd.JointDistributionSequential([
            self.relaxed_latent_steady_state_distribution(batch_size=batch_size),
            lambda _latent_state: self.relaxed_latent_policy(
                latent_state=_latent_state,
                temperature=self.latent_policy_temperature),
            lambda _latent_action, _latent_state: self.relaxed_latent_transition(
                _latent_state,
                _latent_action, ),
        ])

    def action_embedding_function(
            self,
            latent_state: tf.Tensor,
            latent_action: tf.Tensor,
    ) -> tf.Tensor:

        if self.action_discretizer:
            decoder = self.decode_action(
                latent_state=tf.cast(latent_state, dtype=tf.float32),
                latent_action=tf.cast(
                    tf.one_hot(
                        latent_action,
                        depth=self.number_of_discrete_actions),
                    dtype=tf.float32), )
            if self.deterministic_state_embedding:
                return decoder.mode()
            else:
                return decoder.sample()
        else:
            return latent_action

    @staticmethod
    def norm(x: Float, axis: Union[int, List[int], Tuple[int, ...]] = -1):
        """
        to replace tf.norm(x, order=2, axis) which has numerical instabilities (the derivative can yields NaN).
        """
        return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis) + epsilon)

    def cost(self, x: Float, y: Float, space: str):  
        if space != 'state' or self._cost_fn_state is None:
            cost_fn = get_cost_fn(self._cost_fn)
            return sum([
                cost_fn(x_i, y_i) * w for x_i, y_i, w in zip(
                    tf.nest.flatten(x), tf.nest.flatten(y), tf.nest.flatten(self._cost_weights[space]))
            ])
        else:
            return self._cost_fn_state(x, y, self._cost_weights["state"])

    @tf.function
    def __call__(
            self,
            state: Float,
            label: Float,
            action: Float,
            reward: Float,
            next_state: Float,
            next_label: Float,
            sample_key: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
            training: bool = False,
            *args, **kwargs
    ):
        batch_size = tf.shape(action)[0]
        # encoder sampling
        latent_state = self.relaxed_state_encoding(
            state,
            label=label,
            temperature=self.state_encoder_temperature,
            training=training,
        ).sample()
        next_latent_state = self.relaxed_state_encoding(
            next_state,
            label=next_label,
            temperature=self.state_encoder_temperature,
            training=training,
        ).sample()

        if self.policy_based_decoding:
            latent_action = self.relaxed_latent_policy(
                latent_state,
                temperature=self.latent_policy_temperature
            ).sample()
        else:
            latent_action = self.relaxed_action_encoding(
                latent_state,
                action,
                temperature=self.action_encoder_temperature
            ).sample()  # note that latent_action = action when self.action_discretizer is False

        (stationary_latent_state,
         stationary_latent_action,
         next_stationary_latent_state) = self._joint_relaxed_steady_state_distribution(
            batch_size=batch_size
        ).sample()

        # next latent state from the latent transition function
        next_transition_latent_state = self.relaxed_latent_transition(
            latent_state,
            latent_action,
        ).sample()

        # reconstruction loss
        decoder_distribution = tfd.JointDistributionSequential([
            self.decode_state(latent_state, training=training),
            self.decode_action(
                latent_state,
                latent_action),
            self.reward_distribution(
                latent_state,
                latent_action,
                next_latent_state,
                training=training),
            self.decode_state(next_latent_state, training=training)
        ])

        if self.policy_based_decoding:
            mean_decoder_fn = tfd.JointDistributionSequential([
                self.decode_state(latent_state),
                self.action_generator(latent_state),
                self.markov_chain_reward_distribution(latent_state, next_latent_state),
                self.decode_state(next_latent_state)
            ]).mean
        else:
            mean_decoder_fn = decoder_distribution.mean

        decoder_fn = decoder_distribution.sample

        if not self.policy_based_decoding and not self.enforce_upper_bound:
            _state, _action, _reward, _next_state = decoder_fn()
        else:
            _state, _action, _reward, _next_state = mean_decoder_fn()

        reconstruction_loss = (
                self.cost(state, _state, 'state') +
                self.cost(action, _action, 'action') +
                self.cost(reward, _reward, 'reward') +
                self.cost(next_state, _next_state, 'state')
        )

        out = tf.concat([self._recover_fn_obs(tf.stop_gradient(_state[1])) - state[1],
                         self._recover_fn_obs(tf.stop_gradient(_next_state[1])) - next_state[1]], axis=0)
        var = tf.math.reduce_mean(tf.square(out), axis=0)
        # tf.print(var)
        self._obs_variance.assign_add(var)
        self._obs_variance_counter.assign_add(1)
        # tf.print(self._obs_variance)
        # self._obs_variance.append(var)
        if "RepeatPrevious" in (self._env_name or ''):
            rs = tf.concat([state[0], next_state[0]], axis=0)
            ps = tf.concat([_state[0], _next_state[0]], axis=0)
            rs = tf.reshape(rs[:, 1:-4], (-1, self._k, 4))
            ps = tf.reshape(ps[:, 1:-4], (-1, self._k, 4))
            rs = tf.argmax(rs, axis=-1)
            ps = tf.argmax(ps, axis=-1)
            score = tf.reduce_mean(tf.cast(rs == ps, dtype=tf.float32), 0)
            for i in range(self._k):
                self.loss_metrics[f"score_{i}"](score[i])
            self.loss_metrics["score"](tf.reduce_mean(score))

        if self.squared_wasserstein or self.policy_based_decoding or self._random_decoder:
            reconstruction_loss = tf.square(reconstruction_loss)

        if self.policy_based_decoding or self._random_decoder:
            # marginal variance of the reconstruction
            if self.enforce_upper_bound:
                random_action, random_reward = _action, _reward
                random_state, _action, _reward, random_next_state = mean_decoder_fn()
            else:
                random_state, random_action, random_reward, random_next_state = decoder_fn()

            def flatten(_x):
                return tf.concat(
                    [tf.reshape(_component, [tf.shape(_component)[0], -1]) for _component in tf.nest.flatten(_x)],
                    axis=-1)

            random_state = flatten(random_state)
            random_next_state = flatten(random_next_state)
            y = tf.concat([random_state, random_action, random_reward, random_next_state], axis=-1)
            mean = tf.concat([flatten(state), _action, _reward, flatten(_next_state)], axis=-1)
            # marginal variances over all (flat) dimensions
            marginal_variance = tf.reduce_sum(
                # Var(Y) = E_Z [ Var(Y | Z) ] + Var(E[Y | Z]) ; Z = <latent_state, next_latent_state>
                #
                # E_Z[ Var(Y | Z) ] = E_Z E_{Y | Z} [ Y - mean(Y | Z) ]
                tf.square(y - mean) +
                # Var(E[Y | Z]) = E_Z [ mean(Y | Z) - mean(mean(Y | Z)) )]
                tf.square(mean - tf.reduce_mean(mean, axis=0)),
                axis=-1)
        else:
            random_state, random_action, random_reward, random_next_state = \
                _state, _action, _reward, _next_state
            marginal_variance = 0.

        # Wasserstein regularizers and Lipschitz constraint
        if self.policy_based_decoding:
            x = [latent_state, next_transition_latent_state]
            y = [stationary_latent_state, next_stationary_latent_state]
        else:
            x = [latent_state, latent_action, next_transition_latent_state]
            y = [stationary_latent_state, stationary_latent_action, next_stationary_latent_state]
        steady_state_regularizer = tf.squeeze(
            self.steady_state_lipschitz_network(x) - self.steady_state_lipschitz_network(y))
        # steady_state_gradient_penalty = self.compute_gradient_penalty(
        #     x=tf.concat(x, axis=-1),
        #     y=tf.concat(y, axis=-1),
        #     lipschitz_function=lambda _x: self.steady_state_lipschitz_network(
        #         [_x[:, :self.latent_state_size, ...]] +
        #         (
        #             [_x[:, self.latent_state_size: self.latent_state_size + self.number_of_discrete_actions, ...]]
        #             if not self.policy_based_decoding else
        #             []
        #         ) +
        #         [_x[:, -self.latent_state_size:, ...]]))
        steady_state_gradient_penalty = self.compute_gradient_penalty(x, y, self.steady_state_lipschitz_network)

        if self.action_discretizer:
            x = [state, action, latent_state, latent_action, next_latent_state]
            y = [state, action, latent_state, latent_action, next_transition_latent_state]
        else:
            x = [state, action, latent_state, next_latent_state]
            y = [state, action, latent_state, next_transition_latent_state]
        transition_loss_regularizer = tf.squeeze(
            self.transition_loss_lipschitz_network(x) - self.transition_loss_lipschitz_network(y))
        # transition_loss_gradient_penalty = self.compute_gradient_penalty(
        #     x=next_latent_state,
        #     y=next_transition_latent_state,
        #     lipschitz_function=lambda _x: self.transition_loss_lipschitz_network(x[:-1] + [_x]))
        transition_loss_gradient_penalty = self.compute_gradient_penalty(
            [next_latent_state],
            [next_transition_latent_state],
            lambda _x: self.transition_loss_lipschitz_network(x[:-1] + [_x]))

        logits = self.state_encoder_network.get_logits(state, latent_state)
        # entropy_regularizer = self.entropy_regularizer(
        #     state=state,
        #     latent_state=latent_state,
        #     logits=logits,
        #     action=action if not self.policy_based_decoding else None,
        #     include_state_entropy=self.include_state_encoder_entropy,
        #     include_action_entropy=self.include_action_encoder_entropy,
        #     sample_probability=sample_probability, )

        # priority support
        # if self.priority_handler is not None and sample_key is not None:
        #     tf.stop_gradient(
        #         self.priority_handler.update_priority(
        #             keys=sample_key,
        #             latent_states=tf.stop_gradient(tf.cast(tf.round(latent_state), tf.int32)),
        #             loss=tf.stop_gradient(reconstruction_loss +
        #                                   marginal_variance)))

        # loss metrics
        self.loss_metrics['reconstruction_loss'](reconstruction_loss)
        if "state_mse" in self.loss_metrics:
            self.loss_metrics['state_mse'](state, _state)
            self.loss_metrics['state_mse'](next_state, _next_state)
        else:
            for x, y in [(state, _state), (next_state, _next_state)]:
                for i in range(len(x)):
                    self.loss_metrics[f'state_{i:d}_mse'](
                        x[i], [self._recover_fn_state, self._recover_fn_obs][i % 2](y[i]))
        self.loss_metrics['action_mse'](action, random_action)
        self.loss_metrics['reward_mse'](reward, random_reward)
        self.loss_metrics['transition_loss'](transition_loss_regularizer)
        self.loss_metrics['steady_state_regularizer'](steady_state_regularizer)
        self.loss_metrics['gradient_penalty'](
            steady_state_gradient_penalty + transition_loss_gradient_penalty)
        self.loss_metrics['marginal_state_encoder_entropy'](
            self.marginal_state_encoder_entropy(logits=logits, sample_probability=sample_probability))
        if self._state_encoder_type is not EncodingType.DETERMINISTIC:
            self.loss_metrics['state_encoder_entropy'](
                tfd.Independent(
                    tfd.Bernoulli(logits=logits),
                    reinterpreted_batch_ndims=1
                ).entropy())
        if self.latent_policy_network is not None and self.external_latent_policy is not None:
            self.loss_metrics['latent_policy_entropy'](
                self.discrete_latent_policy(latent_state).entropy())
        self.loss_metrics['transition_log_probs'](
            #  self.discrete_latent_transition(
            #      latent_state=tf.round(latent_state),
            #      latent_action=tf.one_hot(
            #          tf.argmax(latent_action, axis=-1),
            #          depth=self.number_of_discrete_actions)
            #  ).log_prob(tf.round(next_latent_state))
            self.relaxed_latent_transition(
                latent_state=latent_state,
                latent_action=latent_action,
                temperature=self.prior_temperature
            ).log_prob(tf.clip_by_value(next_latent_state, 1e-7, 1 - 1e-7)))
        if self._state_encoder_type is not EncodingType.DETERMINISTIC:
            self.loss_metrics['binary_encoding_log_probs'](
                self.binary_encode_state(
                    state=state
                ).log_prob(tf.round(latent_state)[..., self.atomic_prop_dims:]))
        if self.action_discretizer and not self.policy_based_decoding:
            self.loss_metrics['marginal_action_encoder_entropy'](
                self.marginal_action_encoder_entropy(latent_state, action))
            self.loss_metrics['action_encoder_entropy'](
                self.discrete_action_encoding(latent_state, action).entropy())
        elif self.policy_based_decoding or self._random_decoder:
            self.loss_metrics['marginal_variance'](marginal_variance)
            #  if self._random_decoder:
            #      self.loss_metrics['variance'](flatten(decoder_distribution.variance()))
        # if self.include_state_encoder_entropy or self.include_action_encoder_entropy:
        #     self.loss_metrics['entropy_regularizer'](entropy_regularizer)
        # dynamic reward scaling
        self._dynamic_reward_scaling.assign(
            tf.math.minimum(
                self._dynamic_reward_scaling,
                tf.pow(2. * tf.reduce_max(tf.abs(reward)), -1.)))

        if debug:
            tf.print("latent_state", latent_state, summarize=-1)
            tf.print("next_latent_state", next_latent_state, summarize=-1)
            tf.print("next_stationary_latent_state", next_stationary_latent_state, summarize=-1)
            tf.print("next_transition_latent_state", next_transition_latent_state, summarize=-1)
            tf.print("latent_action", latent_action, summarize=-1)
            tf.print("loss", tf.stop_gradient(
                reconstruction_loss + marginal_variance +
                self.wasserstein_regularizer_scale_factor.stationary.scaling * steady_state_regularizer +
                self.wasserstein_regularizer_scale_factor.local_transition_loss.scaling * transition_loss_regularizer))

        return {
            'reconstruction_loss': reconstruction_loss,  # + marginal_variance,
            'steady_state_regularizer': steady_state_regularizer,
            'steady_state_gradient_penalty': steady_state_gradient_penalty,
            'transition_loss_regularizer': transition_loss_regularizer,
            'transition_loss_gradient_penalty': transition_loss_gradient_penalty,
            # 'entropy_regularizer': entropy_regularizer,
        }

    def marginal_state_encoder_entropy(
            self,
            state: Optional[Float] = None,
            latent_state: Optional[Float] = None,
            logits: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
    ) -> Float:

        if logits is None:
            if state is None or latent_state is None:
                raise ValueError("A state and its encoding (i.e., as a latent state) "
                                 "should be provided when logits are not.")

            logits = self.state_encoder_network.get_logits(state, latent_state)

        if sample_probability is None:
            regularizer = tf.reduce_mean(
                - tf.sigmoid(logits) * tf.math.log(tf.reduce_mean(tf.sigmoid(logits), axis=0) + epsilon)
                - tf.sigmoid(-logits) * tf.math.log(tf.reduce_mean(tf.sigmoid(-logits), axis=0) + epsilon),
                axis=0)
        else:
            is_weights = (tf.stop_gradient(tf.reduce_min(sample_probability)) / sample_probability) ** self.is_exponent
            regularizer = tf.reduce_mean(
                - tf.sigmoid(logits) * tf.math.log(
                    tf.reduce_mean(tf.expand_dims(is_weights, -1) * tf.sigmoid(logits), axis=0) + epsilon)
                - tf.sigmoid(-logits) * tf.math.log(
                    tf.reduce_mean(tf.expand_dims(is_weights, -1) * tf.sigmoid(-logits), axis=0) + epsilon),
                axis=0)
        return tf.reduce_sum(regularizer)

    def marginal_action_encoder_entropy(
            self,
            latent_state: Optional[Float] = None,
            action: Optional[Float] = None,
            logits: Optional[Float] = None,
    ) -> Float:
        if logits is None and (latent_state is None or action is None):
            raise ValueError("You should either provide the logits of the action distribution or a latent state"
                             " and an action to compute the marginal entropy")
        if logits is None:
            logits = self.discrete_action_encoding(latent_state, action).logits_parameter()
        batch_size = tf.cast(tf.shape(logits)[0], tf.float32)
        return -1. * tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.softmax(logits) * (
                        tf.reduce_logsumexp(
                            logits - tf.expand_dims(
                                tf.reduce_logsumexp(logits, axis=-1),
                                axis=-1),
                            axis=0) - tf.math.log(batch_size)),
                axis=-1),
            axis=0)

    @tf.function
    def entropy_regularizer(
            self,
            state: tf.Tensor,
            label: Optional[Float] = None,
            latent_state: Optional[Float] = None,
            logits: Optional[Float] = None,
            action: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
            include_state_entropy: bool = True,
            include_action_entropy: bool = True,
            *args, **kwargs
    ) -> Float:
        if latent_state is None:
            if label is None:
                raise ValueError("either a latent state or a label should be provided")
            else:
                latent_state = self.relaxed_state_encoding(
                    state, label=label, temperature=self.state_encoder_temperature)

        regularizer = 0.

        if include_state_entropy:
            if logits is None:
                logits = self.state_encoder_network.get_logits(state, latent_state)
            regularizer += self.marginal_state_encoder_entropy(
                logits=logits,
                sample_probability=sample_probability)
            regularizer -= tfd.Independent(
                tfd.Bernoulli(logits=logits),
                reinterpreted_batch_ndims=1
            ).entropy()

        if include_action_entropy:
            if action is None or not self.action_discretizer:
                regularizer += self.action_entropy_regularizer_scaling * tf.reduce_mean(
                    self.discrete_latent_policy(latent_state).entropy(),
                    axis=0)
            else:
                logits = self.discrete_action_encoding(latent_state, action).logits_parameter()
                regularizer += self.action_entropy_regularizer_scaling * (
                        self.marginal_action_encoder_entropy(logits=logits) -
                        tf.reduce_mean(tfd.Categorical(logits=logits).entropy(), axis=0))
        return regularizer

    @tf.function
    def compute_gradient_penalty_old(
            self,
            x: Float,
            y: Float,
            lipschitz_function: Callable[[Float], Float],
    ):
        print('re_tracing compute_gradient_penalty')
        noise = tf.random.uniform(shape=(tf.shape(x)[0], 1), minval=0., maxval=1.)
        straight_lines = noise * x + (1. - noise) * y
        print(tf.gradients(lipschitz_function(straight_lines), straight_lines))
        gradients = tf.gradients(lipschitz_function(straight_lines), straight_lines)[0]
        return tf.square(self.norm(gradients, axis=1) - 1.)

    @staticmethod
    @tf.function
    def compute_gradient_penalty(
            x: Float,
            y: Float,
            lipschitz_function: Callable[[Float], Float],
    ):
        print('re_tracing compute_gradient_penalty')
        dims = (1, ) * (len(tf.shape(x[0])) - 1)
        noise = tf.random.uniform(shape=(tf.shape(x[0])[0], *dims), minval=0., maxval=1.)
        straight_lines = [noise * x_ + (1. - noise) * y_ for x_, y_ in zip(x, y)]
        with tf.GradientTape() as tape:
            tape.watch(straight_lines)
            output = lipschitz_function(straight_lines)
        gradients = tape.gradient(output, straight_lines)
        gradients = tf.concat([grad_ for grad_ in gradients if grad_ is not None], axis=1)
        gradients = tf.reshape(gradients, (tf.shape(gradients)[0], -1))
        return tf.square(get_cost_fn('norm2')(gradients, axis=1) - 1.)

    def eval(
            self,
            state: Float,
            label: Float,
            action: Float,
            reward: Float,
            next_state: Float,
            next_label: Float,
            sample_probability: Optional[Float] = None,
            additional_transition_batch: Optional[Tuple[Float, ...]] = None,
            *args, **kwargs
    ):
        batch_size = tf.shape(state)[0]
        # sampling
        # encoders
        latent_state = self.binary_encode_state(state, label).sample()
        next_latent_state = self.binary_encode_state(next_state, next_label).sample()
        if self.policy_based_decoding:
            latent_action = tf.cast(self.discrete_latent_policy(latent_state).sample(), tf.float32)
        else:
            latent_action = tf.cast(self.discrete_action_encoding(latent_state, action).sample(), tf.float32)

        # latent steady-state distribution
        stationary_latent_state = self.discrete_latent_steady_state_distribution().sample(batch_size)
        stationary_latent_action = self.discrete_latent_policy(stationary_latent_state).sample()
        next_stationary_latent_state = self.discrete_latent_transition(
            latent_state=stationary_latent_state,
            latent_action=stationary_latent_action
        ).sample()
        next_stationary_latent_state = tf.cast(next_stationary_latent_state, tf.float32)

        # next latent state from the latent transition function
        next_transition_latent_state = self.discrete_latent_transition(
            latent_state,
            latent_action,
        ).sample()

        # reconstruction loss
        # the reward as well as the state and action reconstruction functions are deterministic
        _action, _reward, _next_state = tfd.JointDistributionSequential([
            self.decode_action(
                latent_state,
                latent_action) if not self.policy_based_decoding else
            tfd.Deterministic(loc=self.action_generator(latent_state).mean()),
            self.reward_distribution(
                latent_state,
                latent_action,
                next_latent_state) if not self.policy_based_decoding else
            tfd.Deterministic(loc=self.markov_chain_reward_distribution(latent_state, next_latent_state).mean()),
            self.decode_state(next_latent_state)
        ]).sample()

        reconstruction_loss = (
                tf.norm(action - _action, ord=2, axis=1) +
                tf.norm(reward - _reward, ord=2, axis=1) +
                tf.norm(next_state - _next_state, ord=2, axis=1))
        if self.policy_based_decoding or self.squared_wasserstein:
            reconstruction_loss = tf.square(reconstruction_loss)

        # marginal variance of the reconstruction
        if self.policy_based_decoding:
            random_action, random_reward = tfd.JointDistributionSequential([
                self.decode_action(latent_state, latent_action),
                self.reward_distribution(latent_state, latent_action, next_latent_state),
            ]).sample()
            y = tf.concat([random_action, random_reward, _next_state], axis=-1)
            mean = tf.concat([_action, _reward, _next_state], axis=-1)
            marginal_variance = tf.reduce_sum((y - mean) ** 2. + (mean - tf.reduce_mean(mean, axis=-1)) ** 2., axis=-1)
        else:
            marginal_variance = 0.

        # Wasserstein regularizers and Lipschitz constraint
        if self.policy_based_decoding:
            x = [latent_state, next_transition_latent_state]
            y = [stationary_latent_state, next_stationary_latent_state]
        else:
            x = [latent_state, latent_action, next_transition_latent_state]
            y = [stationary_latent_state, stationary_latent_action, next_stationary_latent_state]
        steady_state_regularizer = tf.squeeze(
            self.steady_state_lipschitz_network(x) - self.steady_state_lipschitz_network(y))

        if self.action_discretizer:
            x = [state, action, latent_state, latent_action, next_latent_state]
            y = [state, action, latent_state, latent_action, next_transition_latent_state]
        else:
            x = [state, action, latent_state, next_latent_state]
            y = [state, action, latent_state, next_transition_latent_state]
        transition_loss_regularizer = tf.squeeze(
            self.transition_loss_lipschitz_network(x) - self.transition_loss_lipschitz_network(y))

        if debug:
            latent_policy = self.discrete_latent_policy(latent_state)
            tf.print("latent policy", latent_policy,
                     '\n latent policy: probs parameter', latent_policy.probs_parameter())
            tf.print("latent action ~ latent policy", latent_policy.sample())
            tf.print("latent_action hist:", tf.cast(tf.argmax(latent_action, axis=1), tf.int64))

        return {
            'reconstruction_loss': reconstruction_loss + marginal_variance,
            'wasserstein_regularizer':
                (self.wasserstein_regularizer_scale_factor.stationary.scaling * steady_state_regularizer +
                 self.wasserstein_regularizer_scale_factor.local_transition_loss.scaling * transition_loss_regularizer),
            'latent_states': tf.concat([tf.cast(latent_state, tf.int64), tf.cast(next_latent_state, tf.int64)], axis=0),
            'latent_actions': (tf.cast(tf.argmax(latent_action, axis=1), tf.int64)
                               if self.action_discretizer else
                               tf.cast(tf.argmax(stationary_latent_action, axis=1), tf.int64))
        }

    @tf.function
    def compute_loss(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
            sample_probability: Optional[tf.Tensor] = None,
            additional_transition_batch: Optional[Tuple[Float]] = None,
            *args, **kwargs
    ):
        output = self(state, label, action, reward, next_state, next_label,
                      sample_key=sample_key,
                      sample_probability=sample_probability,
                      additional_transition_batch=additional_transition_batch,
                      training=True)

        if debug:
            tf.print('call output', output, summarize=-1)

        # Importance sampling weights (is) for prioritized experience replay
        if sample_probability is not None:
            is_weights = (tf.stop_gradient(tf.reduce_min(sample_probability)) / sample_probability) ** self.is_exponent
        else:
            is_weights = 1.

        reconstruction_loss = output['reconstruction_loss']
        wasserstein_loss = (
                self.wasserstein_regularizer_scale_factor.stationary.scaling *
                output['steady_state_regularizer'] +
                self.wasserstein_regularizer_scale_factor.local_transition_loss.scaling *
                output['transition_loss_regularizer']
        )
        gradient_penalty = (
                self.wasserstein_regularizer_scale_factor.stationary.scaling *
                self.wasserstein_regularizer_scale_factor.stationary.gradient_penalty_multiplier *
                output['steady_state_gradient_penalty'] +
                self.wasserstein_regularizer_scale_factor.local_transition_loss.scaling *
                self.wasserstein_regularizer_scale_factor.local_transition_loss.gradient_penalty_multiplier *
                output['transition_loss_gradient_penalty']
        )

        if self.include_state_encoder_entropy:
            entropy_regularizer = self.entropy_regularizer_scale_factor * output['entropy_regularizer']
        elif self.include_action_encoder_entropy:
            entropy_regularizer = output['entropy_regularizer']
        else:
            entropy_regularizer = 0.

        loss = lambda minimize: tf.reduce_mean(
            (-1.) ** (1. - minimize) * is_weights * (
                    minimize * reconstruction_loss +
                    wasserstein_loss +
                    (minimize - 1.) * gradient_penalty -
                    minimize * entropy_regularizer),
            axis=0)

        return {'min': loss(1.), 'max': loss(0.)}

    @property
    def state_encoder_temperature(self):
        return self.encoder_temperature

    @property
    def state_prior_temperature(self):
        return self.prior_temperature

    @property
    def action_encoder_temperature(self):
        return self._action_encoder_temperature

    @action_encoder_temperature.setter
    def action_encoder_temperature(self, value):
        self._action_encoder_temperature = tf.Variable(
            value, dtype=tf.float32, trainable=False, name='action_encoder_temperature')

    @property
    def latent_policy_temperature(self):
        return self._latent_policy_temperature

    @latent_policy_temperature.setter
    def latent_policy_temperature(self, value):
        self._latent_policy_temperature = tf.Variable(
            value, dtype=tf.float32, trainable=False, name='latent_policy_temperature')

    @property
    def inference_variables(self):
        if self.action_discretizer and not self.policy_based_decoding:
            return self.state_encoder_network.trainable_variables + self.action_encoder_network.trainable_variables
        else:
            return self.state_encoder_network.trainable_variables

    @property
    def generator_variables(self):
        variables = []
        if self.action_discretizer:
            variables += self.action_reconstruction_network.trainable_variables
        for network in [self.latent_stationary_network,
                        self.transition_network,
                        self.latent_policy_network,
                        self.reward_network,
                        self.reconstruction_network]:
            if network is not None:
                variables += network.trainable_variables
        return variables

    @property
    def wasserstein_variables(self):
        return (self.steady_state_lipschitz_network.trainable_variables +
                self.transition_loss_lipschitz_network.trainable_variables)

    def _compute_apply_gradients(
            self, state, label, action, reward, next_state, next_label,
            autoencoder_variables=None, wasserstein_regularizer_variables=None,
            sample_key=None, sample_probability=None,
            additional_transition_batch=None,
            step: Int = None,
            *args, **kwargs
    ):
        if autoencoder_variables is None and wasserstein_regularizer_variables is None:
            raise ValueError("Must pass autoencoder and/or wasserstein regularizer variables")
        if step is None:
            step = self.n_critic

        def numerical_error(x, list_of_tensors=False):
            detected = False
            if not list_of_tensors:
                x = [x]
            for value in x:
                if value is not None:
                    detected = detected or tf.reduce_any(tf.logical_or(
                        tf.math.is_nan(value),
                        tf.math.is_inf(value)))
            return detected

        with tf.GradientTape(persistent=True) as tape:
            loss = self.compute_loss(
                state, label, action, reward, next_state, next_label,
                sample_key=sample_key, sample_probability=sample_probability, )

        for optimization_direction, variables in {
            'max': wasserstein_regularizer_variables, 'min': autoencoder_variables
        }.items():
            if (
                    variables is not None and
                    (not debug or not numerical_error(loss[optimization_direction])) and
                    (optimization_direction == 'max' or
                     (step % self.n_critic == 0 and optimization_direction == 'min'))
            ):
                gradients = tape.gradient(loss[optimization_direction], variables)
                optimizer = {
                    'max': self._wasserstein_regularizer_optimizer,
                    'min': self._autoencoder_optimizer,
                }[optimization_direction]

                if self.clip_by_global_norm:
                    gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_by_global_norm)

                if not numerical_error(gradients, list_of_tensors=True):
                    if optimizer is not None:
                        optimizer.apply_gradients(zip(gradients, variables))

                if 'gradients_' + optimization_direction in self.loss_metrics.keys():
                    mean_abs_grads = tf.concat(
                        [tf.reshape(tf.abs(grad), [-1]) for grad in gradients
                         if grad is not None],
                        axis=-1)
                    self.loss_metrics['gradients_' + optimization_direction](mean_abs_grads)

                if self.clip_by_global_norm and (
                        'gradients_' + optimization_direction + '_grad_norm' in self.loss_metrics.keys()):
                    self.loss_metrics['gradients_' + optimization_direction + '_grad_norm'](grad_norm)

                if debug_gradients:
                    for gradient, variable in zip(gradients, variables):
                        tf.print("Gradient for {} (direction={}):".format(variable.name, optimization_direction),
                                 gradient)

        del tape

        return {'loss_minimizer': loss['min'], 'loss_maximizer': loss['max']}

    @tf.function
    def compute_apply_gradients(
            self,
            state: Float,
            label: Float,
            action: Float,
            reward: Float,
            next_state: Float,
            next_label: Float,
            sample_key: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
            additional_transition_batch: Optional[Tuple[Float]] = None,
            step: Int = None,
    ):
        print("wasserstein_mdp compute and apply gradients")
        return self._compute_apply_gradients(
            state, label, action, reward, next_state, next_label,
            autoencoder_variables=self.inference_variables + self.generator_variables,
            wasserstein_regularizer_variables=self.wasserstein_variables,
            sample_key=sample_key, sample_probability=sample_probability,
            additional_transition_batch=additional_transition_batch,
            step=step)

    @tf.function
    def inference_update(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
            sample_probability: Optional[tf.Tensor] = None,
    ):
        return self._compute_apply_gradients(
            state, label, action, reward, next_state, next_label,
            autoencoder_variables=self.generator_variables,
            wasserstein_regularizer_variables=self.wasserstein_variables,
            sample_key=sample_key, sample_probability=sample_probability)

    @tf.function
    def generator_update(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
            sample_probability: Optional[tf.Tensor] = None,
    ):
        return self._compute_apply_gradients(
            state, label, action, reward, next_state, next_label,
            autoencoder_variables=self.generator_variables,
            wasserstein_regularizer_variables=self.wasserstein_variables,
            sample_key=sample_key, sample_probability=sample_probability)

    def mean_latent_bits_used(self, inputs, eps=1e-3, deterministic=True):
        state, label, action, reward, next_state, next_label = inputs[:6]
        latent_distribution = self.binary_encode_state(state, label)
        latent_state = latent_distribution.sample()
        if deterministic:
            mean = tf.reduce_mean(latent_distribution.mode(), axis=0)
        else:
            mean = tf.reduce_mean(latent_distribution.mean(), axis=0)
        check = lambda x: 1. if 1. - eps > x > eps else 0.
        mbu = {'mean_state_bits_used': tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()}
        if self.action_discretizer:
            mean = tf.reduce_mean(
                self.discrete_action_encoding(latent_state, action).probs_parameter()
                if not self.policy_based_decoding else
                self.discrete_latent_policy(latent_state).probs_parameter(),
                axis=0)
            check = lambda x: 1 if 1 - eps > x > eps else 0
            mean_bits_used = tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()

            mbu.update({'mean_action_bits_used': mean_bits_used})
        return mbu

    def estimate_local_losses_from_samples(
            self,
            environment: tf_py_environment.TFPyEnvironment,
            steps: int,
            labeling_function: Callable[[tf.Tensor], tf.Tensor],
            estimate_transition_function_from_samples: bool = False,
            assert_estimated_transition_function_distribution: bool = False,
            replay_buffer_max_frames: Optional[int] = int(1e5),
            reward_scaling: Optional[float] = 1.,
            estimate_value_difference: bool = True,
            *args, **kwargs
    ):
        if self.time_stacked_states:
            labeling_function = lambda x: labeling_function(x)[:, -1, ...]

        return estimate_local_losses_from_samples(
            environment=environment,
            latent_policy=self.get_latent_policy(action_dtype=tf.int64),
            steps=steps,
            latent_state_size=self.latent_state_size,
            number_of_discrete_actions=self.number_of_discrete_actions,
            state_embedding_function=self.state_embedding_function,
            probabilistic_state_embedding=None if self.deterministic_state_embedding else self.binary_encode_state,
            action_embedding_function=self.action_embedding_function,
            latent_reward_function=lambda latent_state, latent_action, next_latent_state: (
                self.reward_distribution(
                    latent_state=tf.cast(latent_state, dtype=tf.float32),
                    latent_action=tf.cast(latent_action, dtype=tf.float32),
                    next_latent_state=tf.cast(next_latent_state, dtype=tf.float32),
                ).mode()),
            labeling_function=labeling_function,
            latent_transition_function=lambda _latent_state, _latent_action: self.discrete_latent_transition(
                tf.cast(_latent_state, tf.float32), tf.cast(_latent_action, tf.float32)),
            estimate_transition_function_from_samples=estimate_transition_function_from_samples,
            replay_buffer_max_frames=replay_buffer_max_frames,
            reward_scaling=reward_scaling,
            atomic_prop_dims=self.atomic_prop_dims,
            estimate_value_difference=estimate_value_difference)

    def eval_and_save(
            self,
            eval_steps: int,
            global_step: tf.Variable,
            dataset: Optional = None,
            dataset_iterator: Optional = None,
            batch_size: Optional[int] = None,
            save_directory: Optional[str] = None,
            log_name: Optional[str] = None,
            train_summary_writer: Optional[tf.summary.SummaryWriter] = None,
            eval_policy_driver: Optional[tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver] = None,
            local_losses_estimator: Optional[Callable] = None,
            *args, **kwargs
    ):

        if (dataset is None) == (dataset_iterator is None or batch_size is None):
            raise ValueError("Must either provide a dataset or a dataset iterator + batch size.")

        if dataset is not None:
            batch_size = eval_steps
            dataset_iterator = iter(dataset.batch(
                batch_size=batch_size,
                drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
            eval_steps = 1 if eval_steps > 0 else 0

        metrics = {
            'eval_loss': tf.metrics.Mean(),
            'eval_reconstruction_loss': tf.metrics.Mean(),
            'eval_wasserstein_regularizer': tf.metrics.Mean(),
        }

        data = {'states': None, 'actions': None}
        score = dict()
        local_losses_metrics = None

        if eval_steps > 0:
            eval_progressbar = Progbar(
                target=(eval_steps + 1) * batch_size, interval=0.1, stateful_metrics=['eval_ELBO'])

            tf.print("\nEvalutation over {} step(s)".format(eval_steps))

            for step in range(eval_steps):
                x = next(dataset_iterator)
                if self._sample_additional_transition:
                    x_prime = next(dataset_iterator)
                else:
                    x_prime = None

                if len(x) >= 8:
                    sample_probability = x[7]
                    # we consider is_exponent=1 for evaluation
                    is_weights = tf.reduce_min(sample_probability) / sample_probability
                else:
                    sample_probability = None
                    is_weights = 1.

                evaluation = self.eval(
                    *(x[:6]), sample_probability=sample_probability, additional_transition_batch=x_prime)
                for value in ('states', 'actions'):
                    latent = evaluation['latent_' + value]
                    data[value] = latent if data[value] is None else tf.concat([data[value], latent], axis=0)
                for value in ('loss', 'reconstruction_loss', 'wasserstein_regularizer'):
                    if value == 'loss':
                        metrics['eval_' + value](tf.reduce_mean(
                            is_weights * (evaluation['reconstruction_loss'] + evaluation['wasserstein_regularizer'])))
                    else:
                        metrics['eval_' + value](tf.reduce_mean(is_weights * evaluation[value]))
                eval_progressbar.add(batch_size, values=[('eval_loss', metrics['eval_loss'].result())])
            tf.print('\n')

        if eval_policy_driver is not None:
            score['eval_policy'] = self.eval_policy(
                eval_policy_driver=eval_policy_driver,
                train_summary_writer=train_summary_writer,
                global_step=global_step
            ).numpy()

        if local_losses_estimator is not None:
            local_losses_metrics = local_losses_estimator()

        if train_summary_writer is not None and eval_steps > 0:
            with train_summary_writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(key, value.result(), step=global_step)
                for value in ('states', 'actions'):
                    if data[value] is not None:
                        if value == 'states':
                            data[value] = tf.reduce_sum(
                                data[value] * 2 ** tf.range(tf.cast(self.latent_state_size, dtype=tf.int64)),
                                axis=-1)
                        tf.summary.histogram('{}_frequency'.format(value[:-1]), data[value],
                                             step=global_step, buckets=32)
                if local_losses_metrics is not None:
                    tf.summary.scalar('local_reward_loss', local_losses_metrics.local_reward_loss, step=global_step)
                    tf.summary.scalar('local_transition_loss',
                                      local_losses_metrics.local_transition_loss, step=global_step)
                    if local_losses_metrics.local_transition_loss_transition_function_estimation is not None:
                        tf.summary.scalar('local_transition_loss_empirical_transition_function',
                                          local_losses_metrics.local_transition_loss_transition_function_estimation,
                                          step=global_step)
                    for key, value in local_losses_metrics.value_difference.items():
                        tf.summary.scalar(key, value, step=global_step)

        if local_losses_metrics is not None:
            tf.print('Local reward loss: {:.2f}'.format(local_losses_metrics.local_reward_loss))
            tf.print('Local transition loss: {:.2f}'.format(local_losses_metrics.local_transition_loss))
            tf.print('Local transition loss (empirical transition function): {:.2f}'
                     ''.format(local_losses_metrics.local_transition_loss_transition_function_estimation))
            score['local_reward_loss'] = local_losses_metrics.local_reward_loss.numpy()
            score['local_transition_loss'] = local_losses_metrics.local_transition_loss.numpy()
            if local_losses_metrics.local_transition_loss_transition_function_estimation is not None and \
                    local_losses_metrics.local_transition_loss_transition_function_estimation \
                    < local_losses_metrics.local_transition_loss:
                score['local_transition_loss'] = \
                    local_losses_metrics.local_transition_loss_transition_function_estimation.numpy()

            for key, value in local_losses_metrics.value_difference.items():
                tf.print(key, value)
            local_losses_metrics.print_time_metrics()

        if eval_steps > 0:
            print('eval loss: ', metrics['eval_loss'].result().numpy())

        if eval_policy_driver is not None:
            self.assign_score(
                score=score,
                checkpoint_model=save_directory is not None,
                save_directory=save_directory,
                model_name='model',
                training_step=global_step.numpy())

        gc.collect()

        return metrics['eval_loss'].result()

    def save(self, save_directory, model_name: str, infos: Optional[Dict] = None, *args, **kwargs):
        import os
        import json

        if infos is None:
            infos = dict()
        else:
            for key, value in infos.items():
                infos[key] = str(value)

        save_path = os.path.join(save_directory, model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save model variables through checkpointing
        optimizer = self.detach_optimizer()
        priority_handler = self.priority_handler
        self.priority_handler = None
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.save(os.path.join(save_path, 'ckpt'))
        self.attach_optimizer(optimizer)
        self.priority_handler = priority_handler

        # dump model infos
        with open(os.path.join(save_path, 'model_infos.json'), 'w') as file:
            json.dump({**self._params, **infos}, file)

        print('Model saved to:', save_path)

    def assign_score(
            self,
            score: Dict[str, float],
            checkpoint_model: bool,
            save_directory: str,
            model_name: str,
            training_step: int,
            save_best_only: bool = True,
    ):
        self._score(score['eval_policy'])
        score['training_step'] = training_step
        self._last_score = score['eval_policy']
        print("assigning score:", score['eval_policy'])

        if checkpoint_model:
            print("save model...")
            import os
            if save_best_only and os.path.exists(os.path.join(save_directory, model_name, 'model_infos.json')):
                with open(os.path.join(save_directory, model_name, 'model_infos.json'), 'r') as f:
                    infos = json.load(f)
                eval_policy = float(infos.get('eval_policy', -1. * np.inf))
                local_transition_loss = float(infos.get('local_transition_loss', np.inf))
                local_reward_loss = float(infos.get('local_reward_loss', np.inf))
                print(
                    "current best model:, eval_policy={:.2f}, local_transitition_loss={:.2f}, local_reward_loss={:.2f}".format(
                        eval_policy, local_transition_loss, local_reward_loss))
                if score['eval_policy'] > eval_policy:
                    print(score['eval_policy'], "better")
                    self.save(save_directory, model_name, score)
                elif np.abs(eval_policy - score['eval_policy']) < epsilon and (
                        'local_transition_loss' in score.keys() and 'local_reward_loss' in score.keys()):
                    if score['local_transition_loss'] < local_transition_loss:
                        print("local_transition_loss better:", score['local_transition_loss'])
                        self.save(save_directory, model_name, score)
                    elif np.abs(score['local_transition_loss'] - local_transition_loss) < epsilon and (
                            score['local_reward_loss'] < local_reward_loss):
                        print("local_reward_loss better:", score['local_reward_loss'])
                        self.save(save_directory, model_name, score)
            else:
                print("saving model")
                self.save(save_directory, model_name, score)


def load(model_path: str):
    with open(os.path.join(model_path, 'model_infos.json'), 'r') as f:
        infos = json.load(f)

    params = dict()
    for key, value in infos.items():
        try:
            params[key] = eval(value)
        except NameError:
            params[key] = value
        except SyntaxError:
            pass

    model = WassersteinMarkovDecisionProcess(**params)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(os.path.join(model_path, 'ckpt-1'))

    return model
