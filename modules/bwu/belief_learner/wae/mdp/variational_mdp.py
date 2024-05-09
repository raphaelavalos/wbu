import os
from collections import namedtuple
import enum
from enum import Enum
from typing import Tuple, Optional, Callable, Dict, Iterator, NamedTuple, List
import numpy as np
import psutil
from absl import logging

from tf_agents.environments.wrappers import TimeLimit
# from keras.saving.saved_model import utils

from belief_learner.utils.old.reinforcement_learning.environments.latent_environment import LatentEmbeddingTFEnvironmentWrapper
from belief_learner.utils.old.reinforcement_learning.environments.no_reward_shaping import NoRewardShapingWrapper
from belief_learner.utils.old.reinforcement_learning.environments.perturbed_env import PerturbedEnvironment
from belief_learner.utils.old.io.video import VideoEmbeddingObserver
from belief_learner.verification.model import TransitionFnDecorator

try:
    import reverb
except ImportError as ie:
    print(ie, "Reverb is not installed on your system, "
              "meaning prioritized experience replay cannot be used.")
import time
import datetime
import gc

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Concatenate, Reshape, Dense, Lambda
from tensorflow.keras.utils import Progbar

import tf_agents.policies.tf_policy
import tf_agents.agents.tf_agent
from tensorflow.keras.layers import TimeDistributed, LSTM
from tensorflow.python.keras.models import Sequential
from tf_agents import specs
from tf_agents.policies import tf_policy, py_tf_eager_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.drivers import dynamic_step_driver, py_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment, tf_environment, py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer, reverb_replay_buffer, reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.policy_step import PolicyStep

from belief_learner.utils.old.policies.latent_policy import LatentPolicyOverRealStateAndActionSpaces
from belief_learner.utils.old.policies.time_stacked_states import TimeStackedStatesPolicyWrapper
from belief_learner.utils.old.policies.epsilon_mimic import EpsilonMimicPolicy
from belief_learner.utils.old.reinforcement_learning.environments import EnvironmentLoader
from belief_learner.utils.old.io.dataset_generator import map_rl_trajectory_to_vae_input, ergodic_batched_labeling_function
from belief_learner.utils.old.replay_buffer_tools import PriorityBuckets, LossPriority, PriorityHandler
from belief_learner.verification.local_losses import estimate_local_losses_from_samples
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb

debug = False
debug_verbosity = -1
debug_gradients = False
check_numerics = False

if check_numerics:
    tf.debugging.enable_check_numerics(stack_height_limit=150)

epsilon = 1e-12


class DatasetComponents(NamedTuple):
    replay_buffer: tf_agents.replay_buffers.replay_buffer.ReplayBuffer
    driver: tf_agents.drivers.py_driver.PyDriver
    initial_collect_driver: tf_agents.drivers.py_driver.PyDriver
    close_fn: Callable
    replay_buffer_num_frames_fn: Callable[[], int]
    wrapped_manager: Optional[tf.train.CheckpointManager]
    dataset: tf.data.Dataset
    dataset_iterator: Iterator
    epsilon_greedy: tf.Variable


class EvaluationCriterion(Enum):
    MAX = enum.auto()
    MEAN = enum.auto()


class VariationalMarkovDecisionProcess(tf.Module):

    def __init__(
            self,
            state_shape: Tuple[int, ...],
            action_shape: Tuple[int, ...],
            reward_shape: Tuple[int, ...],
            label_shape: Tuple[int, ...],
            encoder_network: Model,
            transition_network: Model,
            reward_network: Model,
            decoder_network: Model,
            label_transition_network: Optional[Model] = None,
            latent_policy_network: Optional[Model] = None,
            state_encoder_pre_processing_network: Optional[Model] = None,
            state_decoder_pre_processing_network: Optional[Model] = None,
            time_stacked_states: bool = False,
            latent_state_size: int = 12,
            encoder_temperature: float = 2. / 3,
            prior_temperature: float = 1. / 2,
            encoder_temperature_decay_rate: float = 0.,
            prior_temperature_decay_rate: float = 0.,
            entropy_regularizer_scale_factor: float = 0.,
            entropy_regularizer_decay_rate: float = 0.,
            entropy_regularizer_scale_factor_min_value: float = 0.,
            marginal_entropy_regularizer_ratio: float = 0.,
            kl_scale_factor: float = 1.,
            kl_annealing_growth_rate: float = 0.,
            mixture_components: int = 3,
            max_decoder_variance: Optional[float] = None,
            multivariate_normal_raw_scale_diag_activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.softplus,
            multivariate_normal_full_covariance: bool = False,
            pre_loaded_model: bool = False,
            reset_state_label: bool = True,
            latent_policy_training_phase: bool = False,
            full_optimization: bool = True,
            optimizer: Optional = None,
            evaluation_window_size: int = 5,
            evaluation_criterion: EvaluationCriterion = EvaluationCriterion.MAX,
            action_label_transition_network: Optional[Model] = None,
            action_transition_network: Optional[Model] = None,
            importance_sampling_exponent: Optional[float] = 1.,
            importance_sampling_exponent_growth_rate: Optional[float] = 0.,
            time_stacked_lstm_units: int = 128,
            reward_bounds: Optional[Tuple[float, float]] = None,
            deterministic_state_embedding: bool = True,
    ):

        super(VariationalMarkovDecisionProcess, self).__init__()

        self.state_shape = state_shape
        self.action_shape = action_shape
        self.reward_shape = reward_shape
        self.latent_state_size = latent_state_size
        self.label_shape = label_shape
        self.atomic_prop_dims = np.prod(label_shape) + int(reset_state_label)
        self.mixture_components = mixture_components
        self.full_covariance = multivariate_normal_full_covariance
        self.latent_policy_training_phase = latent_policy_training_phase
        self.full_optimization = full_optimization
        self._optimizer = optimizer
        self.time_stacked_states = time_stacked_states
        self.time_stacked_lstm_units = time_stacked_lstm_units
        self.deterministic_state_embedding = deterministic_state_embedding

        # initialize all tf variables
        self._entropy_regularizer_scale_factor = None
        self._kl_scale_factor = None
        self._initial_kl_scale_factor = None
        self._kl_scale_factor_decay = None
        self._is_exponent = None
        self._initial_is_exponent = None
        self._is_exponent_decay = None
        self._is_exponent_growth_rate = None

        self.encoder_temperature = encoder_temperature
        self.prior_temperature = prior_temperature
        self.entropy_regularizer_scale_factor = (
                entropy_regularizer_scale_factor - entropy_regularizer_scale_factor_min_value)
        self.kl_scale_factor = kl_scale_factor
        self.encoder_temperature_decay_rate = encoder_temperature_decay_rate
        self.prior_temperature_decay_rate = prior_temperature_decay_rate
        self.entropy_regularizer_decay_rate = entropy_regularizer_decay_rate
        self.kl_growth_rate = kl_annealing_growth_rate
        self.max_decoder_variance = max_decoder_variance
        self.is_exponent = importance_sampling_exponent
        self.is_exponent_growth_rate = importance_sampling_exponent_growth_rate

        self.scale_activation = multivariate_normal_raw_scale_diag_activation
        self.entropy_regularizer_scale_factor_min_value = tf.constant(entropy_regularizer_scale_factor_min_value)
        self.marginal_entropy_regularizer_ratio = marginal_entropy_regularizer_ratio

        self.number_of_discrete_actions = -1  # only used if a latent policy network is provided

        try:
            state = Input(shape=state_shape, name="state")
        except TypeError:
            pass

        action = Input(shape=action_shape, name="action")
        self._encoder_softclip = tfb.SoftClip(low=-10., high=10.)  # , hinge_softness=10.)

        if reward_bounds is not None:
            if len(reward_bounds) != 2 or reward_bounds[0] > reward_bounds[1]:
                raise ValueError("Please provide valid reward bounds."
                                 "Values provided: {}".format(str(reward_bounds)))
            self._reward_softclip = tfb.SoftClip(low=reward_bounds[0], high=reward_bounds[1])
        else:
            self._reward_softclip = None

        # the evaluation window contains eiter the N max evaluation scores encountered during training if the evaluation
        # criterion is MAX, or the N last evaluation scores encountered if the evaluation criterion is MEAN.
        self.evaluation_criterion = evaluation_criterion
        self._evaluation_window = tf.Variable(
            initial_value=-1. * np.inf * tf.ones(shape=(evaluation_window_size,)),
            trainable=False,
            name='evaluation_window')

        self._sample_additional_transition = False
        self.priority_handler = None

        if not pre_loaded_model:

            # Encoder network
            if time_stacked_states:
                if state_encoder_pre_processing_network is not None:
                    encoder = TimeDistributed(state_encoder_pre_processing_network)(state)
                else:
                    encoder = state
                encoder = LSTM(units=time_stacked_lstm_units)(encoder)
                encoder = encoder_network(encoder)
            else:
                if state_encoder_pre_processing_network is not None:
                    _state = state_encoder_pre_processing_network(state)
                else:
                    _state = state
                encoder = encoder_network(_state)
            logits_layer = Dense(
                units=latent_state_size - self.atomic_prop_dims,
                # allows avoiding exploding logits values and probability errors after applying a sigmoid
                activation=lambda x: self._encoder_softclip(x),
                name='encoder_latent_distribution_logits'
            )(encoder)
            self.encoder_network = Model(
                inputs=state,
                outputs=logits_layer,
                name='state_encoder')

            # Latent policy network
            latent_state = Input(shape=(latent_state_size,), name="latent_state")
            if latent_policy_network is not None:
                self.latent_policy_network = latent_policy_network(latent_state)
                # we assume actions to be discrete and given in one hot when using a latent policy network
                assert len(self.action_shape) == 1
                self.number_of_discrete_actions = self.action_shape[0]
                self.latent_policy_network = Dense(
                    units=self.number_of_discrete_actions,
                    activation=None,
                    name='latent_policy_one_hot_logits'
                )(self.latent_policy_network)
                self.latent_policy_network = Model(
                    inputs=latent_state,
                    outputs=self.latent_policy_network,
                    name='latent_policy_network')
            else:
                self.latent_policy_network = None
                self.number_of_discrete_actions = -1

            # Transition network
            # inputs are binary concrete random variables, outputs are locations of logistic distributions
            next_label = Input(shape=(self.atomic_prop_dims,), name='next_label')
            if self.number_of_discrete_actions != -1:
                if label_transition_network is not None:
                    transition_network_input = Concatenate(name='transition_network_input')(
                        [latent_state, next_label])
                else:
                    transition_network_input = latent_state
                _transition_network = transition_network(transition_network_input)
                no_latent_state_logits = latent_state_size - self.atomic_prop_dims
                transition_output_layer = Dense(
                    units=no_latent_state_logits * self.number_of_discrete_actions,
                    activation=None,
                    name='transition_network_raw_output_layer'
                )(_transition_network)
                transition_output_layer = Reshape(
                    target_shape=(no_latent_state_logits, self.number_of_discrete_actions),
                    name='transition_network_output_layer_reshape'
                )(transition_output_layer)

                action_transition_output = transition_output_layer

                _action = tf.keras.layers.RepeatVector(
                    int(no_latent_state_logits), name='transition_output_repeat_action')(action)
                transition_output_layer = tf.keras.layers.Multiply(name="multiply_action_transition")(
                    [_action, transition_output_layer])
                transition_output_layer = Lambda(
                    lambda x: tf.reduce_sum(x, axis=-1),
                    name='transition_logits_reduce_sum_action_mask_layer'
                )(transition_output_layer)
            else:
                if label_transition_network is not None:
                    transition_network_input = Concatenate(
                        name="transition_network_input")([latent_state, action, next_label])
                else:
                    transition_network_input = Concatenate(
                        name="transition_network_input")([latent_state, action])
                _transition_network = transition_network(transition_network_input)
                action_transition_output = None
                transition_output_layer = Dense(
                    units=latent_state_size - self.atomic_prop_dims,
                    activation=None,
                    name='latent_transition_distribution_logits'
                )(_transition_network)

            # Label transition network
            # Gives logits of a Bernoulli distribution giving the probability of the next label given the
            # current latent state and the action chosen
            if self.number_of_discrete_actions != -1:
                if label_transition_network is not None:
                    _label_transition_network = label_transition_network(latent_state)
                else:
                    _label_transition_network = _transition_network
                _label_transition_network = Dense(
                    units=self.atomic_prop_dims * self.number_of_discrete_actions,
                    activation=None,
                    name="label_transition_network_raw_output_layer"
                )(_label_transition_network)
                _label_transition_network = Reshape(
                    target_shape=(self.atomic_prop_dims, self.number_of_discrete_actions),
                    name='reshape_label_transition_output'
                )(_label_transition_network)

                self.action_label_transition_network = Model(
                    inputs=latent_state,
                    outputs=_label_transition_network,
                    name='action_label_transition_network')
                self.action_transition_network = Model(
                    inputs=[latent_state, next_label] if label_transition_network is not None else latent_state,
                    outputs=(action_transition_output if label_transition_network is not None else
                             [_label_transition_network, action_transition_output]),
                    name="action_transition_network")

                _action = tf.keras.layers.RepeatVector(
                    int(self.atomic_prop_dims),
                    name='label_transition_output_repeat_action')(action)
                _label_transition_network = tf.keras.layers.Multiply()([_action, _label_transition_network])
                _label_transition_network = Lambda(
                    lambda x: tf.reduce_sum(x, axis=-1),
                    name='label_transition_reduce_sum_action_mask_layer'
                )(_label_transition_network)
            else:
                if label_transition_network is not None:
                    label_transition_network_input = Concatenate(
                        name="label_transition_network_input")([latent_state, action])
                    _label_transition_network = label_transition_network(label_transition_network_input)
                else:
                    _label_transition_network = _transition_network
                _label_transition_network = Dense(
                    units=self.atomic_prop_dims,
                    activation=None,
                    name='next_label_transition_logits'
                )(_label_transition_network)
            self.label_transition_network = Model(
                inputs=[latent_state, action],
                outputs=_label_transition_network,
                name='label_transition_network')
            self.transition_network = Model(
                inputs=([latent_state, action, next_label] if label_transition_network is not None else
                        [latent_state, action]),
                outputs=(transition_output_layer if label_transition_network is not None else
                         [_label_transition_network, transition_output_layer]),
                name="transition_network")

            # Reward network
            next_latent_state = Input(shape=(latent_state_size,), name="next_latent_state")
            if self.number_of_discrete_actions != -1:
                reward_network_input = Concatenate(name="reward_network_input")(
                    [latent_state, next_latent_state])
                _reward_network = reward_network(reward_network_input)
                reward_mean = Dense(
                    units=np.prod(reward_shape) * self.number_of_discrete_actions,
                    activation=None if self._reward_softclip is None else lambda x: self._reward_softclip(x),
                    name='reward_mean_raw_output')(_reward_network)
                reward_mean = Reshape(target_shape=(reward_shape + (self.number_of_discrete_actions,)))(reward_mean)
                _action = tf.keras.layers.RepeatVector(int(np.prod(reward_shape)))(action)
                _action = Reshape(target_shape=(reward_shape + (self.number_of_discrete_actions,)))(_action)
                reward_mean = tf.keras.layers.Multiply(name="multiply_action_reward_stack")(
                    [_action, reward_mean])
                reward_mean = Lambda(
                    lambda x: tf.reduce_sum(x, axis=-1), name='reward_mean_reduce_sum_action_mask_layer'
                )(reward_mean)
                reward_raw_covar = Dense(
                    units=np.prod(reward_shape) * self.number_of_discrete_actions,
                    activation=None,
                    name='reward_covar_raw_output')(_reward_network)
                reward_raw_covar = Reshape(
                    target_shape=reward_shape + (self.number_of_discrete_actions,))(reward_raw_covar)
                reward_raw_covar = tf.keras.layers.Multiply(
                    name='multiply_action_raw_covar_stack')([_action, reward_raw_covar])
                reward_raw_covar = Lambda(
                    lambda x: tf.reduce_sum(x, axis=-1),
                    name='reward_raw_covar_reduce_sum_action_mask_layer'
                )(reward_raw_covar)
            else:
                reward_network_input = Concatenate(name="reward_network_input")(
                    [latent_state, action, next_latent_state])
                _reward_network = reward_network(reward_network_input)
                reward_mean = Dense(
                    units=np.prod(reward_shape),
                    activation=None if self._reward_softclip is None else lambda x: self._reward_softclip(x),
                    name='reward_mean_0')(_reward_network)
                reward_raw_covar = Dense(
                    units=np.prod(reward_shape),
                    activation=None,
                    name='state_reward_raw_diag_covariance_0'
                )(_reward_network)
            reward_mean = Reshape(reward_shape, name='reward_mean')(reward_mean)
            reward_raw_covar = Reshape(
                reward_shape, name='reward_raw_diag_covariance')(reward_raw_covar)
            self.reward_network = Model(
                inputs=[latent_state, action, next_latent_state],
                outputs=[reward_mean, reward_raw_covar],
                name='reward_network')

            # Reconstruction network
            decoder = decoder_network(next_latent_state)

            if time_stacked_states:
                time_dimension = state_shape[0]
                _state_shape = state_shape[1:]

                if decoder.shape[-1] % time_dimension != 0:
                    decoder = Dense(
                        units=decoder.shape[-1] + time_dimension - decoder.shape[-1] % time_dimension)(decoder)

                decoder = Reshape(target_shape=(time_dimension, decoder.shape[-1] // time_dimension))(decoder)
                decoder = LSTM(units=self.time_stacked_lstm_units, return_sequences=True)(decoder)

                if state_decoder_pre_processing_network is not None:
                    decoder = TimeDistributed(state_decoder_pre_processing_network)(decoder)

            else:
                if state_decoder_pre_processing_network is not None:
                    decoder = state_decoder_pre_processing_network(decoder)
                _state_shape = state_shape

            # 1 mean per dimension, nb Normal Gaussian
            decoder_output_mean = Sequential([
                Dense(
                    units=mixture_components * np.prod(_state_shape),
                    activation=None,
                    name='state_decoder_GMM_mean_0'),
                Reshape(
                    target_shape=(mixture_components,) + _state_shape,
                    name="state_decoder_GMM_mean")],
                name="state_decoder_mean")

            if self.full_covariance and len(_state_shape) == 1:
                d = np.prod(_state_shape) * (np.prod(_state_shape) + 1) / 2
                decoder_raw_output = Sequential([
                    Dense(
                        units=mixture_components * d,
                        activation=None,
                        name='state_decoder_GMM_tril_params_0'),
                    Reshape(
                        target_shape=(mixture_components, d,),
                        name='state_decoder_GMM_tril_params_1'),
                    Lambda(
                        lambda x: tfb.FillScaleTriL()(x),
                        name='state_decoder_GMM_scale_tril')],
                    name='state_decoder_scale_tril')
            else:
                # n diagonal co-variance matrices
                decoder_raw_output = Sequential([
                    Dense(
                        units=mixture_components * np.prod(_state_shape),
                        activation=None,
                        name='state_decoder_GMM_raw_diag_covariance_0'),
                    Reshape(
                        (mixture_components,) + _state_shape,
                        name="state_decoder_GMM_raw_diag_covar")],
                    name='state_decoder_raw_diag_covar')

            # prior distribution over the mixture components
            decoder_prior = Sequential([
                Dense(
                    units=mixture_components,
                    activation='softmax',
                    name="state_decoder_GMM_priors")])

            if time_stacked_states:
                decoder_output_mean = TimeDistributed(decoder_output_mean)(decoder)
                decoder_raw_output = TimeDistributed(decoder_raw_output)(decoder)
                decoder_prior = TimeDistributed(decoder_prior)(decoder)
            else:
                decoder_output_mean = decoder_output_mean(decoder)
                decoder_raw_output = decoder_raw_output(decoder)
                decoder_prior = decoder_prior(decoder)

            self.reconstruction_network = Model(
                inputs=next_latent_state,
                outputs=[decoder_output_mean, decoder_raw_output, decoder_prior],
                name='state_reconstruction_network')

        else:
            self.encoder_network = encoder_network
            self.transition_network = transition_network
            self.label_transition_network = label_transition_network
            self.reward_network = reward_network
            self.reconstruction_network = decoder_network
            self.latent_policy_network = latent_policy_network
            self.number_of_discrete_actions = self.action_shape[0] if self.latent_policy_network is not None else -1
            self.action_label_transition_network = action_label_transition_network
            self.action_transition_network = action_transition_network

        self.loss_metrics = {
            'ELBO': tf.keras.metrics.Mean(name='ELBO'),
            'state_mse': tf.keras.metrics.MeanSquaredError(name='state_mse'),
            'reward_mse': tf.keras.metrics.MeanSquaredError(name='reward_mse'),
            'distortion': tf.keras.metrics.Mean(name='distortion'),
            'rate': tf.keras.metrics.Mean(name='rate'),
            # 'annealed_rate': tf.keras.metrics.Mean(name='annealed_rate'),
            'entropy_regularizer': tf.keras.metrics.Mean(name='entropy_regularizer'),
            'encoder_entropy': tf.keras.metrics.Mean(name='encoder_entropy'),
            'marginal_encoder_entropy': tf.keras.metrics.Mean(name='marginal_encoder_entropy'),
            'transition_log_probs': tf.keras.metrics.Mean(name='transition_log_probs'),
            #  'decoder_variance': tf.keras.metrics.Mean(name='decoder_variance')
        }
        self.temperature_metrics = {
            't_1': self.encoder_temperature,
            't_2': self.prior_temperature}
        self._dynamic_reward_scaling = tf.Variable(1., trainable=False)

    def reset_metrics(self):
        for value in self.loss_metrics.values():
            value.reset_states()
        #  super().reset_metrics()

    def attach_optimizer(self, optimizer):
        self._optimizer = optimizer

    def detach_optimizer(self):
        optimizer = self._optimizer
        self._optimizer = None
        return optimizer

    def relaxed_state_encoding(
            self, state: tf.Tensor, temperature: float, label: Optional[tf.Tensor] = None, *args, **kwargs
    ) -> tfd.Distribution:
        """
        Embed the input state and its label (if given) into a Binary Concrete probability distribution over
        a relaxed binary latent representation of the latent state space.
        Note: the Binary Concrete distribution is replaced by a Logistic distribution to avoid underflow issues:
              z ~ BinaryConcrete(logits, temperature) = sigmoid(z_logistic)
              with z_logistic ~ Logistic(loc=logits/temperature, scale=1./temperature))
        """
        logits = self.encoder_network(state)
        if label is not None:
            logits = tf.concat([(label * 2. - 1.) * 1e2, logits], axis=-1)
        return tfd.Independent(
            tfd.Logistic(
                loc=logits / temperature,
                scale=1. / temperature,
                allow_nan_stats=False, ))

    def binary_encode_state(self, state: tf.Tensor, label: Optional[tf.Tensor] = None) -> tfd.Distribution:
        """
        Embed the input state and its label (if given) into a Bernoulli probability distribution over the binary
        representation of the latent state space.
        """
        logits = self.encoder_network(state)
        if label is not None:
            logits = tf.concat([(label * 2. - 1.) * 1e2, logits], axis=-1)
        return tfd.Independent(
            tfd.Bernoulli(
                logits=logits,
                allow_nan_stats=False))

    def decode_state(self, latent_state: tf.Tensor) -> tfd.Distribution:
        """
        Decode a binary latent state to a probability distribution over states of the original MDP.
        """
        [
            reconstruction_mean, reconstruction_raw_covariance, reconstruction_prior_components
        ] = self.reconstruction_network(latent_state)
        if self.max_decoder_variance is None:
            reconstruction_raw_covariance = self.scale_activation(reconstruction_raw_covariance)
        else:
            reconstruction_raw_covariance = tfp.bijectors.SoftClip(
                low=epsilon, high=self.max_decoder_variance ** 0.5).forward(reconstruction_raw_covariance)

        if self.mixture_components == 1:
            reconstruction_mean = (reconstruction_mean[:, 0, ...] if not self.time_stacked_states
                                   else reconstruction_mean[:, :, 0, ...])
            reconstruction_raw_covariance = (reconstruction_raw_covariance[:, 0, ...] if not self.time_stacked_states
                                             else reconstruction_raw_covariance[:, :, 0, ...])
            if self.full_covariance:
                decoder_distribution = tfd.MultivariateNormalTriL(
                    loc=reconstruction_mean,
                    scale_tril=reconstruction_raw_covariance,
                    allow_nan_stats=False, )
            else:
                decoder_distribution = tfd.MultivariateNormalDiag(
                    loc=reconstruction_mean,
                    scale_diag=reconstruction_raw_covariance,
                    allow_nan_stats=False)
        else:
            if self.full_covariance:
                decoder_distribution = tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=reconstruction_prior_components),
                    components_distribution=tfd.MultivariateNormalTriL(
                        loc=reconstruction_mean,
                        scale_tril=reconstruction_raw_covariance,
                        allow_nan_stats=False
                    ),
                    allow_nan_stats=False)
            else:
                decoder_distribution = tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=reconstruction_prior_components),
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=reconstruction_mean,
                        scale_diag=reconstruction_raw_covariance,
                        allow_nan_stats=False
                    ),
                    allow_nan_stats=False)

        if self.time_stacked_states:
            return tfd.Independent(decoder_distribution)
        else:
            return decoder_distribution

    def relaxed_markov_chain_latent_transition(
            self, latent_state: tf.Tensor, temperature: float = 1e-5) -> tfd.Distribution:
        if self._has_dedicated_label_transition_network:
            next_label_logits = self.action_label_transition_network(latent_state)
            next_state_logits = lambda _next_label: self.action_transition_network([latent_state, _next_label])
        else:
            next_label_logits, _next_state_logits = self.action_transition_network(latent_state)
            next_state_logits = lambda _: _next_state_logits

        return tfd.JointDistributionSequential([
            tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=self.latent_policy_network(latent_state)),
                components_distribution=tfd.Independent(
                    tfd.Bernoulli(
                        logits=tf.transpose(next_label_logits, perm=[0, 2, 1]),
                        allow_nan_stats=False, ),
                    reinterpreted_batch_ndims=1),
                allow_nan_stats=False,
            ), lambda _next_label: tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=self.latent_policy_network(latent_state)),
                components_distribution=tfd.Independent(tfd.Logistic(
                    loc=tf.transpose(next_state_logits(_next_label), perm=[0, 2, 1]) / temperature,
                    scale=1. / temperature,
                    allow_nan_stats=False, ), reinterpreted_batch_ndims=1),
                allow_nan_stats=False)])

    def relaxed_latent_transition(
            self, latent_state: tf.Tensor, action: tf.Tensor, next_label: Optional[tf.Tensor] = None,
            temperature: float = 1e-5, *args, **kwargs
    ) -> tfd.Distribution:
        """
        Retrieves a Binary Concrete probability distribution P(z'|z, a) over successor latent states, given a latent
        state z given in relaxed binary representation and an action a.
        Note: the Binary Concrete distribution is replaced by a Logistic distribution to avoid underflow issues:
              z ~ BinaryConcrete(logits, temperature) = sigmoid(z_logistic)
              with z_logistic ~ Logistic(loc=logits / temperature, scale=1. / temperature))
        """
        if self._has_dedicated_label_transition_network:
            next_label_logits = self.label_transition_network([latent_state, action])
            next_state_logits = lambda _next_label: self.transition_network([latent_state, action, _next_label])
        else:
            next_label_logits, _next_state_logits = self.transition_network([latent_state, action])
            next_state_logits = lambda _: _next_state_logits

        if next_label is not None:
            _next_state_logits = next_state_logits(next_label)
            return tfd.Independent(
                tfd.Logistic(
                    loc=_next_state_logits / temperature,
                    scale=1. / temperature,
                    allow_nan_stats=False),
                allow_nan_stats=False, )
        else:
            return tfd.JointDistributionSequential([
                tfd.Independent(
                    tfd.Bernoulli(
                        logits=next_label_logits,
                        allow_nan_stats=False,
                        name='label_transition_distribution'),
                    allow_nan_stats=False),
                lambda _next_label: tfd.Independent(
                    tfd.Logistic(
                        loc=next_state_logits(_next_label) / temperature,
                        scale=1. / temperature,
                        allow_nan_stats=False),
                    allow_nan_stats=False)],
                allow_nan_stats=False)

    def discrete_markov_chain_latent_transition(
            self, latent_state: tf.Tensor) -> tfd.Distribution:
        if self._has_dedicated_label_transition_network:
            next_label_logits = self.action_label_transition_network(latent_state)
            next_state_logits = lambda _next_label: self.action_transition_network([latent_state, _next_label])
        else:
            next_label_logits, _next_state_logits = self.action_transition_network(latent_state)
            next_state_logits = lambda _: _next_state_logits

        return tfd.JointDistributionSequential([
            tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=self.latent_policy_network(latent_state)),
                components_distribution=tfd.Independent(
                    tfd.Bernoulli(
                        logits=tf.transpose(next_label_logits, perm=[0, 2, 1]),
                        dtype=tf.float32,
                        allow_nan_stats=False),
                    reinterpreted_batch_ndims=1)
            ), lambda _next_label: tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=self.latent_policy_network(latent_state)),
                components_distribution=tfd.Independent(tfd.Bernoulli(
                    logits=tf.transpose(next_state_logits(_next_label), perm=[0, 2, 1]),
                    allow_nan_stats=False), reinterpreted_batch_ndims=1))])

    def discrete_latent_transition(
            self, latent_state: tf.Tensor, action: tf.Tensor, next_label: Optional[tf.Tensor] = None
    ) -> tfd.Distribution:
        """
        Retrieves a Bernoulli probability distribution P(z'|z, a) over successor latent states, given a binary latent
        state z and an action a.
        """
        if self._has_dedicated_label_transition_network:
            next_label_logits = self.label_transition_network([latent_state, action])
            next_state_logits = lambda _next_label: self.transition_network([latent_state, action, _next_label])
        else:
            next_label_logits, _next_state_logits = self.transition_network([latent_state, action])
            next_state_logits = lambda _: _next_state_logits

        if next_label is not None:
            _next_state_logits = next_state_logits(next_label)
            return tfd.Independent(tfd.Bernoulli(logits=_next_state_logits, allow_nan_stats=False))
        else:
            return tfd.JointDistributionSequential([
                tfd.Independent(tfd.Bernoulli(
                    logits=next_label_logits,
                    allow_nan_stats=False,
                    dtype=tf.float32)),
                lambda _next_label: tfd.Independent(
                    tfd.Bernoulli(
                        logits=next_state_logits(_next_label),
                        allow_nan_stats=False))])

    def reward_distribution(
            self, latent_state: tf.Tensor, action: tf.Tensor, next_latent_state: tf.Tensor) -> tfd.Distribution:
        """
        Retrieves a probability distribution P(r|z, a, z') over the rewards obtained when the transition z, a, z'
        has been performed.
        """
        [reward_mean, reward_raw_covariance] = self.reward_network([latent_state, action, next_latent_state])
        return tfd.MultivariateNormalDiag(
            loc=reward_mean,
            scale_diag=self.scale_activation(reward_raw_covariance),
            allow_nan_stats=False)

    def discrete_latent_policy(self, latent_state: tf.Tensor):
        return tfd.OneHotCategorical(
            logits=self.latent_policy_network(latent_state),
            allow_nan_stats=False)

    def state_embedding_function(
            self,
            state: tf.Tensor,
            label: Optional[tf.Tensor] = None,
            labeling_function: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
            dtype: tf.dtypes = tf.int32,
    ) -> tf.Tensor:

        if label is not None:
            label = tf.cast(label, dtype=tf.float32)
        elif labeling_function is not None:
            label = labeling_function(state)

        if self.deterministic_state_embedding:
            latent_state = self.binary_encode_state(state, label).mode()
        else:
            latent_state = self.binary_encode_state(state, label).sample()
        return tf.cast(latent_state, dtype)

    def action_embedding_function(
            self,
            latent_state: tf.Tensor,
            latent_action: tf.Tensor,
    ) -> tf.Tensor:
        return latent_action

    def anneal(self):
        for var, decay_rate in [
            (self.encoder_temperature, self.encoder_temperature_decay_rate),
            (self.prior_temperature, self.prior_temperature_decay_rate),
            (self._entropy_regularizer_scale_factor, self.entropy_regularizer_decay_rate),
            (self._kl_scale_factor_decay, self.kl_growth_rate),
            (self._is_exponent_decay, self._is_exponent_growth_rate)
        ]:
            if decay_rate.numpy().all() > 0:
                var.assign(var * (1. - decay_rate))

        for var, var_growth_rate, initial_var_value, decay in [
            (self.kl_scale_factor, self.kl_growth_rate, self._initial_kl_scale_factor, self._kl_scale_factor_decay),
            (self.is_exponent, self.is_exponent_growth_rate, self._initial_is_exponent, self._is_exponent_decay)
        ]:
            if var_growth_rate > 0:
                var.assign(initial_var_value + (1. - initial_var_value) * (1. - decay))

    def call(self, inputs, training=None, mask=None, **kwargs):
        return self.__call__(*inputs, **kwargs)

    def get_config(self):
        pass

    @tf.function
    def __call__(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
            *args, **kwargs
    ):
        if self.latent_policy_training_phase:
            return self.latent_policy_training(state, label, action, reward, next_state, next_label)

        # Logistic samples
        state_encoder_distribution = self.relaxed_state_encoding(state, temperature=self.encoder_temperature)
        next_state_encoder_distribution = self.relaxed_state_encoding(next_state, temperature=self.encoder_temperature)

        # Sigmoid of Logistic samples with location alpha/t and scale 1/t gives Relaxed Bernoulli
        # samples of location alpha and temperature t
        latent_state = tf.concat([label, tf.sigmoid(state_encoder_distribution.sample())], axis=-1)
        next_logistic_latent_state = next_state_encoder_distribution.sample()

        log_q_encoding = next_state_encoder_distribution.log_prob(next_logistic_latent_state)
        if self.latent_policy_network is not None and self.full_optimization:
            log_p_transition = self.relaxed_markov_chain_latent_transition(
                latent_state, temperature=self.prior_temperature
            ).log_prob(next_label, next_logistic_latent_state)
        else:
            log_p_transition = self.relaxed_latent_transition(
                latent_state, action, temperature=self.prior_temperature
            ).log_prob(next_label, next_logistic_latent_state)
        rate = log_q_encoding - log_p_transition

        # retrieve Relaxed Bernoulli samples
        next_latent_state = tf.concat([next_label, tf.sigmoid(next_logistic_latent_state)], axis=-1)

        if self.latent_policy_network is not None and self.full_optimization:
            # log P(a, r, s' | z, z') =  log π(a | z) + log P(r | z, a, z') + log P(s' | z')
            reconstruction_distribution = tfd.JointDistributionSequential([
                self.discrete_latent_policy(latent_state),
                lambda _action: self.reward_distribution(latent_state, _action, next_latent_state),
                self.decode_state(next_latent_state)
            ])
            distortion = -1. * reconstruction_distribution.log_prob(action, reward, next_state)
        else:
            # log P(r, s' | z, a, z') = log P(r | z, a, z') + log P(s' | z')
            reconstruction_distribution = tfd.JointDistributionSequential([
                self.reward_distribution(latent_state, action, next_latent_state),
                self.decode_state(next_latent_state)
            ])
            distortion = -1. * reconstruction_distribution.log_prob(reward, next_state)

        entropy_regularizer = self.entropy_regularizer(
            next_state,
            use_marginal_entropy=True,
            latent_states=next_latent_state)

        # priority support
        if self.priority_handler is not None and sample_key is not None:
            tf.stop_gradient(
                self.priority_handler.update_priority(
                    keys=sample_key,
                    latent_states=tf.stop_gradient(tf.cast(tf.round(latent_state), tf.int32)),
                    loss=tf.stop_gradient(distortion + rate)))

        # metrics
        self.loss_metrics['ELBO'](tf.stop_gradient(-1 * (distortion + rate)))
        reconstruction_sample = reconstruction_distribution.sample()
        self.loss_metrics['state_mse'](next_state, reconstruction_sample[-1])
        self.loss_metrics['reward_mse'](reward, reconstruction_sample[-2])
        self.loss_metrics['distortion'](distortion)
        self.loss_metrics['rate'](rate)
        # self.loss_metrics['annealed_rate'](tf.stop_gradient(self.kl_scale_factor * rate))
        self.loss_metrics['entropy_regularizer'](
            tf.stop_gradient(self.entropy_regularizer_scale_factor * entropy_regularizer))
        self.loss_metrics['transition_log_probs'](
            tf.stop_gradient(
                self.discrete_latent_transition(tf.stop_gradient(tf.round(latent_state)),
                                                action).log_prob(
                    next_label, tf.round(tf.sigmoid(next_logistic_latent_state)))))

        if 'action_mse' in self.loss_metrics:
            self.loss_metrics['action_mse'](action, reconstruction_sample[0])

        if debug:
            tf.print(latent_state, "sampled z")
            tf.print(next_logistic_latent_state, "sampled (logistic) z'")
            tf.print(next_latent_state, "sampled z'")
            tf.print(self.encoder_network([state, label]), "log locations[:-1] -- logits[:-1] of Q")
            tf.print(log_q_encoding, "Log Q(logistic z'|s', l')")
            tf.print(self.transition_network([latent_state, action]), "log-locations P_transition")
            tf.print(log_p_transition, "log P(logistic z'|z, a)")
            tf.print(self.discrete_latent_transition(
                tf.round(latent_state), action
            ).prob(tf.round(tf.sigmoid(next_logistic_latent_state))), "P(round(z') | round(z), a)")
            tf.print(next_latent_state, "sampled z'")
            [reconstruction_mean, _, reconstruction_prior_components] = \
                self.reconstruction_network(next_latent_state)
            tf.print(reconstruction_mean, 'mean(s | z)')
            tf.print(reconstruction_prior_components, 'GMM: prior components')
            tf.print(log_q_encoding - log_p_transition, "log Q(z') - log P(z')")

        return {'distortion': distortion, 'rate': rate, 'entropy_regularizer': entropy_regularizer}

    @tf.function
    def entropy_regularizer(
            self, state: tf.Tensor,
            use_marginal_entropy: bool = False,
            latent_states: Optional[tf.Tensor] = None,
            *args, **kwargs
    ):
        logits = self.encoder_network(state)

        for metric_label in ('encoder_entropy', 'state_encoder_entropy'):
            if metric_label in self.loss_metrics:
                self.loss_metrics[metric_label](
                    tf.stop_gradient(tfd.Independent(tfd.Bernoulli(logits=logits)).entropy()))

        if use_marginal_entropy:
            batch_size = tf.shape(logits)[0]
            marginal_encoder = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(
                    logits=tf.ones(shape=(batch_size, batch_size))),
                components_distribution=tfd.Independent(tfd.RelaxedBernoulli(
                    logits=tf.tile(tf.expand_dims(logits, axis=0), [batch_size, 1, 1]),
                    temperature=self.encoder_temperature,
                    # allow_nan_stats=False
                ), reinterpreted_batch_ndims=1),
                reparameterize=(latent_states is None),
                # allow_nan_stats=False
            )
            if latent_states is None:
                latent_states = marginal_encoder.sample(batch_size)
            else:
                latent_states = latent_states[..., self.atomic_prop_dims:]
            latent_states = tf.clip_by_value(latent_states, clip_value_min=1e-7, clip_value_max=1. - 1e-7)
            marginal_entropy_regularizer = tf.reduce_mean(marginal_encoder.log_prob(latent_states))

            if tf.reduce_any(tf.logical_or(
                    tf.math.is_nan(marginal_entropy_regularizer),
                    tf.math.is_inf(marginal_entropy_regularizer))):
                tf.print("Inf or NaN detected in marginal_encoder_entropy")
                return -1. * tfd.Independent(tfd.Bernoulli(logits=logits, allow_nan_stats=False)).entropy()
            else:
                if 'marginal_encoder_entropy' in self.loss_metrics:
                    self.loss_metrics['marginal_encoder_entropy'](tf.stop_gradient(-1. * marginal_entropy_regularizer))
                return marginal_entropy_regularizer
        else:
            return -1. * tfd.Independent(tfd.Bernoulli(logits=logits, allow_nan_stats=False)).entropy()

    def latent_policy_training(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor
    ):
        latent_distribution = self.relaxed_state_encoding(state, label, temperature=self.encoder_temperature)
        latent_state = latent_distribution.sample()

        latent_policy_distribution = self.discrete_latent_policy(latent_state)

        if 'action_mse' in self.loss_metrics:
            self.loss_metrics['action_mse'](action, latent_policy_distribution.sample())

        return {'distortion': -1. * latent_policy_distribution.log_prob(action), 'rate': 0., 'entropy_regularizer': 0.}

    def eval(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor
    ):
        """
        Evaluate the ELBO by making use of a discrete latent space.
        """

        latent_distribution = self.binary_encode_state(state)
        next_latent_distribution = self.binary_encode_state(next_state)
        latent_state = tf.concat([label, tf.cast(latent_distribution.sample(), tf.float32)], axis=-1)
        next_latent_state_no_label = tf.cast(next_latent_distribution.sample(), tf.float32)

        if self.latent_policy_network is not None and self.full_optimization:
            transition_distribution = self.discrete_markov_chain_latent_transition(
                latent_state)
        else:
            transition_distribution = self.discrete_latent_transition(latent_state, action)
        # rate = next_latent_distribution.kl_divergence(transition_distribution)
        rate = next_latent_distribution.log_prob(next_latent_state_no_label) - transition_distribution.log_prob(
            next_label, next_latent_state_no_label)

        next_latent_state = tf.concat([next_label, next_latent_state_no_label], axis=-1)

        if self.latent_policy_network is not None and self.full_optimization:
            # log P(a, r, s' | z, z') =  log π(a | z) + log P(r | z, a, z') + log P(s' | z')
            reconstruction_distribution = tfd.JointDistributionSequential([
                self.discrete_latent_policy(latent_state),
                lambda _action: self.reward_distribution(latent_state, _action, next_latent_state),
                self.decode_state(next_latent_state)
            ])
            distortion = -1. * reconstruction_distribution.log_prob(action, reward, next_state)
        else:
            # log P(r, s' | z, a, z') = log P(r | z, a, z') + log P(s' | z')
            reconstruction_distribution = tfd.JointDistributionSequential([
                self.reward_distribution(latent_state, action, next_latent_state),
                self.decode_state(next_latent_state)
            ])
            distortion = -1. * reconstruction_distribution.log_prob(reward, next_state)

        return {
            'distortion': distortion,
            'rate': rate,
            'elbo': -1. * (distortion + rate),
            'latent_states': tf.concat([tf.cast(latent_state, tf.int64), tf.cast(next_latent_state, tf.int64)], axis=0),
            'latent_actions': None
        }

    def mean_latent_bits_used(self, inputs, eps=1e-3, deterministic=True):
        """
        Compute the mean number of bits used to represent the latent space of the vae_mdp for the given dataset batch.
        This allows monitoring if the latent space is effectively used by the VAE or if posterior collapse happens.
        """
        mean_bits_used = 0
        s, l = inputs[:2]
        if deterministic:
            mean = tf.reduce_mean(tf.cast(self.binary_encode_state(s, l).mode(), dtype=tf.float32), axis=0)
        else:
            mean = tf.reduce_mean(self.binary_encode_state(s, l).mean(), axis=0)
        check = lambda x: 1 if 1 - eps > x > eps else 0
        mean_bits_used += tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()
        return {'mean_state_bits_used': mean_bits_used}

    @property
    def evaluation_window(self):
        return self._evaluation_window

    @property
    def _has_dedicated_label_transition_network(self):
        return len(self.transition_network.outputs) == 1

    @property
    def encoder_temperature(self):
        return self._encoder_temperature

    @encoder_temperature.setter
    def encoder_temperature(self, value):
        self._encoder_temperature = tf.Variable(
            value, dtype=tf.float32, trainable=False, name='encoder_temperature')

    @property
    def is_exponent(self):
        return self._is_exponent

    @is_exponent.setter
    def is_exponent(self, value):
        if self._is_exponent is None:
            self._is_exponent = tf.Variable(
                value, dtype=tf.float32, trainable=False, name='important_sampling_exponent')
        else:
            self._is_exponent.assign(value)
        if self._initial_is_exponent is None:
            self._initial_is_exponent = tf.Variable(
                value, dtype=tf.float32, trainable=False, name='initial_importance_sampling_exponent')
        else:
            self._initial_is_exponent.assign(value)

    @property
    def is_exponent_growth_rate(self):
        return self._is_exponent_growth_rate

    @is_exponent_growth_rate.setter
    def is_exponent_growth_rate(self, value):
        if self._is_exponent_decay is None:
            self._is_exponent_decay = tf.Variable(1., dtype=tf.float32, trainable=False)
        else:
            self._is_exponent_decay.assign(1.)
        self._is_exponent_growth_rate = tf.constant(value, dtype=tf.float32)

    @property
    def encoder_temperature_decay_rate(self):
        return self._encoder_temperature_decay_rate

    @encoder_temperature_decay_rate.setter
    def encoder_temperature_decay_rate(self, value):
        self._encoder_temperature_decay_rate = tf.constant(value, dtype=tf.float32)

    @property
    def prior_temperature(self):
        return self._prior_temperature

    @prior_temperature.setter
    def prior_temperature(self, value):
        self._prior_temperature = tf.Variable(
            value, dtype=tf.float32, trainable=False, name='prior_temperature')

    @property
    def prior_temperature_decay_rate(self):
        return self._prior_temperature_decay_rate

    @prior_temperature_decay_rate.setter
    def prior_temperature_decay_rate(self, value):
        self._prior_temperature_decay_rate = tf.constant(value, dtype=tf.float32)

    @property
    def entropy_regularizer_scale_factor(self):
        return self._entropy_regularizer_scale_factor + self.entropy_regularizer_scale_factor_min_value

    @entropy_regularizer_scale_factor.setter
    def entropy_regularizer_scale_factor(self, value):
        if self._entropy_regularizer_scale_factor is None:
            self._entropy_regularizer_scale_factor = tf.Variable(
                value, dtype=tf.float32, trainable=False, name='entropy_regularizer_scale_factor')
        else:
            self._entropy_regularizer_scale_factor.assign(value)

    @property
    def entropy_regularizer_decay_rate(self):
        return self._regularizer_decay_rate

    @entropy_regularizer_decay_rate.setter
    def entropy_regularizer_decay_rate(self, value):
        self._regularizer_decay_rate = tf.constant(value, dtype=tf.float32)

    @property
    def kl_scale_factor(self):
        return self._kl_scale_factor

    @kl_scale_factor.setter
    def kl_scale_factor(self, value):
        if self._kl_scale_factor is None:
            self._kl_scale_factor = tf.Variable(value, dtype=tf.float32, trainable=False, name='kl_scale_factor')
        else:
            self._kl_scale_factor.assign(value)

    @property
    def kl_growth_rate(self):
        return self._kl_growth_rate

    @kl_growth_rate.setter
    def kl_growth_rate(self, value):
        if self._initial_kl_scale_factor is None:
            self._initial_kl_scale_factor = tf.Variable(self.kl_scale_factor, dtype=tf.float32, trainable=False)
        else:
            self._initial_kl_scale_factor.assign(self.kl_scale_factor)
        if self._kl_scale_factor_decay is None:
            self._kl_scale_factor_decay = tf.Variable(1., dtype=tf.float32, trainable=False)
        else:
            self._kl_scale_factor_decay.assign(1.)
        self._kl_growth_rate = tf.constant(value, dtype=tf.float32)

    @property
    def inference_variables(self):
        return self.encoder_network.trainable_variables

    @property
    def generator_variables(self):
        variables = []
        for network in [self.reconstruction_network, self.reward_network, self.transition_network]:
            variables += network.trainable_variables
        return variables

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
    ):
        output = self(state, label, action, reward, next_state, next_label, sample_key)
        distortion, rate, entropy_regularizer = output['distortion'], output['rate'], output['entropy_regularizer']
        alpha = self.entropy_regularizer_scale_factor
        beta = self.kl_scale_factor

        # Importance sampling weights (is) for prioritized experience replay
        if sample_probability is not None:
            is_weights = (tf.stop_gradient(tf.reduce_min(sample_probability)) / sample_probability) ** self.is_exponent
        else:
            is_weights = 1.

        return tf.reduce_mean(
            is_weights * (distortion + beta * rate + alpha * entropy_regularizer)
        )

    def _compute_apply_gradients(
            self, state, label, action, reward, next_state, next_label, trainable_variables,
            sample_key=None, sample_probability=None, *args, **kwargs):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(
                state, label, action, reward, next_state, next_label,
                sample_key=sample_key, sample_probability=sample_probability)

        gradients = tape.gradient(loss, trainable_variables)

        if debug_gradients:
            for gradient, variable in zip(gradients, trainable_variables):
                tf.print(gradient, "Gradient for {}".format(variable.name), ' -- variable=', trainable_variables)

        if self._optimizer is not None:
            self._optimizer.apply_gradients(zip(gradients, trainable_variables))
        return {'loss': loss}

    @tf.function
    def compute_apply_gradients(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
            sample_probability: Optional[tf.Tensor] = None,
            *args, **kwargs
    ):
        return self._compute_apply_gradients(
            state, label, action, reward, next_state, next_label, self.trainable_variables,
            sample_key, sample_probability)

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
            state, label, action, reward, next_state, next_label, self.inference_variables,
            sample_key, sample_probability)

    @tf.function
    def latent_policy_update(
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
            self.latent_policy_network.trainable_variables, sample_key, sample_probability)

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
            self.generator_variables, sample_key, sample_probability)

    def initialize_environments(
            self,
            environment_suite,
            env_name: str,
            env_args: Optional[List[str]] = None,
            parallel_environments: bool = False,
            num_parallel_environments: int = 4,
            collect_steps_per_iteration: int = 8,
            environment_seed: Optional[int] = None,
            use_prioritized_replay_buffer: bool = False,
            policy_evaluation_num_episodes: int = 0,
            policy_evaluation_env_name: Optional[str] = None,
            labeling_function: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
            local_losses_evaluation: bool = False,
            local_losses_eval_steps: Optional[int] = 0,
            local_losses_eval_replay_buffer_size: Optional[int] = int(1e5),
            local_losses_reward_scaling: Optional[float] = 1.,
            embed_video_policy_evaluation: bool = False,
            video_path: str = 'video',
            environment_perturbation: float = 3. / 4.,
            time_limit: Optional[int] = None,
            recursive_environment_perturbation: bool = True,
            enforce_no_reward_shaping: bool = False,
            estimate_value_difference: bool = True
    ):
        # reverb replay buffers are not compatible with batched environments
        parallel_environments = parallel_environments and not use_prioritized_replay_buffer
        # get the highest integer from the range(1, num_parallel_env + 1) which can be (integer-)divided by
        # collect_steps_per_iteration
        _num_parallel_environments = num_parallel_environments
        num_parallel_environments = np.argmax(
            np.gcd(np.arange(1, num_parallel_environments + 1), collect_steps_per_iteration)
        ) + 1

        env_loader = EnvironmentLoader(
            environment_suite,
            seed=environment_seed,
            time_stacked_states=self.state_shape[0] if self.time_stacked_states else 1,
            env_args=env_args)

        env_wrappers = []
        if time_limit is not None:
            env_wrappers.append(
                lambda env: TimeLimit(env, time_limit))
        if environment_perturbation > 0.:
            env_wrappers.append(
                lambda env: PerturbedEnvironment(
                    env,
                    perturbation=environment_perturbation,
                    recursive_perturbation=recursive_environment_perturbation))
        if enforce_no_reward_shaping:
            env_wrappers.append(NoRewardShapingWrapper)

        if parallel_environments:
            py_env = parallel_py_environment.ParallelPyEnvironment(
                [lambda: env_loader.load(env_name, env_wrappers=env_wrappers)] * num_parallel_environments)
            env = tf_py_environment.TFPyEnvironment(py_env)
            env.reset()
        else:
            py_env = env_loader.load(env_name, env_wrappers=env_wrappers)
            py_env.reset()
            env = tf_py_environment.TFPyEnvironment(py_env) if not use_prioritized_replay_buffer else py_env

        if policy_evaluation_num_episodes > 0:
            if labeling_function is None:
                raise ValueError('Must pass a labeling function if a policy evaluation environment is requested')
            if policy_evaluation_env_name is None:
                policy_evaluation_env_name = env_name

            py_eval_env = env_loader.load(
                policy_evaluation_env_name,
                env_wrappers=[lambda env: TimeLimit(env, time_limit)] if time_limit is not None else [])
            eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)

            if self.time_stacked_states:
                eval_env = self.wrap_tf_environment(eval_env, lambda x: labeling_function(x)[:, -1, ...])
            else:
                eval_env = self.wrap_tf_environment(eval_env, labeling_function)

            eval_env.reset()
            latent_policy = eval_env.wrap_latent_policy(self.get_latent_policy(action_dtype=tf.int64))
            policy_evaluation_driver = tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver(
                eval_env, latent_policy, num_episodes=policy_evaluation_num_episodes)
            if embed_video_policy_evaluation:
                policy_evaluation_driver.observers.append(VideoEmbeddingObserver(
                    py_env=py_eval_env,
                    file_name=os.path.join(video_path, 'distilled_policy_evaluation'),
                    num_episodes=policy_evaluation_num_episodes))
        else:
            eval_env = None
            policy_evaluation_driver = None

        if local_losses_evaluation and self.latent_policy_network is not None:
            py_local_loss_eval_env = parallel_py_environment.ParallelPyEnvironment(
                [lambda: env_loader.load(env_name, env_wrappers=env_wrappers)] * num_parallel_environments)
            local_losses_eval_env = tf_py_environment.TFPyEnvironment(py_local_loss_eval_env)
            local_losses_eval_env.reset()

            local_losses_estimator = lambda: self.estimate_local_losses_from_samples(
                environment=local_losses_eval_env,
                steps=local_losses_eval_steps,
                labeling_function=labeling_function,
                estimate_transition_function_from_samples=True,
                replay_buffer_max_frames=local_losses_eval_replay_buffer_size,
                reward_scaling=local_losses_reward_scaling,
                estimate_value_difference=estimate_value_difference)

        else:
            local_losses_eval_env = None
            local_losses_estimator = None

        return namedtuple(
            'Environments',
            ['training', 'evaluation', 'policy_evaluation_driver', 'local_losses_eval_env', 'local_losses_estimator'])(
            env, eval_env, policy_evaluation_driver, local_losses_eval_env, local_losses_estimator)

    def initialize_dataset_components(
            self,
            env: py_environment.PyEnvironment,
            policy: tf_agents.policies.tf_policy.TFPolicy,
            labeling_function: Callable[[tf.Tensor], tf.Tensor],
            batch_size: int = 128,
            manager: Optional[tf.train.CheckpointManager] = None,
            use_prioritized_replay_buffer: bool = False,
            replay_buffer_capacity: int = int(1e6),
            priority_exponent: float = 0.6,
            bucket_based_priorities: bool = True,
            discrete_action_space: bool = False,
            collect_steps_per_iteration: int = 8,
            initial_collect_steps: int = int(1e4),
            epsilon_greedy: Optional[tf.Variable] = 0.
    ) -> DatasetComponents:

        if self.time_stacked_states:
            _labeling_function = labeling_function

            def labeling_function(observation):
                return _labeling_function(observation[:, -1, ...])

            policy = TimeStackedStatesPolicyWrapper(policy, history_length=self.state_shape[0])

        if tf.greater(epsilon_greedy, 0.):
            policy = EpsilonMimicPolicy(
                policy=policy,
                latent_policy=LatentPolicyOverRealStateAndActionSpaces(
                    time_step_spec=policy.time_step_spec,
                    action_spec=policy.action_spec,
                    labeling_function=labeling_function,
                    latent_policy=self.get_latent_policy(),
                    state_embedding_function=self.state_embedding_function,
                    action_embedding_function=self.action_embedding_function),
                epsilon=epsilon_greedy)

        # specs
        trajectory_spec = trajectory.from_transition(
            time_step=env.time_step_spec(),
            action_step=policy.policy_step_spec,
            next_time_step=env.time_step_spec())

        if use_prioritized_replay_buffer:

            checkpoint_path = None if manager is None else os.path.join(manager.directory, 'reverb')
            if checkpoint_path is not None:
                reverb_checkpointer = reverb.checkpointers.DefaultCheckpointer(checkpoint_path)
            else:
                reverb_checkpointer = None

            table_name = 'prioritized_replay_buffer'
            table = reverb.Table(
                table_name,
                max_size=replay_buffer_capacity,
                sampler=reverb.selectors.Prioritized(priority_exponent=priority_exponent),
                remover=reverb.selectors.Fifo(),
                rate_limiter=reverb.rate_limiters.MinSize(1))

            reverb_server = reverb.Server([table], checkpointer=reverb_checkpointer)

            replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
                data_spec=policy.collect_data_spec,
                sequence_length=2,
                table_name=table_name,
                local_server=reverb_server)

            if bucket_based_priorities:
                self.priority_handler = PriorityBuckets(
                    replay_buffer=replay_buffer,
                    latent_state_size=self.latent_state_size)
                priority_handler: PriorityHandler = self.priority_handler
            else:
                self.priority_handler = LossPriority(
                    replay_buffer=replay_buffer,
                    max_priority=10., )
                priority_handler: PriorityHandler = self.priority_handler

            add_environment_step = reverb_utils.ReverbTrajectorySequenceObserver(
                py_client=replay_buffer.py_client,
                table_name=table_name,
                sequence_length=2,
                stride_length=1,
                priority=priority_handler.max_priority)

            driver = py_driver.PyDriver(
                env=env,
                policy=py_tf_eager_policy.PyTFEagerPolicy(policy, use_tf_function=True),
                observers=[add_environment_step],
                max_steps=collect_steps_per_iteration)
            initial_collect_driver = py_driver.PyDriver(
                env=env,
                policy=py_tf_eager_policy.PyTFEagerPolicy(policy, use_tf_function=True),
                observers=[add_environment_step],
                max_steps=initial_collect_steps)

            replay_buffer_num_frames = replay_buffer.num_frames

            if manager is not None:
                priority_handler.load_or_initialize_checkpoint(manager.directory)
                model_manager = manager

                def _manager_save(*args, **kwargs):
                    replay_buffer.py_client.checkpoint()
                    priority_handler.checkpoint(*args, **kwargs)
                    model_manager.save(*args, **kwargs)

                manager = namedtuple('CheckpointManagerWithPrioritizedRB', ['save'])(_manager_save)

            def close():
                env.close()
                reverb_server.stop()

        else:
            num_parallel_environments = env.batch_size
            collect_steps_per_iteration = collect_steps_per_iteration // num_parallel_environments
            initial_collect_steps = initial_collect_steps
            replay_buffer_capacity = replay_buffer_capacity // num_parallel_environments

            replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=trajectory_spec,
                batch_size=env.batch_size,
                max_length=replay_buffer_capacity)
            add_batch = replay_buffer.add_batch
            driver = dynamic_step_driver.DynamicStepDriver(
                env, policy, observers=[add_batch], num_steps=collect_steps_per_iteration)
            initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
                env, policy, observers=[add_batch], num_steps=initial_collect_steps)

            replay_buffer_num_frames = lambda: replay_buffer.num_frames().numpy()
            close = lambda: env.close()

            if manager is not None:
                model_manager = manager
                checkpoint_path = os.path.join(manager.directory, 'replay_buffer')
                rb_checkpointer = tf.train.Checkpoint(replay_buffer)
                rb_manager = tf.train.CheckpointManager(
                    checkpoint=rb_checkpointer, directory=checkpoint_path, max_to_keep=1)
                rb_checkpointer.restore(rb_manager.latest_checkpoint)

                def _manager_save(*args, **kwargs):
                    rb_manager.save(*args, **kwargs)
                    model_manager.save(*args, **kwargs)

                manager = namedtuple('CheckpointManagerWithRB', ['save'])(_manager_save)

        if replay_buffer_num_frames() < initial_collect_steps:
            print("Initial collect steps...")
            initial_collect_driver.run(env.current_time_step())

        def dataset_generator(generator_fn):
            return replay_buffer.as_dataset(
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                num_steps=2
            ).map(
                map_func=generator_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                #  deterministic=False  # TF version >= 2.2.0
            )

        dataset = dataset_generator(
            lambda trajectory, buffer_info:
            map_rl_trajectory_to_vae_input(
                trajectory=trajectory,
                labeling_function=ergodic_batched_labeling_function(labeling_function),
                discrete_action=discrete_action_space,
                num_discrete_actions=self.action_shape[0],
                sample_info=buffer_info if use_prioritized_replay_buffer else None))
        dataset_iterator = iter(
            dataset.batch(
                batch_size=batch_size,
                drop_remainder=True
            ).prefetch(tf.data.experimental.AUTOTUNE))

        return DatasetComponents(
            replay_buffer=replay_buffer,
            driver=driver,
            initial_collect_driver=initial_collect_driver,
            close_fn=close,
            replay_buffer_num_frames_fn=replay_buffer_num_frames,
            wrapped_manager=manager,
            dataset=dataset,
            dataset_iterator=dataset_iterator,
            epsilon_greedy=epsilon_greedy)

    def save(self, save_directory, model_name: str, signatures: Optional[Dict] = None, *args, **kwargs):
        if check_numerics:
            tf.debugging.disable_check_numerics()

        _priority_handler = self.priority_handler
        _optimizer = self.detach_optimizer()
        self.priority_handler = None

        (state_shape, label_shape, action_shape, reward_shape) = (
            tuple(shape) for shape in
            [self.state_shape, (self.atomic_prop_dims,), self.action_shape, self.reward_shape])

        if signatures is None:
            signatures = dict()
        else:
            signatures['serving_default'] = self.__call__.get_concrete_function(
                tf.TensorSpec(shape=(None,) + state_shape, dtype=tf.float32, name='state'),
                tf.TensorSpec(shape=(None,) + label_shape, dtype=tf.float32, name='label'),
                tf.TensorSpec(shape=(None,) + action_shape, dtype=tf.float32, name='action'),
                tf.TensorSpec(shape=(None,) + reward_shape, dtype=tf.float32, name='reward'),
                tf.TensorSpec(shape=(None,) + state_shape, dtype=tf.float32, name='next_state'),
                tf.TensorSpec(shape=(None,) + label_shape, dtype=tf.float32, name='next_label'), )

        # with utils.keras_option_scope(save_traces=False):
        tf.saved_model.save(self, os.path.join(save_directory, 'models', model_name), signatures=signatures, )

        self.priority_handler = _priority_handler
        self.attach_optimizer(_optimizer)
        if check_numerics:
            tf.debugging.enable_check_numerics()

    def train_from_policy(
            self,
            policy: tf_agents.policies.tf_policy.TFPolicy,
            environment_suite,
            env_name: str,
            labeling_function: Callable[[tf.Tensor], tf.Tensor],
            env_args: Optional[List] = None,
            env_time_limit: Optional[int] = None,
            epsilon_greedy: Optional[float] = 0.,
            epsilon_greedy_decay_rate: Optional[float] = -1.,
            discrete_action_space: bool = False,
            training_steps: int = int(3e6),
            initial_collect_steps: int = int(1e5),
            collect_steps_per_iteration: Optional[int] = None,
            replay_buffer_capacity: int = int(1e6),
            use_prioritized_replay_buffer: bool = True,
            bucket_based_priorities: bool = True,
            priority_exponent: int = 0.6,
            parallel_environments: bool = True,
            num_parallel_environments: int = 4,
            batch_size: int = 128,
            global_step: Optional[tf.Variable] = None,
            optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(1e-4),
            checkpoint: Optional[tf.train.Checkpoint] = None,
            manager: Optional[tf.train.CheckpointManager] = None,
            log_interval: int = 200,
            checkpoint_interval: int = 250,
            eval_steps: int = int(1e4),
            eval_and_save_model_interval: int = int(1e4),
            train_summary_writer: Optional[tf.summary.SummaryWriter] = None,
            log_name: str = 'vae_training',
            annealing_period: int = 1,
            start_annealing_step: int = 0,
            reset_kl_scale_factor: Optional[float] = None,
            reset_entropy_regularizer: Optional[float] = None,
            display_progressbar: bool = False,
            save_directory: Optional[str] = '.',
            policy_evaluation_num_episodes: int = 30,
            policy_evaluation_env_name: Optional[str] = None,
            environment_seed: Optional[int] = None,
            aggressive_training: bool = False,
            approximate_convergence_error: float = 5e-1,
            approximate_convergence_steps: int = 10,
            aggressive_training_steps: int = int(2e6),
            environment: Optional[tf_agents.environments.py_environment.PyEnvironment] = None,
            dataset_components: Optional[DatasetComponents] = None,
            policy_evaluation_driver: Optional[tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver] = None,
            close_at_the_end: bool = True,
            start_time: Optional[float] = None,
            wall_time: Optional[str] = None,
            memory_limit: Optional[float] = None,
            local_losses_evaluation: bool = False,
            local_losses_eval_steps: Optional[int] = int(3e4),
            local_losses_eval_replay_buffer_size: Optional[int] = int(1e5),
            local_losses_reward_scaling: Optional[float] = 1.,
            estimate_value_difference: bool = True,
            embed_video_evaluation: bool = False,
            environment_perturbation: float = 3. / 4.,
            recursive_environment_perturbation: bool = True,
            enforce_no_reward_shaping: bool = False,
            local_losses_estimator: Optional[Callable] = None,
    ):
        if wall_time is not None:
            if start_time is None:
                start_time = time.time()

            wall_time = wall_time.split(':')
            wall_time = datetime.timedelta(
                hours=int(wall_time[0]),
                minutes=int(wall_time[1]),
                seconds=int(wall_time[2])).total_seconds()

        if collect_steps_per_iteration is None:
            collect_steps_per_iteration = batch_size // 8

        if save_directory is not None:
            save_directory = os.path.join(save_directory, 'saves', env_name)

        # Load checkpoint
        if checkpoint is not None and manager is not None:
            checkpoint.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

        # attach optimizer
        self.attach_optimizer(optimizer)

        if global_step is None:
            if checkpoint is not None:
                global_step = checkpoint.save_counter
            else:
                global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        start_step = global_step.numpy()
        print("Step {} on {}.".format(start_step, training_steps))

        if start_step < start_annealing_step:
            if reset_kl_scale_factor is not None:
                self.kl_scale_factor = reset_kl_scale_factor
            if reset_entropy_regularizer is not None:
                self.entropy_regularizer_scale_factor = reset_entropy_regularizer

        progressbar = Progbar(
            target=None,
            stateful_metrics=list(self.loss_metrics.keys()) + [
                'loss', 't_1', 't_2', 't_1_action', 't_2_action',
                'entropy_regularizer_scale_factor', 'step', "num_episodes", "env_steps",
                "replay_buffer_frames", 'kl_annealing_scale_factor', 'state_rate',
                "state_distortion", 'action_rate', 'action_distortion', 'mean_state_bits_used', 'wis_exponent',
                'priority_logistic_smoothness', 'priority_logistic_mean',
                'priority_logistic_max', 'priority_logistic_min',
                'dynamic_reward_scaling'
            ],
            interval=0.1) if display_progressbar else None

        discrete_action_space = discrete_action_space and (self.latent_policy_network is not None)

        if environment is None or (policy_evaluation_driver is None and policy_evaluation_num_episodes > 0):
            environments = self.initialize_environments(
                environment_suite=environment_suite,
                env_name=env_name,
                env_args=env_args,
                time_limit=env_time_limit,
                labeling_function=labeling_function,
                parallel_environments=parallel_environments,
                num_parallel_environments=num_parallel_environments,
                collect_steps_per_iteration=collect_steps_per_iteration,
                environment_seed=environment_seed,
                use_prioritized_replay_buffer=use_prioritized_replay_buffer,
                policy_evaluation_num_episodes=policy_evaluation_num_episodes,
                policy_evaluation_env_name=policy_evaluation_env_name,
                local_losses_evaluation=local_losses_evaluation,
                local_losses_eval_steps=local_losses_eval_steps,
                local_losses_eval_replay_buffer_size=local_losses_eval_replay_buffer_size,
                local_losses_reward_scaling=local_losses_reward_scaling,
                embed_video_policy_evaluation=embed_video_evaluation,
                video_path=os.path.join(save_directory, 'videos') if save_directory is not None else None,
                environment_perturbation=environment_perturbation,
                recursive_environment_perturbation=recursive_environment_perturbation,
                enforce_no_reward_shaping=enforce_no_reward_shaping,
                estimate_value_difference=estimate_value_difference)

            env = environment if environment is not None else environments.training
            policy_evaluation_driver = policy_evaluation_driver if policy_evaluation_driver is not None \
                else environments.policy_evaluation_driver
            local_losses_estimator = environments.local_losses_estimator
        else:
            env = environment

        if dataset_components is None:
            if epsilon_greedy > 0.:
                epsilon_greedy = tf.Variable(epsilon_greedy, trainable=False, dtype=tf.float32)

                if epsilon_greedy_decay_rate == -1:
                    epsilon_greedy_decay_rate = 1. - tf.exp(
                        (tf.math.log(1e-3) - tf.math.log(epsilon_greedy))
                        / (3. * (training_steps - start_annealing_step) / 5.))
                epsilon_greedy.assign(
                    epsilon_greedy * tf.pow(
                        1. - epsilon_greedy_decay_rate,
                        tf.math.maximum(0., tf.cast(global_step, dtype=tf.float32) - start_annealing_step)))

            dataset_components = self.initialize_dataset_components(
                env=env, policy=policy, labeling_function=labeling_function, batch_size=batch_size, manager=manager,
                use_prioritized_replay_buffer=use_prioritized_replay_buffer,
                replay_buffer_capacity=replay_buffer_capacity, priority_exponent=priority_exponent,
                bucket_based_priorities=bucket_based_priorities, discrete_action_space=discrete_action_space,
                collect_steps_per_iteration=collect_steps_per_iteration, initial_collect_steps=initial_collect_steps,
                epsilon_greedy=epsilon_greedy)
        else:
            epsilon_greedy = dataset_components.epsilon_greedy

        replay_buffer = dataset_components.replay_buffer
        driver = dataset_components.driver
        initial_collect_driver = dataset_components.initial_collect_driver
        close = dataset_components.close_fn
        replay_buffer_num_frames = dataset_components.replay_buffer_num_frames_fn
        manager = dataset_components.wrapped_manager
        dataset_iterator = dataset_components.dataset_iterator
        dataset = dataset_components.dataset

        if replay_buffer_num_frames() < batch_size:
            print("Initial collect steps...")
            initial_collect_driver.run(env.current_time_step())

        if tf.equal(global_step, 0):
            print("Start training")
        else:
            print("Resume training")

        # aggressive training metrics
        best_loss = None
        prev_loss = None
        convergence_error = approximate_convergence_error
        convergence_steps = approximate_convergence_steps
        aggressive_inference_optimization = True
        max_inference_update_steps = int(1e2)
        inference_update_steps = 0

        # wall_time and memory utils
        save_time = 0.
        training_loop_time = 0.
        wall_time_exceeded = False
        memory_used = 0. if memory_limit is None else psutil.Process().memory_info().rss / (1024 ** 3)
        memory_growth = 0.
        memory_limit_exceeded = False

        for _ in range(global_step.numpy(), training_steps):
            _loop_time = time.time()

            # Collect a few steps and save them to the replay buffer.
            driver.run(env.current_time_step())

            additional_training_metrics = {
                "replay_buffer_frames": replay_buffer.num_frames()} if not parallel_environments else {
                "replay_buffer_frames": replay_buffer.num_frames(),
            }
            if tf.math.not_equal(1., self._dynamic_reward_scaling):
                additional_training_metrics['dynamic_reward_scaling'] = self._dynamic_reward_scaling
            if epsilon_greedy > 0.:
                additional_training_metrics['epsilon_greedy'] = epsilon_greedy
            if use_prioritized_replay_buffer and not bucket_based_priorities:
                diff = (self.priority_handler.max_loss.result() - self.priority_handler.min_loss.result())
                if diff != 0:
                    additional_training_metrics[
                        'priority_logistic_smoothness'] = self.priority_handler.max_priority / diff
                additional_training_metrics['priority_logistic_mean'] = diff / 2
                additional_training_metrics['priority_logistic_max'] = self.priority_handler.max_loss.result()
                additional_training_metrics['priority_logistic_min'] = self.priority_handler.min_loss.result()
            if memory_limit is not None:
                additional_training_metrics['memory'] = memory_used

            loss = self.training_step(
                dataset=dataset, dataset_iterator=dataset_iterator, batch_size=batch_size,
                annealing_period=annealing_period, global_step=global_step,
                display_progressbar=display_progressbar,
                start_step=start_step, epoch=0, progressbar=progressbar,
                eval_and_save_model_interval=eval_and_save_model_interval,
                eval_steps=eval_steps,
                save_directory=save_directory, log_name=log_name, train_summary_writer=train_summary_writer,
                log_interval=log_interval, start_annealing_step=start_annealing_step,
                additional_metrics=additional_training_metrics,
                eval_policy_driver=policy_evaluation_driver,
                aggressive_training=aggressive_training and global_step.numpy() < aggressive_training_steps,
                aggressive_update=aggressive_inference_optimization,
                prioritized_experience_replay=use_prioritized_replay_buffer,
                local_losses_estimator=local_losses_estimator)

            if tf.math.logical_and(tf.greater(epsilon_greedy, 0.),
                                   tf.greater_equal(global_step, start_annealing_step)):
                epsilon_greedy.assign(epsilon_greedy * (1. - epsilon_greedy_decay_rate))

            if checkpoint is not None and tf.equal(tf.math.mod(global_step, checkpoint_interval), 0):
                manager.save()

            if wall_time is not None:
                _loop_time = time.time() - _loop_time
                training_loop_time = max(training_loop_time, _loop_time)
                wall_time_exceeded = (time.time() - start_time >= wall_time - 2 * (training_loop_time + save_time))
                if wall_time_exceeded:
                    print('Wall time exceeded.')
                    break

            if memory_limit is not None:
                if tf.equal(tf.math.mod(global_step, log_interval), 0):
                    process = psutil.Process()
                    _memory_used = memory_used
                    memory_used = process.memory_info().rss / (1024 ** 3)  # in GB
                    memory_growth = max(memory_growth, memory_used - _memory_used)
                    memory_limit_exceeded = memory_used + memory_growth > memory_limit
                    if memory_limit_exceeded:
                        print("Memory limit exceeded (used={:.3f}, max growth={:.3f}, limit={:.3f})".format(
                            memory_used, memory_growth, memory_limit))
                        break

            for key, value in loss.items():
                if tf.reduce_any(tf.logical_or(tf.math.is_nan(value), tf.math.is_inf(value))):
                    logging.warning("{} is NaN or Inf: {}".format(key, value))

        score = tf.reduce_mean(self.evaluation_window[self.evaluation_window > - np.inf])

        # save the final model
        if save_directory is not None and checkpoint is not None:
            self.save(
                save_directory,
                os.path.join(log_name, 'step{:d}'.format(global_step.numpy())),
                infos={'step': global_step.numpy(), 'score': score})
        if close_at_the_end:
            close()

        return {'score': score,
                'continue': not (wall_time_exceeded or close_at_the_end or memory_limit_exceeded)}

    def training_step(
            self, dataset, dataset_iterator, batch_size, annealing_period, global_step,
            display_progressbar, start_step, epoch, progressbar, eval_and_save_model_interval,
            eval_steps, save_directory, log_name, train_summary_writer, log_interval,
            start_annealing_step, additional_metrics: Optional[Dict[str, tf.Tensor]] = None,
            eval_policy_driver: Optional[tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver] = None,
            aggressive_training=False, aggressive_update=True, prioritized_experience_replay=False,
            local_losses_estimator=None
    ):
        # state, label_state, action, reward, next_state, label_next_state
        dataset_batch = next(dataset_iterator)

        if self._sample_additional_transition:
            extra_batch = next(dataset_iterator)
        else:
            extra_batch = None

        if additional_metrics is None:
            additional_metrics = {}

        if not aggressive_training and not self.latent_policy_training_phase:
            gradients = self.compute_apply_gradients(
                *dataset_batch, additional_transition_batch=extra_batch, step=global_step)
        elif not aggressive_training and self.latent_policy_training_phase:
            gradients = self.latent_policy_update(*dataset_batch)
        elif aggressive_update:
            gradients = self.inference_update(*dataset_batch)
        else:
            gradients = self.generator_update(*dataset_batch)
        loss = gradients

        if annealing_period > 0 and \
                global_step.numpy() % annealing_period == 0 and global_step.numpy() > start_annealing_step:
            self.anneal()

        # update progressbar
        metrics_key_values = [('step', global_step.numpy())] + \
                             [(key, value) for key, value in loss.items()] + \
                             [(key, value.result()) for key, value in self.loss_metrics.items()] + \
                             [(key, value) for key, value in additional_metrics.items()] + \
                             [(key, value) for key, value in self.temperature_metrics.items()]
        # [(key, value) for key, value in self.mean_latent_bits_used(dataset_batch).items()]
        if annealing_period != 0:
            metrics_key_values.append(('entropy_regularizer_scale_factor', self.entropy_regularizer_scale_factor))
            if self.kl_scale_factor != 1.:
                metrics_key_values.append(('kl_annealing_scale_factor', self.kl_scale_factor))
        if prioritized_experience_replay:
            metrics_key_values.append(('wis_exponent', self.is_exponent))
        if progressbar is not None and display_progressbar:
            progressbar.add(batch_size, values=metrics_key_values)

        # update step
        global_step.assign_add(1)

        # eval, save and log
        if global_step.numpy() % eval_and_save_model_interval == 0:
            self.eval_and_save(dataset=dataset,
                               eval_steps=eval_steps,
                               global_step=global_step, save_directory=save_directory, log_name=log_name,
                               train_summary_writer=train_summary_writer,
                               eval_policy_driver=eval_policy_driver,
                               local_losses_estimator=local_losses_estimator)
            self._dynamic_reward_scaling.assign(1.)

        if global_step.numpy() % log_interval == 0:
            if train_summary_writer is not None:
                with train_summary_writer.as_default():
                    for key, value in metrics_key_values:
                        tf.summary.scalar(key, value, step=global_step)
            # reset metrics
            self.reset_metrics()

        return loss

    def assign_score(
            self,
            score: float,
            checkpoint_model: bool,
            save_directory: str,
            model_name: str,
            training_step: int,
    ):
        """
        Stores the input score into the model evaluation window according to its evaluation criterion.
        If the evaluation window is modified this way and the checkpoint_model flag is set, then a model checkpoint is
        stored into the specified save directory.
        """

        def _checkpoint():
            optimizer = self.detach_optimizer()
            priority_handler = self.priority_handler
            self.priority_handler = None

            eval_checkpoint = tf.train.Checkpoint(model=self)
            eval_checkpoint.save(
                os.path.join(save_directory, 'training_checkpoints', model_name,
                             'ckpt-{:d}'.format(training_step)))

            self.attach_optimizer(optimizer)
            self.priority_handler = priority_handler

        if (self.evaluation_criterion is EvaluationCriterion.MEAN) \
                or tf.reduce_any(self.evaluation_window == -1. * np.inf):
            for i in tf.range(tf.shape(self.evaluation_window)[0]):
                _score = self.evaluation_window[i]
                self.evaluation_window[i].assign(score)
                score = _score
            if checkpoint_model:
                _checkpoint()
        elif self.evaluation_criterion is EvaluationCriterion.MAX:
            for i in tf.range(tf.shape(self.evaluation_window)[0]):
                if self.evaluation_window[i] <= score:
                    self.evaluation_window[i].assign(score)
                    if checkpoint_model:
                        _checkpoint()
                    break

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
            *args, **kwargs,
    ):

        if (dataset is None) == (dataset_iterator is None or batch_size is None):
            raise ValueError("Must either provide a dataset or a dataset iterator + batch size.")

        if dataset is not None:
            batch_size = eval_steps
            dataset_iterator = iter(dataset.batch(
                batch_size=batch_size,
                drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
            eval_steps = 1 if eval_steps > 0 else 0

        metrics = {'eval_elbo': tf.metrics.Mean(),
                   'eval_distortion': tf.metrics.Mean(),
                   'eval_rate': tf.metrics.Mean()}
        data = {'states': None, 'actions': None}
        avg_rewards = None
        local_losses_metrics = None

        if eval_steps > 0:
            eval_progressbar = Progbar(
                target=(eval_steps + 1) * batch_size, interval=0.1, stateful_metrics=['eval_ELBO'])

            tf.print("\nEvalutation over {} step(s)".format(eval_steps))

            for step in range(eval_steps):
                x = next(dataset_iterator)

                if len(x) >= 8:  # then sample_probability and sample_key are provided
                    is_weights = tf.reduce_min(x[7]) / x[7]  # we consider is_exponent=1 for evaluation
                else:
                    is_weights = 1.

                evaluation = self.eval(*(x[:6]))
                for value in ('states', 'actions'):
                    latent = evaluation['latent_' + value]
                    data[value] = latent if data[value] is None else tf.concat([data[value], latent], axis=0)
                for value in ('elbo', 'distortion', 'rate'):
                    metrics['eval_' + value](tf.reduce_mean(is_weights * evaluation[value]))
                eval_progressbar.add(batch_size, values=[('eval_ELBO', metrics['eval_elbo'].result())])
            tf.print('\n')
        if eval_policy_driver is not None:
            avg_rewards = self.eval_policy(
                eval_policy_driver=eval_policy_driver,
                train_summary_writer=train_summary_writer,
                global_step=global_step)

        if local_losses_estimator is not None:
            local_losses_metrics = local_losses_estimator()

        if train_summary_writer is not None and eval_steps > 0:
            buckets = {'states': 32, 'actions': self.number_of_discrete_actions}
            with train_summary_writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(key, value.result(), step=global_step)
                for value in ('states', 'actions'):
                    if data[value] is not None:
                        if value == 'states':
                            data[value] = tf.reduce_sum(
                                data[value] * 2 ** tf.range(tf.cast(self.latent_state_size, dtype=tf.int64)),
                                axis=-1)
                        tf.summary.histogram('{}_frequency'.format(value[:-1]),
                                             data[value],
                                             step=global_step,
                                             buckets=buckets[value])
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

            print('eval ELBO: ', metrics['eval_elbo'].result().numpy())

        if local_losses_metrics is not None:
            tf.print('Local reward loss: {:.2f}'.format(local_losses_metrics.local_reward_loss))
            tf.print('Local transition loss: {:.2f}'.format(local_losses_metrics.local_transition_loss))
            tf.print('Local transition loss (empirical transition function): {:.2f}'
                     ''.format(local_losses_metrics.local_transition_loss_transition_function_estimation))

            for key, value in local_losses_metrics.value_difference.items():
                tf.print(key, value)
            local_losses_metrics.print_time_metrics()

        if eval_policy_driver is not None or eval_steps > 0:
            self.assign_score(
                score=avg_rewards if avg_rewards is not None else metrics['eval_elbo'].result(),
                checkpoint_model=save_directory is not None and log_name is not None,
                save_directory=save_directory,
                model_name=log_name,
                training_step=global_step.numpy())

        gc.collect()

        return metrics['eval_elbo'].result()

    def eval_policy(
            self,
            eval_env: Optional[py_environment.PyEnvironment] = None,
            eval_policy_driver: Optional[tf_agents.drivers.driver.Driver] = None,
            labeling_function: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
            num_eval_episodes: int = 30,
            train_summary_writer: Optional = None,
            global_step: Optional[tf.Variable] = None,
            render: bool = False,
    ):
        if (eval_env is None) == (eval_policy_driver is None):
            raise ValueError('Must either pass an eval_tf_env or an eval_tf_driver.')

        eval_avg_rewards = tf_agents.metrics.tf_metrics.AverageReturnMetric()
        if eval_env is not None:
            if labeling_function is None:
                raise ValueError('Must provide a labeling function if eval_env is provided.')
            eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)
            if self.time_stacked_states:
                latent_eval_env = self.wrap_tf_environment(eval_tf_env, lambda x: labeling_function(x)[:, -1, ...])
            else:
                latent_eval_env = self.wrap_tf_environment(eval_tf_env, labeling_function)
            latent_eval_env.reset()
            latent_policy = latent_eval_env.wrap_latent_policy(self.get_latent_policy(action_dtype=tf.int64))
            eval_policy_driver = tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver(
                latent_eval_env, latent_policy, num_episodes=num_eval_episodes,
                observers=[] if not render else [lambda _: eval_env.render(mode='human')])

        eval_policy_driver.observers.append(eval_avg_rewards)
        eval_policy_driver.run()

        eval_policy_driver.observers.remove(eval_avg_rewards)

        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                tf.summary.scalar('policy_evaluation_avg_rewards', eval_avg_rewards.result(), step=global_step)
        tf.print('eval policy', eval_avg_rewards.result())

        return eval_avg_rewards.result()

    def wrap_tf_environment(
            self,
            tf_env: tf_environment.TFEnvironment,
            labeling_function: Callable[[tf.Tensor], tf.Tensor],
    ) -> tf_environment.TFEnvironment:

        return LatentEmbeddingTFEnvironmentWrapper(
            tf_env=tf_env,
            state_embedding_fn=self.state_embedding_function,
            action_embedding_fn=self.action_embedding_function,
            labeling_fn=labeling_function,
            latent_state_size=self.latent_state_size,
            number_of_discrete_actions=self.number_of_discrete_actions, )

    def get_latent_policy(
            self,
            observation_dtype: tf.dtypes = tf.int32,
            action_dtype: tf.dtypes = tf.int32
    ) -> tf_policy.TFPolicy:

        assert self.latent_policy_network is not None
        action_spec = specs.BoundedTensorSpec(
            shape=(),
            dtype=action_dtype,
            minimum=0,
            maximum=self.number_of_discrete_actions - 1,
            name='action'
        )
        observation_spec = specs.BoundedTensorSpec(
            shape=(self.latent_state_size,),
            dtype=observation_dtype,
            minimum=0,
            maximum=1,
            name='observation'
        )
        time_step_spec = ts.time_step_spec(observation_spec)

        class LatentPolicy(tf_policy.TFPolicy):

            def __init__(
                    self,
                    time_step_spec,
                    action_spec,
                    latent_policy_network
            ):
                super().__init__(time_step_spec, action_spec)
                self._latent_policy_network = latent_policy_network

            def _distribution(self, time_step, policy_state):
                latent_state = tf.cast(time_step.observation, dtype=tf.float32)
                return PolicyStep(
                    tfd.Categorical(
                        logits=self._latent_policy_network(latent_state),
                        dtype=action_dtype),
                    (), ())

        return LatentPolicy(time_step_spec, action_spec, self.latent_policy_network)

    def estimate_local_losses_from_samples(
            self,
            environment: tf_py_environment.TFPyEnvironment,
            steps: int,
            labeling_function: Callable[[tf.Tensor], tf.Tensor],
            estimate_transition_function_from_samples: bool = False,
            assert_estimated_transition_function_distribution: bool = False,
            replay_buffer_max_frames: Optional[int] = int(1e5),
            reward_scaling: Optional[float] = 1.,
            *args, **kwargs
    ):
        if self.latent_policy_network is None:
            raise ValueError('This VAE is not built for policy abstraction.')

        return estimate_local_losses_from_samples(
            environment=environment,
            steps=steps,
            latent_state_size=self.latent_state_size,
            latent_policy=self.get_latent_policy(action_dtype=tf.int64),
            number_of_discrete_actions=self.number_of_discrete_actions,
            state_embedding_function=self.state_embedding_function,
            action_embedding_function=self.action_embedding_function,
            latent_reward_function=lambda latent_state, action, next_latent_state: (
                self.reward_distribution(
                    tf.cast(latent_state, dtype=tf.float32),
                    action,
                    tf.cast(next_latent_state, dtype=tf.float32)).mode()),
            labeling_function=(
                lambda x: labeling_function(x)[:, -1, ...]
            ) if self.time_stacked_states else labeling_function,
            latent_transition_function=lambda state, action: TransitionFnDecorator(
                next_state_distribution=self.discrete_latent_transition(tf.cast(state, tf.float32), action),
                atomic_prop_dims=self.atomic_prop_dims),
            estimate_transition_function_from_samples=estimate_transition_function_from_samples,
            assert_transition_distribution=assert_estimated_transition_function_distribution,
            replay_buffer_max_frames=replay_buffer_max_frames,
            reward_scaling=reward_scaling)


def load(tf_model_path: str, discrete_action=False, step: Optional[int] = None) -> VariationalMarkovDecisionProcess:
    tf_model = tf.saved_model.load(tf_model_path)
    if discrete_action:
        model = VariationalMarkovDecisionProcess(
            state_shape=tuple(tf_model.signatures['serving_default'].structured_input_signature[1]['state'].shape)[1:],
            label_shape=(tf_model.signatures['serving_default'].structured_input_signature[1]['label'].shape[-1] - 1,),
            action_shape=tuple(tf_model.signatures['serving_default'].structured_input_signature[1]['action'].shape)[
                         1:],
            reward_shape=tuple(tf_model.signatures['serving_default'].structured_input_signature[1]['reward'].shape)[
                         1:],
            encoder_network=tf_model.encoder_network,
            transition_network=tf_model.transition_network,
            reward_network=tf_model.reward_network,
            decoder_network=tf_model.reconstruction_network,
            label_transition_network=tf_model.label_transition_network,
            latent_policy_network=tf_model.latent_policy_network,
            latent_state_size=(tf_model.encoder_network.variables[-1].shape[0] +
                               tf_model.signatures['serving_default'].structured_input_signature[1]['label'].shape[-1]),
            encoder_temperature=tf_model._encoder_temperature,
            prior_temperature=tf_model._prior_temperature,
            entropy_regularizer_scale_factor=tf_model._entropy_regularizer_scale_factor,
            kl_scale_factor=tf_model._kl_scale_factor,
            mixture_components=tf.shape(tf_model.reconstruction_network.variables[-1])[-1],
            pre_loaded_model=True,
            action_label_transition_network=tf_model.action_label_transition_network,
            action_transition_network=tf_model.action_transition_network,
            evaluation_window_size=tf.shape(tf_model.evaluation_window)[0], )

    else:
        model = VariationalMarkovDecisionProcess(
            state_shape=tuple(tf_model.signatures['serving_default'].structured_input_signature[1]['state'].shape)[1:],
            label_shape=(tf_model.signatures['serving_default'].structured_input_signature[1]['label'].shape[-1] - 1,),
            action_shape=tuple(tf_model.signatures['serving_default'].structured_input_signature[1]['action'].shape)[
                         1:],
            reward_shape=tuple(tf_model.signatures['serving_default'].structured_input_signature[1]['reward'].shape)[
                         1:],
            encoder_network=tf_model.encoder_network,
            transition_network=tf_model.transition_network,
            reward_network=tf_model.reward_network,
            decoder_network=tf_model.reconstruction_network,
            latent_state_size=(tf_model.encoder_network.variables[-1].shape[0] +
                               tf_model.signatures['serving_default'].structured_input_signature[1]['label'].shape[-1]),
            label_transition_network=tf_model.label_transition_network,
            encoder_temperature=tf_model._encoder_temperature,
            prior_temperature=tf_model._prior_temperature,
            entropy_regularizer_scale_factor=tf_model._entropy_regularizer_scale_factor,
            kl_scale_factor=tf_model._kl_scale_factor,
            mixture_components=tf.shape(tf_model.reconstruction_network.variables[-1])[-1],
            evaluation_window_size=tf.shape(tf_model.evaluation_window)[0],
            pre_loaded_model=True)

    if step is not None:
        path_list = tf_model_path.split(os.sep)
        path_list[path_list.index('models')] = 'training_checkpoints'
        while not os.path.isdir(os.path.join(*path_list)) and len(path_list) > 0:
            path_list.pop()
        if not path_list:
            raise FileNotFoundError('No training checkpoint found for model', model)
        else:
            path_list.append('ckpt-{:d}-1'.format(step))
            checkpoint_path = os.path.join(*path_list)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(checkpoint_path)

    return model
