import os
from typing import Optional, Callable
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Input, Concatenate, Reshape, Dense, Lambda
from tf_agents.environments import tf_py_environment

from belief_learner.wae.mdp import variational_mdp
from belief_learner.utils.old.io import dataset_generator
from belief_learner.wae.mdp.variational_mdp import VariationalMarkovDecisionProcess
from belief_learner.wae.mdp.variational_mdp import epsilon
from belief_learner.verification import estimate_local_losses_from_samples
from belief_learner.verification import TransitionFnDecorator

tfd = tfp.distributions
tfb = tfp.bijectors


class VariationalActionDiscretizer(VariationalMarkovDecisionProcess):

    def __init__(
            self,
            vae_mdp: VariationalMarkovDecisionProcess,
            number_of_discrete_actions: int,
            action_encoder_network: Model,
            action_decoder_network: Model,
            transition_network: Model,
            reward_network: Model,
            latent_policy_network: Model,
            pre_processing_network: Model = Sequential(
                [Dense(units=256, activation='relu'),
                 Dense(units=256, activation='relu')],
                name='pre_processing_network'),
            branching_action_networks: bool = False,
            action_label_transition_network: Optional[Model] = None,
            encoder_temperature: Optional[float] = None,
            prior_temperature: Optional[float] = None,
            encoder_temperature_decay_rate: float = 0.,
            prior_temperature_decay_rate: float = 0.,
            pre_loaded_model: bool = False,
            one_output_per_action: bool = False,
            relaxed_state_encoding: bool = False,
            full_optimization: bool = True,
            reconstruction_mixture_components: int = 1,
            action_entropy_regularizer_scaling: float = 1.,
            importance_sampling_exponent: Optional[float] = None,
            importance_sampling_exponent_growth_rate: Optional[float] = None
    ):

        super().__init__(
            state_shape=vae_mdp.state_shape, action_shape=vae_mdp.action_shape, reward_shape=vae_mdp.reward_shape,
            label_shape=vae_mdp.label_shape, encoder_network=vae_mdp.encoder_network,
            transition_network=vae_mdp.transition_network, label_transition_network=vae_mdp.label_transition_network,
            reward_network=vae_mdp.reward_network,
            decoder_network=vae_mdp.reconstruction_network,
            time_stacked_states=vae_mdp.time_stacked_states,
            latent_state_size=vae_mdp.latent_state_size,
            encoder_temperature=vae_mdp.encoder_temperature.numpy(),
            prior_temperature=vae_mdp.prior_temperature.numpy(),
            encoder_temperature_decay_rate=vae_mdp.encoder_temperature_decay_rate.numpy(),
            prior_temperature_decay_rate=vae_mdp.prior_temperature_decay_rate.numpy(),
            entropy_regularizer_scale_factor=vae_mdp.entropy_regularizer_scale_factor.numpy(),
            entropy_regularizer_decay_rate=vae_mdp.entropy_regularizer_decay_rate.numpy(),
            entropy_regularizer_scale_factor_min_value=vae_mdp.entropy_regularizer_scale_factor_min_value.numpy(),
            marginal_entropy_regularizer_ratio=vae_mdp.marginal_entropy_regularizer_ratio,
            kl_scale_factor=vae_mdp.kl_scale_factor.numpy(),
            kl_annealing_growth_rate=vae_mdp.kl_growth_rate.numpy(),
            mixture_components=vae_mdp.mixture_components,
            multivariate_normal_raw_scale_diag_activation=vae_mdp.scale_activation,
            multivariate_normal_full_covariance=vae_mdp.full_covariance,
            pre_loaded_model=True,
            importance_sampling_exponent=vae_mdp.is_exponent,
            importance_sampling_exponent_growth_rate=vae_mdp.is_exponent_growth_rate,
            optimizer=vae_mdp._optimizer,
            evaluation_window_size=tf.shape(vae_mdp.evaluation_window)[0],
            evaluation_criterion=vae_mdp.evaluation_criterion,
            reward_bounds=None if vae_mdp._reward_softclip is None else (
                vae_mdp._reward_softclip.low, vae_mdp._reward_softclip.high))

        if encoder_temperature is None:
            encoder_temperature = 1. / (number_of_discrete_actions - 1)
        if prior_temperature is None:
            prior_temperature = encoder_temperature / 1.5

        self.number_of_discrete_actions = number_of_discrete_actions
        self._state_vae = vae_mdp
        self.one_output_per_action = one_output_per_action
        self.relaxed_state_encoding = relaxed_state_encoding or full_optimization
        self.full_optimization = full_optimization
        self.mixture_components = reconstruction_mixture_components

        self.state_encoder_temperature = self.encoder_temperature
        self.state_prior_temperature = self.prior_temperature
        self.state_encoder_temperature_decay_rate = self.encoder_temperature_decay_rate
        self.state_prior_temperature_decay_rate = self.prior_temperature_decay_rate

        self.encoder_temperature = tf.Variable(encoder_temperature, dtype=tf.float32, trainable=False)
        self.prior_temperature = tf.Variable(prior_temperature, dtype=tf.float32, trainable=False)
        self.encoder_temperature_decay_rate = tf.constant(encoder_temperature_decay_rate, dtype=tf.float32)
        self.prior_temperature_decay_rate = tf.constant(prior_temperature_decay_rate, dtype=tf.float32)
        self._action_entropy_regularizer_scaling = tf.constant(action_entropy_regularizer_scaling, dtype=tf.float32)
        if importance_sampling_exponent is not None:
            self.is_exponent = importance_sampling_exponent
        if importance_sampling_exponent_growth_rate is not None:
            self.is_exponent_growth_rate = importance_sampling_exponent_growth_rate

        def clone_model(model: tf.keras.Model, copy_name: str = ''):
            model = model_from_json(model.to_json(), custom_objects={'leaky_relu': tf.nn.leaky_relu})
            for layer in model.layers:
                layer._name = copy_name + '_' + layer.name
            model._name = copy_name + '_' + model.name
            return model

        if not pre_loaded_model:
            # action encoder network
            latent_state = Input(shape=(self.latent_state_size,))
            action = Input(shape=self.action_shape)
            next_latent_state = Input(shape=(self.latent_state_size,))
            next_label = Input(shape=(self.atomic_prop_dims,))
            latent_action = Input(shape=(number_of_discrete_actions,)) if not one_output_per_action else None

            action_encoder = Concatenate(name="action_encoder_input")(
                [latent_state, action])
            action_encoder = action_encoder_network(action_encoder)
            action_encoder = Dense(
                units=number_of_discrete_actions,
                activation=None,
                name='action_encoder_exp_one_hot_logits'
            )(action_encoder)
            self.action_encoder = Model(
                inputs=[latent_state, action],
                outputs=action_encoder,
                name="action_encoder")

            # prior over actions
            self.latent_policy_network = latent_policy_network(latent_state)
            self.latent_policy_network = Dense(
                units=self.number_of_discrete_actions,
                activation=None,
                name='action_discretizer_latent_policy_exp_one_hot_logits'
            )(self.latent_policy_network)
            self.latent_policy_network = Model(
                inputs=latent_state,
                outputs=self.latent_policy_network,
                name='action_discretizer_latent_policy_network')

            # discrete actions transition network
            if not one_output_per_action:
                if action_label_transition_network is not None:
                    _transition_network = Concatenate()([latent_state, latent_action, next_label])
                else:
                    _transition_network = Concatenate()([latent_state, latent_action])
                _transition_network = transition_network(_transition_network)
                _next_label_logits = (None if action_label_transition_network is not None else
                                      Dense(units=self.atomic_prop_dims,
                                            activation=None,
                                            name='discrete_action_transition_next_label_logits'
                                      )(_transition_network))
                _next_state_logits = Dense(
                    units=self.latent_state_size - self.atomic_prop_dims,
                    activation=None,
                    name='discrete_action_transition_next_state_logits'
                )(_transition_network)

                self.action_transition_network = Model(
                    inputs=[latent_state, latent_action, next_label] if action_label_transition_network is not None else [latent_state, latent_action],
                    outputs=_next_state_logits if action_label_transition_network is not None else [_next_label_logits, _next_state_logits],
                    name="transition_network")
            else:
                if action_label_transition_network is not None:
                    transition_network_input = Concatenate()([latent_state, next_label])
                else:
                    transition_network_input = latent_state
                transition_network_pre_processing = clone_model(
                    pre_processing_network, 'transition')(transition_network_input)
                transition_outputs_next_label = []
                transition_outputs_next_state = []
                for action in range(number_of_discrete_actions):
                    if branching_action_networks:
                        _transition_network = clone_model(transition_network, str(action))
                    else:
                        _transition_network = transition_network
                    _transition_network = _transition_network(transition_network_pre_processing)
                    _next_state_logits = Dense(
                        units=self.latent_state_size - self.atomic_prop_dims,
                        activation=None,
                        name='transition_next_state_logits_action{}'.format(action)
                    )(_transition_network)
                    transition_outputs_next_state.append(_next_state_logits)
                    if action_label_transition_network is None:
                        _next_label_logits = Dense(
                            units=self.atomic_prop_dims,
                            activation=None,
                            name='transition_next_label_logits_action{}'.format(action)
                        )(_transition_network)
                        transition_outputs_next_label.append(_next_label_logits)
                next_label_logits = Lambda(
                    lambda x: tf.stack(x, axis=1),
                    name="transition_network_output_label"
                )(transition_outputs_next_label) if action_label_transition_network is None else None
                next_state_logits = Lambda(
                    lambda x: tf.stack(x, axis=1),
                    name="transition_network_output_state"
                )(transition_outputs_next_state)
                self.action_transition_network = Model(
                    inputs=latent_state if next_label_logits is not None else [latent_state, next_label],
                    outputs=[next_label_logits, next_state_logits] if next_label_logits is not None else next_state_logits,
                    name="transition_network")

            # label transition network
            if not one_output_per_action and action_label_transition_network is not None:
                _label_transition_network = Concatenate()([latent_state, latent_action])
                _label_transition_network = action_label_transition_network(_label_transition_network)
                _label_transition_network = Dense(
                    units=self.atomic_prop_dims,
                    name='next_label_transition_logits')(_label_transition_network)
                self.action_label_transition_network = Model(
                    inputs=[latent_state, latent_action],
                    outputs=_label_transition_network,
                    name='label_transition_network')
            elif action_label_transition_network is not None:
                if branching_action_networks:
                    _action_label_transition_network = clone_model(action_label_transition_network, str(action))
                else:
                    _action_label_transition_network = action_label_transition_network
                _label_transition_network = _action_label_transition_network(latent_state)
                outputs = []
                for action in range(self.number_of_discrete_actions):
                    _label_transition_network = Dense(
                        units=self.atomic_prop_dims,
                        name='next_label_transition_logits_action{}'.format(action)
                    )(_label_transition_network)
                    outputs.append(_label_transition_network)
                self.action_label_transition_network = Model(
                    inputs=latent_state,
                    outputs=Lambda(
                        lambda x: tf.stack(x, axis=1),
                        name='label_transition_network_output')(outputs),
                    name='label_transition_network')
            else:
                self.action_label_transition_network = None

            # discrete actions reward network
            if not one_output_per_action:
                _reward_network = Concatenate()([latent_state, latent_action, next_latent_state])
                _reward_network = reward_network(_reward_network)
                reward_mean = Dense(
                    units=np.prod(self.reward_shape),
                    activation=None if self._reward_softclip is None else lambda x: self._reward_softclip(x),
                    name='action_reward_mean_0'
                )(_reward_network)
                reward_mean = Reshape(self.reward_shape, name='action_reward_mean')(
                    reward_mean)
                reward_raw_covar = Dense(
                    units=np.prod(self.reward_shape),
                    activation=None,
                    name='action_reward_raw_diag_covariance_0'
                )(_reward_network)
                reward_raw_covar = Reshape(
                    self.reward_shape,
                    name='action_reward_raw_diag_covariance'
                )(reward_raw_covar)
                self.action_reward_network = Model(
                    inputs=[latent_state, latent_action, next_latent_state],
                    outputs=[reward_mean, reward_raw_covar],
                    name="reward_network")
            else:
                reward_network_input = Concatenate()([latent_state, next_latent_state])
                reward_network_pre_processing = clone_model(pre_processing_network, 'reward')(reward_network_input)
                reward_network_outputs = []
                for action in range(number_of_discrete_actions):
                    if branching_action_networks:
                        _reward_network = clone_model(reward_network, str(action))
                    else:
                        _reward_network = reward_network
                    _reward_network = _reward_network(reward_network_pre_processing)
                    reward_mean = Dense(
                        units=np.prod(self.reward_shape),
                        activation=None if self._reward_softclip is None else lambda x: self._reward_softclip(x),
                        name='reward_mean_0_action{}'.format(action)
                    )(_reward_network)
                    reward_mean = Reshape(
                        self.reward_shape,
                        name='action{}_reward_mean'.format(action))(reward_mean)
                    reward_raw_covar = Dense(
                        units=np.prod(self.reward_shape),
                        activation=None,
                        name='reward_raw_diag_covariance_0_action{}'.format(action)
                    )(_reward_network)
                    reward_raw_covar = Reshape(
                        self.reward_shape,
                        name='reward_raw_diag_covariance_action{}'.format(action)
                    )(reward_raw_covar)
                    reward_network_outputs.append([reward_mean, reward_raw_covar])
                reward_network_mean = Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                    list(mean for mean, covariance in reward_network_outputs))
                reward_network_raw_covariance = Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                    list(covariance for mean, covariance in reward_network_outputs))
                self.action_reward_network = Model(
                    inputs=[latent_state, next_latent_state],
                    outputs=[reward_network_mean, reward_network_raw_covariance],
                    name="discrete_actions_reward_network")

            # discrete actions decoder
            if self.mixture_components > 1:
                action_shape = (self.mixture_components,) + self.action_shape
            else:
                action_shape = self.action_shape
            if not one_output_per_action:
                action_decoder = Concatenate()([latent_state, latent_action])
                action_decoder = action_decoder_network(action_decoder)
                action_decoder_mean = Dense(
                    units=self.mixture_components * np.prod(self.action_shape),
                    name='action_decoder_mean_raw_output',
                    activation=None
                )(action_decoder)
                action_decoder_mean = Reshape(
                    target_shape=action_shape,
                    name='action_decoder_mean'
                )(action_decoder_mean)
                action_decoder_raw_covariance = Dense(
                    units=self.mixture_components * np.prod(self.action_shape),
                    name='action_decoder_raw_covariance_output',
                    activation=None
                )(action_decoder)
                action_decoder_raw_covariance = Reshape(
                    target_shape=action_shape,
                    name='action_decoder_diag_covariance'
                )(action_decoder_raw_covariance)
                action_decoder_mixture_categorical_logits = Dense(
                    units=self.mixture_components,
                    activation=None,
                    name='action_decoder_mixture_categorical_logits'
                )(action_decoder)
                self.action_decoder_network = Model(
                    inputs=[latent_state, latent_action],
                    outputs=[
                        action_decoder_mean,
                        action_decoder_raw_covariance,
                        action_decoder_mixture_categorical_logits
                    ],
                    name="action_decoder_network")
            else:
                action_decoder_pre_processing = clone_model(pre_processing_network, 'action')(latent_state)
                action_decoder_outputs = []
                for action in range(number_of_discrete_actions):
                    if branching_action_networks:
                        action_decoder = clone_model(action_decoder_network, str(action))
                    else:
                        action_decoder = action_decoder_network
                    action_decoder = action_decoder(action_decoder_pre_processing)
                    action_decoder_mean = Dense(
                        units=self.mixture_components * np.prod(self.action_shape),
                        activation=None
                    )(action_decoder)
                    action_decoder_mean = Reshape(
                        target_shape=action_shape,
                        name='action{}_decoder_mean'.format(action)
                    )(action_decoder_mean)
                    action_decoder_raw_covariance = Dense(
                        units=self.mixture_components * np.prod(self.action_shape),
                        activation=None
                    )(action_decoder)
                    action_decoder_raw_covariance = Reshape(
                        target_shape=action_shape,
                        name='action{}_decoder_raw_diag_covariance'.format(action)
                    )(action_decoder_raw_covariance)
                    action_decoder_mixture_categorical_logits = Dense(
                        units=self.mixture_components,
                        activation=None,
                        name='action{}_decoder_mixture_categorical_logits'.format(action)
                    )(action_decoder)
                    action_decoder_outputs.append(
                        (action_decoder_mean, action_decoder_raw_covariance, action_decoder_mixture_categorical_logits)
                    )
                action_decoder_mean = Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                    list(mean for mean, covariance, component_logits in action_decoder_outputs))
                action_decoder_raw_covariance = Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                    list(covariance for mean, covariance, component_logits in action_decoder_outputs))
                action_decoder_mixture_categorical_logits = Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                    list(component_logits for mean, covariance, component_logits in action_decoder_outputs))
                self.action_decoder_network = Model(
                    inputs=latent_state,
                    outputs=[
                        action_decoder_mean,
                        action_decoder_raw_covariance,
                        action_decoder_mixture_categorical_logits
                    ],
                    name="action_decoder_network")

        else:
            self.action_encoder = action_encoder_network
            self.latent_policy_network = latent_policy_network
            self.action_transition_network = transition_network
            self.action_reward_network = reward_network
            self.action_decoder_network = action_decoder_network
            self.action_label_transition_network = action_label_transition_network

        self.loss_metrics = {
            'ELBO': tf.keras.metrics.Mean(name='ELBO'),
            'action_mse': tf.keras.metrics.MeanSquaredError(name='action_mse'),
            'reward_mse': tf.keras.metrics.MeanSquaredError(name='reward_mse'),
            'distortion': tf.keras.metrics.Mean(name='distortion'),
            'rate': tf.keras.metrics.Mean(name='rate'),
            # 'annealed_rate': tf.keras.metrics.Mean(name='annealed_rate'),
            'entropy_regularizer': tf.keras.metrics.Mean(name='entropy_regularizer'),
            'transition_log_probs': tf.keras.metrics.Mean(name='transition_log_probs'),
            # 'decoder_divergence': tf.keras.metrics.Mean(name='decoder_divergence'),
            'action_decoder_std': tf.keras.metrics.Mean(name='action_decoder_std')
        }
        if self.full_optimization:
            self.loss_metrics.update({
                'state_mse': tf.keras.metrics.MeanSquaredError(name='state_mse'),
                'state_encoder_entropy': tf.keras.metrics.Mean(name='encoder_entropy'),
                'marginal_encoder_entropy': tf.keras.metrics.Mean(name='marginal_encoder_entropy'),
                'action_encoder_entropy': tf.keras.metrics.Mean(name='action_encoder_entropy'),
                # 'state_decoder_variance': tf.keras.metrics.Mean('decoder_variance'),
                # 'state_rate': tf.keras.metrics.Mean(name='state_rate'),
                # 'action_rate': tf.keras.metrics.Mean(name='action_rate'),
                't_1_state': tf.keras.metrics.Mean(name='state_encoder_temperature'),
                't_2_state': tf.keras.metrics.Mean(name='state_prior_temperature'),
            })
        self.temperature_metrics = {
            't_1': self.state_encoder_temperature,
            't_2': self.state_prior_temperature,
            't_1_action': self.encoder_temperature,
            't_2_action': self.prior_temperature
        }

    def anneal(self):
        super().anneal()
        for var, decay_rate in [
            (self._state_vae.encoder_temperature, self._state_vae.encoder_temperature_decay_rate),
            (self._state_vae.prior_temperature, self._state_vae.prior_temperature_decay_rate),
        ]:
            if decay_rate.numpy().all() > 0:
                var.assign(var * (1. - decay_rate))

    def relaxed_action_encoding(
            self, latent_state: tf.Tensor, action: tf.Tensor, temperature: float
    ) -> tfd.Distribution:
        encoder_logits = self.action_encoder([latent_state, action])
        return tfd.ExpRelaxedOneHotCategorical(temperature=temperature, logits=encoder_logits, allow_nan_stats=False)

    def discrete_action_encoding(self, latent_state: tf.Tensor, action: tf.Tensor) -> tfd.Distribution:
        relaxed_distribution = self.relaxed_action_encoding(latent_state, action, 1e-5)
        log_probs = tf.math.log(relaxed_distribution.probs_parameter() + epsilon)
        return tfd.OneHotCategorical(logits=log_probs, allow_nan_stats=False)

    def discrete_latent_transition(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor,
            relaxed_state_encoding: bool = False, log_latent_action: bool = False
    ) -> tfd.Distribution:

        if not self.one_output_per_action:
            if log_latent_action:
                latent_action = tf.exp(latent_action)

            if self.action_label_transition_network is None:
                next_label_logits, _next_state_logits = self.action_transition_network([latent_state, latent_action])
                next_state_logits = lambda _: _next_state_logits
            else:
                next_label_logits = self.action_label_transition_network([latent_state, latent_action])
                next_state_logits = lambda _next_label: self.action_transition_network(
                    [latent_state, latent_action, _next_label])

            if relaxed_state_encoding:
                return tfd.JointDistributionSequential([
                    tfd.Independent(
                        tfd.Bernoulli(
                            logits=next_label_logits,
                            allow_nan_stats=False,
                            dtype=tf.float32, )),
                    lambda _next_label: tfd.Independent(
                        tfd.Logistic(
                            loc=next_state_logits(_next_label) / self._state_vae.prior_temperature,
                            scale=1. / self._state_vae.prior_temperature,
                            allow_nan_stats=False))])
            else:
                return tfd.JointDistributionSequential([
                    tfd.Independent(
                        tfd.Bernoulli(
                            logits=next_label_logits,
                            allow_nan_stats=False,
                            dtype=tf.float32)),
                    lambda _next_label: tfd.Independent(
                        tfd.Bernoulli(
                            logits=next_state_logits(_next_label),
                            allow_nan_stats=False))])
        else:
            if log_latent_action:
                action_categorical = tfd.Categorical(logits=latent_action, allow_nan_stats=False)
            else:
                action_categorical = tfd.Categorical(probs=latent_action, allow_nan_stats=False)

            if self.action_label_transition_network is None:
                next_label_logits, _next_state_logits = self.action_transition_network(latent_state)
                next_state_logits = lambda _: _next_state_logits
            else:
                next_label_logits = self.action_label_transition_network(latent_state)
                next_state_logits = lambda _next_label: self.action_transition_network(
                    [latent_state, _next_label])

            if relaxed_state_encoding:
                return tfd.JointDistributionSequential([
                    tfd.MixtureSameFamily(
                        mixture_distribution=action_categorical,
                        components_distribution=tfd.Independent(
                            tfd.Bernoulli(
                                logits=next_label_logits,
                                allow_nan_stats=False,
                                dtype=tf.float32),
                            reinterpreted_batch_ndims=1),
                        allow_nan_stats=False),
                    lambda _next_label: tfd.MixtureSameFamily(
                        mixture_distribution=action_categorical,
                        components_distribution=tfd.Independent(
                            tfd.Logistic(
                                loc=next_state_logits(_next_label) / self._state_vae.prior_temperature,
                                scale=1. / self._state_vae.prior_temperature,
                                allow_nan_stats=False),
                            reinterpreted_batch_ndims=1),
                        allow_nan_stats=False)])
            else:
                return tfd.JointDistributionSequential([
                    tfd.MixtureSameFamily(
                        mixture_distribution=action_categorical,
                        components_distribution=tfd.Independent(
                            tfd.Bernoulli(
                                logits=next_label_logits,
                                allow_nan_stats=False,
                                dtype=tf.float32),
                            reinterpreted_batch_ndims=1),
                        allow_nan_stats=False),
                    lambda _next_label: tfd.MixtureSameFamily(
                        mixture_distribution=action_categorical,
                        components_distribution=tfd.Independent(
                            tfd.Bernoulli(
                                logits=next_state_logits(_next_label),
                                allow_nan_stats=False),
                            reinterpreted_batch_ndims=1),
                        allow_nan_stats=False)])

    def reward_distribution(
            self, latent_state, latent_action, next_latent_state, log_latent_action: bool = False,
            disable_mixture_distribution: bool = False
    ) -> tfd.Distribution:

        if not self.one_output_per_action:
            if log_latent_action:
                latent_action = tf.exp(latent_action)

            [reward_mean, reward_raw_covariance] = self.action_reward_network(
                [latent_state, latent_action, next_latent_state])

            return tfd.MultivariateNormalDiag(
                loc=reward_mean,
                scale_diag=self.scale_activation(reward_raw_covariance),
                allow_nan_stats=False)
        else:
            if log_latent_action:
                action_categorical = tfd.Categorical(logits=latent_action, allow_nan_stats=False)
            else:
                action_categorical = tfd.Categorical(probs=latent_action, allow_nan_stats=False)

            [reward_mean, reward_raw_covariance] = self.action_reward_network([latent_state, next_latent_state])

            if not log_latent_action and disable_mixture_distribution:
                # all actions are assumed to be given one-hot
                _latent_action = tf.cast(tf.stop_gradient(tf.argmax(latent_action, axis=-1)), dtype=tf.int32)
                return tfd.MultivariateNormalDiag(
                    loc=tf.stop_gradient(tf.map_fn(
                        fn=lambda i: reward_mean[i, _latent_action[i], ...],
                        elems=tf.range(tf.shape(_latent_action)[0]),
                        fn_output_signature=tf.float32)),
                    scale_diag=tf.stop_gradient(tf.map_fn(
                        fn=lambda i: self.scale_activation(reward_raw_covariance[i, _latent_action[i], ...]),
                        elems=tf.range(tf.shape(_latent_action)[0]),
                        fn_output_signature=tf.float32)),
                    allow_nan_stats=False)
            else:
                return tfd.MixtureSameFamily(
                    mixture_distribution=action_categorical,
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=reward_mean,
                        scale_diag=self.scale_activation(reward_raw_covariance),
                        allow_nan_stats=False
                    ), allow_nan_stats=False)

    def decode_action(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor, log_latent_action: bool = False,
            disable_mixture_distribution: bool = False
    ) -> tfd.Distribution:

        if not self.one_output_per_action:
            if log_latent_action:
                latent_action = tf.exp(latent_action)

            [action_mean, action_raw_covariance, cat_logits] = self.action_decoder_network(
                [latent_state, latent_action])
            if self.mixture_components == 1:
                return tfd.MultivariateNormalDiag(
                    loc=action_mean,
                    scale_diag=self.scale_activation(action_raw_covariance),
                    allow_nan_stats=False)
            else:
                return tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(logits=cat_logits, allow_nan_stats=False),
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=action_mean,
                        scale_diag=self.scale_activation(action_raw_covariance),
                        allow_nan_stats=False),
                    allow_nan_stats=False)
        else:
            if log_latent_action:
                action_categorical = tfd.Categorical(logits=latent_action, allow_nan_stats=False)
            else:
                action_categorical = tfd.Categorical(probs=latent_action, allow_nan_stats=False)

            [action_mean, action_raw_covariance, cat_logits] = self.action_decoder_network(latent_state)

            if self.mixture_components == 1:
                if not log_latent_action and disable_mixture_distribution:
                    # all actions are assumed to be given one-hot
                    _latent_action = tf.cast(tf.stop_gradient(tf.argmax(latent_action, axis=-1)), dtype=tf.int32)
                    return tfd.MultivariateNormalDiag(
                        loc=tf.stop_gradient(tf.map_fn(
                            fn=lambda i: action_mean[i, _latent_action[i], ...],
                            elems=tf.range(tf.shape(_latent_action)[0]),
                            fn_output_signature=tf.float32)),
                        scale_diag=tf.stop_gradient(tf.map_fn(
                            fn=lambda i: self.scale_activation(action_raw_covariance[i, _latent_action[i], ...]),
                            elems=tf.range(tf.shape(_latent_action)[0]),
                            fn_output_signature=tf.float32)),
                        allow_nan_stats=False)
                else:
                    return tfd.MixtureSameFamily(
                        mixture_distribution=action_categorical,
                        components_distribution=tfd.MultivariateNormalDiag(
                            loc=action_mean,
                            scale_diag=self.scale_activation(action_raw_covariance),
                            allow_nan_stats=False),
                        allow_nan_stats=False)
            else:
                return tfd.MixtureSameFamily(
                    mixture_distribution=action_categorical,
                    components_distribution=tfd.MixtureSameFamily(
                        mixture_distribution=tfd.Categorical(logits=cat_logits),
                        components_distribution=tfd.MultivariateNormalDiag(
                            loc=action_mean,
                            scale_diag=self.scale_activation(action_raw_covariance),
                            allow_nan_stats=False),
                        allow_nan_stats=False),
                    allow_nan_stats=False)

    def relaxed_latent_policy(self, latent_state: tf.Tensor, temperature: float):
        return tfd.ExpRelaxedOneHotCategorical(
            temperature=temperature, logits=self.latent_policy_network(latent_state), allow_nan_stats=False)

    def discrete_latent_policy(self, latent_state: tf.Tensor):
        relaxed_distribution = self.relaxed_latent_policy(latent_state, temperature=1e-5)
        log_probs = tf.math.log(relaxed_distribution.probs_parameter() + epsilon)
        return tfd.OneHotCategorical(logits=log_probs, allow_nan_stats=False)

    def action_embedding_function(
            self,
            latent_state: tf.Tensor,
            latent_action: tf.Tensor,
    ) -> tf.Tensor:

        return self.decode_action(
            latent_state=tf.cast(latent_state, dtype=tf.float32),
            latent_action=tf.cast(tf.one_hot(latent_action, depth=self.number_of_discrete_actions), dtype=tf.float32),
            disable_mixture_distribution=True
        ).mode()

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
    ):
        if self.full_optimization:
            return self._full_optimization_call(state, label, action, reward, next_state, next_label, sample_key)

        if self.relaxed_state_encoding:
            logistic_latent_state = self.relaxed_state_encoding(state, self._state_vae.encoder_temperature).sample()
            latent_state = tf.concat([label, tf.sigmoid(logistic_latent_state)], axis=-1)
            next_latent_state_no_label = self._state_vae.relaxed_state_encoding(
                next_state, self._state_vae.encoder_temperature).sample()
        else:
            latent_state = tf.concat([label, tf.cast(self.binary_encode_state(state).sample(), tf.float32)])
            next_latent_state_no_label = tf.cast(self.binary_encode_state(next_state).sample())
        q = self.relaxed_action_encoding(latent_state, action, self.encoder_temperature)
        p = self.relaxed_latent_policy(latent_state, self.prior_temperature)
        log_latent_action = q.sample()

        log_q_log_latent_action = q.log_prob(log_latent_action)
        log_p_log_latent_action = p.log_prob(log_latent_action)

        # transition probability reconstruction
        transition_probability_distribution = \
            self.discrete_latent_transition(
                latent_state, log_latent_action,
                relaxed_state_encoding=self.relaxed_state_encoding, log_latent_action=True)
        if self.relaxed_state_encoding:
            continuous_action_transition = self._state_vae.relaxed_latent_transition(
                latent_state, action, self._state_vae.prior_temperature)
        else:
            continuous_action_transition = self._state_vae.discrete_latent_transition(
                latent_state, action)
        log_p_transition_action = continuous_action_transition.log_prob(next_label, next_latent_state_no_label)
        log_p_transition_latent_action = transition_probability_distribution.log_prob(
            next_label, next_latent_state_no_label)
        log_p_transition = log_p_transition_latent_action - log_p_transition_action

        if self.relaxed_state_encoding:
            next_latent_state = tf.concat([next_label, tf.sigmoid(next_latent_state_no_label)], axis=-1)
        else:
            next_latent_state = tf.concat([next_label, next_latent_state_no_label], axis=-1)

        # rewards reconstruction
        reward_distribution = self.reward_distribution(
            latent_state, log_latent_action, next_latent_state, log_latent_action=True)
        log_p_rewards_action = self._state_vae.reward_distribution(
            latent_state, action, next_latent_state).log_prob(reward)
        log_p_rewards_latent_action = reward_distribution.log_prob(reward)
        log_p_rewards = log_p_rewards_latent_action - log_p_rewards_action

        # action reconstruction
        action_distribution = self.decode_action(latent_state, log_latent_action, log_latent_action=True)
        log_p_action = action_distribution.log_prob(action)

        rate = log_q_log_latent_action - log_p_log_latent_action
        distortion = -1. * (log_p_action + log_p_rewards + log_p_transition)

        entropy_regularizer = self.entropy_regularizer(latent_state, action)

        # priority support
        if self.priority_handler is not None and sample_key is not None:
            tf.stop_gradient(
                self.priority_handler.update_priority(
                    keys=sample_key,
                    latent_states=tf.stop_gradient(tf.cast(tf.round(latent_state), tf.int32)),
                    loss=tf.stop_gradient(distortion + rate)))

        # metrics
        self.loss_metrics['ELBO'](tf.stop_gradient(-1. * (distortion + rate)))
        self.loss_metrics['action_mse'](action, tf.stop_gradient(action_distribution.sample()))
        self.loss_metrics['reward_mse'](reward, tf.stop_gradient(reward_distribution.sample()))
        self.loss_metrics['distortion'](distortion)
        self.loss_metrics['rate'](rate)
        # self.loss_metrics['annealed_rate'](tf.stop_gradient(self.kl_scale_factor * rate))
        self.loss_metrics['entropy_regularizer'](
            tf.stop_gradient(self.entropy_regularizer_scale_factor * entropy_regularizer))
        # if self.one_output_per_action:
        #     self.loss_metrics['decoder_divergence'](self._compute_decoder_jensen_shannon_divergence(z, a_1))
        self.loss_metrics['action_decoder_std'](action_distribution.stddev())

        if variational_mdp.debug:
            tf.print(latent_state, "sampled z", summarize=variational_mdp.debug_verbosity)
            tf.print(next_latent_state, "sampled z'", summarize=variational_mdp.debug_verbosity)
            tf.print(q.logits, "logits of Q_action", summarize=variational_mdp.debug_verbosity)
            tf.print(p.logits, "logits of P_action", summarize=variational_mdp.debug_verbosity)
            tf.print(log_latent_action, "sampled log action from Q", summarize=variational_mdp.debug_verbosity)
            tf.print(log_q_log_latent_action, "log Q(exp_action)", summarize=variational_mdp.debug_verbosity)
            tf.print(log_p_log_latent_action, "log P(exp_action)", summarize=variational_mdp.debug_verbosity)
            tf.print(log_p_rewards, "log P(r | z, â, z')", summarize=variational_mdp.debug_verbosity)
            tf.print(log_p_transition, "log P(z' | z, â)", summarize=variational_mdp.debug_verbosity)
            tf.print(log_p_action, "log P(a | z, â)", summarize=variational_mdp.debug_verbosity)

        return {'distortion': distortion, 'rate': rate, 'entropy_regularizer': entropy_regularizer}

    def _full_optimization_call(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None
    ):
        # sampling from encoder distributions
        latent_state_encoder = self._state_vae.relaxed_state_encoding(state, self._state_vae.encoder_temperature)
        next_latent_state_encoder = self._state_vae.relaxed_state_encoding(next_state, self._state_vae.encoder_temperature)
        latent_state = tf.concat([label, tf.sigmoid(latent_state_encoder.sample())], axis=-1)
        next_logistic_latent_state_no_label = next_latent_state_encoder.sample()
        latent_action_encoder = self.relaxed_action_encoding(latent_state, action, self.encoder_temperature)
        latent_policy = self.relaxed_latent_policy(latent_state, self.prior_temperature)
        log_latent_action = latent_action_encoder.sample()

        # action encoder rate
        log_q_log_latent_action = latent_action_encoder.log_prob(log_latent_action)
        log_pi_log_latent_action = latent_policy.log_prob(log_latent_action)
        action_encoder_rate = log_q_log_latent_action - log_pi_log_latent_action

        # transitions
        transition_probability_distribution = self.discrete_latent_transition(
            latent_state, log_latent_action, relaxed_state_encoding=True, log_latent_action=True)
        log_p_next_latent_state = transition_probability_distribution.log_prob(
            next_label, next_logistic_latent_state_no_label)

        # state encoder rate
        log_q_next_latent_state = next_latent_state_encoder.log_prob(next_logistic_latent_state_no_label)
        state_encoder_rate = log_q_next_latent_state - log_p_next_latent_state

        rate = state_encoder_rate + action_encoder_rate

        next_latent_state = tf.concat([next_label, tf.sigmoid(next_logistic_latent_state_no_label)], axis=-1)

        # Reconstruction
        # log P(a, r, s' | z, â, z') = log P(a | z, â) + log P(r | z, â, z') + log P(s' | z')
        action_distribution = self.decode_action(latent_state, log_latent_action, log_latent_action=True)
        reconstruction_distribution = tfd.JointDistributionSequential([
            action_distribution,
            self.reward_distribution(
                latent_state, log_latent_action, next_latent_state, log_latent_action=True),
            self.decode_state(next_latent_state)
        ])

        distortion = -1. * reconstruction_distribution.log_prob(action, reward, next_state)

        entropy_regularizer = self.entropy_regularizer(
            latent_state, action, state=state,
            use_marginal_entropy=True)

        # priority support
        if self.priority_handler is not None and sample_key is not None:
            tf.stop_gradient(
                self.priority_handler.update_priority(
                    keys=sample_key,
                    latent_states=tf.stop_gradient(tf.cast(tf.round(latent_state), tf.int32)),
                    loss=tf.stop_gradient(distortion + rate)))

        # metrics
        action_sample, reward_sample, next_state_sample = reconstruction_distribution.sample()
        self.loss_metrics['ELBO'](tf.stop_gradient(-1. * (distortion + rate)))
        self.loss_metrics['action_mse'](action, action_sample)
        self.loss_metrics['reward_mse'](reward, reward_sample)
        self.loss_metrics['state_mse'](next_state, next_state_sample)
        # self.loss_metrics['state_rate'](state_encoder_rate)
        # self.loss_metrics['state_encoder_entropy'](self._state_vae.binary_encode(next_state, next_label).entropy())
        self.loss_metrics['action_encoder_entropy'](
            tf.stop_gradient(self.discrete_action_encoding(latent_state, action).entropy()))
        #  self.loss_metrics['state_decoder_variance'](state_distribution.variance())
        # self.loss_metrics['action_rate'](action_encoder_rate)
        self.loss_metrics['distortion'](distortion)
        self.loss_metrics['rate'](rate)
        # self.loss_metrics['annealed_rate'](tf.stop_gradient(self.kl_scale_factor * rate))
        self.loss_metrics['entropy_regularizer'](
            tf.stop_gradient(self.entropy_regularizer_scale_factor * entropy_regularizer))
        self.loss_metrics['t_1_state'].reset_states()
        self.loss_metrics['t_1_state'](self._state_vae.encoder_temperature)
        self.loss_metrics['t_2_state'].reset_states()
        self.loss_metrics['t_2_state'](self._state_vae.prior_temperature)
        self.loss_metrics['transition_log_probs'](
            tf.stop_gradient(
                self.discrete_latent_transition(
                    tf.stop_gradient(tf.round(latent_state)),
                    tf.stop_gradient(tf.math.log(tf.round(tf.exp(log_latent_action)) + epsilon)),
                    log_latent_action=True
                ).log_prob(next_label, tf.stop_gradient(tf.round(tf.sigmoid(next_logistic_latent_state_no_label))))))
        self.loss_metrics['action_decoder_std'](tf.stop_gradient(action_distribution.stddev()))

        return {'distortion': distortion, 'rate': rate, 'entropy_regularizer': entropy_regularizer}

    @tf.function
    def entropy_regularizer(
            self,
            latent_state: tf.Tensor,
            action: tf.Tensor,
            state: Optional[tf.Tensor] = None,
            enforce_deterministic_action_encoder: bool = False,
            use_marginal_entropy: bool = False
    ):
        if state is not None:
            state_regularizer = super().entropy_regularizer(
                state=state,
                use_marginal_entropy=use_marginal_entropy,
                latent_states=latent_state)
        else:
            state_regularizer = 0.
        if self.entropy_regularizer_scale_factor < 0. and not enforce_deterministic_action_encoder:
            action_regularizer = 0.
        else:
            action_regularizer = -1. * self._action_entropy_regularizer_scaling * tf.reduce_mean(
                self.discrete_action_encoding(latent_state, action).entropy(), axis=0)

        return state_regularizer + action_regularizer

    def eval(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor
    ):
        latent_distribution = self.binary_encode_state(state)
        next_latent_distribution = self.binary_encode_state(next_state)
        latent_state = tf.concat([label, tf.cast(latent_distribution.sample(), tf.float32)], axis=-1)
        next_latent_state_no_label = tf.cast(next_latent_distribution.sample(), tf.float32)

        latent_action_encoder = self.discrete_action_encoding(latent_state, action)
        latent_policy = self.discrete_latent_policy(latent_state)
        latent_action = tf.cast(latent_action_encoder.sample(), tf.float32)
        try:
            rate = latent_action_encoder.kl_divergence(latent_policy)
        except tf.errors.InvalidArgumentError:
            rate = latent_action_encoder.log_prob(latent_action) - latent_policy.log_prob(latent_action)

        # transition probability reconstruction
        transition_distribution = self.discrete_latent_transition(
            latent_state, tf.math.log(latent_action + epsilon), log_latent_action=True, relaxed_state_encoding=False)
        log_q_encoding_latent_state = next_latent_distribution.log_prob(next_latent_state_no_label)
        log_p_transition = transition_distribution.log_prob(next_label, next_latent_state_no_label)
        rate += log_q_encoding_latent_state - log_p_transition

        next_latent_state = tf.concat([next_label, next_latent_state_no_label], axis=-1)

        reconstruction_distribution = tfd.JointDistributionSequential([
            self.decode_action(latent_state, tf.math.log(latent_action + epsilon), log_latent_action=True),
            self.reward_distribution(
                latent_state, tf.math.log(latent_action + epsilon), next_latent_state, log_latent_action=True),
            self.decode_state(next_latent_state)
        ])

        distortion = -1. * reconstruction_distribution.log_prob(action, reward, next_state)

        return {
            'distortion': distortion,
            'rate': rate,
            'elbo': -1. * (distortion + rate),
            'latent_states': tf.concat([tf.cast(latent_state, tf.int64), tf.cast(next_latent_state, tf.int64)], axis=0),
            'latent_actions': tf.cast(tf.argmax(latent_action, axis=1), tf.int64)
        }

    def mean_latent_bits_used(self, inputs, eps=1e-3, deterministic=True):
        state, label, action, reward, next_state, next_label = inputs[:6]
        latent_state = tf.cast(self.binary_encode_state(state, label).sample(), tf.float32)
        mean = tf.reduce_mean(self.discrete_action_encoding(latent_state, action).probs_parameter(), axis=0)
        check = lambda x: 1 if 1 - eps > x > eps else 0
        mean_bits_used = tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()

        mbu = {'mean_action_bits_used': mean_bits_used}
        mbu.update(self._state_vae.mean_latent_bits_used(inputs, eps, deterministic))
        return mbu

    def get_state_vae(self) -> VariationalMarkovDecisionProcess:
        return self._state_vae

    @property
    def inference_variables(self):
        variables = []
        for network in [self.action_encoder, self.encoder_network]:
            variables += network.trainable_variables
        return variables

    @property
    def generator_variables(self):
        variables = []
        for network in [self.reconstruction_network, self.action_transition_network, self.action_reward_network]:
            variables += network.trainable_variables
        # if self.full_optimization:
        #     variables.append(self.latent_policy_network)
        return variables

    @property
    def action_discretizer_variables(self):
        variables = []
        for network in [
            self.action_encoder,
            self.action_transition_network,
            self.latent_policy_network,
            self.action_reward_network,
            self.action_decoder_network
        ]:
            variables += network.trainable_variables
        return variables

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
        if self.full_optimization:
            return self._compute_apply_gradients(
                state, label, action, reward, next_state, next_label,
                self.trainable_variables, sample_key, sample_probability)
        else:
            return self._compute_apply_gradients(
                state, label, action, reward, next_state, next_label,
                self.action_discretizer_variables, sample_key, sample_probability)

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

        if self.time_stacked_states:
            labeling_function = lambda x: labeling_function(x)[:, -1, ...]
        _labeling_function = dataset_generator.ergodic_batched_labeling_function(labeling_function)

        return estimate_local_losses_from_samples(
            environment=environment,
            latent_policy=self.get_latent_policy(action_dtype=tf.int64),
            steps=steps,
            latent_state_size=self.latent_state_size,
            number_of_discrete_actions=self.number_of_discrete_actions,
            state_embedding_function=self.state_embedding_function,
            action_embedding_function=self.action_embedding_function,
            latent_reward_function=(
                lambda latent_state, latent_action, next_latent_state:
                self.reward_distribution(
                    latent_state=tf.cast(latent_state, dtype=tf.float32),
                    latent_action=tf.cast(latent_action, dtype=tf.float32),
                    next_latent_state=tf.cast(next_latent_state, dtype=tf.float32),
                    disable_mixture_distribution=True).mode()),
            labeling_function=labeling_function,
            latent_transition_function=lambda state, action: TransitionFnDecorator(
                next_state_distribution=self.discrete_latent_transition(
                    latent_state=tf.cast(state, tf.float32),
                    latent_action=tf.math.log(action + epsilon),
                    log_latent_action=True),
                atomic_prop_dims=self.atomic_prop_dims),
            estimate_transition_function_from_samples=estimate_transition_function_from_samples,
            replay_buffer_max_frames=replay_buffer_max_frames,
            reward_scaling=reward_scaling)


def load(tf_model_path: str, full_optimization: bool = False,
         step: Optional[int] = None) -> VariationalActionDiscretizer:
    tf_model = tf.saved_model.load(tf_model_path)
    state_model = tf_model._state_vae
    state_vae = VariationalMarkovDecisionProcess(
        state_shape=tuple(tf_model.signatures['serving_default'].structured_input_signature[1]['state'].shape)[1:],
        label_shape=(tf_model.signatures['serving_default'].structured_input_signature[1]['label'].shape[-1] - 1,),
        action_shape=tuple(tf_model.signatures['serving_default'].structured_input_signature[1]['action'].shape)[1:],
        reward_shape=tuple(tf_model.signatures['serving_default'].structured_input_signature[1]['reward'].shape)[1:],
        encoder_network=state_model.encoder_network,
        transition_network=state_model.transition_network,
        reward_network=state_model.reward_network,
        label_transition_network=state_model.label_transition_network,
        decoder_network=state_model.reconstruction_network,
        latent_state_size=(tf_model.encoder_network.variables[-1].shape[0] +
                           tf_model.signatures['serving_default'].structured_input_signature[1]['label'].shape[-1]),
        encoder_temperature=state_model._encoder_temperature,
        prior_temperature=state_model._prior_temperature,
        mixture_components=tf.shape(state_model.reconstruction_network.variables[-1])[-1],
        evaluation_window_size=tf.shape(tf_model.evaluation_window)[0],
        pre_loaded_model=True)
    model = VariationalActionDiscretizer(
        vae_mdp=state_vae,
        number_of_discrete_actions=tf_model.action_encoder.variables[-1].shape[0],
        action_encoder_network=tf_model.action_encoder,
        action_decoder_network=tf_model.action_decoder_network,
        transition_network=tf_model.action_transition_network,
        reward_network=tf_model.action_reward_network,
        action_label_transition_network=tf_model.action_label_transition_network,
        latent_policy_network=tf_model.latent_policy_network,
        one_output_per_action=tf_model.action_decoder_network.variables[0].shape[0] == state_vae.latent_state_size,
        encoder_temperature=tf_model._encoder_temperature,
        prior_temperature=tf_model._prior_temperature,
        reconstruction_mixture_components=tf_model.action_decoder_network.variables[-1].shape[0],
        pre_loaded_model=True,
        full_optimization=full_optimization)

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
