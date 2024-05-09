import os
from typing import Union, Tuple, Optional, Callable, Dict
import numpy as np
import copy

from tensorflow.python.keras.metrics import Mean
from tf_agents.typing.types import Float, Int

from belief_learner.networks.lipschitz_functions import ObservationLipschitzFunction
from belief_learner.networks.nn import generate_sequential_model
from belief_learner.utils import get_logger, diff_clip
from belief_learner.utils.costs import get_cost_fn
from belief_learner.utils.decorators import log_usage
from belief_learner.wae.wae_gan import WaeGan
from belief_learner.networks.decoders import StateObservationNormalDecoderN, ObservationNormalDecoder
from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.wae.mdp.wasserstein_mdp import WassersteinMarkovDecisionProcess

import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb
import tensorflow.keras as tfk

logger = get_logger(__name__)


class WassersteinBeliefMDP(WassersteinMarkovDecisionProcess):
    @log_usage
    def __init__(
            self,
            state_shape: Tuple[int, ...],
            observation_shape: Tuple[int, ...],
            latent_embedding_network: ModelArchitecture,
            state_encoder_fc_arch: ModelArchitecture,
            state_encoder_cnn_arch: Optional[ModelArchitecture],
            state_decoder_fc_arch: ModelArchitecture,
            state_decoder_tcnn_arch: Optional[ModelArchitecture],
            observation_encoder_fc_arch: ModelArchitecture,
            observation_encoder_cnn_arch: ModelArchitecture,
            observation_decoder_fc_arch: ModelArchitecture,
            observation_decoder_tcnn_arch: ModelArchitecture,
            latent_deembedding_state_arch: ModelArchitecture,
            latent_deembedding_observation_arch: ModelArchitecture,
            transition_loss_lipschitz_network: ModelArchitecture,
            latent_state_size: int,
            latent_belief_sampler: Optional[Callable[[int], tfd.Distribution]] = None,
            use_wae_gan: bool = True,
            emb_observation_size: int = 8,
            emb_state_size: int = 8,
            wae_gan_regularizer_scale_factor: float = 1.,
            wae_gan_discriminator_arch: ModelArchitecture = ModelArchitecture(
                # base architecture for mnist in WAEs (see WAE paper)
                hidden_units=(512, 512, 512, 512),
                activation='relu',
                name='latent_discriminator'
            ),
            wae_gan_stop_grad: bool = True,
            clip_by_global_norm: Optional[float] = None,
            wae_gan_clip_by_global_norm: Optional[float] = None,
            observation_wae_minimizer_lr: Optional[float] = None,
            observation_wae_maximizer_lr: Optional[float] = None,
            wae_gan_target_update_freq: Optional[Union[int, float]] = None,
            wae_gan_fixed_checkpoint_path: Optional[str] = None,
            cost_fn: str = 'l22',
            cost_fn_obs: Optional[Callable] = None,
            cost_fn_state: Optional[Callable] = None,
            cost_weights: Optional[Dict[str, float]] = None,
            recover_fn_obs: Optional[Callable] = None,
            distillation: bool = True,
            observation_regularizer: bool = False,
            optimizer_name: str = 'Adam',
            weight_decay: Optional[float] = 0.,
            optimizer_epsilon: Optional[float] = 1e-7,
            amsgrad: bool = False,
            *args,
            **kwargs
    ):
        _params = list(locals().items())
        _summary = kwargs.pop('summary', True)
        self._use_wae_gan = use_wae_gan
        self.wae_gan_target_update_freq = wae_gan_target_update_freq
        _norm_weights = [1., 1.]
        if cost_fn_obs is None:
            cost_fn_obs = get_cost_fn(cost_fn)
            _norm_weights[1] = 1. / np.prod(observation_shape)
        if cost_fn_state is None:
            cost_fn_state = get_cost_fn(cost_fn_state)
            _norm_weights[0] = 1. / np.prod(state_shape)

        if weight_decay == 0.:
            weight_decay = None

        original_observation_shape = observation_shape
        self._observation_wae: Optional[WaeGan] = None
        self._gan_target_last_update = None
        self._learn_gan = False
        self._wae_gan_target_update = 0
        if use_wae_gan:
            self._learn_gan = True
            self._observation_wae = WaeGan(
                encoder_fc_arch=observation_encoder_fc_arch,
                decoder_fc_arch=observation_decoder_fc_arch,
                latent_discriminator=wae_gan_discriminator_arch,
                input_shape=original_observation_shape,
                latent_space_size=emb_observation_size,
                encoder_cnn_arch=observation_encoder_cnn_arch,
                decoder_tcnn_arch=observation_decoder_tcnn_arch,
                latent_regularizer_scale_factor=wae_gan_regularizer_scale_factor,
                maximizer_lr=observation_wae_minimizer_lr,
                minimizer_lr=observation_wae_minimizer_lr,
                clip_by_global_norm=wae_gan_clip_by_global_norm,
                cost_fn=cost_fn_obs,
            )
            observation_shape = (emb_observation_size,)
            if wae_gan_fixed_checkpoint_path:
                self._observation_wae.load(wae_gan_fixed_checkpoint_path)
                self._learn_gan = False
            if self.wae_gan_target_update_freq:
                self._observation_wae_target = WaeGan(
                    encoder_fc_arch=observation_encoder_fc_arch,
                    decoder_fc_arch=observation_decoder_fc_arch,
                    latent_discriminator=wae_gan_discriminator_arch,
                    input_shape=original_observation_shape,
                    latent_space_size=emb_observation_size,
                    encoder_cnn_arch=observation_encoder_cnn_arch,
                    decoder_tcnn_arch=observation_decoder_tcnn_arch,
                    latent_regularizer_scale_factor=wae_gan_regularizer_scale_factor,
                    maximizer_lr=None,
                    minimizer_lr=None,
                    clip_by_global_norm=wae_gan_clip_by_global_norm,
                    cost_fn=cost_fn_obs,
                )
                self.update_wae_gan_target(0, 1, force=True)
            cost_fn_obs = get_cost_fn(cost_fn)
            _norm_weights[1] = 1 / np.prod(observation_shape)
            recover_fn_obs = tfb.Identity()

        if use_wae_gan:
            obs_encoder_pre_processing_network_ = tuple()
        elif observation_encoder_cnn_arch is not None:
            obs_encoder_pre_processing_network_ = (observation_encoder_cnn_arch,
                                                   observation_encoder_fc_arch)
        else:
            obs_encoder_pre_processing_network_ = (observation_encoder_fc_arch,)

        # # WARNING: Quick fix to try something -- REMOVE !
        # observation_encoder_fc_arch = observation_encoder_fc_arch.replace(
        #     raw_last=False,
        #     batch_norm=state_encoder_fc_arch.batch_norm,
        #     hidden_units=state_encoder_fc_arch.hidden_units
        # )
        # obs_encoder_pre_processing_network_ = (observation_encoder_fc_arch,)

        if state_encoder_cnn_arch is not None:
            state_encoder_pre_processing_network = (state_encoder_cnn_arch, state_encoder_fc_arch)
        else:
            state_encoder_pre_processing_network = (state_encoder_fc_arch,)

        state_encoder_pre_processing_network = [
            state_encoder_pre_processing_network,
            obs_encoder_pre_processing_network_,
        ]

        def cost_fn_combined_state(x, y, weights):
            g = zip([cost_fn_state, cost_fn_obs], tf.nest.flatten(x), tf.nest.flatten(y),
                    tf.nest.flatten(weights))
            cost = sum([f(x_, y_) * w for f, x_, y_, w in g])
            return cost

        logger.warning(f"cost_weights was set to {cost_weights}")
        if cost_weights is None:
            cost_weights = dict({'state': _norm_weights})
        else:
            a, b = cost_weights.get('state', [1., 1.])
            a *= _norm_weights[0]
            b *= _norm_weights[1]
            cost_weights['state'] = [a, b]
        logger.warning(f"cost_weights now set to {cost_weights} to account for dimensionality difference.")

        kwargs.pop('decoder_network', None)
        super(WassersteinBeliefMDP, self).__init__(
            state_shape=[state_shape, observation_shape],
            state_encoder_network=latent_embedding_network,
            latent_state_size=latent_state_size,
            state_encoder_pre_processing_network=state_encoder_pre_processing_network,
            decoder_network=None,
            summary=False,
            clip_by_global_norm=clip_by_global_norm,
            cost_fn=cost_fn,
            cost_fn_state=cost_fn_combined_state,
            cost_weights=cost_weights,
            recover_fn_obs=recover_fn_obs,
            distillation=distillation,
            transition_loss_lipschitz_network=transition_loss_lipschitz_network,
            *args,
            **kwargs
        )
        self._params = {key: str(value) for key, value in _params}

        self.reconstruction_network = StateObservationNormalDecoderN(
            latent_state_size=latent_state_size,
            observation_decoder_fc_arch=None if use_wae_gan else observation_decoder_fc_arch,
            # observation_decoder_fc_arch=observation_decoder_fc_arch.replace(raw_last=True, batch_norm=state_decoder_fc_arch.batch_norm), ## QUICK FIX !!! None if use_wae_gan else observation_decoder_fc_arch,
            observation_decoder_tcnn_arch=None if use_wae_gan else observation_decoder_tcnn_arch,
            latent_deembedding_observation_arch=latent_deembedding_observation_arch,
            # latent_deembedding_observation_arch=latent_deembedding_observation_arch.replace(raw_last=False),  ## quick fix,,
            latent_deembedding_state_arch=latent_deembedding_state_arch,
            state_decoder_tcnn_arch=state_decoder_tcnn_arch,
            state_decoder_fc_arch=state_decoder_fc_arch,
            state_shape=state_shape,
            observation_shape=observation_shape,
            random_decoder=False,
            atomic_prop_dims=self.atomic_prop_dims,
            emb_observation_size=emb_observation_size,
            emb_state_size=emb_state_size,
            name="state_observation_decoder")

        logger.debug(f"Reconstruction Network --- {self.reconstruction_network}")
        self._random_decoder = False

        self.observation_regularizer = observation_regularizer
        if self.observation_regularizer:
            self._next_state_input = []
            for _input in self._state_input:
                next_input = tfk.Input(shape=_input.shape[1:], name="next_" + _input.name)
                self._next_state_input.append(next_input)
            self.observation_lipschitz_net = ObservationLipschitzFunction(
                mdp_state=self._state_input[0],
                observation=self._state_input[1],
                action=self._action_input,
                next_mdp_state=self._next_state_input[0],
                next_observation=self._next_state_input[1],
                lipschitz_arch=transition_loss_lipschitz_network,
                pre_proc_net=self._get_pre_processing_layers(self._state_encoder_pre_processing_network,
                                                             prefix='observation_lip_pre_process'))
            self.observation_decoder_variance_net = ObservationNormalDecoder(
                latent_state_size=latent_state_size,
                observation_decoder_fc_arch=observation_decoder_fc_arch,
                observation_decoder_tcnn_arch=observation_decoder_tcnn_arch,
                latent_deembedding_observation_arch=latent_deembedding_observation_arch,
                observation_shape=observation_shape,
                name="observation_variance_network",
                emb_state_size=emb_state_size,
                emb_observation_size=emb_observation_size)
            if observation_wae_minimizer_lr is not None:
                self._observation_optimizer_min = tfk.optimizers.get(
                    {'class_name': optimizer_name,
                     'config': {'learning_rate': observation_wae_minimizer_lr,
                                # 'weight_decay': weight_decay,
                                'epsilon': optimizer_epsilon,
                                'amsgrad': amsgrad}})
            else:
                self._observation_optimizer_min = None
            if observation_wae_maximizer_lr is not None:
                self._observation_optimizer_max = tfk.optimizers.get(
                    {'class_name': optimizer_name,
                     'config': {'learning_rate': observation_wae_maximizer_lr,
                                # 'weight_decay': weight_decay,
                                'epsilon': optimizer_epsilon,
                                'amsgrad': amsgrad}})
            else:
                self._observation_optimizer_max = None

            # Add loss
            self.loss_metrics["gradients_obs_regularizer_max"] = \
                Mean("gradients_obs_regularizer_max")
            self.loss_metrics["gradients_obs_regularizer_min"] = \
                Mean("gradients_obs_regularizer_min")
            self.loss_metrics["gradients_obs_regularizer_max_grad_norm"] = \
                Mean("gradients_obs_regularizer_max_grad_norm")
            self.loss_metrics["gradients_obs_regularizer_min_grad_norm"] = \
                Mean("gradients_obs_regularizer_min_grad_norm")

        else:
            self.observation_lipschitz_net = self._observation_optimizer_min = \
                self.observation_decoder_variance_net = self._observation_optimizer_max = None

        # self.loss_metrics['variance'] = Mean(name='variance')
        self._latent_belief_sampler = latent_belief_sampler
        self._wae_gan_stop_grad = wae_gan_stop_grad
        if _summary:
            self.summary()
            if self._observation_wae:
                self._observation_wae.summary()

        if self._observation_wae:
            for key in self._observation_wae.loss_metrics:
                self.loss_metrics[f"gan_{key}"] = self._observation_wae.loss_metrics[key]

    def get_obs_variance(self, latent_state: Optional[Float] = None):
        if self.observation_regularizer and latent_state is not None:
            scale = tfb.Softplus()(self.observation_decoder_variance_net(latent_state))
            return tf.square(scale)
        else:
            return self._obs_variance / tf.cast(self._obs_variance_counter, tf.float32)


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
        if self._use_wae_gan is None:
            return super(WassersteinBeliefMDP, self).compute_loss(
                state, label, action, reward, next_state, next_label, sample_key, sample_probability,
                additional_transition_batch, *args, **kwargs, )

        elif self.observation_regularizer:
            loss = super(WassersteinBeliefMDP, self).compute_loss(
                state, label, action, reward, next_state, next_label, sample_key, sample_probability,
                additional_transition_batch, *args, **kwargs, )

            _state, _observation = state
            _next_state, next_obs_env = next_state
            next_latent_state = self.relaxed_state_encoding(
                next_state, temperature=self.encoder_temperature, label=next_label
            ).sample()
            raw_next_latent_observation_mean = self.reconstruction_network.observation_distribution(
                latent_state=next_latent_state
            ).mean()
            next_latent_observation_mean = self._recover_fn_obs(raw_next_latent_observation_mean)
            flat_mean = tf.reshape(next_latent_observation_mean, (tf.shape(next_latent_observation_mean)[0], -1))
            next_latent_observation_scale = tfb.Softplus()(self.observation_decoder_variance_net(next_latent_state))
            flat_scale = tf.reshape(next_latent_observation_scale, (tf.shape(next_latent_observation_scale)[0], -1))
            ## Clip flat scale
            flat_scale = diff_clip(flat_scale, 0., tf.float32.max, 1e-3)

            next_latent_observation = tfd.TransformedDistribution(
                distribution=tfd.MultivariateNormalDiag(
                    loc=flat_mean,
                    scale_diag=flat_scale, ),
                bijector=tfb.Reshape(self.reconstruction_network._observation_shape)
            ).sample()

            x = [_state, _observation, action, _next_state, next_obs_env]
            y = [_state, _observation, action, _next_state, next_latent_observation]
            obs_discrepancy = tf.squeeze(
                self.observation_lipschitz_net(x) -
                self.observation_lipschitz_net(y)
            )
            gradient_penalty = self.compute_gradient_penalty(
                [next_obs_env],
                [next_latent_observation],
                lambda _x: self.observation_lipschitz_net(x[:-1] + [_x]))
            min_loss = self.wasserstein_regularizer_scale_factor.observation_regularizer.scaling * \
                tf.reduce_mean(obs_discrepancy, axis=0)
            max_loss = -1. * tf.reduce_mean(
                obs_discrepancy -
                self.wasserstein_regularizer_scale_factor.observation_regularizer.gradient_penalty_multiplier
                * gradient_penalty,
                axis=0
            ) * self.wasserstein_regularizer_scale_factor.observation_regularizer.scaling

            if self._observation_optimizer_min is None:
                loss['min'] += min_loss
                loss['max'] += max_loss
            else:
                loss['obs_regularizer_min'] = min_loss
                loss['obs_regularizer_max'] = max_loss

            loss['obs_discrepancy'] = tf.reduce_mean(obs_discrepancy, axis=0)
            loss['obs_regularizer_gradient_penalty'] = tf.reduce_mean(gradient_penalty, 0)

            return loss

        else:
            _state, _observation = state
            _next_state, _next_observation = next_state
            latent_observation = tf.stop_gradient(self.encode_observation(_observation))
            next_latent_observation = tf.stop_gradient(self.encode_observation(_next_observation))

            return super(WassersteinBeliefMDP, self).compute_loss(
                [_state, latent_observation], label, action, reward, [_next_state, next_latent_observation],
                next_label, sample_key, sample_probability, additional_transition_batch, *args, **kwargs)

    @tf.function
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

        optimizers = {
            'max': self._wasserstein_regularizer_optimizer,
            'min': self._autoencoder_optimizer,
            'obs_regularizer_max': self._observation_optimizer_max,
            'obs_regularizer_min': self._observation_optimizer_min
        }

        max_variables = wasserstein_regularizer_variables[:]
        min_variables = autoencoder_variables[:]
        obs_regularizer_max_variables = None
        obs_regularizer_min_variables = None
        with tf.GradientTape(persistent=True) as tape:
            loss = self.compute_loss(
                state, label, action, reward, next_state, next_label,
                sample_key=sample_key, sample_probability=sample_probability, )

        if self.observation_lipschitz_net is not None:
            if optimizers['obs_regularizer_max'] is None:
                max_variables = max_variables + self.observation_lipschitz_net.trainable_variables
            else:
                obs_regularizer_max_variables = self.observation_lipschitz_net.trainable_variables
        if self.observation_decoder_variance_net is not None:
            if optimizers['obs_regularizer_min'] is None:
                min_variables = min_variables + self.observation_decoder_variance_net.trainable_variables
            else:
                obs_regularizer_min_variables = self.observation_decoder_variance_net.trainable_variables

        for optimization_direction, variables in {
            'max': max_variables,
            'min': min_variables,
            'obs_regularizer_max': obs_regularizer_max_variables,
            'obs_regularizer_min': obs_regularizer_min_variables,
        }.items():
            if (
                    variables is not None and
                    ('max' in optimization_direction or
                     ((step + 1) % self.n_critic == 0 and 'min' in optimization_direction))
            ):
                gradients = tape.gradient(loss[optimization_direction], variables)
                optimizer = optimizers[optimization_direction]

                if self.clip_by_global_norm:
                    gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_by_global_norm)
                else:
                    grad_norm = tf.linalg.global_norm(gradients)

                if not numerical_error(gradients, list_of_tensors=True):
                    if optimizer is not None:
                        optimizer.apply_gradients(zip(gradients, variables))
                else:
                    tf.print(
                        "[Warning]",
                        "numerical error for some variable in [",
                        f"WAE BMDP - {optimization_direction}"
                        # *(variable.name for variable in variables), "]"
                    )

                if 'gradients_' + optimization_direction in self.loss_metrics.keys():
                    mean_abs_grads = tf.concat(
                        [tf.reshape(tf.abs(grad), [-1]) for grad in gradients
                         if grad is not None],
                        axis=-1)
                    self.loss_metrics['gradients_' + optimization_direction](mean_abs_grads)

                if 'gradients_' + optimization_direction + '_grad_norm' in self.loss_metrics.keys():
                    self.loss_metrics['gradients_' + optimization_direction + '_grad_norm'](grad_norm)

        del tape

        return loss

    def encode_observation(self, observation):  
        if self._use_wae_gan and self.wae_gan_target_update_freq:
            return self._observation_wae_target.encoder(observation)
        elif self._use_wae_gan:
            return self._observation_wae.encoder(observation)
        else:
            return observation

    def set_evaluation_dataset(self, inputs):
        if self._use_wae_gan:
            self._observation_wae.set_evaluation_dataset(inputs)

    def decode_observation(self, latent_observation):
        if self._use_wae_gan and self.wae_gan_target_update_freq:
            return self._observation_wae_target.decoder(latent_observation)
        elif self._use_wae_gan:
            return self._observation_wae.decoder(latent_observation)
        else:
            return latent_observation

    def update_wae_gan_target(self, train_call: int, tau: Optional[Union[int, float]] = None, force: bool = False):
        if self._learn_gan and (self._use_wae_gan and self.wae_gan_target_update_freq is not None):
            if tau is None:
                tau = self.wae_gan_target_update_freq
            if force or self._gan_target_last_update is None or (train_call - self._gan_target_last_update >= tau):
                self._wae_gan_target_update += 1
                self._observation_wae_target.set_weights(self._observation_wae.get_weights(), tau)
                self._gan_target_last_update = train_call

    def evaluate_gan(self):
        if self._use_wae_gan:
            if self.wae_gan_target_update_freq is None:
                return self._observation_wae.evaluate()
            else:
                evaluation_dict = self._observation_wae.evaluate()
                evaluation_dataset = self._observation_wae._evaluation_dataset
                target_encoding = self.encode_observation(evaluation_dataset)
                current_encoding = self._observation_wae.encoder(evaluation_dataset)
                target_eval = self._observation_wae.compare(target_encoding, current_encoding)
                target_eval = {"target_" + key: value for key, value in target_eval.items()}
                return {**evaluation_dict, **target_eval}
        return {}

    def set_latent_belief_sampler(self, latent_belief_sampler: Callable[[int], tfd.Distribution]):
        self._latent_belief_sampler = latent_belief_sampler

    def _joint_relaxed_steady_state_distribution(self, batch_size: int) -> tfd.Distribution:
        if self._latent_belief_sampler is not None:
            return self._latent_belief_sampler(batch_size)
        else:
            return super(WassersteinBeliefMDP, self)._joint_relaxed_steady_state_distribution(batch_size)

    def load(self, path):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.restore(os.path.join(path, 'ckpt-1'))

    def summary(self):
        super().summary()
        if self._observation_wae:
            self._observation_wae.summary()

    @property
    def gan_target_update(self):
        return self._wae_gan_target_update
