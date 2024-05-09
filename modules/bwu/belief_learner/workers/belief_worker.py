from functools import partial
from typing import Tuple, Callable, Optional, Dict

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from keras.optimizer_v2.learning_rate_schedule import PolynomialDecay
from tensorflow_probability.python.distributions import Distribution
from tf_agents.typing.types import Float, Int, Bool
from tensorflow.keras import layers as tfkl
from belief_learner import ReplayBuffer, ACTION, SUB_BELIEF, NEXT_OBS
from belief_learner.networks.get_model import get_model
from belief_learner.networks.lipschitz_functions import TransitionLossLipschitzFunction
from belief_learner.utils.definitions import RAW_INDEXES, STATE, OBS, IS_RESET_STATE
from belief_learner.utils import merge_first_dims, diff_clip, unmerge_first_dims
from belief_learner.utils.costs import get_cost_fn
from belief_learner.networks.tools import get_activation_fn
from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.networks.fully_connected import _fc_network
from belief_learner.wae.mdp.wasserstein_mdp import WassersteinMarkovDecisionProcess
from belief_learner.workers.worker import Worker
from belief_learner.networks.autoregressive_bernoulli import AutoRegressiveBernoulliNetwork
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb


from belief_learner.utils import get_logger

logger = get_logger(__name__)


class BeliefWorker(Worker):
    __name__ = "Belief Worker"

    def __init__(self,
                 replay_buffer: ReplayBuffer,
                 n_next_state_samples: int,
                 n_state_samples: int,
                 batch_size: int,
                 belief_shape: int,
                 sub_belief_shape: Tuple[int, ...],
                 sub_model_architecture: ModelArchitecture,
                 made_architecture: ModelArchitecture,
                 observation_encoder: Callable[[tf.Tensor], tf.Tensor],
                 observation_encoded_shape: Tuple[int, ...],
                 action_encoded_shape: Tuple[int, ...],
                 latent_transition: Callable[[Float, Float], Distribution],
                 obs_filter: Callable[[Float, Float], Distribution],
                 sub_belief_prior_temperature: float,
                 optimizer: str,
                 clip_by_global_norm: Optional[float],
                 learning_rate: float,
                 latent_state_size: int,
                 state_observation_embedding_fn: Callable[[Float, Float, Float], Float],
                 latent_reward: Callable[[Float, Float, Float], Distribution],
                 transition_lipschitz_net: keras.Model,
                 action_encoder: Callable[[tf.Tensor], tf.Tensor] = tfb.Identity(),
                 filter_variance: Float = 1.,
                 filter_variance_target: Float = 1e-4,
                 filter_variance_decay_steps: int = int(1e6),
                 filter_variance_power_decay: float = 1.5,
                 normalize_log_obs_filter: bool = False,
                 use_normalizing_term: bool = False,
                 use_gru: bool = False,
                 latent_state_to_obs_: Optional[Callable] = None,
                 cost_fn_obs: Optional[Callable] = None,
                 use_running_variance: bool = False,
                 use_learned_variance: bool = False,
                 get_running_variance: Optional[Callable[[Float], Float]] = None,
                 n_critic: int = 5,
                 reward_loss_scale_factor: float = 1.,
                 transition_loss_scale_factor: float = 1.,
                 gradient_penalty_scale_factor: float = 10.,
                 maximizer_lr: float = 3e-4,
                 maximizer_batch_size: int = 128,
                 observation_encoder_cnn_arch: Optional[ModelArchitecture] = None,
                 weight_decay=None,
                 optimizer_epsilon=1e-7,
                 amsgrad=False,
                 **kwargs,
                 ):
        self.use_gru = use_gru
        self.use_normalizing_term = use_normalizing_term
        self.normalize_log_obs_filter = normalize_log_obs_filter
        if normalize_log_obs_filter:
            logger.warning("Normalize log obs filter is set to true !")
        self.latent_state_size = latent_state_size
        self.clip_by_global_norm = clip_by_global_norm
        self.obs_filter = obs_filter
        self.n_state_samples = n_state_samples
        self.n_next_state_samples = n_next_state_samples
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.latent_transition = latent_transition
        # setting up variance for the observation filter
        self.use_running_variance = use_running_variance and not use_learned_variance
        self.use_learned_variance = use_learned_variance
        self.get_running_variance = get_running_variance
        self.filter_variance_target = filter_variance_target

        self.get_obs_variance_and_update = lambda: self._variance(next(self._variance_step_it))
        if self.use_running_variance:
            self.obs_variance = tf.Variable(self.get_running_variance())
        else:
            self.obs_variance = tf.Variable(filter_variance)
            self._variance = PolynomialDecay(
                filter_variance,
                decay_steps=filter_variance_decay_steps,
                end_learning_rate=filter_variance,
                power=filter_variance_power_decay)
            self._variance_decay_step = tf.Variable(1, trainable=False, dtype=tf.int64, )

            def _next_step():
                while True:
                    yield self._variance_decay_step
                    self._variance_decay_step.assign_add(1)

            self._variance_step_it = iter(_next_step())
        self.belief_shape = belief_shape

        if sub_model_architecture.output_dim is None:
            sub_model_architecture = sub_model_architecture._replace(output_dim=sub_belief_shape)
        assert sub_model_architecture.output_dim == sub_belief_shape

        self.sub_model_architecture = sub_model_architecture
        self.sub_belief_shape = sub_belief_shape

        self.action_encoded_shape = action_encoded_shape
        # self.action_encoder = action_encoder
        self.action_encoder = partial(tf.one_hot, depth=action_encoded_shape[0])

        self.observation_encoder = observation_encoder
        self.observation_encoded_shape = observation_encoded_shape
        self.observation_encoder_cnn_arch = observation_encoder_cnn_arch

        self.sub_belief_encoder = self._build_sub_belief_encoder()

        self.made_architecture = made_architecture
        self.sub_belief_prior_temperature = sub_belief_prior_temperature
        self._made = self._build_made()
        self.latent_state_to_obs_ = latent_state_to_obs_

        assert optimizer.lower() in ['adam', 'adamw']
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.get(
                {'class_name': optimizer,
                 'config': {'learning_rate': self.learning_rate,
                            # 'weight_decay': weight_decay,
                            'epsilon': optimizer_epsilon,
                            'amsgrad': amsgrad}})
        self._n_train_calls = 0
        self.cost_fn_obs = cost_fn_obs
        self.state_observation_embedding_fn = state_observation_embedding_fn
        self.latent_reward = latent_reward

        self.lipchitz_transition_fn = transition_lipschitz_net

        self.n_critic = n_critic
        self.reward_loss_scale_factor = reward_loss_scale_factor
        self.transition_loss_scale_factor = transition_loss_scale_factor
        self.gradient_penalty_scale_factor = gradient_penalty_scale_factor
        self.maximizer_lr = maximizer_lr
        self.maximizer_optimizer = tf.keras.optimizers.get(
                {'class_name': optimizer,
                 'config': {'learning_rate': maximizer_lr,
                            'epsilon': optimizer_epsilon}})
        self.maximizer_batch_size = maximizer_batch_size
        self.maximizer_dataset_iter = None


    def sub_belief_encode(self,
                          obs: Float,
                          prev_action: Int,
                          prev_sub_belief: Float,
                          first_timestep: Bool,
                          check: bool = True,
                          training: bool = False,
                          ):
        if check:
            if any(first_timestep):
                assert np.all(prev_sub_belief[first_timestep] == 0.), prev_sub_belief[first_timestep]
        if self.use_gru:
            obs_action_model, gru = self.sub_belief_encoder
            embedding = obs_action_model([self.observation_encoder(obs), self.action_encoder(prev_action)],
                                         training=training)
            prev_sub_belief = tf.convert_to_tensor(prev_sub_belief)
            sub_belief_seq, last_sub_belief = gru(embedding[None], prev_sub_belief, training=training)
        else:
            last_sub_belief = self.sub_belief_encoder([self.observation_encoder(obs),
                                                       self.action_encoder(prev_action),
                                                       tf.stop_gradient(prev_sub_belief), ],
                                                      training=training, )
        return last_sub_belief

    @tf.function
    def sub_belief_encode_seq(self,
                              obs: Float,
                              prev_action: Int,
                              prev_sub_belief: Float,
                              target: bool = False,
                              training: bool = False,
                              ):
        if target:
            raise NotImplementedError()
        is_seq = len(prev_action.shape) == 2
        assert is_seq
        T, B = prev_action.shape[:2]
        if self.use_gru:
            obs_action_model, gru = self.sub_belief_encoder
            embeddings = obs_action_model([
                self.observation_encoder(merge_first_dims(obs, 2)),
                self.action_encoder(merge_first_dims(prev_action, 2))
            ], training=training)
            embeddings = unmerge_first_dims(embeddings, (T, B))
            prev_sub_belief = tf.convert_to_tensor(prev_sub_belief)
            sub_belief_seq, last_sub_belief = gru(embeddings, prev_sub_belief, training=training)
        else:
            sub_belief_seq = []
            for t in range(T):
                prev_sub_belief = self.sub_belief_encoder([self.observation_encoder(obs[t]),
                                                           self.action_encoder(prev_action[t]),
                                                           tf.stop_gradient(prev_sub_belief)],
                                                          training=training)
                sub_belief_seq.append(prev_sub_belief)
            sub_belief_seq = tf.stack(sub_belief_seq, axis=0)
        return sub_belief_seq

    def _build_sub_belief_encoder(self):
        sub_observation_input = keras.Input(self.observation_encoded_shape, name='latent_observation')
        sub_action_input = keras.Input(self.action_encoded_shape, name='latent_action')
        inputs = [sub_observation_input, sub_action_input]
        if not self.use_gru:
            sub_belief_input = keras.Input(self.sub_belief_shape, name='latent_belief')
            inputs.append(sub_belief_input)
        ## Observation CNN
        to_concat = inputs[:]
        if self.observation_encoder_cnn_arch is not None:
            observation_encoder_cnn = get_model(self.observation_encoder_cnn_arch)
            encoded_observation = observation_encoder_cnn(sub_observation_input)
            to_concat[0] = encoded_observation
        concat_input = keras.layers.Concatenate()(to_concat)
        if not self.use_gru and not self.sub_model_architecture.raw_last:
            logger.warning("Sub model architecture raw last is supposed to be true because we add a tanh !")
        elif self.use_gru and self.sub_model_architecture.raw_last:
            logger.warning(
                "Sub model architecture raw last is supposed to be false with gru because there is a gru after !")
        output = _fc_network(
            input_=concat_input,
            hidden_units=self.sub_model_architecture.hidden_units,
            activation=self.sub_model_architecture.activation,
            output_dim=self.sub_model_architecture.output_dim,
            batch_norm=self.sub_model_architecture.batch_norm,
            raw_last=True,
        )
        if not self.use_gru:
            output = tfkl.Activation(activation='tanh')(output)
        name = 'sub_belief_encoder'
        if self.use_gru:
            name += '_obs_action'
        model = keras.Model(inputs=inputs, outputs=output, name=name)
        if not self.use_gru:
            return model
        gru = tfkl.GRU(units=np.prod(self.sub_belief_shape), return_sequences=True, return_state=True, time_major=True)
        return model, gru

    def _build_made(self):
        output_scale = 7.  
        made = AutoRegressiveBernoulliNetwork(
            event_shape=(self.latent_state_size,),
            activation=get_activation_fn(self.made_architecture.activation),
            hidden_units=self.made_architecture.hidden_units,
            conditional_event_shape=self.sub_belief_shape,
            temperature=self.sub_belief_prior_temperature,
            output_softclip=tfb.Chain([tfb.Scale(output_scale), tfb.Tanh(), tfb.Scale(1. / output_scale)]),
            name='autoregressive_belief_network')
        return made

    def _get_weights(self):
        if not self.use_gru:
            return self._made.trainable_variables + self.sub_belief_encoder.trainable_variables
        return self._made.trainable_variables + self.sub_belief_encoder[0].trainable_variables + \
               self.sub_belief_encoder[1].trainable_variables

    @tf.function
    def compute_loss_and_apply_gradients(self, next_obs, action, sub_belief):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    @property
    def n_train_called(self):
        return self._n_train_calls

    def get_sub_belief_weights(self):
        if self.use_gru:
            return self.sub_belief_encoder[0].trainable_variables + self.sub_belief_encoder[1].trainable_variables
        return self.sub_belief_encoder.trainable_variables

    @tf.function
    def compute_seq_loss_and_apply_gradients(self, data):
        with tf.GradientTape() as tape:
            sub_beliefs = self.sub_belief_encode_seq(data[NEXT_OBS], data[ACTION], data[SUB_BELIEF][0],
                                                     training=True)
            loss, (log_p_next_states, log_expectation_to_next_state, log_obs_filter, normalizing_term, obs_dist,
                   obs_variance) = \
                self.compute_seq_loss(
                    prev_sub_belief=data[SUB_BELIEF][0],
                    sub_belief=sub_beliefs,
                    action=data[ACTION][:-1],
                    next_obs=data[NEXT_OBS][:-1],
                    mask=data["mask"][1:],
                )
        return_dict = {
            "belief_loss": loss,
            "log_p_next_states": log_p_next_states,
            "log_expectation_to_next_state": log_expectation_to_next_state,
            "log_obs_filter": log_obs_filter,
            "obs_dist": obs_dist,
            "obs_filter_variance": obs_variance,
        }
        if normalizing_term is not None:
            return_dict["normalizing_term"] = normalizing_term

        grads = tape.gradient(loss, self._get_weights())
        if self.clip_by_global_norm:
            grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_by_global_norm)
        else:
            grad_norm = tf.linalg.global_norm(grads)
        if tf.reduce_any([
            tf.reduce_any(tf.math.logical_or(tf.math.is_nan(grad), tf.math.is_inf(grad))) for grad in grads if
            grad is not None]
        ):
            tf.print("Belief worker got nan in Belief loss.")
        else:
            # tf.print("Belief worker: applying gradient.")
            self.optimizer.apply_gradients(zip(grads, self._get_weights()))
        return_dict["belief_grad_norm"] = grad_norm
        return return_dict

    @tf.function
    def compute_seq_loss(self, prev_sub_belief, sub_belief, action, next_obs, mask):

        T, B = action.shape[:2]
        action_encoded = self.action_encoder(action)
        action_encoded = merge_first_dims(action_encoded, 2)
        sb = tf.concat([prev_sub_belief[None], sub_belief[:-1]], axis=0)
        belief = self._made.relaxed_distribution(conditional_input=merge_first_dims(sb[:-1], 2), training=True)
        next_belief = self._made.relaxed_distribution(conditional_input=merge_first_dims(sb[1:], 2), training=True)
        next_latent_state = next_belief.sample((self.n_next_state_samples,))
        next_latent_state = diff_clip(next_latent_state, 0, 1, 1e-5)
        log_p_next_states = next_belief.log_prob(next_latent_state)
        log_p_next_states = tf.reshape(log_p_next_states, (self.n_next_state_samples, T, B))
        if tf.reduce_any(tf.math.logical_or(tf.math.is_inf(log_p_next_states), tf.math.is_nan(log_p_next_states))):
            tf.print('LOG P(NEXT_STATE)', log_p_next_states, summarize=-1)
            tf.print('NEXT_STATE', next_latent_state, summarize=-1)
            tf.print(
                'INF OR NAN IN NEXT STATE?',
                tf.reduce_any(tf.math.logical_or(tf.math.is_inf(next_latent_state), tf.math.is_nan(next_latent_state))))
        latent_state = belief.sample(self.n_state_samples)
        action_encoded_tiled = tf.tile(action_encoded, [self.n_state_samples, 1])
        latent_next_state_dist = self.latent_transition(tf.stop_gradient(merge_first_dims(latent_state, 2)),
                                                        tf.stop_gradient(action_encoded_tiled))
        next_latent_state_tiled = tf.tile(next_latent_state, [1, self.n_state_samples, 1])
        log_prob_to_next_state = latent_next_state_dist.log_prob(next_latent_state_tiled)
        log_prob_to_next_state = tf.reshape(log_prob_to_next_state,
                                            (self.n_state_samples, self.n_next_state_samples, T * B))
        log_expectation_to_next_state = tf.reduce_logsumexp(
            log_prob_to_next_state - tf.math.log(tf.cast(self.n_state_samples, tf.float32)), 0)
        log_expectation_to_next_state = tf.reshape(log_expectation_to_next_state,
                                                   (self.n_next_state_samples, T, B))

        if tf.reduce_any(tf.math.logical_or(tf.math.is_inf(log_expectation_to_next_state),
                                            tf.math.is_nan(log_expectation_to_next_state))):
            # DEBUG
            tf.print('LOG EXPECTATION TO NEXT STATE)', log_expectation_to_next_state, summarize=-1)
            tf.print('LOG PROB TO NEXT STATE', log_prob_to_next_state, summarize=-1)
            tf.print('NEXT_STATE', next_latent_state, summarize=-1)

        if self.use_learned_variance:
            obs_variance = self.get_running_variance(merge_first_dims(next_latent_state, 2))
        else:
            obs_variance = self.obs_variance.value()
        obs_variance = tf.maximum(obs_variance, self.filter_variance_target * tf.ones_like(obs_variance))
        obs_variance = tf.stop_gradient(obs_variance)
        next_obs = tf.cast(next_obs, tf.float32)
        next_obs_encoded = self.observation_encoder(merge_first_dims(next_obs, 2))
        next_obs_encoded = unmerge_first_dims(next_obs_encoded, (T, B))
        next_obs_encoded_tiled = merge_first_dims(tf.repeat(next_obs_encoded[None], self.n_next_state_samples, 0), 3)
        next_obs_encoded_tiled_r = tf.reshape(next_obs_encoded_tiled, (self.n_next_state_samples * T * B, -1))
        obs_variance = tf.reshape(obs_variance, (self.n_next_state_samples * T * B, -1))

        log_obs_filter = self.obs_filter(next_obs_encoded_tiled_r, obs_variance).log_prob(
            tf.reshape(next_latent_state, (self.n_next_state_samples * T * B, -1))
        )

        log_obs_filter = tf.reshape(log_obs_filter, (self.n_next_state_samples, T, B))
        if self.normalize_log_obs_filter:
            log_obs_filter /= tf.cast(tf.reduce_prod(next_obs.shape[2:]), tf.float32)

        if self.use_normalizing_term:
            latent_state_ = belief.sample((self.n_state_samples,))
            latent_state_ = tf.stop_gradient(latent_state_)
            latent_state_ = merge_first_dims(latent_state_, 2)
            action_encoded_tiled_ = tf.tile(action_encoded, [self.n_state_samples, 1])
            next_latent_state_ = self.latent_transition(latent_state_, action_encoded_tiled_).sample()

            next_obs_encoded_tiled_ = merge_first_dims(tf.repeat(next_obs_encoded[None], self.n_state_samples, 0), 3)
            if self.use_learned_variance:
                _obs_variance = self.get_running_variance(next_latent_state_)
                _obs_variance = tf.maximum(_obs_variance, self.filter_variance_target * tf.ones_like(_obs_variance))
            else:
                _obs_variance = obs_variance
            next_obs_encoded_tiled_ = tf.reshape(next_obs_encoded_tiled_, (self.n_state_samples * B * T, -1))
            _obs_variance = tf.reshape(_obs_variance, (self.n_state_samples * B * T, -1))
            log_obs_ = self.obs_filter(next_obs_encoded_tiled_, _obs_variance).log_prob(next_latent_state_)
            log_obs_ = tf.reshape(log_obs_, (self.n_state_samples, T, B))
            normalizing_term = tf.reduce_logsumexp(log_obs_ - tf.math.log(tf.cast(self.n_state_samples, tf.float32)), 0)
            if self.normalize_log_obs_filter:
                normalizing_term /= tf.cast(tf.reduce_prod(next_obs.shape[2:]), tf.float32)

        mask = tf.cast(mask, tf.float32)
        nbr_elements = tf.reduce_sum(mask)
        loss = log_p_next_states - log_expectation_to_next_state - log_obs_filter
        loss = tf.reduce_mean(loss, axis=0)
        log_p_next_states = tf.reduce_mean(log_p_next_states, axis=0)
        log_expectation_to_next_state = tf.reduce_mean(log_expectation_to_next_state, axis=0)
        log_obs_filter = tf.reduce_mean(log_obs_filter, axis=0)
        if self.use_normalizing_term:
            loss = loss + tf.stop_gradient(normalizing_term)
        loss = tf.reduce_sum(loss * mask) / nbr_elements
        log_p_next_states = tf.reduce_sum(log_p_next_states * mask) / nbr_elements
        log_expectation_to_next_state = tf.reduce_sum(log_expectation_to_next_state * mask) / nbr_elements
        log_obs_filter = tf.reduce_sum(log_obs_filter * mask) / nbr_elements
        if self.use_normalizing_term:
            normalizing_term = tf.reduce_sum(normalizing_term * mask) / nbr_elements
        else:
            normalizing_term = None

        cost_fn = self.cost_fn_obs
        if cost_fn is None:
            cost_fn = get_cost_fn('l2')
        obs_dist = cost_fn(
            next_obs_encoded_tiled,
            self.latent_state_to_obs_(merge_first_dims(next_latent_state, 2)),
        )
        obs_dist = tf.reshape(obs_dist, (self.n_next_state_samples, T, B))
        obs_dist = tf.reduce_mean(obs_dist, 0)
        obs_dist = tf.reduce_sum(obs_dist * mask) / nbr_elements

        if self.use_learned_variance:
            obs_variance = tf.reshape(obs_variance, (self.n_next_state_samples, T, B, -1))
            obs_variance = tf.reduce_mean(obs_variance, 0)
            obs_variance = tf.reduce_sum(obs_variance * mask[..., None], axis=(0,1)) / nbr_elements

        return loss, (log_p_next_states, log_expectation_to_next_state, log_obs_filter, normalizing_term, obs_dist,
                      obs_variance)

    @property
    def obs_filter_variance(self):
        return self._variance(step=self._variance_decay_step.numpy()).numpy().item()

    def update_variance(self):
        if self.use_running_variance:
            self.obs_variance.assign(self.get_running_variance())
        else:
            self.obs_variance.assign(self._variance(next(self._variance_step_it)))

    @tf.function
    def compute_reward_transition_gradient_penalty(
            self,
            state: Float,
            observation: Float,
            is_reset_state: Float,
            action: Float,
            sub_belief: Float,
    ) -> Dict[str, Float]:
        observation = self.observation_encoder(observation)

        belief = self._made.relaxed_distribution(conditional_input=sub_belief, training=False)
        latent_state = self.state_observation_embedding_fn(state, observation, is_reset_state)
        believed_latent_state = belief.sample()

        # transition loss
        next_latent_state_1 = self.latent_transition(latent_state, action).sample()
        next_latent_state_2 = self.latent_transition(believed_latent_state, action).sample()

        # Mean Discrepancy between the observed and believed transition function on the Lipschitz network
        x = [state, observation, action, believed_latent_state, next_latent_state_1]
        y = [state, observation, action, believed_latent_state, next_latent_state_2]

        discrepancy = tf.reduce_mean(
            tf.squeeze(self.lipchitz_transition_fn(x) - self.lipchitz_transition_fn(y)),
            0)

        gradient_penalty = WassersteinMarkovDecisionProcess.compute_gradient_penalty(
            [next_latent_state_1],
            [next_latent_state_2],
            lambda _x: self.lipchitz_transition_fn(x[:-1] + [_x]))
        gradient_penalty = tf.reduce_mean(gradient_penalty, 0)

        return {
            'transition_loss': discrepancy,
            'gradient_penalty': gradient_penalty
        }

    @tf.function
    def compute_reward_transition_regularizers(
            self,
            state: Float,
            observation: Float,
            is_reset_state: Float,
            next_is_reset_state: Float,
            action: Float,
            next_state: Float,
            next_observation: Float,
            prev_sub_belief: Float,
            sub_belief: Float,
            mask: Float,
    ) -> Dict[str, Float]:
        # time dimension, batch size
        T, B = action.shape[:2]

        action = self.action_encoder(action)
        # take the time component into account
        action = merge_first_dims(action, 2)
        state = merge_first_dims(state, 2)
        is_reset_state = tf.cast(merge_first_dims(is_reset_state, 2), tf.float32)[..., None]
        next_is_reset_state = tf.cast(merge_first_dims(next_is_reset_state, 2), tf.float32)[..., None]
        next_state = merge_first_dims(next_state, 2)
        observation = merge_first_dims(observation, 2)
        observation_encoded = self.observation_encoder(observation)
        next_observation = merge_first_dims(next_observation, 2)
        next_observation_encoded = self.observation_encoder(next_observation)

        sb = tf.concat([prev_sub_belief[None], sub_belief[:-1]], axis=0)
        belief = self._made.relaxed_distribution(conditional_input=merge_first_dims(sb[:-1], 2), training=True)

        latent_state = self.state_observation_embedding_fn(state, observation_encoded, is_reset_state)
        believed_latent_state = belief.sample()

        # episode masking
        mask = tf.cast(mask, tf.float32)
        nbr_elements = tf.reduce_sum(mask)

        # reward loss
        next_belief = self._made.relaxed_distribution(conditional_input=merge_first_dims(sb[1:], 2), training=True)
        next_latent_state = self.state_observation_embedding_fn(
            next_state, next_observation_encoded, tf.zeros_like(next_is_reset_state))  
        reward = tf.stop_gradient(self.latent_reward(latent_state, action, next_latent_state).sample())
        next_believed_latent_state = next_belief.sample()
        believed_reward = self.latent_reward(believed_latent_state, action, next_believed_latent_state).sample()
        reward_loss = tf.abs(reward - believed_reward)
        reward_loss = unmerge_first_dims(reward_loss, (T, B))
        reward_loss = tf.squeeze(reward_loss)
        reward_loss = tf.reduce_sum(reward_loss * mask) / nbr_elements

        # transition loss
        next_latent_state_1 = self.latent_transition(latent_state, action).sample()
        next_latent_state_2 = self.latent_transition(believed_latent_state, action).sample()

        # Mean Discrepancy between the observed and believed transition function on the Lipschitz network
        x = [state, observation, action, believed_latent_state, next_latent_state_1]
        y = [state, observation, action, believed_latent_state, next_latent_state_2]

        transition_loss = tf.squeeze(self.lipchitz_transition_fn(x) - self.lipchitz_transition_fn(y))
        transition_loss = unmerge_first_dims(transition_loss, (T, B))
        transition_loss = tf.reduce_sum(transition_loss * mask) / nbr_elements

        return {"reward_loss": reward_loss,
                "transition_loss": transition_loss,}

    @tf.function
    def compute_maximizer_loss_and_apply_grads(self):
        gradient_penalty = []
        for _ in range(self.n_critic):
            state, obs, is_reset_state, action, sub_belief = next(self.maximizer_dataset_iter)
            # tf.print("Actions: ", action[:5])
            with tf.GradientTape() as tape:
                loss = self.compute_reward_transition_gradient_penalty(
                    state=state,
                    observation=obs,
                    is_reset_state=is_reset_state,
                    action=action,
                    sub_belief=sub_belief,
                )

                maximizer_loss = -1. * (
                        loss['transition_loss'] - self.gradient_penalty_scale_factor * loss['gradient_penalty'])
            weights = self.lipchitz_transition_fn.trainable_variables
            grads = tape.gradient(maximizer_loss, weights)
            self.maximizer_optimizer.apply_gradients(zip(grads, weights))
            gradient_penalty.append(loss['gradient_penalty'])

        return tf.reduce_mean(tf.stack(gradient_penalty, axis=0))

    def set_maximizer_dataset(self):
        maximizer_dataset = self.replay_buffer.as_belief_dataset(batch_size=self.maximizer_batch_size,
                                                                 worker='belief')
        self.maximizer_dataset_iter = iter(maximizer_dataset)

