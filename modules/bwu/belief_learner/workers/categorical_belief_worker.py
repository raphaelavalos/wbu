import math
from typing import Optional, Dict

import numpy as np

import tensorflow as tf
from tf_agents.typing.types import Float
from belief_learner.utils import merge_first_dims, unmerge_first_dims
from belief_learner.utils.costs import get_cost_fn
from belief_learner.verification import binary_latent_space
from belief_learner.wae.mdp.wasserstein_mdp import WassersteinMarkovDecisionProcess
from belief_learner.workers.belief_worker import BeliefWorker
from tensorflow_probability import distributions as tfd

from belief_learner.utils import get_logger

logger = get_logger(__name__)


class CategoricalBeliefWorker(BeliefWorker):
    __name__ = "Categorical Belief Worker"

    def __init__(
            self,
            cnn_filters: int = 64,
            filters_variation: Optional[str] = None,  # in [increasing, decreasing]
            cnn_kernels: int = 5,
            kernel_variation: Optional[str] = None,  # in [increasing, decreasing]
            down_sampling_op: str = 'strides',  # in [strides, max_pooling, avg_pooling,]
            up_sampling_op: str = 'strides',  # [in strides, repeat]
            filter_aggregation_op: str = 'conv',  # in [conv, global_max_pooling, global_avg_pooling]
            cnn_activation: str = 'relu',
            use_dtv: bool = False,
            dtv_split_depth: int = 13,
            to_binary_straight_through_gradient: bool = False,
            *args, **kwargs,
    ):
        super(CategoricalBeliefWorker, self).__init__(*args, **kwargs)

        del self._made
        self.cnn_params = {
            'filters': cnn_filters,
            'filter_variation': filters_variation.lower() if filters_variation is not None else None,
            'kernels': cnn_kernels,
            'kernel_variation': kernel_variation.lower() if kernel_variation is not None else None,
            'down_sampling_op': down_sampling_op.lower(),
            'filter_aggregation_op': filter_aggregation_op.lower(),
            'activation': cnn_activation.lower(),
            'up_sampling_op': up_sampling_op.lower(),
        }
        sub_belief_cnns = self._build_sub_belief_cnns()
        self.sub_belief_upscaler = sub_belief_cnns['sub_belief_upscaler']
        self.belief_embedding = sub_belief_cnns['belief_embedding']
        self.use_dtv = use_dtv
        self._to_binary_straight_through_gradient = to_binary_straight_through_gradient
        if self.use_dtv:
            self.latent_space = tf.split(
                binary_latent_space(self.latent_state_size, dtype=tf.float32),
                2**(max(0, self.latent_state_size - dtv_split_depth)))

    def _build_sub_belief_cnns(self):
        sub_belief_shape = np.prod(self.sub_belief_shape)
        power = math.floor(math.log2(sub_belief_shape))
        lower_power_of_2 = 2 ** power
        higher_power_of_2 = 2 ** (power + 1)

        if abs(sub_belief_shape - lower_power_of_2) < abs(sub_belief_shape - higher_power_of_2):
            target_shape = int(lower_power_of_2)
        else:
            target_shape = int(higher_power_of_2)

        filters = [
            max(8, self.cnn_params['filters'] // (i + 1)) if self.cnn_params['filter_variation'] == 'decreasing' else
            self.cnn_params['filters'] * (i + 1) if self.cnn_params['filter_variation'] == 'increasing' else
            self.cnn_params['filters']
            for i in range(self.latent_state_size - int(math.log2(target_shape)))
        ]
        kernels = [
            max(3, self.cnn_params['kernels'] - (i * 2)) if self.cnn_params['kernel_variation'] == 'decreasing' else
            self.cnn_params['kernels'] + (i * 2) if self.cnn_params['kernel_variation'] == 'increasing' else
            self.cnn_params['kernels']
            for i in range(self.latent_state_size - int(math.log2(target_shape)))
        ]

        belief_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2 ** self.latent_state_size,)),
            tf.keras.layers.Reshape((2 ** self.latent_state_size, 1)),
        ])
        for i in range(self.latent_state_size - int(math.log2(target_shape))):
            belief_embedding.add(
                tf.keras.layers.Conv1D(
                    kernel_size=kernels[i],
                    filters=filters[i],
                    padding='same',
                    activation=self.cnn_params['activation'],
                    strides=2 if self.cnn_params['down_sampling_op'] == 'strides' else 1,
                    name=f'belief_embedding_conv1d_{i:d}',)
            )
            if self.cnn_params['down_sampling_op'] == 'max_pooling':
                belief_embedding.add(tf.keras.layers.MaxPooling1D())
            elif self.cnn_params['down_sampling_op'] == 'avg_pooling':
                belief_embedding.add(tf.keras.layers.AveragePooling1D())
        if self.cnn_params['filter_aggregation_op'] == 'global_max_pooling':
            belief_embedding.add(tf.keras.layers.Conv1D(kernel_size=1, filters=target_shape, activation=None, name='belief_embedding_pre_max_pooling_conv'))
            belief_embedding.add(tf.keras.layers.GlobalMaxPooling1D())
        elif self.cnn_params['filter_aggregation_op'] == 'global_avg_pooling':
            belief_embedding.add(tf.keras.layers.Conv1D(kernel_size=1, filters=target_shape, activation=None))
            belief_embedding.add(tf.keras.layers.GlobalAveragePooling1D())
        else:
            belief_embedding.add(tf.keras.layers.Conv1D(kernel_size=1, filters=1, activation=None, name='belief_embedding_output'))
        belief_embedding.add(tf.keras.layers.Flatten())

        if target_shape != sub_belief_shape:
            belief_embedding.add(tf.keras.layers.Dense(sub_belief_shape, activation=None))

        filters.reverse()
        kernels.reverse()

        sub_belief_upscaler = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(sub_belief_shape,))],
            name='sub_belief_upscaler')
        if sub_belief_shape != target_shape:
            sub_belief_upscaler.add(
                tf.keras.layers.Dense(
                    target_shape, activation=self.cnn_params['activation'], name="sub_belief_upscaler_input"))
        sub_belief_upscaler.add(tf.keras.layers.Reshape((target_shape, 1)))
        for i in range(self.latent_state_size - int(math.log2(target_shape))):
            sub_belief_upscaler.add(
                tf.keras.layers.Conv1DTranspose(
                    kernel_size=kernels[i],
                    filters=filters[i],
                    padding='same',
                    activation=self.cnn_params['activation'],
                    strides=2 if self.cnn_params['up_sampling_op'] == 'strides' else 1,
                    name=f'sub_belief_upscaler_cnn_transpose_{i:d}')
            )
            if self.cnn_params['up_sampling_op'] == 'repeat':
                sub_belief_upscaler.add(tf.keras.layers.UpSampling1D(name=f'sub_belief_upscaler_up_sampling_{i:d}'))
        sub_belief_upscaler.add(tf.keras.layers.Conv1DTranspose(
            kernel_size=1,
            filters=1,
            padding='same',
            strides=1,
            activation=None,
            name='sub_belief_upscaler_output'))
        sub_belief_upscaler.add(tf.keras.layers.Flatten())
        
        belief_embedding.summary()
        sub_belief_upscaler.summary()

        return {
            'belief_embedding': belief_embedding,
            'sub_belief_upscaler': sub_belief_upscaler,
        }

    def _get_weights(self):
        if not self.use_gru:
            return self.sub_belief_upscaler.trainable_variables + self.sub_belief_encoder.trainable_variables
        return self.sub_belief_upscaler.trainable_variables + self.sub_belief_encoder[0].trainable_variables + \
               self.sub_belief_encoder[1].trainable_variables

    @tf.function
    def diff_discretize(self, relaxed_one_hot: Float):
        """
        Discretize by rounding and attach the gradient of the input to the discretized one-hot vector
        (straight-through gradients).
        """
        max_val = tf.repeat(tf.reduce_max(relaxed_one_hot, axis=-1)[..., None], 2 ** self.latent_state_size, axis=-1)
        one_hot = tf.cast(relaxed_one_hot >= max_val, tf.float32)
        # attach the gradients
        return one_hot + relaxed_one_hot - tf.stop_gradient(relaxed_one_hot)

    @tf.function
    def one_hot_to_bit_vector(self, one_hot, diff: bool = False):
        """
        Transforms a discrete sample encoded in one-hot to a bit vector (base 2) via bitwise operations.
        If diff is set, enforce the differentiation by replacing the bitwise operations by
        iterative construction of the bit vector.
        """
        if diff and not self._to_binary_straight_through_gradient:
            # has gradient
            binary = []
            for _ in range(self.latent_state_size):
                l, r = tf.split(one_hot, 2, axis=-1)
                b = tf.reduce_sum(r, axis=-1)
                one_hot = l * (1 - b[..., None]) + r * b[..., None]
                binary.append(b)
            binary = tf.stack(binary[::-1], axis=-1)
        else:
            # has no gradient but faster
            binary = tf.cast(
                tf.math.mod(
                    tf.bitwise.right_shift(
                        tf.cast(tf.argmax(one_hot, axis=-1), tf.int32)[..., None],
                        tf.range(self.latent_state_size)),
                    2),
                tf.float32)
        if diff and self._to_binary_straight_through_gradient:
            base10 = tf.reduce_sum(
                one_hot * tf.range(2 ** self.latent_state_size, dtype=tf.float32),
                axis=-1
            )[..., None]
            base10 = tf.where(base10 == 0., 1e-9, base10)
            binary = binary * base10 / tf.stop_gradient(base10)
        return binary

    @tf.function
    def compute_seq_loss(self, prev_sub_belief, sub_belief, action, next_obs, mask):
        T, B = action.shape[:2]
        action_encoded = self.action_encoder(action)
        action_encoded = merge_first_dims(action_encoded, 2)
        sb = tf.concat([prev_sub_belief[None], sub_belief[:-1]], axis=0)
        belief_logits = self.sub_belief_upscaler(merge_first_dims(sb[:-1], 2))
        belief = tfd.OneHotCategorical(logits=belief_logits)
        next_belief_logits = self.sub_belief_upscaler(merge_first_dims(sb[1:], 2))
        # Gumbel softmax
        next_belief = tfd.RelaxedOneHotCategorical(
            temperature=self.sub_belief_prior_temperature,
            logits=next_belief_logits)
        next_latent_state_relaxed = next_belief.sample((self.n_next_state_samples,))
        # discretize and attach gradients
        next_latent_state_one_hot = self.diff_discretize(next_latent_state_relaxed)
        # transform the one-hot encoding to a binary encoding (differentiable)
        next_latent_state = self.one_hot_to_bit_vector(next_latent_state_one_hot, diff=True)
        log_p_next_states = tfd.OneHotCategorical(logits=next_belief_logits).log_prob(next_latent_state_one_hot)
        log_p_next_states = tf.reshape(log_p_next_states, (self.n_next_state_samples, T, B))
        latent_state = belief.sample(self.n_state_samples)
        latent_state = self.one_hot_to_bit_vector(latent_state)
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
        # next_obs_tiled_encoded = self.observation_encoder(next_obs_tiled)
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
            latent_state_ = self.one_hot_to_bit_vector(latent_state_)
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
            obs_variance = tf.reduce_sum(obs_variance * mask[..., None], axis=(0, 1)) / nbr_elements

        return loss, (log_p_next_states, log_expectation_to_next_state, log_obs_filter, normalizing_term, obs_dist,
                      obs_variance)

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

        belief_logits = self.sub_belief_upscaler(sub_belief)
        belief = tfd.RelaxedOneHotCategorical(
            temperature=self.sub_belief_prior_temperature,
            logits=belief_logits)
        believed_latent_state_relaxed = belief.sample()
        believed_latent_state = self.diff_discretize(believed_latent_state_relaxed)
        # transform the one-hot encoding to a binary encoding
        believed_latent_state = self.one_hot_to_bit_vector(believed_latent_state, diff=True)
        latent_state = self.state_observation_embedding_fn(state, observation, is_reset_state)

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
    def transition_dtv(self, latent_state_believed_latent_state_action):
        latent_state, believed_latent_state, action = latent_state_believed_latent_state_action
        latent_state = latent_state[None]
        believed_latent_state = believed_latent_state[None]
        action = action[None]

        acc = 0.
        q = self.latent_transition(latent_state, action)
        p = self.latent_transition(believed_latent_state, action)
        for next_latent_states in self.latent_space:
            acc += tf.reduce_sum(
                tf.abs(q.prob(next_latent_states) - p.prob(next_latent_states)),
                axis=0)
        return .5 * acc

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
        belief_logits = self.sub_belief_upscaler(merge_first_dims(sb[:-1], 2))
        belief = tfd.RelaxedOneHotCategorical(
            temperature=self.sub_belief_prior_temperature,
            logits=belief_logits)
        believed_latent_state_relaxed = belief.sample()
        believed_latent_state = self.diff_discretize(believed_latent_state_relaxed)
        # transform the one-hot encoding to a binary encoding
        believed_latent_state = self.one_hot_to_bit_vector(believed_latent_state, diff=True)
        latent_state = self.state_observation_embedding_fn(state, observation_encoded, is_reset_state)

        # episode masking
        mask = tf.cast(mask, tf.float32)
        nbr_elements = tf.reduce_sum(mask)

        # reward loss
        next_belief_logits = self.sub_belief_upscaler(merge_first_dims(sb[1:], 2))
        next_belief = tfd.RelaxedOneHotCategorical(
            temperature=self.sub_belief_prior_temperature,
            logits=next_belief_logits)
        next_believed_latent_state_relaxed = next_belief.sample()
        next_believed_latent_state = self.diff_discretize(next_believed_latent_state_relaxed)
        # transform the one-hot encoding to a binary encoding
        next_believed_latent_state = self.one_hot_to_bit_vector(next_believed_latent_state, diff=True)

        next_latent_state = self.state_observation_embedding_fn(
            next_state, next_observation_encoded,
            tf.zeros_like(next_is_reset_state))  
        reward = tf.stop_gradient(self.latent_reward(latent_state, action, next_latent_state).sample())
        believed_reward = self.latent_reward(believed_latent_state, action, next_believed_latent_state).sample()
        reward_loss = tf.abs(reward - believed_reward)
        reward_loss = unmerge_first_dims(reward_loss, (T, B))
        reward_loss = tf.squeeze(reward_loss)
        reward_loss = tf.reduce_sum(reward_loss * mask) / nbr_elements

        if self.use_dtv:
            transition_loss = tf.map_fn(
                fn=self.transition_dtv,
                elems=(latent_state, believed_latent_state, action),
                dtype=tf.float32,
                parallel_iterations=10)
        else:
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
                "transition_loss": transition_loss, }
