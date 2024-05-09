from typing import Tuple, Callable, Optional, Any

import tensorflow as tf
from tensorflow import keras as keras
from tensorflow_probability.python.distributions import Distribution
from tf_agents.typing.types import Float
import tensorflow_probability.python.bijectors as tfb

from belief_learner import ReplayBuffer, ACTION, SUB_BELIEF, OBS, NEXT_OBS, STATE, NEXT_STATE
from belief_learner.utils.definitions import IS_RESET_STATE
from belief_learner.workers.belief_worker import BeliefWorker
from belief_learner.workers.categorical_belief_worker import CategoricalBeliefWorker
from belief_learner.workers.rl_worker import A2CWorker
from belief_learner.utils import dict_array_to_dict_tf
from belief_learner.utils.array import dict_array_to_dict_python
from belief_learner.workers.worker import Worker
from belief_learner.networks.model_architecture import ModelArchitecture


class BeliefA2CWorker(Worker):
    __name__ = "BeliefA2CWorker"

    def __init__(self,
                 replay_buffer: ReplayBuffer,
                 n_next_state_samples: int,
                 n_state_samples: int,
                 env_step_per_batch: int,
                 belief_shape: int,
                 sub_belief_shape: Tuple[int, ...],
                 sub_model_architecture: ModelArchitecture,
                 made_architecture: ModelArchitecture,
                 observation_encoder: keras.Model,
                 observation_encoded_shape: Tuple[int, ...],
                 action_encoded_shape: Tuple[int, ...],
                 latent_transition: Callable[[Float, Float], Distribution],
                 obs_filter: Callable[[Float, Float], Distribution],
                 sub_belief_prior_temperature: float,
                 optimizer: str,
                 clip_by_global_norm: Optional[float],
                 learning_rate: float,
                 latent_state_size: int,
                 env_creator: Callable,
                 env_config: dict,
                 nbr_environments: int,
                 multi_process_env: bool,
                 seed: Optional[int],
                 target_update_freq: int,
                 tau: float,
                 gamma: float,
                 policy_architecture: ModelArchitecture,
                 lambda_: float,
                 policy_weight: float,
                 value_weight: float,
                 entropy_weight: float,
                 state_observation_embedding_fn: Callable[[Float, Float, Float], Float],
                 latent_reward: Callable[[Float, Float, Float], Distribution],
                 transition_lipschitz_net: keras.Model,
                 n_belief_updates: int,
                 normalize_advantage: bool = False,
                 action_encoder: Callable[[tf.Tensor], tf.Tensor] = tfb.Identity(),
                 filter_variance: Float = 1.,
                 filter_variance_target: Float = 1e-4,
                 filter_variance_decay_steps: int = int(1e6),
                 filter_variance_power_decay: float = 1.5,
                 normalize_log_obs_filter: bool = False,
                 belief_loss_factor: float = 1.,
                 latent_state_to_obs_: Optional[Callable] = None,
                 cost_fn_obs: Optional[Callable] = None,
                 dual_optim: bool = False,
                 use_normalizing_term: bool = False,
                 learn_sub_belief_encoder: bool = False,
                 use_huber: bool = True,
                 share_network: bool = True,
                 use_gru: bool = True,
                 use_running_variance: bool = False,
                 use_learned_variance: bool = False,
                 get_running_variance: Optional[Callable[[Float], Float]] = None,
                 env_img: Optional[Any] = None,
                 learn_with_img: bool = False,
                 n_step_img: Optional[int] = None,
                 n_train_img: Optional[int] = None,
                 n_critic: int = 5,
                 reward_loss_scale_factor: float = 1.,
                 transition_loss_scale_factor: float = 1.,
                 gradient_penalty_scale_factor: float = 10.,
                 maximizer_lr: float = 3e-4,
                 maximizer_batch_size: int = 128,
                 weight_decay: Optional[float] = 0.,
                 optimizer_epsilon: float = 1e-7,
                 lr_decay_power: Optional[float] = 0.,
                 amsgrad: bool = False,
                 observation_encoder_cnn_arch: Optional[ModelArchitecture] = None,
                 categorical_beliefs: bool = False,
                 belief_cnn_filters: int = 64,
                 belief_filters_variation: Optional[str] = None,  # in [increasing, decreasing]
                 belief_cnn_kernels: int = 5,
                 belief_kernel_variation: Optional[str] = None,  # in [increasing, decreasing]
                 belief_down_sampling_op: str = 'strides',  # in [strides, max_pooling, avg_pooling,]
                 belief_up_sampling_op: str = 'strides',  # [in strides, repeat]
                 belief_filter_aggregation_op: str = 'conv',  # in [conv, global_max_pooling, global_avg_pooling]
                 belief_cnn_activation: str = 'relu',
                 use_dtv: bool = False,
                 dtv_split_depth: int = 13,
                 use_sub_belief: bool = True,
                 to_binary_straight_through_gradient: bool = False,
                 **kwargs):
        self._n_train_calls = 0

        self.learning_rate = learning_rate
        if weight_decay == 0.:
            weight_decay = None
        self.belief_lr = kwargs.get('belief_learning_rate', self.learning_rate)
        if lr_decay_power > 0.:
            maximizer_lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=maximizer_lr, decay_steps=n_belief_updates * n_critic,
                end_learning_rate=1e-6, power=lr_decay_power)
            self.belief_lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.belief_lr, decay_steps=n_belief_updates,
                end_learning_rate=1e-6, power=lr_decay_power)

        belief_kwargs = {
            'replay_buffer': replay_buffer,
            'n_next_state_samples': n_next_state_samples,
            'n_state_samples': n_state_samples,
            'batch_size': env_step_per_batch * nbr_environments,
            'belief_shape': belief_shape,
            'sub_belief_shape': sub_belief_shape,
            'sub_model_architecture': sub_model_architecture,
            'made_architecture': made_architecture,
            'observation_encoder': observation_encoder,
            'observation_encoded_shape': observation_encoded_shape,
            'action_encoded_shape': action_encoded_shape,
            'latent_transition': latent_transition,
            'obs_filter': obs_filter,
            'sub_belief_prior_temperature': sub_belief_prior_temperature,
            'optimizer': optimizer,
            'clip_by_global_norm': kwargs.get('belief_clip_by_global_norm', clip_by_global_norm),
            'learning_rate': self.belief_lr,
            'latent_state_size': latent_state_size,
            'action_encoder': action_encoder,
            'filter_variance': filter_variance,
            'filter_variance_target': filter_variance_target,
            'filter_variance_decay_steps': filter_variance_decay_steps,
            'filter_variance_power_decay': filter_variance_power_decay,
            'normalize_log_obs_filter': normalize_log_obs_filter,
            'use_gru': use_gru,
            'latent_state_to_obs_': latent_state_to_obs_,
            'cost_fn_obs': cost_fn_obs,
            'use_normalizing_term': use_normalizing_term,
            'use_running_variance': use_running_variance and not use_learned_variance,
            'use_learned_variance': use_learned_variance,
            'get_running_variance': get_running_variance,
            'state_observation_embedding_fn': state_observation_embedding_fn,
            'latent_reward': latent_reward,
            'transition_lipschitz_net': transition_lipschitz_net,
            'n_critic': n_critic,
            'reward_loss_scale_factor': reward_loss_scale_factor,
            'transition_loss_scale_factor': transition_loss_scale_factor,
            'gradient_penalty_scale_factor': gradient_penalty_scale_factor,
            'maximizer_lr': maximizer_lr,
            'maximizer_batch_size': maximizer_batch_size,
            'observation_encoder_cnn_arch': observation_encoder_cnn_arch,
            'weight_decay': weight_decay,
            'optimizer_epsilon': optimizer_epsilon,
            'amsgrad': amsgrad,
        }

        if categorical_beliefs:
            belief_kwargs = {
                **belief_kwargs,
                'cnn_filters': belief_cnn_filters,
                'filters_variation': belief_filters_variation,
                'cnn_kernels': belief_cnn_kernels,
                'kernel_variation': belief_kernel_variation,
                'down_sampling_op': belief_down_sampling_op,
                'up_sampling_op': belief_up_sampling_op,
                'filter_aggregation_op': belief_filter_aggregation_op,
                'cnn_activation': belief_cnn_activation,
                'use_dtv': use_dtv,
                'dtv_split_depth': dtv_split_depth,
                'to_binary_straight_through_gradient': to_binary_straight_through_gradient,
            }
            self.belief_worker = CategoricalBeliefWorker(**belief_kwargs)
            self.use_dtv = use_dtv
        else:
            self.belief_worker = BeliefWorker(**belief_kwargs)
            self.use_dtv = False

        self.a2c_worker = A2CWorker(
            env_creator=env_creator,
            env_config=env_config,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            nbr_environments=nbr_environments,
            multi_process_env=multi_process_env,
            sub_belief_updater=self.belief_worker.sub_belief_encode,
            sub_belief_encode_seq=self.belief_worker.sub_belief_encode_seq,
            seed=seed,
            target_update_freq=target_update_freq,
            tau=tau,
            gamma=gamma,
            replay_buffer=replay_buffer,
            env_step_per_batch=env_step_per_batch,
            policy_architecture=policy_architecture,
            lambda_=lambda_,
            policy_weight=policy_weight,
            value_weight=value_weight,
            entropy_weight=entropy_weight,
            clip_by_global_norm=clip_by_global_norm,
            normalize_advantage=normalize_advantage,
            learn_sub_belief_encoder=learn_sub_belief_encoder,
            use_huber=use_huber,
            env_img=env_img,
            learn_with_img=learn_with_img,
            n_step_img=n_step_img,
            n_train_img=n_train_img,
            share_network=share_network,
            sub_belief_upscaler=self.belief_worker.sub_belief_upscaler if categorical_beliefs and not use_sub_belief else None,
            belief_embedding=self.belief_worker.belief_embedding if categorical_beliefs and not use_sub_belief else None
        )
        # **kwargs)
        assert optimizer.lower() in ['adam', 'adamw']
        self.learn_with_img = learn_with_img
        self.clip_by_global_norm = clip_by_global_norm
        self.dual_optim = dual_optim
        self.optimizer = keras.optimizers.Adam(self.learning_rate)
        if self.dual_optim:
            self.belief_clip_by_global_norm = kwargs.get('belief_clip_by_global_norm', self.clip_by_global_norm)
            self.belief_optimizer = tf.keras.optimizers.get(
                {'class_name': optimizer,
                 'config': {'learning_rate': self.belief_lr,
                            # 'weight_decay': weight_decay,
                            'epsilon': optimizer_epsilon,
                            'amsgrad': amsgrad}})
        self.belief_loss_factor = belief_loss_factor
        self.reward_loss_scale_factor = reward_loss_scale_factor
        self.transition_loss_scale_factor = transition_loss_scale_factor

    @tf.function
    def compute_loss_and_apply_grad(self, data):
        print("tracing compute_loss_and_apply_grad")
        with tf.GradientTape(persistent=True) as tape:
            sub_beliefs = self.belief_worker.sub_belief_encode_seq(
                data[NEXT_OBS], data[ACTION], data[SUB_BELIEF][0], training=True)
            policy_loss, value_loss, entropy = self.a2c_worker.compute_loss(
                data,
                sub_beliefs if self.a2c_worker.learn_sub_belief_encoder else tf.stop_gradient(sub_beliefs))
            belief_loss, (log_p_next_states, log_expectation_to_next_state, log_obs_filter, normalizing_term, obs_dist,
                          obs_variance) = \
                self.belief_worker.compute_seq_loss(
                    prev_sub_belief=data[SUB_BELIEF][0],
                    sub_belief=sub_beliefs,
                    action=data[ACTION][:-1],
                    next_obs=data[NEXT_OBS][:-1],
                    mask=data["mask"][1:],
                )

            regularization = self.belief_worker.compute_reward_transition_regularizers(
                state=data[STATE][:-1],
                is_reset_state=data[IS_RESET_STATE][:-1],
                next_is_reset_state=data[IS_RESET_STATE][1:],
                observation=data[OBS][:-1],
                action=data[ACTION][:-1],
                next_state=data[NEXT_STATE][:-1],
                next_observation=data[NEXT_OBS][:-1],
                prev_sub_belief=data[SUB_BELIEF][0],
                sub_belief=sub_beliefs,
                mask=data["mask"][1:], )

            _belief_loss = belief_loss
            belief_loss += self.reward_loss_scale_factor * regularization['reward_loss']
            belief_loss += self.transition_loss_scale_factor * regularization['transition_loss']

            a2c_loss = self.a2c_worker.policy_weight * policy_loss + self.a2c_worker.value_weight * value_loss - \
                       self.a2c_worker.entropy_weight * entropy
            loss = self.belief_loss_factor * belief_loss + a2c_loss
        return_dict = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "belief_loss": _belief_loss,
            "reward_regularizer": regularization['reward_loss'],
            "transition_regularizer": regularization['transition_loss'],
            "loss": loss,
            "log_p_next_states": log_p_next_states,
            "log_expectation_to_next_state": log_expectation_to_next_state,
            "log_obs_filter": log_obs_filter,
            "obs_dist": obs_dist,
            # "obs_filter_variance": obs_variance,
        }
        if normalizing_term is not None:
            return_dict["normalizing_term"] = normalizing_term

        if self.dual_optim:
            rl_update_weights = self.a2c_worker._get_weights()
            if self.a2c_worker.learn_sub_belief_encoder:
                rl_update_weights += self.belief_worker.get_sub_belief_weights()
            grads_rl = tape.gradient(a2c_loss, rl_update_weights)
            if self.clip_by_global_norm:
                grads_rl, grad_norm_rl = tf.clip_by_global_norm(grads_rl, self.clip_by_global_norm)
            else:
                grad_norm_rl = tf.linalg.global_norm(grads_rl)
            if tf.reduce_any([
                tf.reduce_any(tf.math.logical_or(tf.math.is_nan(grad), tf.math.is_inf(grad))) for grad in grads_rl if
                grad is not None]
            ):
                tf.print("Belief A2C worker got nan in A2C loss.")
            else:
                # tf.print("Belief worker: applying gradient.")
                self.optimizer.apply_gradients(zip(grads_rl, rl_update_weights))
            return_dict["grad_norm_rl"] = grad_norm_rl

            # Belief
            grads_belief = tape.gradient(belief_loss, self.belief_worker._get_weights())
            if self.clip_by_global_norm:
                grads_belief, grad_norm_belief = tf.clip_by_global_norm(grads_belief, self.belief_clip_by_global_norm)
            else:
                grad_norm_belief = tf.linalg.global_norm(grads_belief)
            if tf.reduce_any([
                tf.reduce_any(tf.math.logical_or(tf.math.is_nan(grad), tf.math.is_inf(grad))) for grad in grads_belief
                if grad is not None]
            ):
                tf.print("Belief A2C worker got nan in Belief loss.")
            else:
                # tf.print("Belief worker: applying gradient.")
                self.belief_optimizer.apply_gradients(zip(grads_belief, self.belief_worker._get_weights()))
            return_dict["grad_norm_belief"] = grad_norm_belief

        else:
            grads = tape.gradient(loss, self._get_weights())
            # tf.print("belief grad norm : ", tf.linalg.global_norm(grads[:-12]))
            # tf.print("policy grad norm : ", tf.linalg.global_norm(grads[-12:]))
            if self.clip_by_global_norm:
                grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_by_global_norm)
            else:
                grad_norm = tf.linalg.global_norm(grads)
            # tf.print("belief grad norm after grad clip : ", tf.linalg.global_norm(grads[:-12]))
            # tf.print("policy grad norm after grad clip: ", tf.linalg.global_norm(grads[-12:]))
            if tf.reduce_any([
                tf.reduce_any(tf.math.logical_or(tf.math.is_nan(grad), tf.math.is_inf(grad))) for grad in grads if
                grad is not None]
            ):
                tf.print("Belief A2C worker got nan in loss.")
            else:
                # tf.print("Belief worker: applying gradient.")
                self.optimizer.apply_gradients(zip(grads, self._get_weights()))

            return_dict["grad_norm"] = grad_norm

        return return_dict

    def train(self):
        self._n_train_calls += 1
        self.belief_worker.update_variance()
        episode_ended_stats, data = self.a2c_worker.train_interact()
        data = dict_array_to_dict_tf(data)
        if not self.use_dtv:
            return_dict_maximizer = {
                'belief_gradient_penalty': self.belief_worker.compute_maximizer_loss_and_apply_grads()
            }
            return_dict_maximizer = dict_array_to_dict_python(return_dict_maximizer)
        else:
            return_dict_maximizer = dict()
        if self.a2c_worker.learn_with_img:
            return_dict_belief = self.belief_worker.compute_seq_loss_and_apply_gradients(data)
            return_dict_belief = dict_array_to_dict_python(return_dict_belief)
            return_dict_rl = self.a2c_worker.train_img()
            return_dict = {**return_dict_rl, **return_dict_belief, **return_dict_maximizer}
        else:
            return_dict = self.compute_loss_and_apply_grad(data)
            return_dict = dict_array_to_dict_python(return_dict)
            return_dict = {**return_dict, **return_dict_maximizer}
        return_dict["episode_ended_stats"] = episode_ended_stats
        if False and not (self.belief_worker.use_running_variance or self.belief_worker.use_learned_variance):
            obs_filter_variance = self.belief_worker.obs_variance.value().numpy().item()
            return_dict["obs_filter_variance"] = obs_filter_variance
        return return_dict

    def interact(self, n: int, training: bool = True, force_random_action: bool = False,
                 fill_replay_buffer: Optional[bool] = None):
        assert not training
        return self.a2c_worker.interact(n, training, force_random_action, fill_replay_buffer)

    @property
    def env_steps(self):
        return self.a2c_worker.env_steps

    @property
    def n_episode(self):
        return self.a2c_worker.n_episode

    @property
    def nbr_env(self):
        return self.a2c_worker.nbr_env

    @property
    def nbr_actions(self):
        return self.a2c_worker.nbr_actions

    def evaluate(self, n: int):
        return self.a2c_worker.evaluate(n)

    @property
    def n_train_called(self):
        return self._n_train_calls

    def save(self, path: str):
        raise NotImplementedError()

    def load(self, path: str):
        raise NotImplementedError()

    def _get_weights(self):
        return self.belief_worker._get_weights() + self.a2c_worker._get_weights()

    def set_maximizer_dataset(self):
        self.belief_worker.set_maximizer_dataset()
