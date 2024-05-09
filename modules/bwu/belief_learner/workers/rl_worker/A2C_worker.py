from typing import Optional, Callable, List, Tuple, Dict

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from tf_agents.typing.types import Float
from tensorflow_probability import distributions as tfd

from belief_learner import ReplayBuffer
from belief_learner.buffers.data_store import DataStore
from belief_learner.utils.definitions import SUB_BELIEF, REWARD, DONE, ACTION, DictArray, NEXT_OBS, OBS, STATE, \
    NEXT_STATE
from belief_learner.utils.distributions import StochasticDistribution
from belief_learner.networks.dense import generate_rl_model
from belief_learner.workers.rl_worker.rl_worker import RLWorker
from belief_learner.utils import merge_first_dims, unmerge_first_dims, dict_array_to_dict_tf

from belief_learner.utils.decorators import log_usage
from belief_learner.utils.policy import compute_gae_tf3

from belief_learner.networks.model_architecture import ModelArchitecture


class A2CWorker(RLWorker):

    @log_usage
    def __init__(
            self,
            env_creator: Callable,
            env_config: dict,
            optimizer_name: str,
            learning_rate: float,
            nbr_environments: int,
            multi_process_env: bool,
            sub_belief_updater: Callable,
            seed: Optional[int],
            target_update_freq: int,
            tau: float,
            gamma: float,
            replay_buffer: ReplayBuffer,
            env_step_per_batch: int,
            policy_architecture: ModelArchitecture,
            lambda_: float,
            policy_weight: float,
            value_weight: float,
            entropy_weight: float,
            clip_by_global_norm: Optional[float],
            sub_belief_weights: Optional[List] = None,
            belief_optimizer: Optional = None,
            clip_by_global_norm_sub_belief: Optional = None,
            learn_sub_belief_encoder: bool = False,
            learn_sub_belief_encoder_same_opt: bool = False,
            normalize_advantage: bool = False,
            share_network: bool = False,
            sub_belief_encode_seq=None,
            _debug_use_state=False,
            _debug_best_action=False,
            use_huber=False,
            _include_in_weights=None,
            env_img=None,
            learn_with_img: bool = False,
            n_step_img: Optional[int] = None,
            n_train_img: Optional[int] = None,
            sub_belief_upscaler: tf.keras.Model = None,
            belief_embedding: tf.keras.Model = None,
            **kwargs,
    ):
        super().__init__(env_creator, env_config, optimizer_name, learning_rate, nbr_environments, multi_process_env,
                         sub_belief_updater, seed, target_update_freq, tau, gamma, replay_buffer, env_step_per_batch,
                         None, None, _debug_use_state, env_img, _debug_best_action)
        self._include_in_weights = _include_in_weights
        self.clip_by_global_norm_sub_belief = clip_by_global_norm_sub_belief
        self.belief_optimizer = belief_optimizer
        self.sub_belief_weights = sub_belief_weights
        self.learn_sub_belief_encoder = learn_sub_belief_encoder
        self.learn_sub_belief_encoder_same_opt = learn_sub_belief_encoder_same_opt
        assert not (learn_sub_belief_encoder_same_opt and learn_sub_belief_encoder)
        self.clip_by_global_norm = clip_by_global_norm
        self._current_data_store: list[DictArray] = []
        self._lambda = lambda_
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self._explore_startegy = None
        self.normalize_advantage = normalize_advantage
        self.share_network = share_network
        assert belief_embedding == sub_belief_upscaler == None or None not in [belief_embedding, sub_belief_upscaler]
        self.sub_belief_upscaler = sub_belief_upscaler
        self.belief_embedding = belief_embedding

        self.learn_with_img = learn_with_img
        self.n_step_img = n_step_img
        self.n_train_img = n_train_img

        input_dim = self.replay_buffer.elements_scheme[SUB_BELIEF].shape
        if self._debug_use_state:
            input_dim = self.replay_buffer.elements_scheme[STATE].shape

        self.policy_architecture = policy_architecture._replace(
            input_dim=input_dim,
            output_dim=(self.nbr_actions,),
        )

        self.model = generate_rl_model(
            self.policy_architecture._replace(name='policy'),
            dueling=False,
            actor_critic=share_network,
            sub_belief_upscaler=sub_belief_upscaler,
            belief_embedding=belief_embedding
        )

        if not share_network:
            self.v_model = generate_rl_model(
                self.policy_architecture._replace(output_dim=(1,), name='value'),
                dueling=False,
                actor_critic=False,
                sub_belief_upscaler=sub_belief_upscaler,
                belief_embedding=belief_embedding
            )

        self.models = ['model']
        if not self.share_network:
            self.models.append('v_model')
        self.assets = []
        self.distribution = StochasticDistribution
        self.sub_belief_encode_seq = sub_belief_encode_seq
        self._alt = sub_belief_encode_seq is not None
        assert self._alt
        if self._alt:
            self._alt_data_store = DataStore(self.replay_buffer._reset_obs_state[OBS],
                                             self.replay_buffer.default_prev_action,
                                             self._prev_sub_belief[0].copy(),
                                             self.replay_buffer.elements_scheme,
                                             self.batch_size // self.nbr_env,
                                             reset_state=self.replay_buffer._reset_obs_state[STATE],
                                             use_state=True)
            self._alt_data_store_img = DataStore(self.replay_buffer._reset_obs_state[OBS],
                                                 self.replay_buffer.default_prev_action,
                                                 self._prev_sub_belief[0].copy(),
                                                 self.replay_buffer.elements_scheme,
                                                 self.batch_size // self.nbr_env,
                                                 reset_state=self.replay_buffer._reset_obs_state[STATE],
                                                 use_state=True)
        self._use_huber = use_huber
        self._huber_loss = tfk.losses.Huber(reduction=tfk.losses.Reduction.NONE)

    def _interact(self, training: bool = True, force_random_action: bool = False,
                  fill_replay_buffer: Optional[bool] = None) -> Tuple[DictArray, List[Dict]]:
        if fill_replay_buffer is None:
            fill_replay_buffer = training
        data, episode_ended_stats = super()._interact(fill_replay_buffer, force_random_action)
        if training:
            if self._alt:
                self._alt_data_store.add(data)
            else:
                self._current_data_store.append(data)
        return data, episode_ended_stats

    @tf.function
    def compute_loss_and_apply_gradients(self, data):
        with tf.GradientTape(persistent=self.learn_sub_belief_encoder) as tape:
            sub_beliefs = self.sub_belief_encode_seq(data[NEXT_OBS], data[ACTION], data[SUB_BELIEF][0])
            # print(np.absolute(data['next_sub_belief'] - sub_beliefs.numpy())[:-1].max())
            policy_loss, value_loss, entropy = self.compute_loss(data, sub_beliefs)
            loss = self.policy_weight * policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy
        grads = tape.gradient(loss, self._get_weights())
        grad_norm = None
        if self.clip_by_global_norm:
            grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_by_global_norm)
        self.optimizer.apply_gradients(zip(grads, self._get_weights()))
        grads_norm = [grad_norm]
        if self.learn_sub_belief_encoder:
            grads_sub_belief = tape.gradient(loss, self.sub_belief_weights)
            grad_norm_sub_belief = None
            if self.clip_by_global_norm_sub_belief:
                grads_sub_belief, grad_norm_sub_belief = tf.clip_by_global_norm(grads_sub_belief,
                                                                                self.clip_by_global_norm_sub_belief)
            self.belief_optimizer.apply_gradients(zip(grads_sub_belief, self.sub_belief_weights))
            grads_norm.append(grad_norm_sub_belief)
        return loss, (policy_loss, value_loss, entropy), grads_norm

    @tf.function
    def compute_loss(self, data, sub_beliefs):
        T, B = sub_beliefs.shape[:2]
        action = tf.one_hot(data[ACTION][1:], depth=self.n_actions)
        if not self._debug_use_state:
            logit_policy, value = self.compute_policy_value(merge_first_dims(sub_beliefs, 2), training=True)
        else:
            next_state = data[NEXT_STATE]
            logit_policy, value = self.compute_policy_value(merge_first_dims(next_state, 2), training=True)
        logit_policy = unmerge_first_dims(logit_policy, (T, B))[:-1]
        log_prob = tf.math.log_softmax(logit_policy, -1)
        prob = tf.math.exp(log_prob)
        log_prob_selected = tf.reduce_sum(log_prob * action, -1)

        value = unmerge_first_dims(value, (T, B))[..., 0]
        # tf.print("Value max :", tf.reduce_max(tf.math.abs(value)))
        mask = data["mask"]
        policy_mask = tf.cast(mask[1:], tf.float32)
        advantage = compute_gae_tf3(
            data[REWARD][1:],
            value[:-1] * policy_mask,
            value[1:] * policy_mask,
            data[DONE][1:],
            self.gamma,
            self._lambda)
        target_value = tf.stop_gradient(
            value[:-1] + advantage)  
        if self._use_huber:
            value_loss = tf.reduce_sum(
                self._huber_loss(tf.reshape(target_value, (-1, 1)), tf.reshape(value[:-1], (-1, 1)))
                * tf.reshape(policy_mask, (-1,))
            ) / tf.reduce_sum(tf.cast(policy_mask, tf.float32))
        else:
            value_loss = .5 * tf.reduce_sum(tf.square(value[:-1] - target_value) * policy_mask) / tf.reduce_sum(
                tf.cast(policy_mask, tf.float32))
        if self.normalize_advantage: 
            raise ValueError
            advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-10)
        policy_loss = - tf.reduce_sum(policy_mask * log_prob_selected * tf.stop_gradient(advantage)) / tf.reduce_sum(
            tf.cast(policy_mask, tf.float32))
        entropy = - tf.reduce_sum(policy_mask[..., None] * log_prob * prob) / tf.reduce_sum(
            tf.cast(policy_mask, tf.float32))
        return policy_loss, value_loss, entropy

    def train_interact(self):
        self._current_data_store = []
        if self._alt and self._episode_id is not None:  
            self._alt_data_store.reset(episode_nbrs=self._episode_id.copy(),
                                       prev_sub_belief=self._prev_sub_belief.copy(),
                                       prev_action=self._prev_action.copy(),
                                       obs=self._current_state_obs[OBS],
                                       is_first=self._timestep == 0,
                                       state=self._current_state_obs[STATE],
                                       prev_obs=self._prev_state_obs[OBS].copy(),
                                       prev_state=self._prev_state_obs[STATE].copy(),
                                       )
        episode_ended_stats = self.interact(self.batch_size // self.nbr_env, True)
        data = self._alt_data_store.time_output()
        return episode_ended_stats, data

    def train(self):
        episode_ended_stats, data = self.train_interact()
        loss, (policy_loss, value_loss, entropy), grads_norm = self.compute_loss_and_apply_gradients(data)
        stats = {
            'policy_loss': policy_loss.numpy().item(),
            'value_loss': value_loss.numpy().item(),
            'entropy_loss': entropy.numpy().item(),
            # 'n_train_call': self._n_train_calls,
            'episode_ended_stats': episode_ended_stats,
            **dict(zip(['grad_norm', 'grad_norm_sub_belief'], map(lambda x: x.numpy().item(), grads_norm)))
        }

        return stats

    def value_model(self, belief, training: bool = False):
        if self.share_network:
            value = self.model(belief, training=training)[1]
        else:
            value = self.v_model(belief, training=training)
        return value

    def policy_model(self, belief, training: bool = False):
        policy = self.model(belief, training=training)
        if self.share_network:
            policy = policy[0]
        return policy

    def compute_policy_value(self, belief, training: bool = False):
        if self.share_network:
            policy, value = self.model(belief, training=training)
        else:
            policy = self.policy_model(belief, training=training)
            value = self.value_model(belief, training=training)
        return policy, value

    def _get_weights(self):
        if self.share_network:
            weights = self.model.trainable_variables
        else:
            weights = self.model.trainable_variables + self.v_model.trainable_variables
        if self._include_in_weights is not None:
            for e in self._include_in_weights:
                weights += e.trainable_variables
            # weights += self.sub_belief_weights
        weights = [weight for weight in weights if 'sub_belief_upscaler' not in weight.name]
        return weights

    @tf.function
    def _train_img(self, latent_state, prev_action, prev_sub_belief):
        latent_state_extended = (latent_state, prev_action, prev_sub_belief)
        with tf.GradientTape() as tape:
            sub_belief, policy_logit, value, actions, rewards, dones, next_value = self.interact_img(self.n_step_img,
                                                                                                     *latent_state_extended)
            policy_loss, value_loss, entropy = self.compute_loss_img(policy_logit, value, actions, rewards, dones,
                                                      next_value)
            a2c_loss = self.policy_weight * policy_loss + self.value_weight * value_loss - \
                       self.entropy_weight * entropy
        grads = tape.gradient(a2c_loss, self._get_weights())
        if self.clip_by_global_norm:
            grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_by_global_norm)
        else:
            grad_norm = tf.linalg.global_norm(grads)
        self.optimizer.apply_gradients(zip(grads, self._get_weights()))
        dones_ratio = tf.reduce_mean(tf.cast(dones, tf.float32))
        return a2c_loss, (policy_loss, value_loss, entropy), grad_norm, dones_ratio

    @tf.function
    def compute_loss_img(self, policy_logit, value, actions, rewards, dones, next_value):
        value = tf.squeeze(value, -1)
        next_value = tf.squeeze(next_value, -1)
        rewards = tf.squeeze(rewards, -1)
        next_values = tf.concat([value[1:], next_value[None]], axis=0)

        advantage = compute_gae_tf3(
            rewards,
            value,
            next_values,
            dones,
            self.gamma,
            self._lambda)
        target_value = value + advantage
        target_value = tf.stop_gradient(target_value)
        if self._use_huber:
            raise NotImplementedError()

        actions_one_hot = tf.one_hot(actions, self.n_actions)
        log_prob = tf.nn.log_softmax(policy_logit, axis=-1)
        log_prob_selected = tf.reduce_sum(actions_one_hot * log_prob, axis=-1)
        prob = tf.exp(log_prob)

        value_loss = .5 * tf.reduce_mean(tf.square(value - target_value))
        if self.normalize_advantage:  
            raise ValueError
            advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-10)
        policy_loss = - tf.reduce_mean(log_prob_selected * tf.stop_gradient(advantage))
        entropy = tf.reduce_mean(-tf.reduce_sum(log_prob * prob, axis=-1))
        return policy_loss, value_loss, entropy

    def train_img(self):
        assert self.learn_with_img
        assert self.n_step_img is not None
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'grad_norm': [],
            'dones_ratio': [],
        }
        for _ in range(self.n_train_img):
            prev_action = tf.convert_to_tensor(self._prev_action)
            prev_sub_belief = tf.convert_to_tensor(self._prev_sub_belief)
            latent_state = self.env_img.init(dict_array_to_dict_tf(self._current_state_obs))
            a2c_loss, (policy_loss, value_loss, entropy), grad_norm, dones_ratio = self._train_img(latent_state, prev_action, prev_sub_belief)
            stats["policy_loss"].append(policy_loss.numpy().item())
            stats["value_loss"].append(value_loss.numpy().item())
            stats["entropy_loss"].append(entropy.numpy().item())
            stats["grad_norm"].append(grad_norm.numpy().item())
            stats["dones_ratio"].append(dones_ratio.numpy().item())

        stats['policy_loss'] = np.mean(stats["policy_loss"])
        stats['value_loss'] = np.mean(stats["value_loss"])
        stats['entropy_loss'] = np.mean(stats["entropy_loss"])
        stats['grad_norm'] = np.mean(stats["grad_norm"])
        stats['dones_ratio'] = np.mean(stats["dones_ratio"])
        return stats

    @tf.function
    def interact_img(self, n: int, latent_state: Float, prev_action: Float, prev_sub_belief: Float,
                     training: bool = True) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Performs n interaction between the agent and the environments.

        :param n: Number of interactions
        :param training: If true exploration is activated and interaction will be saved to the replay buffer.
        """
        # Setting things up
        sub_belief = []
        policy_logit = []
        value = []
        actions = []
        rewards = []
        dones = []
        latent_state_extended = (latent_state, prev_action, prev_sub_belief)
        for _ in range(n):
            latent_state_extended, sub_belief_, policy_logit_, value_, actions_, rewards_, dones_ = self._interact_img(
                *latent_state_extended, training)
            sub_belief.append(sub_belief_)
            policy_logit.append(policy_logit_)
            value.append(value_)
            actions.append(actions_)
            rewards.append(rewards_)
            dones.append(dones_)

        sub_belief = tf.stack(sub_belief, 0)
        policy_logit = tf.stack(policy_logit, 0)
        value = tf.stack(value, axis=0)
        actions = tf.stack(actions, axis=0)
        rewards = tf.stack(rewards, axis=0)
        dones = tf.stack(dones)

        next_value = self._compute_next_value_img(*latent_state_extended)

        return sub_belief, policy_logit, value, actions, rewards, dones, next_value

    @tf.function
    def _compute_next_value_img(self, latent_state: Float, prev_action: Float, prev_sub_belief: Float):
        state, obs = self.env_img.latent_state_state_obs(latent_state)
        sub_belief = self.sub_belief_encode_seq(obs=obs[None],
                                                prev_action=prev_action[None],
                                                prev_sub_belief=prev_sub_belief,
                                                )[0]
        next_value = self.value_model(sub_belief, True)
        return next_value

    @tf.function
    def _interact_img(self, latent_state: Float, prev_action: Float, prev_sub_belief: Float, training: bool = True) -> \
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        state, obs = self.env_img.latent_state_state_obs(latent_state)
        # might move latent_state-> obs here
        sub_belief = self.sub_belief_encode_seq(obs=obs[None],
                                                prev_action=prev_action[None],
                                                prev_sub_belief=prev_sub_belief,
                                                )[0]

        if self._debug_use_state:
            policy_logit, value = self.compute_policy_value(state, training=training)
        else:
            policy_logit, value = self.compute_policy_value(sub_belief, training=training)

        actions = self.distribution(policy_logit).sample()

        new_latent_state, rewards, dones = self.env_img.step(latent_state, actions)

        prev_action = actions
        prev_sub_belief = sub_belief

        if tf.reduce_any(dones):
            prev_sub_belief = (1. - tf.cast(dones[..., None], tf.float32)) * prev_sub_belief
            prev_action = (1 - tf.cast(dones, tf.int32)) * prev_action

        return (new_latent_state, prev_action, prev_sub_belief), sub_belief, policy_logit, value, actions, rewards, dones

    def evaluate_img(self):
        if "RepeatPrevious" not in self._evaluate_env.envs[0].__class__.__name__:
            return ''
        env = self._evaluate_env
        state_obs_real = env.reset()
        prev_action = tf.zeros_like(self._prev_action_img[:1])
        prev_sub_belief = tf.zeros_like(self._prev_sub_belief_img[:1])
        latent_state = self.env_img.init(dict_array_to_dict_tf(state_obs_real))
        done = False
        latent_state_extended = (latent_state, prev_action, prev_sub_belief)
        actions = []
        rewards_img = []
        rewards = []
        state_obs_img = []
        i = 0
        while not done and (i < 120):
            state_obs_img.append(self.env_img.latent_state_state_obs(latent_state))
            next_latent_state_extended, sub_belief, policy_logit, value, action, reward_img, done = self._interact_img(*latent_state_extended, training=False)
            done = done.numpy().item()
            actions.append(action)
            rewards_img.append(reward_img)
            latent_state_extended = next_latent_state_extended
            latent_state = latent_state_extended[0]
            i += 1
        states_img, obs_img = zip(*state_obs_img)
        states_img = tf.concat(states_img, axis=0).numpy()
        states_img = [states_img[:, :1],
                      states_img[:, 1:-4].reshape((states_img.shape[0], -1, 4)).argmax(-1),
                      states_img[:, -4:]]
        states_img = np.concatenate(states_img, axis=-1)
        obs_img = tf.concat(obs_img, axis=0).numpy()
        obs_img = [obs_img[:, :1],
                   obs_img[:, 1:].argmax(-1)[..., None]]
        obs_img = np.concatenate(obs_img, axis=-1)

        actions = tf.concat(actions, axis=0).numpy().flatten()
        rewards_img = tf.concat(rewards_img, axis=0).numpy().flatten()
        string = ''
        for el in zip(states_img, obs_img, actions, rewards_img):
            string += '\n'.join(map(str,map(lambda x: x.tolist(), el)))
            string += '\n'
        return string
