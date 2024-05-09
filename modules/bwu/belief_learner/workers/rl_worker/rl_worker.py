import os
from typing import Optional, Dict, Callable, Union, List, Tuple, Any
from functools import partial
import gym
from copy import deepcopy
import numpy as np
import tensorflow as tf
from belief_learner.utils.env.vec_env.dummy_vec_env import DummyVecEnv
from tensorflow.keras.optimizers import Optimizer, Adam

from belief_learner.utils.definitions import SUB_BELIEF, EPISODE_NBR, TIMESTEP, ACTION, REWARD, STATE, OBS, DONE, DictArray, \
    NEXT_OBS, NEXT_SUB_BELIEF, NEXT_STATE
from belief_learner.buffers.replay_buffer import ReplayBuffer
from belief_learner.workers.worker import Worker
from belief_learner.utils import array_to_np

from belief_learner.utils import get_logger

logger = get_logger(__name__)


episode_stat_empty = {
    'timestep': 0,
    'cum_rewards': 0
}


class RLWorker(Worker):

    __name__ = "RL Worker"
    def __init__(self,
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
                 activation_fn: Union[str, Callable],
                 hidden_layer_sizes: List[int],
                 _debug_use_state: bool = False,
                 env_img: Optional[Any] = None,
                 _debug_best_action: bool = False,
                 ):
        """

        :param env_creator: Function that creates an environment.
        :param nbr_environments: Number of parallel environments to use.
        :param multi_process_env: If true the parallel environments will be run in subprocesses.
        :param sub_belief_updater: Function that computes the belief update.
        :param seed: Seed of the environments.
        :param replay_buffer: Replay buffer to store experiences.
        """
        # Environments creation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.gamma = gamma
        self.env_config = env_config.copy()
        self.env_creator = env_creator
        env_creator = partial(self.env_creator, **env_config)
        self.nbr_env = nbr_environments
        self._evaluate_env = DummyVecEnv([env_creator])
        if self.nbr_env == 1 and multi_process_env:
            logger.warning("Requesting multiprocessing for the environments but without parallelized environments. "
                           "Setting multi_process_env to False.")
            multi_process_env = False
        self.multi_process_env = multi_process_env
        MEnv = DummyVecEnv
        if self.multi_process_env:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            MEnv = SubprocVecEnv
        self.envs = MEnv([env_creator] * self.nbr_env)
        self.seed_input = seed
        self.seed_output = self.envs.seed(seed)
        self.nbr_actions = self.envs.action_space.n

        self.batch_size = env_step_per_batch * self.nbr_env
        self.replay_buffer = replay_buffer
        self.sub_belief_updater = sub_belief_updater

        self.activation_fn = activation_fn
        self.hidden_layer_sizes = hidden_layer_sizes

        # Variable allocations
        self._current_state_obs: Optional[Dict[str, np.ndarray]] = None
        self._prev_state_obs: Optional[Dict[str, np.ndarray]] = None
        self._prev_sub_belief: Optional[np.ndarray] = np.zeros(
            (self.nbr_env,) + self.replay_buffer.elements_scheme[SUB_BELIEF].shape,
            self.replay_buffer.elements_scheme[SUB_BELIEF].dtype)
        self._prev_action: Optional[np.ndarray] = np.zeros((self.nbr_env,) + self.envs.action_space.shape,
                                                           dtype=self.envs.action_space.dtype)

        self._current_state_obs_img: Optional[Dict[str, np.ndarray]] = None
        self._prev_sub_belief_img: Optional[np.ndarray] = np.zeros(
            (self.nbr_env,) + self.replay_buffer.elements_scheme[SUB_BELIEF].shape,
            self.replay_buffer.elements_scheme[SUB_BELIEF].dtype)
        self._prev_action_img: Optional[np.ndarray] = np.zeros((self.nbr_env,) + self.envs.action_space.shape,
                                                           dtype=self.envs.action_space.dtype)
        self._timestep_img = None

        self._episode_id: Optional[np.ndarray] = None
        self._n_reset_state: Optional[np.ndarray] = None
        self._timestep: Optional[np.ndarray] = None
        self._env_has_been_reset = False

        self.network_target_pairs = []
        self.target_update_freq = target_update_freq
        self.tau = tau

        # self.policy_model = None

        self._explore_strategy = None
        self._epsilon = 1.
        self._epsilon_decay = .9995
        self._epsilon_min = .05
        assert isinstance(self.envs.action_space, gym.spaces.Discrete)
        self.n_actions = self.envs.action_space.n
        self.optimizer: Optimizer = Adam(learning_rate=learning_rate)
        self._n_train_calls = 0
        self._t_env = 0
        self._cum_rewards = np.zeros((self.nbr_env,))
        self._cum_rewards_undis = np.zeros((self.nbr_env,))
        self._env_steps = 0
        self._n_episode = 0
        self.models = []
        self.assets = ['_epsilon']
        self._debug_use_state = _debug_use_state
        self._debug_best_action = _debug_best_action
        self.env_img = env_img

    def _first_env_reset(self):
        """
        Method to reset the environment and initialize the variables.
        """
        assert not self._env_has_been_reset
        self._env_has_been_reset = True
        self._current_state_obs = self.envs.reset()
        self._prev_state_obs = {
            STATE: np.repeat(self.replay_buffer._reset_obs_state[STATE].copy(), self.nbr_env, 0),
            OBS: np.repeat(self.replay_buffer._reset_obs_state[OBS].copy(), self.nbr_env, 0),
        }
        self._episode_id = np.array([self.replay_buffer.new_episode(1) for _ in range(self.nbr_env)])
        self._timestep = np.zeros_like(self._episode_id)
        self._n_reset_state = np.ones_like(self._episode_id)
        # self._episode_running_stats = {ep_id: episode_stat_empty.copy() for ep_id in self._episode_id}
        self._prev_action[:] = self.replay_buffer.default_prev_action
        state, obs = self._current_state_obs['state'], self._current_state_obs['obs']
        self.replay_buffer.add(
            {
                EPISODE_NBR: self._episode_id,
                TIMESTEP: self._timestep + self._n_reset_state,
                STATE: state,
                OBS: obs,
            }
        )

    def evaluate(self, n: int):
        return [self._evaluate() for _ in range(n)]

    def restart_all_envs_next_interact(self):
        self._env_has_been_reset = False

    def _evaluate(self):
        env = self._evaluate_env
        state_obs = env.reset()
        # state, obs = state_obs["state"], state_obs["obs"]
        prev_action = np.zeros((1,) + self.envs.action_space.shape,
                               dtype=self.envs.action_space.dtype)
        prev_sub_belief = np.zeros(
            (1,) + self.replay_buffer.elements_scheme[SUB_BELIEF].shape,
            self.replay_buffer.elements_scheme[SUB_BELIEF].dtype)
        first_timestep = np.array([True])
        done = False
        cum_rewards = 0
        i = 0
        obs = state_obs[OBS]
        values = []
        while not done:
            sub_belief = self.sub_belief_updater(
                obs=obs,
                prev_action=prev_action,
                prev_sub_belief=prev_sub_belief,
                first_timestep=first_timestep,
            )
            if self._debug_use_state:
                model_output, value = self.compute_policy_value(state_obs[STATE])
            else:
                model_output, value = self.compute_policy_value(sub_belief)
            values.append(array_to_np(value))
            model_output_np = array_to_np(model_output)
            actions = self.distribution(model_output_np).sample()
            new_state_obs, rewards, dones, info = env.step(actions)
            cum_rewards += self.gamma ** i * rewards[0]
            first_timestep = np.zeros_like(first_timestep)
            done = dones[0]
            obs = new_state_obs[OBS]
            prev_action = actions
            prev_sub_belief = sub_belief
            i += 1
        values = np.stack(values).flatten()
        stats = {
            "cum_rewards": cum_rewards,
            "value_max": np.max(values).item(),
            "value_mean": np.mean(values).item(),
            "value_min": np.min(values).item(),
            "timestep": i,
        }
        return stats

    def interact(self, n: int, training: bool = True, force_random_action: bool = False, fill_replay_buffer: Optional[bool] = None):
        """
        Performs n interaction between the agent and the environments.

        :param n: Number of interactions
        :param training: If true exploration is activated and interaction will be saved to the replay buffer.
        """
        episode_ended_stats = []
        for _ in range(n):
            data, episode_ended_stats_ = self._interact(training, force_random_action, fill_replay_buffer)
            if episode_ended_stats_:
                episode_ended_stats.append(episode_ended_stats_)
        # logger.warning(episode_ended_stats)
        return episode_ended_stats



    def _interact(self, training: bool = True, force_random_action: bool = False) -> Tuple[DictArray, List[Dict]]:
        episode_ended_stats = []

        if not self._env_has_been_reset:
            self._first_env_reset()
        state, obs = self._current_state_obs[STATE], self._current_state_obs[OBS]
        first_timestep = self._timestep == 0

        if any(first_timestep):
            assert np.all(self._prev_action[first_timestep] == self.replay_buffer.default_prev_action), self._prev_action[first_timestep]
            assert np.all(self._prev_sub_belief[first_timestep] == 0.), self._prev_sub_belief[first_timestep]

        # The reset_state is known before hand no need to pass it
        sub_belief = self.sub_belief_updater(obs=obs,
                                             prev_action=self._prev_action,
                                             prev_sub_belief=self._prev_sub_belief,
                                             first_timestep=first_timestep,
                                             )
        if tf.reduce_any(tf.math.logical_or(tf.math.is_inf(sub_belief), tf.math.is_nan(sub_belief))):
            logger.warning("RL worker got nan from sub_belief_encoder")
            logger.warning(self._get_weights)
            logger.warning(f"sub_belief"
                           f"; {sub_belief}")
        if self._debug_best_action:
            actions = state[:, 1:5].argmax(-1)
        else:
            if self._debug_use_state:
                model_output = self.policy_model(state)
            else:
                model_output = self.policy_model(sub_belief)
            model_output_np = array_to_np(model_output)
            if np.any(np.isnan(model_output_np)):
                logger.warning("RL worker got nan computing prob logits.")
                logger.warning(f"sub_belief {sub_belief.numpy()}")
                logger.warning(f"weights: {self._get_weights}")
                logger.warning(f"model_output_np: {model_output_np}")
            if training:
                actions = self.distribution(model_output_np).sample()
                actions = self.explore(actions, force_random_action)
            else:
                actions = self.distribution(model_output_np).sample()  
        new_state_obs, rewards, dones, info = self.envs.step(actions)

        self._prev_action = actions

        sub_belief = array_to_np(sub_belief)

        data = {
            EPISODE_NBR: np.copy(self._episode_id),
            TIMESTEP: np.copy(self._timestep + self._n_reset_state),
            SUB_BELIEF: np.copy(sub_belief),
            ACTION: np.copy(actions),
            REWARD: np.copy(rewards),
            DONE: np.copy(dones),
        }

        self._cum_rewards += self.gamma ** self._timestep * rewards
        self._cum_rewards_undis += rewards

        # Adding elements of the current timestep
        self.replay_buffer.add(data)

        data[NEXT_OBS] = np.copy(new_state_obs[OBS])
        data[NEXT_SUB_BELIEF] = np.zeros_like(sub_belief)
        data[OBS] = np.copy(obs)
        data[NEXT_STATE] = np.copy(new_state_obs[STATE])
        data[STATE] = np.copy(state)
        data['is_first'] = self._timestep == 0

        # Adding elements of the next timestep and get new episode number.
        if sum(dones):
            new_state = np.stack([info[i]['terminal_observation'][STATE] for i in dones.nonzero()[0]], 0)
            new_obs = np.stack([info[i]['terminal_observation'][OBS] for i in dones.nonzero()[0]], 0)
            reset_state_count = np.array([info[i]['reset_state_count'] for i in dones.nonzero()[0]])

            new_sub_belief = self.sub_belief_updater(
                obs=new_obs,
                prev_action=actions[dones],
                prev_sub_belief=sub_belief[dones],
                first_timestep=self._timestep[dones] == 0
            )

            new_sub_belief = array_to_np(new_sub_belief)

            last_episode_data = {
                EPISODE_NBR: self._episode_id[dones],
                TIMESTEP: self._timestep[dones] + self._n_reset_state[dones] + 1,
                STATE: np.copy(new_state),
                OBS: np.copy(new_obs),
                SUB_BELIEF: np.copy(new_sub_belief),
            }

            self.replay_buffer.add(last_episode_data, last=True)

            data[NEXT_OBS][dones] = np.copy(new_obs)
            # if self._debug_use_state:
            data[NEXT_STATE][dones] = np.copy(new_state)
            data[NEXT_SUB_BELIEF][dones] = np.copy(new_sub_belief)

            episode_ended_stats = [
                {
                    'episode_id': int(ep_id),
                    'timestep': int(t) + 1,
                    'cum_reward': float(r),
                    'cum_reward_undiscounted': float(ru),
                }
                for ep_id, t, r, ru in zip(self._episode_id[dones],
                                           self._timestep[dones],
                                           self._cum_rewards[dones],
                                           self._cum_rewards_undis[dones])
            ]

            new_ep_nbrs = []
            for ep_nbr, rs_count in zip(self._episode_id[dones], reset_state_count):
                self.replay_buffer.episode_end(ep_nbr)
                new_ep_nbrs.append(self.replay_buffer.new_episode(reset_state_count=rs_count))
            self._episode_id[dones] = new_ep_nbrs
            self._cum_rewards[dones] = 0
            self._cum_rewards_undis[dones] = 0
            self._timestep[dones] = 0
            self._n_reset_state[dones] = reset_state_count
            self._prev_action[dones] = 0
            sub_belief[dones] = 0
            self._n_episode += sum(dones)
            self._current_state_obs[STATE][dones] = self.replay_buffer._reset_obs_state[STATE].copy()
            self._current_state_obs[OBS][dones] = self.replay_buffer._reset_obs_state[OBS].copy()

        self._timestep[~dones] += 1
        self._prev_sub_belief = sub_belief
        self._prev_state_obs = self._current_state_obs.copy()
        self._current_state_obs = new_state_obs
        state, obs = self._current_state_obs['state'], self._current_state_obs['obs']
        self.replay_buffer.add(
            {
                EPISODE_NBR: self._episode_id,
                TIMESTEP: self._timestep + self._n_reset_state,
                STATE: np.copy(state),
                OBS: np.copy(obs),
            }
        )

        if training:
            self._epsilon = max(self._epsilon * self._epsilon_decay, self._epsilon_min)

        self._env_steps += self.nbr_env

        return data, episode_ended_stats


    def explore(self, actions, force_random_action: bool = False):
        if self._explore_strategy is None and not force_random_action:
            return actions
        elif self._explore_strategy == 'epsilon_greedy' or force_random_action:
            random_actions = np.random.random(len(actions)) < max(self._epsilon, int(force_random_action))
            explore_actions = np.random.randint(0, self.n_actions, actions.shape)
            explore_actions = np.where(random_actions, explore_actions, actions)
            return explore_actions
        else:
            raise ValueError(f"Unknown exploration strategy {self._explore_strategy}.")

    @property
    def epsilon(self):
        return self._epsilon

    def update_target_networks(self):
        assert 0 < self.tau <= 1
        for network, target in self.network_target_pairs:
            if self.tau >= 1:
                target.set_weights(network.get_weights())
            else:
                target.set_weights(
                    [self.tau * target_weight + (1 - self.tau) * network_weight for network_weight, target_weight in
                     zip(network.get_weights(), target.get_weights())])

    def train(self):
        raise NotImplementedError()

    def _get_weights(self):
        raise NotImplementedError()

    @property
    def env_steps(self):
        return self._env_steps
    @property
    def n_episode(self):
        return self._n_episode

    @property
    def n_train_called(self):
        return self._n_train_calls

    def save(self, path: str):
        assert os.path.exists(path), f"Path {path} does not exist"
        for key in self.models:
            model = getattr(self, key)
            model.save_weights(os.path.join(path, f"rl_worker_{key}.h5"))
        optimizer_weights = self.optimizer.get_weights()
        np.save(os.path.join(path, 'rl_worker_optimizer_weights.npy'), optimizer_weights, allow_pickle=True)

    def load(self, path: str):
        assert os.path.exists(path), f"Path {path} does not exist"
        for key in self.models:
            model = getattr(self, key)
            model.load_weights(os.path.join(path, f"rl_worker_{key}.h5"))
        optimizer_weights = np.load(os.path.join(path, 'rl_worker_optimizer_weights.npy'), allow_pickle=True)
        if self.optimizer.iterations != 0:
            self.optimizer.set_weights(optimizer_weights)
        else:
            logger.warning("Optimizer was not loaded.")
