from typing import Optional, Callable, Union, List

import numpy as np
import tensorflow as tf

from belief_learner import ReplayBuffer
from belief_learner.utils.definitions import SUB_BELIEF, REWARD, DONE, ACTION, NEXT_OBS
from belief_learner.utils.distributions import DeterministicDistribution
from belief_learner.networks.dense import generate_rl_model
from belief_learner.workers.rl_worker.rl_worker import RLWorker


class DQNWorker(RLWorker):
    def __init__(self,
                 double_dqn: bool,
                 dueling_dqn: bool,
                 activation_fn: Union[str, Callable],
                 hidden_layer_sizes: List[int],
                 target_update_freq: Union[int, float],
                 env_creator: Callable,
                 env_config: dict,
                 nbr_environments: int,
                 multi_process_env: bool,
                 sub_belief_updater, seed: Optional[int],
                 replay_buffer: ReplayBuffer,
                 env_step_per_batch: int,
                 optimizer_name: str,
                 learning_rate: float,
                 tau: float,
                 gamma: float,
                 ):

        super().__init__(env_creator=env_creator,
                         env_config=env_config,
                         nbr_environments=nbr_environments,
                         multi_process_env=multi_process_env,
                         sub_belief_updater=sub_belief_updater,
                         seed=seed,
                         replay_buffer=replay_buffer,
                         optimizer_name=optimizer_name,
                         target_update_freq=target_update_freq,
                         learning_rate=learning_rate,
                         tau=tau,
                         gamma=gamma,
                         env_step_per_batch=env_step_per_batch,
                         activation_fn=activation_fn,
                         hidden_layer_sizes=hidden_layer_sizes)
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.nbr_actions = self.envs.action_space.n

        self.dqn_model = generate_rl_model(
            input_shape=self.replay_buffer.elements_scheme[SUB_BELIEF].shape,
            dueling=self.dueling_dqn,
            output_units=self.nbr_actions,
            activation_fn=self.activation_fn,
            hidden_layer_sizes=self.hidden_layer_sizes,
        )

        self.target_dqn_model = generate_rl_model(
            input_shape=self.replay_buffer.elements_scheme[SUB_BELIEF].shape,
            dueling=self.dueling_dqn,
            output_units=self.nbr_actions,
            activation_fn=self.activation_fn,
            hidden_layer_sizes=self.hidden_layer_sizes,
        )

        self.target_dqn_model.set_weights(self.dqn_model.get_weights())
        self.distribution = DeterministicDistribution
        self.policy_model = self.dqn_model
        self._explore_strategy = 'epsilon_greedy'
        self.network_target_pairs.append((self.dqn_model, self.target_dqn_model))

    def train(self):
        self._n_train_calls += 1
        batch = self.replay_buffer.sample('rl', self.batch_size)
        sub_belief = batch[SUB_BELIEF]
        reward = batch[REWARD]
        action = batch[ACTION]
        next_obs = batch[NEXT_OBS]
        done = batch[DONE]
        next_sub_belief = self.sub_belief_updater(
            obs=next_obs,
            prev_action=action,
            prev_sub_belief=sub_belief,
            first_timestep=np.zeros_like(done),
        )

        target_q_value_tp1 = self.target_dqn_model(next_sub_belief)
        if self.double_dqn:
            q_value_tp1 = self.policy_model(next_sub_belief)
            next_action = tf.math.argmax(q_value_tp1, -1)
        else:
            next_action = tf.math.argmax(target_q_value_tp1, -1)

        target_q_value_tp1_selected = tf.reduce_sum(tf.one_hot(next_action, self.n_actions) * target_q_value_tp1, -1)
        target = reward + self.gamma * (1 - done) * target_q_value_tp1_selected
        target = tf.stop_gradient(target)

        with tf.GradientTape() as tape:
            q_value = self.policy_model(sub_belief)
            q_value_selected = tf.reduce_sum(tf.one_hot(action, self.n_actions) * q_value, -1)

            td_error = (q_value_selected - target)
            loss = .5 * tf.reduce_mean(tf.square(td_error))
            if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                print('error')

        grads = tape.gradient(loss, self._get_weights())
        self.optimizer.apply_gradients(zip(grads, self._get_weights()))

        if self._n_train_calls % self.target_update_freq == 0:
            print('update_target_networks')
            self.update_target_networks()

        stats = {
            'loss': loss.numpy(),
            'n_train_call': self._n_train_calls,
        }

        return stats

    def _get_weights(self):
        return self.policy_model.weights
