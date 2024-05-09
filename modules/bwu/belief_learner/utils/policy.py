import numpy as np
import tensorflow as tf


def sub_belief_updater_pass(obs: np.ndarray,
                            prev_action: np.ndarray,
                            prev_sub_belief: np.ndarray,
                            first_timestep: bool = False,
                            ):
    return obs

def compute_gae(reward: np.ndarray, next_value: np.ndarray, done: np.ndarray, gamma: float, lambda_: float):
    y = np.zeros_like(next_value)
    not_done = 1. - done
    y[-1] = reward[-1] + not_done[-1] * gamma * next_value[-1]
    for t in range(y.shape[0] - 2, -1, -1):
        y[t] = reward[t] + not_done[t] * ((1 - lambda_) * gamma * next_value[t] + gamma * y[t + 1])
    return y


# @tf.function
def compute_gae_tf(reward, next_value, done, gamma, lambda_):
    not_done = 1. - tf.cast(done, tf.float32)
    reward = tf.cast(reward, tf.float32)
    y = []
    y.append(reward[-1] + not_done[-1] * gamma * next_value[-1])
    for t in range(next_value.shape[0] - 2, -1, -1):
        y.append(reward[t] + not_done[t] * ((1 - lambda_) * gamma * next_value[t] + gamma * y[-1]))
    y = y[::-1]
    y = tf.stack(y,)
    y = tf.cast(y, tf.float32)
    return y


def compute_gae_tf2(reward, value, next_value, done, gamma, lambda_,):
    not_done = 1. - tf.cast(done, tf.float32)
    reward = tf.cast(reward, tf.float32)

    advantage = []
    advantage.append(reward[-1] + (gamma * next_value[-1] * not_done[-1]) - value[-1])
    for t in range(next_value.shape[0] - 2, -1, -1):
        delta = reward[t] + (gamma * next_value[t] * not_done[t]) - value[t]
        advantage.append(delta + (gamma * lambda_ * advantage[-1] * not_done[t]))
    advantage = advantage[::-1]
    advantage = tf.stack(advantage)
    advantage = tf.cast(advantage, tf.float32)
    return advantage

@tf.function
def compute_gae_tf3(reward, value, next_value, done, gamma, lambda_):
    not_done = 1. - tf.cast(done, tf.float32)
    reward = tf.cast(reward, tf.float32)
    delta = reward + (gamma * next_value * not_done) - value
    advantage = [delta[-1]]
    for t in range(next_value.shape[0] - 2, -1, -1):
        advantage.append(delta[t] + (gamma * lambda_ * advantage[-1] * not_done[t]))
    advantage = advantage[::-1]
    advantage = tf.stack(advantage)
    advantage = tf.cast(advantage, tf.float32)
    return advantage