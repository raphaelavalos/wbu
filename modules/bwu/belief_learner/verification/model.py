from collections import namedtuple
from typing import Callable, Optional
import math

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments import TFEnvironment
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.typing.types import Float
from tf_agents.utils import common

from belief_learner.utils.old.reinforcement_learning.environments.latent_environment import LatentEmbeddingTFEnvironmentWrapper
from belief_learner.utils.old.io.dataset_generator import map_rl_trajectory_to_vae_input, ergodic_batched_labeling_function
from belief_learner.verification import binary_latent_space

tfd = tfp.distributions


class TransitionFunction:
    """
    Class representing an MDP transition function where probabilities are encoded via
    a transition matrix (specifically, via a SparseTensor).

    Args:
        transition_matrix: the SparseTensor representing the probability transition matrix;
                           expected size: [S, A, S], where S is the number of states and A the number of actions of
                           the MDP.
        backup_transition_function: backup transition function (mapping state-action pairs to a probability
                                    distribution) that will be used if a state-action pair entry doesn't exist in the
                                    transition matrix.
        assert_distribution: whether to check if the transition matrix is sound or not (the probabilities of
                             each state-action pair entry should sum up to one).
        split_label_from_latent_space: if set, the 'prob' method obtained via __call__ will require 2 input arguments,
                                       in binary: the label of the next latent state, followed by the representation of
                                       the next latent state without the bits of the next latent state.
                                       Concretely, this means the next latent state can be retrieved via:
                                       next_latent_state = tf.concat([label, next_latent_state_no_label], axis=-1)
    """

    def __init__(
            self,
            transition_matrix: tf.sparse.SparseTensor,
            backup_transition_function: Optional[Callable[[tf.Tensor, tf.Tensor], tfd.Distribution]] = None,
            assert_distribution: bool = False,
            split_label_from_latent_space: bool = False
    ):
        self.num_states = transition_matrix.dense_shape[0]
        # using TF operators to compute log2 caused numerical issues on some machines
        self.latent_state_size = int(math.log2(self.num_states))
        self.num_actions = transition_matrix.dense_shape[1]
        self.transitions = transition_matrix
        self.split_label_from_latent_space = split_label_from_latent_space
        self.enabled_actions = tf.cast(
            tf.sparse.reduce_sum(self.transitions, axis=-1, output_is_sparse=True),
            dtype=tf.bool)
        if assert_distribution:
            state_action_pairs = tf.sparse.reduce_sum(self.transitions, axis=-1, output_is_sparse=True)
            tf.debugging.assert_near(state_action_pairs.values, 1.)
        self.backup_transition_function = backup_transition_function
        self.latent_state_space = binary_latent_space(self.latent_state_size, dtype=tf.float32)

    def to_dense(self, *args, **kwargs) -> tf.Tensor:
        """
        Returns: The dense representation of the transition function tensor
        """
        return tf.sparse.to_dense(self.transitions)

    def __call__(self, latent_state: tf.Tensor, latent_action: tf.Tensor):
        """
        Gives the transition distribution formed by the input latent state and latent action.
        The returned distribution is a namedtuple with the following method 'prob':
            prob(next_latent_state, full_latent_state=False), which returns either the probability of
                - the next_latent_state, given in binary;
                - the full matrix entry P(.|latent_state, latent_action) if full_latent_state is set, i.e.,
                  the probability of going to all the next latent states, given the input latent state and actions.
        Args:
            latent_state: the latent state, given in binary (expected size: log_2 S)
            latent_action: the latent action, given in one-hot (expected size: A)

        Returns: the distribution P(. | latent_state, latent_action)
        """
        latent_state = tf.cast(latent_state, tf.int32)
        state = tf.reduce_sum(latent_state * 2 ** tf.range(self.latent_state_size), axis=-1)
        action = tf.argmax(latent_action, axis=-1)
        if self.backup_transition_function is None:
            backup_transition_fn = lambda *args: namedtuple('zeros_backup', ['prob'])(lambda *x: 0.)
        else:
            backup_transition_fn = self.backup_transition_function

        @tf.function
        def _get_prob_value(transition):
            latent_state, state, action, next_latent_state_no_label, next_label = transition
            if self.split_label_from_latent_space:
                next_latent_state = tf.cast(tf.concat([next_label, next_latent_state_no_label], axis=-1), tf.int32)
            else:
                next_latent_state = next_latent_state_no_label
            next_state = tf.reduce_sum(
                tf.cast(next_latent_state, tf.float32) * 2. ** tf.range(self.latent_state_size, dtype=tf.float32),
                axis=-1)

            # check if the action has been visited in the given state during the transition sparse tensor construction
            action_is_enabled = tf.squeeze(
                tf.sparse.to_dense(
                    tf.sparse.slice(self.enabled_actions, [state, action], [1, 1])))

            if action_is_enabled:
                probs = tf.squeeze(tf.sparse.slice(self.transitions, [state, action, next_state], [1, 1, 1]).values)
                # if the entry of the slice is empty, then the probability is 0
                return 0. if tf.equal(tf.size(probs), 0) else probs
            # if not, then use a backup transition function to retrieve the probability distribution for [state, action]
            else:
                if self.split_label_from_latent_space:
                    next_latent_state = [tf.expand_dims(next_label, axis=0),
                                         tf.expand_dims(next_latent_state_no_label, axis=0)]
                else:
                    next_latent_state = [tf.expand_dims(next_latent_state, 0)]
                return tf.squeeze(
                    backup_transition_fn(
                        tf.expand_dims(latent_state, axis=0),
                        tf.expand_dims(tf.one_hot(action, depth=tf.cast(self.num_actions, tf.int32)), axis=0)
                    ).prob(*next_latent_state))

        @tf.function
        def _prob(*value, **kwargs):
            if self.split_label_from_latent_space:
                next_label, next_latent_state_no_label = value
                next_latent_state = tf.concat(next_label, next_latent_state_no_label, axis=-1)
            else:
                next_label = next_latent_state_no_label = next_latent_state = value[0]
            full_latent_state_space = kwargs.get('full_latent_state_space', False)
            if full_latent_state_space:
                if self.split_label_from_latent_space:
                    return _probs_row(next_label, next_latent_state_no_label)
                else:
                    return _probs_row(next_latent_state)
            else:
                return tf.map_fn(
                    fn=_get_prob_value,
                    elems=(latent_state, state, action, next_latent_state_no_label, next_label),
                    fn_output_signature=tf.float32)

        @tf.function
        def _probs_row(*value):
            action_is_enabled = tf.squeeze(tf.sparse.to_dense(tf.sparse.slice(
                self.enabled_actions, [state[0, ...], action[0, ...]], [1, 1])))
            if action_is_enabled:
                return tf.squeeze(tf.sparse.to_dense(tf.sparse.slice(
                    self.transitions, [state[0, ...], action[0, ...], 0], [1, 1, self.num_states])))
            else:
                _distr = backup_transition_fn(
                    latent_state,
                    tf.one_hot(action, depth=tf.cast(self.num_actions, tf.int32)))
                try:
                    return _distr.prob(*value, full_latent_state_space=True)
                except TypeError:
                    return _distr.prob(*value)

        return namedtuple('next_state_transition_distribution', ['prob'])(_prob)


class TransitionFunctionCopy(TransitionFunction):
    """
    Copies a transition function to a transition matrix (encoded through a SparseTensor)

    Args:
        transition_function: mapping from state-action pairs to distributions, implementing the prob method
        epsilon: probability tolerance error
    """

    def __init__(
            self,
            num_states: int,
            num_actions: int,
            transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
            epsilon: float = 1e-6,
            split_label_from_latent_space: bool = False,
            atomic_prop_dims: Optional[int] = None,
    ):
        if split_label_from_latent_space and atomic_prop_dims is None:
            raise ValueError("You need provide atomic_prop_dims if split_label_from_latent_space is set.")
        latent_state_size = int(math.log2(num_states))
        latent_state_space = binary_latent_space(latent_state_size, dtype=tf.float32)
        action_space = tf.one_hot(tf.range(num_actions), depth=tf.cast(num_actions, dtype=tf.int32))
        repeated_action_space = tf.repeat(action_space, repeats=num_states, axis=0)
        state_action_space = tf.tile(latent_state_space, multiples=[num_actions, 1])

        @tf.function
        def _get_sparse_entry(latent_state):
            state = tf.tile(tf.expand_dims(latent_state, 0), [num_states * num_actions, 1])
            action = repeated_action_space
            _distr = transition_function(state, action)
            if split_label_from_latent_space:
                next_latent_states = [
                    state_action_space[..., :atomic_prop_dims],
                    state_action_space[..., atomic_prop_dims:]]
            else:
                next_latent_states = [state_action_space]
            probs = _distr.prob(*next_latent_states)
            # sparsify
            probs = tf.where(
                condition=probs > epsilon,
                x=probs,
                y=tf.zeros_like(probs))
            # normalize
            probs = tf.reshape(probs, [num_actions, num_states])
            probs = probs / tf.reduce_sum(tf.expand_dims(probs, 1), axis=-1)
            indices = tf.where(tf.math.not_equal(probs, tf.zeros_like(probs)))
            values = tf.gather_nd(probs, indices)
            return tf.sparse.SparseTensor(
                indices=indices,
                values=values,
                dense_shape=[num_actions, num_states])

        transitions = tf.map_fn(
            fn=_get_sparse_entry,
            elems=latent_state_space,
            fn_output_signature=tf.SparseTensorSpec(shape=[num_actions, num_states]))

        #  with tf.device('/CPU:0'):
        #      transitions = tf.sparse.reorder(transitions)

        super(TransitionFunctionCopy, self).__init__(
            transition_matrix=transitions,
            split_label_from_latent_space=split_label_from_latent_space,)
        self.atomic_prop_dims = atomic_prop_dims


class RewardFunctionCopy:

    def __init__(
            self,
            num_states: int,
            num_actions: int,
            reward_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], Float],
            transition_function: Optional[Callable[[tf.Tensor, tf.Tensor], tfd.Distribution]] = None,
            split_label_from_latent_space: bool = False,
            atomic_prop_dims: Optional[int] = None,
            epsilon: float = 1e-6,
    ):
        if split_label_from_latent_space and atomic_prop_dims is None:
            raise ValueError("You need provide atomic_prop_dims if split_label_from_latent_space is set.")

        self.latent_state_size = int(math.log2(num_states))
        latent_state_space = binary_latent_space(self.latent_state_size, dtype=tf.float32)
        action_space = tf.one_hot(tf.range(num_actions), depth=tf.cast(num_actions, dtype=tf.int32))
        repeated_action_space = tf.repeat(action_space, repeats=num_states, axis=0)
        state_action_space = tf.tile(latent_state_space, multiples=[num_actions, 1])

        @tf.function
        def _get_sparse_entry(latent_state):
            state = tf.tile(tf.expand_dims(latent_state, 0), [num_states * num_actions, 1])
            action = repeated_action_space
            rewards = tf.squeeze(reward_function(state, action, state_action_space))
            if transition_function is not None:
                _distr = transition_function(state, action)
                if split_label_from_latent_space:
                    next_latent_states = [
                        state_action_space[..., :atomic_prop_dims],
                        state_action_space[..., atomic_prop_dims:]]
                else:
                    next_latent_states = [state_action_space]
                probs = _distr.prob(*next_latent_states)
                rewards = tf.where(
                    condition=probs > epsilon,
                    x=rewards,
                    y=tf.zeros_like(rewards))
            rewards = tf.reshape(rewards, [num_actions, num_states])
            indices = tf.where(tf.greater(tf.abs(rewards), epsilon))
            values = tf.gather_nd(rewards, indices)
            return tf.sparse.SparseTensor(
                indices=indices,
                values=values,
                dense_shape=[num_actions, num_states])

        self.transitions = tf.map_fn(
            fn=_get_sparse_entry,
            elems=latent_state_space,
            fn_output_signature=tf.SparseTensorSpec(shape=[num_actions, num_states]))
        self.num_states = num_states

    def to_dense(self) -> tf.Tensor:
        """
        Returns: The dense representation of the transition function tensor
        """
        return tf.sparse.to_dense(self.transitions)

    @tf.function
    def __call__(self, latent_state: tf.Tensor, latent_action: tf.Tensor, *args, **kwargs):
        """
        Get the full entry for R(latent_state, latent_action, .)
        latent_state, latent_actions are assumed to be batched, but only the first batch element is used to provide
        the entry (the others are ignored).

        Args:
            latent_state: batched binary latent state
            latent_action: batched one-hot encoded action

        Returns: the rewards for all the next states, given latent state and latent action.
        """
        latent_state = tf.cast(latent_state[0, ...], tf.int32)
        state = tf.reduce_sum(latent_state[0, ...] * 2 ** tf.range(self.latent_state_size), axis=-1)
        action = tf.argmax(latent_action[0, ...], axis=-1)

        return tf.squeeze(tf.sparse.to_dense(tf.sparse.slice(
            self.transitions, [state, action, 0], [1, 1, self.num_states])))


class TransitionFrequencyEstimator(TransitionFunction):
    def __init__(
            self,
            latent_states: tf.Tensor,
            latent_actions: tf.Tensor,
            next_latent_states: tf.Tensor,
            backup_transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
            assert_distribution: bool = False,
            split_label_from_latent_space: bool = False
    ):
        latent_states = tf.cast(latent_states, dtype=tf.int32)
        next_latent_states = tf.cast(next_latent_states, dtype=tf.int32)
        latent_state_size = tf.shape(latent_states)[1]  # first axis is batch, second is latent state size
        num_states = 2 ** latent_state_size
        num_actions = tf.shape(latent_actions)[1]  # first axis is batch, second is a one-hot vector

        @tf.function
        def compute_transition_counter():
            states = tf.reduce_sum(latent_states * 2 ** tf.range(latent_state_size), axis=-1)
            actions = tf.cast(tf.argmax(latent_actions, axis=-1), dtype=tf.int32)
            next_states = tf.reduce_sum(next_latent_states * 2 ** tf.range(latent_state_size), axis=-1)

            # flat transition indices
            transitions = states * num_actions * num_states + actions * num_states + next_states
            transitions, _, count = tf.unique_with_counts(transitions)
            transitions = tf.stack([transitions // (num_states * num_actions),  # state index
                                    (transitions // num_states) % num_actions,  # action index
                                    transitions % num_states],  # next state index
                                   axis=-1)
            transitions = tf.cast(transitions, dtype=tf.int64)
            transition_counter = tf.sparse.SparseTensor(
                indices=transitions,
                values=tf.cast(count, tf.float32),
                dense_shape=(num_states, num_actions, num_states))
            return tf.sparse.reorder(transition_counter)

        transition_counter = compute_transition_counter()
        probs = tf.Variable(tf.cast(transition_counter.values, dtype=tf.float32), trainable=False)
        i = tf.Variable(0, trainable=False)
        j = tf.Variable(0, trainable=False)

        @tf.function
        def _compute_transition_probabilities(
                transition_counter: tf.sparse.SparseTensor,
                probs: tf.Variable,
                i: tf.Variable,
                j: tf.Variable):
            state_action_pair_counter = tf.sparse.reduce_sum(transition_counter, axis=-1, output_is_sparse=True)
            indices = transition_counter.indices[..., :-1]
            while i < tf.shape(probs)[0]:
                if tf.reduce_all(indices[i] == state_action_pair_counter.indices[j], axis=-1):
                    probs[i].assign(tf.cast(transition_counter.values[i], dtype=tf.float32) /
                                    tf.cast(state_action_pair_counter.values[j], dtype=tf.float32))
                    i.assign_add(1)
                else:
                    j.assign_add(1)  # works only if indices are ordered

            transition_tensor = tf.sparse.SparseTensor(
                indices=transition_counter.indices,
                values=probs,
                dense_shape=(num_states, num_actions, num_states))

            return tf.sparse.reorder(transition_tensor)

        super(TransitionFrequencyEstimator, self).__init__(
            transition_matrix=_compute_transition_probabilities(transition_counter, probs, i, j),
            backup_transition_function=backup_transition_function,
            split_label_from_latent_space=split_label_from_latent_space)

    @tf.function
    def to_dense(self, backup_tensor: Optional[tf.Tensor] = None, *args, **kwargs) -> tf.Tensor:
        """
        Args:
            backup_tensor: a backup tensor for zero entries
        Returns:
            the dense representation of this transition function
        """
        dense_tensor = tf.sparse.to_dense(self.transitions)
        if backup_tensor is not None:
            tf.debugging.assert_near(tf.reduce_sum(backup_tensor, axis=-1), 1.)
            x = tf.where(
                condition=tf.repeat(
                    tf.expand_dims(tf.sparse.to_dense(self.enabled_actions), -1),
                    repeats=tf.shape(dense_tensor)[-1],
                    axis=-1),
                x=dense_tensor,
                y=backup_tensor)
            return x
        else:
            return dense_tensor

class TransitionFnDecorator:
    """
    Decorates a latent transition function P with a new prob function so that:
    P_new(s' | s, a) = P(l(s'), [s' without label] | s, a)
    with l being the labeling function

    Usage:
    ```python
    decorated_transition_fn = lambda state, action: TransitionFnDecorator(
        next_state_distribution=transition_fn(state, action),
        atomic_prop_dims=atomic_prop_dims)
    ```
    """

    def __init__(self, next_state_distribution, atomic_prop_dims):
        self.next_state_distribution = next_state_distribution
        self.atomic_prop_dims = atomic_prop_dims

    def prob(self, latent_state, *args, **kwargs):
        return self.next_state_distribution.prob(
            latent_state[..., :self.atomic_prop_dims],
            latent_state[..., self.atomic_prop_dims:],
            *args, **kwargs)

def estimate_latent_transition_function_from_samples(
        environment: TFEnvironment,
        n_steps,
        state_embedding_function,
        action_embedding_function,
        labeling_function,
        latent_state_size,
        number_of_discrete_actions,
        latent_policy,
        backup_transition_fn,
):
    latent_environment = LatentEmbeddingTFEnvironmentWrapper(
        tf_env=environment,
        state_embedding_fn=state_embedding_function,
        action_embedding_fn=action_embedding_function,
        labeling_fn=labeling_function,
        latent_state_size=latent_state_size,
        number_of_discrete_actions=number_of_discrete_actions,)
    # set the latent policy over real states
    policy = latent_environment.wrap_latent_policy(latent_policy)
    trajectory_spec = trajectory.from_transition(
        time_step=latent_environment.time_step_spec(),
        action_step=policy.policy_step_spec,
        next_time_step=latent_environment.time_step_spec(),
    )
    # replay_buffer
    replay_buffer = TFUniformReplayBuffer(
        data_spec=trajectory_spec,
        batch_size=latent_environment.batch_size,
        max_length=n_steps,
        # to retrieve all the transitions when single_deterministic_pass is True
        dataset_window_shift=1,
        dataset_drop_remainder=True)
    # initialize driver
    observers = [replay_buffer.add_batch]
    driver = DynamicStepDriver(
        env=latent_environment,
        policy=policy,
        num_steps=n_steps,
        observers=observers)
    driver.run = common.function(driver.run)
    driver.run()

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        num_steps=2,
        # whether to gather transitions only once or not
        single_deterministic_pass=True,
    ).map(
        map_func=lambda trajectory, _: map_rl_trajectory_to_vae_input(
            trajectory=trajectory,
            include_latent_states=True,
            discrete_action=True,
            num_discrete_actions=number_of_discrete_actions,
            labeling_function=ergodic_batched_labeling_function(labeling_function)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).batch(
        batch_size=replay_buffer.num_frames(),
        drop_remainder=False)
    dataset_iterator = iter(dataset)

    _, label, latent_state, latent_action, _, _, next_label, next_latent_state = next(dataset_iterator)

    return TransitionFrequencyEstimator(
        latent_state, latent_action, next_latent_state,
        backup_transition_function=backup_transition_fn,
        assert_distribution=True, )
