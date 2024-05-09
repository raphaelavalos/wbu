from collections import namedtuple
from typing import Optional, Callable, Dict, Any
import time

import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import tf_policy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.typing.types import Float, Bool
from tf_agents.utils import common
import tensorflow_probability.python.distributions as tfd

from belief_learner.utils.old.reinforcement_learning import metrics
from belief_learner.utils.old.io.dataset_generator import map_rl_trajectory_to_vae_input, \
    ergodic_batched_labeling_function, is_reset_state
from belief_learner.utils.old.reinforcement_learning.environments.latent_environment import LatentEmbeddingTFEnvironmentWrapper
from belief_learner.verification import binary_latent_space
from belief_learner.verification.model import TransitionFrequencyEstimator, TransitionFunctionCopy, RewardFunctionCopy
from belief_learner.verification.value_iteration import value_iteration, vi_tensor_size


def estimate_local_losses_from_samples(
        environment: TFPyEnvironment,
        latent_policy: tf_policy.TFPolicy,
        steps: int,
        latent_state_size: int,
        number_of_discrete_actions: int,
        state_embedding_function: Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor],
        action_embedding_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        latent_reward_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        labeling_function: Callable[[tf.Tensor], tf.Tensor],
        latent_transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution] = None,
        estimate_transition_function_from_samples: bool = False,
        assert_transition_distribution: bool = False,
        fill_in_replay_buffer: bool = True,
        replay_buffer_max_frames: int = int(1e5),
        reward_scaling: Optional[float] = 1.,
        atomic_prop_dims: Optional[int] = None,
        probabilistic_state_embedding: Optional[Callable[[tf.Tensor, tf.Tensor], tfd.Distribution]] = None,
        estimate_value_difference: bool = False,
        gamma: float = 0.99,
        epsilon: float = 1e-6,
        memory_limit: int = int(4.5e9),
):
    """
    Estimates reward and probability local losses from samples.

    Args:
        environment: batched TFPyEnvironment
        latent_policy: a policy sets over latent states and producing (one-hot) latent actions
        steps: number of environment steps used to perform the estimation
        latent_state_size: binary size of the latent space
        number_of_discrete_actions: number of discrete actions
        state_embedding_function: mapping from real states and labels to binary states (of type int32)
        action_embedding_function: mapping from real states and discrete actions (given in base 10 with type int32)
                                   to real actions.
                                   Important: latent actions are here assumed to be given in base 10 and not in
                                   one-hot since environments generally process discrete actions represented in
                                   base 10 by convention.
        latent_reward_function: mapping from (binary) latent state, (one-hot) latent actions and (binary) next latent
                                state to rewards. Latent states and actions are assumed to be of type int32 while
                                returned reward type is float32.
        labeling_function: labeling function (mapping from real states to subset of atomic propositions)
        latent_transition_function: mapping from (binary) latent states and (one-hot) actions to
                                    a probability distribution over latent states.
                                    A probability distribution object is assumed to have a prob method (e.g.,
                                    tensorflow probability distribution objects) such that
                                    distribution.prob(next_state) is the probability of transitioning to the next
                                    latent state
        estimate_transition_function_from_samples: whether to estimate the latent transition function from samples
                                                   or not. If True, the latent transition function estimated this
                                                   way stores the probability matrix into a sparse tensor.
        assert_transition_distribution: whether to assert the transition function is correctly computed or not
        fill_in_replay_buffer: whether to fill in the replay buffer or not before the loss estimation.
        replay_buffer_max_frames: maximum number of frames to be contained in the replay buffer.
        atomic_prop_dims: number of atomic propositions; in other words the rightmost label shape.
        probabilistic_state_embedding: underlying (conditional) distribution of the state embedding function.
                                       If provided, this distribution will be used for estimating the local
                                       transition loss.
    Returns: a namedtuple containing an estimation of the local reward and probability losses
    """
    time_metrics = dict()
    time_metrics['start'] = time.time()

    if latent_transition_function is None and not estimate_transition_function_from_samples:
        raise ValueError('no latent transition function provided')

    # scale the reward function
    _latent_reward_fn = latent_reward_function
    latent_reward_function = lambda _state, _action, _next_state: (
            reward_scaling * _latent_reward_fn(_state, _action, _next_state))
    # generate environment wrapper for discrete actions
    latent_environment = LatentEmbeddingTFEnvironmentWrapper(
        tf_env=environment,
        state_embedding_fn=state_embedding_function,
        action_embedding_fn=action_embedding_function,
        labeling_fn=labeling_function,
        latent_state_size=latent_state_size,
        number_of_discrete_actions=number_of_discrete_actions,
        reward_scaling=reward_scaling)
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
        max_length=replay_buffer_max_frames,
        # to retrieve all the transitions when single_deterministic_pass is True
        dataset_window_shift=1,
        dataset_drop_remainder=True)
    # initialize driver
    observers = [replay_buffer.add_batch]
    if estimate_value_difference:
        observers.append(metrics.AverageDiscountedReturnMetric(
            gamma=gamma,
            batch_size=latent_environment.batch_size))
    driver = DynamicStepDriver(
        env=latent_environment,
        policy=policy,
        num_steps=replay_buffer_max_frames if fill_in_replay_buffer else steps,
        observers=observers)
    driver.run = common.function(driver.run)
    # collect environment steps
    driver.run()
    
    time_metrics['collect'] = time.time() - time_metrics['start']

    # retrieve dataset from the replay buffer
    generator = lambda trajectory, _: map_rl_trajectory_to_vae_input(
        trajectory=trajectory,
        include_latent_states=True,
        discrete_action=True,
        num_discrete_actions=number_of_discrete_actions,
        labeling_function=ergodic_batched_labeling_function(labeling_function))

    def _sample_from_replay_buffer(num_transitions: Optional[int] = None, single_deterministic_pass=False):
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            num_steps=2,
            # whether to gather transitions only once or not
            single_deterministic_pass=single_deterministic_pass,
        ).map(
            map_func=generator,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).batch(
            batch_size=num_transitions if num_transitions else replay_buffer.num_frames(),
            drop_remainder=False)
        dataset_iterator = iter(dataset)

        state, label, latent_state, latent_action, reward, next_state, next_label, next_latent_state = \
            next(dataset_iterator)

        return namedtuple(
            'ErgodicMDPTransitionSample',
            ['state', 'label', 'latent_state', 'latent_action', 'reward', 'next_state', 'next_label',
             'next_latent_state', 'batch_size'])(
            state, label, latent_state, latent_action, reward, next_state, next_label,
            next_latent_state, tf.shape(state)[0])
    
    time_metrics['local_reward_loss'] = time.time()

    samples = _sample_from_replay_buffer(num_transitions=steps)
    (state, label, latent_state, latent_action, reward, next_state, next_label,
     next_latent_state) = (
        samples.state, samples.label, samples.latent_state, samples.latent_action, samples.reward,
        samples.next_state, samples.next_label, samples.next_latent_state)

    if atomic_prop_dims is None:
        atomic_prop_dims = latent_state_size - tf.shape(
            state_embedding_function(state, None)
        )[-1]

    local_reward_loss = estimate_local_reward_loss(
        state, label, latent_action, reward, next_state, next_label,
        latent_reward_function, latent_state, next_latent_state, )

    time_metrics['local_reward_loss'] = time.time() - time_metrics['local_reward_loss']

    if estimate_transition_function_from_samples:
        
        time_metrics['empirical_model_generation'] = time.time()
        _samples = _sample_from_replay_buffer(single_deterministic_pass=True)
        _transition_function_estimation_num_frames = _samples.batch_size.numpy()
        empirical_latent_transition_function = TransitionFrequencyEstimator(
            _samples.latent_state, _samples.latent_action, _samples.next_latent_state,
            backup_transition_function=latent_transition_function,
            assert_distribution=assert_transition_distribution, )

        time_metrics['empirical_model_generation'] = time.time() - time_metrics['empirical_model_generation']
    else:
        empirical_latent_transition_function = None
        _transition_function_estimation_num_frames = 0
        time_metrics['transition_function_estimation'] = 0
    
    time_metrics['local_transition_loss'] = time.time()

    local_transition_loss = estimate_local_transition_loss(
        state, label, latent_action, next_state, next_label,
        latent_transition_function, latent_state, next_latent_state,
        probabilistic_state_embedding=probabilistic_state_embedding)

    time_metrics['local_transition_loss'] = time.time() - time_metrics['local_transition_loss']

    time_metrics['local_transition_loss_2'] = time.time()
    if empirical_latent_transition_function is not None:
        local_transition_loss_transition_fn_estimation = estimate_local_transition_loss(
            state, label, latent_action, next_state, next_label,
            empirical_latent_transition_function, latent_state, next_latent_state,
            probabilistic_state_embedding=probabilistic_state_embedding)
    else:
        local_transition_loss_transition_fn_estimation = None

    time_metrics['local_transition_loss_2'] = time.time() - time_metrics['local_transition_loss_2']
    
    value_diff = dict()
    if estimate_value_difference:
        
        time_metrics['model_generation'] = time.time()
        # Model generation
        _latent_transition_fn = latent_transition_function
        latent_transition_function = TransitionFunctionCopy(
            num_states=tf.cast(tf.pow(2, latent_state_size), dtype=tf.int32),
            num_actions=number_of_discrete_actions,
            transition_function=latent_transition_function,
            epsilon=epsilon)
        latent_reward_function = RewardFunctionCopy(
            num_states=tf.cast(tf.pow(2, latent_state_size), dtype=tf.int32),
            num_actions=number_of_discrete_actions,
            reward_function=latent_reward_function,
            transition_function=_latent_transition_fn,
            epsilon=epsilon)
        time_metrics['model_generation'] = time.time() - time_metrics['model_generation']

        transition_functions = {"latent_transition_function": latent_transition_function}
        if empirical_latent_transition_function is not None:
            _model_gen_time = time.time()
            # empirical_latent_transition_function.merge(
            #     latent_transition_function, epsilon)
            _to_dense = empirical_latent_transition_function.to_dense
            empirical_latent_transition_function.to_dense = lambda: _to_dense(
                backup_tensor=latent_transition_function.to_dense())
            time_metrics['empirical_model_generation'] += time.time() - _model_gen_time
            transition_functions["empirical_latent_transition_function"] = empirical_latent_transition_function
        for name, transition_fn in transition_functions.items():
            time_metrics['value_diff_' + name] = time.time()
            # memory handling 
            try:
                memory_used = tf.config.experimental.get_memory_info('GPU:0')['current']
            except ValueError as error:
                memory_used = memory_limit
            device = 'GPU' if memory_used + vi_tensor_size(
                    latent_state_size,
                    number_of_discrete_actions,
                    episodic_return=True
            ) < memory_limit else 'CPU'
            if memory_used < memory_limit and device == 'CPU':
                print("[VI] Tensor computation switched to CPU for value iteration.")
            with tf.device('/{}:0'.format(device)):
                values = compute_values_from_initial_distribution(
                    latent_state_size=latent_state_size,
                    atomic_prop_dims=atomic_prop_dims,
                    original_state=state,
                    number_of_discrete_actions=number_of_discrete_actions,
                    latent_policy=latent_policy,
                    latent_transition_fn=transition_fn,
                    latent_reward_function=latent_reward_function,
                    epsilon=epsilon,
                    gamma=gamma,
                    stochastic_state_embedding=(
                        lambda _state: probabilistic_state_embedding(
                            _state, ergodic_batched_labeling_function(labeling_function)(_state))
                    ) if probabilistic_state_embedding else (
                        lambda _state: tfd.Independent(
                            tfd.Deterministic(loc=state_embedding_function(
                                _state, ergodic_batched_labeling_function(labeling_function)(_state))),
                            reinterpreted_batch_ndims=1)), )
                value_diff["value_diff_" + name] = tf.abs(
                    observers[-1].result() - values)
                time_metrics['value_diff_' + name] = time.time() - time_metrics['value_diff_' + name]

    def print_time_metrics():
        print("Time metrics:")

        def replace(_dict: Dict[str, Any], key_dict: Dict[str, str]):
            _new_dict = dict()
            for old_key, value in _dict.items():
                new_key = key_dict.get(old_key, None)
                if new_key is not None:
                    _new_dict[new_key] = value
                else:
                    _new_dict[old_key] = value
            return _new_dict

        for _name, _time in replace(
                time_metrics,
                {
                    'collect':
                        "Fill in the Replay Buffer ({:d} frames)".format(replay_buffer_max_frames),
                    'local_reward_loss':
                        "Estimate the local reward loss function (from {:d} transitions)".format(steps),
                    'local_transition_loss':
                        "Estimate the local transition loss function (from {:d} transitions)".format(steps),
                    'transition_function_estimation':
                        "Time to build the transition function via frequency estimation"
                        " (from {:d} transitions)".format(_transition_function_estimation_num_frames),
                    'local_transition_loss_2':
                        "Estimate the local transition loss function via the frequency"
                        "-estimated transition function:".format(steps),
                    'value_diff_latent_transition_function': "Value difference",
                    "value_diff_empirical_latent_transition_function":
                        "Value difference (empirical latent transition function)",
                    "model_generation": "Model generation",
                    "empirical_model_generation":
                        "Transition model generation (empirical frequency estimation, from {:d} transitions)".format(
                            steps)
                }
        ).items():
            if _name != 'start':
                print("    {}: {:.3f}".format(_name, _time))

    replay_buffer.clear()

    return namedtuple(
        'LocalLossesEstimationMetrics',
        ['local_reward_loss', 'local_transition_loss', 'local_transition_loss_transition_function_estimation',
         'print_time_metrics', 'time_metrics', 'value_difference'])(
        local_reward_loss, local_transition_loss, local_transition_loss_transition_fn_estimation,
        print_time_metrics, time_metrics, value_diff)


@tf.function
def estimate_local_reward_loss(
        state: tf.Tensor,
        label: tf.Tensor,
        latent_action: tf.Tensor,
        reward: tf.Tensor,
        next_state: tf.Tensor,
        next_label: tf.Tensor,
        latent_reward_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        latent_state: Optional[tf.Tensor] = None,
        next_latent_state: Optional[tf.Tensor] = None,
        state_embedding_function: Optional[Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]] = None,
):
    if latent_state is None:
        latent_state = state_embedding_function(state, label)
    if next_latent_state is None:
        next_latent_state = state_embedding_function(next_state, next_label)

    return tf.reduce_mean(
        tf.abs(reward - latent_reward_function(latent_state, latent_action, next_latent_state)))


@tf.function
def estimate_local_transition_loss(
        state: tf.Tensor,
        label: tf.Tensor,
        latent_action: tf.Tensor,
        next_state: tf.Tensor,
        next_label: tf.Tensor,
        latent_transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
        latent_state: Optional[tf.Tensor] = None,
        next_latent_state: Optional[tf.Tensor] = None,
        state_embedding_function: Optional[Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]] = None,
        probabilistic_state_embedding: Optional[Callable[[tf.Tensor, tf.Tensor], tfd.Distribution]] = None
):
    if latent_state is None:
        latent_state = state_embedding_function(state, label)
    if next_latent_state is None:
        next_latent_state = state_embedding_function(next_state, tf.cast(next_label, tf.float32))

    next_latent_state = tf.cast(next_latent_state, tf.float32)

    if probabilistic_state_embedding:
        latent_state_space = binary_latent_space(tf.shape(latent_state)[-1], dtype=tf.float32)

        @tf.function
        def total_variation(transition):
            _latent_state, _latent_action, _next_state, _next_label = transition
            tile = lambda t: tf.tile(
                tf.expand_dims(t, 0),
                [tf.shape(latent_state_space)[0], 1])

            latent_transition_distribution = latent_transition_function(
                tile(_latent_state), tile(_latent_action))
            embedding_distribution = probabilistic_state_embedding(
                tile(_next_state), tile(_next_label))
            return .5 * tf.reduce_sum(
                tf.abs(
                    embedding_distribution.prob(latent_state_space) -
                    latent_transition_distribution.prob(
                        latent_state_space,
                        full_latent_state_space=True)),
                axis=0)

        return tf.reduce_mean(
            tf.map_fn(
                fn=total_variation,
                elems=[latent_state, latent_action, next_state, next_label],
                fn_output_signature=tf.float32))

    else:
        latent_transition_distribution = latent_transition_function(latent_state, latent_action)
        return tf.reduce_mean(1. - latent_transition_distribution.prob(next_latent_state))


@tf.function
def compute_values_from_initial_distribution(
        latent_state_size: int,
        atomic_prop_dims: int,
        original_state: tf.Tensor,
        number_of_discrete_actions: int,
        latent_policy: Callable[[tf.Tensor], tfd.OneHotCategorical],
        latent_transition_fn: TransitionFunctionCopy,
        latent_reward_function: RewardFunctionCopy,
        epsilon: Float,
        gamma: Float,
        stochastic_state_embedding: Callable[[tf.Tensor], tfd.Distribution],
        v_init: Optional[Float] = None,
        absorbing_states: Optional[Callable[[tf.Tensor], Bool]] = None,
):
    latent_state_space = binary_latent_space(latent_state_size)

    p_init = stochastic_state_embedding(
        tf.tile(tf.zeros_like(original_state[:1, ...]), [tf.shape(latent_state_space)[0], 1])
    ).prob(latent_state_space)
    is_reset_state_test_fn = lambda latent_state: is_reset_state(latent_state, atomic_prop_dims)

    values = value_iteration(
        latent_state_size=latent_state_size,
        num_actions=number_of_discrete_actions,
        transition_fn=latent_transition_fn,
        reward_fn=latent_reward_function,
        gamma=gamma,
        policy=PolicyDecorator(latent_policy),
        epsilon=epsilon,
        is_reset_state_test_fn=(
            is_reset_state_test_fn if absorbing_states is None else
            lambda latent_state: tf.logical_or(
                is_reset_state_test_fn(latent_state),
                absorbing_states(latent_state))),
        episodic_return=tf.equal(tf.reduce_max(p_init), 1.) or (absorbing_states is not None),
        error_type='absolute',
        v_init=v_init,
        transition_matrix=latent_transition_fn.to_dense(),
        reward_matrix=latent_reward_function.to_dense(),)

    if tf.equal(tf.reduce_max(p_init), 1.):
        # deterministic reset
        reset_state = stochastic_state_embedding(
            tf.tile(tf.zeros_like(original_state[:1, ...]), [tf.shape(latent_state_space)[0], 1])
        ).sample()
        reset_state = tf.cast(reset_state, tf.float32)

        latent_action_space = tf.one_hot(
            indices=tf.range(number_of_discrete_actions),
            depth=tf.cast(number_of_discrete_actions, tf.int32),
            dtype=tf.float32)

        p_init = tf.reduce_sum(
            tf.transpose(
                PolicyDecorator(latent_policy)(
                    reset_state
                ).probs_parameter()
            ) * tf.map_fn(
                fn=lambda latent_action: latent_transition_fn(
                    reset_state,
                    tf.tile(tf.expand_dims(latent_action, 0), [tf.shape(latent_state_space)[0], 1]),
                ).prob(
                    tf.cast(latent_state_space, tf.float32),
                    full_latent_state_space=True),
                elems=latent_action_space),
            axis=0) * (1. - tf.cast(is_reset_state_test_fn(latent_state_space), tf.float32))

        return tf.reduce_sum(
            p_init * values
        ) / tf.reduce_sum(p_init)

    else:
        return tf.reduce_sum(p_init * values, axis=0)


class PolicyDecorator:
    """
    Decorates a TFPolicy π to retrieve a mapping from states to the space of distributions over actions.
    """

    def __init__(self, policy: tf_policy.TFPolicy):
        self._policy = policy

    def __call__(self, state, *args, **kwargs):
        return self._policy.distribution(
            ts.transition(observation=state, reward=tf.zeros_like(state)), *args, **kwargs
        ).action
