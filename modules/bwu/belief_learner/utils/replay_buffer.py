from typing import Callable, Optional

import gym

from belief_learner.utils.definitions import OBS, STATE, ACTION, SUB_BELIEF


def elements_scheme_builder(env_maker: Callable[[dict], gym.Env], env_config: dict,
                            env: Optional[gym.Env] = None,
                            sub_belief_space: Optional[gym.Space] = None):
    if env is None:
        env_config = env_config.copy()
        env = env_maker(**env_config)

    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert OBS in list(env.observation_space.keys())
    assert STATE in list(env.observation_space.keys())

    if sub_belief_space is None:
        sub_belief_space = env.observation_space[OBS]

    elements_scheme = {
        OBS: env.observation_space[OBS],
        STATE: env.observation_space[STATE],
        ACTION: env.action_space,
        SUB_BELIEF: sub_belief_space,
    }
    return elements_scheme
