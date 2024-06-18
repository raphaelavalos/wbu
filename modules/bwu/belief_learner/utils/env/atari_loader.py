from typing import Tuple, Union, Any, NamedTuple, Dict

import gym
import numpy as np
from gym import spaces

from belief_learner.utils.env.perturbed_env_wrapper import PerturbedGymWrapper

try:
    import cv2  # pytype:disable=import-error

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None

GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]

from belief_learner.utils.logging import get_logger

logger = get_logger(__name__)
# Taken from stable_baselines3

class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: the environment to wrap
    :param noop_max: the maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: the environment to wrap
    """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: the environment to wrap
    """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> GymStepReturn:
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)

    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs) -> GymObs:
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.

    :param env: the environment
    """

    def __init__(self, env: gym.Env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward: float) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: the environment
    :param width:
    :param height:
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=env.observation_space.dtype
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class AtariWrapper(gym.Wrapper):
    """
    Atari 2600 preprocessings

    Specifically:

    * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}

    :param env: gym environment
    :param noop_max: max number of no-ops
    :param frame_skip: the frequency at which the agent experiences the game.
    :param screen_size: resize Atari frame
    :param terminal_on_life_loss: if True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    """

    def __init__(
            self,
            env: gym.Env,
            noop_max: int = 30,
            frame_skip: int = 4,
            screen_size: int = 84,
            terminal_on_life_loss: bool = True,
            clip_reward: bool = True,
    ):
        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)


class AtariStateWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, one_hot_state: bool = False, as_bits: bool = False):
        super().__init__(env)
        assert one_hot_state + as_bits != 2, "one_hot_state and as_bits are exclusive"
        self.one_hot_state = one_hot_state
        self.as_bits = as_bits
        if one_hot_state:
            state_space = gym.spaces.Box(0., 1., (self.env.ale.getRAMSize() * 256,))
        elif as_bits:
            state_space = gym.spaces.Box(0., 1., (self.env.ale.getRAMSize() * 8, ))
        else:
            state_space = gym.spaces.Box(0., 1., (self.env.ale.getRAMSize(),))
        self.observation_space = gym.spaces.Dict(
            {
                "obs": self.env.observation_space,
                "state": state_space,
            }
        )

    def observation(self, observation):
        state = self.env.ale.getRAM()
        if self.one_hot_state:
            state = np.eye(256,)[state].flatten()
        elif self.as_bits:
            state = np.unpackbits(state[..., None], axis=1, count=8).flatten().astype(float)
            # TODO: look into making it more efficient (not float)
        else:
            state = state / 255.
        state_obs = {'obs': observation, 'state': state}
        return state_obs


class FlickeringObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, p_blank: float):
        super().__init__(env)
        self.p_blank = p_blank

    def observation(self, observation):
        if self.unwrapped.np_random.random() < self.p_blank:
            if isinstance(observation, dict):
                observation["obs"] = np.zeros_like(observation["obs"])
            else:
                observation = np.zeros_like(observation)
        return observation


class FloatObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0., 1., env.observation_space.shape)

    def observation(self, observation):
        return observation / 255.


class OnlyState(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space["obs"] = self.observation_space["state"]

    def observation(self, observation):
        observation["obs"] = observation["state"].copy()
        return observation


class HorizonReduceWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, horizon: int):
        super(HorizonReduceWrapper, self).__init__(env)
        self.horizon = horizon
        self.count = 0

    def reset(self, **kwargs):
        self.count = 0
        return super(HorizonReduceWrapper, self).reset()

    def step(self, action):
        obs, reward, done, info = super(HorizonReduceWrapper, self).step(action)
        self.count += 1
        if self.count >= self.horizon:
            assert done, "HorizonReduceWrapper horizon is reached and done is not True. " \
                         "This will lead to ReplayBuffer issues, increase the wrapper horizon."
        return obs, reward, done, info


def atari_env_maker(env_name: str,
                    horizon: int = 10000,
                    noop_max: int = 30,
                    frame_skip: int= 4,
                    screen_size: int= 84,
                    terminal_on_life_loss: bool = True,
                    clip_reward: bool = True,
                    p_blank: float = .5,
                    perturbation: float = 0.75,
                    recursive_perturbation: bool = True,
                    enforce_reset_to_null_once: bool = True,
                    reset_state_pad: float = 0.,
                    init_in_reset_state: bool = False,
                    reset_state_in_info: bool = True,
                    only_state: bool = False,
                    one_hot_state: bool = False,
                    as_bits: bool = False,
                    ):
    assert "NoFrameskip" in env_name
    env = gym.make(env_name)
    # logger.warning(f"Setting _max_episode_steps to 10k instead of {env._max_episode_steps} (no safe guard in place yet).")
    # env._max_episode_steps = 50000
    env.horizon = env._max_episode_steps
    env = AtariWrapper(env, noop_max=noop_max, frame_skip=frame_skip, screen_size=screen_size,
                       terminal_on_life_loss=terminal_on_life_loss, clip_reward=clip_reward)
    env = HorizonReduceWrapper(env, horizon)
    env = FloatObs(env)
    env = FlickeringObs(env, p_blank)
    env = AtariStateWrapper(env, one_hot_state=one_hot_state, as_bits=as_bits)
    env = PerturbedGymWrapper(env,
                              perturbation,
                              recursive_perturbation,
                              enforce_reset_to_null_once,reset_state_pad,
                              init_in_reset_state,
                              reset_state_in_info)
    if only_state:
        logger.warn("Only state should be true only for debugging !")
        env = OnlyState(env)
    return env
