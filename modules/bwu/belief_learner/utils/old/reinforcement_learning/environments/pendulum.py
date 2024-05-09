import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv


class PendulumRandomInit(PendulumEnv):
    def reset(self):
        high = np.array([np.pi, 8])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()
