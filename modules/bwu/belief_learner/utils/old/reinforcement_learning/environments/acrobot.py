import numpy as np
from gym.envs.classic_control.acrobot import AcrobotEnv


class AcrobotEnvRandomInit(AcrobotEnv):
    def reset(self):
        high = np.array([np.pi, np.pi, self.MAX_VEL_1, self.MAX_VEL_2])
        loc = np.zeros(shape=(4,))
        scale = np.array([np.pi / 9, np.pi / 9, np.pi / 6, np.pi / 4])
        self.state = np.clip(np.random.normal(loc, scale), -high, high)
        return self._get_ob()
