import numpy as np

from belief_learner.utils.definitions import SUB_BELIEF, OBS, ACTION, REWARD, NEXT_OBS, EPISODE_NBR, DONE, \
    NEXT_SUB_BELIEF, IS_RESET_STATE, STATE, NEXT_STATE, inv_next_keys
from copy import deepcopy


class DataStore:
    def __init__(self, reset_obs, reset_action, reset_sub_belief, element_scheme, max_t, reset_state = None,
                 use_state=False):
        self._reset_obs = reset_obs
        self._reset_action = reset_action
        self._reset_sub_belief = reset_sub_belief
        self._element_scheme = element_scheme
        self.max_t = max_t
        self._reset_state = reset_state
        self.use_state = use_state
        self.data = {}
        self._keys = [OBS, ACTION, REWARD, DONE, NEXT_OBS, SUB_BELIEF, NEXT_SUB_BELIEF, IS_RESET_STATE]
        if self.use_state:
            self._keys.extend([STATE, NEXT_STATE])
        self._base_dict = {k: [v] for k, v in zip(self._keys, (self._reset_obs, self._reset_action, 0., False,
                                                               self._reset_obs, self._reset_sub_belief,
                                                               self._reset_sub_belief, True, self._reset_state,
                                                               self._reset_state))}

    def reset(self, episode_nbrs, prev_sub_belief, prev_action, obs, is_first, prev_obs, state=None, prev_state=None):
        self.data = {}

        for i, episode_nbr in enumerate(episode_nbrs):
            self._init_episode_first(
                episode_nbr=episode_nbr,
                next_obs=obs[i],
                sub_belief=prev_sub_belief[i],
                action=prev_action[i],
                is_reset_state=is_first[i],
                next_state=state[i],
                obs=prev_obs[i],
                state=prev_state[i]
            )

    def add(self, data):
        for i, episode_nbr in enumerate(data[EPISODE_NBR]):
            if episode_nbr not in self.data:
                assert data['is_first'][i]
                state = None
                if self.use_state:
                    state = data[STATE][i]
                self._init_episode_first(episode_nbr=episode_nbr, next_obs=data[OBS][i], next_state=state,
                                         obs=self._reset_obs.copy(), state=self._reset_state.copy(),)

            for key in self.data[episode_nbr]:
                if key == IS_RESET_STATE:
                    self.data[episode_nbr][key].append(False)
                else:
                    self.data[episode_nbr][key].append(data[key][i])
            if len(self.data[episode_nbr][NEXT_SUB_BELIEF]) > 1:
                self.data[episode_nbr][NEXT_SUB_BELIEF][-2] = data[SUB_BELIEF][i].copy()

    def time_output(self,):
        max_length = self.max_t + 1
        nbr_episodes = len(self.data)
        data = {}
        for key in self._keys:
            key_ = inv_next_keys.get(key, key)
            key_shape = self._element_scheme[key_].shape
            key_dtype = self._element_scheme[key_].dtype
            data[key] = np.zeros((max_length, nbr_episodes,) + key_shape, dtype=key_dtype)
            for i, episode_data in enumerate(self.data.values()):
                episode_key_data = np.stack(episode_data[key])
                data[key][:len(episode_key_data), i] = episode_key_data
        data["mask"] = np.zeros((max_length, nbr_episodes), bool)
        for i, episode_data in enumerate(self.data.values()):
            data["mask"][:len(episode_data[OBS]), i] = True
        return data

    def output(self,):
        pass

    def _init_episode_first(self, episode_nbr, next_obs, obs, sub_belief=None, action=None, is_reset_state=True,
                            next_state=None, state=None):
        self.data[episode_nbr] = deepcopy(self._base_dict)
        self.data[episode_nbr][NEXT_OBS] = [next_obs.copy()]
        self.data[episode_nbr][OBS] = [obs.copy()]
        if sub_belief is not None:
            self.data[episode_nbr][SUB_BELIEF] = [sub_belief]
        if action is not None:
            self.data[episode_nbr][ACTION] = [action]
        if not is_reset_state:
            self.data[episode_nbr][IS_RESET_STATE] = [is_reset_state]
        if next_state is not None:
            self.data[episode_nbr][NEXT_STATE] = [next_state.copy()]
        if state is not None:
            self.data[episode_nbr][STATE] = [state.copy()]

    def _init_episode(self, episode_nbr):
        data = {key: [] for key in self._keys}
        self.data[episode_nbr] = data
