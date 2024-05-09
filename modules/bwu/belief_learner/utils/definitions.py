from typing import Union, Dict, Any

import numpy as np
import tensorflow as tf

Array = Union[np.ndarray, tf.Tensor]
DictArray = Dict[Any, Array]

STATE = 'state'
NEXT_STATE = 'next_state'
ACTION = 'action'
PREV_ACTION = 'prev_action'
OBS = 'obs'
NEXT_OBS = 'next_obs'
REWARD = 'reward'
DONE = 'done'
INFO = 'info'
SUB_BELIEF = 'sub_belief'
PREV_SUB_BELIEF = 'prev_sub_belief'
NEXT_SUB_BELIEF = 'next_sub_belief'
EPISODE_NBR = 'episode_nbr'
TIMESTEP = 'timestep'
IS_RESET_STATE = 'is_reset_state'
NEXT_IS_RESET_STATE = 'next_is_reset_state'
RAW_INDEXES = 'raw_indexes'

next_keys = {
    STATE: NEXT_STATE,
    OBS: NEXT_OBS,
    SUB_BELIEF: NEXT_SUB_BELIEF,
    IS_RESET_STATE: NEXT_IS_RESET_STATE,
}

inv_next_keys = {
    NEXT_STATE: STATE,
    NEXT_OBS: OBS,
    NEXT_SUB_BELIEF: SUB_BELIEF,
    NEXT_IS_RESET_STATE: IS_RESET_STATE,
}

prev_keys = {
    ACTION: PREV_ACTION
}
