import time
from typing import Optional, List, Dict, Union
import psutil
import os
import logging
import itertools
import numpy as np

_LOGGER_SET = False


def set_logger(filename: Optional[str] = None):
    global _LOGGER_SET
    if not _LOGGER_SET:
        logging.basicConfig(
            filename=filename,
            format='%(asctime)s  %(name)s  {%(lineno)d} %(levelname)s: %(message)s',
            level=logging.DEBUG,
        )
        _LOGGER_SET = True


def get_mem():
    return psutil.Process().memory_info().vms / 1024**3


def combine_stats(stats: List[Dict]) -> Union[Dict, List]:
    same_keys = all([set(o.keys()) == set(stats[0].keys()) for o in stats[1:]])
    assert same_keys
    elements = {}
    for key in stats[0].keys():
        elements[key] = _merge(list(itertools.chain(*[o[key] if isinstance(o[key], list) else [o[key]] for o in stats])))

    actor_worker_name = 'BeliefA2CWorker' if 'BeliefA2CWorker' in elements else "RL Worker"
    belief_worker_name = 'BeliefA2CWorker' if 'BeliefA2CWorker' in elements else "Belief Worker"
    if actor_worker_name in elements and elements[actor_worker_name].get("episode_ended_stats", False):
        elements[actor_worker_name]["episode_ended_stats"] = list(filter(lambda x: len(x) > 0,
                                                                   elements[actor_worker_name]["episode_ended_stats"]))
        if elements[actor_worker_name]["episode_ended_stats"]:
            elements[actor_worker_name]["episode_ended_stats"] = _merge(
                list(map(lambda x: x[0], itertools.chain(*elements[actor_worker_name]["episode_ended_stats"]))))
            elements[actor_worker_name]["episode_ended_stats"].pop('episode_id')

    if belief_worker_name in elements and "obs_filter_variance" in elements[belief_worker_name]:
        obs_filter_variance = np.array(elements[belief_worker_name]["obs_filter_variance"])
        obs_filter_variance = np.mean(obs_filter_variance, 0)

        if obs_filter_variance.shape in ((), (1,)):
            obs_filter_variance = obs_filter_variance.item()
        else:
            obs_filter_variance = [e.item() for e in obs_filter_variance]
        elements[belief_worker_name]["obs_filter_variance"] = obs_filter_variance

    return elements

def _merge(stats: List[Dict]):
    same_keys = all([set(o.keys()) == set(stats[0].keys()) for o in stats[1:]])
    assert same_keys
    elements = {key: [stat[key] for stat in stats] for key in stats[0].keys()}
    elements = {key: np.mean(value) if isinstance(value[0], (int, float)) else value for key, value in elements.items()}
    return elements

class TimeManager:
    def __init__(self, name: Optional[str] = None):
        self._enter_time = None
        self._exit_time = None
        self._durations = []
        self._n_calls = 0
        self._in = False
        self._name = str(name) if name is not None else ''

    def __enter__(self):
        self._in = True
        self._enter_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_time = time.perf_counter()
        self._in = False
        self._n_calls += 1
        self._durations.append(self._exit_time - self._enter_time)

    def durations(self) -> List[float]:
        durations = self._durations.copy()
        if self._in:
            durations.append(time.perf_counter() - self._enter_time)
        return durations

    @property
    def mean_duration(self) -> float:
        if self._n_calls == 0:
            return float('nan')
        return sum(self._durations) / self._n_calls

    @property
    def total_duration(self) -> float:
        return sum(self._durations)

    @property
    def calls(self) -> int:
        return self._n_calls

    def __repr__(self):
        return f"Time Manager ({self._name}) : mean time {self.mean_duration} sec - {self.calls} calls"

