from belief_learner.utils.stats import get_mem
from belief_learner.utils.stats import TimeManager
from belief_learner.utils.logging import get_logger

import inspect
from functools import wraps
import pprint


def _get_name(func, *args):
    return func.__qualname__


def time_memory(func):
    @wraps(func)
    def t(*args, **kwargs):
        func_ = getattr(func, '__wrapped__', func)
        name = _get_name(func_, *args)
        logger = get_logger(func_.__module__)
        logger.debug(f'Entering {name}')
        start_mem = get_mem()
        with TimeManager() as time_manager:
            output = func(*args, **kwargs)
        logger.info(f'{name} took {time_manager.total_duration:.3f}s '
                     f'and increased memory of {get_mem() - start_mem:.3f} GiB.')
        return output

    return t


def log_usage(func):
    @wraps(func)
    def t(*args, **kwargs):
        func_ = getattr(func, '__wrapped__', func)
        name = _get_name(func_, *args)
        logger = get_logger(inspect.getmodule(func).__name__)
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args.pop('self', None)
        logger.debug(f"Entering {name}(\n {pprint.pformat(func_args, indent=4)[1:-1]}\n)")
        output = func(*args, **kwargs)
        return output

    return t
