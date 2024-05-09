import logging
from typing import Union

logger = logging.getLogger("bwu")
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s  %(name)-40s  {%(lineno)03d} %(levelname)-8s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class TFFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("Model was constructed with shape")


logging.getLogger('tensorflow').addFilter(TFFilter())


def get_logger(name: str):
    logger_ = logger.getChild(name)
    return logger_

def set_logger_level(level: Union[int, str]):
    logger.setLevel(level)

def file_handler_to_logger(path: str):
    fh = logging.FileHandler(path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)