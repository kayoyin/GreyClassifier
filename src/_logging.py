"""
Script for the logger.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

__DEFAULT_LOG_PATH__ = "logs"
__DEFAULT_MAX_SIZE__ = 1 * 1024 * 1024
__DEFAULT_LOG_FORMAT__ = "%(asctime)s - %(name)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s"


def check_path(path: str):
    """Routine that checks the path.
    """
    assert isinstance(path, str)
    if not os.path.exists(path):
        os.makedirs(path)


def get_logger(name: str, path: str = __DEFAULT_LOG_PATH__, max_size: int = __DEFAULT_MAX_SIZE__):
    """Routine that builds a logger.
    """

    assert isinstance(name, str)
    assert isinstance(path, str)
    assert isinstance(max_size, int)

    # Step 1: check the path.
    check_path(path)

    # Step 2: get the logger.
    logger_name = name
    logger = logging.getLogger(logger_name)

    # Step 3: build the rotating file handler.
    rotating_format = logging.Formatter(__DEFAULT_LOG_FORMAT__)
    rotating_handler = RotatingFileHandler("{}/{}.log".format(path, logger_name), mode="a", maxBytes=max_size, backupCount=2, encoding=None, delay=0)
    rotating_handler.setLevel(logging.INFO)
    rotating_handler.setFormatter(rotating_format)

    # Step 4: build the rotating file handler.
    console_format = logging.Formatter(__DEFAULT_LOG_FORMAT__)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_format)

    # Step 5: Set the logger.
    if not len(logger.handlers):
        logger.addHandler(console_handler)
        logger.addHandler(rotating_handler)
        logger.setLevel(logging.INFO)

    return logger
