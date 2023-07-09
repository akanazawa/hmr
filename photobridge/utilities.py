"""
Module with various utilities
"""

import logging
import os

import box
import yaml


def read_yaml(path: str) -> box.Box:
    """Read content of yaml file from path

    :param path: path to yaml file
    :type path: str
    :return: box.Box representation of file content
    """

    with open(path, encoding="utf-8") as file:

        return box.Box(yaml.safe_load(file))


def get_logger(path: str) -> logging.Logger:
    """
    Returns a logger configured to write to a file
    :param path: path to file logger should write to
    :return: logger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("image_retrieval")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
