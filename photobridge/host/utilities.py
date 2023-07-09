"""
Module with host-side utilities
"""

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
