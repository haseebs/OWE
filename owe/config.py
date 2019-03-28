import logging
from copy import deepcopy
from typing import Any


logger = logging.getLogger("owe")


class Config:
    """
    Stores config settings immutably.
    """
    __conf = {}

    @staticmethod
    def get(name: str) -> Any:
        try:
            return deepcopy(Config.__conf[name])
        except KeyError:
            logger.warning(f"Config setting: '{name}' not found. Returning None!")
            return

    @staticmethod
    def set(name: str, value: Any):
        try:
            Config.__conf[name] = deepcopy(value)
        except: # Goes here because torch.device is not serializable (its bugged)
            Config.__conf[name] = value
