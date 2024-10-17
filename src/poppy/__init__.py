"""
bayesian-poppy: Bayesian posterior post-processing in python
"""

import logging
from importlib.metadata import PackageNotFoundError, version

from .backend import set_backend
from .poppy import Poppy

set_backend()

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Poppy",
    "set_backend",
]
