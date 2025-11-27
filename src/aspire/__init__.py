"""
aspire: Accelerated Sequential Posterior Inference via REuse
"""

import logging
from importlib.metadata import PackageNotFoundError, version

from .aspire import Aspire
from .samples import Samples

try:
    __version__ = version("aspire")
except PackageNotFoundError:
    __version__ = "unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Aspire",
    "Samples",
]
