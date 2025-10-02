import inspect
import logging
from typing import Any

from ..history import FlowHistory
from ..transforms import BaseTransform, IdentityTransform

logger = logging.getLogger(__name__)


class Flow:
    xp = None  # type: Any

    def __init__(
        self,
        dims: int,
        device: Any,
        data_transform: BaseTransform | None = None,
    ):
        self.dims = dims
        self.device = device

        if data_transform is None:
            data_transform = IdentityTransform(self.xp)
            logger.info("No data_transform provided, using IdentityTransform.")

        self.data_transform = data_transform

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, x):
        raise NotImplementedError

    def sample_and_log_prob(self, n_samples):
        raise NotImplementedError

    def fit(self, samples, **kwargs) -> FlowHistory:
        raise NotImplementedError

    def fit_data_transform(self, x):
        return self.data_transform.fit(x)

    def rescale(self, x):
        return self.data_transform.forward(x)

    def inverse_rescale(self, x):
        return self.data_transform.inverse(x)

    def config_dict(self):
        """Return a dictionary of the configuration of the flow.

        This can be used to recreate the flow by passing the dictionary
        as keyword arguments to the constructor.

        This is automatically populated with the arguments passed to the
        constructor.

        Returns
        -------
        config : dict
            The configuration dictionary.
        """
        return getattr(self, "_init_args", {})

    def save(self, h5_file, path="flow"):
        raise NotImplementedError

    @classmethod
    def load(cls, h5_file, path="flow"):
        raise NotImplementedError

    def __new__(cls, *args, **kwargs):
        # Create instance
        obj = super().__new__(cls)
        # Inspect the subclass's __init__ signature
        sig = inspect.signature(cls.__init__)
        bound = sig.bind_partial(obj, *args, **kwargs)
        bound.apply_defaults()
        # Save args (excluding self)
        obj._init_args = {
            k: v for k, v in bound.arguments.items() if k != "self"
        }
        return obj
