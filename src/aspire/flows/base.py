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
