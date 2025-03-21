from typing import Any

from ..transforms import DataTransform

class Flow:

    def __init__(
            self,
            dims: int,
            device: Any,
            data_transform: DataTransform = None,
        ):
        self.dims = dims
        self.device = device
        self.data_transform = data_transform

    def log_prob(self, x):
        raise NotImplementedError
    
    def sample(self, x):
        raise NotImplementedError

    def sample_and_log_prob(self, n_samples):
        raise NotImplementedError

    def fit(self, samples, **kwargs):
        raise NotImplementedError

    def fit_data_transform(self, x):
        return self.data_transform.fit(x)

    def rescale(self, x):
        return self.data_transform.forward(x)

    def inverse_rescale(self, x):
        return self.data_transform.inverse(x)
