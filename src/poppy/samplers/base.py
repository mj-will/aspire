from typing import Callable


from ..samples import Samples
from ..flows.base import Flow


class Sampler:

    def __init__(
        self, 
        log_likelihood: Callable,
        log_prior: Callable,
        dims: int,
        flow: Flow,
        xp: Callable,
        parameters: list[str] | None = None,
    ):
        self.flow = flow
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.dims = dims
        self.xp = xp
        self.parameters = parameters
        self.history = None

    def sample(self, n_samples: int) -> Samples:
        raise NotImplementedError
