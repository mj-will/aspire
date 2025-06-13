import logging
from typing import Callable

from ..flows.base import Flow
from ..samples import Samples
from ..utils import track_calls

logger = logging.getLogger(__name__)


class Sampler:
    """Base class for all samplers.

    Parameters
    ----------
    log_likelihood : Callable
        The log likelihood function.
    log_prior : Callable
        The log prior function.
    dims : int
        The number of dimensions.
    flow : Flow
        The flow object.
    xp : Callable
        The array backend to use.
    parameters : list[str] | None
        The list of parameter names. If None, any samples objects will not
        have the parameters names specified.
    """

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
        self._log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.dims = dims
        self.xp = xp
        self.parameters = parameters
        self.history = None
        self.n_likelihood_evaluations = 0

    @track_calls
    def sample(self, n_samples: int) -> Samples:
        raise NotImplementedError

    def log_likelihood(self, samples: Samples) -> Samples:
        """Computes the log likelihood of the samples.

        Also tracks the number of likelihood evaluations.
        """
        self.n_likelihood_evaluations += len(samples)
        return self._log_likelihood(samples)

    def config_dict(self, include_sample_calls: bool = True) -> dict:
        """
        Returns a dictionary with the configuration of the sampler.

        Parameters
        ----------
        include_sample_calls : bool
            Whether to include the sample calls in the configuration.
            Default is True.
        """
        config = {}
        if include_sample_calls:
            if hasattr(self, "sample") and hasattr(self.sample, "calls"):
                config["sample_calls"] = {
                    "args": self.sample.calls.args,
                    "kwargs": self.sample.calls.kwargs,
                }
            else:
                logger.warning(
                    "Sampler does not have a sample method with calls attribute."
                )
        return config
