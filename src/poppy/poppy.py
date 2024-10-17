from typing import Callable

from .flows import get_flow_wrapper
from .samples.base import BaseSamples


class Poppy:
    """Posterior post-processing.

    Parameters
    ----------
    log_likelihood : Callable
        The log likelihood function.
    log_prior : Callable
        The log prior function.
    dims : int
        The number of dimensions.
    flow_matching : bool
        Whether to use flow matching.
    **kwargs
        Keyword arguments to pass to the flow.
    """

    def __init__(
        self,
        *,
        log_likelihood: Callable,
        log_prior: Callable,
        dims: int,
        flow_matching: bool = False,
        **kwargs,
    ) -> None:
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.dims = dims
        self._flow = get_flow_wrapper(flow_matching)(dims=self.dims, **kwargs)

    @property
    def flow(self):
        """The normalizing flow object."""
        return self._flow

    def convert_to_samples(
        self,
        x,
        log_likelihood=None,
        log_prior=None,
        log_q=None,
        evaluate: bool = True,
    ) -> BaseSamples:
        from .samples import Samples

        samples = Samples(
            x=x,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            log_q=log_q,
        )

        if evaluate:
            if log_likelihood is None:
                samples.log_likelihood = self.log_likelihood(samples)
            if log_prior is None:
                samples.log_prior = self.log_prior(samples)
            if log_q is None:
                samples.log_q = self.flow.log_prob(samples.x)
            samples.compute_weights()
        return samples

    def fit(self, samples: BaseSamples, **kwargs) -> dict:
        samples = samples.to_backend()
        return self.flow.fit(samples.x, **kwargs)

    def sample_posterior(
        self, n_samples: int = 1, return_numpy: bool = True
    ) -> BaseSamples:
        x, log_q = self.flow.sample(n_samples)
        samples = self.convert_to_samples(x, log_q=log_q)
        return samples.to_numpy() if return_numpy else samples
