import logging
from typing import Callable

from .flows import get_flow_wrapper
from .samples.base import BaseSamples
from .transforms.base import DataTransform


logger = logging.getLogger(__name__)
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
        parameters: list[str] | None = None,
        periodic_parameters: list[str] | None = None,
        prior_bounds: dict[str, tuple[float, float]] | None = None,
        bounded_to_unbounded: bool = True,
        flow_matching: bool = False,
        device: str | None = None,
        **kwargs,
    ) -> None:
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.dims = dims
        self.parameters = parameters
        self.device = device
        data_transform = DataTransform(
            parameters=parameters,
            prior_bounds=prior_bounds,
            periodic_parameters=periodic_parameters,
            bounded_to_unbounded=bounded_to_unbounded,
            device=self.device,
        )
        self._flow = get_flow_wrapper(flow_matching)(
            dims=self.dims, data_transform=data_transform, **kwargs
        )

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
            parameters=self.parameters,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            log_q=log_q,
        )

        if evaluate:
            if log_prior is None:
                samples.log_prior = self.log_prior(samples)
            if log_likelihood is None:
                samples.log_likelihood = self.log_likelihood(samples)
            samples.compute_weights()
        return samples

    def fit(self, samples: BaseSamples, **kwargs) -> dict:
        self.training_samples = samples
        samples = samples.to_backend()

        return self.flow.fit(samples.x, **kwargs)

    def sample_posterior(
        self, n_samples: int = 1, return_numpy: bool = True
    ) -> BaseSamples:
        x, log_q = self.flow.sample(n_samples)
        samples = self.convert_to_samples(x, log_q=log_q)
        samples = samples.to_numpy() if return_numpy else samples
        logger.info("Sample summary:")
        logger.info(samples)
        return samples
