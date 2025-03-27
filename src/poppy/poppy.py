import logging
import multiprocessing as mp
from typing import Callable

from .flows import get_flow_wrapper
from .samples import Samples
from .transforms import DataTransform

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
        bounded_transform: str = "probit",
        flow_matching: bool = False,
        device: str | None = None,
        xp: None = None,
        flow_backend: str = "zuko",
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.dims = dims
        self.parameters = parameters
        self.device = device
        self.eps = eps

        self.periodic_parameters = periodic_parameters
        self.prior_bounds = prior_bounds
        self.bounded_to_unbounded = bounded_to_unbounded
        self.bounded_transform = bounded_transform
        self.flow_matching = flow_matching
        self.flow_backend = flow_backend
        self.flow_kwargs = kwargs
        self.xp = xp

        self._flow = None

    @property
    def flow(self):
        """The normalizing flow object."""
        return self._flow

    @property
    def sampler(self):
        """The sampler object."""
        return self._sampler

    def convert_to_samples(
        self,
        x,
        log_likelihood=None,
        log_prior=None,
        log_q=None,
        evaluate: bool = True,
        xp=None,
    ) -> Samples:
        if xp is None:
            xp = self.xp
        samples = Samples(
            x=x,
            parameters=self.parameters,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            log_q=log_q,
            xp=xp,
        )

        if evaluate:
            if log_prior is None:
                logger.info("Evaluating log prior")
                samples.log_prior = samples.xp.to_device(
                    self.log_prior(samples), samples.device
                )
            if log_likelihood is None:
                logger.info("Evaluating log likelihood")
                samples.log_likelihood = samples.xp.to_device(
                    self.log_likelihood(samples), samples.device
                )
            samples.compute_weights()
        return samples

    def init_flow(self):
        if self.flow_backend == "zuko":
            import array_api_compat.torch as xp
        elif self.flow_backend == "flowjax":
            import jax.numpy as xp
        data_transform = DataTransform(
            parameters=self.parameters,
            prior_bounds=self.prior_bounds,
            periodic_parameters=self.periodic_parameters,
            bounded_to_unbounded=self.bounded_to_unbounded,
            bounded_transform=self.bounded_transform,
            device=self.device,
            xp=xp,
            eps=self.eps,
        )
        FlowClass = get_flow_wrapper(
            backend=self.flow_backend, flow_matching=self.flow_matching
        )

        self._flow = FlowClass(
            dims=self.dims,
            device=self.device,
            data_transform=data_transform,
            **self.flow_kwargs,
        )

    def fit(self, samples: Samples, **kwargs) -> dict:
        if self.xp is None:
            self.xp = samples.xp

        if self.flow is None:
            self.init_flow()

        self.training_samples = samples
        logger.info(f"Training with {len(samples.x)} samples")
        history = self.flow.fit(samples.x, **kwargs)
        return history

    def init_sampler(self, sampler_type: str):
        if sampler_type == "importance":
            from .samplers.importance import ImportanceSampler as SamplerClass
        elif sampler_type == "emcee":
            from .samplers.mcmc import Emcee as SamplerClass
        elif sampler_type == "smc":
            from .samplers.smc import EmceeSMC as SamplerClass
        else:
            raise ValueError

        sampler = SamplerClass(
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
            dims=self.dims,
            flow=self.flow,
            xp=self.xp,
        )
        return sampler

    def sample_posterior(
        self,
        n_samples: int = 1,
        sampler: str = "importance",
        xp=None,
        **kwargs,
    ) -> Samples:
        """Draw samples from the posterior distribution.

        Parameters
        ----------
        n_samples : int
            The number of sample to draw.

        Returns
        -------
        samples : Samples
            Samples object contain samples and their corresponding weights.
        """
        self._sampler = self.init_sampler(sampler)
        samples = self._sampler.sample(n_samples, **kwargs)
        if xp is not None:
            samples = samples.to_namespace(xp)
        samples.parameters = self.parameters
        logger.info("Sample summary:")
        logger.info(samples)
        return samples

    def enable_pool(self, pool: mp.Pool):
        """Context manager to temporarily replace the log_likelihood method
        with a version that uses a multiprocessing pool to parallelize
        computation.

        Parameters
        ----------
        pool : multiprocessing.Pool
            The pool to use for parallel computation.
        """
        from .utils import PoolHandler

        return PoolHandler(self, pool)

    def config_dict(self) -> dict:
        return {
            # "log_likelihood": self.log_likelihood,
            # "log_prior": self.log_prior,
            "dims": self.dims,
            "parameters": self.parameters,
            "periodic_parameters": self.periodic_parameters,
            "prior_bounds": self.prior_bounds,
            "bounded_to_unbounded": self.bounded_to_unbounded,
            # "bounded_transform": self.bounded_transform,
            "flow_matching": self.flow_matching,
            # "device": self.device,
            # "xp": self.xp,
            "flow_backend": self.flow_backend,
            "flow_kwargs": self.flow_kwargs,
            "eps": self.eps,
        }

    def save_config(self, filename: str) -> None:
        """Save the configuration to a JSON file."""
        import json

        with open(filename, "w") as f:
            json.dump(self.config_dict(), f, indent=4)
