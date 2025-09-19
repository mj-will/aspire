import logging
import multiprocessing as mp
from inspect import signature
from typing import Any, Callable

import h5py

from .flows import get_flow_wrapper
from .history import History
from .samples import Samples
from .transforms import (
    CompositeTransform,
    FlowPreconditioningTransform,
    FlowTransform,
)
from .utils import recursively_save_to_h5_file

logger = logging.getLogger(__name__)


class Aspire:
    """Accelerated Sequential Posterior Inference via REuse (aspire).

    Parameters
    ----------
    log_likelihood : Callable
        The log likelihood function.
    log_prior : Callable
        The log prior function.
    dims : int
        The number of dimensions.
    parameters : list[str] | None
        The list of parameter names. If None, any samples objects will not
        have the parameters names specified.
    periodic_parameters : list[str] | None
        The list of periodic parameters.
    prior_bounds : dict[str, tuple[float, float]] | None
        The bounds for the prior. If None, some parameter transforms cannot
        be applied.
    bounded_to_unbounded : bool
        Whether to transform bounded parameters to unbounded ones.
    bounded_transform : str
        The transformation to use for bounded parameters. Options are
        'logit', 'exp', or 'tanh'.
    device : str | None
        The device to use for the flow. If None, the default device will be
        used. This is only used when using the PyTorch backend.
    xp : Callable | None
        The array backend to use. If None, the default backend will be
        used.
    flow_backend : str
        The backend to use for the flow. Options are 'zuko' or 'flowjax'.
    flow_matching : bool
        Whether to use flow matching.
    eps : float
        The epsilon value to use for data transforms.
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
        bounded_transform: str = "logit",
        device: str | None = None,
        xp: Callable | None = None,
        flow_backend: str = "zuko",
        flow_matching: bool = False,
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

    @property
    def n_likelihood_evaluations(self):
        """The number of likelihood evaluations."""
        if hasattr(self, "_sampler"):
            return self._sampler.n_likelihood_evaluations
        else:
            return None

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
        FlowClass, xp = get_flow_wrapper(
            backend=self.flow_backend, flow_matching=self.flow_matching
        )

        data_transform = FlowTransform(
            parameters=self.parameters,
            prior_bounds=self.prior_bounds,
            bounded_to_unbounded=self.bounded_to_unbounded,
            bounded_transform=self.bounded_transform,
            device=self.device,
            xp=xp,
            eps=self.eps,
        )

        # Check if FlowClass takes `parameters` as an argument
        flow_init_params = signature(FlowClass.__init__).parameters
        if "parameters" in flow_init_params:
            self.flow_kwargs["parameters"] = self.parameters.copy()

        logger.info(f"Configuring {FlowClass} with kwargs: {self.flow_kwargs}")

        self._flow = FlowClass(
            dims=self.dims,
            device=self.device,
            data_transform=data_transform,
            **self.flow_kwargs,
        )

    def fit(self, samples: Samples, **kwargs) -> History:
        if self.xp is None:
            self.xp = samples.xp

        if self.flow is None:
            self.init_flow()

        self.training_samples = samples
        logger.info(f"Training with {len(samples.x)} samples")
        history = self.flow.fit(samples.x, **kwargs)
        return history

    def get_sampler_class(self, sampler_type: str) -> Callable:
        """Get the sampler class based on the sampler type.

        Parameters
        ----------
        sampler_type : str
            The type of sampler to use. Options are 'importance', 'emcee', or 'smc'.
        """
        if sampler_type == "importance":
            from .samplers.importance import ImportanceSampler as SamplerClass
        elif sampler_type == "emcee":
            from .samplers.mcmc import Emcee as SamplerClass
        elif sampler_type == "emcee_smc":
            from .samplers.smc.emcee import EmceeSMC as SamplerClass
        elif sampler_type == "minipcn":
            from .samplers.mcmc import MiniPCN as SamplerClass
        elif sampler_type in ["smc", "minipcn_smc"]:
            from .samplers.smc.minipcn import MiniPCNSMC as SamplerClass
        elif sampler_type == "blackjax_smc":
            from .samplers.smc.blackjax import BlackJAXSMC as SamplerClass
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
        return SamplerClass

    def init_sampler(
        self,
        sampler_type: str,
        preconditioning: str | None = None,
        preconditioning_kwargs: dict | None = None,
        **kwargs,
    ) -> Callable:
        """Initialize the sampler for posterior sampling.

        Parameters
        ----------
        sampler_type : str
            The type of sampler to use. Options are 'importance', 'emcee', or 'smc'.
        """
        SamplerClass = self.get_sampler_class(sampler_type)

        if sampler_type != "importance" and preconditioning is None:
            preconditioning = "default"

        preconditioning = preconditioning.lower() if preconditioning else None

        if preconditioning is None or preconditioning == "none":
            transform = None
        elif preconditioning in ["standard", "default"]:
            preconditioning_kwargs = preconditioning_kwargs or {}
            preconditioning_kwargs.setdefault("affine_transform", False)
            preconditioning_kwargs.setdefault("bounded_to_unbounded", False)
            preconditioning_kwargs.setdefault("bounded_transform", "logit")
            transform = CompositeTransform(
                parameters=self.parameters,
                prior_bounds=self.prior_bounds,
                periodic_parameters=self.periodic_parameters,
                xp=self.xp,
                device=self.device,
                **preconditioning_kwargs,
            )
        elif preconditioning == "flow":
            preconditioning_kwargs = preconditioning_kwargs or {}
            preconditioning_kwargs.setdefault("affine_transform", False)
            transform = FlowPreconditioningTransform(
                parameters=self.parameters,
                flow_backend=self.flow_backend,
                flow_kwargs=self.flow_kwargs,
                flow_matching=self.flow_matching,
                periodic_parameters=self.periodic_parameters,
                bounded_to_unbounded=self.bounded_to_unbounded,
                prior_bounds=self.prior_bounds,
                xp=self.xp,
                device=self.device,
                **preconditioning_kwargs,
            )
        else:
            raise ValueError(f"Unknown preconditioning: {preconditioning}")

        sampler = SamplerClass(
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
            dims=self.dims,
            prior_flow=self.flow,
            xp=self.xp,
            preconditioning_transform=transform,
            **kwargs,
        )
        return sampler

    def sample_posterior(
        self,
        n_samples: int = 1000,
        sampler: str = "importance",
        xp: Any = None,
        return_history: bool = False,
        preconditioning: str | None = None,
        preconditioning_kwargs: dict | None = None,
        **kwargs,
    ) -> Samples:
        """Draw samples from the posterior distribution.

        If using a sampler that calls an external sampler, e.g.
        :code:`minipcn` then keyword arguments for this sampler should be
        specified in :code:`sampler_kwargs`. For example:

        .. code-block:: python

            aspire = aspire(...)
            aspire.sample_posterior(
                n_samples=1000,
                sampler="minipcn_smc",
                adaptive=True,
                sampler_kwargs=dict(
                    n_steps=100,
                    step_fn="tpcn",
                )
            )

        Parameters
        ----------
        n_samples : int
            The number of sample to draw.
        sampler: str
            Sampling algorithm to use for drawing the posterior samples.
        xp: Any
            Array API for the final samples.
        return_history : bool
            Whether to return the history of the sampler.
        preconditioning: str
            Type of preconditioning to apply in the sampler. Options are
            'default', 'flow', or 'none'. If not specified, the default
            will depend on the sampler being used. The importance sampler
            will default to 'none' and the other samplers to 'default'
        preconditioning_kwargs: dict
            Keyword arguments to pass to the preconditioning transform.
        kwargs : dict
            Keyword arguments to pass to the sampler. These are passed
            automatically to the init method of the sampler or to the sample
            method.

        Returns
        -------
        samples : Samples
            Samples object contain samples and their corresponding weights.
        """
        SamplerClass = self.get_sampler_class(sampler)
        # Determine sampler initialization parameters
        # and remove them from kwargs
        sampler_init_kwargs = signature(SamplerClass.__init__).parameters
        sampler_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in sampler_init_kwargs and k != "self"
        }
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in sampler_init_kwargs or k == "self"
        }

        self._sampler = self.init_sampler(
            sampler,
            preconditioning=preconditioning,
            preconditioning_kwargs=preconditioning_kwargs,
            **sampler_kwargs,
        )
        samples = self._sampler.sample(n_samples, **kwargs)
        if xp is not None:
            samples = samples.to_namespace(xp)
        samples.parameters = self.parameters
        logger.info(f"Sampled {len(samples)} samples from the posterior")
        logger.info(
            f"Number of likelihood evaluations: {self.n_likelihood_evaluations}"
        )
        logger.info("Sample summary:")
        logger.info(samples)
        if return_history:
            return samples, self._sampler.history
        else:
            return samples

    def enable_pool(self, pool: mp.Pool, **kwargs):
        """Context manager to temporarily replace the log_likelihood method
        with a version that uses a multiprocessing pool to parallelize
        computation.

        Parameters
        ----------
        pool : multiprocessing.Pool
            The pool to use for parallel computation.
        """
        from .utils import PoolHandler

        return PoolHandler(self, pool, **kwargs)

    def config_dict(
        self, include_sampler_config: bool = True, **kwargs
    ) -> dict:
        """Return a dictionary with the configuration of the aspire object.

        Parameters
        ----------
        include_sampler_config : bool
            Whether to include the configuration of the sampler. Default is
            True.
        kwargs : dict
            Additional keyword arguments to pass to the :py:meth:`config_dict`
            method of the sampler.
        """
        config = {
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
        if include_sampler_config:
            config["sampler_config"] = self.sampler.config_dict(**kwargs)
        return config

    def save_config(
        self, h5_file: h5py.File, path="aspire_config", **kwargs
    ) -> None:
        """Save the configuration to an HDF5 file.

        Parameters
        ----------
        h5_file : h5py.File
            The HDF5 file to save the configuration to.
        path : str
            The path in the HDF5 file to save the configuration to.
        kwargs : dict
            Additional keyword arguments to pass to the :py:meth:`config_dict`
            method.
        """
        recursively_save_to_h5_file(
            h5_file,
            path,
            self.config_dict(**kwargs),
        )

    def save_config_to_json(self, filename: str) -> None:
        """Save the configuration to a JSON file."""
        import json

        with open(filename, "w") as f:
            json.dump(self.config_dict(), f, indent=4)

    def sample_flow(self, n_samples: int = 1, xp=None) -> Samples:
        """Sample from the flow directly.

        Includes the data transform, but does not compute
        log likelihood or log prior.
        """
        if self.flow is None:
            self.init_flow()
        x, log_q = self.flow.sample_and_log_prob(n_samples)
        samples = Samples(x=x, log_q=log_q, xp=xp, parameters=self.parameters)
        return samples
