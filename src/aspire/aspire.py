import copy
import logging
import multiprocessing as mp
import pickle
from contextlib import contextmanager
from inspect import signature
from typing import Any, Callable

import h5py

from .flows import get_flow_wrapper
from .flows.base import Flow
from .history import History
from .samplers.base import Sampler
from .samples import Samples
from .transforms import (
    CompositeTransform,
    FlowPreconditioningTransform,
    FlowTransform,
)
from .utils import (
    AspireFile,
    function_id,
    load_from_h5_file,
    recursively_save_to_h5_file,
    resolve_xp,
)

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
    flow : Flow | None
        The flow object, if it already exists.
        If None, a new flow will be created.
    flow_backend : str
        The backend to use for the flow. Options are 'zuko' or 'flowjax'.
    flow_matching : bool
        Whether to use flow matching.
    eps : float
        The epsilon value to use for data transforms.
    dtype : Any | str | None
        The data type to use for the samples, flow and transforms.
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
        flow: Flow | None = None,
        flow_backend: str = "zuko",
        flow_matching: bool = False,
        eps: float = 1e-6,
        dtype: Any | str | None = None,
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
        self.dtype = dtype

        self._flow = flow
        self._sampler = None

    @property
    def flow(self):
        """The normalizing flow object."""
        return self._flow

    @flow.setter
    def flow(self, flow: Flow):
        """Set the normalizing flow object."""
        self._flow = flow

    @property
    def sampler(self) -> Sampler | None:
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
            dtype=self.dtype,
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
            dtype=self.dtype,
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
            dtype=self.dtype,
            **self.flow_kwargs,
        )

    def fit(
        self,
        samples: Samples,
        checkpoint_path: str | None = None,
        checkpoint_save_config: bool = True,
        overwrite: bool = False,
        **kwargs,
    ) -> History:
        """Fit the normalizing flow to the provided samples.

        Parameters
        ----------
        samples : Samples
            The samples to fit the flow to.
        checkpoint_path : str | None
            Path to save the checkpoint. If None, no checkpoint is saved.
        checkpoint_save_config : bool
            Whether to save the Aspire configuration to the checkpoint.
        overwrite : bool
            Whether to overwrite an existing flow in the checkpoint file.
        kwargs : dict
            Keyword arguments to pass to the flow's fit method.
        """
        if self.xp is None:
            self.xp = samples.xp

        if self.flow is None:
            self.init_flow()

        self.training_samples = samples
        logger.info(f"Training with {len(samples.x)} samples")
        history = self.flow.fit(samples.x, **kwargs)
        defaults = getattr(self, "_checkpoint_defaults", None)
        if checkpoint_path is None and defaults:
            checkpoint_path = defaults["path"]
            checkpoint_save_config = defaults["save_config"]
        saved_config = (
            defaults.get("saved_config", False) if defaults else False
        )
        if checkpoint_path is not None:
            with AspireFile(checkpoint_path, "a") as h5_file:
                if checkpoint_save_config and not saved_config:
                    if "aspire_config" in h5_file:
                        del h5_file["aspire_config"]
                    self.save_config(h5_file, include_sampler_config=False)
                    if defaults is not None:
                        defaults["saved_config"] = True
                # Save flow only if missing or overwrite=True
                if "flow" in h5_file:
                    if overwrite:
                        del h5_file["flow"]
                        self.save_flow(h5_file)
                else:
                    self.save_flow(h5_file)
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
        preconditioning: str
            Type of preconditioning to apply in the sampler. Options are
            'default', 'flow', or 'none'.
        preconditioning_kwargs: dict
            Keyword arguments to pass to the preconditioning transform.
        kwargs : dict
            Keyword arguments to pass to the sampler.
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
                dtype=self.dtype,
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
                dtype=self.dtype,
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
            dtype=self.dtype,
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
        checkpoint_path: str | None = None,
        checkpoint_every: int = 1,
        checkpoint_save_config: bool = True,
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
        checkpoint_path : str | None
            Path to save the checkpoint. If None, no checkpoint is saved unless
            within an :py:meth:`auto_checkpoint` context or a custom callback
            is provided.
        checkpoint_every : int
            Frequency (in number of sampler iterations) to save the checkpoint.
        checkpoint_save_config : bool
            Whether to save the Aspire configuration to the checkpoint.
        kwargs : dict
            Keyword arguments to pass to the sampler. These are passed
            automatically to the init method of the sampler or to the sample
            method.

        Returns
        -------
        samples : Samples
            Samples object contain samples and their corresponding weights.
        """
        if (
            sampler == "importance"
            and hasattr(self, "_resume_sampler_type")
            and self._resume_sampler_type
        ):
            sampler = self._resume_sampler_type

        if "resume_from" not in kwargs and hasattr(
            self, "_resume_from_default"
        ):
            kwargs["resume_from"] = self._resume_from_default
            if hasattr(self, "_resume_overrides"):
                kwargs.update(self._resume_overrides)
            if hasattr(self, "_resume_n_samples") and n_samples == 1000:
                n_samples = self._resume_n_samples

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
        self._last_sampler_type = sampler
        # Auto-checkpoint convenience: set defaults for checkpointing to a single file
        defaults = getattr(self, "_checkpoint_defaults", None)
        if checkpoint_path is None and defaults:
            checkpoint_path = defaults["path"]
            checkpoint_every = defaults["every"]
            checkpoint_save_config = defaults["save_config"]
        saved_flow = defaults.get("saved_flow", False) if defaults else False
        saved_config = (
            defaults.get("saved_config", False) if defaults else False
        )
        if checkpoint_path is not None:
            kwargs.setdefault("checkpoint_file_path", checkpoint_path)
            kwargs.setdefault("checkpoint_every", checkpoint_every)
            with AspireFile(checkpoint_path, "a") as h5_file:
                if checkpoint_save_config:
                    if "aspire_config" in h5_file:
                        del h5_file["aspire_config"]
                    self.save_config(
                        h5_file,
                        include_sampler_config=True,
                        include_sample_calls=False,
                    )
                    saved_config = True
                    if defaults is not None:
                        defaults["saved_config"] = True
                if (
                    self.flow is not None
                    and not saved_flow
                    and "flow" not in h5_file
                ):
                    self.save_flow(h5_file)
                    saved_flow = True
                    if defaults is not None:
                        defaults["saved_flow"] = True

        samples = self._sampler.sample(n_samples, **kwargs)
        self._last_sample_posterior_kwargs = {
            "n_samples": n_samples,
            "sampler": sampler,
            "xp": xp,
            "return_history": return_history,
            "preconditioning": preconditioning,
            "preconditioning_kwargs": preconditioning_kwargs,
            "sampler_init_kwargs": sampler_kwargs,
            "sample_kwargs": copy.deepcopy(kwargs),
        }
        if checkpoint_path is not None:
            with AspireFile(checkpoint_path, "a") as h5_file:
                if checkpoint_save_config and not saved_config:
                    if "aspire_config" in h5_file:
                        del h5_file["aspire_config"]
                    self.save_config(
                        h5_file,
                        include_sampler_config=True,
                        include_sample_calls=False,
                    )
                    if defaults is not None:
                        defaults["saved_config"] = True
                if (
                    self.flow is not None
                    and not saved_flow
                    and "flow" not in h5_file
                ):
                    self.save_flow(h5_file)
                    if defaults is not None:
                        defaults["saved_flow"] = True
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

    @classmethod
    def resume_from_file(
        cls,
        file_path: str,
        *,
        log_likelihood: Callable,
        log_prior: Callable,
        sampler: str | None = None,
        checkpoint_path: str = "checkpoint",
        checkpoint_dset: str = "state",
        flow_path: str = "flow",
        config_path: str = "aspire_config",
        resume_kwargs: dict | None = None,
    ):
        """
        Recreate an Aspire object from a single file and prepare to resume sampling.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file containing config, flow, and checkpoint.
        log_likelihood : Callable
            Log-likelihood function (required, not pickled).
        log_prior : Callable
            Log-prior function (required, not pickled).
        sampler : str
            Sampler type to use (e.g., 'smc', 'minipcn_smc', 'emcee_smc'). If None,
            will attempt to infer from saved config or checkpoint metadata.
        checkpoint_path : str
            HDF5 group path where the checkpoint is stored.
        checkpoint_dset : str
            Dataset name within the checkpoint group.
        flow_path : str
            HDF5 path to the saved flow.
        config_path : str
            HDF5 path to the saved Aspire config.
        resume_kwargs : dict | None
            Optional overrides to apply when resuming (e.g., checkpoint_every).
        """
        (
            aspire,
            checkpoint_bytes,
            checkpoint_state,
            sampler_config,
            saved_sampler_type,
            n_samples,
        ) = cls._build_aspire_from_file(
            file_path=file_path,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            checkpoint_path=checkpoint_path,
            checkpoint_dset=checkpoint_dset,
            flow_path=flow_path,
            config_path=config_path,
        )

        sampler_config = sampler_config or {}
        sampler_config.pop("sampler_class", None)

        if checkpoint_bytes is not None:
            aspire._resume_from_default = checkpoint_bytes
            aspire._resume_sampler_type = (
                sampler
                or saved_sampler_type
                or (
                    checkpoint_state.get("sampler")
                    if checkpoint_state
                    else None
                )
            )
            aspire._resume_n_samples = n_samples
            aspire._resume_overrides = resume_kwargs or {}
            aspire._resume_sampler_config = sampler_config
        aspire._checkpoint_defaults = {
            "path": file_path,
            "every": 1,
            "save_config": False,
            "save_flow": False,
            "saved_config": False,
            "saved_flow": False,
        }
        return aspire

    @contextmanager
    def auto_checkpoint(
        self,
        path: str,
        every: int = 1,
        save_config: bool = True,
        save_flow: bool = True,
    ):
        """
        Context manager to auto-save checkpoints, config, and flow to a file.

        Within the context, sample_posterior will default to writing checkpoints
        to the given path with the specified frequency, and will append config/flow
        after sampling.
        """
        prev = getattr(self, "_checkpoint_defaults", None)
        self._checkpoint_defaults = {
            "path": path,
            "every": every,
            "save_config": save_config,
            "save_flow": save_flow,
            "saved_config": False,
            "saved_flow": False,
        }
        try:
            yield self
        finally:
            if prev is None:
                if hasattr(self, "_checkpoint_defaults"):
                    delattr(self, "_checkpoint_defaults")
            else:
                self._checkpoint_defaults = prev

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
            "log_likelihood": function_id(self.log_likelihood),
            "log_prior": function_id(self.log_prior),
            "dims": self.dims,
            "parameters": self.parameters,
            "periodic_parameters": self.periodic_parameters,
            "prior_bounds": self.prior_bounds,
            "bounded_to_unbounded": self.bounded_to_unbounded,
            "bounded_transform": self.bounded_transform,
            "flow_matching": self.flow_matching,
            "device": self.device,
            "xp": self.xp.__name__ if self.xp else None,
            "flow_backend": self.flow_backend,
            "flow_kwargs": self.flow_kwargs,
            "eps": self.eps,
        }
        if hasattr(self, "_last_sampler_type"):
            config["sampler_type"] = self._last_sampler_type
        if include_sampler_config:
            if self.sampler is None:
                raise ValueError("Sampler has not been initialized.")
            config["sampler_config"] = self.sampler.config_dict(**kwargs)
        return config

    def save_config(
        self, h5_file: h5py.File | AspireFile, path="aspire_config", **kwargs
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

    def save_flow(self, h5_file: h5py.File, path="flow") -> None:
        """Save the flow to an HDF5 file.

        Parameters
        ----------
        h5_file : h5py.File
            The HDF5 file to save the flow to.
        path : str
            The path in the HDF5 file to save the flow to.
        """
        if self.flow is None:
            raise ValueError("Flow has not been initialized.")
        self.flow.save(h5_file, path=path)

    def load_flow(self, h5_file: h5py.File, path="flow") -> None:
        """Load the flow from an HDF5 file.

        Parameters
        ----------
        h5_file : h5py.File
            The HDF5 file to load the flow from.
        path : str
            The path in the HDF5 file to load the flow from.
        """
        FlowClass, xp = get_flow_wrapper(
            backend=self.flow_backend, flow_matching=self.flow_matching
        )
        logger.debug(f"Loading flow of type {FlowClass} from {path}")
        self._flow = FlowClass.load(h5_file, path=path)

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

    # --- Resume helpers ---
    @staticmethod
    def _build_aspire_from_file(
        file_path: str,
        log_likelihood: Callable,
        log_prior: Callable,
        checkpoint_path: str,
        checkpoint_dset: str,
        flow_path: str,
        config_path: str,
    ):
        """Construct an Aspire instance, load flow, and gather checkpoint metadata from file."""
        with AspireFile(file_path, "r") as h5_file:
            if config_path not in h5_file:
                raise ValueError(
                    f"Config path '{config_path}' not found in {file_path}"
                )
            config_dict = load_from_h5_file(h5_file, config_path)
            try:
                checkpoint_bytes = h5_file[checkpoint_path][checkpoint_dset][
                    ...
                ].tobytes()
            except Exception:
                logger.warning(
                    "Checkpoint not found at %s/%s in %s; will resume without a checkpoint.",
                    checkpoint_path,
                    checkpoint_dset,
                    file_path,
                )
                checkpoint_bytes = None

        sampler_config = config_dict.pop("sampler_config", None)
        saved_sampler_type = config_dict.pop("sampler_type", None)
        if isinstance(config_dict.get("xp"), str):
            config_dict["xp"] = resolve_xp(config_dict["xp"])
        config_dict["log_likelihood"] = log_likelihood
        config_dict["log_prior"] = log_prior

        aspire = Aspire(**config_dict)

        with AspireFile(file_path, "r") as h5_file:
            if flow_path in h5_file:
                logger.info(f"Loading flow from {flow_path} in {file_path}")
                aspire.load_flow(h5_file, path=flow_path)
            else:
                raise ValueError(
                    f"Flow path '{flow_path}' not found in {file_path}"
                )

        n_samples = None
        checkpoint_state = None
        if checkpoint_bytes is not None:
            try:
                checkpoint_state = pickle.loads(checkpoint_bytes)
                samples_saved = (
                    checkpoint_state.get("samples")
                    if checkpoint_state
                    else None
                )
                if samples_saved is not None:
                    n_samples = len(samples_saved)
                    if aspire.xp is None and hasattr(samples_saved, "xp"):
                        aspire.xp = samples_saved.xp
            except Exception:
                logger.warning(
                    "Failed to decode checkpoint; proceeding without resume state."
                )

        return (
            aspire,
            checkpoint_bytes,
            checkpoint_state,
            sampler_config,
            saved_sampler_type,
            n_samples,
        )
