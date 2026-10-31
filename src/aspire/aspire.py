import copy
import importlib
import logging
import multiprocessing as mp
import pickle
import warnings
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from typing import Any, Callable

import h5py
from array_api_compat import array_namespace
from sequince.checkpoint import SMCCheckpoint
from sequince.flows import (
    NormalizingFlow,
    flow_state_dict,
    load_flow_state_dict,
)
from sequince.transforms import (
    FlowTransform,
)

from .compat import LegacyFlow, _adapt_flow
from .history import FlowHistory, History
from .samplers.base import Sampler, load_checkpoint
from .samples import Samples
from .utils import (
    AspireFile,
    build_particle_transform,
    function_id,
    get_flow_class,
    load_from_h5_file,
    recursively_save_to_h5_file,
    resolve_dtype,
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
    flow : NormalizingFlow | LegacyFlow | None
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
        flow: NormalizingFlow | LegacyFlow | None = None,
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

        self._flow = _adapt_flow(flow) if flow is not None else None
        self._sampler = None

    @property
    def flow(self):
        """The normalizing flow object."""
        return self._flow

    @flow.setter
    def flow(self, flow: NormalizingFlow | LegacyFlow):
        """Set the normalizing flow object."""
        self._flow = _adapt_flow(flow)

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
        FlowClass = get_flow_class(
            backend=self.flow_backend, flow_matching=self.flow_matching
        )
        xp = FlowClass.xp

        particle_transform = build_particle_transform(
            parameters=self.parameters,
            prior_bounds=self.prior_bounds,
            periodic_parameters=None,
            bounded_to_unbounded=self.bounded_to_unbounded,
            bounded_transform=self.bounded_transform,
            eps=self.eps,
        )

        # Check if FlowClass takes `parameters` as an argument
        flow_init_params = signature(FlowClass.__init__).parameters
        if "parameters" in flow_init_params:
            self.flow_kwargs["parameters"] = self.parameters.copy()

        logger.info(f"Configuring {FlowClass} with kwargs: {self.flow_kwargs}")

        self._flow = FlowClass(
            dims=self.dims,
            device=self.device or "cpu",
            particle_transform=particle_transform,
            dtype=(
                resolve_dtype(self.dtype, xp)
                if self.dtype is not None
                else None
            ),
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

        if self.parameters is None and samples.parameters is not None:
            self.parameters = samples.parameters.copy()

        if self.flow is None:
            self.init_flow()
        elif getattr(self, "_skip_flow_training", False) and not overwrite:
            logger.info(
                "Skipping flow training because a checkpointed flow was loaded."
            )
            return FlowHistory()

        self.training_samples = samples
        logger.info(f"Training with {len(samples.x)} samples")
        fit_result = self.flow.fit(samples.x, **kwargs)
        if isinstance(fit_result, FlowHistory):
            history = fit_result
        else:
            fit_history = getattr(self.flow, "fit_history", {})
            history = FlowHistory(
                training_loss=list(fit_history.get("training_loss", [])),
                validation_loss=list(fit_history.get("validation_loss", [])),
            )
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
            from .samplers.smc import EmceeSMC as SamplerClass
        elif sampler_type == "blackjax_smc":
            from .samplers.smc import BlackJAXSMC as SamplerClass
        elif sampler_type == "minipcn":
            from .samplers.mcmc import MiniPCN as SamplerClass
        elif sampler_type == "smc":
            from .samplers.smc import SMCSampler as SamplerClass
        elif sampler_type == "minipcn_smc":
            from .samplers.smc import MiniPCNSMC as SamplerClass
        else:
            from importlib.metadata import entry_points

            # Fetch any custom sampler registered via an entry point in the
            # aspire.samplers group
            entry_points_dict = {
                ep.name: ep for ep in entry_points(group="aspire.samplers")
            }

            if sampler_type in entry_points_dict:
                SamplerClass = entry_points_dict[sampler_type].load()
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
            preconditioning_kwargs = dict(preconditioning_kwargs or {})
            preconditioning_kwargs.setdefault("affine_transform", False)
            preconditioning_kwargs.setdefault("bounded_to_unbounded", False)
            preconditioning_kwargs.setdefault("bounded_transform", "logit")
            transform = build_particle_transform(
                parameters=self.parameters,
                prior_bounds=self.prior_bounds,
                periodic_parameters=self.periodic_parameters,
                eps=self.eps,
                **preconditioning_kwargs,
            )
        elif preconditioning == "flow":
            preconditioning_kwargs = dict(preconditioning_kwargs or {})
            fit_kwargs = preconditioning_kwargs.pop("fit_kwargs", None)
            flow_kwargs = dict(
                preconditioning_kwargs.pop("flow_kwargs", self.flow_kwargs)
            )
            preconditioning_kwargs.setdefault("affine_transform", False)
            preconditioning_kwargs.setdefault(
                "bounded_to_unbounded", self.bounded_to_unbounded
            )
            preconditioning_kwargs.setdefault(
                "bounded_transform", self.bounded_transform
            )
            particle_transform = build_particle_transform(
                parameters=self.parameters,
                periodic_parameters=None,
                prior_bounds=self.prior_bounds,
                eps=self.eps,
                **preconditioning_kwargs,
            )
            FlowClass = get_flow_class(
                backend=self.flow_backend,
                flow_matching=self.flow_matching,
            )
            flow_xp = FlowClass.xp
            transform_flow = FlowClass(
                dims=self.dims,
                device=self.device or "cpu",
                particle_transform=particle_transform,
                dtype=(
                    resolve_dtype(self.dtype, flow_xp)
                    if self.dtype is not None
                    else None
                ),
                **flow_kwargs,
            )
            transform = FlowTransform(
                transform_flow,
                fit_kwargs=fit_kwargs,
            )
        else:
            raise ValueError(f"Unknown preconditioning: {preconditioning}")

        sampler = SamplerClass(
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
            dims=self.dims,
            proposal=self.flow,
            xp=self.xp,
            dtype=self.dtype,
            preconditioning_transform=transform,
            parameters=self.parameters,
            **kwargs,
        )
        return sampler

    def sample_posterior(
        self,
        n_samples: int | None = None,
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

        For SMC, mutation options belong in :code:`sampler_kwargs`.
        For standalone MCMC, pass chain and mutation options directly.
        For example:

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
        n_samples : int | None
            The number of sample to draw. If None, the behavior will depend on
            the sampler.
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
            HDF5 (.h5/.hdf5) or pickle (.pkl/.pickle) checkpoint path. Pickle
            files contain sampler state only; config and flow are saved only
            in HDF5 files. If None, no checkpoint is saved unless
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
            if hasattr(self, "_resume_n_samples") and n_samples is None:
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
        if kwargs.get("checkpoint_callback") is not None:
            kwargs.setdefault("checkpoint_every", checkpoint_every)
        pickle_checkpoint = checkpoint_path is not None and Path(
            checkpoint_path
        ).suffix.lower() in (".pkl", ".pickle")
        if checkpoint_path is not None:
            # Check if sampler supports checkpointing
            signatured_sample = signature(self._sampler.sample)
            if not {"checkpoint_file_path", "checkpoint_every"}.issubset(
                signatured_sample.parameters
            ):
                logger.warning(
                    f"Sampler {sampler} does not support checkpointing. Checkpoint will not be saved."
                )
            else:
                kwargs.setdefault("checkpoint_file_path", checkpoint_path)
                kwargs.setdefault("checkpoint_every", checkpoint_every)
            if not pickle_checkpoint:
                with AspireFile(checkpoint_path, "a") as h5_file:
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
        if checkpoint_path is not None and not pickle_checkpoint:
            with AspireFile(checkpoint_path, "a") as h5_file:
                if checkpoint_save_config and not saved_config:
                    if "aspire_config" in h5_file:
                        del h5_file["aspire_config"]
                    self.save_config(
                        h5_file,
                        include_sampler_config=False,
                    )
                    if "sampler_config" in h5_file:
                        del h5_file["sampler_config"]
                    self.save_sampler_config(
                        h5_file,
                        include_sample_calls="last",
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
            checkpoint,
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

        aspire._set_resume_defaults(
            checkpoint=checkpoint,
            checkpoint_state=checkpoint_state,
            sampler_config=sampler_config,
            saved_sampler_type=saved_sampler_type,
            n_samples=n_samples,
            sampler=sampler,
            resume_kwargs=resume_kwargs,
        )
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
        resume: bool = False,
    ):
        """
        Context manager to auto-save sampler checkpoints to a file.

        Within the context, sample_posterior will default to writing checkpoints
        to the given path with the specified frequency. HDF5 files also store
        Aspire configuration and the fitted flow; pickle files store only
        native Sequince sampler state.

        Parameters
        ----------
        path : str
            HDF5 (.h5/.hdf5) or pickle (.pkl/.pickle) checkpoint path.
            Pickle files store sampler state only; the existing Aspire
            configuration and fitted flow are used when resuming.
        every : int
            Frequency (in number of sampler iterations) to save the checkpoint.
        save_config : bool
            Whether to save the Aspire configuration in an HDF5 checkpoint.
        save_flow : bool
            Whether to save the flow in an HDF5 checkpoint.
        resume : bool
            Whether to attempt to resume from an existing checkpoint at the path.
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
        resume_attrs = [
            "_resume_from_default",
            "_resume_sampler_type",
            "_resume_n_samples",
            "_resume_overrides",
            "_resume_sampler_config",
            "_skip_flow_training",
        ]
        prev_resume_state = {
            attr: getattr(self, attr)
            for attr in resume_attrs
            if hasattr(self, attr)
        }
        path = Path(path)
        if resume and path.is_file():
            logger.info(f"Resuming from checkpoint file at {path}")
            (
                checkpoint,
                checkpoint_state,
                sampler_config,
                saved_sampler_type,
                n_samples,
                checkpoint_xp,
            ) = self._load_resume_data(
                path,
                checkpoint_path="checkpoint",
                checkpoint_dset="state",
                config_path="aspire_config",
            )
            requested_n_samples = self._resume_n_samples_from_sampler_config(
                sampler_config
            )
            if requested_n_samples is not None:
                n_samples = requested_n_samples
            if path.suffix.lower() not in (".pkl", ".pickle"):
                self._load_flow_from_file(
                    path,
                    flow_path="flow",
                    required=False,
                )
            self._set_resume_defaults(
                checkpoint=checkpoint,
                checkpoint_state=checkpoint_state,
                sampler_config=sampler_config,
                saved_sampler_type=(
                    saved_sampler_type
                    or getattr(self, "_last_sampler_type", None)
                ),
                n_samples=n_samples,
            )
            self._skip_flow_training = self.flow is not None
            if self.xp is None and checkpoint_xp is not None:
                self.xp = checkpoint_xp
        try:
            yield self
        finally:
            for attr in resume_attrs:
                if attr in prev_resume_state:
                    setattr(self, attr, prev_resume_state[attr])
                elif hasattr(self, attr):
                    delattr(self, attr)
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
        self, include_sampler_config: bool = False, **kwargs
    ) -> dict:
        """Return a dictionary with the configuration of the aspire object.

        Parameters
        ----------
        include_sampler_config : bool
            Whether to include the configuration of the sampler. Default is
            False.
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
        if include_sampler_config:
            if hasattr(self, "_last_sampler_type"):
                config["sampler_type"] = self._last_sampler_type
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

    def save_sampler_config(
        self,
        h5_file: h5py.File | AspireFile,
        path: str = "sampler_config",
        **kwargs,
    ) -> None:
        """Save the sampler configuration to an HDF5 file.

        By default, this will save the configuration of the last sampler used in
        a call to :py:meth:`sample_posterior`.
        Set :code:`include_sample_class='all' to include all the calls to
        :py:meth:`sample_posterior` and their corresponding sampler
        configurations.

        Parameters
        ----------
        h5_file : h5py.File
            The HDF5 file to save the sampler configuration to.
        path : str
            The path in the HDF5 file to save the sampler configuration to.
        kwargs : dict
            Additional keyword arguments to pass to the :py:meth:`config_dict`
            method of the sampler.
        """
        config = self.sampler.config_dict(**kwargs) if self.sampler else {}
        if hasattr(self, "_last_sampler_type"):
            config["sampler_type"] = self._last_sampler_type
        recursively_save_to_h5_file(
            h5_file,
            path,
            config,
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
        init_args = dict(getattr(self.flow, "_init_args", {}))
        payload = pickle.dumps(
            {
                "module": type(self.flow).__module__,
                "class": type(self.flow).__qualname__,
                "init_args": init_args,
                "state": flow_state_dict(self.flow),
            }
        )
        if path in h5_file:
            del h5_file[path]
        h5_file.create_dataset(path, data=list(payload), dtype="u1")

    def load_flow(self, h5_file: h5py.File, path="flow") -> None:
        """Load the flow from an HDF5 file.

        Parameters
        ----------
        h5_file : h5py.File
            The HDF5 file to load the flow from.
        path : str
            The path in the HDF5 file to load the flow from.
        """
        if isinstance(h5_file[path], h5py.Group):
            raise ValueError(
                "Legacy Aspire saved flows are not compatible with the "
                "Sequince Density backend. Refit the flow and start a new "
                "checkpoint file."
            )
        payload = pickle.loads(bytes(h5_file[path][()]))
        module = importlib.import_module(payload["module"])
        FlowClass = module
        for part in payload["class"].split("."):
            FlowClass = getattr(FlowClass, part)
        init_args = dict(payload["init_args"])
        extra_kwargs = init_args.pop("kwargs", {})
        self._flow = FlowClass(**extra_kwargs, **init_args)
        load_flow_state_dict(self._flow, payload["state"])

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
        samples = Samples(
            x=x,
            log_q=log_q,
            xp=xp,
            parameters=self.parameters,
            dtype=self.dtype,
        )
        return samples

    # --- Resume helpers ---
    @staticmethod
    def _load_resume_data(
        file_path: str,
        checkpoint_path: str,
        checkpoint_dset: str,
        config_path: str,
        sampler_config_path: str = "sampler_config",
    ) -> tuple[
        SMCCheckpoint | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        str | None,
        int | None,
        Any,
    ]:
        """Load a native checkpoint and any saved Aspire resume metadata."""
        if Path(file_path).suffix.lower() in (".pkl", ".pickle"):
            checkpoint = load_checkpoint(file_path)
            return (
                checkpoint,
                checkpoint.to_dict(),
                None,
                None,
                len(checkpoint.state.particles),
                array_namespace(checkpoint.state.particles),
            )
        with AspireFile(file_path, "r") as h5_file:
            config_dict = (
                load_from_h5_file(h5_file, config_path)
                if config_path in h5_file
                else None
            )
            standalone_sampler_config = (
                load_from_h5_file(h5_file, sampler_config_path)
                if sampler_config_path in h5_file
                else None
            )
            checkpoint_group = h5_file.get(checkpoint_path)
            if checkpoint_group is not None and (
                "runtime" in checkpoint_group
                or "runtime.previous" in checkpoint_group
            ):
                checkpoint = SMCCheckpoint.load_hdf5(
                    h5_file,
                    path=checkpoint_path,
                    dsetname="runtime",
                )
                checkpoint_state = checkpoint.to_dict()
                n_samples = len(checkpoint.state.particles)
                checkpoint_xp = array_namespace(checkpoint.state.particles)
            else:
                if (
                    checkpoint_group is not None
                    and checkpoint_dset in checkpoint_group
                ):
                    raise ValueError(
                        "Legacy Aspire checkpoints are not compatible with "
                        "the Sequince backend. Start a new sampling run."
                    )
                logger.warning(
                    "Checkpoint not found at %s/%s in %s; will resume "
                    "without a checkpoint.",
                    checkpoint_path,
                    checkpoint_dset,
                    file_path,
                )
                checkpoint = None
                checkpoint_state = None
                n_samples = None
                checkpoint_xp = None

        sampler_config = None
        saved_sampler_type = None
        used_legacy_sampler_config = False
        if config_dict is not None:
            saved_sampler_type = config_dict.get("sampler_type")
            sampler_config = config_dict.get("sampler_config")
            used_legacy_sampler_config = sampler_config is not None
        if standalone_sampler_config is not None:
            saved_sampler_type = (
                saved_sampler_type
                or standalone_sampler_config.get("sampler_type")
            )
            if sampler_config is None:
                sampler_config = dict(standalone_sampler_config)
                sampler_config.pop("sampler_type", None)
                used_legacy_sampler_config = False
        elif used_legacy_sampler_config:
            warnings.warn(
                (
                    f"Loaded sampler metadata from legacy '{config_path}' "
                    f"path in {file_path}; please migrate to "
                    f"'{sampler_config_path}'."
                ),
                DeprecationWarning,
                stacklevel=2,
            )

        return (
            checkpoint,
            checkpoint_state,
            sampler_config,
            saved_sampler_type,
            n_samples,
            checkpoint_xp,
        )

    def _set_resume_defaults(
        self,
        *,
        checkpoint: SMCCheckpoint | None,
        checkpoint_state: dict[str, Any] | None,
        sampler_config: dict[str, Any] | None,
        saved_sampler_type: str | None,
        n_samples: int | None,
        sampler: str | None = None,
        resume_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Configure default resume state for future sample_posterior calls."""
        if checkpoint is None:
            return

        sampler_config = sampler_config or {}
        sampler_config.pop("sampler_class", None)

        self._resume_from_default = checkpoint
        self._resume_sampler_type = (
            sampler
            or saved_sampler_type
            or (checkpoint_state.get("sampler") if checkpoint_state else None)
        )
        self._resume_n_samples = n_samples
        self._resume_overrides = resume_kwargs or {}
        self._resume_sampler_config = sampler_config

    @staticmethod
    def _resume_n_samples_from_sampler_config(
        sampler_config: dict | None,
    ) -> int | None:
        """Infer the original sample_posterior n_samples from saved sample calls."""
        if not isinstance(sampler_config, dict):
            return None

        sample_calls = sampler_config.get("sample_calls")
        if not isinstance(sample_calls, dict):
            return None

        sample_kwargs = sample_calls.get("kwargs")
        if (
            isinstance(sample_kwargs, dict)
            and sample_kwargs.get("n_final_samples") is not None
        ):
            try:
                return int(sample_kwargs["n_final_samples"])
            except (TypeError, ValueError):
                return None

        sample_args = sample_calls.get("args")
        if hasattr(sample_args, "__len__") and not isinstance(
            sample_args, (str, bytes, dict)
        ):
            if len(sample_args) == 0:
                sample_args = None
        else:
            sample_args = None

        if sample_args is not None:
            try:
                return int(sample_args[0])
            except (TypeError, ValueError):
                return None

        if isinstance(sample_kwargs, dict) and "n_samples" in sample_kwargs:
            try:
                return int(sample_kwargs["n_samples"])
            except (TypeError, ValueError):
                return None

        return None

    def _load_flow_from_file(
        self,
        file_path: str,
        flow_path: str = "flow",
        required: bool = True,
    ) -> bool:
        """Load a saved flow from file onto the current Aspire instance."""
        with AspireFile(file_path, "r") as h5_file:
            if flow_path in h5_file:
                logger.info(f"Loading flow from {flow_path} in {file_path}")
                self.load_flow(h5_file, path=flow_path)
                return True
        if required:
            raise ValueError(
                f"Flow path '{flow_path}' not found in {file_path}"
            )
        logger.warning(
            "Flow not found at %s in %s; continuing without loading a flow.",
            flow_path,
            file_path,
        )
        return False

    @classmethod
    def _build_aspire_from_file(
        cls,
        file_path: str,
        log_likelihood: Callable,
        log_prior: Callable,
        checkpoint_path: str,
        checkpoint_dset: str,
        flow_path: str,
        config_path: str,
    ) -> tuple[
        "Aspire",
        SMCCheckpoint | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        str | None,
        int | None,
    ]:
        """Construct an Aspire instance, load flow, and gather checkpoint metadata from file."""
        if Path(file_path).suffix.lower() in (".pkl", ".pickle"):
            raise ValueError(
                "Pickle checkpoints contain sampler state only. Resume on "
                "an existing Aspire instance with sample_posterior("
                "resume_from=file_path) or auto_checkpoint(resume=True)."
            )
        with AspireFile(file_path, "r") as h5_file:
            if config_path not in h5_file:
                raise ValueError(
                    f"Config path '{config_path}' not found in {file_path}"
                )
            config_dict = load_from_h5_file(h5_file, config_path)

        (
            checkpoint,
            checkpoint_state,
            sampler_config,
            saved_sampler_type,
            n_samples,
            checkpoint_xp,
        ) = Aspire._load_resume_data(
            file_path=file_path,
            checkpoint_path=checkpoint_path,
            checkpoint_dset=checkpoint_dset,
            config_path=config_path,
        )

        sampler_config = config_dict.pop("sampler_config", sampler_config)
        saved_sampler_type = config_dict.pop(
            "sampler_type", saved_sampler_type
        )
        if isinstance(config_dict.get("xp"), str):
            config_dict["xp"] = resolve_xp(config_dict["xp"])
        config_dict["log_likelihood"] = log_likelihood
        config_dict["log_prior"] = log_prior

        aspire = Aspire(**config_dict)

        aspire._load_flow_from_file(
            file_path,
            flow_path=flow_path,
            required=True,
        )

        if aspire.xp is None and checkpoint_xp is not None:
            aspire.xp = checkpoint_xp

        requested_n_samples = cls._resume_n_samples_from_sampler_config(
            sampler_config
        )
        if requested_n_samples is not None:
            n_samples = requested_n_samples

        return (
            aspire,
            checkpoint,
            checkpoint_state,
            sampler_config,
            saved_sampler_type,
            n_samples,
        )
