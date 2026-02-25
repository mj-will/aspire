from __future__ import annotations

import importlib
import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import numpy as np
from array_api_compat import (
    array_namespace,
)
from array_api_compat.common._typing import Array
from array_api_extra import default_dtype
from matplotlib.figure import Figure

from .utils import (
    asarray,
    convert_dtype,
    decode_dtype,
    encode_dtype,
    infer_device,
    logsumexp,
    recursively_save_to_h5_file,
    resolve_dtype,
    safe_to_device,
    to_numpy,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BaseSamples:
    """Class for storing samples and corresponding weights.

    If :code:`xp` is not specified, all inputs will be converted to match
    the array type of :code:`x`.
    """

    x: Array
    """Array of samples, shape (n_samples, n_dims)."""
    log_likelihood: Array | None = None
    """Log-likelihood values for the samples."""
    log_prior: Array | None = None
    """Log-prior values for the samples."""
    log_q: Array | None = None
    """Log-probability values under the proposal distribution."""
    parameters: list[str] | None = None
    """List of parameter names."""
    dtype: Any | str | None = None
    """Data type of the samples.

    If None, the default dtype for the array namespace will be used.
    """
    xp: Callable | None = None
    """
    The array namespace to use for the samples.

    If None, the array namespace will be inferred from the type of :code:`x`.
    """
    device: Any = None
    """Device to store the samples on.

    If None, the device will be inferred from the array namespace of :code:`x`.
    """

    def __post_init__(self):
        if self.xp is None:
            self.xp = array_namespace(self.x)
        # Numpy arrays need to be on the CPU before being converted
        if self.dtype is not None:
            self.dtype = resolve_dtype(self.dtype, self.xp)
        else:
            # Fall back to default dtype for the array namespace
            self.dtype = default_dtype(self.xp)
        self.x = self.array_to_namespace(self.x, dtype=self.dtype)
        if self.device is None:
            self.device = infer_device(self.x, self.xp)
        if self.log_likelihood is not None:
            self.log_likelihood = self.array_to_namespace(
                self.log_likelihood, dtype=self.dtype
            )
        if self.log_prior is not None:
            self.log_prior = self.array_to_namespace(
                self.log_prior, dtype=self.dtype
            )
        if self.log_q is not None:
            self.log_q = self.array_to_namespace(self.log_q, dtype=self.dtype)

        if self.parameters is None:
            self.parameters = [f"x_{i}" for i in range(self.dims)]

    @property
    def dims(self):
        """Number of dimensions (parameters) in the samples."""
        if self.x is None:
            return 0
        return self.x.shape[1] if self.x.ndim > 1 else 1

    def to_numpy(self, dtype: Any | str | None = None):
        logger.debug("Converting samples to numpy arrays")
        import array_api_compat.numpy as np

        if dtype is not None:
            dtype = resolve_dtype(dtype, np)
        else:
            dtype = convert_dtype(self.dtype, np)
        return self.__class__(
            x=self.x,
            parameters=self.parameters,
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
            log_q=self.log_q,
            xp=np,
        )

    def to_namespace(self, xp, dtype: Any | str | None = None):
        if dtype is None:
            dtype = convert_dtype(self.dtype, xp)
        else:
            dtype = resolve_dtype(dtype, xp)
        logger.debug("Converting samples to {} namespace", xp)
        return self.__class__(
            x=self.x,
            parameters=self.parameters,
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
            log_q=self.log_q,
            xp=xp,
            device=self.device,
            dtype=dtype,
        )

    def array_to_namespace(self, x, dtype=None):
        """Convert an array to the same namespace as the samples"""
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = resolve_dtype(dtype, self.xp)
        else:
            kwargs["dtype"] = self.dtype
        x = asarray(x, self.xp, **kwargs)
        x = safe_to_device(x, self.device, self.xp)
        return x

    def to_dict(self, flat: bool = True, copy: bool = True):
        """Convert the samples to a dictionary.

        Parameters
        ----------
        flat : bool
            If True, the samples are stored as separate keys for each parameter.
            If False, the samples are stored in a "samples" key as a dictionary
            of parameter arrays.
        copy : bool
            If True, the arrays in the dictionary are deep-copied. If False, they
            are not copied and may share memory with the original samples.

        Returns
        -------
        dict
            A dictionary representation of the samples.
        """
        out = {}
        for f in fields(self):
            name = f.name
            if name in ["x", "xp"]:
                continue
            value = getattr(self, name)
            if value is None:
                out[name] = None
            else:
                # This could be improved
                try:
                    out[name] = deepcopy(value) if copy else value
                except Exception:
                    out[name] = value

        out["xp"] = self.xp
        samples = dict(zip(self.parameters, self.x.T, strict=True))
        if flat:
            out.update(samples)
        else:
            out["samples"] = samples
        return out

    @classmethod
    def from_dict(cls, dictionary):
        """Load samples from a dictionary.

        The dictionary can either be in a flat format, where the samples are
        stored as separate keys for each parameter, or in a nested format, where
        the samples are stored in a "samples" key as a dictionary of parameter.
        """
        dictionary = dictionary.copy()
        if "samples" in dictionary:
            samples = dictionary.pop("samples")
            parameters = dictionary.pop("parameters")
            if parameters is None:
                parameters = sorted(samples.keys())
            x = np.stack([samples[p] for p in parameters], axis=-1)
        else:
            parameters = dictionary.pop("parameters")
            if parameters is None:
                raise ValueError(
                    "Parameters must be provided if samples are not nested in a 'samples' key"
                )
            x = np.stack([dictionary[p] for p in parameters], axis=-1)
            for p in parameters:
                dictionary.pop(p, None)
        return cls(x=x, parameters=parameters, **dictionary)

    def to_dataframe(self, include: list[str] | None = None) -> "pd.DataFrame":
        """Convert the samples to a pandas DataFrame.

        Only includes samples, log_likelihood, log_prior, and log_q by default,
        since additional fields have varying shapes and may not be compatible
        with a DataFrame format.

        Parameters
        ----------
        include : list[str] | None
            List of fields to include in the DataFrame. If None, includes x,
            log_likelihood, log_prior, and log_q. x is always included
            irrespective of the value of include.

        Returns
        -------
        pd.DataFrame
            A DataFrame representation of the samples.
        """
        import pandas as pd

        data = {}

        samples = dict(zip(self.parameters, self.x.T, strict=True))
        data.update(samples)

        if include is None:
            include = ["log_likelihood", "log_prior", "log_q"]

        for key in include:
            if getattr(self, key) is not None:
                data[key] = getattr(self, key)
            else:
                data[key] = np.full(len(self.x), np.nan)
        return pd.DataFrame(data)

    def plot_corner(
        self,
        parameters: list[str] | None = None,
        fig: Figure | None = None,
        **kwargs,
    ):
        """Plot a corner plot of the samples.

        Parameters
        ----------
        parameters : list[str] | None
            List of parameters to plot. If None, all parameters are plotted.
            Figure to plot on. If None, a new figure is created.
        **kwargs : dict
            Additional keyword arguments to pass to corner.corner(). Kwargs
            are deep-copied before use.
        """
        import corner

        kwargs = deepcopy(kwargs)
        kwargs.setdefault("labels", self.parameters)

        if parameters is not None:
            indices = [self.parameters.index(p) for p in parameters]
            kwargs["labels"] = parameters
            x = self.x[:, indices] if self.x.ndim > 1 else self.x[indices]
        else:
            x = self.x
        fig = corner.corner(to_numpy(x), fig=fig, **kwargs)
        return fig

    def __str__(self):
        out = (
            f"No. samples: {len(self.x)}\nNo. parameters: {self.x.shape[-1]}\n"
        )
        return out

    def _encode_for_hdf5(self, flat=True):
        """Encode the samples for storage in an HDF5 file."""
        dictionary = self.to_numpy().to_dict(flat=flat)
        dictionary["dtype"] = encode_dtype(self.xp, self.dtype)
        dictionary["xp"] = self.xp.__name__
        return dictionary

    def save(self, h5_file, path="samples", flat=False):
        """Save the samples to an HDF5 file.

        This converts the samples to numpy and then to a dictionary.

        Parameters
        ----------
        h5_file : h5py.File
            The HDF5 file to save to.
        path : str
            The path in the HDF5 file to save to.
        flat : bool
            If True, save the samples as a flat dictionary.
            If False, save the samples as a nested dictionary.
        """
        dictionary = self._encode_for_hdf5(flat=flat)
        recursively_save_to_h5_file(h5_file, path, dictionary)

    @classmethod
    def _decode_from_dictionary(cls, dictionary):
        """Decode the samples from a dictionary loaded from an HDF5 file."""
        dictionary["xp"] = importlib.import_module(dictionary["xp"])
        dictionary["dtype"] = decode_dtype(
            dictionary["xp"], dictionary["dtype"]
        )
        return cls.from_dict(dictionary)

    @classmethod
    def load(cls, h5_file, path="samples"):
        """Load the samples from an HDF5 file."""
        from .utils import load_from_h5_file

        dictionary = load_from_h5_file(h5_file, path)
        return cls._decode_from_dictionary(dictionary)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx) -> BaseSamples:
        return self.__class__(
            x=self.x[idx],
            log_likelihood=self.log_likelihood[idx]
            if self.log_likelihood is not None
            else None,
            log_prior=self.log_prior[idx]
            if self.log_prior is not None
            else None,
            log_q=self.log_q[idx] if self.log_q is not None else None,
            parameters=self.parameters,
            dtype=self.dtype,
        )

    def __setitem__(self, idx, value: BaseSamples):
        raise NotImplementedError("Setting items is not supported")

    @classmethod
    def concatenate(cls, samples: list[BaseSamples]) -> BaseSamples:
        """Concatenate multiple Samples objects."""
        if not samples:
            raise ValueError("No samples to concatenate")
        if not all(s.parameters == samples[0].parameters for s in samples):
            raise ValueError("Parameters do not match")
        if not all(s.xp == samples[0].xp for s in samples):
            raise ValueError("Array namespaces do not match")
        if not all(s.dtype == samples[0].dtype for s in samples):
            raise ValueError("Dtypes do not match")
        xp = samples[0].xp
        return cls(
            x=xp.concatenate([s.x for s in samples], axis=0),
            log_likelihood=xp.concatenate(
                [s.log_likelihood for s in samples], axis=0
            )
            if all(s.log_likelihood is not None for s in samples)
            else None,
            log_prior=xp.concatenate([s.log_prior for s in samples], axis=0)
            if all(s.log_prior is not None for s in samples)
            else None,
            log_q=xp.concatenate([s.log_q for s in samples], axis=0)
            if all(s.log_q is not None for s in samples)
            else None,
            parameters=samples[0].parameters,
            dtype=samples[0].dtype,
        )

    @classmethod
    def from_samples(cls, samples: BaseSamples, **kwargs) -> BaseSamples:
        """Create a Samples object from a BaseSamples object."""
        xp = kwargs.pop("xp", samples.xp)
        device = kwargs.pop("device", samples.device)
        dtype = kwargs.pop("dtype", samples.dtype)
        if dtype is not None:
            dtype = resolve_dtype(dtype, xp)
        else:
            dtype = convert_dtype(samples.dtype, xp)
        return cls(
            x=samples.x,
            log_likelihood=samples.log_likelihood,
            log_prior=samples.log_prior,
            log_q=samples.log_q,
            parameters=samples.parameters,
            xp=xp,
            device=device,
            **kwargs,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # replace xp (callable) with module name string
        if self.xp is not None:
            state["xp"] = (
                self.xp.__name__ if hasattr(self.xp, "__name__") else None
            )
        return state

    def __setstate__(self, state):
        # Restore xp by checking the namespace of x
        state["xp"] = array_namespace(state["x"])
        # device may be string; leave as-is or None
        device = state.get("device")
        if device is not None and "jax" in getattr(
            state["xp"], "__name__", ""
        ):
            device = None
        state["device"] = device
        self.__dict__.update(state)


@dataclass
class Samples(BaseSamples):
    """Class for storing samples and corresponding weights.

    If :code:`xp` is not specified, all inputs will be converted to match
    the array type of :code:`x`.
    """

    log_w: Array = field(init=False)
    weights: Array = field(init=False)
    evidence: float = field(init=False)
    evidence_error: float = field(init=False)
    log_evidence: float | None = None
    log_evidence_error: float | None = None
    effective_sample_size: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        if all(
            x is not None
            for x in [self.log_likelihood, self.log_prior, self.log_q]
        ):
            self.compute_weights()
        else:
            self.log_w = None
            self.weights = None
            self.evidence = None
            self.evidence_error = None
            self.effective_sample_size = None

    @property
    def efficiency(self):
        """Efficiency of the weighted samples.

        Defined as ESS / number of samples.
        """
        if self.log_w is None:
            raise RuntimeError("Samples do not contain weights!")
        return self.effective_sample_size / len(self.x)

    def compute_weights(self):
        """Compute the posterior weights."""
        self.log_w = self.log_likelihood + self.log_prior - self.log_q
        self.log_evidence = asarray(logsumexp(self.log_w), self.xp) - math.log(
            len(self.x)
        )
        self.weights = self.xp.exp(self.log_w)
        self.evidence = self.xp.exp(self.log_evidence)
        n = len(self.x)
        self.evidence_error = self.xp.sqrt(
            self.xp.sum((self.weights - self.evidence) ** 2) / (n * (n - 1))
        )
        self.log_evidence_error = self.xp.abs(
            self.evidence_error / self.evidence
        )
        log_w = self.log_w - self.xp.max(self.log_w)
        self.effective_sample_size = self.xp.exp(
            asarray(logsumexp(log_w) * 2 - logsumexp(log_w * 2), self.xp)
        )

    @property
    def scaled_weights(self):
        return self.xp.exp(self.log_w - self.xp.max(self.log_w))

    def rejection_sample(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        log_u = asarray(
            np.log(rng.uniform(size=len(self.x))), self.xp, device=self.device
        )
        log_w = self.log_w - self.xp.max(self.log_w)
        accept = log_w > log_u
        return self.__class__(
            x=self.x[accept],
            log_likelihood=self.log_likelihood[accept],
            log_prior=self.log_prior[accept],
            dtype=self.dtype,
        )

    def plot_corner(self, include_weights: bool = True, **kwargs):
        kwargs = deepcopy(kwargs)
        if (
            include_weights
            and self.weights is not None
            and "weights" not in kwargs
        ):
            kwargs["weights"] = to_numpy(self.scaled_weights)
        return super().plot_corner(**kwargs)

    def __str__(self):
        out = super().__str__()
        if self.log_evidence is not None:
            out += f"Log evidence: {self.log_evidence:.2f} +/- {self.log_evidence_error:.2f}\n"
        if self.log_w is not None:
            out += (
                f"Effective sample size: {self.effective_sample_size:.1f}\n"
                f"Efficiency: {self.efficiency:.2f}\n"
            )
        return out

    def to_namespace(self, xp):
        return self.__class__(
            x=asarray(self.x, xp, dtype=self.dtype),
            parameters=self.parameters,
            log_likelihood=asarray(self.log_likelihood, xp, dtype=self.dtype)
            if self.log_likelihood is not None
            else None,
            log_prior=asarray(self.log_prior, xp, dtype=self.dtype)
            if self.log_prior is not None
            else None,
            log_q=asarray(self.log_q, xp, dtype=self.dtype)
            if self.log_q is not None
            else None,
            log_evidence=asarray(self.log_evidence, xp, dtype=self.dtype)
            if self.log_evidence is not None
            else None,
            log_evidence_error=asarray(
                self.log_evidence_error, xp, dtype=self.dtype
            )
            if self.log_evidence_error is not None
            else None,
        )

    def to_numpy(self):
        return self.__class__(
            x=to_numpy(self.x),
            parameters=self.parameters,
            log_likelihood=to_numpy(self.log_likelihood)
            if self.log_likelihood is not None
            else None,
            log_prior=to_numpy(self.log_prior)
            if self.log_prior is not None
            else None,
            log_q=to_numpy(self.log_q) if self.log_q is not None else None,
            log_evidence=self.log_evidence
            if self.log_evidence is not None
            else None,
            log_evidence_error=self.log_evidence_error
            if self.log_evidence_error is not None
            else None,
        )

    def to_dataframe(self, include: list[str] | None = None) -> "pd.DataFrame":
        """Convert the samples to a pandas DataFrame.

        By default, includes log_likelihood, log_prior, log_q, and log_w.
        See parent class for more details.

        Parameters
        ----------
        include : list[str] | None
            List of fields to include in the DataFrame. If None, includes
            log_likelihood, log_prior, log_q, and log_w.

        Returns
        -------
        pd.DataFrame
            A DataFrame representation of the samples.
        """
        if include is None:
            include = ["log_likelihood", "log_prior", "log_q", "log_w"]
        return super().to_dataframe(include)

    def __getitem__(self, idx):
        sliced = super().__getitem__(idx)
        sliced.log_evidence = self.log_evidence
        sliced.log_evidence_error = self.log_evidence_error

        if self.log_w is not None:
            sliced.log_w = self.array_to_namespace(self.log_w[idx])
            if self.weights is not None:
                sliced.weights = self.array_to_namespace(self.weights[idx])
            else:
                sliced.weights = sliced.xp.exp(sliced.log_w)
            log_w = sliced.log_w - sliced.xp.max(sliced.log_w)
            sliced.effective_sample_size = sliced.xp.exp(
                asarray(logsumexp(log_w) * 2 - logsumexp(log_w * 2), sliced.xp)
            )
        return sliced


@dataclass
class MCMCSamples(BaseSamples):
    """Class for storing MCMC samples and chain metadata.

    Samples are stored flattened in :code:`x`, with :code:`chain_shape`
    capturing the original chain layout (excluding the parameter dimension).
    """

    chain_shape: tuple[int, ...] | None = None
    """Shape of the chain excluding the parameter dimension."""
    thin: int | None = None
    """Thinning factor used to produce the chain, if any."""
    burn_in: int | None = None
    """Number of burn-in steps removed, if any."""
    autocorrelation_time: Array | None = None
    """Autocorrelation time per dimension, if available."""
    minimum_chain_ndim: ClassVar[int] = 2
    """Minimum required dimensionality for the input chain."""

    def __post_init__(self):
        super().__post_init__()
        if self.chain_shape is None:
            self.chain_shape = (len(self.x),)
        else:
            expected = int(np.prod(self.chain_shape))
            if expected != len(self.x):
                raise ValueError(
                    "chain_shape does not match the number of samples"
                )

    @classmethod
    def from_chain(
        cls,
        chain,
        log_likelihood=None,
        log_prior=None,
        log_q=None,
        parameters=None,
        xp=None,
        dtype: Any | str | None = None,
        device: Any = None,
        thin: int | None = None,
        burn_in: int | None = None,
        autocorrelation_time: float | None = None,
        **extra_kwargs,
    ) -> "MCMCSamples":
        """Create samples from a chain array.

        Parameters
        ----------
        chain
            MCMC chain with shape (..., n_dims). For a single chain, this
            is typically (n_steps, n_dims). For ensemble samplers, use
            (n_steps, n_walkers, n_dims).
        log_likelihood, log_prior, log_q
            Optional arrays matching the chain shape without the last
            dimension. They will be flattened to align with :code:`x`.
        parameters
            Optional list of parameter names.
        xp
            Optional array namespace. If None, inferred from :code:`chain`.
        dtype
            Optional dtype to use for stored arrays.
        device
            Optional device to place arrays on.
        thin, burn_in
            Optional metadata describing how the chain was processed.
        """
        if xp is None:
            xp = array_namespace(chain)
        chain = asarray(chain, xp)
        if chain.ndim < cls.minimum_chain_ndim:
            raise ValueError(
                f"chain must have at least {cls.minimum_chain_ndim} dimensions"
            )
        chain_shape = chain.shape[:-1]
        dims = chain.shape[-1]
        x = xp.reshape(chain, (-1, dims))
        kwargs = dict(
            x=x,
            log_likelihood=cls._flatten_chain_values(log_likelihood, xp),
            log_prior=cls._flatten_chain_values(log_prior, xp),
            log_q=cls._flatten_chain_values(log_q, xp),
            parameters=parameters,
            xp=xp,
            device=device,
            dtype=dtype,
            chain_shape=chain_shape,
            thin=thin,
            burn_in=burn_in,
            autocorrelation_time=autocorrelation_time,
        )
        kwargs.update(extra_kwargs)
        return cls(**kwargs)

    @staticmethod
    def _flatten_chain_values(values, xp):
        if values is None:
            return None
        values = asarray(values, xp)
        return xp.reshape(values, (-1,))

    @property
    def chain(self):
        """The chain reshaped to its original layout"""
        return self.xp.reshape(self.x, (*self.chain_shape, self.dims))

    @property
    def n_steps(self) -> int:
        """Number of steps in the chain, excluding walkers"""
        return self.chain_shape[0]

    @property
    def n_chains(self) -> int:
        """Number of parallel chains (e.g. walkers), if applicable."""
        return self.chain_shape[1] if len(self.chain_shape) > 1 else 1

    def _reshape_like_chain(self, values):
        if values is None:
            return None
        return self.xp.reshape(values, self.chain_shape)

    def _chain_post_process_index(self, burn_in: int, thin: int):
        return slice(burn_in, None, thin)

    def _post_process_constructor_kwargs(self) -> dict[str, Any]:
        return {}

    def post_process(self, burn_in: int = 0, thin: int = 1) -> MCMCSamples:
        """Return a new MCMCSamples object with burn-in discarded and/or thinned."""
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if thin <= 0:
            raise ValueError("thin must be a positive integer")
        if burn_in == 0 and thin == 1:
            logger.warning(
                "No burn-in or thinning applied, returning original samples"
            )
            return self  # No processing needed
        index = self._chain_post_process_index(burn_in, thin)
        chain = self.chain[index]
        chain_shape = (chain.shape[0],) + chain.shape[1:-1]
        x = chain.reshape((-1, self.dims))
        log_likelihood = (
            self._reshape_like_chain(self.log_likelihood)[index].reshape(-1)
            if self.log_likelihood is not None
            else None
        )
        log_prior = (
            self._reshape_like_chain(self.log_prior)[index].reshape(-1)
            if self.log_prior is not None
            else None
        )
        log_q = (
            self._reshape_like_chain(self.log_q)[index].reshape(-1)
            if self.log_q is not None
            else None
        )
        return self.__class__(
            x=x,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            log_q=log_q,
            parameters=self.parameters,
            xp=self.xp,
            device=self.device,
            dtype=self.dtype,
            chain_shape=chain_shape,
            thin=self.thin * thin if self.thin is not None else thin,
            burn_in=(self.burn_in + burn_in)
            if self.burn_in is not None
            else burn_in,
            autocorrelation_time=self.autocorrelation_time,
            **self._post_process_constructor_kwargs(),
        )

    def __getitem__(self, idx):
        sliced = super().__getitem__(idx)
        sliced.chain_shape = (len(sliced.x),)
        sliced.thin = self.thin
        sliced.burn_in = self.burn_in
        sliced.autocorrelation_time = self.autocorrelation_time
        return sliced


@dataclass
class PTMCMCSamples(MCMCSamples):
    """Class for storing parallel-tempered MCMC samples."""

    betas: Array | None = None
    """Inverse temperatures for the chains.

    Should be a 1D array of shape (n_temps,) in decreasing order, starting at 1.
    """

    minimum_chain_ndim: ClassVar[int] = 3
    """Minimum required dimensionality for the PTMCMC input chain."""

    def __post_init__(self):
        super().__post_init__()
        # Ensure beta are decreasing and match the number of temperatures in the chain
        if self.betas is not None:
            self.betas = self.array_to_namespace(self.betas, dtype=self.dtype)
            if self.betas.ndim != 1:
                raise ValueError("betas must be a 1D array")
            if len(self.betas) != self.n_temps:
                raise ValueError(
                    "Length of betas must match the number of temperatures in the chain"
                )
            if not (self.betas[0] == 1):
                raise ValueError("betas must start at 1")
            if not self.xp.all(self.xp.diff(self.betas) < 0):
                raise ValueError("betas must be in decreasing order")

    @classmethod
    def from_chain(
        cls,
        chain,
        betas=None,
        log_likelihood=None,
        log_prior=None,
        log_q=None,
        parameters=None,
        xp=None,
        dtype: Any | str | None = None,
        device: Any = None,
        thin: int | None = None,
        burn_in: int | None = None,
        autocorrelation_time: float | None = None,
    ) -> "PTMCMCSamples":
        """Create samples from a parallel-tempered MCMC chain.

        Parameters
        ----------
        chain
            PTMCMC chain with shape (n_temps, ..., n_dims). A common layout
            is (n_temps, n_steps, n_walkers, n_dims).
        betas
            Optional inverse temperatures with shape (n_temps,).
        log_likelihood, log_prior, log_q
            Optional arrays matching the chain shape without the last
            dimension. They will be flattened to align with :code:`x`.
        parameters
            Optional list of parameter names.
        xp
            Optional array namespace. If None, inferred from :code:`chain`.
        dtype
            Optional dtype to use for stored arrays.
        device
            Optional device to place arrays on.
        thin, burn_in
            Optional metadata describing how the chain was processed.
        """
        return super().from_chain(
            chain=chain,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            log_q=log_q,
            parameters=parameters,
            xp=xp,
            dtype=dtype,
            device=device,
            betas=betas,
            thin=thin,
            burn_in=burn_in,
            autocorrelation_time=autocorrelation_time,
        )

    @property
    def n_temps(self) -> int:
        """Number of temperatures in the parallel-tempered chain."""
        return self.chain_shape[0]

    def _chain_post_process_index(self, burn_in: int, thin: int):
        return (slice(None), slice(burn_in, None, thin))

    def _post_process_constructor_kwargs(self) -> dict[str, Any]:
        return {"betas": self.betas}

    def at_temperature(self, index: int) -> MCMCSamples:
        chain = self.chain[index]
        log_likelihood = self._reshape_like_chain(self.log_likelihood)
        log_prior = self._reshape_like_chain(self.log_prior)
        log_q = self._reshape_like_chain(self.log_q)
        return MCMCSamples.from_chain(
            chain=chain,
            log_likelihood=None
            if log_likelihood is None
            else log_likelihood[index],
            log_prior=None if log_prior is None else log_prior[index],
            log_q=None if log_q is None else log_q[index],
            parameters=self.parameters,
            xp=self.xp,
            dtype=self.dtype,
            device=self.device,
            thin=self.thin,
            burn_in=self.burn_in,
            autocorrelation_time=self.autocorrelation_time[index]
            if self.autocorrelation_time is not None
            else None,
        )

    def cold_chain(self) -> MCMCSamples:
        return self.at_temperature(0)

    def log_evidence_thermodynamic_integration(
        self, burn_in_fraction: float | None = 0.1, method: str = "variance"
    ) -> tuple[float, float]:
        """Compute the log evidence using thermodynamic integration.

        Notes
        -----
        By default, follows the implementation outlined in Section 2.1.3 of
        Annis et al. (2019) [1]_. If method="variance", the error is estimated
        using the variance of the TI estimate across samples as in Eq. (37).
        If method="coarse", the error is estimated by comparing to a coarser
        integration using every other temperature, as in the original ptemcee
        implementation.

        .. [1] Annis, J., et al. https://doi.org/10.1016/j.jmp.2019.01.005

        Parameters
        ----------
        burn_in_fraction
            Fraction of initial samples to discard as burn-in. If None, no burn-in is discarded.
            This is applied independently to each temperature chain before integration.
        method
            Method for estimating the uncertainty in the log evidence.
            Options are "variance" or "coarse". See Notes for details.

        Returns
        -------
        log_evidence
            The estimated log evidence from thermodynamic integration.
        log_evidence_error
            An estimate of the uncertainty in the log evidence.
        """
        if self.betas is None:
            raise ValueError("Betas must be provided to compute evidence")
        logl_chain = self._reshape_like_chain(self.log_likelihood)
        istart = (
            int(logl_chain.shape[1] * burn_in_fraction)
            if burn_in_fraction is not None
            else 0
        )
        # Discard burn-in and flatten chain dimensions to a per-temperature sample axis.
        logl_chain = logl_chain[:, istart:]
        logl_chain = logl_chain.reshape(logl_chain.shape[0], -1)
        if logl_chain.shape[1] == 0:
            raise ValueError(
                "No samples available after burn-in for TI evidence"
            )

        # Integrate from low to high temperature as in Eq. (35).
        order = np.argsort(self.betas)
        betas = self.betas[order]
        logls = logl_chain[order]

        mean_logls = np.mean(logls, axis=1)
        log_evidence = np.trapezoid(mean_logls, betas)
        if method == "variance":
            # Eq. (36): TI_i for each aligned sample index across temperatures.
            ti_per_sample = np.trapezoid(logls, betas, axis=0)
            # Eq. (37): Var(mu_TI) = Var(TI) / n.
            n = ti_per_sample.shape[0]
            var_mu_ti = np.var(ti_per_sample) / n
            log_evidence_error = math.sqrt(float(var_mu_ti))
        elif method == "coarse":
            # Alternative error estimate by comparing to a coarser integration using every other temperature.
            # Copied from the original implementation in ptemcee
            betas = betas[::-1]
            logls = mean_logls[::-1]
            betas0 = np.copy(betas)
            if betas[-1] != 0:
                logger.warning(
                    "Hottest chain is not at beta=0, duplicating hottest chain with beta=0 for error estimation"
                )
                betas = np.concatenate((betas0, [0]))
                betas2 = np.concatenate((betas0[::2], [0]))

                # Duplicate mean log-likelihood of hottest chain as a best guess for beta = 0.
                logls2 = np.concatenate((logls[::2], [logls[-1]]))
                logls = np.concatenate((logls, [logls[-1]]))
            else:
                betas2 = np.concatenate((betas0[:-1:2], [0]))
                logls2 = np.concatenate((logls[:-1:2], [logls[-1]]))

            log_evidence_2 = -np.trapz(logls2, betas2)
            log_evidence_error = abs(log_evidence - log_evidence_2)
        else:
            raise ValueError(
                f"Invalid method for log evidence error estimation: {method}"
            )

        return float(log_evidence), float(log_evidence_error)

    def log_evidence_stepping_stone(
        self, burn_in_fraction: float | None = 0.1
    ) -> float:
        """Compute the log evidence using the stepping-stone estimator.

        Notes
        -----
        Follows the implementation outlined in Section 2.2.3 of Annis et al. (2019) [1]_

        .. [1] Annis, J., et al. https://doi.org/10.1016/j.jmp.2019.01.005

        Parameters
        ----------
        burn_in_fraction
            Fraction of initial samples to discard as burn-in. If None, no burn-in is discarded.
            This is applied independently to each temperature chain before integration.

        Returns
        -------
        log_evidence
            The estimated log evidence from thermodynamic integration.
        log_evidence_error
            An estimate of the uncertainty in the log evidence.
        """
        if self.betas is None:
            raise ValueError("Betas must be provided to compute evidence")
        if self.betas[-1] != 0:
            raise ValueError(
                "Stepping stone estimator requires the hottest chain to be at beta=0"
            )
        logl_chain = self._reshape_like_chain(self.log_likelihood)
        istart = (
            int(logl_chain.shape[1] * burn_in_fraction)
            if burn_in_fraction is not None
            else 0
        )
        # Discard burn-in steps
        logl_chain = logl_chain[:, istart:]
        # Combine the walker and step dimensions for easier indexing but
        # keep the temperature dimension separate
        logl_chain = logl_chain.reshape(logl_chain.shape[0], -1)
        order = np.argsort(self.betas)[::-1]
        betas = self.betas[order]
        logls = logl_chain[order]

        log_evidence = 0.0
        var_log_ss = 0.0
        n_samples = logls.shape[1]
        if n_samples == 0:
            raise ValueError(
                "No samples available after burn-in for stepping-stone evidence"
            )
        for i in range(len(betas) - 1):
            dbeta = betas[i] - betas[i + 1]  # positive
            # Equation (51): log r_j = log(mean(exp(dbeta * logL)))).
            a = dbeta * logls[i + 1]
            a_max = np.max(a)
            exp_shift = np.exp(a - a_max)
            mean_shift = float(np.mean(exp_shift))
            log_evidence += math.log(mean_shift) + float(a_max)

            # Equation (53): Var(log SS) = (1/n^2) * sum_j sum_i (exp(a_i)/r_j)^2
            ratio = exp_shift / mean_shift
            var_log_ss += float(np.sum(ratio**2))

        var_log_ss /= n_samples**2
        return float(log_evidence), math.sqrt(float(var_log_ss))

    def plot_chain(
        self,
        beta_index: int,
        n_walkers: int | None = None,
        burn_in: int = 0,
        parameters: list[str] | None = None,
        fig: Figure | None = None,
        **kwargs,
    ):
        import matplotlib.pyplot as plt

        chain = self.chain

        if parameters is not None:
            if self.parameters is None:
                raise ValueError(
                    "Cannot specify parameters to plot if samples do not have parameter names"
                )
            param_indices = [self.parameters.index(p) for p in parameters]
        else:
            param_indices = range(chain.shape[-1])

        fig, axs = plt.subplots(len(param_indices), 1, sharex=True)
        for count, idx in enumerate(param_indices):
            axs[count].plot(chain[beta_index, :, :n_walkers, idx], **kwargs)
            axs[count].axvline(burn_in, color="r", linestyle="--")
        fig.suptitle(f"$\\beta = {self.betas[beta_index]}$")
        return fig

    def __getitem__(self, idx):
        chain = self.chain[:, idx, ...]
        log_likelihood = self._reshape_like_chain(self.log_likelihood)
        log_prior = self._reshape_like_chain(self.log_prior)
        log_q = self._reshape_like_chain(self.log_q)
        return self.__class__.from_chain(
            chain=chain,
            betas=self.betas,
            log_likelihood=None
            if log_likelihood is None
            else log_likelihood[:, idx, ...],
            log_prior=None if log_prior is None else log_prior[:, idx, ...],
            log_q=None if log_q is None else log_q[:, idx, ...],
            parameters=self.parameters,
            xp=self.xp,
            dtype=self.dtype,
            device=self.device,
            thin=self.thin,
            burn_in=self.burn_in,
            autocorrelation_time=self.autocorrelation_time,
        )


@dataclass
class SMCSamples(BaseSamples):
    beta: float | None = None
    """Temperature parameter for the current samples."""
    log_evidence: float | None = None
    """Log evidence estimate for the current samples."""
    log_evidence_error: float | None = None
    """Log evidence error estimate for the current samples."""

    def log_p_t(self, beta):
        log_p_T = self.log_likelihood + self.log_prior
        return (1 - beta) * self.log_q + beta * log_p_T

    def unnormalized_log_weights(self, beta: float) -> Array:
        return (self.beta - beta) * self.log_q + (beta - self.beta) * (
            self.log_likelihood + self.log_prior
        )

    def log_evidence_ratio(self, beta: float) -> float:
        log_w = self.unnormalized_log_weights(beta)
        return logsumexp(log_w) - math.log(len(self.x))

    def log_evidence_ratio_variance(self, beta: float) -> float:
        """Estimate the variance of the log evidence ratio using the delta method.

        Defined as Var(log Z) = Var(w) / (E[w])^2 where w are the unnormalized weights.
        """
        log_w = self.unnormalized_log_weights(beta)
        m = self.xp.max(log_w)
        u = self.xp.exp(log_w - m)
        mean_w = self.xp.mean(u)
        var_w = self.xp.var(u)
        return (
            var_w / (len(self) * (mean_w**2)) if mean_w != 0 else self.xp.nan
        )

    def log_weights(self, beta: float) -> Array:
        log_w = self.unnormalized_log_weights(beta)
        if self.xp.isnan(log_w).any():
            raise ValueError(f"Log weights contain NaN values for beta={beta}")
        log_evidence_ratio = logsumexp(log_w) - math.log(len(self.x))
        return log_w + log_evidence_ratio

    def resample(
        self,
        beta,
        n_samples: int | None = None,
        rng: np.random.Generator = None,
    ) -> "SMCSamples":
        if beta == self.beta and n_samples is None:
            logger.warning(
                "Resampling with the same beta value, returning identical samples"
            )
            return self
        if rng is None:
            rng = np.random.default_rng()
        if n_samples is None:
            n_samples = len(self.x)
        log_w = self.log_weights(beta)
        w = to_numpy(self.xp.exp(log_w - logsumexp(log_w)))
        idx = rng.choice(len(self.x), size=n_samples, replace=True, p=w)
        return self.__class__(
            x=self.x[idx],
            log_likelihood=self.log_likelihood[idx],
            log_prior=self.log_prior[idx],
            log_q=self.log_q[idx],
            beta=beta,
            dtype=self.dtype,
            parameters=self.parameters,
        )

    def __str__(self):
        out = super().__str__()
        if self.log_evidence is not None:
            out += f"Log evidence: {self.log_evidence:.2f}\n"
        return out

    def to_standard_samples(self):
        """Convert the samples to standard samples."""
        return Samples(
            x=self.x,
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
            xp=self.xp,
            parameters=self.parameters,
            log_evidence=self.log_evidence,
            log_evidence_error=self.log_evidence_error,
        )

    def to_numpy(self):
        return self.__class__(
            x=to_numpy(self.x),
            parameters=self.parameters,
            log_likelihood=to_numpy(self.log_likelihood)
            if self.log_likelihood is not None
            else None,
            log_prior=to_numpy(self.log_prior)
            if self.log_prior is not None
            else None,
            log_q=to_numpy(self.log_q) if self.log_q is not None else None,
            beta=self.beta,
            log_evidence=self.log_evidence
            if self.log_evidence is not None
            else None,
            log_evidence_error=self.log_evidence_error
            if self.log_evidence_error is not None
            else None,
        )

    def __getitem__(self, idx):
        sliced = super().__getitem__(idx)
        sliced.beta = self.beta
        sliced.log_evidence = self.log_evidence
        sliced.log_evidence_error = self.log_evidence_error
        return sliced
