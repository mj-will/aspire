import logging
from typing import Any, Protocol, runtime_checkable

import array_api_compat.numpy as np

from .utils import asarray, to_numpy

logger = logging.getLogger(__name__)


@runtime_checkable
class Proposal(Protocol):
    """Interface for proposal distributions used by samplers.

    A proposal must at minimum expose ``sample_and_log_prob`` and
    ``log_prob``. Optional methods such as ``fit`` or ``save``/``load``
    are used when available.
    """

    xp: Any  # Array namespace used internally

    def sample_and_log_prob(self, n_samples: int): ...

    def log_prob(self, x): ...


class GaussianProposal:
    """Multivariate Gaussian proposal distribution.

    Parameters
    ----------
    dims : int
        Number of dimensions.
    mean : array-like | None
        Mean vector. If None, must call :meth:`fit` before sampling.
    cov : array-like | None
        Covariance matrix. If None, must call :meth:`fit` before sampling.
    frozen : bool
        If True, calling :meth:`fit` is a no-op and the pre-specified
        ``mean``/``cov`` are never overwritten. Default is False.
    """

    xp = np

    def __init__(self, dims: int, mean=None, cov=None, frozen: bool = False):
        self.dims = dims
        self.mean = np.asarray(mean) if mean is not None else None
        self.cov = np.asarray(cov) if cov is not None else None
        self.frozen = frozen
        self._validate_parameters()

    def _validate_parameters(self):
        if self.dims < 1:
            raise ValueError("dims must be a positive integer.")
        if (self.mean is None) != (self.cov is None):
            raise ValueError(
                "mean and cov must either both be set or both be None."
            )
        if self.mean is not None and self.mean.shape != (self.dims,):
            raise ValueError(
                f"mean must have shape ({self.dims},), got {self.mean.shape}."
            )
        if self.cov is not None and self.cov.shape != (
            self.dims,
            self.dims,
        ):
            raise ValueError(
                "cov must have shape "
                f"({self.dims}, {self.dims}), got {self.cov.shape}."
            )
        if self.frozen and self.mean is None:
            raise ValueError("A frozen proposal requires both mean and cov.")

    def _check_fitted(self):
        if self.mean is None or self.cov is None:
            raise RuntimeError(
                "GaussianProposal has not been fitted yet. "
                "Call fit() before sampling or evaluating log_prob."
            )

    def fit(self, x, **kwargs):
        """Fit the proposal to samples by computing mean and covariance.

        If ``frozen=True`` was set at construction, this is a no-op and the
        pre-specified ``mean``/``cov`` are preserved.

        Parameters
        ----------
        x : array-like
            Samples of shape ``(n_samples, dims)``.
        """
        if self.frozen:
            return
        x_np = to_numpy(x)
        if x_np.ndim != 2 or x_np.shape[1] != self.dims:
            raise ValueError(
                f"x must have shape (n_samples, {self.dims}), "
                f"got {x_np.shape}."
            )
        if x_np.shape[0] < 2:
            raise ValueError("At least two samples are required to fit.")
        self.mean = np.mean(x_np, axis=0)
        self.cov = np.atleast_2d(np.cov(x_np, rowvar=False))
        return

    def sample_and_log_prob(self, n_samples: int, xp=None):
        """Draw samples and compute their log-probability.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        xp : module | None
            Array namespace for the returned arrays. Defaults to ``self.xp``.

        Returns
        -------
        x : array, shape ``(n_samples, dims)``
        log_q : array, shape ``(n_samples,)``
        """
        xp = xp if xp is not None else self.xp
        self._check_fitted()
        if n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        x = np.random.multivariate_normal(
            mean=self.mean,
            cov=self.cov,
            size=n_samples,
        ).reshape(n_samples, self.dims)
        log_q = self.log_prob(x)
        return asarray(x, xp=xp), asarray(log_q, xp=xp)

    def log_prob(self, x, xp=None):
        """Evaluate the log-probability of samples.

        Parameters
        ----------
        x : array-like, shape ``(n_samples, dims)``
        xp : module | None
            Array namespace for the returned array. Defaults to ``self.xp``.

        Returns
        -------
        log_q : array, shape ``(n_samples,)``
        """
        xp = xp if xp is not None else self.xp
        self._check_fitted()
        x_np = np.asarray(to_numpy(x))
        if x_np.ndim == 1:
            if self.dims == 1:
                x_np = x_np.reshape(-1, 1)
            elif x_np.shape == (self.dims,):
                x_np = x_np.reshape(1, self.dims)
        if x_np.ndim != 2 or x_np.shape[1] != self.dims:
            raise ValueError(
                f"x must have shape (n_samples, {self.dims}), "
                f"got {x_np.shape}."
            )

        sign, log_det_cov = np.linalg.slogdet(self.cov)
        if sign <= 0:
            raise ValueError("cov must be positive definite.")
        delta = x_np - self.mean
        quadratic = np.sum(
            delta * np.linalg.solve(self.cov, delta.T).T,
            axis=1,
        )
        normalizer = self.dims * np.log(2.0 * np.pi) + log_det_cov
        log_q = -0.5 * (normalizer + quadratic)
        return asarray(log_q, xp=xp)

    def config_dict(self):
        return {
            "proposal_class": "GaussianProposal",
            "dims": self.dims,
            "frozen": self.frozen,
        }

    def save(self, h5_file, path: str = "proposal"):
        """Save mean and covariance to an HDF5 file."""
        self._check_fitted()
        grp = h5_file.require_group(path)
        grp.create_dataset("mean", data=self.mean)
        grp.create_dataset("cov", data=self.cov)
        grp.attrs["dims"] = self.dims
        grp.attrs["frozen"] = self.frozen

    @classmethod
    def load(cls, h5_file, path: str = "proposal"):
        """Load from an HDF5 file."""
        grp = h5_file[path]
        dims = int(grp.attrs["dims"])
        mean = grp["mean"][...]
        cov = grp["cov"][...]
        frozen = bool(grp.attrs.get("frozen", False))
        return cls(dims=dims, mean=mean, cov=cov, frozen=frozen)
