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

    def fit(self, samples, **kwargs): ...

    def config_dict(self): ...

    def save(self, h5_file, path: str = "proposal"): ...

    @classmethod
    def load(cls, h5_file, path: str = "proposal"): ...


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
            return None
        x_np = to_numpy(x)
        self.mean = np.mean(x_np, axis=0)
        self.cov = (
            np.cov(x_np.T)
            if x_np.shape[1] > 1
            else np.atleast_2d(np.var(x_np))
        )
        return None

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
        from scipy.stats import multivariate_normal

        xp = xp if xp is not None else self.xp
        self._check_fitted()
        dist = multivariate_normal(mean=self.mean, cov=self.cov)
        x = dist.rvs(size=n_samples)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        log_q = dist.logpdf(x)
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
        from scipy.stats import multivariate_normal

        xp = xp if xp is not None else self.xp
        self._check_fitted()
        x_np = to_numpy(x)
        dist = multivariate_normal(mean=self.mean, cov=self.cov)
        return asarray(dist.logpdf(x_np), xp=xp)

    def config_dict(self):
        return {"proposal_class": "GaussianProposal", "dims": self.dims}

    def save(self, h5_file, path: str = "proposal"):
        """Save mean and covariance to an HDF5 file."""
        self._check_fitted()
        grp = h5_file.require_group(path)
        grp.create_dataset("mean", data=self.mean)
        grp.create_dataset("cov", data=self.cov)
        grp.attrs["dims"] = self.dims

    @classmethod
    def load(cls, h5_file, path: str = "proposal"):
        """Load from an HDF5 file."""
        grp = h5_file[path]
        dims = int(grp.attrs["dims"])
        mean = grp["mean"][...]
        cov = grp["cov"][...]
        return cls(dims=dims, mean=mean, cov=cov)
