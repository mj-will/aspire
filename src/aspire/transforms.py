import logging
import math
from typing import Any

from array_api_compat import device as get_device
from array_api_compat import is_torch_namespace

from .flows import get_flow_wrapper
from .utils import (
    copy_array,
    logit,
    sigmoid,
    update_at_indices,
)

logger = logging.getLogger(__name__)


class BaseTransform:
    """Base class for data transforms.

    Parameters
    ----------
    xp : Callable
        The array API namespace to use (e.g., numpy, torch).
    dtype : Any, optional
        The data type to use for the transform. If not provided, defaults to
        the default dtype of the array API namespace if available.
    """

    def __init__(self, xp, dtype=None):
        self.xp = xp
        if is_torch_namespace(self.xp) and dtype is None:
            dtype = self.xp.get_default_dtype()
        self.dtype = dtype

    def fit(self, x):
        """Fit the transform to the data."""
        raise NotImplementedError("Subclasses must implement fit method.")

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method.")

    def inverse(self, y):
        raise NotImplementedError("Subclasses must implement inverse method.")


class IdentityTransform(BaseTransform):
    """Identity transform that does nothing to the data."""

    def fit(self, x):
        return copy_array(x, xp=self.xp)

    def forward(self, x):
        return copy_array(x, xp=self.xp), self.xp.zeros(
            len(x), device=get_device(x)
        )

    def inverse(self, y):
        return copy_array(y, xp=self.xp), self.xp.zeros(
            len(y), device=get_device(y)
        )


class CompositeTransform(BaseTransform):
    def __init__(
        self,
        parameters: list[int],
        periodic_parameters: list[int] = None,
        prior_bounds: list[tuple[float, float]] = None,
        bounded_to_unbounded: bool = True,
        bounded_transform: str = "probit",
        affine_transform: bool = True,
        device=None,
        xp: None = None,
        eps: float = 1e-6,
        dtype: Any = None,
    ):
        super().__init__(xp=xp, dtype=dtype)
        if prior_bounds is None:
            logger.warning(
                "Missing prior bounds, some transforms may not be applied."
            )
        if periodic_parameters and not prior_bounds:
            raise ValueError(
                "Must specify prior bounds to use periodic parameters."
            )
        self.parameters = parameters
        self.periodic_parameters = periodic_parameters or []
        self.bounded_to_unbounded = bounded_to_unbounded
        self.bounded_transform = bounded_transform
        self.affine_transform = affine_transform

        self.eps = eps
        self.device = device

        if prior_bounds is None:
            self.prior_bounds = None
            self.bounded_parameters = None
            lower_bounds = None
            upper_bounds = None
        else:
            logger.info(f"Prior bounds: {prior_bounds}")
            self.prior_bounds = {
                k: self.xp.asarray(
                    prior_bounds[k], device=device, dtype=self.dtype
                )
                for k in self.parameters
            }
            if bounded_to_unbounded:
                self.bounded_parameters = [
                    p
                    for p in parameters
                    if self.xp.isfinite(self.prior_bounds[p]).all()
                    and p not in self.periodic_parameters
                ]
            else:
                self.bounded_parameters = None
            lower_bounds = self.xp.asarray(
                [self.prior_bounds[p][0] for p in parameters],
                device=device,
                dtype=self.dtype,
            )
            upper_bounds = self.xp.asarray(
                [self.prior_bounds[p][1] for p in parameters],
                device=device,
                dtype=self.dtype,
            )

        if self.periodic_parameters:
            logger.info(f"Periodic parameters: {self.periodic_parameters}")
            self.periodic_mask = self.xp.asarray(
                [p in self.periodic_parameters for p in parameters],
                dtype=bool,
                device=device,
            )
            self._periodic_transform = PeriodicTransform(
                lower=lower_bounds[self.periodic_mask],
                upper=upper_bounds[self.periodic_mask],
                xp=self.xp,
            )
        if self.bounded_parameters:
            logger.info(f"Bounded parameters: {self.bounded_parameters}")
            self.bounded_mask = self.xp.asarray(
                [p in self.bounded_parameters for p in parameters], dtype=bool
            )
            if self.bounded_transform == "probit":
                BoundedClass = ProbitTransform
            elif self.bounded_transform == "logit":
                BoundedClass = LogitTransform
            else:
                raise ValueError(
                    f"Unknown bounded transform: {self.bounded_transform}"
                )

            self._bounded_transform = BoundedClass(
                lower=lower_bounds[self.bounded_mask],
                upper=upper_bounds[self.bounded_mask],
                xp=self.xp,
                eps=self.eps,
            )

        if self.affine_transform:
            logger.info(f"Affine transform applied to: {self.parameters}")
            self._affine_transform = AffineTransform(xp=self.xp)
        else:
            self._affine_transform = None

    def fit(self, x):
        x = copy_array(x, xp=self.xp)
        if self.periodic_parameters:
            logger.debug(
                f"Fitting periodic transform to parameters: {self.periodic_parameters}"
            )
            x = update_at_indices(
                x,
                (slice(None), self.periodic_mask),
                self._periodic_transform.fit(x[:, self.periodic_mask]),
            )
        if self.bounded_parameters:
            logger.debug(
                f"Fitting bounded transform to parameters: {self.bounded_parameters}"
            )
            x = update_at_indices(
                x,
                (slice(None), self.bounded_mask),
                self._bounded_transform.fit(x[:, self.bounded_mask]),
            )
        if self.affine_transform:
            logger.debug("Fitting affine transform to all parameters.")
            x = self._affine_transform.fit(x)
        return x

    def forward(self, x):
        x = copy_array(x, xp=self.xp)
        x = self.xp.atleast_2d(x)
        log_abs_det_jacobian = self.xp.zeros(len(x), device=self.device)
        if self.periodic_parameters:
            y, log_j_periodic = self._periodic_transform.forward(
                x[..., self.periodic_mask]
            )
            x = update_at_indices(x, (slice(None), self.periodic_mask), y)
            log_abs_det_jacobian += log_j_periodic

        if self.bounded_parameters:
            y, log_j_bounded = self._bounded_transform.forward(
                x[..., self.bounded_mask]
            )
            x = update_at_indices(x, (slice(None), self.bounded_mask), y)
            log_abs_det_jacobian += log_j_bounded

        if self.affine_transform:
            x, log_j_affine = self._affine_transform.forward(x)
            log_abs_det_jacobian += log_j_affine
        return x, log_abs_det_jacobian

    def inverse(self, x):
        x = copy_array(x, xp=self.xp)
        x = self.xp.atleast_2d(x)
        log_abs_det_jacobian = self.xp.zeros(len(x), device=self.device)
        if self.affine_transform:
            x, log_j_affine = self._affine_transform.inverse(x)
            log_abs_det_jacobian += log_j_affine

        if self.bounded_parameters:
            y, log_j_bounded = self._bounded_transform.inverse(
                x[..., self.bounded_mask]
            )
            x = update_at_indices(x, (slice(None), self.bounded_mask), y)
            log_abs_det_jacobian += log_j_bounded

        if self.periodic_parameters:
            y, log_j_periodic = self._periodic_transform.inverse(
                x[..., self.periodic_mask]
            )
            x = update_at_indices(x, (slice(None), self.periodic_mask), y)
            log_abs_det_jacobian += log_j_periodic

        return x, log_abs_det_jacobian

    def new_instance(self, xp=None):
        return self.__class__(
            parameters=self.parameters,
            periodic_parameters=self.periodic_parameters,
            prior_bounds=self.prior_bounds,
            bounded_to_unbounded=self.bounded_to_unbounded,
            bounded_transform=self.bounded_transform,
            device=self.device,
            xp=xp or self.xp,
            eps=self.eps,
        )


class FlowTransform(CompositeTransform):
    """Subclass of CompositeTransform that uses a Flow for transformations.

    Does not support periodic transforms.
    """

    def __init__(
        self,
        parameters: list[int],
        prior_bounds: list[tuple[float, float]] = None,
        bounded_to_unbounded: bool = True,
        bounded_transform: str = "probit",
        affine_transform: bool = True,
        device=None,
        xp=None,
        eps=1e-6,
        dtype=None,
    ):
        super().__init__(
            parameters=parameters,
            periodic_parameters=[],
            prior_bounds=prior_bounds,
            bounded_to_unbounded=bounded_to_unbounded,
            bounded_transform=bounded_transform,
            affine_transform=affine_transform,
            device=device,
            xp=xp,
            eps=eps,
            dtype=dtype,
        )

    def new_instance(self, xp=None):
        return self.__class__(
            parameters=self.parameters,
            prior_bounds=self.prior_bounds,
            bounded_to_unbounded=self.bounded_to_unbounded,
            bounded_transform=self.bounded_transform,
            device=self.device,
            xp=xp or self.xp,
            eps=self.eps,
        )


class PeriodicTransform(BaseTransform):
    name: str = "periodic"
    requires_prior_bounds: bool = True

    def __init__(self, lower, upper, xp, dtype=None):
        super().__init__(xp=xp, dtype=dtype)
        self.lower = xp.asarray(lower, dtype=dtype)
        self.upper = xp.asarray(upper, dtype=dtype)
        self._width = upper - lower
        self._shift = None

    def fit(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        y = self.lower + (x - self.lower) % self._width
        return y, self.xp.zeros(y.shape[0], device=get_device(y))

    def inverse(self, y):
        x = self.lower + (y - self.lower) % self._width
        return x, self.xp.zeros(x.shape[0], device=get_device(x))


class ProbitTransform(BaseTransform):
    name: str = "probit"
    requires_prior_bounds: bool = True

    def __init__(self, lower, upper, xp, eps=1e-6, dtype=None):
        self.lower = xp.asarray(lower, dtype=dtype)
        self.upper = xp.asarray(upper, dtype=dtype)
        self._scale_log_abs_det_jacobian = -xp.log(upper - lower).sum()
        self.eps = eps
        self.xp = xp

    def fit(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        from scipy.special import erfinv

        y = (x - self.lower) / (self.upper - self.lower)
        y = self.xp.clip(y, self.eps, 1.0 - self.eps)
        y = erfinv(2 * y - 1) * math.sqrt(2)
        log_abs_det_jacobian = (
            0.5 * (math.log(2 * math.pi) + y**2).sum(-1)
            + self._scale_log_abs_det_jacobian
        )
        return y, log_abs_det_jacobian

    def inverse(self, y):
        from scipy.special import erf

        log_abs_det_jacobian = (
            -(0.5 * (math.log(2 * math.pi) + y**2)).sum(-1)
            - self._scale_log_abs_det_jacobian
        )
        x = 0.5 * (1 + erf(y / math.sqrt(2)))
        x = (self.upper - self.lower) * x + self.lower
        return x, log_abs_det_jacobian


class LogitTransform(BaseTransform):
    name: str = "logit"
    requires_prior_bounds: bool = True

    def __init__(self, lower, upper, xp, eps=1e-6, dtype=None):
        self.lower = xp.asarray(lower, dtype=dtype)
        self.upper = xp.asarray(upper, dtype=dtype)
        self._scale_log_abs_det_jacobian = -xp.log(upper - lower).sum()
        self.eps = eps
        self.xp = xp

    def fit(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        y = (x - self.lower) / (self.upper - self.lower)
        y, log_abs_det_jacobian = logit(y, eps=self.eps)
        log_abs_det_jacobian += self._scale_log_abs_det_jacobian
        return y, log_abs_det_jacobian

    def inverse(self, y):
        x, log_abs_det_jacobian = sigmoid(y)
        log_abs_det_jacobian -= self._scale_log_abs_det_jacobian
        x = (self.upper - self.lower) * x + self.lower
        return x, log_abs_det_jacobian


class AffineTransform(BaseTransform):
    name: str = "affine"
    requires_prior_bounds: bool = False

    def __init__(self, xp):
        self._mean = None
        self._std = None
        self.xp = xp

    def fit(self, x):
        self._mean = x.mean(0)
        self._std = x.std(0)
        self.log_abs_det_jacobian = -self.xp.log(self.xp.abs(self._std)).sum()
        return self.forward(x)[0]

    def forward(self, x):
        y = (x - self._mean) / self._std
        return y, self.log_abs_det_jacobian * self.xp.ones(
            y.shape[0], device=get_device(y)
        )

    def inverse(self, y):
        x = y * self._std + self._mean
        return x, -self.log_abs_det_jacobian * self.xp.ones(
            y.shape[0], device=get_device(y)
        )


class FlowPreconditioningTransform(BaseTransform):
    def __init__(
        self,
        parameters: list[int],
        flow_backend: str = "zuko",
        prior_bounds: list[tuple[float, float]] = None,
        bounded_to_unbounded: bool = True,
        bounded_transform: str = "probit",
        affine_transform: bool = True,
        periodic_parameters: list[int] = None,
        device=None,
        xp=None,
        eps=1e-6,
        dtype=None,
        flow_matching: bool = False,
        flow_kwargs: dict[str, Any] = None,
        fit_kwargs: dict[str, Any] = None,
    ):
        super().__init__(xp=xp, dtype=dtype)

        self.parameters = parameters
        self.periodic_parameters = periodic_parameters or []
        self.prior_bounds = prior_bounds
        self.bounded_to_unbounded = bounded_to_unbounded
        self.bounded_transform = bounded_transform
        self.affine_transform = affine_transform
        self.eps = eps
        self.device = device or "cpu"
        self.flow_backend = flow_backend
        self.flow_matching = flow_matching
        self.flow_kwargs = flow_kwargs or {}
        self.fit_kwargs = fit_kwargs or {}

        FlowClass = get_flow_wrapper(
            backend=flow_backend, flow_matching=flow_matching
        )
        transform = CompositeTransform(
            parameters=parameters,
            periodic_parameters=periodic_parameters,
            prior_bounds=prior_bounds,
            bounded_to_unbounded=bounded_to_unbounded,
            bounded_transform=bounded_transform,
            affine_transform=affine_transform,
            device=device,
            xp=FlowClass.xp,
            eps=eps,
            dtype=dtype,
        )

        self._data_transform = transform
        self._FlowClass = FlowClass
        self.flow = None

    def fit(self, x):
        self.flow = self._FlowClass(
            dims=len(self.parameters),
            device=self.device,
            data_transform=self._data_transform,
            **self.flow_kwargs,
        )
        self.flow.fit(x, **self.fit_kwargs)
        return self.flow.forward(x, xp=self.xp)[0]

    def forward(self, x):
        return self.flow.forward(x, xp=self.xp)

    def inverse(self, y):
        return self.flow.inverse(y, xp=self.xp)

    def new_instance(self, xp=None):
        return self.__class__(
            parameters=self.parameters,
            periodic_parameters=self.periodic_parameters,
            prior_bounds=self.prior_bounds,
            bounded_to_unbounded=self.bounded_to_unbounded,
            bounded_transform=self.bounded_transform,
            affine_transform=self.affine_transform,
            device=self.device,
            xp=xp or self.xp,
            eps=self.eps,
            dtype=self.dtype,
            flow_backend=self.flow_backend,
            flow_matching=self.flow_matching,
            flow_kwargs=self.flow_kwargs,
            fit_kwargs=self.fit_kwargs,
        )
