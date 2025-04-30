import copy
import logging
import math
from typing import Any

from array_api_compat import is_torch_namespace
from scipy.special import erf, erfinv

from .utils import logit, sigmoid, update_at_indices

logger = logging.getLogger(__name__)


class DataTransform:
    def __init__(
        self,
        parameters: list[int],
        periodic_parameters: list[int],
        prior_bounds: list[tuple[float, float]],
        bounded_to_unbounded: bool = True,
        bounded_transform: str = "probit",
        device=None,
        xp: None = None,
        eps: float = 1e-6,
        dtype: Any = None,
    ):
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

        self.xp = xp
        self.device = device
        self.eps = eps

        if is_torch_namespace(self.xp) and dtype is None:
            dtype = self.xp.get_default_dtype()
        self.dtype = dtype

        if prior_bounds is None:
            self.prior_bounds = None
            self.bounded_parameters = None
            lower_bounds = None
            upper_bounds = None
        else:
            logger.info(f"Prior bounds: {prior_bounds}")
            self.prior_bounds = {
                k: self.xp.asarray(prior_bounds[k], device=device, dtype=self.dtype) for k in self.parameters
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
        logger.info(f"Affine transform applied to: {self.parameters}")
        self.affine_transform = AffineTransform(xp=self.xp)

    def fit(self, x):
        x = copy.copy(x)
        if self.periodic_parameters:
            logger.debug(
                f"Fitting periodic transform to parameters: {self.periodic_parameters}"
            )
            x = update_at_indices(
                x,
                (slice(None), self.periodic_mask),
                self._periodic_transform.fit(x[:, self.periodic_mask])
            )
        if self.bounded_parameters:
            logger.debug(
                f"Fitting bounded transform to parameters: {self.bounded_parameters}"
            )
            x = update_at_indices(
                x,
                (slice(None), self.bounded_mask),
                self._bounded_transform.fit(x[:, self.bounded_mask])
            )
        return self.affine_transform.fit(x)

    def forward(self, x):
        x = copy.copy(x)
        log_abs_det_jacobian = self.xp.zeros(len(x), device=self.device)
        if self.periodic_parameters:
            y, log_j_periodic = self._periodic_transform.forward(x[:, self.periodic_mask])
            x = update_at_indices(x, (slice(None), self.periodic_mask), y)
            log_abs_det_jacobian += log_j_periodic

        if self.bounded_parameters:
            y, log_j_bounded = self._bounded_transform.forward(x[:, self.bounded_mask])
            x = update_at_indices(x, (slice(None), self.bounded_mask), y)
            log_abs_det_jacobian += log_j_bounded

        x, log_j_affine = self.affine_transform.forward(x)
        log_abs_det_jacobian += log_j_affine
        return x, log_abs_det_jacobian

    def inverse(self, x):
        x = copy.copy(x)
        log_abs_det_jacobian = self.xp.zeros(len(x), device=self.device)
        x, log_j_affine = self.affine_transform.inverse(x)
        log_abs_det_jacobian += log_j_affine

        if self.bounded_parameters:
            y, log_j_bounded = self._bounded_transform.inverse(x[:, self.bounded_mask])
            x = update_at_indices(x, (slice(None), self.bounded_mask), y)
            log_abs_det_jacobian += log_j_bounded

        if self.periodic_parameters:
            y, log_j_periodic = self._periodic_transform.inverse(x[:, self.periodic_mask])
            x = update_at_indices(x, (slice(None), self.periodic_mask), y)
            log_abs_det_jacobian += log_j_periodic

        return x, log_abs_det_jacobian
    
    def new_instance(self):
        return self.__class__(
            parameters=self.parameters,
            periodic_parameters=self.periodic_parameters,
            prior_bounds=self.prior_bounds,
            bounded_to_unbounded=self.bounded_to_unbounded,
            bounded_transform=self.bounded_transform,
            device=self.device,
            xp=self.xp,
            eps=self.eps,
        )


class PeriodicTransform(DataTransform):
    name: str = "periodic"
    requires_prior_bounds: bool = True

    def __init__(self, lower, upper, xp, dtype=None):
        self.lower = xp.asarray(lower, dtype=dtype)
        self.upper = xp.asarray(upper, dtype=dtype)
        self._width = upper - lower
        self._shift = None
        self.xp = xp

    def fit(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        y = self.lower + (x - self.lower) % self._width
        return y, self.xp.zeros(y.shape[0], device=y.device)

    def inverse(self, y):
        x = self.lower + (y - self.lower) % self._width
        return x, self.xp.zeros(x.shape[0], device=x.device)


class ProbitTransform(DataTransform):
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
        y = (x - self.lower) / (self.upper - self.lower)
        y = self.xp.clip(y, self.eps, 1.0 - self.eps)
        y = erfinv(2 * y - 1) * math.sqrt(2)
        log_abs_det_jacobian = (
            0.5 * (math.log(2 * math.pi) + y**2).sum(-1)
            + self._scale_log_abs_det_jacobian
        )
        return y, log_abs_det_jacobian

    def inverse(self, y):
        log_abs_det_jacobian = (
            -(0.5 * (math.log(2 * math.pi) + y**2)).sum(-1)
            - self._scale_log_abs_det_jacobian
        )
        x = 0.5 * (1 + erf(y / math.sqrt(2)))
        x = (self.upper - self.lower) * x + self.lower
        return x, log_abs_det_jacobian
    

class LogitTransform(DataTransform):

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



class AffineTransform(DataTransform):
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
            y.shape[0], device=y.device
        )

    def inverse(self, y):
        x = y * self._std + self._mean
        return x, -self.log_abs_det_jacobian * self.xp.ones(
            y.shape[0], device=y.device
        )
