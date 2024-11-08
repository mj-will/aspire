import numpy as np
import copy
from typing import Callable
import logging

from ..backend import get_backend
from .utils import get_transforms

logger = logging.getLogger(__name__)

class DataTransform:

    to_backend: Callable = None

    def __init__(
        self,
        parameters: list[int],
        periodic_parameters: list[int],
        prior_bounds: list[tuple[float, float]],
        bounded_to_unbounded: bool = True,
        device=None,
    ):
        if prior_bounds is None:
            logger.warning(
                "Missing prior bounds, some transforms may not be applied."
            )
        if periodic_parameters and not prior_bounds:
            raise ValueError("Must specify prior bounds to use periodic parameters.")
        self.parameters = parameters
        if periodic_parameters:
            logger.warning("Periodic parameters are not implemented yet.")
        self.periodic_parameters = []
        self.bounded_to_unbounded = bounded_to_unbounded

        backend = get_backend()
        self.to_backend = backend.from_numpy
        self.device = device

        if prior_bounds is None:
            self.prior_bounds = None
            self.bounded_parameters = None
            lower_bounds = None
            upper_bounds = None
        else:
            logger.info(f"Prior bounds: {prior_bounds}")
            self.prior_bounds = {k: self.to_backend(v, self.device) for k, v in prior_bounds.items()}
            if bounded_to_unbounded:
                self.bounded_parameters = [
                    p for p in parameters
                    if np.isfinite(prior_bounds[p]).all() and p not in self.periodic_parameters
                ]
            else:
                self.bounded_parameters = None
            lower_bounds = self.to_backend([self.prior_bounds[p][0] for p in parameters], self.device)
            upper_bounds = self.to_backend([self.prior_bounds[p][1] for p in parameters], self.device)

        transforms = get_transforms()

        if self.periodic_parameters:
            logger.info(f"Periodic parameters: {self.periodic_parameters}")
            self.periodic_mask = self.to_backend(np.array([p in self.periodic_parameters for p in parameters]), self.device, dtype=bool)
            self.periodic_transform = transforms.Periodic(
                lower=lower_bounds[self.periodic_mask],
                upper=upper_bounds[self.periodic_mask],
            )
        if self.bounded_parameters:
            logger.info(f"Bounded parameters: {self.bounded_parameters}")
            self.bounded_mask = self.to_backend(np.array([p in self.bounded_parameters for p in parameters]), dtype=bool)
            self.bounded_transform = transforms.Bounded(
                lower=lower_bounds[self.bounded_mask],
                upper=upper_bounds[self.bounded_mask],
            )
        logger.info(f"Affine transform applied to: {self.parameters}")
        self.affine_transform = transforms.Affine()

    def fit(self, x):
        x = copy.copy(x)
        if self.periodic_parameters:
            logger.debug(f"Fitting periodic transform to parameters: {self.periodic_parameters}")
            x[:, self.periodic_mask] = self.periodic_transform.fit(x[:, self.periodic_mask])
        if self.bounded_parameters:
            logger.debug(f"Fitting bounded transform to parameters: {self.bounded_parameters}")
            x[:, self.bounded_mask] = self.bounded_transform.fit(x[:, self.bounded_mask])
        return self.affine_transform.fit(x)


    def forward(self, x):
        x = copy.copy(x)
        log_abs_det_jacobian = self.to_backend(np.zeros(x.shape[0]), self.device)
        if self.periodic_parameters:
            x[:, self.periodic_mask], log_j_periodic = self.periodic_transform.forward(x[:, self.periodic_mask])
            log_abs_det_jacobian += log_j_periodic
        
        if self.bounded_parameters:
            x[:, self.bounded_mask], log_j_bounded = self.bounded_transform.forward(x[:, self.bounded_mask])
            log_abs_det_jacobian += log_j_bounded

        x, log_j_affine = self.affine_transform.forward(x)
        log_abs_det_jacobian += log_j_affine
        return x, log_abs_det_jacobian

    def inverse(self, x):
        x = copy.copy(x)
        log_abs_det_jacobian = self.to_backend(np.zeros(x.shape[0]), self.device)
        x, log_j_affine = self.affine_transform.inverse(x)
        log_abs_det_jacobian += log_j_affine

        if self.bounded_parameters:
            x[:, self.bounded_mask], log_j_bounded = self.bounded_transform.inverse(x[:, self.bounded_mask])
            log_abs_det_jacobian += log_j_bounded

        if self.periodic_parameters:
            x[:, self.periodic_mask], log_j_periodic = self.periodic_transform.inverse(x[:, self.periodic_mask])
            log_abs_det_jacobian += log_j_periodic

        return x, log_abs_det_jacobian
