import math
import torch

from .base import DataTransform


class PeriodicTransform(DataTransform):

    name: str = "periodic"
    requires_prior_bounds: bool = True

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self._width = upper - lower
        self._shift = None

    def fit(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        y = self.lower + (x - self.lower) % self._width
        return y, torch.zeros(y.shape[0], device=y.device)

    def inverse(self, y):
        x = self.lower + (y - self.lower) % self._width
        return x, torch.zeros(x.shape[0], device=x.device)


class ProbitTransform(DataTransform):

    name: str = "probit"
    requires_prior_bounds: bool = True

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self._scale_log_abs_det_jacobian = -torch.log(upper - lower).sum()
        self.eps = torch.finfo(torch.get_default_dtype()).eps

    def fit(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        y = (x - self.lower) / (self.upper - self.lower)
        y = torch.clamp(y, self.eps, 1.0 - self.eps)
        y = torch.erfinv(2 * y - 1)  * math.sqrt(2)
        log_abs_det_jacobian = (
            0.5 * (math.log(2 * math.pi) + y ** 2).sum(-1)
            + self._scale_log_abs_det_jacobian
        )
        return y, log_abs_det_jacobian

    def inverse(self, y):
        log_abs_det_jacobian = (
            -(0.5 * (math.log(2 * math.pi) + y ** 2)).sum(-1)
            - self._scale_log_abs_det_jacobian
        )
        x = 0.5 * (1 + torch.erf(y / math.sqrt(2)))
        x = (self.upper - self.lower) * x + self.lower
        return x, log_abs_det_jacobian


class AffineTransform(DataTransform):

    name: str = "affine"
    requires_prior_bounds: bool = False

    def __init__(self):
        self._mean = None 
        self._std = None

    def fit(self, x):
        self._mean = x.mean(0)
        self._std = x.std(0)
        self.log_abs_det_jacobian = -torch.log(torch.abs(self._std)).sum()
        return self.forward(x)[0]

    def forward(self, x):
        y = (x - self._mean) / self._std
        return y, self.log_abs_det_jacobian * torch.ones(y.shape[0], device=y.device)
    
    def inverse(self, y):
        x = y * self._std + self._mean
        return x, -self.log_abs_det_jacobian * torch.ones(y.shape[0], device=y.device)
