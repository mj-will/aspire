from collections import namedtuple
import importlib

from ..backend import get_backend

KnownTransforms = namedtuple(
    "KnownTransforms", ["Periodic", "Bounded", "Affine"]
)


def get_transforms(backend=None):
    if backend is None:
        backend = get_backend().name
    transforms = importlib.import_module(f".transforms.{backend}", "poppy")
    return KnownTransforms(
        Periodic=transforms.PeriodicTransform,
        Bounded=transforms.ProbitTransform,
        Affine=transforms.AffineTransform,
    )