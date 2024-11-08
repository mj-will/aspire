from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable

__backend = None
KNOWN_BACKENDS = ("jax", "torch")


@dataclass(frozen=True)
class Backend:
    name: str
    Samples: Callable = field(init=False)
    to_numpy: Callable = field(init=False)
    from_numpy: Callable = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "Samples",
            getattr(
                import_module(f".samples.{self.name}", "poppy"),
                f"{self.name.capitalize()}Samples",
            ),
        )
        object.__setattr__(
            self,
            "to_numpy",
            getattr(
                import_module(".samples.numpy", "poppy"),
                f"{self.name}_to_numpy",
            ),
        )
        object.__setattr__(
            self,
            "from_numpy",
            getattr(
                import_module(f".samples.{self.name}", "poppy"),
                f"numpy_to_{self.name}",
            ),
        )
        from . import samples

        samples.Samples = self.Samples
        samples.to_numpy = self.to_numpy
        samples.from_numpy = self.from_numpy


def set_backend(backend: str = None):
    if backend is None:
        backend = "torch"
    global __backend
    if backend not in KNOWN_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend}. Known backends are {list(KNOWN_BACKENDS.keys())}"
        )
    __backend = Backend(backend)


def get_backend():
    if __backend is None:
        set_backend()
    return __backend
