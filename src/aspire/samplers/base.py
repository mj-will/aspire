import logging
import pickle
from pathlib import Path
from typing import Any, Callable

from ..flows.base import Flow
from ..samples import Samples
from ..transforms import IdentityTransform
from ..utils import AspireFile, asarray, dump_state, track_calls

logger = logging.getLogger(__name__)


class Sampler:
    """Base class for all samplers.

    Parameters
    ----------
    log_likelihood : Callable
        The log likelihood function.
    log_prior : Callable
        The log prior function.
    dims : int
        The number of dimensions.
    flow : Flow
        The flow object.
    xp : Callable
        The array backend to use.
    parameters : list[str] | None
        The list of parameter names. If None, any samples objects will not
        have the parameters names specified.
    """

    def __init__(
        self,
        log_likelihood: Callable,
        log_prior: Callable,
        dims: int,
        prior_flow: Flow,
        xp: Callable,
        dtype: Any | str | None = None,
        parameters: list[str] | None = None,
        preconditioning_transform: Callable | None = None,
    ):
        self.prior_flow = prior_flow
        self._log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.dims = dims
        self.xp = xp
        self.dtype = dtype
        self.parameters = parameters
        self.history = None
        self.n_likelihood_evaluations = 0
        self._last_checkpoint_state: dict | None = None
        self._last_checkpoint_bytes: bytes | None = None
        if preconditioning_transform is None:
            self.preconditioning_transform = IdentityTransform(xp=self.xp)
        else:
            self.preconditioning_transform = preconditioning_transform

    def fit_preconditioning_transform(self, x):
        """Fit the data transform to the data."""
        x = asarray(
            x,
            xp=self.preconditioning_transform.xp,
            dtype=self.preconditioning_transform.dtype,
        )
        return self.preconditioning_transform.fit(x)

    @track_calls
    def sample(self, n_samples: int) -> Samples:
        raise NotImplementedError

    def log_likelihood(self, samples: Samples) -> Samples:
        """Computes the log likelihood of the samples.

        Also tracks the number of likelihood evaluations.
        """
        self.n_likelihood_evaluations += len(samples)
        return self._log_likelihood(samples)

    def config_dict(self, include_sample_calls: bool = False) -> dict:
        """
        Returns a dictionary with the configuration of the sampler.

        Parameters
        ----------
        include_sample_calls : bool
            Whether to include the sample calls in the configuration.
            Default is False.
        """
        config = {"sampler_class": self.__class__.__name__}
        if include_sample_calls:
            if hasattr(self, "sample") and hasattr(self.sample, "calls"):
                config["sample_calls"] = self.sample.calls.to_dict(
                    list_to_dict=True
                )
            else:
                logger.warning(
                    "Sampler does not have a sample method with calls attribute."
                )
        return config

    # --- Checkpointing helpers shared across samplers ---
    def _checkpoint_extra_state(self) -> dict:
        """Sampler-specific extras for checkpointing (override in subclasses)."""
        return {}

    def _restore_extra_state(self, state: dict) -> None:
        """Restore sampler-specific extras (override in subclasses)."""
        _ = state  # no-op for base

    def build_checkpoint_state(
        self,
        samples: Samples,
        iteration: int | None = None,
        meta: dict | None = None,
    ) -> dict:
        """Prepare a serializable checkpoint payload for the sampler state."""
        checkpoint_samples = samples
        base_state = {
            "sampler": self.__class__.__name__,
            "iteration": iteration,
            "samples": checkpoint_samples,
            "config": self.config_dict(include_sample_calls=False),
            "parameters": self.parameters,
            "meta": meta or {},
        }
        base_state.update(self._checkpoint_extra_state())
        return base_state

    def serialize_checkpoint(
        self, state: dict, protocol: int | None = None
    ) -> bytes:
        """Serialize a checkpoint state to bytes with pickle."""
        protocol = (
            pickle.HIGHEST_PROTOCOL if protocol is None else int(protocol)
        )
        return pickle.dumps(state, protocol=protocol)

    def default_checkpoint_callback(self, state: dict) -> None:
        """Store the latest checkpoint (state + pickled bytes) on the sampler."""
        self._last_checkpoint_state = state
        self._last_checkpoint_bytes = self.serialize_checkpoint(state)

    def default_file_checkpoint_callback(
        self, file_path: str | Path | None
    ) -> Callable[[dict], None]:
        """Return a simple default callback that overwrites an HDF5 file."""
        if file_path is None:
            return self.default_checkpoint_callback
        file_path = Path(file_path)
        lower_path = file_path.name.lower()
        if not lower_path.endswith((".h5", ".hdf5")):
            raise ValueError(
                "Checkpoint file must be an HDF5 file (.h5 or .hdf5)."
            )

        def _callback(state: dict) -> None:
            with AspireFile(file_path, "a") as h5_file:
                self.save_checkpoint_to_hdf(
                    state, h5_file, path="checkpoint", dsetname="state"
                )
            self.default_checkpoint_callback(state)

        return _callback

    def save_checkpoint_to_hdf(
        self,
        state: dict,
        h5_file,
        path: str = "sampler_checkpoints",
        dsetname: str | None = None,
        protocol: int | None = None,
    ) -> None:
        """Save a checkpoint state into an HDF5 file as a pickled blob."""
        if dsetname is None:
            iter_str = state.get("iteration", "unknown")
            dsetname = f"iter_{iter_str}"
        dump_state(
            state,
            h5_file,
            path=path,
            dsetname=dsetname,
            protocol=protocol or pickle.HIGHEST_PROTOCOL,
        )

    def load_checkpoint_from_file(
        self,
        file_path: str | Path,
        h5_path: str = "checkpoint",
        dsetname: str = "state",
    ) -> dict:
        """Load a checkpoint dictionary from .pkl or .hdf5 file."""
        file_path = Path(file_path)
        lower_path = file_path.name.lower()
        if lower_path.endswith((".h5", ".hdf5")):
            with AspireFile(file_path, "r") as h5_file:
                data = h5_file[h5_path][dsetname][...]
                checkpoint_bytes = data.tobytes()
        else:
            with open(file_path, "rb") as f:
                checkpoint_bytes = f.read()
        return pickle.loads(checkpoint_bytes)

    def restore_from_checkpoint(
        self, source: str | bytes | dict
    ) -> tuple[Samples, dict]:
        """Restore sampler state from a checkpoint source."""
        if isinstance(source, str):
            state = self.load_checkpoint_from_file(source)
        elif isinstance(source, bytes):
            state = pickle.loads(source)
        elif isinstance(source, dict):
            state = source
        else:
            raise TypeError("Unsupported checkpoint source type.")

        samples_saved = state.get("samples")
        if samples_saved is None:
            raise ValueError("Checkpoint missing samples.")

        samples = Samples.from_samples(
            samples_saved, xp=self.xp, dtype=self.dtype
        )
        # Allow subclasses to restore sampler-specific components
        self._restore_extra_state(state)
        return samples, state

    @property
    def last_checkpoint_state(self) -> dict | None:
        """Return the most recent checkpoint state stored by the default callback."""
        return self._last_checkpoint_state

    @property
    def last_checkpoint_bytes(self) -> bytes | None:
        """Return the most recent pickled checkpoint produced by the default callback."""
        return self._last_checkpoint_bytes
