from __future__ import annotations

from typing import TYPE_CHECKING

from array_api_compat import array_namespace
from sequince import TemperedPosterior
from sequince.utils.random import resolve_rng

from ..samples import Samples
from ..utils import track_calls
from .base import Sampler

if TYPE_CHECKING:
    from sequince.utils.random import RNGLike


class ImportanceSampler(Sampler):
    @track_calls
    def sample(self, n_samples: int, rng: RNGLike = None) -> Samples:
        rng = resolve_rng(rng)
        posterior = TemperedPosterior(
            self._log_prior_array,
            self._log_likelihood_array,
            log_proposal=self._proposal(),
        )
        state = self._initial_state(
            n_samples,
            posterior=posterior,
            rng=rng,
        )
        components = state.evaluation.components
        samples = Samples(
            state.particles,
            log_q=components["log_proposal"],
            log_prior=components["log_prior"],
            log_likelihood=components["log_likelihood"],
            xp=array_namespace(state.particles),
            parameters=self.parameters,
            dtype=self.dtype,
        )
        samples.compute_weights()
        return samples
