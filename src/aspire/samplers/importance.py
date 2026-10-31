from __future__ import annotations

from typing import TYPE_CHECKING

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
        """Draw proposal samples and compute posterior importance weights.

        Parameters
        ----------
        n_samples : int
            Number of particles to draw from the fitted proposal.
        rng : RandomGenerator, numpy.random.Generator, int or None, optional
            Random generator or seed used to draw the proposal particles.

        Returns
        -------
        Samples
            Particles of shape ``(n_samples, dims)`` with proposal, prior,
            and likelihood log densities, importance weights, and evidence
            estimates.
        """
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
        return Samples.from_state(
            state,
            parameters=self.parameters,
            dtype=self.dtype,
        )
