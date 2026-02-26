from typing import Callable

import numpy as np

from ..samples import MCMCSamples, Samples, to_numpy
from ..utils import AspireFile, track_calls
from .base import Sampler


class MCMCSampler(Sampler):
    """Base class for MCMC samplers."""

    chain_checkpoint_path = "checkpoint"
    """Path within checkpoint file to save MCMC chain checkpoints.

    The default is "checkpoint".
    """
    chain_dataset_name = "mcmc_chain"
    """Name of chain entry within checkpoint file to save MCMC checkpoints."""

    def __init__(
        self,
        log_likelihood,
        log_prior,
        dims,
        prior_flow,
        xp,
        dtype=None,
        parameters=None,
        preconditioning_transform=None,
        rng=None,
    ):
        super().__init__(
            log_likelihood,
            log_prior,
            dims,
            prior_flow,
            xp,
            dtype,
            parameters,
            preconditioning_transform,
        )
        self.rng = rng or np.random.default_rng()

    def draw_initial_samples(self, n_samples: int) -> Samples:
        """Draw initial samples from the prior flow."""
        # Flow may propose samples outside prior bounds, so we may need
        # to try multiple times to get enough valid samples.
        n_samples_drawn = 0
        samples = None
        while n_samples_drawn < n_samples:
            x, log_q = self.prior_flow.sample_and_log_prob(n_samples)
            new_samples = Samples(
                x,
                xp=self.xp,
                log_q=log_q,
                dtype=self.dtype,
                parameters=self.parameters,
            )
            new_samples.log_prior = new_samples.array_to_namespace(
                self.log_prior(new_samples)
            )
            valid = self.xp.isfinite(new_samples.log_prior)
            n_valid = int(self.xp.sum(valid))
            if n_valid > 0:
                if samples is None:
                    samples = new_samples[valid]
                else:
                    samples = Samples.concatenate(
                        [samples, new_samples[valid]]
                    )
                n_samples_drawn += n_valid

        if n_samples_drawn > n_samples:
            samples = samples[:n_samples]

        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
        )
        return samples

    def log_prob(self, z):
        """Compute the log probability of the samples.

        Input samples are in the transformed space.
        """
        x, log_abs_det_jacobian = self.preconditioning_transform.inverse(z)
        samples = Samples(x, xp=self.xp, dtype=self.dtype)
        samples.log_prior = self.log_prior(samples)
        samples.log_likelihood = self.log_likelihood(samples)
        log_prob = (
            samples.log_likelihood
            + samples.log_prior
            + samples.array_to_namespace(log_abs_det_jacobian)
        )
        return to_numpy(log_prob).flatten()

    def default_mcmc_chain_file_checkpoint_callback(
        self, file_path: str | None
    ) -> Callable[[dict], None]:
        """Return a callback that saves MCMC checkpoints as native HDF5 samples."""
        if file_path is None:
            return self.default_checkpoint_callback
        callback = self.default_file_checkpoint_callback(file_path)
        _ = callback  # validates extension and path early

        def _mcmc_chain_callback(state: dict) -> None:
            samples = state.get("samples")
            if samples is None:
                raise ValueError("Checkpoint missing samples.")
            chain_path = (
                f"{self.chain_checkpoint_path}/{self.chain_dataset_name}"
            )
            with AspireFile(file_path, "a") as h5_file:
                if chain_path in h5_file:
                    del h5_file[chain_path]
                samples.save(h5_file, path=chain_path, flat=False)
                group = h5_file[chain_path]
                if (
                    hasattr(samples, "chain_shape")
                    and samples.chain_shape is not None
                ):
                    group.attrs["chain_shape"] = np.asarray(
                        samples.chain_shape, dtype=int
                    )
                iteration = state.get("iteration")
                if iteration is not None:
                    group.attrs["iteration"] = int(iteration)
                stage = state.get("meta", {}).get("stage")
                if stage is not None:
                    group.attrs["stage"] = str(stage)
                sampler_name = state.get("sampler")
                if sampler_name is not None:
                    group.attrs["sampler"] = str(sampler_name)
            self.default_checkpoint_callback(state)

        return _mcmc_chain_callback

    def checkpoint_mcmc_chain(
        self,
        samples: Samples,
        iteration: int | None = None,
        checkpoint_callback: Callable[[dict], None] | None = None,
        checkpoint_every: int | None = None,
        checkpoint_file_path: str | None = None,
    ) -> None:
        """Save an MCMC chain checkpoint."""
        if checkpoint_every is not None and checkpoint_every <= 0:
            return
        if checkpoint_callback is None and checkpoint_file_path is not None:
            checkpoint_callback = (
                self.default_mcmc_chain_file_checkpoint_callback(
                    checkpoint_file_path
                )
            )
        if checkpoint_callback is None:
            return
        state = self.build_checkpoint_state(
            samples=samples, iteration=iteration, meta={"stage": "mcmc_chain"}
        )
        checkpoint_callback(state)


class Emcee(MCMCSampler):
    @track_calls
    def sample(
        self,
        n_samples: int,
        nwalkers: int = None,
        nsteps: int = 500,
        rng=None,
        discard=0,
        checkpoint_callback: Callable[[dict], None] | None = None,
        checkpoint_every: int | None = None,
        checkpoint_file_path: str | None = None,
        **kwargs,
    ) -> Samples:
        from emcee import EnsembleSampler

        nwalkers = nwalkers or n_samples
        self.sampler = EnsembleSampler(
            nwalkers,
            self.dims,
            log_prob_fn=self.log_prob,
            vectorize=True,
        )

        rng = rng or self.rng or np.random.default_rng()

        samples = self.draw_initial_samples(nwalkers)
        p0 = samples.x

        z0 = to_numpy(self.preconditioning_transform.fit(p0))

        self.sampler.run_mcmc(z0, nsteps, **kwargs)

        chain = self.sampler.get_chain(discard=discard)

        # Transform chain back to original space
        chain_z = chain.reshape(-1, self.dims)
        chain_x, log_jacobian = self.preconditioning_transform.inverse(chain_z)
        chain_x = chain_x.reshape(chain.shape)
        # Create MCMCSamples
        samples_mcmc = MCMCSamples.from_chain(
            chain=chain_x,
            parameters=self.parameters,
            xp=self.xp,
            dtype=self.dtype,
            burn_in=discard,
        )
        self.checkpoint_mcmc_chain(
            samples=samples_mcmc,
            iteration=nsteps,
            checkpoint_callback=checkpoint_callback,
            checkpoint_every=checkpoint_every,
            checkpoint_file_path=checkpoint_file_path,
        )

        return samples_mcmc


class MiniPCN(MCMCSampler):
    @track_calls
    def sample(
        self,
        n_samples,
        rng=None,
        target_acceptance_rate=0.234,
        n_steps=100,
        thin=1,
        burnin=0,
        last_step_only=False,
        step_fn="tpcn",
        checkpoint_callback: Callable[[dict], None] | None = None,
        checkpoint_every: int | None = None,
        checkpoint_file_path: str | None = None,
    ):
        from minipcn import Sampler

        rng = rng or self.rng or np.random.default_rng()
        p0 = self.draw_initial_samples(n_samples).x

        z0 = to_numpy(self.preconditioning_transform.fit(p0))

        self.sampler = Sampler(
            log_prob_fn=self.log_prob,
            step_fn=step_fn,
            rng=rng,
            dims=self.dims,
            target_acceptance_rate=target_acceptance_rate,
        )

        chain, history = self.sampler.sample(z0, n_steps=n_steps)
        _ = history

        # Transform the full chain back to original space once so checkpoints
        # always capture pre-burn/pre-thin samples.
        chain_z_full = chain.reshape(-1, self.dims)
        chain_x_full, _ = self.preconditioning_transform.inverse(chain_z_full)
        chain_x_full = chain_x_full.reshape(chain.shape)
        full_chain_samples = MCMCSamples.from_chain(
            chain=chain_x_full,
            parameters=self.parameters,
            xp=self.xp,
            dtype=self.dtype,
            thin=1,
            burn_in=0,
        )
        self.checkpoint_mcmc_chain(
            samples=full_chain_samples,
            iteration=n_steps,
            checkpoint_callback=checkpoint_callback,
            checkpoint_every=checkpoint_every,
            checkpoint_file_path=checkpoint_file_path,
        )

        if last_step_only:
            x = chain_x_full[-1]
            samples_mcmc = Samples(
                x, xp=self.xp, dtype=self.dtype, parameters=self.parameters
            )
            samples_mcmc.log_prior = samples_mcmc.array_to_namespace(
                self.log_prior(samples_mcmc)
            )
            samples_mcmc.log_likelihood = samples_mcmc.array_to_namespace(
                self.log_likelihood(samples_mcmc)
            )
        else:
            # Apply burn-in and thinning to the full chain for returned samples.
            samples_mcmc = full_chain_samples.post_process(
                burn_in=burnin, thin=thin
            )

        return samples_mcmc


class ParallelTemperedMCMCSampler(MCMCSampler):
    """Wrapper for Parallel Tempered MCMC Samplers"""

    def log_likelihood_wrapper(self, z):
        """Wrapper for log-likelihood that takes array inputs."""
        x, log_abs_det_jacobian = self.preconditioning_transform.inverse(z)
        samples = Samples(x, xp=self.xp, dtype=self.dtype)
        samples.log_prior = self.log_prior(samples)
        samples.log_likelihood = self.log_likelihood(samples)
        return to_numpy(samples.log_likelihood + log_abs_det_jacobian)

    def log_prior_wrapper(self, z):
        """Wrapper for log-prior that takes array inputs."""
        x, _ = self.preconditioning_transform.inverse(z)
        samples = Samples(x, xp=self.xp, dtype=self.dtype)
        # Skip Jacobian to avoid double counting in log_prior and
        # log_likelihood
        return self.log_prior(samples)
