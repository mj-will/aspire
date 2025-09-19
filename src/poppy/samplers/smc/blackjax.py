import logging
from functools import partial

import numpy as np

from ...samples import SMCSamples
from ...utils import asarray, to_numpy, track_calls
from .base import SMCSampler

logger = logging.getLogger(__name__)


class BlackJAXSMC(SMCSampler):
    """BlackJAX SMC sampler."""

    def __init__(
        self,
        log_likelihood,
        log_prior,
        dims,
        prior_flow,
        xp,
        parameters=None,
        preconditioning_transform=None,
        rng: np.random.Generator | None = None,  # New parameter
    ):
        # For JAX compatibility, we'll keep the original xp
        super().__init__(
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            dims=dims,
            prior_flow=prior_flow,
            xp=xp,
            parameters=parameters,
            preconditioning_transform=preconditioning_transform,
        )
        self.key = None
        self.rng = rng or np.random.default_rng()

    def log_prob(self, x, beta=None):
        """Log probability function compatible with BlackJAX."""
        # Convert to original xp format for computation
        if hasattr(x, "__array__"):
            x_original = asarray(x, self.xp)
        else:
            x_original = x

        # Transform back to parameter space
        x_params, log_abs_det_jacobian = (
            self.preconditioning_transform.inverse(x_original)
        )
        samples = SMCSamples(x_params, xp=self.xp)

        # Compute log probabilities
        log_q = self.prior_flow.log_prob(samples.x)
        samples.log_q = samples.array_to_namespace(log_q)
        samples.log_prior = samples.array_to_namespace(self.log_prior(samples))
        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
        )

        # Compute target log probability
        log_prob = samples.log_p_t(
            beta=beta
        ).flatten() + samples.array_to_namespace(log_abs_det_jacobian)

        # Handle NaN values
        log_prob = self.xp.where(
            self.xp.isnan(log_prob), -self.xp.inf, log_prob
        )

        return log_prob

    @track_calls
    def sample(
        self,
        n_samples: int,
        n_steps: int = None,
        adaptive: bool = True,
        target_efficiency: float = 0.5,
        target_efficiency_rate: float = 1.0,
        n_final_samples: int | None = None,
        sampler_kwargs: dict | None = None,
        rng_key=None,
    ):
        """Sample using BlackJAX SMC.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        n_steps : int
            Number of SMC steps.
        adaptive : bool
            Whether to use adaptive tempering.
        target_efficiency : float
            Target efficiency for adaptive tempering.
        n_final_samples : int | None
            Number of final samples to return.
        sampler_kwargs : dict | None
            Additional arguments for the BlackJAX sampler.
            - algorithm: str, one of "nuts", "hmc", "rwmh", "random_walk"
            - n_steps: int, number of MCMC steps per mutation
            - step_size: float, step size for HMC/NUTS
            - inverse_mass_matrix: array, inverse mass matrix
            - sigma: float or array, proposal covariance for random walk MH
            - num_integration_steps: int, integration steps for HMC
        rng_key : jax.random.key| None
            JAX random key for reproducibility.
        """
        self.sampler_kwargs = sampler_kwargs or {}
        self.sampler_kwargs.setdefault("n_steps", 5 * self.dims)
        self.sampler_kwargs.setdefault("algorithm", "nuts")
        self.sampler_kwargs.setdefault("step_size", 1e-3)
        self.sampler_kwargs.setdefault("inverse_mass_matrix", None)
        self.sampler_kwargs.setdefault("sigma", 0.1)  # For random walk MH

        # Initialize JAX random key
        if rng_key is None:
            import jax

            self.key = jax.random.key(42)
        else:
            self.key = rng_key

        return super().sample(
            n_samples,
            n_steps=n_steps,
            adaptive=adaptive,
            target_efficiency=target_efficiency,
            target_efficiency_rate=target_efficiency_rate,
            n_final_samples=n_final_samples,
        )

    def mutate(self, particles, beta, n_steps=None):
        """Mutate particles using BlackJAX MCMC."""
        import blackjax
        import jax

        logger.debug("Mutating particles with BlackJAX")

        # Split the random key
        self.key, subkey = jax.random.split(self.key)

        # Transform particles to latent space
        z = self.fit_preconditioning_transform(particles.x)

        # Convert to JAX arrays
        z_jax = jax.numpy.asarray(to_numpy(z))

        # Create log probability function for this beta
        log_prob_fn = partial(self._jax_log_prob, beta=beta)

        # Choose BlackJAX algorithm
        algorithm = self.sampler_kwargs["algorithm"].lower()

        n_steps = n_steps or self.sampler_kwargs["n_steps"]

        if algorithm == "rwmh" or algorithm == "random_walk":
            # Initialize Random Walk Metropolis-Hastings sampler
            sigma = self.sampler_kwargs.get("sigma", 0.1)

            # BlackJAX RMH expects a transition function, not a covariance
            if isinstance(sigma, (int, float)):
                # Create a multivariate normal proposal function
                def proposal_fn(key, position):
                    return position + sigma * jax.random.normal(
                        key, position.shape
                    )
            else:
                # For more complex covariance structures
                if len(sigma) == self.dims:
                    # Diagonal covariance
                    sigma_diag = jax.numpy.array(sigma)

                    def proposal_fn(key, position):
                        return position + sigma_diag * jax.random.normal(
                            key, position.shape
                        )
                else:
                    # Full covariance matrix
                    sigma_matrix = jax.numpy.array(sigma)

                    def proposal_fn(key, position):
                        return position + jax.random.multivariate_normal(
                            key, jax.numpy.zeros(self.dims), sigma_matrix
                        )

            rwmh = blackjax.rmh(log_prob_fn, proposal_fn)

            # Initialize states for each particle
            n_particles = z_jax.shape[0]
            keys = jax.random.split(subkey, n_particles)

            # Vectorized initialization and sampling
            def init_and_sample(key, z_init):
                state = rwmh.init(z_init)

                def one_step(state, key):
                    state, info = rwmh.step(key, state)
                    return state, (state, info)

                keys = jax.random.split(key, n_steps)
                final_state, (states, infos) = jax.lax.scan(
                    one_step, state, keys
                )
                return final_state, infos

            # Vectorize over particles
            final_states, all_infos = jax.vmap(init_and_sample)(keys, z_jax)

            # Extract final positions
            z_final = final_states.position

            # Calculate acceptance rates
            acceptance_rates = jax.numpy.mean(all_infos.is_accepted, axis=1)
            mean_acceptance = jax.numpy.mean(acceptance_rates)

        elif algorithm == "nuts":
            # Initialize step size and mass matrix if not provided
            inverse_mass_matrix = self.sampler_kwargs["inverse_mass_matrix"]
            if inverse_mass_matrix is None:
                inverse_mass_matrix = jax.numpy.eye(self.dims)

            step_size = self.sampler_kwargs["step_size"]

            # Initialize NUTS sampler
            nuts = blackjax.nuts(
                log_prob_fn,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
            )

            # Initialize states for each particle
            n_particles = z_jax.shape[0]
            keys = jax.random.split(subkey, n_particles)

            # Vectorized initialization and sampling
            def init_and_sample(key, z_init):
                state = nuts.init(z_init)

                def one_step(state, key):
                    state, info = nuts.step(key, state)
                    return state, (state, info)

                keys = jax.random.split(key, self.sampler_kwargs["n_steps"])
                final_state, (states, infos) = jax.lax.scan(
                    one_step, state, keys
                )
                return final_state, infos

            # Vectorize over particles
            final_states, all_infos = jax.vmap(init_and_sample)(keys, z_jax)

            # Extract final positions
            z_final = final_states.position

            # Calculate acceptance rates
            acceptance_rates = jax.numpy.mean(all_infos.is_accepted, axis=1)
            mean_acceptance = jax.numpy.mean(acceptance_rates)

        elif algorithm == "hmc":
            # Initialize HMC sampler
            hmc = blackjax.hmc(
                log_prob_fn,
                step_size=self.sampler_kwargs["step_size"],
                num_integration_steps=self.sampler_kwargs.get(
                    "num_integration_steps", 10
                ),
                inverse_mass_matrix=(
                    self.sampler_kwargs["inverse_mass_matrix"]
                    or jax.numpy.eye(self.dims)
                ),
            )

            # Similar vectorized sampling as NUTS
            n_particles = z_jax.shape[0]
            keys = jax.random.split(subkey, n_particles)

            def init_and_sample(key, z_init):
                state = hmc.init(z_init)

                def one_step(state, key):
                    state, info = hmc.step(key, state)
                    return state, (state, info)

                keys = jax.random.split(key, self.sampler_kwargs["n_steps"])
                final_state, (states, infos) = jax.lax.scan(
                    one_step, state, keys
                )
                return final_state, infos

            final_states, all_infos = jax.vmap(init_and_sample)(keys, z_jax)
            z_final = final_states.position
            acceptance_rates = jax.numpy.mean(all_infos.is_accepted, axis=1)
            mean_acceptance = jax.numpy.mean(acceptance_rates)

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Convert back to parameter space
        z_final_np = to_numpy(z_final)
        x_final = self.preconditioning_transform.inverse(z_final_np)[0]

        # Store MCMC diagnostics
        self.history.mcmc_acceptance.append(float(mean_acceptance))

        # Create new samples
        samples = SMCSamples(x_final, xp=self.xp, beta=beta)
        samples.log_q = samples.array_to_namespace(
            self.prior_flow.log_prob(samples.x)
        )
        samples.log_prior = samples.array_to_namespace(self.log_prior(samples))
        samples.log_likelihood = samples.array_to_namespace(
            self.log_likelihood(samples)
        )

        if samples.xp.isnan(samples.log_q).any():
            raise ValueError("Log proposal contains NaN values")

        return samples

    def _jax_log_prob(self, z, beta):
        """JAX-compatible log probability function."""
        import jax.numpy as jnp

        # Single particle version for JAX
        z_expanded = jnp.expand_dims(z, 0)  # Add batch dimension
        log_prob = self.log_prob(z_expanded, beta=beta)
        return log_prob[0]  # Remove batch dimension
