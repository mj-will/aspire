aspire: Accelerated Sequential Posterior Inference via REuse
============================================================

``aspire`` is a lightweight framework for reusing existing posterior samples
and normalizing flows to accelerate Bayesian inference. It focuses on
practical workflows: fit a flow, adaptively run Sequential Monte Carlo (SMC),
MCMC or importance samplers, and visualise or export the resulting samples.

Key capabilities
----------------

- Fit flow-based proposals (PyTorch or JAX backends) with automatic handling of
  bounded and periodic parameters.
- Run adaptive SMC (MiniPCN or BlackJAX kernels) and importance sampling with
  detailed diagnostic histories.
- Inspect results via convenience helpers for evidence estimates, corner plots,
  and HDF5/JSON export.

Quick start
-----------

.. code-block:: python

    import numpy as np
    from aspire import Aspire, Samples

    def log_likelihood(samples):
        x = samples.x
        return -0.5 * np.sum(x**2, axis=-1)

    def log_prior(samples):
        return -0.5 * np.sum(samples.x**2, axis=-1)

    init = Samples(np.random.normal(size=(2_000, 4)))

    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        dims=4,
        parameters=[f"x{i}" for i in range(4)],
    )
    aspire.fit(init, n_epochs=20)
    posterior = aspire.sample_posterior(
        sampler="smc",
        n_samples=500,
        sampler_kwargs=dict(n_steps=100),
    )

    posterior.plot_corner()

Use the sections below for environment setup, conceptual guidance, runnable
examples, and the complete API reference.

.. toctree::
    :maxdepth: 2
    :caption: Contents

    installation
    user_guide
    checkpointing
    recipes
    multiprocessing
    examples
    entry_points
    API Reference </autoapi/aspire/index>


.. toctree::
    :maxdepth: 2
    :caption: Related Packages

    aspire-bilby <https://aspire.readthedocs.io/projects/aspire-bilby/en/latest/>
