Multiprocessing
===============

Use :meth:`aspire.Aspire.enable_pool` to run your likelihood (and optionally
prior) in parallel across a :class:`multiprocessing.Pool`. The helper swaps the
``map_fn`` argument expected by your log-likelihood / log-prior for
``pool.map`` while the context is active, then restores the original methods.

Prepare a map-aware likelihood
------------------------------

Your likelihood must accept ``map_fn``. A minimal
pattern:

.. code-block:: python

    import numpy as np


    def _global_log_likelihood(x):
        # Expensive likelihood computation for a single sample `x`
        return -np.sum(x**2)  # Example likelihood

    def log_likelihood(samples, map_fn=map):
        logl = -np.inf * np.ones(len(samples.x))
        if samples.log_prior is None:
            raise RuntimeError("log-prior has not been evaluated!")
        mask = np.isfinite(samples.log_prior, dtype=bool)
        x = np.asarray(samples.x[mask, :], dtype=float)
        logl[mask] = np.fromiter(
            map_fn(_global_log_likelihood, x),
            dtype=float,
        )
        return logl

Swap in a multiprocessing pool
------------------------------

Wrap your sampling call inside ``enable_pool`` to parallelize the map step:

.. code-block:: python

    import multiprocessing as mp
    from aspire import Aspire

    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,   # must also accept map_fn if parallelize_prior=True
        dims=4,
        parameters=["a", "b", "c", "d"],
    )

    with mp.Pool() as pool, aspire.enable_pool(pool):
        samples, history = aspire.sample_posterior(
            sampler="smc",
            n_samples=1_000,
            return_history=True,
        )

Notes
-----

- By default only the likelihood is parallelized; set
  ``aspire.enable_pool(pool, parallelize_prior=True)`` if your prior also
  accepts ``map_fn``.
- ``enable_pool`` closes the pool on exit unless you pass ``close_pool=False``.
- The context manager itself is implemented by
  :class:`aspire.utils.PoolHandler`; if you need finer control (for example,
  reusing the same pool across multiple ``Aspire`` instances) you can
  instantiate it directly.
