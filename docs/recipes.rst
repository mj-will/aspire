Practical recipes
=================

Checking the prior when evaluating the likelihood
-------------------------------------------------

By default, Aspire samplers always evaluate the log-prior before the
log-likelihood. This allows users to check the prior support and skip
likelihood evaluations for samples that lie outside the prior bounds.

.. code-block:: python

    import aspire
    import numpy as np


    def log_likelihood(samples: aspire.Samples) -> np.ndarray:
        if samples.log_prior is None:
            raise RuntimeError("log-prior has not been evaluated!")
        # Return -inf for samples with invalid prior
        logl = np.full(samples.n_samples, -np.inf, dtype=float)
        # Only evaluate the likelihood where the prior is finite
        mask = np.isfinite(samples.log_prior, dtype=bool)
        # Valid samples
        x = samples.x[mask, :]
        logl[mask] = -np.sum(x**2, axis=1)  # Example likelihood
        return logl


Checking the flow distribution
------------------------------

It can be useful to inspect the flow-based proposal distribution before sampling
from the posterior. You can do this by drawing samples from the flow after fitting
and comparing them to the initial samples:


.. code-block:: python

    from aspire import Aspire, Samples
    from aspire.plot import plot_comparison

    # Define the initial samples
    initial_samples = Samples(...)

    # Define the Aspire instance
    aspire = Aspire(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        ...
    )

    # Fit the flow to the initial samples
    fit_history = aspire.fit(initial_samples)

    # Draw samples from the flow
    flow_samples = aspire.sample_flow(10_000)

    # Plot a comparison between initial samples and flow samples
    fig = plot_comparison(
        initial_samples,
        flow_samples,
        per_samples_kwargs=[
            dict(include_weights=False, color="C0"),
            dict(include_weights=False, color="C1"),
        ],
        labels=["Initial samples", "Flow samples"],
    )
    # Save or show the figure
    fig.savefig("flow_comparison.png")
