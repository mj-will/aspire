Examples
========

The repository ships with runnable scripts that demonstrate typical Aspire
workflows. Execute them from the examples directory after installing the relevant
extras.

Sequential Monte Carlo (MiniPCN)
--------------------------------

.. literalinclude:: ../examples/smc_example.py
   :language: python
   :linenos:
   :lines: 1-80
   :caption: ``examples/smc_example.py`` (excerpt)

Run the full example:

.. code-block:: console

   $ python smc_example.py

The script demonstrates how to:

- Build contrived mixtures of Gaussians for testing,
- Fit a Neural Spline Flow to biased initial samples,
- Run adaptive MiniPCN-SMC via :meth:`aspire.Aspire.sample_posterior`,
- Plot diagnostics (loss curves, beta schedule, corner plots).
