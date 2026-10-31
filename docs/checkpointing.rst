Checkpointing and Resuming
==========================

Aspire provides a few simple patterns to resume long runs.

.. warning::

   Saved flows and sampler checkpoints from the pre-Sequince Aspire backend
   cannot be resumed by this backend. Refit the proposal and start a new
   checkpoint file. Existing sample datasets can still be loaded independently.

Saving checkpoints while sampling
---------------------------------

- Pass ``checkpoint_path`` (an HDF5 file) to :py:meth:`aspire.Aspire.sample_posterior`
  to write checkpoints as the sampler runs. Use ``checkpoint_every`` to control
  frequency and ``checkpoint_save_config/flow`` to control what metadata is saved.
- For a convenience wrapper, wrap your sampling in
  ``with aspire.auto_checkpoint("run.h5", every=1): ...``. Inside the context,
  ``sample_posterior`` will default to checkpointing to that file, and the config/flow
  will be updated as needed.

What gets saved
^^^^^^^^^^^^^^^

- Sequince stores its runtime checkpoint under ``/checkpoint/runtime`` in the
  HDF5 file, with iteration history alongside it.
- Aspire writes ``/aspire_config`` for Aspire-level configuration,
  ``/sampler_config`` for sampler metadata, and ``/flow`` for the saved flow.
  If these already exist, they are overwritten when saving.
- For native Sequince checkpoints, sampler metadata may be read from either
  ``/sampler_config`` or an embedded ``/aspire_config`` entry. This does not
  make pre-migration checkpoint payloads compatible.

Resuming from a file
--------------------

- Use :py:meth:`aspire.Aspire.resume_from_file` to rebuild an Aspire instance and flow
  from a checkpoint file:

  .. code-block:: python

      aspire = Aspire.resume_from_file(
          "run.h5",
          log_likelihood=log_likelihood,
          log_prior=log_prior,
      )
      # Optionally continue checkpointing to the same file
      with aspire.auto_checkpoint("run.h5", every=1):
          samples = aspire.sample_posterior(...)

- ``resume_from_file`` loads config, flow, and the last checkpoint (if present), and
  primes the instance to resume sampling; you can still override sampler kwargs when
  calling ``sample_posterior``.

Resuming with ``auto_checkpoint``
---------------------------------

- If you use ``auto_checkpoint(..., resume=True)``, it will attempt to load the flow
  and config from the checkpoint file when entering the context. If a checkpoint is
  found, it will also load that and prime the instance to resume sampling. If no
  checkpoint is found, it will simply load the flow and config (if present) and start
  fresh sampling.

  .. code-block:: python

      aspire = Aspire(...)

      with aspire.auto_checkpoint("run.h5", every=1, resume=True):
          # Flow with be loaded from the file if present and training will be skipped
          aspire.fit(...)
          # If a checkpoint was found, sampling will resume from that; otherwise it will start from scratch
          samples = aspire.sample_posterior(...)



Manual resume via ``sample_posterior`` args
-------------------------------------------

- If you have a checkpoint blob (bytes or dict) already, you can pass it directly:

  .. code-block:: python

      samples = aspire.sample_posterior(
          n_samples=...,
          sampler="smc",
          resume_from=checkpoint_bytes_or_dict,
          checkpoint_path="run.h5",  # optional: keep writing checkpoints
      )

- To resume from a file without using ``resume_from_file``, load the checkpoint bytes
  and flow yourself, then call ``sample_posterior``:

  .. code-block:: python

      from aspire.utils import AspireFile
      from sequince.checkpoint import SMCCheckpoint

      aspire = Aspire(..., flow_backend="zuko")
      with AspireFile("run.h5", "r") as f:
          aspire.load_flow(f, path="flow")
          checkpoint = SMCCheckpoint.load_hdf5(f, path="checkpoint")
      samples = aspire.sample_posterior(
          n_samples=..., sampler="smc", resume_from=checkpoint
      )

Notes and tips
--------------

- Checkpoint files must be HDF5 (``.h5``/``.hdf5``).
- If a checkpoint is missing in the file (e.g., sampling never wrote one), the flow
  and config are still loaded; you can simply start sampling again and checkpointing
  will continue to the same file.
- For manual control, you can always call ``save_config`` / ``save_flow`` yourself
  on an :class:`aspire.utils.AspireFile`.
- SMC and MCMC samplers accept native Sequince iteration callbacks through
  ``callbacks``. Each callback receives ``(state, context, result)`` after an
  iteration and may return an updated context.
- They also accept ``checkpoint_callback`` and ``checkpoint_every`` for custom
  persistence. The callback receives a committed
  :class:`sequince.checkpoint.SMCCheckpoint`, which can be serialized directly
  or inspected before being sent elsewhere.
- When resuming, you must still specify the keyword arguments for :code:`sample_posterior`.
