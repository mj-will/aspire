Checkpointing and Resuming
==========================

Aspire provides a few simple patterns to resume long runs.

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

- The sampler stores checkpoints under ``/checkpoint/state`` in the HDF5 file.
- Aspire writes ``/aspire_config`` (with ``sampler_type`` and ``sampler_config``) and
  ``/flow``. If these already exist, they are overwritten when saving.

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
          samples = aspire.sample_posterior()

- ``resume_from_file`` loads config, flow, and the last checkpoint (if present), and
  primes the instance to resume sampling; you can still override sampler kwargs when
  calling ``sample_posterior``.

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

      aspire = Aspire(..., flow_backend="zuko")
      with AspireFile("run.h5", "r") as f:
          aspire.load_flow(f, path="flow")
          # Standard layout is /checkpoint/state; adjust if you used a different path
          checkpoint_bytes = f["checkpoint"]["state"][...].tobytes()
      samples = aspire.sample_posterior(
          n_samples=..., sampler="smc", resume_from="run.h5"
          # or resume_from=checkpoint_bytes if you prefer to pass bytes directly
      )

Notes and tips
--------------

- Checkpoint files must be HDF5 (``.h5``/``.hdf5``).
- If a checkpoint is missing in the file (e.g., sampling never wrote one), the flow
  and config are still loaded; you can simply start sampling again and checkpointing
  will continue to the same file.
- For manual control, you can always call ``save_config`` / ``save_flow`` yourself
  on an :class:`aspire.utils.AspireFile`.
- SMC samplers also accept a custom ``checkpoint_callback`` and ``checkpoint_every`` if
  you want full control over how checkpoints are persisted or inspected. Provide a
  callable that accepts the checkpoint state dict; from there you can, for example,
  serialize to another format or push to remote storage.
