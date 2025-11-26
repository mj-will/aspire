Entry Points
============

Aspire uses Python's entry point system to allow for extensibility and
integration with external libraries. This mechanism enables users to register
custom components that can be seamlessly integrated into the Aspire framework.

.. _custom_flows:

Custom Flows
------------

Aspire supports custom flow implementations via the
``aspire.flows`` entry point group. To register a new flow backend, define an
entry point in your ``pyproject.toml`` like so:

.. code-block:: toml

    [project.entry-points."aspire.flows"]
    myflow = "my_module:MyFlowClass"


The specified class must inherit from :class:`aspire.flows.base.Flow` (or one
of the existing flow wrappers), implement the required methods and define the
``xp`` attribute which specifies the array namespace.
You can then select your custom flow by setting
``flow_backend="myflow"`` when initializing Aspire.

For an example see ``GWFlow`` in ``aspire-gw`` (https://github.com/mj-will/aspire-gw).
