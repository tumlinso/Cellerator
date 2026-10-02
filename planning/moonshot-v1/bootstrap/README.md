# Safe copy helpers

`check_package.py` verifies an immutable package manifest. `check_plans.py` checks local IDs, references and graph consistency; it is not the native Todo validator.

`install.py` previews by default and copies into new planning directories only with `--apply`. It checks independent repository roots, source revision, markers and destination absence. An explicit reviewed source-drift override does not grant authority to import a plan. Existing dirty work is reported, not reset.

`stage_source.py` previews or copies a role payload into a new `experiments/moonshot-v1`; take the matching source claim first. It never overwrites an existing directory.

The helpers do not call agents, mutate Todo storage, apply plans, or change library source. Copies to two repositories are not globally atomic. If a copy fails, inspect the newly created partial directory; the tool intentionally does not erase possible concurrent changes in a cleanup routine.
