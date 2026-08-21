# Cellerator execution session v1

`execution_session` extends the existing `Cellerator::runtime` substrate. It is
not a wrapper around `DeviceMathContext` and does not depend on CP-Math.

Preparation has three explicit allocation lifetimes:

- structure-persistent storage survives value generations;
- plan-persistent storage survives launches for one prepared strategy;
- graph-stable storage has an address that cannot change after sealing.

Each registered stream has its own transient arena and library handles. Streams
may be caller-owned or session-owned. Caller-owned streams are never destroyed
by session teardown. Library creation, device discovery, and workspace growth
are preparation operations. `bind_launch` is valid only after `seal_session`;
it performs no CUDA call, allocation, hashing, device selection, or
synchronization.

The fixed plan, projection, and order-transform caches accept already-computed
semantic keys. They deliberately do not hash pointers or structures in the hot
path. Higher layers own key construction and cache-value destruction.

CE-ARCH-60 removed the experimental CP-Math runtime island after its useful
consumers and evidence moved to the operation core, explicit launch bindings,
and compatibility targets. The completed ownership migration is:

| Experimental ownership | Session v1 ownership |
| --- | --- |
| owned execution stream | registered caller stream or explicit pool slot |
| one mutable `WorkspacePool` | one pre-reserved transient arena per stream |
| embedded cuBLAS/cuSPARSE caches | prepared handles per registered stream |
| prepared launch pointers | external launch bindings |
| implicit workspace growth | preparation reserve followed by sealing |

The session records a single device performance class. `device_fleet_view`
allows later multi-device planning without adding distributed execution or
fleet metadata to every launch record.
