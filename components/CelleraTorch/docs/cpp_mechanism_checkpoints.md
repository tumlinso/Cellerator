# C++ mechanism checkpoints

A C++ composite model has two parameter owners. Ordinary libtorch layers remain
Torch-owned and can use their normal `torch::serialize::OutputArchive` and
`InputArchive` paths. A prepared mechanism's coefficient leaf aliases storage
owned by Cellerator, so save its logical values with `Mechanism::snapshot()`
and restore them with `Mechanism::restore()` after preparing the replacement
native owner. Save and reload the optimizer state alongside the ordinary model
archive and native logical snapshot.

Exclude mechanism leaves from generic `torch::nn::Module::save/load` archives.
In particular, loading a whole composite module can replace or rebind a
registered parameter's storage, which breaks the native pointer and readiness
contract. The next forward or guarded update rejects that replaced storage.
The native snapshot restore path rebuilds the FP32 master and derived precision
plane while keeping the prepared owner authoritative.

The installed-consumer fixture in
[`tests/mechanism_test.cc`](../tests/mechanism_test.cc) demonstrates this split:
it archives the ordinary `pre` and `post` layers, saves Adam state separately,
and stores the mechanism values through `snapshot()`. Reload constructs a new
prepared owner, loads only the dense layer archives, restores native values,
then checks predictions and the next optimizer step.
