# Part Two separation audit

The Part One compiler and final link graph contain no NVRTC or general-JIT
target. All validated execution enters through AOT objects, prelinked providers,
or explicit native artifacts. The retained writable CEIR/object contracts are
versioned seams, not runtime-compilation prerequisites.

CellShard appears only at the concrete application boundary and through the
narrow opaque materialization request. The request describes a compiler-owned
portable schedule; it does not implement deep persistence, distribution, or
runtime evolution. Source and target searches found no hidden Part Two link
dependency. Both deferred families remain marked non-prerequisites in the
machine-readable capability matrix.
