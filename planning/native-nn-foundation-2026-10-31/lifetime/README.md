# Native query lifetime regressions

The query convenience path now retains one typed `PredictionFrame` per batch.
It owns rvalue model/input/base/request/source values and custom full-head
results used by borrowed gathers, and shares one lifetime
token across the returned predictions. For an rvalue conditioned invocation,
the frame also retains the whole invocation as an upstream lifetime anchor;
this is required even when the conditioner returns a value such as
`std::span`, because that value can still borrow from the invocation's input.
That upstream invocation uses one additional shared allocation. Lvalue
dependencies remain caller-owned. Direct model/readout calls do not allocate
this frame and keep caller-managed result lifetimes. The typed frame itself is
one shared allocation per batch, with no per-prediction owner allocations.

## ASan evidence

Run the three captured reproducers and the expanded native query test with:

```sh
bash planning/native-nn-foundation-2026-10-31/lifetime/run_asan.sh
```

Toolchain: `g++-12 (Ubuntu 12.4.0-2ubuntu1~24.04.1) 12.4.0`, C++20, AddressSanitizer
(`-fsanitize=address -fno-omit-frame-pointer`), linked to the configured CPU
LibTorch installation. Runtime options were
`ASAN_OPTIONS=detect_leaks=0:halt_on_error=1`. All three standalone reproducers
printed `42`; the expanded test was silent. The script exited 0 with no
AddressSanitizer report. Binaries are written under `/tmp` only.

The two compact reproducer sources are retained here as
[`query_view_repro.cc`](query_view_repro.cc) and
[`conditioned_view_repro.cc`](conditioned_view_repro.cc). The expanded test is
[`query_readouts_test.cc`](../../../tests/nn/native/query_readouts_test.cc).
It checks a move-only base result borrowed by span, a selector-returned array,
rvalue input/model ownership, value/reference/span-returning conditioners,
multiple requests sharing one frame, and preserved ATen input gradients.

The additional [`full_gather_repro.cc`](full_gather_repro.cc) exercises an
ordinary decoder returning a custom owning result, followed by a gather that
returns `std::span`. ASan confirmed the original heap-use-after-free
([captured report](full-gather-before.log)). The full result is now retained
in an optional typed slot in the existing frame, without an additional owner
allocation. The Tensor `index_select` path still returns an owning tensor.

## CPU tests

```sh
cmake --build /tmp/cellerator-native-nn-cmake --target \
  native_nn_query_readouts_test native_nn_cell_query_integration_test \
  native_nn_supervision_test -j2
ctest --test-dir /tmp/cellerator-native-nn-cmake \
  -R 'native_nn_(query_readouts|cell_query_integration|supervision)_test' \
  --output-on-failure
```

Result: all 3 tests passed. `git diff --check` also passed. No GPU execution was
run.
