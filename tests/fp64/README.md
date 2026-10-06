# FP64 native qualification

This standalone target builds against the repository's current
`Cellerator::prepared_relation_cuda` target. It does not carry a second
relation kernel. The native executable exercises the four relation/feature
tuples, forward and transpose application, widths `1, 3, 15, 16, 17, 33, 65,
257`, two value generations, signed weights, and irregular rows including an
empty row. The prepared provider currently rejects repeated endpoints because
its retained masked topology cannot preserve distinct edge slots; the harness
checks that rejection. The shared CSR traversal is separately tested with
duplicate edges and a logical value map. The Python driver compares each emitted device
result with SciPy CSR using values rounded to their stored operand dtypes.

Configure and build outside the repository:

```sh
cmake -S tests/fp64 -B /tmp/cellerator-fp64-build \
  -DCELLERATOR_SOURCE_DIR="$PWD" -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build /tmp/cellerator-fp64-build --target fp64_qualification -j2
```

The standalone project also defines `fp64_cost_bench` and build-only copies of
the current FP16 witnesses (`fp64_existing_adapter_test`, `forward`,
`transpose`, `lifecycle`, and `overhead`). These copies repair only the former
private `p->logical_map` field path and relative include after materializing in
the build directory; the repository's archived witness sources are unchanged.
The older `preparation` and `generation` witnesses call removed `submit`/map
internals and are excluded as source-incompatible.

Run `compare_scipy.py` under the foreground CUDA controller, with the actual
device and stream-interlock settings supplied by that controller:

```sh
python3 tests/fp64/compare_scipy.py \
  --binary /tmp/cellerator-fp64-build/fp64_qualification \
  --receipt /tmp/cellerator-fp64-scipy.json
```

The native harness also checks homogeneous float/double elementwise multiply,
in-place squaring, affine combination, and partial-overlap rejection. FP64
relation comparisons use `rtol=atol=3e-13`; the float32-only tuple uses
`2e-6`. A composed `W*x`, `W*(x*x)`, and `E[x^2]-E[x]^2` pipeline is checked
against SciPy and a centered reference for well-conditioned and cancellation-
heavy values. The cancellation bound propagates `gamma_n` error from the
first and second moments; it does not imply universally stable variance.

For independent memory-safety validation, run the same binary through
Compute Sanitizer under the CUDA controller:

```sh
compute-sanitizer --tool memcheck /tmp/cellerator-fp64-build/fp64_qualification
```

Measure cost after building `fp64_cost_bench`; first build the source-bound
historical FP32 wrapper, then run the measured harness through the foreground
CUDA controller:

```sh
python3 tests/fp64/run_checks.py --phase all \
  --build /tmp/cellerator-fp64-build --output planning/fp64/evidence
```

The gate runner owns foreground CUDA controller admission and passes the
baseline library explicitly to the cost harness. For a cost-only run, its
equivalent commands are:

```sh
python3 tests/fp64/measure_costs.py --build-baseline-only \
  --repository "$PWD" --binary /tmp/cellerator-fp64-build/fp64_cost_bench \
  --baseline-library /tmp/cellerator-fp64-build/libhistorical_csr_baseline.so
python3 tests/fp64/run_checks.py --phase costs \
  --build /tmp/cellerator-fp64-build --output planning/fp64/evidence
```

The measurements separate preparation, transfers, resident execution, and
end-to-end work; report persistent bytes and provider temporary bytes. Do not
infer performance or memory cost from dtype size or this correctness target.
