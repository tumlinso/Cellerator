# Private-port derivatives

`ops.forward(h,e,d,weights,widths,src,dst,generations=(0,0,0,0))`
returns flattened `y` and a saved tape. Every numerical argument is a flattened
finite NumPy float32 array; metadata arrays are flattened int64 arrays.
Widths are positive private actor widths. The common interface width is
`P=e.size/h.size`. Each actor contributes row-major `E_i[P,H_i]` and
`D_i[H_i,P]` blocks. Every ordered directed edge contributes one independent
weight, including duplicate `(src,dst)` pairs and weights currently equal to zero.

The result is `y_i = D_i sum_{edges with dst=i} w_edge E_src h_src`.
The caller owns the local law `phi_i`; this operation computes transport alone.
`vjp(tape,g,current_generations=None)` returns `(dh,de,dd,dweights)`.
`jvp(tape,dh,de,dd,dweights,current_generations=None)` returns `dy`.
Supplying current generations rejects a mismatch in any saved operand generation.
The four generations correspond to h, e, d, weights. Saved values and metadata
are copies backed by immutable bytes. Derivatives never mutate the saved primal.

The native C++17 CPU float32 implementation is compiled on first use with `c++`
and loaded through ctypes. A cache keyed by source SHA256, compiler version and compile flags
lives in the operating system temporary directory. No production operator
registration or GPU execution is supplied by this experimental module.

Run `python3 -B experiments/moonshot-parallel-v1/diff/ports/test_ports.py`.
The six tests cover heterogeneous widths, repeated directed edges, all input
and parameter finite differences, zero edge-weight derivatives, JVP finite
differences, the VJP/JVP adjoint identity, empty edges, immutable saved values,
stale-generation rejection, malformed public Tape construction, and admission checks.

Public Tape construction validates the same extents, dtypes, positive dimensions,
edge indices and generations as forward, then snapshots immutable copies. The
exported C function is a private ctypes ABI: Python establishes pointer buffer
lengths. Its own admission rejects invalid mode, scalar dimensions, required
null pointers, signed indexing overflow, nonpositive widths and out-of-range
edges; arbitrary raw pointer allocation lengths remain the caller's obligation.
