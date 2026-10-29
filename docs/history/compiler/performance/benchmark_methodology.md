# Compiler benchmark methodology v1

Every accepted record identifies the hardware and topology, OS, compiler
toolchain, Cellerator source commit and dirty state, benchmark-source and binary
hashes, selected biological profile and hash, input identity, and exact baseline.

Cold mode starts with no reusable compiler process, parsed source, profile,
cache, or generated artifact. Warm mode names the retained state and reports
both first and steady-state samples. Use at least two warmups and an odd number
of at least eleven measured repetitions. Preserve raw samples; summarize median,
median absolute deviation, minimum, maximum, and a bootstrap 95% interval.

Record wall time, CPU time, peak resident memory, output and intermediate bytes,
diagnostic count, and exit status. Contamination includes competing workload,
thermal or clock change, cache-state mismatch, source mutation, and failed
benchmark-mutex acquisition. Contaminated runs are retained but rejected for
comparison. GPU-involving generated-program runs additionally require the GPU
lease; compiler-only host runs require the repository benchmark mutex.
