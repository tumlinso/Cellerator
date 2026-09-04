# Profile-aware compile benchmark v1

The benchmark compiles one immutable relation-field source against two profile
states. The source semantics fingerprint is identical for both compilations;
the profile and candidate-search fingerprints differ deterministically.

The compiler copies a bounded pointer-free profile state into compiler-owned
cold data, propagates structural, numerical, and reuse evidence into explicit
candidate-search inputs, and records profile-load time, propagation time, and
compiler working-set bytes. No runtime state, storage policy, CUDA dependency,
or Part Two JIT mechanism is introduced.

Validation uses `ce_ccp1_d03_015`. A standalone optimized host benchmark can be
built from `bench/compiler/profile/profile_aware_compile_benchmark.cc` and
`src/compiler/profile/deliver_the_first_profile_aware_compile_benchmark.cc`.
It reports both deterministic fingerprints and measured mean compile latency;
timings are evidence only and are deliberately excluded from fingerprints.

Reference measurement on 2026-09-04 used GCC 13.3.0, C++17, `-O3`, Linux
x86_64, 100,000 warm-cache compilations per state, and no CUDA. Both states
used 808 bytes of bounded compiler working memory. The baseline state selected
2 candidates and the recurrent state selected 5; their search fingerprints
were `7461185111684886846` and `9166336267884262530`, while both retained source
fingerprint `12204476778501210427`. Exact nanosecond timings are emitted by each
run because they are host-load dependent and are not promoted as a performance
claim.
