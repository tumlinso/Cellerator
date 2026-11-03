# Receipt-bound measurement qualification

One serialized packet passed the receipt, source, telemetry and per-arm evidence join. Neither experimental arm met the strict greater-than-5% latency reduction gate. The [machine-readable export](qualification.json) contains all paired observations, retained source identities, failures and sanitizer records.

Frozen contract SHA-256: `237da579f5068fc26469f52c79b7ae33beb1ab0af057129a27c186193c5c8147`. Frozen measurement matrix SHA-256: `c2d40e206fa21692a704a65e029a159f38cb023183e4cc7a28481b4f35749241`.

The measurement is one 10x replay scenario: teacher gate, grouped-linear FP32 body, incremental condition updates, one-step reuse, tile 32, one staging buffer, and six independent paired rounds on the same V100. The boundary includes cold execution preparation, the recurrent complete horizon, and output recovery; fixture, oracle, reset and warmup are excluded. The fastest matched control among scalar, bitmap, direct, library, and library-batched was bitmap at 465.30 ms median.

| Experimental arm | Median latency reduction vs bitmap | Paired speedup, 95% interval | Decision |
| --- | ---: | ---: | --- |
| cohort-direct | -2.55% | 0.9751 [0.9368, 1.0095] | rejected: no median improvement |
| cohort-shared | 0.49% | 1.0050 [0.9718, 1.0553] | practically inconclusive: latency reduction does not exceed 5% |

The 95% interval is the paired bootstrap interval for the median speedup, not a range of observed timings. This is a bounded software and performance finding for the stated scenario; remaining matrix coverage is incomplete.

All four representative sanitizer tools passed six complete-chain gate/body cases on the exact measured binary and source archive. The export retains these four receipts as its primary evidence and also preserves the earlier source-identical results. This representative FP16 synthetic cohort-shared coverage is not exhaustive. See [sanitizer evidence](sanitizer-coverage.json).

Earlier failed attempts are preserved:
- `3f4dee5a-1c99-4e60-a35c-f1b137cced29`: worker status `storage_over_cap`; source archive `1f88bab3706ab3d06173660e82e61c83bb0ab6e05cf55025994fb623bd778170`; binary `a2604bd79dfbfac9cc137e4223a03e5b6698352defad2c42b2c10f1a7805a585`. The first arm was retained, but the matrix was incomplete.
- `85dc73fb-3cbd-4408-939b-5a9720256b5e`: worker status `failed`; source archive `1f88bab3706ab3d06173660e82e61c83bb0ab6e05cf55025994fb623bd778170`; binary `a2604bd79dfbfac9cc137e4223a03e5b6698352defad2c42b2c10f1a7805a585`. The full 42-arm matrix is retained, but telemetry was unclean: {'target utilization has no attributable process': 30}.

The successful packet used study serialization and sampled all-device telemetry with retained, output-bound native child lifetimes for retrospective attribution. Sampling cannot establish host-wide exclusivity or exclude activity between observations. The prior concurrent subprocess probe is inconclusive. Snapshot manifests were independently reconstructed from immutable snapshot bytes and verified against the snapshot fingerprints; the original SDK manifest files were unavailable.
