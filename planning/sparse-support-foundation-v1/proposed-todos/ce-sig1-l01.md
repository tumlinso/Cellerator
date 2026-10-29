# CE-SIG1-L01: Implement learned gate and compile lifecycle

Implement shared learned entity/mechanism embeddings and condition parameters; bounded proposal scoring over existing postings, recent co-activation and seeded exploration; sampled pair/triple participant proposals; soft-gate training followed by pruned-forward fine-tuning with the declared straight-through estimator. Record estimator, threshold, temperature and derivative-candidate policy. Add learn-then-compile lifecycle with trainable values mutable after routing compilation.

Acceptance: distinguish bounded discovery coverage/misses from exact retrieval; compare against unpruned and shuffled-support controls; preserve an explicit FP32 path and correct storage/multiply/accumulate/gradient precision declarations. No claim of exact gradients over unsampled dormant candidates.
