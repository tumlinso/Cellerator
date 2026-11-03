# Early correctness evidence retention

The first compact STE index (`ste-validation.json`) was written before lane
reuse. Autobench subsequently evicted some of the referenced raw successful
packet directories under its ordinary current/previous retention policy. The
recorded identities, hashes and Scheduler receipts still exist, but the missing
raw bytes cannot be rechecked and the old index alone is not reproducible
evidence for those checks.

The controller now copies each completed packet to `runs/packets` before
Autobench can import or evict it. The current `correctness-evidence.json` is
generated only from those preserved packet copies and the independently retained
Scheduler receipts. STE correctness and its missing sanitizer modes have been
rerun; the current index retains all four sanitizer modes. Teacher sanitizer
packets for all four modes are also retained, including a rerun of memcheck.
The earlier ordinary teacher packet was evicted; the retained sanitizer packets
execute the same teacher correctness fixture. The failed consumer packet was
separately preserved before import.
