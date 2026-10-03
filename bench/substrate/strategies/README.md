# Bounded host cost sample

```
python3 -B bench/substrate/strategies/check.py --build-dir /tmp/ce-is1-strategies-sample --sdk-prefix /tmp/ce-is1-sdk-a --output /tmp/ce-is1-strategies-sample.json
```

This one FP64 toy relation has four source rows, three destination rows and six
contributions (five distinct pairs). It records averages over 512 repetitions after
16 warmups, with a compiler barrier between repetitions. Preparation includes actual
plan validation and both native route preparations. Publication copies owned plan
metadata; migration converts caller state into the selected physical order. Forward
and input VJP include actual native execution and both physical conversions. Consumer
checks are included; the installed provider retains its producer compilation flags.
These timings are illustrative cost inputs, not scalable throughput evidence or a
winning-strategy recommendation. No private sibling implementation is compiled.

A direct native vector multiplication plus explicit canonical reduction is a control.
Repair timing records cold metadata repair; subsequent native re-preparation and
publication remain separate costs. The recorded publication byte count covers
permutation arrays, not an allocator footprint or a complete persistence transaction.

The deferred CUDA occupancy evaluator remains deferred. This tiny host measurement
provides no evidence that launch/transfer overhead or a new evaluator would amortize.
The current CPU Cellpack exact oracle remains available. No GPU launch was performed.
