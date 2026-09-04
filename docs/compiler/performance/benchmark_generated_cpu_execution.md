# Generated CPU execution benchmark v1

Relation and multi-operation fields compare generated CPU, direct C++, and the
existing Cellerator runtime. Complete cost separates preparation, transforms,
packing, execution, reuse, peak memory, and output-order recovery. Correctness
uses one independent reference and exact domain/order identities. Eleven raw
samples per cell run under the benchmark mutex; setup is never hidden in reuse.
