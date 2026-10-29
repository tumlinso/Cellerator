# CE-GEO geometry-layout evaluation: partial v12 evidence

Status: **stopped before the candidate timing matrix**. This is a preserved
partial run, not a layout-performance conclusion. The root requested stopping
after identifying a quadratic host setup path in the immutable v12 package, to
avoid repeating it for every candidate. No complete-cost comparison, confidence
interval, regression assessment, or break-even result was produced.

## Frozen identities and workload

- Cellerator source revision at launch: `e55fc7c3d570c6880db0f73fcb39b573ffbaf73d`.
- Immutable Torch v12 identity SHA-256:
  `99da528db6f4ba14bec983b6dfe31f12a065dbdc8b39c0dfb224ed0e969fca07`.
- Torch extension SHA-256:
  `6a85eebb37d26bf7ddc94a74026a67723d5592470a30b3a55349f2fa95b6071c`.
- KDA checkout revision: `277c79b8cdd1d13c34a0dfd132b8bb504bcb5afc`.
  Its pre-existing dirty files were outside this experiment and were not
  changed here.
- KDA task SHA-256:
  `05f6e91423f2f149b6a3d054ff98bedeb2fdc6cf0e641019cf3d96b31772d45d`.
  It requires 21 paired samples, two exploration rounds, and two confirmation
  rounds. All five candidate selectors and image hashes prepared successfully
  on CPU after correcting a task-relative adapter path. The earlier failed
  prepare record remains in `../kda_workspace/`.
- Prepared-image manifest SHA-256:
  `6ab55eb2476f306188de3a105c711af7e6072bc37c27e876fe628b9761a86ed7`.
- Small fixture manifest SHA-256:
  `c33facc18f330739411c1f293609c237689685bd4003099ca79b331fded144be`.
- Large ranked fixture manifest SHA-256:
  `674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32`.
- The only completed GPU numerical run used `large_ranked`: N=4,096,
  E=262,144, B=32, rank count 512, on `sm_70`.

## Completed validation

Candidate `hybrid8` completed large-ranked Compute Sanitizer memcheck under the
admitted job. The log reports `ERROR SUMMARY: 0 errors`. The frozen SIMT oracle
checks passed for inference output, training parameter gradient, and loss;
both paths retained 519,961 nonzero derivative-only probability gradients,
there were zero read tickets left, and the maximum parameter difference after
the one training update was `2.9802322387695312e-08`.

The exact image was
`prepared_images/large-ranked/hybrid8.cegeo`, SHA-256
`5283fe30383d09713383f82e4580781c50688025857d998868be4e1fd77ad924`. It maps
165,425 support edges through the MMA route and 96,719 through 256 explicit
residual regions (7,687 tiles). This verifies that one hybrid residual image
preserves the tested forward and VJP behavior on the declared large fixture.
The prepared-image manifest records 210.819734 ms of host geometry compilation
for this image; total incremental module/image setup and break-even still
require the uninstrumented paired measurements.

The run JSON contains setup and allocation diagnostics measured while
Compute Sanitizer was active. They are retained as raw evidence and are not used
as normal setup or performance measurements.

## Scheduler attempts and cleanup

Both attempts used the stable GPU1 resource
`gpu:testlsserver:GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505`, with an exclusive
8 GiB admission estimate, a 3-sample zero-utilization precondition, and a
finite timeout.

1. Job `4a5e1199-c304-4cb5-8313-d96a61943703`, attempt
   `26f99b25-0a41-4ad6-ad17-df4ad58e6bdf`, failed before candidate validation:
   the geometry driver emitted `geometry_edge_ids`, while the shared training
   driver expected `geometry_edge_ids_in_support_order`. This harness key was
   fixed and checked on CPU across canonical, reordered, hybrid8 and hybrid16
   image maps. The scheduler receipt reports `cleanup_verified: true`.
2. Job `46491975-fdc9-4123-a6ce-f4b7c28c74a8`, attempt
   `5c523a20-538e-4211-9e9b-547db82328ab`, passed the quiescence precondition;
   all three GPU1 observations showed no process and 0% utilization. Hybrid8
   memcheck then passed as above. The root directed cancellation during the
   subsequent hybrid16 memcheck after finding a costly v12 setup path. The
   scheduler reports terminal `cancelled`, `cleanup_verified: true`, and the
   systemd unit absent after stop. The post-cancel scheduler inventory showed
   GPU1 available, no process, and all 16,929,259,520 bytes free; its one
   utilization sample was 3%, so it is not described as quiescent after the
   job.

   The KDA timing matrix did not start. Hybrid16 memcheck, small and event-local
   GPU validation, all 21-sample candidate timings, and break-even analysis
   remain unrun.

## Setup limitation and next qualified attempt

The root's source review identified an O(P^2) scalar-ID duplicate check in the
v12 training constructor at
`src/compute/operation/indexed_mechanism/training.cu`. The reported P≈300,000
case implies about 45 billion comparisons for one owner setup. This source
review, rather than a geometry timing result, is why the v12 candidate matrix
was stopped.

The current KDA candidate manifests and prepared selectors are bound to v12.
For a v13 rerun, record the new immutable package identity, prepare the same
five candidates in a fresh comparison workspace, and retain these v12
memcheck, harness-failure and cancellation records. Do not interpret the v12
partial run as evidence that topology reordering or hybrid images help or hurt.

## Consumed guidance identities

Local source SHA-256 values for the methods used in this evaluation:

| Guidance file | SHA-256 |
|---|---|
| CUDA skill entry | `3992c5b0e85409cce8002fcfe2c6caed63df6451daad9bae28ccd64a381d717a` |
| Device Scheduler host execution | `311edd1ea9055c70914bddebe7b5c6fbe36789cfaafa4491c153b2e713fbc90d` |
| Benchmark standardization | `6c8a7c7547234add7bc785ab2ceca3ca52873349e717f4ae44c29a6a83891d79` |
| Sparse biological workloads | `81672cd93c3d50aa002cda559eabcbddcce873ec2f2824f40168e47f97382df1` |
| Machine-aligned design | `78b489b241c5be8907f36f8f26e70edd39c9d4bfb26e337d0fe7eb0ce693e779` |
| Compute Sanitizer playbook | `80a687fe3de534226b7f3ae8496f5e36f49fbaf7c9678e5f7f0a52f3e3fbdb7f` |
| Volta router | `54f91d1e42395eac485862a0970d74ea641642f63b96ffbdc2381ca52a0096b7` |
| KDA agent entry | `e7f85c69274b6524c5fc4e13f3cffec8c7d42469f23a4e1e1c0555c47451bbc6` |
| KDA experiment engine | `57ee81aa0300553d2f154d4c289435c730ef905872fe97b21f9c2a6d37e93b78` |
| KernelWiki entry | `90174aa1f6c86bd3156cf25be807d20ad233d3abab3d3b5ac9310a87e3e9a35e` |
