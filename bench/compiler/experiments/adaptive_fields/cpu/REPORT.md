# Adaptive execution fields: CPU experiments

Frozen repository source: `c63f591d04ef74b7566937d161c80936dcc4cf05`. Harness compiler: `g++ 13.3.0`; the E2 CPU framework adapter uses Torch `2.7.0+cu126`. No GPU code was run.

## Decisions

| Experiment | Decision | Main result |
|---|---|---|
| E1 composition | Supported in the local adapter | Parsed anonymous, named and nested field forms normalized to one graph; that graph selected existing CPU relation and segment callbacks. Three small fixtures matched an independent FP64 oracle. The production vector semantic IR path remains disconnected. |
| E2 semantic registration | Supported for the narrow descriptor | A compiled ordinary C++ add helper and one CPU ATen overload were invoked. The local selector fused relation/add/softmax only with a semantic body and legal effects; explicit and epilogue results matched. No production registry or Torch bridge was changed. |
| E3 artifact lifecycle | Unresolved | Real support publication and restoration plus real planning-key validation were exercised. A local artifact model covered rebinding, workset changes, structure/order staleness and counters. That dependency model is not wired to production caches. |
| E4 runtime compilation | Supported as a CPU prototype | Existing object compiler plus ordinary shared linking and `dlopen` produced invoked code with runtime value rebinding. Cold, same-process and second-process reuse were exercised, along with identity changes. |
| E6 CPU publication/lifetime | Supported in the local adapter | Real compiled modules were held through blocked calls and safely retired after lease release; stale, failed and cancelled replacements were rejected, and fresh retries passed. |

## Evidence and boundaries

- E1 uses the real field parsers and existing CPU relation and segment implementations. The temporary graph parser/inliner handles only the experiment grammar; it is not a language frontend. The current source-to-semantic-IR vertical slice does not express `p = segment_softmax(Ax)` as a typed connected vector region.
- E2 calls Torch `add` on CPU contiguous same-shaped FP32 tensors with `alpha=1`; a broadcast input is explicitly rejected. A local compiled C++ helper publishes its mathematical body in a local descriptor. A write to bias and an opaque operation block crossing the relation/add rewrite boundary. Fusion preserves relation-sum-then-bias FP32 operation order.
- E3's real manager published logical edge 30 and revoked 10 at structure epoch 5, and checkpoint restoration preserved IDs. Its local counters were `compile=1,index=3,projection=3,workset=3,value_operand=4,rebind=1,pack=0`. Value operands were actually materialized on the primal/derivative union and refreshed after value rebinding and workset changes. The current planning key omits value and active-support generations; the local materialization is not production cache evidence.
- E4's generated source depends on standard compiler headers only. The compiler depfile records the generated source under `-MMD`; standard-library header file contents are not separately hashed, so compiler command/version stands in for those inputs. Topology identity changes produced new generic artifacts; this was not loop specialization by CSR pattern. Single timing probes are plumbing evidence only.
- E6 uses a deterministic CPU worker with one pending slot and real generated `.so` modules. A shared lease held an old executable through an in-flight call while a replacement was published. The independent lifetime verifier confirmed the latest retry and retirement behavior. Device streams and GPU tape lifetimes remain untested.
- All experiment adapters, manifests, helper code, and generated files remain under this directory. No production interfaces, defaults, kernels, application algorithms, or scientific estimators changed.

Structured records: `results/e1.json` through `results/e4.json`, plus `runtime/E6_cpu.json`; source/toolchain manifest: `results/evidence-manifest.json`.
