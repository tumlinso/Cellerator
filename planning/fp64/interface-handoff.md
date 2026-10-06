# FP64 native interface handoff

Owner task: CE-FP64-001; run CE-FP64-RUN-1.

Supported relation weight/feature/output tuples: f32/f32/f32, f32/f64/f64, f64/f32/f64, f64/f64/f64. New tuples multiply and accumulate in FP64. Existing f16 behavior remains bounded to its current envelope.

The user approved widening operation_descriptor.input_scale and destination_scale to double. FP32 execution retains float coefficient rounding. Device state/result views gain trailing dtype metadata defaulting to f32. FP64 values get typed publication; the existing f32 API stays available. Rebuild native/binding consumers after these layout changes.

Native API names: `device_numeric_values_binding<T>` underlies `device_f32_values_binding` and `device_f64_values_binding`; publish with `publish_f32_values` or `publish_f64_values`. Set `device_state_view::dtype` and `device_result_view::dtype` explicitly for mixed or FP64 storage. The homogeneous `multiply<T>`, `square<T>`, and `axpby<T>` functions are declared in `device_elementwise.hh` and linked by the existing prepared-relation CUDA library. Exact output/input aliases are permitted for these elementwise operations; partial overlap fails before submission.

CE-PYBIND-ABSORB retains all binding/CMake changes. CE-MOM-PROBE-01 retains native_numeric. Root owns final numerical/GPU acceptance. The user delivered a warning to the other owner and confirmed it may resume after this task was claimed. Project Control currently supports messages only within one run; this note supplies the cross-run source reference.

The Python/CelleraTorch migration proceeds in the isolated `codex/python-bindings` worktree; FP64 numerical edits stay in the canonical checkout. FP64 leaves root CMake and optional build wiring unchanged. Integration must retain the migration's binding/package/model-operation changes and rebuild consumers of `relation_semantics.hh`, `prepared_relation.hh`, and `project.hh`. The new elementwise declarations use the existing prepared-relation CUDA library. Neither workstream should restore an older copy of these shared headers over the other's changes.

The user authorized changing only both tasks parallel policies to parallel_safe; this retained the moments claim and scopes. This FP64 task then narrowed its own read scopes to avoid overlap.

Qualification uses shared-host CUDA controller admission, NumPy/SciPy references and measured memory/time costs. No FP64 gradient, Tensor Core, or unrelated provider support is claimed.

The user selected preserving the prepared provider's existing explicit duplicate-endpoint rejection. Its masked topology schema has one slot per endpoint; duplicate arithmetic is qualified in the shared CSR kernel. No structural schema or gradient mapping change is included.

Context routing: ctxpp status is degraded-text-routing, semantic=false and incomplete/stale. Canonical source inspection is used; generated indexes are not edited by hand.
