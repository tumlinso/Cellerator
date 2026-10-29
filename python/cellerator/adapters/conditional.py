"""Explicit identity bridge from Python plans to native conditional dispatch.

No Python hash becomes a native identity. Callers supply stable numeric mappings;
expert revisions and origin identities are checked for collisions before use.
The optional native module is loaded only when a backend is constructed.
"""
from __future__ import annotations

import importlib
import operator
from types import MappingProxyType
from collections.abc import Mapping

from ..execution.residency import ParameterKey


def _unsigned(value, bits):
    if isinstance(value, bool):
        raise TypeError("native identity components must be integers, not bool")
    result = operator.index(value)
    if not 0 <= result < (1 << bits):
        raise ValueError(f"native identity component does not fit uint{bits}")
    return result


class NativeIdentityMap:
    """Injective, caller-owned encoding of logical identities, independent of layout.

    expert_keys maps ParameterKey to (uint64 expert_id, uint64 revision).
    origins maps (selection.origin, selection.row_id) to native
    (uint32 process_rank, uint64 instance_id, uint64 row_id). Native row_id is an
    identity, not an offset into the input array. Mappings must be shared across
    participants when they exchange native plans; no process-local hash is used.
    """
    def __init__(self, expert_keys: Mapping, origins: Mapping):
        keys, rows = {}, {}
        experts, native_experts = {}, {}
        for key, encoded in expert_keys.items():
            if not isinstance(key, ParameterKey) or len(encoded) != 2:
                raise TypeError("expert mapping requires ParameterKey -> (expert, revision)")
            native_key = (_unsigned(encoded[0], 64), _unsigned(encoded[1], 64))
            if key.expert_id in experts and experts[key.expert_id] != native_key[0]:
                raise ValueError("one logical expert cannot change native identity across revisions")
            if native_key[0] in native_experts and native_experts[native_key[0]] != key.expert_id:
                raise ValueError("distinct logical experts collide in native identity")
            experts[key.expert_id] = native_key[0]
            native_experts[native_key[0]] = key.expert_id
            keys[key] = native_key
        if len(set(keys.values())) != len(keys):
            raise ValueError("distinct parameter revisions collide in native identity")
        for identity, encoded in origins.items():
            if len(identity) != 2 or len(encoded) != 3:
                raise ValueError("origin mapping requires (origin, row) -> (rank, instance, row)")
            rows[identity] = (_unsigned(encoded[0], 32), _unsigned(encoded[1], 64),
                              _unsigned(encoded[2], 64))
        if len(set(rows.values())) != len(rows):
            raise ValueError("distinct origins collide in native identity")
        self.expert_keys = MappingProxyType(keys)
        self.origins = MappingProxyType(rows)
        self._reverse_experts = {value: key for key, value in keys.items()}
        self._reverse_origins = {value: key for key, value in rows.items()}

    def records(self, selections):
        return tuple((*self.expert_keys[selected.key],
                      *self.origins[selected.origin, selected.row_id], selected.weight)
                     for selected in selections)


    def dispatch_plan(self, selections, *, native_module=None):
        """Build the existing host plan, also accepted by CudaDispatchPlan.

        This constructs CPU metadata only; CUDA upload remains an explicit
        operation with an admitted device and caller-owned stream.
        """
        native = (importlib.import_module("cellerator._conditional_native")
                  if native_module is None else native_module)
        return native.DispatchPlan(self.records(selections))


class NativeConditionalBackend:
    """ExecutionEngine callback using the existing native CPU dispatcher.

    Inputs map (origin, row_id) to contiguous one-dimensional NumPy feature
    vectors. callback(key, parameters, values, selection_indices) returns a
    contiguous NumPy output matrix. Parameters are the engine's exact resident
    revision payload; selection_indices refer to the original whole worklist.
    Outputs stay unweighted: the caller performs exactly one owner reduction.

    FP16 input storage is promoted by the native CPU adapter to FP32 callback
    arithmetic/output. FP32 and FP64 remain explicit alternatives. This adapter
    does not launch CUDA or distributed collectives.
    """
    def __init__(self, identities: NativeIdentityMap, callback, *, input_features: int,
                 output_features: int, native_module=None):
        if not isinstance(identities, NativeIdentityMap) or not callable(callback):
            raise TypeError("provide NativeIdentityMap and a callable numerical backend")
        self.identities = identities
        self.callback = callback
        self.input_features = operator.index(input_features)
        self.output_features = operator.index(output_features)
        if min(self.input_features, self.output_features) <= 0:
            raise ValueError("feature counts must be positive")
        self.native = (importlib.import_module("cellerator._conditional_native")
                       if native_module is None else native_module)
        if not hasattr(self.native, "execute_occurrences"):
            raise ImportError("native conditional module must provide execute_occurrences; rebuild it")

    def __call__(self, request):
        import numpy as np
        selected = request.selections
        if any(item.key != request.decision.key for item in selected):
            raise ValueError("execution request substitutes an expert revision")
        plan = self.identities.dispatch_plan(selected, native_module=self.native)
        origin_ids = tuple(self.identities._reverse_origins[
            (origin.rank, origin.instance, origin.row)] for origin in plan.origins)
        rows = tuple(np.asarray(request.inputs[identity]) for identity in origin_ids)
        if any(row.ndim != 1 or len(row) != self.input_features or not row.flags.c_contiguous
               for row in rows):
            raise ValueError("native inputs require contiguous vectors of the declared width")
        if rows and any(row.dtype != rows[0].dtype for row in rows):
            raise TypeError("native input rows must share one explicit storage dtype")
        inputs = np.stack(rows) if rows else np.empty((0, self.input_features), dtype=np.float16)

        def execute(key, values, origins, local_indices):
            logical_key = self.identities._reverse_experts[key.expert_id, key.revision]
            if logical_key != request.decision.key:
                raise ValueError("native dispatch changed the exact selected expert revision")
            indices = tuple(request.selection_indices[index] for index in local_indices)
            return self.callback(logical_key, request.parameters, values, indices)

        return self.native.execute_occurrences(plan, inputs, self.output_features, execute)
