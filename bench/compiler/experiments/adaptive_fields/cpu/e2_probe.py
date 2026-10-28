"""Small operation-registration prototype. Uses the installed CPU Torch ATen path."""
from __future__ import annotations

import json
import ctypes
import subprocess
from pathlib import Path

import torch


def aten_add_same_shape_f32(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if (lhs.dtype != torch.float32 or rhs.dtype != torch.float32 or
            lhs.shape != rhs.shape or not lhs.is_contiguous() or
            not rhs.is_contiguous() or lhs.device.type != "cpu"):
        raise TypeError("supported overload is contiguous same-shape CPU FP32")
    return torch.add(lhs, rhs, alpha=1)


def main() -> None:
    x = torch.tensor([1.0, -2.0, 4.0], dtype=torch.float32)
    b = torch.tensor([0.5, 3.0, -1.0], dtype=torch.float32)
    # Exercise the selected ATen overload (and its autograd registration).
    x.requires_grad_(True)
    b.requires_grad_(True)
    y = aten_add_same_shape_f32(x, b)
    y.square().sum().backward()
    assert torch.equal(y.detach(), torch.tensor([1.5, 1.0, 3.0]))
    assert torch.equal(x.grad, 2 * y.detach()) and torch.equal(b.grad, x.grad)
    rejected_broadcast = False
    try:
        aten_add_same_shape_f32(x.detach(), torch.ones(1, dtype=torch.float32))
    except TypeError:
        rejected_broadcast = True
    assert rejected_broadcast

    # These are local descriptors, not a production operation registry.
    # The ordinary C++ helper's published mathematical body is represented as
    # the same-shape componentwise decomposition: add(x,b). The portable body
    # and external overload therefore admit relation-epilogue fusion.
    helper_src = Path(__file__).with_name("e2_cpp_helper.cc")
    helper_so = Path(__file__).parent / "artifacts" / "libe2_cpp_helper.so"
    helper_so.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["g++", "-std=c++20", "-O2", "-shared", "-fPIC",
                    str(helper_src), "-o", str(helper_so)], check=True)
    helper_lib = ctypes.CDLL(str(helper_so))
    helper = helper_lib.probe_add_f32
    helper.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float), ctypes.c_uint64]
    helper.restype = None
    lhs_values, rhs_values = [1.25, -3.0], [0.5, 2.0]
    lhs = (ctypes.c_float * 2)(*lhs_values)
    rhs = (ctypes.c_float * 2)(*rhs_values)
    helper_out = (ctypes.c_float * 2)()
    helper(lhs, rhs, helper_out, 2)
    assert list(helper_out) == [1.75, -1.0]

    descriptors = {
        "cpp_helper_add": {
            "identity": "probe.math.add", "version": 1,
            "operands": ["tensor<f32>[N]", "tensor<f32>[N]"],
            "result": "tensor<f32>[N]", "effects": {"reads": ["lhs", "rhs"], "writes": []},
            "numerics": "IEEE FP32 componentwise add, preserving operation order",
            "body": "lhs[i] + rhs[i]", "actual_callable": str(helper_so),
            "lowering": "local relation epilogue callback",
        },
        "aten_add": {
            "identity": "aten::add.Tensor", "version": 1,
            "overload": "same-shape contiguous CPU FP32, alpha=1, no broadcast",
            "operands": ["tensor<f32>[N]", "tensor<f32>[N]"],
            "result": "tensor<f32>[N]", "effects": {"reads": ["lhs", "rhs"], "writes": []},
            "numerics": "selected ATen overload, same-shaped FP32 componentwise add",
            "body": "lhs[i] + rhs[i]",
            "lowering": "local relation epilogue callback",
        },
    }
    # The rewrite is legal only with the semantic body and disjoint effects.
    before = ["relation_apply(A_vals,x)->y", "aten_add(y,b)->u", "segment_softmax(u,s)->p"]
    after = ["relation_apply_epilogue(A_vals,x,b)->u", "segment_softmax(u,s)->p"]
    effects_only = {"identity": "native.bump", "effects": {"reads": ["x"], "writes": ["b"]}}
    opaque = {"identity": "opaque.native", "effects": "unknown", "barrier": True}
    ops = {"relation_apply": {"body": "csr_dot", "effects": {"reads": ["A_vals", "x"], "writes": ["y"]}},
           "add": descriptors["aten_add"],
           "segment_softmax": {"body": "stable_segment_softmax", "effects": {"reads": ["u"], "writes": ["p"]}}}

    def select_fusion(sequence, registry):
        # Search a connected relation/add/softmax chain, then prove that moving
        # the add into the relation epilogue crosses no write or opaque effect.
        for i, relation in enumerate(sequence):
            if relation["kind"] != "relation_apply":
                continue
            j = next((k for k in range(i+1, len(sequence)) if sequence[k]["kind"] == "add"), None)
            if j is None:
                continue
            k = next((k for k in range(j+1, len(sequence)) if sequence[k]["kind"] == "segment_softmax"), None)
            if k is None:
                continue
            triple = [relation, sequence[j], sequence[k]]
            add = registry.get("add")
            if not add or not add.get("body") or add.get("barrier"):
                return sequence, False
            if triple[1]["lhs"] != triple[0]["out"] or triple[2]["input"] != triple[1]["out"]:
                continue
            # Map formal descriptor operands to the graph's concrete symbols.
            add_reads = {triple[1]["lhs"], triple[1]["rhs"]}
            for crossed in sequence[j+1:k]:
                contract = registry.get(crossed["kind"])
                if not contract or contract.get("barrier") or contract.get("effects") == "unknown":
                    return sequence, False
                if add_reads.intersection(contract.get("effects", {}).get("writes", [])):
                    return sequence, False
            fused = {"kind": "relation_apply_epilogue", "inputs": [triple[0]["relation"],
                      triple[0]["state"], triple[1]["rhs"]], "out": triple[1]["out"],
                      "numeric_order": "relation sum then fp32 bias add"}
            return sequence[:i] + [fused] + sequence[i+1:j] + sequence[j+1:k] + [triple[2]] + sequence[k+1:], True
        return sequence, False

    pipeline = [{"kind": "relation_apply", "relation": "A_vals", "state": "x", "out": "y"},
                {"kind": "add", "lhs": "y", "rhs": "b", "out": "u"},
                {"kind": "segment_softmax", "input": "u", "out": "p"}]
    selected, did_fuse = select_fusion(pipeline, ops)
    assert did_fuse and selected[0]["kind"] == "relation_apply_epilogue"
    no_body = dict(descriptors["aten_add"]);no_body.pop("body")
    assert not select_fusion(pipeline, {**ops, "add": no_body})[1]

    # A declared write to b forbids moving or folding the add across bump.
    # This actual invocation distinguishes the original sequence from a bad
    # reordered execution.
    mutable_bias = torch.tensor([0.25], dtype=torch.float32)
    old_bias = mutable_bias.clone()
    def native_bump(): mutable_bias.add_(10.0)
    native_bump()
    ordered = torch.tensor([2.0]) + mutable_bias
    wrong_reordered = (torch.tensor([2.0]) + old_bias)  # add moved before write
    assert ordered.item() == 12.25 and wrong_reordered.item() == 2.25
    effects_sequence = [pipeline[0], pipeline[1],
                        {"kind": "native.bump", "reads": ["x"], "writes": ["b"]}, pipeline[2]]
    effects_registered = {**ops, "native.bump": effects_only}
    effects_plan = effects_sequence
    assert effects_registered["native.bump"]["effects"]["writes"] == ["b"]
    effects_selected, effects_fused = select_fusion(effects_sequence, effects_registered)
    assert not effects_fused and effects_selected == effects_sequence
    opaque_sequence = [pipeline[0], pipeline[1], {"kind": "opaque.native"}, pipeline[2]]
    opaque_plan = opaque_sequence
    opaque_selected, opaque_fused = select_fusion(opaque_sequence, {**ops, "opaque.native": opaque})
    assert not opaque_fused and opaque_selected == opaque_sequence

    # Execute both the unfused Torch composition and the local fused epilogue
    # callback on CSR data, including a negative weight and a singleton row.
    rowptr = [0, 2, 2, 3]
    col = [0, 2, 1]
    weight = torch.tensor([0.5, -0.25, 1.25], dtype=torch.float32)
    state = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
    bias = torch.tensor([0.1, -0.4, 0.8], dtype=torch.float32)
    explicit_relation = torch.zeros(3)
    for row in range(3):
        for edge in range(rowptr[row], rowptr[row + 1]):
            explicit_relation[row] += weight[edge] * state[col[edge]]
    explicit = torch.cat((torch.softmax(torch.add(explicit_relation[:2], bias[:2]), dim=0),
                          torch.softmax(torch.add(explicit_relation[2:], bias[2:]), dim=0)))

    def fused_relation_epilogue() -> torch.Tensor:
        result = torch.zeros(3)
        for row in range(3):
            acc = torch.tensor(0.0)
            for edge in range(rowptr[row], rowptr[row + 1]):
                acc = acc + weight[edge] * state[col[edge]]
            result[row] = acc + bias[row]
        return torch.cat((torch.softmax(result[:2], dim=0), torch.softmax(result[2:], dim=0)))

    fused = fused_relation_epilogue()
    torch.testing.assert_close(fused, explicit, rtol=1e-6, atol=1e-7)

    report = {
        "decision": "supported",
        "actual_torch": torch.__version__,
        "actual_operation": "torch.add(..., alpha=1) on CPU contiguous same-shaped FP32 tensors",
        "forward_and_autograd": "pass",
        "ordinary_cpp_helper": {"compiled_and_invoked": True, "output": list(helper_out)},
        "unsupported_broadcast": "rejected by local adapter",
        "descriptors": descriptors,
        "rewrite": {"before": before, "after": after, "local_callback": True},
        "rewrite_execution": {"explicit": explicit.tolist(), "fused": fused.tolist(),
                              "max_abs_error": float((explicit-fused).abs().max())},
        "effects_only_native": {"contract": effects_only, "ordering": effects_plan,
                                 "fusion_across_write": False,
                                 "actual_order_sensitive_values": {"declared_order": ordered.item(),
                                   "incorrect_moved_add": wrong_reordered.item()}},
        "opaque_native": {"contract": opaque, "ordering": opaque_plan,
                          "barrier_preserved": True},
        "limitations": [
            "registration, typed checks, rewrite selection, and epilogue callback are experiment-local",
            "ATen was invoked through Python CPU dispatch; no Torch-to-Cellerator production adapter was exercised",
            "the C++ helper body was manually declared in its experiment-local descriptor, not inferred from source",
            "no general Torch module capture or arbitrary overload support",
        ],
    }
    out = Path(__file__).parent / "results" / "e2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("E2 selected ATen add forward/autograd, descriptor checks, rewrite legality and barrier records PASS")


if __name__ == "__main__":
    main()
