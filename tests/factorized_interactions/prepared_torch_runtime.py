"""CUDA runtime checks for the optional prepared ATen/Python adapter.

This is invoked only from the Scheduler-admitted correctness packet.  The
mathematical expectations come from the scalar NumPy oracle in reference.py;
the test does not use Torch autograd as its reference.
"""

from __future__ import annotations

from dataclasses import replace
import gc

import numpy as np
import torch

from reference import assert_fp32_contract, evaluate
from prepared_torch_records import prepared_factorized_records
from workloads import make_workload


def _make_bundle(problem, dy: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "x": torch.tensor(problem.x, device=device, dtype=torch.float32,
                          requires_grad=True),
        "z": torch.tensor(problem.z, device=device, dtype=torch.float32,
                          requires_grad=True),
        "alpha": torch.tensor(problem.alpha, device=device, dtype=torch.float32,
                              requires_grad=True),
        "bias": torch.tensor(problem.bias, device=device, dtype=torch.float32,
                             requires_grad=True),
        "gamma": torch.tensor(problem.gamma, device=device, dtype=torch.float32,
                              requires_grad=True),
        "dy": torch.tensor(dy, device=device, dtype=torch.float32),
    }


def _run_forward_backward(plan, problem, dy: np.ndarray, route: str, stream,
                          delay_cycles: int = 0):
    with torch.cuda.stream(stream):
        if delay_cycles:
            torch.cuda._sleep(delay_cycles)
        bundle = _make_bundle(problem, dy, torch.device("cuda:0"))
        output = _run_bundle_forward_backward(plan, bundle, route)
    return bundle, output


def _run_bundle_forward_backward(plan, bundle, route: str):
    output = plan(
        bundle["x"], bundle["z"], bundle["alpha"], bundle["bias"],
        bundle["gamma"], output_dtype=torch.float32, route=route)
    (output * bundle["dy"]).sum().backward()
    return output


def _compare(problem, dy: np.ndarray, bundle, output: torch.Tensor,
             route: str) -> dict:
    reference = evaluate(problem, dy=dy, output_dtype=np.float32)
    actual_output = output.detach().cpu().numpy()
    assert_fp32_contract(actual_output, reference.y_float64,
                         reference.sum_abs_y_contributions,
                         f"Python {route} output")

    gradient_names = {
        "x": "dx", "z": "dz", "alpha": "dalpha",
        "bias": "dbias", "gamma": "dgamma",
    }
    checks = {"y": _max_error(actual_output, reference.y_float64)}
    for input_name, reference_name in gradient_names.items():
        actual = bundle[input_name].grad.detach().cpu().numpy()
        expected = getattr(reference.vjp, reference_name)
        scale = getattr(reference.vjp_scale, reference_name)
        assert_fp32_contract(actual, expected, scale,
                             f"Python {route} gradient {input_name}")
        checks[reference_name] = _max_error(actual, expected)
    return checks


def _max_error(actual: np.ndarray, expected: np.ndarray) -> dict:
    error = np.abs(np.asarray(actual, dtype=np.float64)
                   - np.asarray(expected, dtype=np.float64))
    if not error.size:
        return {"max_abs_error": 0.0}
    return {"max_abs_error": float(error.max())}


def _clear_gradients(bundle) -> None:
    for name in ("x", "z", "alpha", "bias", "gamma"):
        bundle[name].grad = None


def run_prepared_torch_runtime_checks() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("prepared ATen/Python checks require an admitted CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "the exclusive Scheduler packet must expose exactly its selected GPU")
    if not hasattr(torch.cuda, "_sleep"):
        raise RuntimeError("this CUDA Torch build lacks the stream-ordering delay primitive")

    from cellerator.torch import PreparedFactorizedInteraction

    device = torch.device("cuda:0")
    problem, dy = make_workload(
        "correctness_ordered_tied_zero", np.float32, np.float32)
    plan = PreparedFactorizedInteraction(
        prepared_factorized_records(problem),
        inputs=problem.x.shape[0], contexts=problem.z.shape[0],
        outputs=problem.target_count, alpha_rows=problem.alpha.shape[0],
        biases=problem.bias.shape[0], gammas=problem.gamma.shape[0],
        pattern_id=0xF17E, structure_id=0x20261102, structure_epoch=7)

    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)

    # The first call creates/uploads the device owner on A, after deliberate
    # queued work.  The second call reuses the same plan on B while A may still
    # be uploading.  Correct execution depends on the adapter's event ordering.
    bundle_a, output_a = _run_forward_backward(
        plan, problem, dy, "packet", stream_a, delay_cycles=10_000_000)
    bundle_b, output_b = _run_forward_backward(
        plan, problem, dy, "direct", stream_b)
    stream_a.synchronize()
    stream_b.synchronize()

    errors = {
        "stream_a_packet": _compare(problem, dy, bundle_a, output_a, "packet"),
        "stream_b_direct": _compare(problem, dy, bundle_b, output_b, "direct"),
    }

    # Mutate the same live parameter tensor only after both earlier VJPs have
    # completed, then call through the same prepared owner on another stream.
    _clear_gradients(bundle_a)
    updated_gamma = problem.gamma.copy()
    updated_gamma += np.float32(0.125)
    updated_problem = replace(problem, gamma=updated_gamma)
    with torch.cuda.stream(stream_b), torch.no_grad():
        bundle_a["gamma"].add_(0.125)
    with torch.cuda.stream(stream_b):
        output_updated = _run_bundle_forward_backward(plan, bundle_a, "packet")
    stream_b.synchronize()
    errors["updated_parameter_packet"] = _compare(
        updated_problem, dy, bundle_a, output_updated, "updated packet")
    if torch.equal(output_a, output_updated):
        raise AssertionError("updated gamma contents did not change the output")

    # PyTorch must reject mutation of a primal saved for backward before the
    # native VJP reads it.
    with torch.cuda.stream(stream_b):
        bundle_mutation = _make_bundle(problem, dy, device)
        output_mutation = plan(
            bundle_mutation["x"], bundle_mutation["z"],
            bundle_mutation["alpha"], bundle_mutation["bias"],
            bundle_mutation["gamma"], output_dtype=torch.float32, route="packet")
    stream_b.synchronize()
    with torch.no_grad():
        bundle_mutation["x"].add_(0.01)
    try:
        with torch.cuda.stream(stream_b):
            output_mutation.sum().backward()
    except RuntimeError as error:
        if "modified by an inplace operation" not in str(error):
            raise
        mutation_guard = "passed"
    else:
        raise AssertionError("autograd accepted mutation of a saved source tensor")

    # The adapter explicitly supports first-order differentiation only.
    with torch.cuda.stream(stream_b):
        bundle_higher_order = _make_bundle(problem, dy, device)
        output_higher_order = plan(
            bundle_higher_order["x"], bundle_higher_order["z"],
            bundle_higher_order["alpha"], bundle_higher_order["bias"],
            bundle_higher_order["gamma"], output_dtype=torch.float32,
            route="packet")
    stream_b.synchronize()
    try:
        with torch.cuda.stream(stream_b):
            torch.autograd.grad(output_higher_order.sum(),
                                bundle_higher_order["x"], create_graph=True)
    except RuntimeError as error:
        if "first-order autograd only" not in str(error):
            raise
        higher_order_guard = "passed"
    else:
        raise AssertionError("autograd accepted a create_graph=True VJP")

    stream_a.synchronize()
    stream_b.synchronize()
    del output_a, output_b, output_updated, output_mutation, output_higher_order
    gc.collect()
    return {
        "schema": "cellerator.factorized_interactions.prepared_torch_runtime.v1",
        "torch_version": torch.__version__,
        "device_count": torch.cuda.device_count(),
        "routes": ["packet", "direct"],
        "stream_owner_reuse": "passed",
        "parameter_update_reuse": "passed",
        "saved_tensor_mutation_guard": mutation_guard,
        "first_order_only_guard": higher_order_guard,
        "gradient_groups_checked": ["x", "z", "alpha", "bias", "gamma"],
        "contract_max_abs_errors": errors,
        "timed": False,
    }
