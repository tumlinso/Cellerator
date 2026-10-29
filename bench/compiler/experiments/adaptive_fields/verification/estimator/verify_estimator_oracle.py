#!/usr/bin/env python3
"""CPU-only independent reference checks for the E5 estimator contract.

This does not invoke Cellerator native/CUDA code. It fixes the declared
straight-through gate estimator and recurrent transition equations so a later
native/field-plan result can be compared against an independent oracle.
"""
from __future__ import annotations

import torch


def verify_gate_vjp() -> dict[str, object]:
    # Four logical candidates: two admitted, two dormant. The derivative
    # workset has positives plus one sampled dormant candidate, repeated for
    # two cells. The final dormant candidate is intentionally omitted.
    scores = torch.tensor([-1.2, 0.4, 1.3, -0.7], dtype=torch.float64,
                          requires_grad=True)
    temperature = 0.7
    threshold = 0.5
    p = torch.sigmoid(scores / temperature)
    admitted = p >= threshold
    q = p + (p * admitted.to(p.dtype) - p).detach()
    assert admitted.tolist() == [False, True, True, False]
    assert q.detach().tolist()[0] == 0.0

    # Each derivative item has its own dL/dq contribution. Duplicate gate
    # indices model work items in different cells and must scatter-add.
    gate_indices = torch.tensor([1, 2, 0, 1, 2, 0], dtype=torch.int64)
    upstream_items = torch.tensor([0.3, -1.1, 2.0, 0.8, 0.2, -0.5],
                                  dtype=torch.float64)
    (q.index_select(0, gate_indices) * upstream_items).sum().backward()

    upstream_by_candidate = torch.zeros_like(scores)
    upstream_by_candidate.scatter_add_(0, gate_indices, upstream_items)
    explicit = upstream_by_candidate * p.detach() * (1.0 - p.detach()) / temperature
    torch.testing.assert_close(scores.grad, explicit, rtol=1e-14, atol=1e-14)
    assert scores.grad[0].item() != 0.0  # closed but sampled dormant candidate
    assert scores.grad[3].item() == 0.0  # dormant and not sampled
    assert upstream_by_candidate[1].item() == 1.1  # repeated index accumulated
    return {
        "admitted": admitted.tolist(),
        "derivative_gate_indices": gate_indices.tolist(),
        "upstream_by_candidate": upstream_by_candidate.tolist(),
        "score_vjp": scores.grad.tolist(),
        "omitted_dormant_vjp": scores.grad[3].item(),
    }


def verify_recurrent_vjp() -> dict[str, object]:
    # Two small tanh transitions, matching the mathematical transition family
    # used by the native latent operator, but intentionally independent of its
    # implementation and precision path.
    torch.manual_seed(712)
    state0 = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    events = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    matrices_a = (torch.randn(2, 3, 3, dtype=torch.float64) * 0.2).requires_grad_()
    matrices_b = (torch.randn(2, 3, 2, dtype=torch.float64) * 0.2).requires_grad_()
    biases = (torch.randn(2, 3, dtype=torch.float64) * 0.1).requires_grad_()
    state = state0
    for step in range(2):
        state = torch.tanh(state @ matrices_a[step].T
                           + events[step] @ matrices_b[step].T
                           + biases[step])
    upstream = torch.tensor([[0.6, -1.0, 0.25]], dtype=torch.float64)
    loss = (state * upstream).sum()
    loss.backward()

    # Independently apply the reverse recurrence: dz through tanh, then
    # matrix/event VJPs. This is the declared recurrent VJP, not a finite
    # difference through a hard gate.
    with torch.no_grad():
        states = [state0.detach()]
        pres = []
        for step in range(2):
            pre = (states[-1] @ matrices_a[step].detach().T
                   + events[step].detach() @ matrices_b[step].detach().T
                   + biases[step].detach())
            pres.append(pre)
            states.append(torch.tanh(pre))
        grad_state = upstream.clone()
        grad_a = torch.zeros_like(matrices_a)
        grad_b = torch.zeros_like(matrices_b)
        grad_bias = torch.zeros_like(biases)
        grad_events = torch.zeros_like(events)
        for step in (1, 0):
            grad_pre = grad_state * (1.0 - states[step + 1].square())
            grad_a[step] = grad_pre.T @ states[step]
            grad_b[step] = grad_pre.T @ events[step].detach().unsqueeze(0)
            grad_bias[step] = grad_pre.sum(dim=0)
            grad_events[step] = grad_pre @ matrices_b[step].detach()
            grad_state = grad_pre @ matrices_a[step].detach()

    torch.testing.assert_close(state0.grad, grad_state, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(events.grad, grad_events, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(matrices_a.grad, grad_a, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(matrices_b.grad, grad_b, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(biases.grad, grad_bias, rtol=1e-13, atol=1e-13)
    return {
        "steps": 2,
        "loss": loss.item(),
        "state0_vjp_l2": state0.grad.norm().item(),
        "events_vjp_l2": events.grad.norm().item(),
        "transition_a_vjp_l2": matrices_a.grad.norm().item(),
        "transition_b_vjp_l2": matrices_b.grad.norm().item(),
        "bias_vjp_l2": biases.grad.norm().item(),
    }


if __name__ == "__main__":
    print("gate_vjp", verify_gate_vjp())
    print("recurrent_vjp", verify_recurrent_vjp())
