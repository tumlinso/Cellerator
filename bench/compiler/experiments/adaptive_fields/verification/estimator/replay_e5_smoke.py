#!/usr/bin/env python3
"""Replay one exported E5 current-event trace with an independent CPU Torch oracle.

This script consumes the raw artifact emitted by gpu/e5_stage_adapter.py. It
does not import that adapter, Cellerator bindings, or CUDA. The equations below
are a small dense Torch composition of the declared transition, gate, relation,
and cross-entropy semantics; captured native arrays are used only as comparison
points and exact-workset input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class RelationWithDerivativeWorkset(torch.autograd.Function):
    """Primal relation forward with score VJP over the declared D workset."""

    @staticmethod
    def forward(ctx, inputs, edge_values, q, p_offsets, p_ordinals, p_gates,
                d_offsets, d_ordinals, d_gates, sources, destinations):
        cells, _, width = inputs.shape
        out = torch.zeros((cells, int(destinations.max()) + 1, width),
                          dtype=torch.float32)
        for cell in range(cells):
            for item in range(int(p_offsets[cell]), int(p_offsets[cell + 1])):
                edge = int(p_ordinals[item])
                gate_index = int(p_gates[item])
                out[cell, int(destinations[edge])] += (
                    inputs[cell, int(sources[edge])].float()
                    * edge_values[edge].half().float()
                    * q[gate_index].float())
        ctx.save_for_backward(inputs, edge_values, q, p_offsets, p_ordinals,
                              p_gates, d_offsets, d_ordinals, d_gates,
                              sources, destinations)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (inputs, edge_values, q, p_offsets, p_ordinals, p_gates,
         d_offsets, d_ordinals, d_gates, sources, destinations) = ctx.saved_tensors
        cells, source_count, width = inputs.shape
        grad_inputs = torch.zeros((cells, source_count, width), dtype=torch.float32)
        grad_values = torch.zeros_like(edge_values)
        grad_q = torch.zeros_like(q)
        # Input and shared-value VJPs come from actual primal forward items.
        for cell in range(cells):
            for item in range(int(p_offsets[cell]), int(p_offsets[cell + 1])):
                edge = int(p_ordinals[item])
                gate_index = int(p_gates[item])
                source = int(sources[edge])
                destination = int(destinations[edge])
                weight = edge_values[edge].half().float()
                upstream = grad_output[cell, destination]
                grad_inputs[cell, source] += upstream * weight * q[gate_index].float()
                grad_values[edge] += (upstream * inputs[cell, source].float()
                                      * q[gate_index].float()).sum()
        # The declared surrogate gate VJP uses all derivative candidates,
        # including derivative-only dormant items, and scatter-adds duplicates.
        for cell in range(cells):
            for item in range(int(d_offsets[cell]), int(d_offsets[cell + 1])):
                edge = int(d_ordinals[item])
                gate_index = int(d_gates[item])
                source = int(sources[edge])
                destination = int(destinations[edge])
                grad_q[gate_index] += (
                    grad_output[cell, destination] * inputs[cell, source].float()
                    * edge_values[edge].half().float()).sum()
        return (grad_inputs.to(inputs.dtype), grad_values, grad_q,
                None, None, None, None, None, None, None, None)


def tensor(z, *, dtype=None):
    return torch.as_tensor(np.array(z, copy=True), dtype=dtype)


def block(parameters, metadata, name):
    entry = metadata[name]
    begin = int(entry["offset"])
    count = int(entry["numel"])
    return parameters[begin:begin + count].reshape(entry["shape"])


def close_report(reference, observed):
    ref = reference.detach().float()
    got = observed.detach().float()
    delta = (ref - got).abs()
    scale = got.abs().clamp_min(1e-8)
    return {
        "shape": list(ref.shape),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "max_rel": float((delta / scale).max()) if delta.numel() else 0.0,
        "rms_abs": float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
    }


def run(path: Path, *, fixture: str = "current_event_all") -> dict:
    arrays = np.load(path, allow_pickle=False)
    initial_key = f"initial_parameters_{fixture}"
    prefix = f"direct_{fixture}"
    parameters = torch.nn.Parameter(tensor(arrays[initial_key], dtype=torch.float32))
    metadata = {}
    for name in ("entity_embedding", "rank_embedding", "scorer",
                 "transition_A", "transition_B", "transition_bias"):
        entry = arrays[f"block_{name}_offset_shape"]
        metadata[name] = {"offset": int(entry[0]), "numel": int(entry[1]),
                          "shape": [int(value) for value in entry[2:]]}
    embeddings = block(parameters, metadata, "entity_embedding")
    scorer = block(parameters, metadata, "scorer").reshape(-1)
    matrices_a = block(parameters, metadata, "transition_A")
    matrices_b = block(parameters, metadata, "transition_B")
    biases = block(parameters, metadata, "transition_bias")
    edge_ids = tensor(arrays[f"support_edge_ids_{fixture}"], dtype=torch.int64)
    sources = tensor(arrays[f"support_sources_{fixture}"], dtype=torch.int64)
    destinations = tensor(arrays[f"support_destinations_{fixture}"], dtype=torch.int64)
    events = tensor(arrays["event_ids"], dtype=torch.int64)
    targets = tensor(arrays["targets"], dtype=torch.int64)
    cells, steps = events.shape
    state = torch.zeros((cells, matrices_a.shape[0], matrices_a.shape[1]))
    step_reports = []
    final_output = None

    for step in range(steps):
        stem = f"{prefix}_step{step}"
        event = events[:, step]
        event_input = embeddings.index_select(0, event)
        affected_offsets = tensor(arrays[f"{stem}_latent_affected_offsets"], dtype=torch.int64)
        affected = tensor(arrays[f"{stem}_latent_affected_mechanisms"], dtype=torch.int64)
        # The Torch caller re-casts the whole recurrent state to FP16 before
        # each native call; untouched lanes therefore also pass through that
        # storage conversion before the native identity-copy path.
        next_state = state.half().float().clone()
        for cell in range(cells):
            for item in range(int(affected_offsets[cell]), int(affected_offsets[cell + 1])):
                mechanism = int(affected[item])
                z = state[cell, mechanism].half().float()
                u = event_input[cell].half().float()
                # The native operation contract is a shared transition
                # z' = tanh(Az + Bu + b): mechanism IDs select state lanes,
                # while A/B/b are shared coefficient matrices. The experiment
                # owner layout stores a leading mechanism dimension, but the
                # current kernel consumes the first shared matrix block.
                a = matrices_a[0].half().float()
                b = matrices_b[0].half().float()
                bias = biases[0].half().float()
                next_state[cell, mechanism] = torch.tanh(F.linear(z, a, bias)
                                                         + F.linear(u, b))
        native_state = tensor(arrays[f"{stem}_latent_state_output"], dtype=torch.float32)
        transition_error = close_report(next_state, native_state)

        current_input = torch.zeros((cells, embeddings.shape[0], embeddings.shape[1]))
        current_input[torch.arange(cells), event] = event_input
        context = (current_input + next_state.float().mean(dim=1, keepdim=True)).expand(
            -1, embeddings.shape[0], -1).contiguous()
        candidate_offsets = tensor(arrays[f"{stem}_score_candidate_cell_offsets"], dtype=torch.int64)
        candidate_ordinals = tensor(arrays[f"{stem}_score_candidate_edge_ordinals"], dtype=torch.int64)
        candidate_gate = tensor(arrays[f"{stem}_score_candidate_gate_value_indices"], dtype=torch.int64)
        scores = []
        for cell in range(cells):
            for item in range(int(candidate_offsets[cell]), int(candidate_offsets[cell + 1])):
                edge = int(candidate_ordinals[item])
                scores.append((context[cell, sources[edge]] * embeddings[destinations[edge]]
                               * scorer).sum() / (embeddings.shape[-1] ** 0.5))
        score_tensor = torch.stack(scores) if scores else parameters.new_empty((0,))
        probabilities = torch.sigmoid(score_tensor)
        admitted = tensor(arrays[f"{stem}_score_admitted"], dtype=torch.bool)
        native_probabilities = tensor(arrays[f"{stem}_score_probabilities"], dtype=torch.float32)
        score_error = close_report(probabilities, native_probabilities)
        q = probabilities + (probabilities * admitted.float() - probabilities).detach()

        edge_values = (embeddings[sources] * embeddings[destinations] * scorer).sum(dim=-1)
        edge_values = edge_values / (embeddings.shape[-1] ** 0.5)
        native_values = tensor(arrays[f"{prefix}_step{step}_edge_values"], dtype=torch.float32)
        value_error = close_report(edge_values, native_values)
        p_offsets = tensor(arrays[f"{stem}_selected_primal_cell_offsets"], dtype=torch.int64)
        p_ordinals = tensor(arrays[f"{stem}_selected_primal_edge_ordinals"], dtype=torch.int64)
        p_gates = tensor(arrays[f"{stem}_selected_primal_gate_value_indices"], dtype=torch.int64)
        d_offsets = tensor(arrays[f"{stem}_selected_derivative_cell_offsets"], dtype=torch.int64)
        d_ordinals = tensor(arrays[f"{stem}_selected_derivative_edge_ordinals"], dtype=torch.int64)
        d_gates = tensor(arrays[f"{stem}_selected_derivative_gate_value_indices"], dtype=torch.int64)
        relation_input = current_input.to(torch.float16)
        relation_output = RelationWithDerivativeWorkset.apply(
            relation_input, edge_values, q, p_offsets, p_ordinals, p_gates,
            d_offsets, d_ordinals, d_gates, sources, destinations)
        native_output = tensor(arrays[f"{stem}_relation_output"], dtype=torch.float32)
        output_error = close_report(relation_output, native_output)
        step_reports.append({"step": step, "transition": transition_error,
                             "gate_scores_probabilities": score_error,
                             "edge_values": value_error, "relation_output": output_error,
                             "candidate_count": int(candidate_ordinals.numel()),
                             "primal_count": int(p_ordinals.numel()),
                             "derivative_count": int(d_ordinals.numel())})
        state = next_state
        final_output = relation_output

    logits = (final_output.float() * embeddings.unsqueeze(0)).sum(dim=-1)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    native_final = tensor(arrays[f"{prefix}_final_output"], dtype=torch.float32)
    native_loss = tensor(arrays[f"{prefix}_loss"], dtype=torch.float64)
    native_gradient = tensor(arrays[f"{prefix}_owner_gradient"], dtype=torch.float32)
    gradient_by_block = {}
    for name, entry in metadata.items():
        begin, count = entry["offset"], entry["numel"]
        gradient_by_block[name] = close_report(
            parameters.grad[begin:begin + count], native_gradient[begin:begin + count])
    return {
        "artifact": str(path),
        "fixture": fixture,
        "cpu_only": True,
        "candidate_identity_count": int(edge_ids.numel()),
        "step_count": steps,
        "steps": step_reports,
        "final_output": close_report(final_output, native_final),
        "loss_abs": abs(loss.item() - native_loss.item()),
        "loss_torch_cpu": loss.item(),
        "loss_native": native_loss.item(),
        "owner_gradient_by_block": gradient_by_block,
        "owner_gradient": close_report(parameters.grad, native_gradient),
        "derivative_only_total": sum(
            item["derivative_count"] - item["primal_count"] for item in step_reports),
        "repeated_derivative_gate_index_steps": [
            step for step in range(steps)
            if len(set(tensor(arrays[f"{prefix}_step{step}_selected_derivative_gate_value_indices"],
                              dtype=torch.int64).tolist())) <
            tensor(arrays[f"{prefix}_step{step}_selected_derivative_gate_value_indices"],
                   dtype=torch.int64).numel()],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--fixture", default="current_event_all")
    args = parser.parse_args()
    print(json.dumps(run(args.artifact, fixture=args.fixture), indent=2))
