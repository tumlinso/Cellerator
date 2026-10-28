#!/usr/bin/env python3
"""Matched GPU follow-up: current-event vs observed window-history relation input.

This driver imports the frozen torch-v12-validator package and its fixture,
workset, and train-step helpers. It intentionally never reads validator-only
fixture fields or masks the 24-destination readout.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from collections import Counter
from pathlib import Path

import torch

ROOT = Path("/home/tumlinson/Cellerator")
DEFAULT_SNAPSHOT = ROOT / "build-sparse-support/evidence/torch-v12-validator"
SNAPSHOT_ENV = os.environ.get("CELLERATOR_SPARSE_SUPPORT_SNAPSHOT")
SNAPSHOT = Path(SNAPSHOT_ENV or str(DEFAULT_SNAPSHOT)).resolve()
PACKAGE = SNAPSHOT / "package"
EXAMPLE_DIR = SNAPSHOT / "examples/sparse_support"
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(EXAMPLE_DIR))

import train_synthetic as base  # noqa: E402
from cellerator.torch.sparse_support import (  # noqa: E402
    KaryoplastEventSlice,
    compile_sparse_support_ce_geo,
    make_next_event_fixture,
    prepare_sparse_slice,
)
from cellerator.torch import _torch  # noqa: E402


SEED = 41027
EPOCHS = 50
SOFT_EPOCHS = 10
TRAIN_CELLS = 32
STEPS = 12
HELDOUT_CELLS = 512
ENTITIES = 24
THRESHOLD = 0.5


def public_fixture(seed: int, cells: int):
    fixture = make_next_event_fixture(seed=seed, cells=cells,
                                      steps=STEPS, entities=ENTITIES)
    # Ensure even the experiment driver passes no validator-only fields onward.
    return {key: value for key, value in fixture.items()
            if not key.startswith("validator")}


def history_provider(module, index, fixture, device):
    provider = base.event_provider(module, index, fixture, device)

    def wrapped(position, history, timestamps, windows):
        work = provider(position, history, timestamps, windows)
        sources = work[4] if len(work) == 5 else None
        if sources is None:
            raise RuntimeError("v12 event provider did not supply observed window sources")
        module._experiment_history_sources = sources
        return work

    return wrapped


def install_history_relation(module):
    """Pass the same count/time-window occurrence aggregate used by gate context."""
    original_relation = module.relation

    def relation(self, inputs, edge_values, gates, primal, derivative):
        sources = getattr(self, "_experiment_history_sources", None)
        if sources is None or len(sources) != inputs.shape[0]:
            raise RuntimeError("windowed source rows are missing for a relation step")
        embedding = self.block("entity_embedding")
        # The relation receives FP16, while entity embeddings and the gate's
        # existing history aggregate use FP32 accumulation.
        history = embedding.new_zeros(inputs.shape)
        for cell, row in enumerate(sources):
            if row:
                indices = torch.as_tensor(row, device=embedding.device,
                                          dtype=torch.int64)
                history[cell].index_add_(
                    0, indices, embedding.index_select(0, indices))
        return original_relation(history.to(torch.float16), edge_values, gates,
                                 primal, derivative)

    module.relation = types.MethodType(relation, module)


def train_events(module, index, fixture, device, optimizer, *, prune, history_readout):
    base.reset_event_stream_state(module)
    provider = base.event_provider(module, index, fixture, device)
    if history_readout:
        provider = history_provider(module, index, fixture, device)
    output = module(fixture["events"], provider,
                    timestamps=fixture["timestamps"],
                    windows=fixture["count_windows"], prune=prune,
                    threshold=THRESHOLD,
                    derivative_seed=int(fixture["seed"]))
    loss = module.training_loss(output, fixture["targets"].to(device))
    loss.backward()
    base.assert_transition_gradients(module)
    base.guarded_step(module, optimizer)
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach()), module.gate_admission_rate()


@torch.no_grad()
def evaluate_events(module, index, fixture, device, *, prune, history_readout):
    base.reset_event_stream_state(module)
    provider = base.event_provider(module, index, fixture, device)
    if history_readout:
        provider = history_provider(module, index, fixture, device)
    encoded = module(fixture["events"], provider,
                     timestamps=fixture["timestamps"],
                     windows=fixture["count_windows"], prune=prune,
                     threshold=THRESHOLD,
                     derivative_seed=int(fixture["seed"]))
    logits = module.next_event_logits(encoded)
    targets = fixture["targets"].to(device)
    predictions = logits.argmax(dim=-1)
    return {
        "loss_all_24": float(torch.nn.functional.cross_entropy(logits, targets)),
        "accuracy_all_24": float((predictions == targets).float().mean()),
        "prediction_counts_all_24": [
            int((predictions == entity).sum()) for entity in range(ENTITIES)],
        "confusion_target_rows_predicted_0_to_23": [
            [int(((targets == target) & (predictions == entity)).sum())
             for entity in range(ENTITIES)]
            for target in range(16, 20)],
        "target_counts": [int((targets == target).sum())
                          for target in range(16, 20)],
        "mean_true_target_rank_all_24": float(
            (1 + (logits > logits.gather(1, targets[:, None])).sum(dim=1)).float().mean()),
        "gate_admission": module.gate_admission_rate(),
    }


def support_seed(seed):
    support = base.pair_support(seed)
    shuffled = base.shuffled_support(support, seed + 17)
    return support, shuffled


def prepare_model(support, discriminator, seed, device, *, history_readout):
    axis_words = (101, 102, 201, 202, 301, 302, 401, 402 + discriminator)
    compiled = compile_sparse_support_ce_geo(
        support, source_count=ENTITIES, destination_count=ENTITIES,
        projection_id=discriminator)
    module = prepare_sparse_slice(
        KaryoplastEventSlice, support, entities=ENTITIES, mechanisms=4,
        rank_count=8, device=device, parameter_axis_words=axis_words,
        seed=seed, ce_geo_image=compiled["image"],
        source_group_members=compiled["source_group_members"],
        destination_group_members=compiled["destination_group_members"],
        geometry_edge_ids=compiled["geometry_edge_ids_in_support_order"],
        projection_id=discriminator, max_cells=HELDOUT_CELLS,
        max_primal_items=HELDOUT_CELLS * max(1, len(support.edge_ids)))
    if history_readout:
        install_history_relation(module)
    return module


def main():
    if not SNAPSHOT_ENV:
        raise RuntimeError("GPU trial requires an explicit immutable snapshot path")
    identity_path = SNAPSHOT / "identity.json"
    identity_bytes = identity_path.read_bytes()
    identity_sha256 = hashlib.sha256(identity_bytes).hexdigest()
    identity = json.loads(identity_bytes)
    expected_identity = os.environ.get("CELLERATOR_SPARSE_SUPPORT_EXPECTED_IDENTITY_SHA256")
    if not expected_identity:
        raise RuntimeError("GPU trial requires the qualified identity digest")
    if identity_sha256 != expected_identity:
        raise RuntimeError("frozen snapshot identity does not match the requested digest")
    frozen_files = identity.get("files", {})
    files_to_check = {
        "examples/sparse_support/train_synthetic.py": EXAMPLE_DIR / "train_synthetic.py",
        "package/cellerator/torch/sparse_support.py": PACKAGE / "cellerator/torch/sparse_support.py",
        "package/cellerator/torch/_torch.cpython-313-x86_64-linux-gnu.so":
            PACKAGE / "cellerator/torch/_torch.cpython-313-x86_64-linux-gnu.so",
    }
    for relative, path in files_to_check.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != frozen_files.get(relative):
            raise RuntimeError(f"selected snapshot file hash mismatch: {relative}")
    torch.manual_seed(SEED)
    device = torch.device("cuda", 0)
    main_support, shuffled = support_seed(SEED)
    controls = {
        "learned_pruned": (main_support, 2, 3, True),
        "soft_only": (main_support, 6, 7, False),
        "shuffled_pruned": (shuffled, 4, 5, True),
    }
    models = {}
    optimizers = {}
    indexes = {}
    pair_initial_equal = {}
    for name, (support, current_discriminator, history_discriminator,
               prune_after_warmup) in controls.items():
        current = prepare_model(support, current_discriminator, SEED + 1,
                                0, history_readout=False)
        history = prepare_model(support, history_discriminator, SEED + 1,
                                0, history_readout=True)
        equal = torch.equal(current.parameters_flat.detach(),
                            history.parameters_flat.detach())
        if not equal:
            raise RuntimeError(f"initial parameters differ in paired control {name}")
        pair_initial_equal[name] = equal
        for tag, module in (("current", current), ("history", history)):
            key = f"{name}/{tag}"
            models[key] = module
            indexes[key] = _torch.NativeReverseIndex(module._support_index)
            optimizers[key] = torch.optim.Adam([module.parameters_flat], lr=0.02)

    epoch_rows = []
    for epoch in range(EPOCHS):
        fixture = public_fixture(SEED + epoch, TRAIN_CELLS)
        prune = epoch >= SOFT_EPOCHS
        for name, (_, _, _, prune_after_warmup) in controls.items():
            pair = {}
            for readout in ("current", "history"):
                key = f"{name}/{readout}"
                loss, admission = train_events(
                    models[key], indexes[key], fixture, device, optimizers[key],
                    prune=(prune if prune_after_warmup else False),
                    history_readout=(readout == "history"))
                pair[readout] = {"loss": loss, "gate_admission": admission}
            epoch_rows.append({"epoch": epoch, "control": name,
                               "pruned": bool(prune_after_warmup and prune),
                               **pair})

    heldout = public_fixture(SEED + 200_000, HELDOUT_CELLS)
    target_counts = Counter(int(x) for x in heldout["targets"].tolist())
    constant_prior = {
        "majority_target_id": min(target_counts, key=lambda x: (-target_counts[x], x)),
        "majority_accuracy": max(target_counts.values()) / HELDOUT_CELLS,
        "class_counts_16_to_19": [target_counts.get(i, 0) for i in range(16, 20)],
        "target_count": HELDOUT_CELLS,
    }
    heldout_results = {}
    for key, module in models.items():
        readout = key.split("/")[-1]
        heldout_results[key] = {}
        for label, prune in (("soft", False), ("pruned", True)):
            heldout_results[key][label] = evaluate_events(
                module, indexes[key], heldout, device, prune=prune,
                history_readout=(readout == "history"))

    training_summary = {}
    for name in controls:
        training_summary[name] = {}
        for readout in ("current", "history"):
            rows = [row for row in epoch_rows if row["control"] == name]
            losses = [row[readout]["loss"] for row in rows]
            admissions = [row[readout]["gate_admission"] for row in rows]
            training_summary[name][readout] = {
                "epoch_0_loss": losses[0],
                "epoch_9_loss": losses[9],
                "epoch_10_loss": losses[10],
                "epoch_49_loss": losses[-1],
                "minimum_loss": min(losses),
                "epoch_49_gate_admission": admissions[-1],
            }
    report = {
        "experiment": "event_history_relation_readout_v1",
        "package_snapshot": str(SNAPSHOT),
        "package_identity_sha256": identity_sha256,
        "extension_sha256": hashlib.sha256(
            (PACKAGE / "cellerator/torch/_torch.cpython-313-x86_64-linux-gnu.so").read_bytes()
        ).hexdigest(),
        "training_driver_sha256": hashlib.sha256(
            (EXAMPLE_DIR / "train_synthetic.py").read_bytes()).hexdigest(),
        "module_source_sha256": hashlib.sha256(
            (PACKAGE / "cellerator/torch/sparse_support.py").read_bytes()).hexdigest(),
        "seed": SEED,
        "epochs": EPOCHS,
        "soft_epochs": SOFT_EPOCHS,
        "train_cells": TRAIN_CELLS,
        "heldout_cells": HELDOUT_CELLS,
        "steps": STEPS,
        "support_edges": len(main_support.edge_ids),
        "support_growth": 0,
        "gate_threshold": THRESHOLD,
        "readout_logits": ENTITIES,
        "learning_rate": 0.02,
        "heldout_seed": SEED + 200_000,
        "target_constant_prior": constant_prior,
        "pair_initial_parameters_equal": pair_initial_equal,
        "validator_only_fields_passed": [],
        "training_summary": training_summary,
        "heldout_all24_results": heldout_results,
    }
    print("EVENT_HISTORY_TRIAL_JSON=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
