"""CPU checks for the generic Torch/native owner lifecycle bridge."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn

from cellerator.torch.mechanism import (  # noqa: E402
    attach_native_owner, guarded_step, load_checkpoint, save_checkpoint,
)


class _FakeHandle:
    device = 0

    def __init__(self, values):
        self.values = values.clone()
        self.generation = 0
        self.poisoned = False
        self.parameter = None


class _FakeSparseAPI:
    @staticmethod
    def sparse_validate_parameters(handle, parameter):
        assert parameter.numel() == handle.values.numel()
        handle.parameter = parameter

    @staticmethod
    def sparse_preflight_write(handle):
        assert not handle.poisoned

    @staticmethod
    def sparse_begin_update(handle):
        assert not handle.poisoned

    @staticmethod
    def sparse_publish_update(handle):
        handle.values.copy_(handle.parameter.detach())
        handle.generation += 1

    @staticmethod
    def sparse_poison(handle):
        handle.poisoned = True

    @staticmethod
    def sparse_snapshot(handle):
        return handle.values.clone(), {"blocks": [("shared", 0, 2)]}

    @staticmethod
    def sparse_restore(handle, values, metadata):
        assert isinstance(values, torch.Tensor)
        assert values.device.type == "cpu"
        assert values.dtype == torch.float32 and values.is_contiguous()
        assert metadata == {"blocks": [("shared", 0, 2)]}
        handle.values.copy_(torch.as_tensor(values))
        if handle.parameter is not None:
            handle.parameter.data.copy_(handle.values)


class _SparseOwnerModule(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.shared = nn.Parameter(initial.clone())
        self.handle = _FakeHandle(initial)
        attach_native_owner(self, self.handle, self.shared, _FakeSparseAPI)
        self.window = 0

    def _sparse_checkpoint_state(self):
        return {"window": self.window}

    def _prepare_sparse_checkpoint_state(self, state):
        if set(state) != {"window"}:
            raise ValueError("bad state")
        return state

    def _restore_sparse_checkpoint_state(self, state):
        self.window = state["window"]


class NativeOwnerProtocolTest(unittest.TestCase):
    def test_guarded_step_preserves_one_sparse_owner_leaf_and_publishes_once(self):
        module = _SparseOwnerModule(torch.tensor([1.0, -2.0]))
        parameter = module.shared
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        parameter.square().sum().backward()

        self.assertTrue(guarded_step(module, optimizer))
        self.assertIs(module.shared, parameter)
        self.assertIs(module._cellerator_native_owner.parameter, parameter)
        self.assertEqual(module.handle.generation, 1)
        self.assertTrue(torch.allclose(module.handle.values, parameter.detach()))

    def test_sparse_checkpoint_restores_owner_optimizer_and_window(self):
        source = _SparseOwnerModule(torch.tensor([1.0, -2.0]))
        source_optimizer = torch.optim.Adam([source.shared], lr=0.1)
        source.shared.square().sum().backward()
        guarded_step(source, source_optimizer)
        source.window = 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sparse-owner.pt"
            save_checkpoint(path, source, source_optimizer)

            target = _SparseOwnerModule(torch.tensor([9.0, 8.0]))
            target_optimizer = torch.optim.Adam([target.shared], lr=0.5)
            load_checkpoint(path, target, target_optimizer)

        self.assertTrue(torch.equal(target.shared, source.shared))
        self.assertTrue(torch.equal(target.handle.values, source.handle.values))
        self.assertEqual(target.window, 7)
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.1)
        self.assertTrue(torch.equal(target_optimizer.state[target.shared]["exp_avg"],
                                    source_optimizer.state[source.shared]["exp_avg"]))
