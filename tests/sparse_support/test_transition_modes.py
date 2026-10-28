"""CPU checks for semantic transition views and checkpoint mode preflight."""
from __future__ import annotations

import unittest

import torch

from cellerator.torch.execution import (
    _TRANSITION_ROUTING_POLICY, _TRANSITION_STORAGE_LAYOUT,
    _validate_transition_checkpoint_state, transition_parameter_views,
)


class TransitionModeContractTest(unittest.TestCase):
    def _blocks(self):
        flat = torch.nn.Parameter(torch.arange(3 * 4 + 3 * 6 + 3 * 2,
                                               dtype=torch.float32))
        a = flat[:12].view(3, 2, 2)
        b = flat[12:30].view(3, 2, 3)
        bias = flat[30:].view(3, 2)
        return flat, a, b, bias

    def test_shared_views_alias_only_block_zero_and_gradients_return_to_flat_owner(self):
        flat, a, b, bias = self._blocks()
        shared_a, shared_b, shared_bias = transition_parameter_views(
            a, b, bias, "shared")
        self.assertEqual(tuple(shared_a.shape), (2, 2))
        self.assertEqual(tuple(shared_b.shape), (2, 3))
        self.assertEqual(tuple(shared_bias.shape), (2,))
        self.assertEqual(shared_a.untyped_storage().data_ptr(), flat.untyped_storage().data_ptr())
        self.assertEqual(shared_b.untyped_storage().data_ptr(), flat.untyped_storage().data_ptr())
        self.assertEqual(shared_bias.untyped_storage().data_ptr(), flat.untyped_storage().data_ptr())
        (shared_a.sum() + shared_b.sum() + shared_bias.sum()).backward()
        expected = torch.zeros_like(flat)
        expected[:4] = 1
        expected[12:18] = 1
        expected[30:32] = 1
        self.assertTrue(torch.equal(flat.grad, expected))

    def test_per_mechanism_views_retain_stacks_and_flat_parameter_owner(self):
        _, a, b, bias = self._blocks()
        actual = transition_parameter_views(a, b, bias, "per_mechanism")
        self.assertIs(actual[0], a)
        self.assertIs(actual[1], b)
        self.assertIs(actual[2], bias)

    def test_bad_mode_and_misaligned_stacks_are_rejected(self):
        _, a, b, bias = self._blocks()
        with self.assertRaisesRegex(ValueError, "transition_mode"):
            transition_parameter_views(a, b, bias, "implicit")
        with self.assertRaisesRegex(ValueError, "aligned"):
            transition_parameter_views(a, b[:2], bias, "shared")

    def test_checkpoint_preflight_maps_untagged_legacy_to_shared(self):
        _validate_transition_checkpoint_state({"structure_epoch": 4}, "shared")
        tagged = {
            "transition_mode": "shared",
            "transition_storage_layout": _TRANSITION_STORAGE_LAYOUT,
            "transition_routing_policy": _TRANSITION_ROUTING_POLICY,
        }
        _validate_transition_checkpoint_state(tagged, "shared")

    def test_checkpoint_preflight_rejects_mismatch_and_unknown_contracts(self):
        per_mechanism = {
            "transition_mode": "per_mechanism",
            "transition_storage_layout": _TRANSITION_STORAGE_LAYOUT,
            "transition_routing_policy": _TRANSITION_ROUTING_POLICY,
        }
        with self.assertRaisesRegex(ValueError, "mode does not match"):
            _validate_transition_checkpoint_state(per_mechanism, "shared")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            _validate_transition_checkpoint_state({"transition_mode": "shared"}, "shared")
        bad_layout = dict(per_mechanism, transition_storage_layout="unknown")
        with self.assertRaisesRegex(ValueError, "storage layout"):
            _validate_transition_checkpoint_state(bad_layout, "per_mechanism")
        bad_routing = dict(per_mechanism, transition_routing_policy="unknown")
        with self.assertRaisesRegex(ValueError, "routing policy"):
            _validate_transition_checkpoint_state(bad_routing, "per_mechanism")


if __name__ == "__main__":
    unittest.main()
