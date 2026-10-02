"""Fresh-owner checkpoint witnesses combining the trajectory mechanisms."""
import json
import unittest
import numpy as np
import torch
from flow import SylvesterFlow
from delta import DeltaLedger
from polynomial import polynomial


class ContinuationTests(unittest.TestCase):
    def test_flow_checkpoint_rebuilds_cache_and_preserves_parameter_gradients(self):
        torch.manual_seed(91)
        dtype = torch.float64
        L = (torch.randn(3, 3, dtype=dtype) * 0.07).requires_grad_()
        R = (torch.randn(3, 3, dtype=dtype) * 0.07).requires_grad_()
        flow = SylvesterFlow(L, R, definition_id="checkpoint-law", structure_generation=4,
                             parameter_generation=8)
        warm = torch.randn(3, 3, dtype=dtype)
        flow(warm, 0.2, training=False)
        restored = SylvesterFlow.from_checkpoint(json.loads(json.dumps(flow.checkpoint())))
        self.assertEqual(restored.cache_hits, 0)
        self.assertEqual(restored.cache_misses, 0)
        torch.testing.assert_close(flow(warm, 0.2, training=False),
                                   restored(warm, 0.2, training=False))
        self.assertEqual(restored.cache_misses, 1)
        # Continue through a nonlinear matrix-polynomial readout. Fresh factors
        # must preserve the original flow dependence, including the time step.
        X1 = warm.clone().requires_grad_()
        X2 = warm.clone().requires_grad_()
        dt1 = torch.tensor(0.14, dtype=dtype, requires_grad=True)
        dt2 = dt1.detach().clone().requires_grad_()
        M = torch.randn(3, 3, dtype=dtype) * 0.05
        outputs = []
        gradients = []
        for owner, X, dt in ((flow, X1, dt1), (restored, X2, dt2)):
            Z = owner(X, dt, training=True)
            output = polynomial(Z, owner.L, owner.R, M)
            outputs.append(output)
            gradients.append(torch.autograd.grad(output.square().sum(),
                                                  (X, owner.L, owner.R, dt)))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for before, after in zip(*gradients):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_ledger_checkpoint_retains_pending_subthreshold_changes(self):
        W = np.array([[1.2, -0.3, 0.4], [-0.2, 0.7, 1.1]], dtype=np.float64)
        original = DeltaLedger(W, np.zeros(3), tau=0.1,
                               parameter_generation=3, structure_generation=5)
        original.update(np.array([0.06, 0.0, 0.02]))
        restored = DeltaLedger.from_checkpoint(json.loads(json.dumps(original.checkpoint())))
        np.testing.assert_array_equal(original.x_sent, restored.x_sent)
        np.testing.assert_array_equal(original.y, restored.y)
        for x in ([0.12, 0.04, 0.08], [0.15, 0.16, 0.2], [0.21, 0.19, 0.3]):
            y1, bound1 = original.update(np.array(x))
            y2, bound2 = restored.update(np.array(x))
            np.testing.assert_array_equal(y1, y2)
            np.testing.assert_array_equal(bound1, bound2)
            np.testing.assert_array_equal(original.x_sent, restored.x_sent)
            # Bound concerns the instantaneous fixed linear map only.
            self.assertTrue(np.all(np.abs(W @ x - y1) <= bound1 + 1e-14))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
