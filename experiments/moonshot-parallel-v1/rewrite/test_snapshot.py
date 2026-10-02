"""Numerical and publication-lifetime witnesses for snapshot.py."""
import unittest
from unittest.mock import patch
import torch
from snapshot import Coordinate, Snapshot, SnapshotController, SavedTape

DT = torch.float64

def tensor(values):
    return torch.tensor(values, dtype=DT)


def fixture():
    return Snapshot((Coordinate('a', 2), Coordinate('b', 7)), tensor([2., -1.]),
                    tensor([[1., 2.], [0., 3.]]), tensor([[2., -1.]]),
                    replica_maps=((0, 1, 0), (1, 0)))


def gradient(controller, steps=3):
    tape = controller.acquire_tape(steps)
    output = tape.output.detach()
    return output, tape.backward()


class SnapshotTests(unittest.TestCase):
    def assertTensor(self, left, right):
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)

    def test_invertible_output_and_all_gradient_pullbacks(self):
        old = SnapshotController(fixture())
        output, grads = gradient(old)
        t = tensor([[2., 1.], [1., 1.]])
        inv = torch.linalg.inv(t)
        new = old.publish_transform(t, (Coordinate('q0', 0), Coordinate('q1', 0)),
                                    expected_epoch=0, expected_coordinates=fixture().coordinates,
                                    replica_maps=((1, 0, 1),))
        result, migrated = gradient(old)
        self.assertTensor(result, output)
        self.assertTensor(t.T @ migrated['state'], grads['state'])
        self.assertTensor(t.T @ migrated['law'] @ inv.T, grads['law'])
        self.assertTensor(migrated['readout'] @ inv.T, grads['readout'])
        self.assertEqual((new.structure_epoch, new.value_generation), (1, 1))
        self.assertTensor(old.replica_values()[0], new.state[[1, 0, 1]])

    def quotient_fixture(self):
        e = tensor([[1., 0.], [1., 0.], [0., 1.]])
        p = tensor([[.5, .5, 0.], [0., 0., 1.]])
        b = tensor([[2., 1.], [0., 3.]])
        snap = Snapshot(tuple(Coordinate(f's{i}', 0) for i in range(3)),
                        e @ tensor([2., -1.]), e @ b @ p, tensor([[1., 2., -1.]]))
        return snap, p, e

    def test_invariant_quotient_outputs_and_tangent_gradient(self):
        snap, p, e = self.quotient_fixture()
        controller = SnapshotController(snap)
        output, old_grads = gradient(controller)
        controller.publish_quotient(p, e, (Coordinate('q0', 0), Coordinate('q1', 0)),
                                    expected_epoch=0, expected_coordinates=snap.coordinates)
        result, new_grads = gradient(controller)
        self.assertTensor(result, output)
        self.assertTensor(new_grads['state'], e.T @ old_grads['state'])
        self.assertTensor(new_grads['law'], e.T @ old_grads['law'] @ p.T)
        self.assertTensor(new_grads['readout'], old_grads['readout'] @ p.T)

    def test_quotient_validation_is_atomic(self):
        snap, p, e = self.quotient_fixture()
        for bad_p, bad_e, bad_snap in ((p*2, e, snap),
              (p, e, Snapshot(snap.coordinates, snap.state, torch.eye(3, dtype=DT)*0 + tensor([[1., 0., 0.], [0., 2., 0.], [0., 0., 1.]]), snap.readout)),
              (p, e, Snapshot(snap.coordinates, tensor([1., 2., 3.]), snap.law, snap.readout))):
            controller = SnapshotController(bad_snap)
            before = controller.snapshot
            with self.assertRaises(ValueError):
                controller.publish_quotient(bad_p, bad_e, (Coordinate('q0', 0), Coordinate('q1', 0)),
                    expected_epoch=0, expected_coordinates=snap.coordinates)
            self.assertTensor(controller.snapshot.state, before.state)
            self.assertTensor(controller.snapshot.law, before.law)
            self.assertEqual(controller.snapshot.coordinates, before.coordinates)

    def test_tapes_block_every_publication_and_restore(self):
        snap = fixture()
        c = SnapshotController(snap)
        held = c.acquire_tape()
        checkpoint = c.export_checkpoint()
        kwargs = dict(expected_epoch=0, expected_coordinates=snap.coordinates)
        calls = (lambda: c.publish_transform(torch.eye(2, dtype=DT), snap.coordinates, **kwargs),
                 lambda: c.publish_quotient(torch.eye(2, dtype=DT), torch.eye(2, dtype=DT), snap.coordinates, **kwargs),
                 lambda: c.restore_checkpoint(checkpoint, **kwargs))
        for call in calls:
            with self.assertRaisesRegex(RuntimeError, 'drain'):
                call()
        grads = held.backward()
        self.assertEqual(set(grads), {'state', 'law', 'readout'})
        for operation in (held.backward, lambda: held.output):
            with self.assertRaises(RuntimeError):
                operation()
        c.assert_drained()
        c.publish_transform(torch.eye(2, dtype=DT), snap.coordinates, **kwargs)
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            c.publish_transform(torch.eye(2, dtype=DT), snap.coordinates, **kwargs)
        wrong = (Coordinate('a', 3), Coordinate('b', 7))
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            c.restore_checkpoint(checkpoint, expected_epoch=1, expected_coordinates=wrong)

    def test_identity_order_and_invalid_transform_are_atomic(self):
        snap = fixture()
        c = SnapshotController(snap)
        kwargs = dict(expected_epoch=0, expected_coordinates=snap.coordinates)
        for matrix, coords in ((tensor([[0., 0.], [0., 0.]]), snap.coordinates),
                               (torch.eye(2, dtype=DT), (Coordinate('same', 0), Coordinate('same', 1))),
                               (tensor([[float('nan'), 0.], [0., 1.]]), snap.coordinates)):
            with self.assertRaises(ValueError):
                c.publish_transform(matrix, coords, **kwargs)
            self.assertEqual(c.snapshot.structure_epoch, 0)
        with self.assertRaises(RuntimeError):
            c.publish_transform(torch.eye(2, dtype=DT), snap.coordinates,
                expected_epoch=0, expected_coordinates=tuple(reversed(snap.coordinates)))

    def test_replica_copy_and_sum_adjoint(self):
        c = SnapshotController(fixture())
        self.assertTensor(c.replica_values()[0], tensor([2., -1., 2.]))
        self.assertTensor(c.replica_adjoint((tensor([1., 2., 3.]), tensor([5., 7.]))), tensor([11., 7.]))
        with self.assertRaises(ValueError):
            c.replica_adjoint((tensor([1., 2.]), tensor([1., 2.])))

    def test_checkpoint_complete_validation_and_defensive_cloning(self):
        snap = fixture()
        c = SnapshotController(snap)
        snap.state.fill_(99)
        self.assertTensor(c.snapshot.state, tensor([2., -1.]))
        external = c.snapshot
        external.law.fill_(99)
        payload = c.export_checkpoint()
        fresh = SnapshotController.from_checkpoint(payload)
        original_output, original_grads = gradient(c)
        restored_output, restored_grads = gradient(fresh)
        self.assertTensor(original_output, restored_output)
        for name in original_grads:
            self.assertTensor(original_grads[name], restored_grads[name])
        payload['state'].fill_(99)
        self.assertTensor(fresh.snapshot.state, tensor([2., -1.]))
        for key, value in (('schema_version', 2), ('structure_epoch', -1),
                           ('value_generation', True), ('state', tensor([1.])),
                           ('law', tensor([[float('inf'), 0.], [0., 1.]])),
                           ('replica_maps', ((2,),)), ('coordinates', (('a', 0), ('a', 1)))):
            malformed = c.export_checkpoint()
            malformed[key] = value
            with self.assertRaises(ValueError):
                SnapshotController.from_checkpoint(malformed)
        malformed = c.export_checkpoint()
        malformed.pop('readout')
        with self.assertRaises(ValueError):
            SnapshotController.from_checkpoint(malformed)
        malformed = c.export_checkpoint()
        malformed['extra'] = 1
        with self.assertRaises(ValueError):
            SnapshotController.from_checkpoint(malformed)

    def test_malformed_public_snapshots_and_atomic_restore_failure(self):
        snap = fixture()
        for bad in (None,
                    Snapshot(None, snap.state, snap.law, snap.readout),
                    Snapshot(snap.coordinates, None, snap.law, snap.readout),
                    Snapshot(snap.coordinates, snap.state, snap.law, snap.readout, replica_maps=(None,)),
                    Snapshot((Coordinate('a', True), Coordinate('b', 0)), snap.state, snap.law, snap.readout)):
            with self.assertRaises(ValueError):
                SnapshotController(bad)
        c = SnapshotController(snap)
        malformed = c.export_checkpoint()
        malformed['law'] = tensor([[1.]])
        with self.assertRaises(ValueError):
            c.restore_checkpoint(malformed, expected_epoch=0, expected_coordinates=snap.coordinates)
        self.assertTensor(c.snapshot.law, snap.law)
        self.assertEqual(c.snapshot.structure_epoch, 0)
        c.assert_drained()
        with self.assertRaises(ValueError):
            c.acquire_tape('bad')
        c.assert_drained()
        held = c.acquire_tape()
        with self.assertRaises(ValueError):
            held.backward(tensor([1., 2.]))
        with self.assertRaises(RuntimeError):
            c.assert_drained()
        held.backward()
        c.assert_drained()

    def test_direct_tape_construction_requires_controller_admission(self):
        c = SnapshotController(fixture())
        with self.assertRaisesRegex(TypeError, 'acquired through'):
            SavedTape(c, 1)
        with self.assertRaisesRegex(TypeError, 'acquired through'):
            SavedTape(c, 1, _admission=object())
        c.assert_drained()
        with patch.object(torch.Tensor, '__matmul__', side_effect=RuntimeError('graph build failed')):
            with self.assertRaisesRegex(RuntimeError, 'graph build failed'):
                c.acquire_tape(1)
        c.assert_drained()
        self.assertEqual(c.snapshot.structure_epoch, 0)
        tape = c.acquire_tape(1)
        self.assertEqual(len(c._tapes), 1)
        with self.assertRaisesRegex(RuntimeError, 'drain'):
            c.publish_transform(torch.eye(2, dtype=DT), fixture().coordinates,
                expected_epoch=0, expected_coordinates=fixture().coordinates)
        tape.release()
        tape.release()
        c.assert_drained()
        c.publish_transform(torch.eye(2, dtype=DT), fixture().coordinates,
            expected_epoch=0, expected_coordinates=fixture().coordinates)

    def test_context_release_and_zero_step_gradients(self):
        c = SnapshotController(fixture())
        with c.acquire_tape(0) as tape:
            self.assertTensor(tape.output, fixture().readout @ fixture().state)
            self.assertTensor(tape.backward()['law'], torch.zeros((2, 2), dtype=DT))
        c.assert_drained()
        with self.assertRaises(ValueError):
            c.acquire_tape(-1)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
