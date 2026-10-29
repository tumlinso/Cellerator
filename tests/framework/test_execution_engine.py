from concurrent.futures import Future
from dataclasses import replace
import threading
import unittest

from cellerator.execution import (
    BoundExecution, DeviceSpec, ExecutionEngine, ExpertPlacement, ExpertSpec,
    ExpertWorklist, HardwareTopology, LinkSpec, ParameterKey, ParameterRevision,
    ParameterStore, PlacementPlanner, PlanExecutionError, ResidencyError,
    ResidencyManager, SelectedWork, StaleRevisionError,
)


class EngineTests(unittest.TestCase):
    def fixture(self, backend=None, transfer=None):
        manager = ResidencyManager({"cpu": 1024}, transfer or (lambda value, *_: value))
        store = ParameterStore(manager)
        key = ParameterKey("gain", 0)
        store.register(ParameterRevision(key, 3, 4))
        work = ExpertWorklist.from_selections([
            SelectedWork(key, "cpu", "b", 0.25),
            SelectedWork(key, "cpu", "a", 1.0),
            SelectedWork(key, "cpu", "b", 0.75),
        ])
        topology = HardwareTopology([
            DeviceSpec("host", kind="host", memory_bytes=1024),
            DeviceSpec("cpu", memory_bytes=1024),
        ], [LinkSpec("host", "cpu", 1000, bidirectional=True)])
        plan = PlacementPlanner(topology, [ExpertSpec(key, 4)],
            [ExpertPlacement(key, "host", executable=False)]).plan(
                work.demands(activation_bytes_per_row=4, output_bytes_per_row=4))
        engine = ExecutionEngine(store, backend or (lambda request: tuple(
            request.inputs[selection.row_id] * request.parameters
            for selection in request.selections)))
        return engine, store, key, BoundExecution(plan, work, "structure-1")

    def test_executes_original_occurrences_and_exact_old_revision(self):
        engine, store, key, bound = self.fixture()
        with engine:
            store.publish(key, ParameterRevision(ParameterKey("gain", 1), 7, 4))
            result = engine.submit(bound, {"a": 2, "b": 5}, work_revision="structure-1").result()
            self.assertEqual(result.outputs, (15, 6, 15))
            self.assertEqual(tuple(x.weight for x in result.selections), (0.25, 1, 0.75))
            self.assertEqual(store.current("gain").revision, 1)
            self.assertEqual(store.prune("gain"), (key,))

    def test_rejects_changed_work_revision_and_substituted_rows(self):
        engine, store, key, bound = self.fixture()
        with engine:
            with self.assertRaises(StaleRevisionError):
                engine.submit(bound, {}, work_revision="structure-2")
            demand = replace(bound.plan.decisions[0].demand, work_ids=("a", "b", "b"))
            decision = replace(bound.plan.decisions[0], demand=demand)
            wrong = replace(bound, plan=replace(bound.plan, decisions=(decision,)))
            with self.assertRaises(ValueError):
                engine.submit(wrong, {}, work_revision="structure-1")
            omitted = replace(bound, plan=replace(bound.plan, decisions=()))
            with self.assertRaises(ValueError):
                engine.submit(omitted, {}, work_revision="structure-1")

    def test_backend_future_holds_pins_and_cleanup_precedes_result(self):
        completion = Future()
        entered = threading.Event()
        def backend(request):
            entered.set()
            return completion
        engine, store, key, bound = self.fixture(backend)
        with engine:
            future = engine.submit(bound, {}, work_revision="structure-1")
            self.assertTrue(entered.wait(3))
            self.assertFalse(store.residency.evict(key, "cpu"))
            store.publish(key, ParameterRevision(ParameterKey("gain", 1), 7, 4))
            with self.assertRaises(ResidencyError):
                store.prune("gain")
            completion.set_result([1, 2, 3])
            self.assertEqual(future.result(3).outputs, (1, 2, 3))
            self.assertEqual(store.prune("gain"), (key,))

    def test_failure_releases_pins_and_preserves_partial_report(self):
        engine, store, key, bound = self.fixture(lambda request: (_ for _ in ()).throw(ValueError("backend")))
        with engine:
            with self.assertRaises(PlanExecutionError) as caught:
                engine.submit(bound, {}, work_revision="structure-1").result()
            self.assertEqual(caught.exception.partial.completed_decisions, ())
            self.assertIsInstance(caught.exception.__cause__, ValueError)
            self.assertTrue(store.residency.evict(key, "cpu"))

    def test_async_transfer_is_awaited_before_backend(self):
        transfer = Future()
        entered = threading.Event()
        engine, store, key, bound = self.fixture(
            lambda request: entered.set() or (request.parameters,) * 3,
            lambda *_: transfer)
        with engine:
            result = engine.submit(bound, {}, work_revision="structure-1")
            self.assertFalse(entered.is_set())
            transfer.set_result(11)
            self.assertEqual(result.result(3).outputs, (11, 11, 11))

    def test_collective_retains_pins_beyond_forward(self):
        engine, store, key, bound = self.fixture()
        with engine:
            ticket = engine.submit_collective([key], "cpu", lambda params, x: params[key] * x, 2)
            self.assertEqual(ticket.result(), 6)
            self.assertFalse(store.residency.evict(key, "cpu"))
            store.publish(key, ParameterRevision(ParameterKey("gain", 1), 7, 4))
            with self.assertRaises(ResidencyError):
                store.prune("gain")
            ticket.close()
            ticket.close()
            self.assertEqual(store.prune("gain"), (key,))

    def test_stale_update_cannot_overwrite_new_parameters(self):
        engine, store, key, _ = self.fixture()
        with engine:
            next_revision = ParameterRevision(ParameterKey("gain", 1), 9, 4,
                master_parameters=9.0, gradients=2.0, optimizer_state={"step": 1})
            store.publish(key, next_revision)
            with self.assertRaises(StaleRevisionError):
                store.publish(key, ParameterRevision(ParameterKey("gain", 2), 2, 4))
            self.assertEqual(store.snapshot([next_revision.key])[0].master_parameters, 9.0)


class TorchBindingTests(unittest.TestCase):
    def test_exact_state_binding_preserves_input_and_parameter_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch unavailable")
        from cellerator.adapters.distributed import DistributedSparseExpertBank
        from cellerator.adapters.torch import SparseRouting
        from cellerator.execution import TorchCollectiveBinding
        module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(99)
        bank = DistributedSparseExpertBank({0: module}, [0], output_size=1)
        weight = torch.tensor([[3.0]], requires_grad=True)
        manager = ResidencyManager({"cpu": 1024}, lambda payload, *_: payload)
        store = ParameterStore(manager)
        key = ParameterKey("linear", 4)
        store.register(ParameterRevision(key, {"weight": weight}, 4))
        routing = SparseRouting(torch.tensor([0, 0]), torch.tensor([0, 1, 2]))
        binding = TorchCollectiveBinding(bank, {key: 0}, routing)
        x = torch.tensor([[2.0], [5.0]], requires_grad=True)
        with ExecutionEngine(store, lambda request: ()) as engine:
            with engine.submit_collective([key], "cpu", binding, x) as ticket:
                output = ticket.result()
                self.assertEqual(output.tolist(), [[6], [15]])
                output.sum().backward()
                self.assertEqual(x.grad.tolist(), [[3], [3]])
                self.assertEqual(weight.grad.tolist(), [[7]])
                self.assertFalse(manager.evict(key, "cpu"))
            self.assertTrue(manager.evict(key, "cpu"))
        self.assertEqual(module.weight.item(), 99)  # functional binding restores bank


if __name__ == "__main__":
    unittest.main()
