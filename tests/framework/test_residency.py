"""Backend-neutral capacity, transfer, pinning and revision smoke tests."""
from concurrent.futures import Future
import unittest

from cellerator.execution.residency import (
    CapacityError, MemoryCategory, ParameterKey, ResidencyError, ResidencyManager,
)


class ResidencyTests(unittest.TestCase):
    def manager(self, transfer=None, *, capacities=None, disposed=None):
        calls = []

        def copy(payload, source, target):
            calls.append((payload, source, target))
            return payload if transfer is None else transfer(payload, source, target)

        def dispose(payload, device):
            disposed.append((payload, device))

        manager = ResidencyManager(capacities or {"gpu0": 8}, copy,
                                  dispose=dispose if disposed is not None else None)
        manager.calls = calls
        return manager

    def register(self, manager, name="expert", revision=1, *, nbytes=4,
                 category=MemoryCategory.PARAMETER, source="host"):
        key = ParameterKey(name, revision)
        payload = {"expert": name, "revision": revision}
        manager.register(key, payload, nbytes=nbytes, category=category, source=source)
        return key, payload

    def test_demand_lease_pins_copy_until_release_and_host_backing_survives(self):
        disposed = []
        manager = self.manager(disposed=disposed)
        key, host = self.register(manager)
        self.assertFalse(manager.resident(key, "gpu0"))
        lease = manager.acquire(key, "gpu0")
        resident_payload = lease.result()
        self.assertEqual(resident_payload, host)
        self.assertTrue(manager.resident(key, "gpu0"))
        self.assertEqual(manager.usage("gpu0")[MemoryCategory.PARAMETER], 4)
        self.assertFalse(manager.evict(key, "gpu0"))
        lease.release()
        lease.release()
        self.assertTrue(manager.evict(key, "gpu0"))
        self.assertEqual(disposed, [(host, "gpu0")])
        self.assertEqual(manager.usage("gpu0")[MemoryCategory.PARAMETER], 0)
        self.assertEqual(manager.acquire(key, "gpu0").result(), host)

    def test_async_transfer_reserves_capacity_and_shares_one_transfer(self):
        transfer = Future()
        calls = []

        def delayed(payload, source, target):
            calls.append((payload, source, target))
            return transfer

        manager = self.manager(delayed, capacities={"gpu0": 4})
        key, payload = self.register(manager, nbytes=4)
        first = manager.acquire(key, "gpu0")
        second = manager.acquire(key, "gpu0")
        completion_eviction = []
        first.future.add_done_callback(
            lambda _: completion_eviction.append(manager.evict(key, "gpu0")))
        self.assertFalse(first.future.cancel())
        with self.assertRaises(RuntimeError):
            first.future.set_result("forged")
        self.assertEqual(len(calls), 1)
        self.assertFalse(manager.resident(key, "gpu0"))
        self.assertEqual(manager.usage("gpu0")[MemoryCategory.PARAMETER], 4)
        with self.assertRaises(CapacityError):
            other, _ = self.register(manager, "other", nbytes=1)
            manager.acquire(other, "gpu0")
        transfer.set_result(("copy", payload))
        self.assertEqual(completion_eviction, [False])
        self.assertEqual(first.result(), ("copy", payload))
        self.assertEqual(second.result(), ("copy", payload))
        self.assertTrue(manager.resident(key, "gpu0"))
        first.release()
        self.assertFalse(manager.evict(key, "gpu0"))
        second.release()
        self.assertTrue(manager.evict(key, "gpu0"))

    def test_failed_shared_transfer_drops_reservation_and_all_pins(self):
        transfer = Future()
        calls = [0]

        def once_then_copy(payload, *_):
            calls[0] += 1
            return transfer if calls[0] == 1 else payload

        manager = self.manager(once_then_copy, capacities={"gpu0": 4})
        key, _ = self.register(manager, nbytes=4)
        first = manager.acquire(key, "gpu0")
        second = manager.acquire(key, "gpu0")
        transfer.set_exception(RuntimeError("copy failed"))
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            first.result()
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            second.result()
        self.assertEqual(manager.usage("gpu0")[MemoryCategory.PARAMETER], 0)
        self.assertFalse(manager.resident(key, "gpu0"))
        first.release()
        second.release()
        retry = manager.acquire(key, "gpu0")
        self.assertEqual(retry.result(), {"expert": "expert", "revision": 1})
        retry.release()

    def test_capacity_evicts_only_unpinned_completed_entries_in_lru_order(self):
        disposed = []
        manager = self.manager(capacities={"gpu0": 8}, disposed=disposed)
        a, a_payload = self.register(manager, "a", nbytes=4)
        b, b_payload = self.register(manager, "b", nbytes=4)
        la = manager.acquire(a, "gpu0")
        la.result()
        la.release()
        lb = manager.acquire(b, "gpu0")
        lb.result()
        lb.release()
        # Touch a so b is the oldest once the next 4-byte request arrives.
        la = manager.acquire(a, "gpu0")
        la.result()
        la.release()
        lc, _ = self.register(manager, "c", nbytes=4)
        lc_lease = manager.acquire(lc, "gpu0")
        lc_lease.result()
        self.assertTrue(manager.resident(a, "gpu0"))
        self.assertTrue(manager.resident(c := lc, "gpu0"))
        self.assertFalse(manager.resident(b, "gpu0"))
        self.assertIn((b_payload, "gpu0"), disposed)
        self.assertNotIn((a_payload, "gpu0"), disposed)
        self.assertFalse(manager.evict(b, "gpu0"))
        lb.release()
        lc_lease.release()

    def test_insufficient_room_does_not_partially_evict_cache_entries(self):
        manager = self.manager(capacities={"gpu0": 8})
        a, _ = self.register(manager, "cache", nbytes=4)
        b, _ = self.register(manager, "pinned", nbytes=4)
        c, _ = self.register(manager, "large", nbytes=8)
        cache = manager.acquire(a, "gpu0")
        pinned = manager.acquire(b, "gpu0")
        cache.result()
        pinned.result()
        cache.release()
        with self.assertRaises(CapacityError):
            manager.acquire(c, "gpu0")
        self.assertTrue(manager.resident(a, "gpu0"))
        self.assertTrue(manager.resident(b, "gpu0"))
        self.assertFalse(manager.resident(c, "gpu0"))
        pinned.release()

    def test_prefetch_is_unpinned_and_demand_can_join_it(self):
        transfer = Future()
        manager = self.manager(lambda *_: transfer, capacities={"gpu0": 4})
        key, payload = self.register(manager, nbytes=4)
        future = manager.prefetch(key, "gpu0")
        self.assertIsInstance(future, Future)
        self.assertFalse(manager.resident(key, "gpu0"))
        lease = manager.acquire(key, "gpu0")
        self.assertIs(lease.future, future)
        self.assertFalse(manager.evict(key, "gpu0"))
        transfer.set_result(payload)
        self.assertEqual(lease.result(), payload)
        lease.release()
        self.assertTrue(manager.evict(key, "gpu0"))

    def test_prefetch_cannot_be_invalidated_inside_completion_callback(self):
        transfer = Future()
        manager = self.manager(lambda *_: transfer, capacities={"gpu0": 4})
        old, _ = self.register(manager, revision=1, nbytes=4)
        latest, _ = self.register(manager, revision=2, nbytes=4)
        future = manager.prefetch(old, "gpu0")
        observed = []

        def invalidate_during_completion(_):
            try:
                manager.invalidate("expert")
            except ResidencyError as exc:
                observed.append(str(exc))

        future.add_done_callback(invalidate_during_completion)
        transfer.set_result({"old": "copy"})
        self.assertEqual(observed, ["cannot invalidate pinned or in-flight revisions"])
        self.assertTrue(manager.resident(old, "gpu0"))
        self.assertEqual(manager.invalidate("expert"), (old,))
        self.assertTrue(manager.resident(latest, "host"))

    def test_explicit_exact_resident_source_is_used_and_pinned_through_transfer(self):
        remote = Future()
        calls = []

        def route(payload, source, target):
            calls.append((payload, source, target))
            return remote if target == "gpu0" else {"on": target, "source_payload": payload}

        manager = self.manager(route, capacities={"gpu0": 4, "gpu1": 4})
        key, host = self.register(manager, nbytes=4)
        source_lease = manager.acquire(key, "gpu1")
        gpu1_payload = source_lease.result()
        source_lease.release()

        destination = manager.acquire(key, "gpu0", source="gpu1")
        self.assertEqual(calls[-1], (gpu1_payload, "gpu1", "gpu0"))
        self.assertEqual(destination.source, "gpu1")
        self.assertFalse(manager.evict(key, "gpu1"))  # transfer pin
        destination.release()
        self.assertFalse(manager.evict(key, "gpu1"))  # held through in-flight copy
        remote.set_result({"on": "gpu0", "source_payload": gpu1_payload})
        self.assertTrue(manager.evict(key, "gpu1"))
        self.assertTrue(manager.evict(key, "gpu0"))
        self.assertEqual(manager.acquire(key, "gpu1").result(), gpu1_payload)
        self.assertEqual(host, {"expert": "expert", "revision": 1})

    def test_explicit_source_failure_releases_source_pin(self):
        remote = Future()
        manager = self.manager(lambda payload, source, target:
                               remote if target == "gpu0" else payload,
                               capacities={"gpu0": 4, "gpu1": 4})
        key, _ = self.register(manager, nbytes=4)
        source = manager.acquire(key, "gpu1")
        source.result()
        source.release()
        target = manager.acquire(key, "gpu0", source="gpu1")
        remote.set_exception(RuntimeError("copy failed"))
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            target.result()
        self.assertTrue(manager.evict(key, "gpu1"))
        target.release()

    def test_explicit_source_must_be_exact_and_capacity_failure_unpins_it(self):
        manager = self.manager(capacities={"gpu0": 4, "gpu1": 4, "blocked": 4})
        old, _ = self.register(manager, revision=1, nbytes=4)
        new, _ = self.register(manager, revision=2, nbytes=4)
        old_lease = manager.acquire(old, "gpu1")
        old_lease.result()
        old_lease.release()
        with self.assertRaises(KeyError):
            manager.acquire(new, "gpu0", source="gpu1")
        # A caller-owned full source location leaves no room for destination.
        filler = ParameterKey("filler", 1)
        manager.register(filler, object(), nbytes=4, source="blocked",
                         category=MemoryCategory.CELL_STATE)
        with self.assertRaises(CapacityError):
            manager.acquire(old, "blocked", source="gpu1")
        self.assertTrue(manager.evict(old, "gpu1"))

    def test_memory_categories_are_accounted_separately(self):
        manager = self.manager(capacities={"gpu0": 10})
        a, _ = self.register(manager, "expert", nbytes=4,
                             category=MemoryCategory.PARAMETER)
        b, _ = self.register(manager, "cell", nbytes=3,
                             category=MemoryCategory.CELL_STATE)
        c, _ = self.register(manager, "optimizer", nbytes=2,
                             category=MemoryCategory.OPTIMIZER)
        leases = [manager.acquire(key, "gpu0") for key in (a, b, c)]
        for lease in leases:
            lease.result()
        usage = manager.usage("gpu0")
        self.assertEqual(usage[MemoryCategory.PARAMETER], 4)
        self.assertEqual(usage[MemoryCategory.CELL_STATE], 3)
        self.assertEqual(usage[MemoryCategory.OPTIMIZER], 2)
        self.assertEqual(usage[MemoryCategory.ACTIVATION], 0)
        for lease in leases:
            lease.release()

    def test_invalidation_is_atomic_and_preserves_latest_by_default(self):
        disposed = []
        manager = self.manager(disposed=disposed)
        old, old_host = self.register(manager, revision="old")
        new, new_host = self.register(manager, revision="new")
        old_lease = manager.acquire(old, "gpu0")
        old_lease.result()
        new_lease = manager.acquire(new, "gpu0")
        new_lease.result()
        with self.assertRaises(ResidencyError):
            manager.invalidate("expert")
        self.assertEqual(manager.latest_revision("expert"), "new")
        self.assertTrue(manager.resident(old, "gpu0"))
        old_lease.release()
        removed = manager.invalidate("expert")
        self.assertEqual(removed, (old,))
        self.assertFalse(manager.resident(old, "gpu0"))
        self.assertTrue(manager.resident(new, "gpu0"))
        self.assertEqual(disposed, [(old_host, "gpu0")])
        self.assertEqual(manager.acquire(new, "gpu0").result(), new_host)
        new_lease.release()

    def test_invalidation_can_explicitly_retain_a_chosen_revision(self):
        manager = self.manager()
        old, _ = self.register(manager, revision=1)
        latest, _ = self.register(manager, revision=2)
        self.assertEqual(manager.invalidate("expert", keep_revision=1), (latest,))
        self.assertEqual(manager.latest_revision("expert"), 1)
        lease = manager.acquire(old, "gpu0")
        self.assertEqual(lease.result(), {"expert": "expert", "revision": 1})
        lease.release()
        with self.assertRaises(KeyError):
            manager.acquire(latest, "gpu0")

    def test_none_is_a_valid_explicit_revision_to_retain(self):
        manager = self.manager()
        none_key, _ = self.register(manager, revision=None)
        latest, _ = self.register(manager, revision="new")
        self.assertEqual(manager.invalidate("expert", keep_revision=None), (latest,))
        self.assertIsNone(manager.latest_revision("expert"))
        lease = manager.acquire(none_key, "gpu0")
        self.assertEqual(lease.result(), {"expert": "expert", "revision": None})
        lease.release()

    def test_failed_transfer_to_unmanaged_target_cleans_up_without_masking_error(self):
        attempts = [0]

        def fail_once(payload, source, target):
            attempts[0] += 1
            if target == "remote-unmanaged":
                raise RuntimeError("remote copy failed")
            return payload

        manager = self.manager(fail_once)
        key, _ = self.register(manager)
        lease = manager.acquire(key, "remote-unmanaged")
        with self.assertRaisesRegex(RuntimeError, "remote copy failed"):
            lease.result()
        self.assertFalse(manager.resident(key, "remote-unmanaged"))
        lease.release()
        # A subsequent managed request can still reserve its full capacity.
        retry = manager.acquire(key, "gpu0")
        self.assertEqual(retry.result(), {"expert": "expert", "revision": 1})
        retry.release()

    def test_registered_source_is_caller_owned_and_not_evictable(self):
        manager = self.manager(capacities={"host": 4, "gpu0": 4})
        key, payload = self.register(manager, nbytes=4, source="host")
        self.assertTrue(manager.resident(key, "host"))
        self.assertFalse(manager.evict(key, "host"))
        self.assertEqual(manager.usage("host")[MemoryCategory.PARAMETER], 4)
        self.assertEqual(manager.acquire(key, "host").result(), payload)

    def test_duplicate_revision_and_bad_capacities_are_rejected(self):
        manager = self.manager()
        key, _ = self.register(manager)
        with self.assertRaises(ValueError):
            manager.register(key, object(), nbytes=1)
        with self.assertRaises(ValueError):
            ResidencyManager({"gpu0": -1}, lambda *args: None)
        with self.assertRaises(TypeError):
            ResidencyManager({"gpu0": True}, lambda *args: None)
        with self.assertRaises(TypeError):
            manager.register(ParameterKey("x", 1), object(), nbytes=1,
                             category="parameter")

    def test_disposal_errors_are_observable_after_accounting_eviction(self):
        manager = ResidencyManager({"gpu0": 4}, lambda payload, *_: payload,
                                   dispose=lambda *_: (_ for _ in ()).throw(RuntimeError("dispose")))
        key, _ = self.register(manager)
        lease = manager.acquire(key, "gpu0")
        lease.result()
        lease.release()
        self.assertTrue(manager.evict(key, "gpu0"))
        self.assertEqual(len(manager.disposal_errors), 1)
        self.assertEqual(manager.usage("gpu0")[MemoryCategory.PARAMETER], 0)


if __name__ == "__main__":
    unittest.main()
