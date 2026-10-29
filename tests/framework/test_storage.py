import unittest
import numpy as np

from cellerator.state.storage import SoAStateStore, StateHandle, StorageSpec


class DynamicStorageTests(unittest.TestCase):
    def test_growth_reuse_and_stale_incarnation_handle(self):
        store = SoAStateStore({"rna": StorageSpec((), np.float32),
                               "latent": StorageSpec((3,), np.float16)}, capacity=1)
        a = store.insert("a", {"rna": 1.0, "latent": [1, 2, 3]})
        initial_layout = store.layout_generation
        b = store.insert("b", {"rna": 2.0, "latent": [4, 5, 6]})
        self.assertGreater(store.capacity, 1)
        self.assertGreater(store.layout_generation, initial_layout)
        old_row = store.row_index(b)
        reusable_row = store.row_index(a)
        store.remove(a)
        c = store.insert("c", {"rna": 3.0, "latent": [7, 8, 9]})
        self.assertEqual(store.row_index(c), reusable_row)
        with self.assertRaises(KeyError):
            store.get(a, "rna")
        self.assertEqual(store.get(b, "latent").tolist(), [4, 5, 6])
        self.assertEqual(store.get("c", "rna"), np.float32(3))
        self.assertGreaterEqual(old_row, 0)

    def test_compaction_reorders_physical_rows_without_changing_handles(self):
        store = SoAStateStore({"activity": ((), np.float32)}, capacity=4)
        a = store.insert("a", {"activity": 10})
        b = store.insert("b", {"activity": 20})
        c = store.insert("c", {"activity": 30})
        store.remove(b)
        before = store.layout_generation
        layout = store.compact(("c", "a"))
        self.assertEqual(layout.physical_ids, ("c", "a"))
        self.assertEqual(store.gather((c, a), "activity").tolist(), [30.0, 10.0])
        self.assertEqual(store.row_index(a), 1)
        self.assertEqual(store.row_index(c), 0)
        self.assertEqual(store.handle("a"), a)
        self.assertGreater(store.layout_generation, before)
        self.assertEqual(store.structure_generation, 4)

    def test_views_generations_sparse_selection_and_state_reference(self):
        store = SoAStateStore({"counts": ((2,), np.int32), "gate": ((), np.bool_)})
        handles = [store.insert(k, {"counts": [i, i + 1], "gate": False})
                   for i, k in enumerate(("x", "y", "z"))]
        structure, value = store.structure_generation, store.value_generation
        store.update(handles[1], {"counts": [8, 9], "gate": True})
        self.assertEqual(store.structure_generation, structure)
        self.assertEqual(store.value_generation, value + 1)
        ref = store.state_reference(handles[1], "counts")
        ref.set(np.asarray([12, 13], dtype=np.int32))
        self.assertEqual(store.get(handles[1], "counts").tolist(), [12, 13])
        selection = store.select(("z", "x"))
        self.assertEqual([h.identity for h in selection], ["z", "x"])
        view = store.view(handles[0], "gate")
        view[...] = True
        store.mark_values_changed()
        self.assertTrue(store.get(handles[0], "gate"))

    def test_state_reference_cannot_alias_reinserted_identity(self):
        store = SoAStateStore({"activity": ((), np.float32)}, capacity=1)
        first = store.insert("cell", {"activity": 1.0})
        ref = store.state_reference(first, "activity")
        store.remove(first)
        replacement = store.insert("cell", {"activity": 99.0})
        self.assertNotEqual(first.incarnation, replacement.incarnation)
        with self.assertRaises(KeyError):
            ref.get()
        with self.assertRaises(KeyError):
            ref.set(np.float32(2.0))
        self.assertEqual(store.get(replacement, "activity"), np.float32(99.0))


if __name__ == "__main__":
    unittest.main()
