"""CPU contract tests for the thin Python prepared-routing adapter."""
import unittest
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                      / "python" / "cellerator" / "torch"))
from prepared_routing import (
    _PreparedRouteIndexCache,
    PreparedEntityMechanismRouting,
    PreparedRoutingIdentity,
)


class _FakeNativeRouter:
    def __init__(self, *arguments):
        self.arguments = arguments
        self.queries = []

    def query(self, entity_ids, *identity):
        self.queries.append((entity_ids, identity))
        return "offsets", "mechanisms"


class _FakeNativeModule:
    def __init__(self):
        self.router = None

    def PreparedEntityMechanismRouting(self, *arguments):
        self.router = _FakeNativeRouter(*arguments)
        return self.router


class PreparedRoutingAdapterTests(unittest.TestCase):
    def test_preserves_route_rows_and_all_dependency_keys(self):
        native = _FakeNativeModule()
        identity = PreparedRoutingIdentity(11, 12, 13, 14, 15, 16)
        rows = ((2, 1, 1), (), (0, 2))
        router = PreparedEntityMechanismRouting(
            native, rows, mechanism_count=3, identity=identity, device=4)

        self.assertEqual(native.router.arguments,
                         (rows, 3, 3, 11, 12, 13, 14, 15, 16, 4))
        ids = object()
        self.assertEqual(router.query(ids, identity=identity),
                         ("offsets", "mechanisms"))
        self.assertEqual(native.router.queries, [(ids, identity.as_tuple())])

    def test_rejects_stale_dependency_key_before_native_launch(self):
        native = _FakeNativeModule()
        identity = PreparedRoutingIdentity(11, 12, 13, 14, 15, 16)
        router = PreparedEntityMechanismRouting(
            native, ((0,),), mechanism_count=1, identity=identity, device=0)
        stale = replace(identity, route_generation=17)
        with self.assertRaisesRegex(ValueError, "dependencies changed"):
            router.query(object(), identity=stale)
        self.assertEqual(native.router.queries, [])

    def test_rejects_zero_identity_fields(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            PreparedRoutingIdentity(11, 12, 13, 0, 15, 16)

    def test_zero_published_generation_is_valid_and_activity_only_queries_reuse(self):
        native = _FakeNativeModule()
        cache = _PreparedRouteIndexCache()
        binding_generation = cache.observe_binding(2, 0x1234, 0x5678)
        identity = PreparedRoutingIdentity(11, 12, 13, 14, 0,
                                           binding_generation)
        rows = ((2, 1, 1), (), (0, 2))
        router, prepared = cache.get(native, rows, 3, identity, 2)
        self.assertTrue(prepared)
        self.assertEqual(router.query("activity-a", identity=identity),
                         ("offsets", "mechanisms"))

        same_router, prepared = cache.get(native, rows, 3, identity, 2)
        self.assertIs(same_router, router)
        self.assertFalse(prepared)
        same_router.query("activity-b", identity=identity)
        self.assertEqual(native.router.arguments[0], rows)
        self.assertEqual(len(native.router.queries), 2)

        # A published value generation changes the route dependency key.
        value_identity = replace(identity, route_generation=1)
        value_router, prepared = cache.get(native, rows, 3, value_identity, 2)
        self.assertTrue(prepared)
        self.assertIsNot(value_router, router)

        # A structure epoch change invalidates even when route rows are equal.
        structure_identity = replace(value_identity, structure_epoch=15)
        structure_router, prepared = cache.get(
            native, rows, 3, structure_identity, 2)
        self.assertTrue(prepared)
        self.assertIsNot(structure_router, value_router)

        # A native binding change advances the module-owned binding generation
        # and invalidates the previous projection before it can be queried.
        next_binding = cache.observe_binding(2, 0x1234, 0x9abc)
        self.assertEqual(next_binding, binding_generation + 1)
        binding_identity = replace(structure_identity,
                                   binding_generation=next_binding)
        binding_router, prepared = cache.get(
            native, rows, 3, binding_identity, 2)
        self.assertTrue(prepared)
        self.assertIsNot(binding_router, structure_router)


if __name__ == "__main__":
    unittest.main()
