"""Safe biological meaning sidecars for ordinary and native model weights."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from cellerator._declarations import Identity
from cellerator.biology import FeatureBinding, IdentityCodec
from cellerator.torch.manifest import (
    BiologicalManifestError,
    biological_manifest,
    load_biological_manifest,
    save_biological_manifest,
    validate_biological_manifest,
)
from cellerator.torch.features import FeatureEncoder
from cellerator.torch.relations import PreparedRelation
from cellerator.torch.views import ModelWithViews


class BoundModel(nn.Module):
    def __init__(self, binding: FeatureBinding):
        super().__init__()
        self.encoder = nn.Embedding(32, 4)
        self.binding = binding
        self.shared = nn.Linear(4, 4)
        self.left = self.shared
        self.right = self.shared

    def cellerator_biological_descriptors(self, *, identity_codec=None):
        return {
            "feature_binding": self.binding.to_manifest(codec=identity_codec),
        }

    def forward(self, ids):
        return self.shared(self.encoder(ids))


class ViewsModel(BoundModel):
    def __init__(self, binding, views):
        super().__init__(binding)
        self._declared_views = tuple(views)
        self.selector_calls = 0

    def biological_view_descriptors(self, *, identity_codec=None):
        # The test descriptors are already data-only; this method must not run
        # any selector supplied by the model.
        return self._declared_views


class BiologicalManifestTests(unittest.TestCase):
    def _binding(self, identities=("ENSG-A", "ENSG-B"), model=(7, 2), *, metadata=None):
        return FeatureBinding.from_ids(
            identities, model, domain="gene", metadata=metadata or {
                "organism": "example-species", "assembly": "v1",
                "measurement": "RNA", "transform": "log1p",
            }, canonical_indices=(0, 1),
        )

    def test_feature_binding_preserves_canonical_and_noncontiguous_model_ids(self):
        binding = self._binding()
        descriptor = binding.to_manifest()
        self.assertEqual(descriptor["identities"], [
            {"type": "str", "value": "ENSG-A"},
            {"type": "str", "value": "ENSG-B"},
        ])
        self.assertEqual(descriptor["canonical_indices"], [0, 1])
        self.assertEqual(descriptor["model_indices"], [7, 2])
        restored = FeatureBinding.from_manifest(descriptor)
        self.assertEqual(restored.to_manifest(), descriptor)

    def test_bound_feature_encoder_exports_its_semantic_binding(self):
        binding = self._binding()
        encoder = FeatureEncoder(16, 4, binding=binding)
        manifest = biological_manifest(encoder)
        self.assertEqual(manifest["declarations"], [{
            "module_path": "",
            "kind": "feature_binding",
            "descriptor": binding.to_manifest(),
        }])
        self.assertEqual(biological_manifest(FeatureEncoder(16, 4))["declarations"], [])

    def test_model_with_views_exports_binding_and_selector_declaration(self):
        selector_calls = []

        def selector(result, _batch):
            selector_calls.append(True)
            return result

        def build(source_id):
            model = ModelWithViews(BoundModel(self._binding()))
            model.add_view(
                "atac", nn.Identity(), source=selector, source_id=source_id,
                output_binding=self._binding(
                    identities=("peak-a", "peak-b"), model=(0, 1),
                    metadata={"reference": "hg38", "measurement": "ATAC",
                              "transform": "binary"},
                ),
            )
            return model

        missing_id = ModelWithViews(BoundModel(self._binding()))
        missing_id.add_view("atac", nn.Identity(), source=selector)
        with self.assertRaisesRegex(ValueError, "source_id"):
            biological_manifest(missing_id)

        model = build("project.selector.atac.v1")
        manifest = biological_manifest(model)
        self.assertEqual(selector_calls, [])
        view = manifest["views"][0]
        self.assertEqual(view["name"], "atac")
        self.assertEqual(view["source_kind"], "callable")
        self.assertEqual(view["source_id"], "project.selector.atac.v1")
        self.assertEqual(
            view["output_binding"],
            self._binding(
                identities=("peak-a", "peak-b"), model=(0, 1),
                metadata={"reference": "hg38", "measurement": "ATAC",
                          "transform": "binary"},
            ).to_manifest(),
        )
        with self.assertRaisesRegex(BiologicalManifestError, "views"):
            validate_biological_manifest(build("project.selector.changed"), manifest)

    def test_native_128_bit_identity_roundtrips_without_custom_code(self):
        identity = Identity(0xFEDCBA9876543210, 0x123456789ABCDEF0)
        binding = FeatureBinding.from_ids((identity, "gene-B"), (11, 4), domain="gene")
        manifest = biological_manifest(BoundModel(binding))
        descriptor = manifest["declarations"][0]["descriptor"]["identities"][0]
        self.assertEqual(descriptor, {
            "type": "identity128", "low": identity.low, "high": identity.high,
        })
        restored = FeatureBinding.from_manifest(
            manifest["declarations"][0]["descriptor"]
        )
        self.assertEqual(restored.identities, (identity, "gene-B"))
        self.assertEqual(restored.model_indices, (11, 4))

    def test_custom_identity_codec_is_explicit_and_manifest_never_decodes_it(self):
        class GeneID:
            def __init__(self, value):
                self.value = value

            def __hash__(self):
                return hash(self.value)

            def __eq__(self, other):
                return isinstance(other, GeneID) and self.value == other.value

        decoder_calls = []
        codec = IdentityCodec(
            "example.gene-id",
            lambda identity: {"id": identity.value},
            lambda payload: decoder_calls.append(payload) or GeneID(payload["id"]),
            version=3,
        )
        binding = FeatureBinding.from_ids(
            (GeneID("g-1"),), (12,), domain="gene", identity_codec=codec,
        )
        model = BoundModel(binding)
        manifest = biological_manifest(model, identity_codec=codec)
        encoded = manifest["declarations"][0]["descriptor"]["identities"][0]
        self.assertEqual(encoded["type"], "custom")
        self.assertEqual(encoded["codec"], "example.gene-id")
        self.assertEqual(decoder_calls, [])
        restored = FeatureBinding.from_manifest(
            manifest["declarations"][0]["descriptor"],
            codecs={codec.name: codec},
        )
        self.assertEqual(restored.identities, (GeneID("g-1"),))
        self.assertEqual(len(decoder_calls), 1)

    def test_custom_identity_without_codec_fails_without_repr_fallback(self):
        class OpaqueID:
            pass

        binding = FeatureBinding.from_ids((OpaqueID(),), (0,))
        with self.assertRaisesRegex(TypeError, "explicit IdentityCodec"):
            binding.to_manifest()

    def test_manifest_rejects_wrong_binding_before_weight_load_callback(self):
        original = BoundModel(self._binding())
        manifest = biological_manifest(original)
        incompatible = BoundModel(self._binding(
            identities=("ENSG-A", "ENSG-C"), model=(7, 2),
        ))
        load_called = False

        def load_weights():
            nonlocal load_called
            validate_biological_manifest(incompatible, manifest)
            load_called = True

        with self.assertRaisesRegex(BiologicalManifestError, "declarations"):
            load_weights()
        self.assertFalse(load_called)

        wrong_model_ids = BoundModel(self._binding(
            identities=("ENSG-A", "ENSG-B"), model=(7, 3),
        ))
        with self.assertRaisesRegex(BiologicalManifestError, "declarations"):
            validate_biological_manifest(wrong_model_ids, manifest)

    def test_manifest_rejects_reference_metadata_and_output_order_changes(self):
        original_binding = self._binding()
        original = ViewsModel(original_binding, [{
            "name": "atac",
            "output_binding": self._binding(
                identities=("peak-1", "peak-2"), model=(0, 1),
                metadata={"reference": "hg38", "measurement": "ATAC",
                          "transform": "binary"},
            ).to_manifest(),
            "queryable": True,
            "source_kind": "mapping_key",
            "source_key": "regulatory",
            "source_id": None,
        }])
        manifest = biological_manifest(original)

        changed_binding = self._binding()
        changed = ViewsModel(changed_binding, [{
            "name": "atac",
            "output_binding": self._binding(
                identities=("peak-2", "peak-1"), model=(1, 0),
                metadata={"reference": "hg38", "measurement": "ATAC",
                          "transform": "binary"},
            ).to_manifest(),
            "queryable": True,
            "source_kind": "mapping_key",
            "source_key": "regulatory",
            "source_id": None,
        }])
        with self.assertRaisesRegex(BiologicalManifestError, "views"):
            validate_biological_manifest(changed, manifest)

        changed_reference = ViewsModel(changed_binding, [{
            "name": "atac",
            "output_binding": self._binding(
                identities=("peak-1", "peak-2"), model=(0, 1),
                metadata={"reference": "mm10", "measurement": "ATAC",
                          "transform": "binary"},
            ).to_manifest(),
            "queryable": True,
            "source_kind": "mapping_key",
            "source_key": "regulatory",
            "source_id": None,
        }])
        with self.assertRaisesRegex(BiologicalManifestError, "views"):
            validate_biological_manifest(changed_reference, manifest)

        changed_source = ViewsModel(original_binding, [{
            "name": "atac",
            "output_binding": self._binding(
                identities=("peak-1", "peak-2"), model=(0, 1),
                metadata={"reference": "hg38", "measurement": "ATAC",
                          "transform": "binary"},
            ).to_manifest(),
            "queryable": True,
            "source_kind": "mapping_key",
            "source_key": "different_representation",
            "source_id": None,
        }])
        with self.assertRaisesRegex(BiologicalManifestError, "views"):
            validate_biological_manifest(changed_source, manifest)

    def test_callable_selector_requires_caller_id_and_is_never_executed(self):
        binding = self._binding()
        missing_id = ViewsModel(binding, [{
            "name": "readout", "output_binding": None, "queryable": False,
            "source_kind": "callable", "source_key": None, "source_id": None,
        }])
        with self.assertRaisesRegex(BiologicalManifestError, "source_id"):
            biological_manifest(missing_id)

        with_id = ViewsModel(binding, [{
            "name": "readout", "output_binding": None, "queryable": False,
            "source_kind": "callable", "source_key": None,
            "source_id": "project.selector.v1",
        }])
        manifest = biological_manifest(with_id)
        self.assertEqual(with_id.selector_calls, 0)
        self.assertEqual(manifest["views"][0]["source_id"], "project.selector.v1")

    def test_explicit_prepared_relation_records_occurrence_order_and_duplicates(self):
        relation = PreparedRelation.from_csr(
            [0, 2, 2], [1, 1],
            source_order=("g1", "g1", "g2"),
            target_order=("program-a", "program-b"),
            source_offsets=(0, 3), target_offsets=(0, 2),
            structural_generation=9,
        )
        manifest = biological_manifest(nn.Linear(2, 2), relations={"regulation": relation})
        declaration = manifest["relations"][0]["descriptor"]
        self.assertEqual(declaration["source_order"], ["g1", "g1", "g2"])
        self.assertEqual(declaration["col_indices"], [1, 1])
        self.assertEqual(declaration["structural_generation"], 9)
        validate_biological_manifest(
            nn.Linear(2, 2), manifest, relations={"regulation": relation},
        )
        changed_order = PreparedRelation.from_csr(
            [0, 2, 2], [1, 1],
            source_order=("g2", "g1", "g1"),
            target_order=("program-a", "program-b"),
            source_offsets=(0, 3), target_offsets=(0, 2),
            structural_generation=9,
        )
        with self.assertRaisesRegex(BiologicalManifestError, "relations"):
            validate_biological_manifest(
                nn.Linear(2, 2), manifest, relations={"regulation": changed_order},
            )

    def test_weight_sidecar_hash_and_filename_pairing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            model = BoundModel(self._binding())
            weights = directory / "weights.pt"
            torch.save(model.state_dict(), weights)
            sidecar = directory / "weights.bio.json"
            saved = save_biological_manifest(model, sidecar, weights_path=weights)
            loaded = load_biological_manifest(sidecar, weights_path=weights)
            self.assertEqual(loaded, saved)
            validate_biological_manifest(model, loaded, weights_path=weights)
            state = torch.load(weights, weights_only=True)
            self.assertTrue(all(isinstance(value, torch.Tensor) for value in state.values()))
            weights.write_bytes(b"different weight bytes")
            with self.assertRaisesRegex(BiologicalManifestError, "SHA-256"):
                validate_biological_manifest(model, loaded, weights_path=weights)

    def test_shared_parameter_ownership_and_tensor_only_state_dict_are_unchanged(self):
        model = BoundModel(self._binding())
        before = model.state_dict()
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in before.values()))
        self.assertIs(model.left.weight, model.right.weight)
        parameter_ids = [id(parameter) for parameter in model.parameters()]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))
        manifest = biological_manifest(model)
        validate_biological_manifest(model, manifest)
        self.assertIs(model.left.weight, model.right.weight)
        self.assertEqual(set(model.state_dict()), set(before))

    def test_payload_data_and_duplicate_json_keys_are_rejected(self):
        class BadDescriptor(nn.Module):
            def cellerator_biological_descriptors(self, *, identity_codec=None):
                return {"feature_binding": {"history": ["cell event"]}}

        with self.assertRaisesRegex(TypeError, "payload field"):
            biological_manifest(BadDescriptor())

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(
                '{"format":"cellerator-biological-model-manifest",'
                '"version":1,"version":1,"declarations":[],'
                '"relations":[],"views":[],"weights":null}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BiologicalManifestError, "duplicate JSON"):
                load_biological_manifest(path)

    def test_plain_module_is_valid_without_claiming_biological_meaning(self):
        manifest = biological_manifest(nn.Sequential(nn.Linear(3, 2), nn.ReLU()))
        self.assertEqual(manifest["declarations"], [])
        self.assertEqual(manifest["relations"], [])
        self.assertEqual(manifest["views"], [])


if __name__ == "__main__":
    unittest.main()
