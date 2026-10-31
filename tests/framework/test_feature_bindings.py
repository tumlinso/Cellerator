"""CPU contracts for biological identity bindings and query coordinates."""

from dataclasses import dataclass
import unittest

import torch

from cellerator._declarations import Axis, Identity
from cellerator.biology import (
    FeatureBinding,
    IdentityCodec,
    Prediction,
    PredictionQuery,
)
from cellerator.torch.features import FeatureEncoder


class FeatureBindingTests(unittest.TestCase):
    def test_named_alignment_preserves_requested_order_and_gradient(self):
        binding = FeatureBinding.from_ids(
            ("gene-A", "gene-B", "gene-C"),
            (2, 7, 11),
            domain="gene",
            canonical_indices=(40, 12, 91),
            metadata={"organism": "testus", "reference": {"release": "v3"}},
        )
        alignment = binding.prepare(
            ("gene-C", "gene-A", "gene-B"), ("gene-B", "gene-C")
        )
        self.assertEqual(alignment.source_order, ("gene-C", "gene-A", "gene-B"))
        self.assertEqual(alignment.requested_ids, ("gene-B", "gene-C"))
        self.assertEqual(alignment.source_positions, (2, 0))
        self.assertEqual(alignment.canonical_indices, (12, 91))
        self.assertEqual(alignment.model_indices, (7, 11))

        values = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        selected = alignment.gather(values, axis=1)
        torch.testing.assert_close(selected, torch.tensor([[3.0, 1.0]]))
        selected.sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([[1.0, 0.0, 1.0]]))

    def test_alias_resolution_ambiguity_and_canonical_precedence(self):
        binding = FeatureBinding.from_ids(
            ("canonical", "other"),
            aliases={"canonical": ("shared", "other", "canonical-alias"),
                     "other": ("shared",)},
        )
        self.assertEqual(binding.model_index("canonical"), 0)
        # Exact canonical identities take precedence over aliases.
        self.assertEqual(binding.model_index("other"), 1)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            binding.model_index("shared")
        with self.assertRaisesRegex(ValueError, "unique after alias"):
            binding.prepare(("canonical", "canonical-alias"), ("canonical",))

    def test_occurrence_mapping_retains_event_repeats_and_order(self):
        binding = FeatureBinding.from_ids(
            ("gene-a", "gene-b"), (4, 13), domain="gene"
        )
        mapped = binding.map_occurrences(("gene-b", "gene-a", "gene-b"))
        self.assertEqual(mapped.source_order, ("gene-b", "gene-a", "gene-b"))
        self.assertEqual(mapped.requested_ids, ("gene-b", "gene-a", "gene-b"))
        self.assertEqual(mapped.source_positions, (0, 1, 2))
        self.assertEqual(mapped.model_indices, (13, 4, 13))
        tokens = binding.map_occurrences((13, 4, 13), id_space="model")
        self.assertEqual(tokens.requested_ids, ("gene-b", "gene-a", "gene-b"))
        self.assertEqual(tokens.model_indices, (13, 4, 13))

    def test_unknown_policies_are_explicit_and_drop_keeps_input_extent(self):
        binding = FeatureBinding.from_ids(
            ("a", "b"), unknown_policy="drop"
        )
        with self.assertRaisesRegex(KeyError, "unknown feature"):
            FeatureBinding.from_ids(("a", "b")).prepare(("a", "unknown"))
        prepared = binding.prepare(("a", "unknown", "b"), ("b", "unknown", "a"))
        self.assertEqual(prepared.source_order, ("a", "unknown", "b"))
        self.assertEqual(prepared.requested_ids, ("b", "a"))
        self.assertEqual(prepared.source_positions, (2, 0))
        self.assertEqual(prepared.dropped_ids, ("unknown", "unknown"))
        values = torch.tensor([[2.0, 99.0, 4.0]])
        torch.testing.assert_close(prepared.gather(values, axis=1), torch.tensor([[4.0, 2.0]]))

    def test_equal_shapes_with_different_meaning_are_incompatible(self):
        first = FeatureBinding.from_ids(("gene-a", "gene-b"), metadata={"assembly": "A"})
        second = FeatureBinding.from_ids(("peak-a", "peak-b"), metadata={"assembly": "A"})
        third = FeatureBinding.from_ids(("gene-a", "gene-b"), metadata={"assembly": "B"})
        with self.assertRaisesRegex(ValueError, "identities"):
            first.assert_compatible(second)
        with self.assertRaisesRegex(ValueError, "metadata"):
            first.assert_compatible(third)
        with self.assertRaisesRegex(ValueError, "incompatible feature binding"):
            third.validate_alignment(first.prepare(("gene-a", "gene-b")))

    def test_manifest_roundtrips_builtin_and_custom_identities(self):
        axis = Axis(Identity(101, 102), Identity(201, 202),
                    Identity(301, 302), Identity(401, 402), 2)
        binding = FeatureBinding.from_ids(
            (Identity(1, 2), ("chr1", 20, 25)),
            (3, 8),
            domain="peak",
            axis=axis,
            metadata={"assembly": "GRCh38", "measurement": {"layer": "counts"}},
        )
        restored = FeatureBinding.from_manifest(binding.to_manifest())
        binding.assert_compatible(restored)
        self.assertEqual(restored.axis, axis)

        @dataclass(frozen=True)
        class Peak:
            chromosome: str
            start: int
            end: int

        codec = IdentityCodec(
            "test.peak",
            lambda value: {"chromosome": value.chromosome,
                           "start": value.start, "end": value.end},
            lambda value: Peak(value["chromosome"], value["start"], value["end"]),
        )
        custom = FeatureBinding.from_ids(
            (Peak("chr2", 10, 14),), identity_codec=codec
        )
        descriptor = custom.to_manifest()
        restored_custom = FeatureBinding.from_manifest(descriptor, codecs={codec.name: codec})
        self.assertEqual(restored_custom.identities, custom.identities)
        with self.assertRaisesRegex(TypeError, "explicit IdentityCodec"):
            FeatureBinding.from_ids((Peak("chr2", 10, 14),)).to_manifest()


class GeneVocabularyBindingTests(unittest.TestCase):
    def test_gene_vocabulary_keeps_canonical_and_noncontiguous_token_ids_distinct(self):
        try:
            from cellerator.tokenization import (
                GeneRecord,
                GeneVocabulary,
                GeneVocabularyMetadata,
                UnknownIdentityPolicy,
            )
        except ImportError as exc:
            self.skipTest(f"optional native tokenization is unavailable: {exc}")

        metadata = GeneVocabularyMetadata()
        metadata.source = "test"
        metadata.organism = "species"
        metadata.assembly = "assembly-1"
        metadata.release = "release-2"
        vocabulary = GeneVocabulary.from_records(
            [
                GeneRecord("gene-z", "SYMBOL_Z", ["old-z"], 17),
                GeneRecord("gene-a", "SYMBOL_A", ["old-a"], 4),
            ],
            metadata,
            UnknownIdentityPolicy.error,
        )
        binding = FeatureBinding.from_gene_vocabulary(vocabulary, model_capacity=18)
        self.assertEqual(binding.identities, ("gene-a", "gene-z"))
        self.assertEqual(binding.canonical_indices, (0, 1))
        self.assertEqual(binding.model_indices, (4, 17))
        alignment = binding.prepare(("SYMBOL_Z", "gene-a"), ("gene-a", "old-z"))
        self.assertEqual(alignment.requested_ids, ("gene-a", "gene-z"))
        self.assertEqual(alignment.source_positions, (1, 0))
        self.assertEqual(alignment.canonical_indices, (0, 1))
        self.assertEqual(alignment.model_indices, (4, 17))

        encoder = FeatureEncoder(18, 6, value_size=None, binding=binding).cpu()
        names = ("gene-z", "gene-a", "gene-z")
        encoded_bound = encoder.encode_bound(names)
        model_rows = torch.tensor([17, 4, 17], dtype=torch.long)
        torch.testing.assert_close(encoded_bound, encoder(model_rows))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            FeatureEncoder(17, 6, value_size=None, binding=binding)


class AnnDataBindingTests(unittest.TestCase):
    def test_sparse_anndata_features_align_by_binding_identity(self):
        try:
            from scipy import sparse
        except ImportError as exc:
            self.skipTest(f"optional SciPy sparse support is unavailable: {exc}")
        from cellerator.adapters.anndata import from_adata_bound

        class FakeAnnData:
            obs_names = ("cell-1", "cell-2")
            var_names = ("symbol-b", "unmodeled", "gene-a")
            layers = {}
            raw = None
            X = sparse.csr_matrix([[2.0, 99.0, 1.0], [3.0, 98.0, 4.0]])

        binding = FeatureBinding.from_ids(
            ("gene-a", "gene-b"),
            (8, 12),
            domain="gene",
            aliases={"gene-b": ("symbol-b",)},
            unknown_policy="drop",
        )
        view, alignment = from_adata_bound(
            FakeAnnData(), binding, requested_ids=("gene-b", "gene-a")
        )
        self.assertEqual(view.feature_ids, ("gene-b", "gene-a"))
        self.assertEqual(view.cell_ids, ("cell-1", "cell-2"))
        self.assertTrue(sparse.issparse(view.values))
        self.assertEqual(view.values.getformat(), "csr")
        self.assertEqual(alignment.source_positions, (0, 2))
        self.assertEqual(view.values.toarray().tolist(), [[2.0, 1.0], [3.0, 4.0]])


class PredictionRecordTests(unittest.TestCase):
    def test_prediction_records_preserve_borrowed_values_and_repeated_samples(self):
        values = torch.randn(3, 2, requires_grad=True)
        query = PredictionQuery(
            "atac",
            feature_ids=("peak-b", "peak-a"),
            sample_ids=("cell-1", "cell-1"),
            row_offsets=(0, 2, 3),
            conditioning=values,
            context_metadata={"experiment": "perturbation-7"},
        )
        prediction = Prediction(
            values,
            feature_ids=query.feature_ids,
            sample_ids=query.sample_ids,
            row_offsets=(0, 2, 3),
            context=query.conditioning,
        )
        self.assertIs(prediction.values, values)
        self.assertIs(prediction.context, values)
        self.assertIs(query.conditioning, values)
        self.assertEqual(prediction.sample_ids, ("cell-1", "cell-1"))
        self.assertEqual(prediction.feature_ids, ("peak-b", "peak-a"))
        with self.assertRaises(TypeError):
            query.context_metadata["new"] = "value"
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            PredictionQuery("rna", row_offsets=(0, 2, 1))


if __name__ == "__main__":
    unittest.main()
