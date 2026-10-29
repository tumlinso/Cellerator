"""Independent public-API checks for ATen tokenization.

CPU tests are safe to run directly. CUDA cases are explicitly opt-in because
the shared host requires scheduler admission for every GPU execution.
"""

import os
import unittest

import numpy as np
import torch

from cellerator import tokenization as tok


def cpu_tensor(value, dtype=None):
    return torch.as_tensor(value, dtype=dtype, device="cpu")


def ragged(batch):
    offsets = batch.offsets.to("cpu").numpy()
    ids = batch.token_ids.to("cpu").numpy()
    return [ids[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]


def make_vocab(identities, tokens=None, policy=tok.UnknownIdentityPolicy.error):
    if tokens is None:
        tokens = identities
    return tok.make_vocabulary(
        cpu_tensor(identities, torch.int64), cpu_tensor(tokens, torch.int64), policy
    )


class TokenizationCpuTests(unittest.TestCase):
    def test_vocabulary_mapping_preserves_repeated_events_and_attributes(self):
        vocab = make_vocab([3, 5, 9], [300, 500, 900])
        events = cpu_tensor([9, 3, 9, 5, 3], torch.int64)
        offsets = cpu_tensor([0, 3, 3, 5], torch.int64)
        time = cpu_tensor([4.0, 5.0, 8.0, 0.0, 2.0])
        encoded = tok.event_stream(events, offsets, vocab, {"time": time})
        self.assertEqual(ragged(encoded), [[900, 300, 900], [], [500, 300]])
        torch.testing.assert_close(encoded.attributes["time"], time)

    def test_unknown_identity_error_drop_and_explicit_mapping(self):
        identities = cpu_tensor([1, 7, 5], torch.int64)
        offsets = cpu_tensor([0, 3], torch.int64)
        strict = make_vocab([1, 5])
        with self.assertRaises((ValueError, RuntimeError)):
            tok.event_stream(identities, offsets, strict)

        dropped = make_vocab([1, 5], policy=tok.UnknownIdentityPolicy.drop)
        self.assertEqual(ragged(tok.event_stream(identities, offsets, dropped)), [[1, 5]])

        mapped = tok.make_vocabulary(
            cpu_tensor([1, 5], torch.int64), cpu_tensor([10, 50], torch.int64),
            tok.UnknownIdentityPolicy.map_to_token, 99,
        )
        self.assertEqual(ragged(tok.event_stream(identities, offsets, mapped)), [[10, 99, 50]])

    def test_unknown_drop_does_not_consume_ranked_top_k(self):
        # OOV features dropped by policy must not occupy the requested k slots.
        vocab = make_vocab([2], [22], policy=tok.UnknownIdentityPolicy.drop)
        values = cpu_tensor([[2.0, 1.0]])
        features = cpu_tensor([1, 2], torch.int64)
        options = tok.RankedFeatureOptions()
        options.max_per_row = 1
        result = tok.ranked_feature(values, features, vocab, options)
        self.assertEqual(ragged(result), [[22]])

    def test_dense_rank_uses_feature_identity_for_ties_and_truncation(self):
        feature_ids = cpu_tensor([30, 10, 20, 40], torch.int64)
        values = cpu_tensor([[5.0, 5.0, 0.0, -2.0], [0.0, 7.0, 7.0, 1.0]])
        # Identity order and token order intentionally differ.
        vocab = make_vocab([10, 20, 30, 40], [101, 202, 303, 404])
        options = tok.RankedFeatureOptions()
        options.max_per_row = 1
        result = tok.ranked_feature(values, feature_ids, vocab, options)
        self.assertEqual(ragged(result), [[101], [101]])

    def test_dense_sparse_agreement_and_original_nonzero_eligibility(self):
        feature_ids = cpu_tensor([5, 7, 8, 9], torch.int64)
        values = cpu_tensor([[2.0, 0.0, -4.0, 1.0], [0.0, 3.0, 0.0, 0.0]])
        vocab = make_vocab([5, 7, 8, 9], [50, 70, 80, 90])
        options = tok.RankedFeatureOptions()
        options.scores = cpu_tensor([[1.0, 9.0, -2.0, 1.0], [4.0, 1.0, 3.0, 8.0]])
        dense = tok.ranked_feature(values, feature_ids, vocab, options)
        csr = values.to_sparse_csr()
        csr_options = tok.RankedFeatureOptions()
        csr_options.scores = cpu_tensor([[1.0, 0.0, -2.0, 1.0],
                                         [0.0, 1.0, 0.0, 0.0]]).to_sparse_csr()
        sparse = tok.ranked_feature(csr, feature_ids, vocab, csr_options)
        # The zero-valued feature has the largest supplied score but remains
        # ineligible; a negative measurement remains eligible by default.
        expected = [[50, 90, 80], [70]]
        self.assertEqual(ragged(dense), expected)
        self.assertEqual(ragged(sparse), expected)

    def test_csr_scores_must_share_the_value_topology(self):
        values = cpu_tensor([[2.0, 0.0, 1.0]]).to_sparse_csr()
        # Same BxF dimensions, but scores store a different column pattern.
        scores = cpu_tensor([[0.0, 5.0, 0.0]]).to_sparse_csr()
        options = tok.RankedFeatureOptions()
        options.scores = scores
        vocab = make_vocab([1, 2, 3])
        with self.assertRaises((ValueError, RuntimeError)):
            tok.ranked_feature(values, cpu_tensor([1, 2, 3], torch.int64),
                               vocab, options)

    def test_feature_axis_permutation_does_not_change_token_sequence(self):
        feature_ids = cpu_tensor([4, 2, 8, 6], torch.int64)
        values = cpu_tensor([[0.5, 4.0, 3.0, 0.0], [2.0, 0.0, 0.0, 7.0]])
        vocab = make_vocab([2, 4, 6, 8], [20, 40, 60, 80])
        baseline = tok.ranked_feature(values, feature_ids, vocab)
        permutation = cpu_tensor([2, 0, 3, 1], torch.int64)
        permuted = tok.ranked_feature(
            values.index_select(1, permutation),
            feature_ids.index_select(0, permutation), vocab,
        )
        self.assertEqual(ragged(permuted), ragged(baseline))

    def test_ranked_measurements_attributes_and_gradient(self):
        values = cpu_tensor([[1.0, 4.0, 2.0]], torch.float64).requires_grad_()
        features = cpu_tensor([1, 2, 3], torch.int64)
        vocab = make_vocab([1, 2, 3], [101, 102, 103])
        metadata = cpu_tensor([10.0, 20.0, 30.0], torch.float64).requires_grad_()
        options = tok.RankedFeatureOptions()
        options.max_per_row = 2
        options.retain_measurements = True
        options.retain_scores = True
        result = tok.ranked_feature(values, features, vocab, options, {"quality": metadata})
        self.assertEqual(ragged(result), [[102, 103]])
        self.assertIn("measurement", result.attributes)
        self.assertIn("score", result.attributes)
        result.attributes["quality"].sum().backward()
        torch.testing.assert_close(metadata.grad, cpu_tensor([0.0, 1.0, 1.0], torch.float64))

    def test_geneformer_style_formula_against_numpy(self):
        # Pinned reference: Geneformer source revision
        # b36d210733fe66fc5514bb7e25205eb37b85eea4,
        # geneformer/tokenizer.py lines 284-315:
        # https://huggingface.co/ctheodoris/Geneformer/blob/b36d210733fe66fc5514bb7e25205eb37b85eea4/geneformer/tokenizer.py
        # The fixture uses synthetic counts/statistics, then independently
        # applies that fixed-count/median scaling and descending rank/truncation.
        counts = np.array([[3, 9, 0, 4], [1, 0, 6, 2]], dtype=np.float64)
        total = np.array([16, 9], dtype=np.float64)
        medians = np.array([2, 3, 1, 4], dtype=np.float64)
        features = np.array([40, 10, 30, 20], dtype=np.int64)
        tokens = {10: 5, 20: 6, 30: 7, 40: 8}
        vocab = make_vocab(list(tokens), list(tokens.values()))
        output = tok.geneformer_style(
            cpu_tensor(counts), cpu_tensor(total), cpu_tensor(medians),
            cpu_tensor(features), vocab, 3,
        )
        normalized = counts / total[:, None] * 10_000.0 / medians[None, :]
        expected = []
        for row in normalized:
            present = np.flatnonzero(counts[len(expected)] > 0)
            order = sorted(present, key=lambda j: (-row[j], features[j]))[:3]
            expected.append([tokens[int(features[j])] for j in order])
        self.assertEqual(ragged(output), expected)

    def test_invalid_profile_reference_factors_fail(self):
        vocab = make_vocab([1, 2])
        with self.assertRaises((ValueError, RuntimeError)):
            tok.geneformer_style(
                cpu_tensor([[2.0, 1.0]]), cpu_tensor([3.0]),
                cpu_tensor([1.0, 0.0]), cpu_tensor([1, 2], torch.int64),
                vocab, 2,
            )

    def test_geneformer_ignores_unretained_reference_factors(self):
        vocab = make_vocab(
            [1], [101], policy=tok.UnknownIdentityPolicy.drop
        )
        result = tok.geneformer_style(
            cpu_tensor([[2.0, 1.0]]), cpu_tensor([3.0]),
            cpu_tensor([1.0, 0.0]), cpu_tensor([1, 99], torch.int64),
            vocab, 2,
        )
        self.assertEqual(ragged(result), [[101]])

    def test_nonzero_feature_median_batches_even_and_empty_columns(self):
        a = cpu_tensor([[0.0, 1.0, 8.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
        b = cpu_tensor([[5.0, 9.0, 0.0, 0.0]])
        result = tok.nonzero_feature_median([a, b], feature_block_size=2)
        torch.testing.assert_close(result.medians[:3], cpu_tensor([4.0, 5.0, 8.0]))
        torch.testing.assert_close(result.nonzero_counts, cpu_tensor([2, 2, 1, 0]))
        torch.testing.assert_close(result.valid, cpu_tensor([True, True, True, False]))
        self.assertTrue(torch.isnan(result.medians[3]))

    def test_nonzero_feature_median_sparse_matches_dense(self):
        dense = cpu_tensor([[0.0, 2.0, 5.0], [4.0, 0.0, 0.0], [2.0, 6.0, 0.0]])
        sparse = dense.to_sparse_csr()
        dense_stats = tok.nonzero_feature_median([dense], feature_block_size=1)
        sparse_stats = tok.nonzero_feature_median([sparse], feature_block_size=1)
        torch.testing.assert_close(sparse_stats.medians, dense_stats.medians)
        torch.testing.assert_close(sparse_stats.nonzero_counts, dense_stats.nonzero_counts)

    def test_nonzero_feature_median_rejects_nonfinite_and_noncanonical_csr(self):
        with self.assertRaises((ValueError, RuntimeError)):
            tok.nonzero_feature_median([cpu_tensor([[1.0, float("nan")]])])
        # CSR columns are deliberately descending within the row, so entries
        # are not canonical even though their shape and indices are in range.
        noncanonical = torch.sparse_csr_tensor(
            cpu_tensor([0, 2], torch.int64), cpu_tensor([1, 0], torch.int64),
            cpu_tensor([2.0, 1.0]), size=(1, 2), check_invariants=False,
        )
        with self.assertRaises((ValueError, RuntimeError)):
            tok.nonzero_feature_median([noncanonical], feature_block_size=1)

    def test_nonzero_feature_median_sparse_wide_blocked_shape(self):
        # A wide sparse shape exercises exact feature blocking while keeping
        # the expected result independently computable from the handful of
        # stored entries.
        rows, columns = 64, 4096
        row_ids = np.arange(rows, dtype=np.int64)
        col_ids = (row_ids * 61 + 7) % columns
        values = np.arange(1, rows + 1, dtype=np.float32)
        dense = np.zeros((rows, columns), dtype=np.float32)
        dense[row_ids, col_ids] = values
        csr = cpu_tensor(dense).to_sparse_csr()
        stats = tok.nonzero_feature_median([csr], feature_block_size=31)
        expected = np.full((columns,), np.nan, dtype=np.float32)
        counts = np.zeros((columns,), dtype=np.int64)
        for column in np.unique(col_ids):
            present = values[col_ids == column]
            expected[column] = np.median(present)
            counts[column] = present.size
        np.testing.assert_allclose(stats.medians.numpy(), expected, equal_nan=True)
        np.testing.assert_array_equal(stats.nonzero_counts.numpy(), counts)

    def test_pack_pad_round_trip_preserves_attributes(self):
        vocab = make_vocab([2, 4, 8])
        ids = cpu_tensor([4, 2, 8], torch.int64)
        offsets = cpu_tensor([0, 2, 2, 3], torch.int64)
        attr = cpu_tensor([[1, 2], [3, 4], [5, 6]], torch.int32)
        batch = tok.event_stream(ids, offsets, vocab, {"coordinate": attr})
        padded = tok.pad_sequences(batch, 0)
        self.assertEqual(padded.token_ids.shape, (3, 2))
        torch.testing.assert_close(padded.valid.sum(1), cpu_tensor([2, 0, 1]))
        repacked = tok.pack_padded(padded.token_ids, padded.lengths, padded.attributes)
        self.assertEqual(ragged(repacked), ragged(batch))
        torch.testing.assert_close(repacked.attributes["coordinate"], attr)

    def test_caller_controlled_boundary_tokens_and_attributes(self):
        vocab = make_vocab([1, 2])
        batch = tok.event_stream(
            cpu_tensor([1, 2], torch.int64), cpu_tensor([0, 1, 1, 2], torch.int64),
            vocab, {"time": cpu_tensor([4.0, 9.0])},
        )
        prefix = tok.TokenBatch(cpu_tensor([7, 7, 7], torch.int64),
                                cpu_tensor([0, 1, 2, 3], torch.int64),
                                {"time": cpu_tensor([-1.0, -1.0, -1.0])})
        suffix = tok.TokenBatch(cpu_tensor([8, 8, 8], torch.int64),
                                cpu_tensor([0, 1, 2, 3], torch.int64),
                                {"time": cpu_tensor([10.0, 10.0, 10.0])})
        result = tok.add_boundaries(batch, prefix, suffix)
        self.assertEqual(ragged(result), [[7, 1, 8], [7, 8], [7, 2, 8]])
        torch.testing.assert_close(
            result.attributes["time"], cpu_tensor([-1.0, 4.0, 10.0,
                                                    -1.0, 10.0,
                                                    -1.0, 9.0, 10.0]),
        )

    def test_invalid_offsets_and_attribute_extent_rejected(self):
        vocab = make_vocab([1, 2])
        with self.assertRaises((ValueError, RuntimeError)):
            tok.event_stream(cpu_tensor([1, 2]), cpu_tensor([0, 2, 1]), vocab)
        with self.assertRaises((ValueError, RuntimeError)):
            tok.event_stream(cpu_tensor([1, 2]), cpu_tensor([0, 2]), vocab,
                             {"bad": cpu_tensor([1])})

    def test_buffer_append_subset_overflow_reserve_and_compact(self):
        vocab = make_vocab([1, 2, 3])
        storage = tok.SequenceBuffer.create([2, 1, 0], {"x": cpu_tensor([0.0])})
        first = tok.event_stream(
            cpu_tensor([1, 2]), cpu_tensor([0, 1, 2]), vocab,
            {"x": cpu_tensor([1.0, 2.0])},
        )
        storage.append_checked(cpu_tensor([0, 1], torch.int64), first)
        original_data_ptr = storage.token_storage.data_ptr()
        second = tok.event_stream(
            cpu_tensor([3, 2]), cpu_tensor([0, 1, 2]), vocab,
            {"x": cpu_tensor([3.0, 4.0])},
        )
        status = storage.append_into(cpu_tensor([0, 1], torch.int64), second)
        torch.testing.assert_close(status.succeeded, cpu_tensor([True, False]))
        self.assertEqual(storage.token_storage.data_ptr(), original_data_ptr)
        compact = storage.compact()
        self.assertEqual(ragged(compact), [[1, 3], [2], []])
        torch.testing.assert_close(compact.attributes["x"], cpu_tensor([1., 3., 2.]))
        grown = storage.reserve([4, 3, 1])
        second_one = tok.event_stream(
            cpu_tensor([2]), cpu_tensor([0, 1]), vocab,
            {"x": cpu_tensor([4.0])},
        )
        status = grown.append_into(cpu_tensor([1], torch.int64), second_one)
        torch.testing.assert_close(status.succeeded, cpu_tensor([True]))
        self.assertEqual(ragged(grown.compact()), [[1, 3], [2, 2], []])

    def test_buffer_rejects_gradient_tracked_payload(self):
        storage = tok.SequenceBuffer.create([2], {"x": cpu_tensor([0.])})
        vocab = make_vocab([1])
        updates = tok.event_stream(
            cpu_tensor([1]), cpu_tensor([0, 1]), vocab,
            {"x": cpu_tensor([2.0], torch.float32).requires_grad_()},
        )
        with self.assertRaises((ValueError, RuntimeError)):
            storage.append_into(cpu_tensor([0], torch.int64), updates)


@unittest.skipUnless(
    os.environ.get("CELLERATOR_TOKENIZATION_TEST_CUDA") == "1",
    "CUDA tests require explicit Device Scheduler admission and opt-in",
)
class TokenizationCudaTests(unittest.TestCase):
    def test_cuda_unspecified_device_uses_current_device(self):
        actual = torch.device("cuda", torch.cuda.current_device())
        storage = tok.SequenceBuffer.create(
            [2], {"value": torch.empty(0, device="cuda")}, device="cuda"
        )
        updates = tok.TokenBatch(
            torch.tensor([7], dtype=torch.int64, device="cuda"),
            torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
            {"value": torch.tensor([2.0], device="cuda")},
        )
        storage.append_checked(
            torch.tensor([0], dtype=torch.int64, device="cuda"), updates
        )
        self.assertEqual(storage.token_storage.device, actual)
        self.assertEqual(ragged(storage.compact()), [[7]])

    def test_current_stream_rank_and_append_preserves_temporary_lifetimes(self):
        self.assertTrue(torch.cuda.is_available(), "opted into CUDA but no CUDA device is available")
        device = torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            vocab = tok.make_vocabulary(
                torch.tensor([1, 2, 3], device=device),
                torch.tensor([11, 22, 33], device=device),
            )
            result = tok.ranked_feature(
                torch.tensor([[2.0, 0.0, 5.0]], device=device),
                torch.tensor([1, 2, 3], device=device), vocab,
            )
            self.assertEqual(result.token_ids.device, device)
            storage = tok.SequenceBuffer.create(
                [2, 2], {"time": torch.zeros(1, device=device)}, str(device)
            )
            row_ids = torch.tensor([1, 2], dtype=torch.int64, device=device)
            row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
            attributes = {"time": torch.tensor([3.0, 7.0], device=device)}
            encoded = tok.event_stream(row_ids, row_offsets, vocab, attributes)
            status = storage.append_into(
                torch.tensor([0, 1], dtype=torch.int64, device=device), encoded
            )
            del row_ids, row_offsets, attributes, encoded
            compact = storage.compact()
        stream.synchronize()
        self.assertEqual(ragged(result), [[33, 11]])
        self.assertEqual(ragged(compact), [[11], [22]])
        torch.testing.assert_close(
            compact.attributes["time"].cpu(), torch.tensor([3.0, 7.0])
        )
        torch.testing.assert_close(status.succeeded.cpu(), torch.tensor([True, True]))

    def test_cuda_rank_profile_and_medians_match_cpu_references(self):
        device = torch.device("cuda", torch.cuda.current_device())
        values_cpu = np.array([[3.0, 0.0, 1.0], [0.0, 6.0, 2.0]], dtype=np.float64)
        features_cpu = np.array([40, 10, 30], dtype=np.int64)
        vocab = tok.make_vocabulary(
            torch.tensor([10, 30, 40], dtype=torch.int64, device=device),
            torch.tensor([5, 7, 9], dtype=torch.int64, device=device),
        )
        values = torch.tensor(values_cpu, device=device)
        features = torch.tensor(features_cpu, dtype=torch.int64, device=device)
        dense = tok.ranked_feature(values, features, vocab)
        sparse = tok.ranked_feature(values.to_sparse_csr(), features, vocab)
        self.assertEqual(ragged(dense), [[9, 7], [5, 7]])
        self.assertEqual(ragged(sparse), [[9, 7], [5, 7]])

        totals = np.array([4.0, 8.0], dtype=np.float64)
        medians = np.array([2.0, 3.0, 1.0], dtype=np.float64)
        profile = tok.geneformer_style(
            values, torch.tensor(totals, device=device),
            torch.tensor(medians, device=device), features, vocab, 2,
        )
        scores = values_cpu / totals[:, None] * 10_000.0 / medians[None, :]
        expected = []
        for row, count_row in enumerate(values_cpu):
            present = np.flatnonzero(count_row > 0)
            order = sorted(present, key=lambda j: (-scores[row, j], features_cpu[j]))[:2]
            token_by_identity = {10: 5, 30: 7, 40: 9}
            expected.append([token_by_identity[int(features_cpu[j])] for j in order])
        self.assertEqual(ragged(profile), expected)

        stats = tok.nonzero_feature_median(
            [values.to_sparse_csr()], feature_block_size=2
        )
        np.testing.assert_allclose(stats.medians.cpu().numpy(), [3.0, 6.0, 1.5])
        np.testing.assert_array_equal(stats.nonzero_counts.cpu().numpy(), [1, 1, 2])
        self.assertTrue(bool(stats.valid.all().item()))

    def test_cuda_append_subset_overflow_reserve_and_attributes(self):
        device = torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            vocab = tok.make_vocabulary(
                torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device),
                torch.tensor([11, 22, 33, 44], dtype=torch.int64, device=device),
            )
            buffer = tok.SequenceBuffer.create(
                [2, 1, 2], {"position": torch.empty((0, 2), device=device)}, str(device)
            )
            token_ptr = buffer.token_storage.data_ptr()
            attribute_ptr = buffer.attribute_storage["position"].data_ptr()
            status = torch.empty((2,), dtype=torch.bool, device=device)
            selected = torch.tensor([0, 2], dtype=torch.int64, device=device)

            first = tok.event_stream(
                torch.tensor([1, 3, 4], dtype=torch.int64, device=device),
                torch.tensor([0, 1, 3], dtype=torch.int64, device=device), vocab,
                {"position": torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device=device)},
            )
            first_status = buffer.append_into(selected, first, status)
            torch.testing.assert_close(first_status.succeeded, torch.tensor([True, True], device=device))
            self.assertEqual(first_status.succeeded.data_ptr(), status.data_ptr())

            second = tok.event_stream(
                torch.tensor([2, 1], dtype=torch.int64, device=device),
                torch.tensor([0, 1, 2], dtype=torch.int64, device=device), vocab,
                {"position": torch.tensor([[7., 8.], [9., 10.]], device=device)},
            )
            second_status = buffer.append_into(selected, second, status)
            torch.testing.assert_close(second_status.succeeded, torch.tensor([True, False], device=device))
            self.assertEqual(second_status.succeeded.data_ptr(), status.data_ptr())
            self.assertEqual(buffer.token_storage.data_ptr(), token_ptr)
            self.assertEqual(buffer.attribute_storage["position"].data_ptr(), attribute_ptr)
            torch.testing.assert_close(buffer.lengths, torch.tensor([2, 0, 2], device=device))
            compact = buffer.compact()
            self.assertEqual(ragged(compact), [[11, 22], [], [33, 44]])
            torch.testing.assert_close(
                compact.attributes["position"],
                torch.tensor([[1., 2.], [7., 8.], [3., 4.], [5., 6.]], device=device),
            )

            grown = buffer.reserve([4, 1, 3])
            last = tok.event_stream(
                torch.tensor([1, 2], dtype=torch.int64, device=device),
                torch.tensor([0, 1, 2], dtype=torch.int64, device=device), vocab,
                {"position": torch.tensor([[11., 12.], [13., 14.]], device=device)},
            )
            final_status = grown.append_into(selected, last)
            torch.testing.assert_close(final_status.succeeded, torch.tensor([True, True], device=device))
            final = grown.compact()
        stream.synchronize()
        self.assertEqual(ragged(final), [[11, 22, 11], [], [33, 44, 22]])
        torch.testing.assert_close(
            final.attributes["position"],
            torch.tensor([[1., 2.], [7., 8.], [11., 12.],
                          [3., 4.], [5., 6.], [13., 14.]], device=device),
        )

    def test_cuda_invalid_offsets_and_corrupt_length_do_not_write(self):
        device = torch.device("cuda", torch.cuda.current_device())
        vocab = tok.make_vocabulary(
            torch.tensor([1, 2], dtype=torch.int64, device=device),
            torch.tensor([11, 22], dtype=torch.int64, device=device),
        )
        selected = torch.tensor([0, 1], dtype=torch.int64, device=device)
        storage = tok.SequenceBuffer.create(
            [3, 3], {"time": torch.empty(0, device=device)}, str(device)
        )
        storage.token_storage.fill_(123)
        storage.attribute_storage["time"].fill_(123.0)
        valid = tok.event_stream(
            torch.tensor([1, 2], dtype=torch.int64, device=device),
            torch.tensor([0, 1, 2], dtype=torch.int64, device=device), vocab,
            {"time": torch.tensor([1.0, 2.0], device=device)},
        )
        storage.append_checked(selected, valid)
        before_tokens = storage.token_storage.clone()
        before_attrs = storage.attribute_storage["time"].clone()
        before_lengths = storage.lengths.clone()
        malformed = tok.TokenBatch(
            torch.tensor([1], dtype=torch.int64, device=device),
            torch.tensor([0, 1, 0], dtype=torch.int64, device=device),
            {"time": torch.tensor([8.0], device=device)},
        )
        status_buffer = torch.empty((2,), dtype=torch.bool, device=device)
        malformed_result = storage.append_into(selected, malformed, status_buffer)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(malformed_result.succeeded, torch.tensor([False, False], device=device))
        torch.testing.assert_close(storage.token_storage, before_tokens)
        torch.testing.assert_close(storage.attribute_storage["time"], before_attrs)
        torch.testing.assert_close(storage.lengths, before_lengths)

        corrupt = tok.SequenceBuffer.create([1, 1], device=str(device))
        corrupt.token_storage.fill_(123)
        corrupt_before = corrupt.token_storage.clone()
        corrupt.lengths[0] = 2
        one = tok.event_stream(
            torch.tensor([1], dtype=torch.int64, device=device),
            torch.tensor([0, 1, 1], dtype=torch.int64, device=device), vocab,
        )
        corrupt_status = corrupt.append_into(selected, one)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(corrupt_status.succeeded, torch.tensor([False, True], device=device))
        torch.testing.assert_close(corrupt.lengths, torch.tensor([2, 0], device=device))
        torch.testing.assert_close(corrupt.token_storage, corrupt_before)

    def test_cuda_side_stream_wait_and_append_input_lifetime(self):
        device = torch.device("cuda", torch.cuda.current_device())
        producer = torch.cuda.Stream(device=device)
        consumer = torch.cuda.Stream(device=device)
        ready = torch.cuda.Event()
        complete = torch.cuda.Event()
        with torch.cuda.stream(producer):
            vocab = tok.make_vocabulary(
                torch.tensor([1, 2], dtype=torch.int64, device=device),
                torch.tensor([11, 22], dtype=torch.int64, device=device),
            )
            identities = torch.tensor([1, 2], dtype=torch.int64, device=device)
            offsets = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
            times = torch.tensor([17.0, 23.0], device=device)
            ready.record(producer)

        with torch.cuda.stream(consumer):
            consumer.wait_event(ready)
            storage = tok.SequenceBuffer.create(
                [1, 1], {"time": torch.empty(0, device=device)}, str(device)
            )
            selected = torch.tensor([0, 1], dtype=torch.int64, device=device)
            encoded = tok.event_stream(identities, offsets, vocab, {"time": times})
            result = storage.append_into(selected, encoded)
            del identities, offsets, times, encoded, selected
            # Allocate on the original producer stream after dropping the
            # append inputs. This pressures the allocator to reuse their
            # blocks while the consumer may still be reading them.
            with torch.cuda.stream(producer):
                pressure = [torch.empty((1 << 18,), device=device) for _ in range(8)]
                for tensor in pressure:
                    tensor.fill_(5.0)
            packed = storage.compact()
            complete.record(consumer)

        complete.synchronize()
        producer.synchronize()
        torch.testing.assert_close(result.succeeded, torch.tensor([True, True], device=device))
        self.assertEqual(ragged(packed), [[11], [22]])
        torch.testing.assert_close(packed.attributes["time"], torch.tensor([17.0, 23.0], device=device))


if __name__ == "__main__":
    unittest.main()
