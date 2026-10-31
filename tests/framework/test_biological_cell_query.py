"""Generic cell queries retain their shared program and live information."""
import unittest

from cellerator import Cell, Prediction, PredictionQuery


class _Program:
    def __call__(self, data):
        return data

    def query(self, data, request):
        self.last = (data, request)
        return Prediction(data, request.feature_ids, request.sample_ids)


class CellQueryTests(unittest.TestCase):
    def test_query_supplies_identity_without_copying_payload_or_request(self):
        source = Cell([1.0], id="cell-A")
        program = _Program()
        bound = source.bind(program)
        request = PredictionQuery("rna", feature_ids=("gene-A",))
        result = bound.query(request)
        self.assertIs(result.values, source.data)
        self.assertIsNone(request.sample_ids)
        self.assertEqual(result.sample_ids, ("cell-A",))
        source.data = [2.0]
        self.assertIs(bound.query(request).values, source.data)
        self.assertIs(bound.program, program)

    def test_explicit_window_identity_is_preserved(self):
        program = _Program()
        request = PredictionQuery("rna", sample_ids=("window-1",))
        Cell([1.0], id="cell-A").bind(program).query(request)
        self.assertIs(program.last[1], request)

    def test_missing_program_and_missing_query_support_fail_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "no program"):
            Cell().query(PredictionQuery("rna"))
        with self.assertRaisesRegex(TypeError, "biological queries"):
            Cell().bind(lambda value: value).query(PredictionQuery("rna"))


if __name__ == "__main__":
    unittest.main()
