"""CPU checks for the prepared Torch record identity adapter."""

from __future__ import annotations

import unittest

import numpy as np

from prepared_torch_records import prepared_factorized_records
from workloads import make_workload


class PreparedTorchRecordTests(unittest.TestCase):
    def test_stable_ids_are_nonzero_and_logical_indices_stay_zero_based(self):
        problem, _ = make_workload(
            "correctness_ordered_tied_zero", np.float32, np.float32)
        records = prepared_factorized_records(problem)

        self.assertEqual(
            [record["logical_id"] for record in records],
            list(range(1, len(problem.mechanisms) + 1)),
        )
        self.assertEqual(records[0]["inputs"],
                         list(problem.mechanisms[0].participants))
        self.assertEqual(records[0]["target"], problem.mechanisms[0].target)
        self.assertEqual(records[0]["alpha"], problem.mechanisms[0].alpha_id)
        self.assertEqual(records[0]["bias"], problem.mechanisms[0].bias_id)
        self.assertEqual(records[0]["terms"], list(zip(
            problem.mechanisms[0].context_ids,
            problem.mechanisms[0].gamma_ids)))


if __name__ == "__main__":
    unittest.main()
