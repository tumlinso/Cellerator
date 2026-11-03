from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recorded_receipts import ReceiptValidationError, validate_receipt_and_pointer


GPU_UUID = "GPU-12345678-1234-5678-9abc-def012345678"
OTHER_GPU_UUID = "GPU-87654321-4321-8765-cba9-876543210fed"
RESOURCE_ID = f"gpu:test-host:{GPU_UUID}"
SCHEDULER_JOB_ID = "sched-job-test-01"
SCHEDULER_ATTEMPT_ID = "sched-attempt-test-01"
AUTOBENCH_ATTEMPT_ID = "autobench-attempt-test-01"
SOURCE_FINGERPRINT = "source-fingerprint-test"
BUILD_FINGERPRINT = "build-fingerprint-test"
BUILD_OUTPUT_IDENTITY = "build-output-identity-test"


def scheduler_unit(attempt_id: str) -> str:
    digest = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:32]
    return f"device-scheduler-attempt-{digest}.service"


def fixture() -> tuple[dict, dict, dict]:
    unit = scheduler_unit(SCHEDULER_ATTEMPT_ID)
    cgroup = f"/user.slice/user-1000.slice/user@1000.service/app.slice/{unit}"
    job = {
        "id": SCHEDULER_JOB_ID,
        "state": "succeeded",
        "cleanup_verified": True,
        "attempt": {
            "attempt_id": SCHEDULER_ATTEMPT_ID,
            "job_id": SCHEDULER_JOB_ID,
            "state": "succeeded",
            "cleanup_verified": True,
            "execution_context": {
                "allocations": [{"resource_id": RESOURCE_ID, "quantities": {"memory_bytes": 1}}],
                "resources": [{
                    "resource_id": RESOURCE_ID,
                    "host_id": "test-host",
                    "kind": "gpu",
                    "attributes": {"uuid": GPU_UUID},
                }],
            },
            "execution_handle": {
                "external_id": unit,
                "metadata": {"control_group": cgroup},
            },
        },
        "spec": {
            "resources": [{"kind": "gpu", "selectors": {"resource_id": RESOURCE_ID}}],
        },
    }
    worker = {
        "status": "succeeded",
        "returncode": 0,
        "scheduler_job_id": SCHEDULER_JOB_ID,
        "scheduler_attempt_id": SCHEDULER_ATTEMPT_ID,
    }
    device = [{"resource_id": RESOURCE_ID, "kind": "gpu", "uuid": GPU_UUID}]
    identity = {
        "source_fingerprint": SOURCE_FINGERPRINT,
        "build_fingerprint": BUILD_FINGERPRINT,
        "device_identity": device,
    }
    result = {"job": job, "worker": worker, "identity": identity}
    pointer = {
        "fingerprint": "pointer-fingerprint-test",
        "attempt": AUTOBENCH_ATTEMPT_ID,
        "result": dict(worker),
        "source_fingerprint": SOURCE_FINGERPRINT,
        "build_fingerprint": BUILD_FINGERPRINT,
        "build_output_identity": BUILD_OUTPUT_IDENTITY,
        "device_identity": device,
        "scheduler_job_id": SCHEDULER_JOB_ID,
        "cleanup_verified": True,
    }
    context = {
        "schema": "cellerator.factorized_interactions.outer_execution_context.v1",
        "status": "observed",
        "scheduler_job_id": SCHEDULER_JOB_ID,
        "scheduler_attempt_id": SCHEDULER_ATTEMPT_ID,
        "autobench_attempt_id": AUTOBENCH_ATTEMPT_ID,
        "device_uuid": GPU_UUID,
        "cgroup": cgroup,
        "unit_name": unit,
    }
    return result, pointer, context


class RecordedReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_json(self, name: str, value: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return path

    def assert_rejected(self, receipt: dict, pointer: dict, context: dict) -> None:
        row = receipt.get("autobench_results_row", receipt)
        if isinstance(row, dict) and "result_json" in row and "fingerprint" not in row:
            row["fingerprint"] = pointer["fingerprint"]
        receipt_path = self.write_json("receipt.json", receipt)
        pointer_path = self.write_json("pointer.json", pointer)
        with self.assertRaises(ReceiptValidationError):
            validate_receipt_and_pointer(receipt_path, pointer_path, context, GPU_UUID)

    def test_official_details_row_binds_receipt_pointer_and_exact_file_hashes(self) -> None:
        result, pointer, context = fixture()
        row = {
            "id": AUTOBENCH_ATTEMPT_ID,
            "job_id": SCHEDULER_JOB_ID,
            "fingerprint": pointer["fingerprint"],
            "state": "succeeded",
            "result_json": json.dumps(result, sort_keys=True),
        }
        receipt_path = self.write_json("receipt-row.json", row)
        pointer_path = self.write_json("current.json", pointer)

        observed = validate_receipt_and_pointer(receipt_path, pointer_path, context, GPU_UUID)

        self.assertEqual(observed["scheduler_job_id"], SCHEDULER_JOB_ID)
        self.assertEqual(observed["scheduler_attempt_id"], SCHEDULER_ATTEMPT_ID)
        self.assertEqual(observed["autobench_attempt_id"], AUTOBENCH_ATTEMPT_ID)
        self.assertEqual(observed["device_uuid"], GPU_UUID)
        self.assertEqual(observed["unit_name"], scheduler_unit(SCHEDULER_ATTEMPT_ID))
        self.assertEqual(observed["build_output_identity"], BUILD_OUTPUT_IDENTITY)
        self.assertEqual(observed["receipt_sha256"], hashlib.sha256(receipt_path.read_bytes()).hexdigest())
        self.assertEqual(observed["pointer_sha256"], hashlib.sha256(pointer_path.read_bytes()).hexdigest())

    def test_operator_wrapper_can_supply_receipt_and_current_pointer_together(self) -> None:
        result, pointer, context = fixture()
        wrapper = {
            "schema": "cellerator.factorized_interactions.finished_evidence.v1",
            "autobench_results_row": {
                "id": AUTOBENCH_ATTEMPT_ID,
                "job_id": SCHEDULER_JOB_ID,
                "fingerprint": pointer["fingerprint"],
                "state": "succeeded",
                "result_json": json.dumps(result, sort_keys=True),
            },
            "current_pointer": pointer,
            "report_path": "qualification-report.json",
            "report_sha256": "report-hash-bound-by-assembler",
        }
        wrapper_path = self.write_json("finished-evidence.json", wrapper)

        observed = validate_receipt_and_pointer(wrapper_path, wrapper_path, context, GPU_UUID)

        digest = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
        self.assertEqual(observed["receipt_sha256"], digest)
        self.assertEqual(observed["pointer_sha256"], digest)

    def test_raw_decoded_result_requires_independent_autobench_attempt_identity(self) -> None:
        result, pointer, context = fixture()
        receipt_path = self.write_json("raw.json", result)
        pointer_path = self.write_json("pointer.json", pointer)
        context_without_id = dict(context)
        context_without_id.pop("autobench_attempt_id")
        with self.assertRaisesRegex(ReceiptValidationError, "Autobench attempt ID"):
            validate_receipt_and_pointer(receipt_path, pointer_path, context_without_id, GPU_UUID)

    def test_raw_decoded_result_uses_context_attempt_identity(self) -> None:
        result, pointer, context = fixture()
        receipt_path = self.write_json("raw.json", result)
        pointer_path = self.write_json("pointer.json", pointer)
        self.assertEqual(
            validate_receipt_and_pointer(receipt_path, pointer_path, context, GPU_UUID)["autobench_attempt_id"],
            AUTOBENCH_ATTEMPT_ID,
        )

    def test_rejects_scheduler_or_autobench_attempt_mismatch(self) -> None:
        result, pointer, context = fixture()
        context["scheduler_attempt_id"] = "different-attempt"
        self.assert_rejected({
            "id": AUTOBENCH_ATTEMPT_ID, "job_id": SCHEDULER_JOB_ID, "state": "succeeded",
            "result_json": json.dumps(result),
        }, pointer, context)

        result, pointer, context = fixture()
        pointer["attempt"] = "different-autobench-attempt"
        self.assert_rejected({
            "id": AUTOBENCH_ATTEMPT_ID, "job_id": SCHEDULER_JOB_ID, "state": "succeeded",
            "result_json": json.dumps(result),
        }, pointer, context)

    def test_rejects_result_row_pointer_fingerprint_mismatch(self) -> None:
        result, pointer, context = fixture()
        row = {
            "id": AUTOBENCH_ATTEMPT_ID,
            "job_id": SCHEDULER_JOB_ID,
            "fingerprint": "stale-fingerprint",
            "state": "succeeded",
            "result_json": json.dumps(result),
        }
        self.assert_rejected(row, pointer, context)

    def test_rejects_unverified_job_or_attempt_cleanup(self) -> None:
        result, pointer, context = fixture()
        result["job"]["attempt"]["cleanup_verified"] = False
        self.assert_rejected({
            "id": AUTOBENCH_ATTEMPT_ID, "job_id": SCHEDULER_JOB_ID, "state": "succeeded",
            "result_json": json.dumps(result),
        }, pointer, context)

        result, pointer, context = fixture()
        result["job"]["cleanup_verified"] = False
        self.assert_rejected({
            "id": AUTOBENCH_ATTEMPT_ID, "job_id": SCHEDULER_JOB_ID, "state": "succeeded",
            "result_json": json.dumps(result),
        }, pointer, context)

    def test_rejects_wrong_unit_or_control_group(self) -> None:
        result, pointer, context = fixture()
        context["unit_name"] = "device-scheduler-attempt-wrong.service"
        self.assert_rejected({
            "id": AUTOBENCH_ATTEMPT_ID, "job_id": SCHEDULER_JOB_ID, "state": "succeeded",
            "result_json": json.dumps(result),
        }, pointer, context)

        result, pointer, context = fixture()
        result["job"]["attempt"]["execution_handle"]["metadata"]["control_group"] += "/other.service"
        self.assert_rejected({
            "id": AUTOBENCH_ATTEMPT_ID, "job_id": SCHEDULER_JOB_ID, "state": "succeeded",
            "result_json": json.dumps(result),
        }, pointer, context)

    def test_rejects_grant_selector_and_device_identity_mismatch(self) -> None:
        result, pointer, context = fixture()
        result["job"]["attempt"]["execution_context"]["resources"][0]["attributes"]["uuid"] = OTHER_GPU_UUID
        self.assert_rejected({
            "id": AUTOBENCH_ATTEMPT_ID, "job_id": SCHEDULER_JOB_ID, "state": "succeeded",
            "result_json": json.dumps(result),
        }, pointer, context)

        result, pointer, context = fixture()
        result["job"]["spec"]["resources"][0]["selectors"]["resource_id"] = "gpu:other-host:GPU-other"
        self.assert_rejected({
            "id": AUTOBENCH_ATTEMPT_ID, "job_id": SCHEDULER_JOB_ID, "state": "succeeded",
            "result_json": json.dumps(result),
        }, pointer, context)

        result, pointer, context = fixture()
        pointer["device_identity"][0]["resource_id"] = "gpu:other-host:GPU-other"
        self.assert_rejected({
            "id": AUTOBENCH_ATTEMPT_ID, "job_id": SCHEDULER_JOB_ID, "state": "succeeded",
            "result_json": json.dumps(result),
        }, pointer, context)

    def test_rejects_worker_and_pointer_result_disagreement(self) -> None:
        result, pointer, context = fixture()
        result["worker"]["scheduler_attempt_id"] = "other-scheduler-attempt"
        self.assert_rejected({
            "id": AUTOBENCH_ATTEMPT_ID, "job_id": SCHEDULER_JOB_ID, "state": "succeeded",
            "result_json": json.dumps(result),
        }, pointer, context)

        result, pointer, context = fixture()
        pointer["result"]["returncode"] = 7
        self.assert_rejected({
            "id": AUTOBENCH_ATTEMPT_ID, "job_id": SCHEDULER_JOB_ID, "state": "succeeded",
            "result_json": json.dumps(result),
        }, pointer, context)


if __name__ == "__main__":
    unittest.main()
