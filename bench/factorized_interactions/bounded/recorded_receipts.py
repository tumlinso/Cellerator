"""Bind retained Autobench pointers to completed Device Scheduler receipts.

This module is read-only.  It verifies recorded identities; it neither submits
work nor treats process environment markers as allocation authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from collections.abc import Mapping
from typing import Any


class ReceiptValidationError(ValueError):
    """Recorded evidence is incomplete, inconsistent, or not successful."""


_GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptValidationError(message)


def _string(value: Any, field: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"missing or invalid {field}")
    return value


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), "evidence path is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptValidationError("evidence file is not valid JSON") from exc
    _require(isinstance(value, dict), "evidence root must be an object")
    return value, raw


def _scheduler_unit(attempt_id: str) -> str:
    """Match Device Scheduler's systemd adapter attempt_unit_name()."""
    digest = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:32]
    return f"device-scheduler-attempt-{digest}.service"


def _device_record(records: Any, resource_id: str, expected_uuid: str, label: str) -> None:
    _require(isinstance(records, list), f"{label} device identity is missing")
    gpu_records = [row for row in records if isinstance(row, Mapping) and
                   str(row.get("kind", "")).casefold() == "gpu"]
    _require(len(gpu_records) == 1, f"{label} must identify exactly one granted GPU")
    record = gpu_records[0]
    _require(record.get("resource_id") == resource_id and record.get("uuid") == expected_uuid,
             f"{label} GPU identity differs from the Scheduler grant")


def _unwrap_receipt(value: dict[str, Any], context: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Return decoded result_json and its independently bound Autobench ID."""
    # The operator's immutable export wraps the official `results --details`
    # row.  The row's `id` is Autobench's attempt ID; it is not Scheduler's ID.
    if isinstance(value.get("autobench_results_row"), Mapping):
        value = dict(value["autobench_results_row"])

    autobench_id: Any = None
    if "result_json" in value:
        row = value
        result_json = row.get("result_json")
        _require(isinstance(result_json, str), "Autobench result_json must be serialized JSON")
        try:
            decoded = json.loads(result_json)
        except json.JSONDecodeError as exc:
            raise ReceiptValidationError("Autobench result_json is invalid") from exc
        _require(isinstance(decoded, dict), "Autobench result_json must decode to an object")
        autobench_id = row.get("id")
        _require(str(row.get("state", "")).casefold() == "succeeded",
                 "Autobench result row is not successful")
        row_job_id = row.get("job_id")
        if row_job_id is not None:
            _require(isinstance(row_job_id, str) and row_job_id,
                     "Autobench result row has an invalid Scheduler job ID")
        value = decoded
    else:
        decoded = value
        autobench_id = decoded.get("autobench_attempt_id")
        if autobench_id is None:
            autobench_id = context.get("autobench_attempt_id")

    _require(isinstance(value.get("job"), Mapping), "Scheduler job receipt is missing")
    _require(isinstance(value.get("worker"), Mapping), "Autobench worker receipt is missing")
    _require(isinstance(value.get("identity"), Mapping), "Autobench identity is missing")
    autobench_id = _string(autobench_id, "Autobench attempt ID")
    return value, autobench_id


def _unwrap_pointer(value: dict[str, Any]) -> dict[str, Any]:
    wrapped = value.get("current_pointer")
    if isinstance(wrapped, Mapping):
        return dict(wrapped)
    return value


def _gpu_grant(job: Mapping[str, Any]) -> tuple[str, str]:
    attempt = job.get("attempt")
    _require(isinstance(attempt, Mapping), "Scheduler attempt is missing")
    execution = attempt.get("execution_context")
    _require(isinstance(execution, Mapping), "Scheduler execution grant is missing")
    allocations = execution.get("allocations")
    resources = execution.get("resources")
    _require(isinstance(allocations, list) and isinstance(resources, list),
             "Scheduler execution grant is incomplete")

    allocation_ids = [row.get("resource_id") for row in allocations if isinstance(row, Mapping)]
    resource_ids = [row.get("resource_id") for row in resources if isinstance(row, Mapping)]
    _require(len(allocation_ids) == len(allocations) and len(resource_ids) == len(resources),
             "Scheduler execution grant contains malformed records")
    _require(all(isinstance(item, str) and bool(item.strip())
                 for item in (*allocation_ids, *resource_ids)),
             "Scheduler execution grant contains invalid resource IDs")
    _require(len(set(allocation_ids)) == len(allocation_ids) and
             len(set(resource_ids)) == len(resource_ids) and set(allocation_ids) == set(resource_ids),
             "Scheduler allocations and descriptors do not match by resource identity")

    gpu_resources = [row for row in resources if str(row.get("kind", "")).casefold() == "gpu"]
    _require(len(gpu_resources) == 1, "Scheduler execution grant must contain exactly one GPU")
    resource = gpu_resources[0]
    resource_id = _string(resource.get("resource_id"), "GPU resource ID")
    _require(resource_id in allocation_ids, "GPU resource descriptor has no matching allocation")
    attributes = resource.get("attributes")
    _require(isinstance(attributes, Mapping), "GPU resource attributes are missing")
    device_uuid = _string(attributes.get("uuid"), "granted GPU UUID")
    _require(_GPU_UUID.fullmatch(device_uuid) is not None, "Scheduler grant has an invalid GPU UUID")

    spec = job.get("spec")
    if isinstance(spec, Mapping) and "resources" in spec:
        requested = spec.get("resources")
        _require(isinstance(requested, list), "Scheduler requested resources are malformed")
        gpu_requests = [row for row in requested if isinstance(row, Mapping) and
                        str(row.get("kind", "")).casefold() == "gpu"]
        _require(all(isinstance(row, Mapping) for row in requested),
                 "Scheduler requested resources contain malformed records")
        if gpu_requests:
            _require(len(gpu_requests) == 1,
                     "single-GPU receipt has an ambiguous GPU request")
            selectors = gpu_requests[0].get("selectors", {})
            _require(isinstance(selectors, Mapping), "GPU request selectors are malformed")
            requested_id = selectors.get("resource_id")
            if requested_id is not None:
                _require(requested_id == resource_id,
                         "requested GPU resource differs from the Scheduler grant")

    return resource_id, device_uuid


def validate_receipt_and_pointer(
    receipt_path: Path,
    pointer_path: Path,
    context: Mapping[str, Any],
    expected_uuid: str,
) -> dict[str, Any]:
    """Validate a finished Autobench/Scheduler receipt and its current pointer.

    `receipt_path` may contain an official Autobench details row, its decoded
    `{job, worker, identity}` result object (with an explicit Autobench attempt
    ID in that object or `context`), or the operator's
    `{autobench_results_row, current_pointer, ...}` export.  `pointer_path` may
    be a standalone `current.json` or that same wrapper export.
    """
    _require(isinstance(context, Mapping), "observed execution context is missing")
    _require(isinstance(expected_uuid, str) and _GPU_UUID.fullmatch(expected_uuid) is not None,
             "expected UUID must be one explicit GPU UUID")

    receipt_root, receipt_bytes = _read_json(Path(receipt_path))
    pointer_root, pointer_bytes = _read_json(Path(pointer_path))
    receipt, autobench_attempt_id = _unwrap_receipt(receipt_root, context)
    pointer = _unwrap_pointer(pointer_root)

    job = receipt["job"]
    worker = receipt["worker"]
    identity = receipt["identity"]
    attempt = job.get("attempt")
    _require(isinstance(attempt, Mapping), "Scheduler attempt is missing")

    scheduler_job_id = _string(job.get("id", job.get("job_id")), "Scheduler job ID")
    scheduler_attempt_id = _string(attempt.get("attempt_id"), "Scheduler attempt ID")
    if job.get("id") is not None and job.get("job_id") is not None:
        _require(job["id"] == job["job_id"], "Scheduler job identifiers disagree")
    if attempt.get("job_id") is not None:
        _require(attempt["job_id"] == scheduler_job_id, "Scheduler attempt belongs to a different job")
    _require(str(job.get("state", "")).casefold() in {"succeeded", "completed"},
             "Scheduler job is not successful")
    if attempt.get("state") is not None:
        _require(str(attempt.get("state")).casefold() in {"succeeded", "completed"},
                 "Scheduler attempt is not successful")
    _require(job.get("cleanup_verified") is True and attempt.get("cleanup_verified") is True,
             "Scheduler job and attempt cleanup are not both verified")

    _require(str(worker.get("status", "")).casefold() == "succeeded",
             "Autobench worker is not successful")
    returncode = worker.get("returncode")
    _require(isinstance(returncode, int) and not isinstance(returncode, bool) and returncode == 0,
             "Autobench worker did not return zero")
    _require(worker.get("scheduler_job_id") == scheduler_job_id and
             worker.get("scheduler_attempt_id") == scheduler_attempt_id,
             "Autobench worker Scheduler identity differs from its receipt")

    if isinstance(receipt_root.get("autobench_results_row"), Mapping):
        row = receipt_root["autobench_results_row"]
    elif "result_json" in receipt_root:
        row = receipt_root
    else:
        row = None
    if row is not None and row.get("job_id") is not None:
        _require(row.get("job_id") == scheduler_job_id,
                 "Autobench row Scheduler job ID differs from its receipt")

    _require(context.get("status") == "observed", "outer execution context is not an observed process record")
    _require(context.get("scheduler_job_id") == scheduler_job_id and
             context.get("scheduler_attempt_id") == scheduler_attempt_id,
             "observed process context Scheduler identity differs from the receipt")
    _require(context.get("device_uuid") == expected_uuid,
             "observed process context GPU differs from the expected UUID")

    unit_name = _scheduler_unit(scheduler_attempt_id)
    cgroup = _string(context.get("cgroup"), "observed cgroup")
    observed_unit = _string(context.get("unit_name"), "observed Scheduler unit")
    _require(observed_unit == unit_name, "observed Scheduler unit does not match the attempt")
    cgroup_path = PurePosixPath(cgroup)
    _require(cgroup_path.is_absolute() and ".." not in cgroup_path.parts and
             cgroup_path.parts.count(unit_name) == 1,
             "observed cgroup does not identify the exact Scheduler attempt unit")

    handle = attempt.get("execution_handle")
    if handle is not None:
        _require(isinstance(handle, Mapping), "Scheduler execution handle is malformed")
        external_id = handle.get("external_id")
        if external_id is not None:
            _require(external_id == unit_name, "Scheduler execution handle unit differs from the attempt")
        metadata = handle.get("metadata")
        if metadata is not None:
            _require(isinstance(metadata, Mapping), "Scheduler execution handle metadata is malformed")
            control_group = metadata.get("control_group")
            if control_group is not None:
                _require(control_group == cgroup,
                         "Scheduler execution control group differs from the observed cgroup")
        direct_control_group = handle.get("control_group")
        if direct_control_group is not None:
            _require(direct_control_group == cgroup,
                     "Scheduler execution control group differs from the observed cgroup")

    resource_id, granted_uuid = _gpu_grant(job)
    _require(granted_uuid == expected_uuid, "Scheduler grant differs from the expected GPU UUID")
    _device_record(identity.get("device_identity"), resource_id, granted_uuid,
                   "Autobench receipt")
    _device_record(pointer.get("device_identity"), resource_id, granted_uuid,
                   "Autobench pointer")

    _require(pointer.get("fingerprint") is not None and
             isinstance(pointer.get("fingerprint"), str) and bool(pointer["fingerprint"].strip()),
             "Autobench current pointer fingerprint is missing")
    if row is not None:
        row_fingerprint = row.get("fingerprint")
        _require(isinstance(row_fingerprint, str) and bool(row_fingerprint.strip()) and
                 row_fingerprint == pointer["fingerprint"],
                 "Autobench result row fingerprint differs from its current pointer")
    _require(pointer.get("attempt") == autobench_attempt_id,
             "Autobench pointer attempt does not match its result row")
    _require(pointer.get("scheduler_job_id") == scheduler_job_id,
             "Autobench pointer Scheduler job ID differs from its receipt")
    _require(pointer.get("cleanup_verified") is True,
             "Autobench current pointer does not attest verified cleanup")
    _require(pointer.get("source_fingerprint") == identity.get("source_fingerprint") and
             isinstance(identity.get("source_fingerprint"), str) and
             bool(identity.get("source_fingerprint")),
             "Autobench source fingerprint differs from its receipt")
    _require(pointer.get("build_fingerprint") == identity.get("build_fingerprint") and
             isinstance(identity.get("build_fingerprint"), str) and
             bool(identity.get("build_fingerprint")),
             "Autobench build fingerprint differs from its receipt")
    build_output_identity = _string(pointer.get("build_output_identity"),
                                    "Autobench build output identity")

    pointer_result = pointer.get("result")
    _require(isinstance(pointer_result, Mapping), "Autobench pointer worker result is missing")
    for field in ("status", "returncode", "scheduler_job_id", "scheduler_attempt_id"):
        _require(pointer_result.get(field) == worker.get(field),
                 "Autobench pointer worker identity differs from its receipt")

    return {
        "scheduler_job_id": scheduler_job_id,
        "scheduler_attempt_id": scheduler_attempt_id,
        "autobench_attempt_id": autobench_attempt_id,
        "device_uuid": granted_uuid,
        "cgroup": cgroup,
        "unit_name": unit_name,
        "source_fingerprint": identity["source_fingerprint"],
        "build_fingerprint": identity["build_fingerprint"],
        "build_output_identity": build_output_identity,
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "pointer_sha256": hashlib.sha256(pointer_bytes).hexdigest(),
    }
