"""Versioned little-endian wire files for an out-of-process native runner."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

from reference import Mechanism, Problem, validate_problem


MAGIC_INPUT = b"CFINT001"
MAGIC_RESULT = b"CFOUT001"
HEADER = struct.Struct("<8s10Q3I")
MECHANISM = struct.Struct("<QQQQII4Q")
TERM = struct.Struct("<QQ")
TYPE_CODE = {np.dtype(np.float16): 1, np.dtype(np.float32): 2}
CODE_TYPE = {value: key for key, value in TYPE_CODE.items()}
MAX_CASE_SHARD_BYTES = 32 * 1024 * 1024


def _little_array(array: np.ndarray, dtype: np.dtype) -> bytes:
    value = np.ascontiguousarray(array, dtype=dtype)
    return value.tobytes(order="C")


def _case_header(problem: Problem, output_dtype: np.dtype | type) -> bytes:
    x_dtype = np.asarray(problem.x).dtype
    parameter_dtypes = tuple(np.asarray(a).dtype for a in
                             (problem.alpha, problem.bias, problem.gamma))
    if x_dtype not in TYPE_CODE:
        raise TypeError("x storage must be FP16 or FP32")
    if np.asarray(problem.z).dtype != x_dtype:
        raise TypeError("x and z storage dtypes must match in this wire protocol")
    if any(dtype != parameter_dtypes[0] for dtype in parameter_dtypes):
        raise TypeError("alpha, bias and gamma storage dtypes must match")
    if parameter_dtypes[0] not in TYPE_CODE:
        raise TypeError("parameter storage must be FP16 or FP32")
    output_dtype = np.dtype(output_dtype)
    if output_dtype not in TYPE_CODE:
        raise TypeError("output storage must be FP16 or FP32")
    total_terms = sum(len(m.context_ids) for m in problem.mechanisms)
    counts = (
        problem.x.shape[2], problem.x.shape[1], problem.x.shape[0],
        problem.z.shape[0], problem.target_count, problem.alpha.shape[0],
        problem.bias.size, problem.gamma.size, len(problem.mechanisms), total_terms,
    )
    return HEADER.pack(MAGIC_INPUT, *counts, TYPE_CODE[x_dtype],
                       TYPE_CODE[parameter_dtypes[0]], TYPE_CODE[output_dtype])


def _encode_case(problem: Problem, dy: np.ndarray,
                 output_dtype: np.dtype | type) -> bytes:
    """Encode the canonical CFINT001 byte stream independent of transport."""
    validate_problem(problem)
    dy = np.asarray(dy)
    expected_dy_shape = (problem.target_count, problem.x.shape[1], problem.x.shape[2])
    if dy.shape != expected_dy_shape or dy.dtype != np.float32:
        raise ValueError("dy must be an FP32 [target,channel,cell] array")
    chunks = [_case_header(problem, output_dtype)]
    for mechanism_id, mechanism in enumerate(problem.mechanisms):
        if len(mechanism.participants) not in (2, 3, 4):
            raise ValueError("mechanism arity must be 2, 3 or 4")
        participants = tuple(mechanism.participants) + (0,) * (4 - len(mechanism.participants))
        chunks.append(MECHANISM.pack(
            mechanism_id, mechanism.target, mechanism.alpha_id, mechanism.bias_id,
            len(mechanism.participants), len(mechanism.context_ids), *participants,
        ))
        for context_id, gamma_id in zip(mechanism.context_ids, mechanism.gamma_ids):
            chunks.append(TERM.pack(context_id, gamma_id))
    chunks.extend((
        _little_array(problem.x, np.dtype("<f2") if np.asarray(problem.x).dtype == np.float16
                      else np.dtype("<f4")),
        _little_array(problem.z, np.dtype("<f2") if np.asarray(problem.z).dtype == np.float16
                      else np.dtype("<f4")),
        _little_array(problem.alpha, np.dtype("<f2") if np.asarray(problem.alpha).dtype == np.float16
                      else np.dtype("<f4")),
        _little_array(problem.bias, np.dtype("<f2") if np.asarray(problem.bias).dtype == np.float16
                      else np.dtype("<f4")),
        _little_array(problem.gamma, np.dtype("<f2") if np.asarray(problem.gamma).dtype == np.float16
                      else np.dtype("<f4")),
        _little_array(dy, np.dtype("<f4")),
    ))
    return b"".join(chunks)


def write_case(path: Path, problem: Problem, dy: np.ndarray,
               output_dtype: np.dtype | type = np.float16) -> dict:
    """Write a native runner input and return its content identity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _encode_case(problem, dy, output_dtype)
    path.write_bytes(payload)
    return {"path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def verify_case_shards(identity: dict) -> None:
    """Verify the exact ordered CFINT001 byte stream and each shard's bound."""
    paths = [Path(path) for path in identity["parts"]]
    limit = identity["shard_bytes"]
    if not paths or len(paths) != len(set(paths)) or not 0 < limit <= MAX_CASE_SHARD_BYTES:
        raise ValueError("invalid case shard list or size limit")
    digest = hashlib.sha256()
    total = 0
    for index, path in enumerate(paths):
        size = path.stat().st_size
        if not 0 < size <= limit or (index + 1 < len(paths) and size != limit):
            raise ValueError("invalid case shard size")
        total += size
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    if total != identity["bytes"] or digest.hexdigest() != identity["sha256"]:
        raise ValueError("case shard stream differs from its frozen identity")


def write_sharded_case(path: Path, problem: Problem, dy: np.ndarray,
                       output_dtype: np.dtype | type = np.float16, *,
                       shard_bytes: int = MAX_CASE_SHARD_BYTES) -> dict:
    """Write CFINT001 unchanged in bounded ordered files and verify their bytes.

    A single small case retains its ordinary path and legacy runner interface.
    Larger cases use numbered shards, each no larger than 32 MiB. The identity
    hashes the concatenated stream, not a manifest or individual records.
    """
    if not isinstance(shard_bytes, int) or not 0 < shard_bytes <= MAX_CASE_SHARD_BYTES:
        raise ValueError("shard_bytes must be in 1..32 MiB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _encode_case(problem, dy, output_dtype)
    data = memoryview(payload)
    parts = []
    for index, offset in enumerate(range(0, len(payload), shard_bytes)):
        part = path if len(payload) <= shard_bytes else path.with_name(
            f"{path.name}.part{index:05d}")
        with part.open("wb") as stream:
            stream.write(data[offset:offset + shard_bytes])
        parts.append(str(part))
    identity = {"path": parts[0], "parts": parts, "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "shard_bytes": shard_bytes}
    verify_case_shards(identity)
    return identity


def read_case(path: Path) -> tuple[Problem, np.ndarray, np.dtype]:
    """Read a frozen case file and reconstruct semantic ordered records."""
    payload = memoryview(Path(path).read_bytes())
    if len(payload) < HEADER.size:
        raise ValueError("native case file lacks a complete header")
    fields = HEADER.unpack(payload[:HEADER.size])
    magic, *rest = fields
    if magic != MAGIC_INPUT:
        raise ValueError("native case has an unknown magic/version")
    counts = tuple(int(x) for x in rest[:10])
    activation_code, parameter_code, output_code = (int(x) for x in rest[10:])
    if activation_code not in CODE_TYPE or parameter_code not in CODE_TYPE \
            or output_code not in CODE_TYPE:
        raise ValueError("native case has an unknown storage dtype code")
    (cells, channels, occurrences, contexts, targets, alpha_rows, biases,
     gammas, mechanism_count, term_count) = counts
    activation_dtype = CODE_TYPE[activation_code].newbyteorder("<")
    parameter_dtype = CODE_TYPE[parameter_code].newbyteorder("<")
    offset = HEADER.size
    mechanisms = []
    terms_seen = 0
    for _ in range(mechanism_count):
        end = offset + MECHANISM.size
        if end > len(payload):
            raise ValueError("truncated native case mechanism table")
        (logical_id, target, alpha_id, bias_id, arity, local_term_count,
         p0, p1, p2, p3) = MECHANISM.unpack(payload[offset:end])
        offset = end
        if arity not in (2, 3, 4):
            raise ValueError("native case has unsupported mechanism arity")
        context_ids = []
        gamma_ids = []
        for _ in range(local_term_count):
            end = offset + TERM.size
            if end > len(payload):
                raise ValueError("truncated native case term table")
            context_id, gamma_id = TERM.unpack(payload[offset:end])
            offset = end
            context_ids.append(int(context_id))
            gamma_ids.append(int(gamma_id))
        terms_seen += local_term_count
        participants = (p0, p1, p2, p3)[:arity]
        # Logical IDs are required to be stable and unique by the host contract.
        if logical_id != len(mechanisms):
            raise ValueError("wire-format logical IDs must preserve record order")
        mechanisms.append(Mechanism(
            target=int(target), participants=tuple(int(x) for x in participants),
            context_ids=tuple(context_ids), gamma_ids=tuple(gamma_ids),
            bias_id=int(bias_id), alpha_id=int(alpha_id),
        ))
    if terms_seen != term_count:
        raise ValueError("native case term total differs from header")
    arrays = {}
    specifications = (
        ("x", (occurrences, channels, cells), activation_dtype),
        ("z", (contexts, cells), activation_dtype),
        ("alpha", (alpha_rows, channels), parameter_dtype),
        ("bias", (biases,), parameter_dtype),
        ("gamma", (gammas,), parameter_dtype),
        ("dy", (targets, channels, cells), np.dtype("<f4")),
    )
    for name, shape, dtype in specifications:
        arrays[name], offset = _read_array(payload, offset, shape, dtype)
    if offset != len(payload):
        raise ValueError("native case contains unaccounted trailing bytes")
    problem = Problem(arrays["x"], arrays["z"], arrays["alpha"],
                      arrays["bias"], arrays["gamma"], tuple(mechanisms), targets)
    validate_problem(problem)
    return problem, arrays["dy"], CODE_TYPE[output_code]


def write_result(path: Path, problem: Problem, y: np.ndarray, vjp,
                 output_dtype: np.dtype | type, repeat_vjp=None) -> dict:
    """Write native result files in tests and adapter tooling."""
    output_dtype = np.dtype(output_dtype)
    if output_dtype not in TYPE_CODE:
        raise TypeError("output storage must be FP16 or FP32")
    expected_y = (problem.target_count, problem.x.shape[1], problem.x.shape[2])
    if np.asarray(y).shape != expected_y:
        raise ValueError("output shape differs from the case")
    total_terms = sum(len(m.context_ids) for m in problem.mechanisms)
    counts = (
        problem.x.shape[2], problem.x.shape[1], problem.x.shape[0],
        problem.z.shape[0], problem.target_count, problem.alpha.shape[0],
        problem.bias.size, problem.gamma.size, len(problem.mechanisms), total_terms,
    )
    chunks = [HEADER.pack(MAGIC_RESULT, *counts, TYPE_CODE[output_dtype], 2,
                          2 if repeat_vjp is not None else 1),
              _little_array(y, output_dtype.newbyteorder("<"))]
    gradient_shapes = (("dx", problem.x.shape), ("dz", problem.z.shape),
                       ("dalpha", problem.alpha.shape), ("dbias", problem.bias.shape),
                       ("dgamma", problem.gamma.shape))
    for gradients in (vjp,) if repeat_vjp is None else (vjp, repeat_vjp):
        for name, expected in gradient_shapes:
            gradient = np.asarray(getattr(gradients, name))
            if gradient.shape != expected:
                raise ValueError(f"{name} shape differs from the case")
            chunks.append(_little_array(gradient, np.dtype("<f4")))
    payload = b"".join(chunks)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def _read_array(data: memoryview, offset: int, shape: tuple[int, ...],
                dtype: np.dtype) -> tuple[np.ndarray, int]:
    count = int(np.prod(shape, dtype=np.int64))
    end = offset + count * dtype.itemsize
    if end > len(data):
        raise ValueError("truncated native result file")
    array = np.frombuffer(data[offset:end], dtype=dtype, count=count).reshape(shape).copy()
    return array, end


def read_result(path: Path, expected_problem: Problem,
                expected_output_dtype: np.dtype | type) -> dict[str, np.ndarray]:
    """Read output plus five FP32 VJPs; reject shape/dtype/tail mismatches."""
    payload = memoryview(Path(path).read_bytes())
    if len(payload) < HEADER.size:
        raise ValueError("native result file lacks a complete header")
    fields = HEADER.unpack(payload[:HEADER.size])
    magic, *rest = fields
    if magic != MAGIC_RESULT:
        raise ValueError("native result has an unknown magic/version")
    counts = tuple(int(x) for x in rest[:10])
    type_codes = tuple(int(x) for x in rest[10:])
    expected_counts = (
        expected_problem.x.shape[2], expected_problem.x.shape[1],
        expected_problem.x.shape[0], expected_problem.z.shape[0],
        expected_problem.target_count, expected_problem.alpha.shape[0],
        expected_problem.bias.size, expected_problem.gamma.size,
        len(expected_problem.mechanisms),
        sum(len(m.context_ids) for m in expected_problem.mechanisms),
    )
    if counts != expected_counts:
        raise ValueError(f"native result extents {counts} differ from expected {expected_counts}")
    if type_codes[0] != TYPE_CODE[np.dtype(expected_output_dtype)]:
        raise ValueError("native result output dtype differs from the requested dtype")
    if type_codes[1] != 2 or type_codes[2] not in (1, 2):
        raise ValueError("native result must declare FP32 VJPs and one or two VJP sets")
    vjp_set_count = type_codes[2]
    output_type = CODE_TYPE[type_codes[0]].newbyteorder("<")
    offset = HEADER.size
    arrays = {}
    shapes = {
        "y": (expected_problem.target_count, expected_problem.x.shape[1],
             expected_problem.x.shape[2]),
        "dx": expected_problem.x.shape,
        "dz": expected_problem.z.shape,
        "dalpha": expected_problem.alpha.shape,
        "dbias": expected_problem.bias.shape,
        "dgamma": expected_problem.gamma.shape,
    }
    for name, shape in shapes.items():
        dtype = output_type if name == "y" else np.dtype("<f4")
        arrays[name], offset = _read_array(payload, offset, shape, dtype)
    if vjp_set_count == 2:
        for name in ("dx", "dz", "dalpha", "dbias", "dgamma"):
            arrays[f"{name}_repeat"], offset = _read_array(
                payload, offset, shapes[name], np.dtype("<f4"))
    if offset != len(payload):
        raise ValueError("native result contains unaccounted trailing bytes")
    return arrays
