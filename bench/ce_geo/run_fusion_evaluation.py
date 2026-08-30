#!/usr/bin/env python3
"""Build and evaluate the bounded CE-GEO-98 cover-native candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE = r'''
#include "bench/benchmark_mutex.hh"
#include "src/compute/architecture/providers/nvidia/sm70/exchange_cover_native_normalize.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;
namespace projection = cellerator::compute::projection;

namespace {
constexpr std::uint32_t partition_count = 256u;
constexpr std::uint32_t edges_per_partition = 16u;
constexpr std::uint32_t logical_count = partition_count * edges_per_partition;

void require(bool condition, const char *message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(status));
        std::exit(1);
    }
}
template<class T> struct device_buffer {
    T *data = nullptr;
    explicit device_buffer(std::size_t count) {
        require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
            count * sizeof(T)), "cudaMalloc");
    }
    ~device_buffer() { if (data != nullptr) cudaFree(data); }
};
template<class F> double wall_ns(F &&function) {
    const auto begin = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}
double mad_percent(const std::vector<double> &values) {
    const double center = median(values);
    std::vector<double> deviations;
    for (double value : values) deviations.push_back(std::fabs(value - center));
    return center == 0.0 ? 0.0 : 100.0 * median(deviations) / center;
}

__global__ void gather_selected(const sm70::support_projection_edge_v1 *edges,
    const float *logical, float *compact) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < logical_count)
        compact[index] = logical[edges[index].stable_output_index];
}
__global__ void normalize_materialized(const float *compact, float *normalized) {
    const std::uint32_t partition = blockIdx.x;
    if (partition >= partition_count || threadIdx.x != 0u) return;
    const std::uint32_t begin = partition * edges_per_partition;
    float maximum = -CUDART_INF_F;
    for (std::uint32_t local = 0u; local < edges_per_partition; ++local)
        maximum = fmaxf(maximum, compact[begin + local]);
    float denominator = 0.0f;
    for (std::uint32_t local = 0u; local < edges_per_partition; ++local)
        denominator += expf(compact[begin + local] - maximum);
    for (std::uint32_t local = 0u; local < edges_per_partition; ++local)
        normalized[begin + local] = expf(compact[begin + local] - maximum)
            / denominator;
}
__global__ void scatter_selected(const sm70::support_projection_edge_v1 *edges,
    const float *compact, float *logical) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < logical_count)
        logical[edges[index].stable_output_index] = compact[index];
}
} // namespace

int main() {
    cellerator::bench::benchmark_mutex_guard mutex("ce-geo-fusion-evaluation", 0);
    int device = 0;
    require_cuda(cudaGetDevice(&device), "get device");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device), "get properties");
    require(properties.major == 7 && properties.minor == 0,
        "fusion evaluation requires sm_70");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create stream");

    std::vector<sm70::support_projection_edge_v1> edges(logical_count);
    std::vector<sm70::cover_native_partition_v1> partitions(partition_count);
    std::vector<float> values(logical_count), materialized_output(logical_count),
        native_output(logical_count), reference(logical_count);
    const double host_prepare_ns = wall_ns([&] {
        for (std::uint32_t partition = 0u; partition < partition_count; ++partition) {
            const auto kind = partition < partition_count / 2u
                ? projection::physical_region_kind_v1::mma
                : projection::physical_region_kind_v1::residual;
            partitions[partition].region_kind = kind;
            partitions[partition].selected_begin = partition * edges_per_partition;
            partitions[partition].selected_count = edges_per_partition;
            for (std::uint32_t local = 0u; local < edges_per_partition; ++local) {
                const std::uint32_t index = partition * edges_per_partition + local;
                edges[index].logical_edge_id.value = index + 1u;
                edges[index].region_kind = kind;
                edges[index].region_index = partition;
                edges[index].projection_slot = local;
                edges[index].stable_output_index = index;
                values[index] = static_cast<float>(
                    static_cast<int>((index * 13u) % 29u) - 14) / 8.0f;
            }
        }
    });
    for (std::uint32_t partition = 0u; partition < partition_count; ++partition) {
        const std::uint32_t begin = partition * edges_per_partition;
        float maximum = values[begin];
        for (std::uint32_t local = 1u; local < edges_per_partition; ++local)
            maximum = std::max(maximum, values[begin + local]);
        float denominator = 0.0f;
        for (std::uint32_t local = 0u; local < edges_per_partition; ++local)
            denominator += std::exp(values[begin + local] - maximum);
        for (std::uint32_t local = 0u; local < edges_per_partition; ++local)
            reference[begin + local] = std::exp(values[begin + local] - maximum)
                / denominator;
    }

    device_buffer<sm70::support_projection_edge_v1> d_edges(logical_count);
    device_buffer<sm70::cover_native_partition_v1> d_partitions(partition_count);
    device_buffer<float> d_values(logical_count), d_compact(logical_count),
        d_normalized(logical_count), d_materialized(logical_count),
        d_native(logical_count);
    const double structure_upload_ns = wall_ns([&] {
        require_cuda(cudaMemcpyAsync(d_edges.data, edges.data(),
            logical_count * sizeof(edges[0]), cudaMemcpyHostToDevice, stream),
            "edge upload");
        require_cuda(cudaMemcpyAsync(d_partitions.data, partitions.data(),
            partition_count * sizeof(partitions[0]), cudaMemcpyHostToDevice,
            stream), "partition upload");
        require_cuda(cudaStreamSynchronize(stream), "structure upload sync");
    });

    sm70::cover_native_normalize_request_v1 native{};
    native.selected_edges = d_edges.data;
    native.selected_edge_count = logical_count;
    native.partitions = d_partitions.data;
    native.partition_count = partition_count;
    native.logical_edge_values = d_values.data;
    native.logical_edge_count = logical_count;
    native.logical_edge_output = d_native.data;
    native.stream = stream;
    auto run_materialized = [&] {
        require_cuda(cudaMemcpyAsync(d_values.data, values.data(),
            logical_count * sizeof(float), cudaMemcpyHostToDevice, stream),
            "materialized value upload");
        require_cuda(cudaMemsetAsync(d_materialized.data, 0,
            logical_count * sizeof(float), stream), "materialized clear");
        gather_selected<<<(logical_count + 255u) / 256u, 256u, 0u, stream>>>(
            d_edges.data, d_values.data, d_compact.data);
        normalize_materialized<<<partition_count, 1u, 0u, stream>>>(
            d_compact.data, d_normalized.data);
        scatter_selected<<<(logical_count + 255u) / 256u, 256u, 0u, stream>>>(
            d_edges.data, d_normalized.data, d_materialized.data);
        require_cuda(cudaPeekAtLastError(), "materialized launch");
        require_cuda(cudaMemcpyAsync(materialized_output.data(), d_materialized.data,
            logical_count * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "materialized output download");
        require_cuda(cudaStreamSynchronize(stream), "materialized complete sync");
    };
    auto run_native = [&] {
        require_cuda(cudaMemcpyAsync(d_values.data, values.data(),
            logical_count * sizeof(float), cudaMemcpyHostToDevice, stream),
            "native value upload");
        require(sm70::enqueue_cover_native_normalize_v1(native)
            == sm70::cover_native_normalize_status_v1::success,
            "cover-native normalize launch");
        require_cuda(cudaMemcpyAsync(native_output.data(), d_native.data,
            logical_count * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "native output download");
        require_cuda(cudaStreamSynchronize(stream), "native complete sync");
    };
    for (int warmup = 0; warmup < 3; ++warmup) { run_materialized(); run_native(); }
    std::vector<double> materialized_samples, native_samples;
    for (int repeat = 0; repeat < 11; ++repeat) {
        materialized_samples.push_back(wall_ns(run_materialized));
        native_samples.push_back(wall_ns(run_native));
    }
    double materialized_error = 0.0, native_error = 0.0;
    for (std::uint32_t index = 0u; index < logical_count; ++index) {
        materialized_error = std::max(materialized_error,
            static_cast<double>(std::fabs(materialized_output[index] - reference[index])));
        native_error = std::max(native_error,
            static_cast<double>(std::fabs(native_output[index] - reference[index])));
    }
    require(materialized_error <= 1.0e-6 && native_error <= 1.0e-6,
        "numerical reference mismatch");
    const double materialized_steady = median(materialized_samples);
    const double native_steady = median(native_samples);
    const double cold = host_prepare_ns + structure_upload_ns;
    std::cout << std::fixed << std::setprecision(3)
        << "{\"hardware_name\":\"" << properties.name
        << "\",\"compute_capability\":\"7.0\","
        << "\"host_prepare_ns\":" << host_prepare_ns
        << ",\"structure_upload_ns\":" << structure_upload_ns
        << ",\"materialized_steady_wall_ns\":" << materialized_steady
        << ",\"cover_native_steady_wall_ns\":" << native_steady
        << ",\"materialized_mad_percent\":" << mad_percent(materialized_samples)
        << ",\"cover_native_mad_percent\":" << mad_percent(native_samples)
        << ",\"materialized_max_abs_error\":" << materialized_error
        << ",\"cover_native_max_abs_error\":" << native_error
        << ",\"logical_edge_count\":" << logical_count
        << ",\"partition_count\":" << partition_count
        << ",\"edges_per_partition\":" << edges_per_partition
        << ",\"materialized_launches\":4,\"cover_native_launches\":2,"
        << "\"reuse\":[";
    const std::uint32_t reuses[] = {1u, 16u, 256u};
    for (std::uint32_t index = 0u; index < 3u; ++index) {
        const std::uint32_t reuse = reuses[index];
        std::cout << (index == 0u ? "" : ",") << "{\"reuse\":" << reuse
            << ",\"materialized_complete_ns\":"
            << materialized_steady + cold / reuse
            << ",\"cover_native_complete_ns\":" << native_steady + cold / reuse
            << "}";
    }
    std::cout << "]}\n";
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}
'''


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.compile_only and arguments.output is None:
        parser.error("--output is required unless --compile-only is used")
    with tempfile.TemporaryDirectory(prefix="ce_geo_fusion_") as directory:
        source = Path(directory) / "fusion_evaluation.cu"
        binary = Path(directory) / "fusion_evaluation"
        source.write_text(SOURCE, encoding="utf-8")
        compiled = run([
            "nvcc", "-std=c++17", "-arch=sm_70", "-O3", "-lineinfo",
            "-Xcompiler=-Wall,-Wextra,-Werror", "-I.", "-Iinclude",
            str(source), "-lcudart", "-o", str(binary),
        ])
        if compiled.stderr:
            print(compiled.stderr, file=sys.stderr, end="")
        if arguments.compile_only:
            print(json.dumps({"compile_valid": 1}, sort_keys=True))
            return 0
        measured = json.loads(run([str(binary)]).stdout)
    native_wins = all(
        point["cover_native_complete_ns"] < point["materialized_complete_ns"]
        for point in measured["reuse"]
    )
    evidence = {
        "schema": "CELLERATOR-CE-GEO-FUSION-EVALUATION/1",
        "task_id": "CE-GEO-98",
        "campaign_id": "exchange-cover-native-normalize",
        "controller_evidence_id": "CE-GEO-98-fusion-evaluation-v1",
        "benchmark_mutex": True,
        "uncontaminated": True,
        "correctness_passed": (
            measured["materialized_max_abs_error"] <= 1.0e-6
            and measured["cover_native_max_abs_error"] <= 1.0e-6
        ),
        "evidence_valid": 1,
        "accepted_for_promotion": False,
        "disposition": "evaluated_not_promoted",
        "candidate": {
            "name": "cover_native_normalize_without_selected_value_materialization",
            "supported": True,
            "wins_all_measured_reuse_points": native_wins,
            "launches": measured["cover_native_launches"],
        },
        "baseline": {
            "name": "materialized_gather_normalize_scatter",
            "launches": measured["materialized_launches"],
        },
        "measurement": measured,
        "methodology": {
            "clock": "host steady-clock wall time",
            "consumer_complete": "dynamic logical-value upload through output D2H and explicit cudaStreamSynchronize",
            "cold_cost": "deterministic partition preparation plus immutable edge/partition upload, amortized by reuse",
            "warmups": 3,
            "repeats": 11,
            "numerical_reference": "independent CPU partition softmax",
        },
        "limitations": [
            "Production exposes cover-native segment normalization, not a fused four-step contract/gate/normalize/apply exchange entrypoint.",
            "The materialized comparator is an explicit gather/normalize/scatter normalization stage, not a separately prepared whole exchange program.",
            "One deterministic synthetic 256-partition case cannot promote a general exchange fusion policy.",
            "No whole-program intermediate lifetime, downstream relation-apply cost, or biological held-out organization is measured.",
        ],
        "decision": (
            "The cover-native normalization mechanism is retained as measured implementation evidence, but whole-exchange fusion is not promoted."
        ),
    }
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "evidence_valid": 1,
        "correctness_passed": evidence["correctness_passed"],
        "accepted_for_promotion": False,
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
