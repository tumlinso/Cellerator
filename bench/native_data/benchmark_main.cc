#include "aten_baseline.hh"
#include "workload.hh"

#if defined(CELLERATOR_NATIVE_DATA_SPECIALIST)
#include "distributed_adapter.hh"
#endif

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using cellerator::bench::native_data::AtenTwoDeviceBaseline;
using cellerator::bench::native_data::CycleResult;
using cellerator::bench::native_data::TimingBreakdown;
using cellerator::bench::native_data::Workload;

#if defined(CELLERATOR_NATIVE_DATA_SPECIALIST)
using Runner = cellerator::bench::native_data::DistributedGatedProductAdapter;
constexpr std::string_view backend_name = "cellerator_distributed_gated_product";
#else
using Runner = AtenTwoDeviceBaseline;
constexpr std::string_view backend_name = "aten_eager_two_device";
#endif

struct Options {
    std::string case_name = "reference_active";
    std::string mode = "correctness";
    std::string output_path;
    int device0 = 0;
    int device1 = 1;
    std::size_t warmup = 2;
    std::size_t samples = 7;
    float learning_rate = 0.01f;
};

std::string quote(std::string_view text) {
    std::string result = "\"";
    for (const char value : text) {
        if (value == '"' || value == '\\') result.push_back('\\');
        if (value == '\n') {
            result += "\\n";
        } else if (value == '\r') {
            result += "\\r";
        } else {
            result.push_back(value);
        }
    }
    result.push_back('"');
    return result;
}

std::string hash_hex(std::uint64_t value) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << value;
    return stream.str();
}

template<class T>
void write_number(std::ostream& output, T value) {
    if constexpr (std::is_floating_point_v<T>) {
        if (!std::isfinite(value)) {
            output << "null";
            return;
        }
    }
    output << std::setprecision(10) << value;
}

void write_array(std::ostream& output, const std::vector<float>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        write_number(output, values[index]);
    }
    output << ']';
}

void write_timing(std::ostream& output, const TimingBreakdown& timing) {
    output << "{\"host_pack_ms\":"; write_number(output, timing.host_pack_ms);
    output << ",\"upload_ms\":"; write_number(output, timing.upload_ms);
    output << ",\"readiness_ms\":"; write_number(output, timing.readiness_ms);
    output << ",\"forward_ms\":"; write_number(output, timing.forward_ms);
    output << ",\"backward_ms\":"; write_number(output, timing.backward_ms);
    output << ",\"reduction_ms\":"; write_number(output, timing.reduction_ms);
    output << ",\"maintenance_ms\":"; write_number(output, timing.maintenance_ms);
    output << ",\"update_ms\":"; write_number(output, timing.update_ms);
    output << ",\"replica_refresh_ms\":"; write_number(output, timing.replica_refresh_ms);
    output << ",\"canonicalization_ms\":"; write_number(output, timing.canonicalization_ms);
    output << ",\"validation_export_ms\":"; write_number(output, timing.validation_export_ms);
    output << ",\"full_cycle_ms\":"; write_number(output, timing.full_cycle_ms);
    output << ",\"transfer_bytes\":" << timing.transfer_bytes;
    output << ",\"peak_allocated_bytes\":" << timing.peak_allocated_bytes;
    output << ",\"peak_reserved_bytes\":" << timing.peak_reserved_bytes;
    output << ",\"condition_transfer_bytes\":" << timing.condition_transfer_bytes;
    output << ",\"participant_transfer_bytes\":" << timing.participant_transfer_bytes;
    output << ",\"parameter_transfer_bytes\":" << timing.parameter_transfer_bytes;
    output << ",\"output_transfer_bytes\":" << timing.output_transfer_bytes;
    output << ",\"backward_transfer_bytes\":" << timing.backward_transfer_bytes;
    output << ",\"replica_refresh_bytes\":" << timing.replica_refresh_bytes;
    output << ",\"prepared_device_bytes\":" << timing.prepared_device_bytes;
    output << ",\"prepared_staging_bytes\":" << timing.prepared_staging_bytes;
    output << ",\"prepared_host_payload_bytes\":" << timing.prepared_host_payload_bytes;
    output << ",\"omitted_participant_transfers\":" << timing.omitted_participant_transfers;
    output << ",\"projection_refreshes\":" << timing.projection_refreshes;
    output << ",\"projected_participant_values\":" << timing.projected_participant_values;
    output << ",\"peer_forward\":" << (timing.peer_forward ? "true" : "false");
    output << ",\"peer_reverse\":" << (timing.peer_reverse ? "true" : "false");
    output << ",\"update_includes_replica_refresh\":"
           << (timing.update_includes_replica_refresh ? "true" : "false") << '}';
}

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view name(argv[index]);
        const auto require_value = [&]() -> std::string_view {
            if (index + 1 >= argc) throw std::invalid_argument(std::string(name) + " needs a value");
            return argv[++index];
        };
        if (name == "--case") options.case_name = require_value();
        else if (name == "--mode") options.mode = require_value();
        else if (name == "--output") options.output_path = require_value();
        else if (name == "--device0") options.device0 = std::stoi(std::string(require_value()));
        else if (name == "--device1") options.device1 = std::stoi(std::string(require_value()));
        else if (name == "--warmup") options.warmup = std::stoull(std::string(require_value()));
        else if (name == "--samples") options.samples = std::stoull(std::string(require_value()));
        else if (name == "--learning-rate") options.learning_rate = std::stof(std::string(require_value()));
        else if (name == "--help") {
            std::cout << "--case reference_active|reference_inactive|low|high "
                         "--mode correctness|timing --output PATH "
                         "--device0 N --device1 N --warmup N --samples N --learning-rate F\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (options.mode != "correctness" && options.mode != "timing")
        throw std::invalid_argument("mode must be correctness or timing");
    if (options.output_path.empty()) throw std::invalid_argument("--output is required");
    if (!options.samples) throw std::invalid_argument("samples must be positive");
    return options;
}

Workload make_workload(std::string_view name) {
    using namespace cellerator::bench::native_data;
    if (name == "reference_active") return make_reference_active();
    if (name == "reference_inactive") return make_reference_inactive();
    if (name == "low") return make_low_activity_disjoint();
    if (name == "high") return make_high_activity_overlap();
    throw std::invalid_argument("unknown workload case");
}

struct DeviceRecord {
    std::string name;
    std::string uuid;
    int major = 0;
    int minor = 0;
    std::uint64_t memory_bytes = 0;
};

DeviceRecord device_record(int device) {
    cudaDeviceProp properties{};
    const auto status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaGetDeviceProperties: ") + cudaGetErrorString(status));
    std::ostringstream uuid;
    uuid << std::hex << std::setfill('0');
    for (const auto byte : properties.uuid.bytes)
        uuid << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(byte));
    return {properties.name, uuid.str(), properties.major, properties.minor, properties.totalGlobalMem};
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    if (values.size() % 2) return values[middle];
    return (values[middle - 1] + values[middle]) * 0.5;
}

void write_samples(std::ostream& output, const std::vector<CycleResult>& samples,
                   const CycleResult* cold_start,
                   const std::vector<CycleResult>& warmups) {
    if (samples.empty()) {
        output << "{\"count\":0,\"cold_start\":";
        if (cold_start) write_timing(output, cold_start->timing);
        else output << "null";
        output << ",\"warmup\":{\"count\":0,\"full_cycle_ms_total\":0,\"samples\":[]},\"samples\":[]}";
        return;
    }
    const auto median_for = [&](double TimingBreakdown::*field) {
        std::vector<double> values;
        values.reserve(samples.size());
        for (const auto& sample : samples) values.push_back(sample.timing.*field);
        return median(std::move(values));
    };
    output << "{\"count\":" << samples.size();
    output << ",\"cold_start\":";
    if (cold_start) write_timing(output, cold_start->timing);
    else output << "null";
    double warmup_total = 0.0;
    for (const auto& warmup : warmups) warmup_total += warmup.timing.full_cycle_ms;
    output << ",\"warmup\":{\"count\":" << warmups.size()
           << ",\"full_cycle_ms_total\":";
    write_number(output, warmup_total);
    output << ",\"samples\":[";
    for (std::size_t index = 0; index < warmups.size(); ++index) {
        if (index) output << ',';
        write_timing(output, warmups[index].timing);
    }
    output << "]}";
    output << ",\"median_forward_ms\":"; write_number(output, median_for(&TimingBreakdown::forward_ms));
    output << ",\"median_backward_ms\":"; write_number(output, median_for(&TimingBreakdown::backward_ms));
    output << ",\"median_reduction_ms\":"; write_number(output, median_for(&TimingBreakdown::reduction_ms));
    output << ",\"median_update_ms\":"; write_number(output, median_for(&TimingBreakdown::update_ms));
    output << ",\"median_replica_refresh_ms\":"; write_number(output, median_for(&TimingBreakdown::replica_refresh_ms));
    output << ",\"median_full_cycle_ms\":"; write_number(output, median_for(&TimingBreakdown::full_cycle_ms));
    output << ",\"samples\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index) output << ',';
        write_timing(output, samples[index].timing);
    }
    output << "]}";
}

void write_result(const Options& options, const Workload& workload,
                  const Runner& baseline,
                  const std::vector<CycleResult>& samples,
                  const CycleResult* cold_start,
                  const std::vector<CycleResult>& warmups,
                  const CycleResult* correctness, const CycleResult* next) {
    const auto gpu0 = device_record(options.device0);
    const auto gpu1 = device_record(options.device1);
    const auto& prep = baseline.preparation();
    std::ofstream output(options.output_path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open result output path");

    output << "{\"schema\":\"cellerator.native_data.gated_product_benchmark.v1\"";
    output << ",\"backend\":" << quote(backend_name);
#if defined(CELLERATOR_NATIVE_DATA_SPECIALIST)
    output << ",\"memory_accounting\":{\"peak_allocator_metrics_available\":false"
           << ",\"prepared_device_bytes\":\"exact fixed Cellerator device-buffer capacity\""
           << ",\"peak_allocated_bytes\":\"unavailable for direct CUDA allocations\"}";
#else
    output << ",\"memory_accounting\":{\"peak_allocator_metrics_available\":true"
           << ",\"prepared_device_bytes\":\"sum of persistent ATen CUDA tensor storage\""
           << ",\"peak_allocated_bytes\":\"ATen caching allocator peak after per-cycle reset\"}";
#endif
    output << ",\"case\":" << quote(workload.name);
    output << ",\"mode\":" << quote(options.mode);
    output << ",\"devices\":[{\"visible_index\":" << options.device0
           << ",\"name\":" << quote(gpu0.name) << ",\"uuid\":" << quote(gpu0.uuid)
           << ",\"compute_capability\":" << gpu0.major << '.' << gpu0.minor
           << ",\"memory_bytes\":" << gpu0.memory_bytes << "},{\"visible_index\":" << options.device1
           << ",\"name\":" << quote(gpu1.name) << ",\"uuid\":" << quote(gpu1.uuid)
           << ",\"compute_capability\":" << gpu1.major << '.' << gpu1.minor
           << ",\"memory_bytes\":" << gpu1.memory_bytes << "}]";
    output << ",\"workload\":{\"batch\":" << workload.batch
           << ",\"coordinates\":" << workload.coordinates
           << ",\"conditions\":" << workload.conditions
           << ",\"parameters\":" << workload.parameters
           << ",\"groups\":" << workload.groups()
           << ",\"outputs\":" << workload.output_coordinate.size()
           << ",\"max_argument_slots\":"
           << *std::max_element(workload.argument_count.begin(), workload.argument_count.end())
           << ",\"active_groups\":" << prep.active_groups
           << ",\"active_fraction\":";
    write_number(output, static_cast<double>(prep.active_groups) / workload.groups());
    const auto owner0 = std::count(workload.group_owner.begin(), workload.group_owner.end(), 0);
    const auto owner1 = std::count(workload.group_owner.begin(), workload.group_owner.end(), 1);
    output << ",\"group_owners\":[" << owner0 << ',' << owner1 << ']';
    output << ",\"gate_argument_min\":"; write_number(output, prep.gate_argument_min);
    output << ",\"gate_argument_max\":"; write_number(output, prep.gate_argument_max);
    output << ",\"value_storage_dtype\":\"fp16\",\"regulator_dtype\":\"fp32\",\"compute_dtype\":\"fp32\",\"parameter_dtype\":\"fp32\",\"output_dtype\":\"fp32\",\"vjp_dtype\":\"fp32\"}";
    output << ",\"configuration\":{\"device0\":" << options.device0
           << ",\"device1\":" << options.device1
           << ",\"warmup\":" << options.warmup
           << ",\"samples\":" << options.samples
           << ",\"cold_start_cycles\":" << (options.mode == "timing" ? 1 : 0)
           << ",\"learning_rate\":"; write_number(output, options.learning_rate);
    output << "}";
    output << ",\"placement\":{\"owner_devices\":{\"expression\":" << options.device0
           << ",\"regulators\":" << options.device1
           << ",\"theta_phi_tau\":" << options.device0
           << ",\"cotangent\":" << options.device0
           << "},\"canonical_devices\":{\"output\":" << options.device0
           << ",\"dx\":" << options.device0
           << ",\"dz\":" << options.device1
           << ",\"dtheta_dphi_dtau\":" << options.device0
           << "},\"expression_representation\":"
#if defined(CELLERATOR_NATIVE_DATA_SPECIALIST)
           << quote("device0 full packed source; device1 active-group projection")
#else
           << quote("FP16 dense-coordinate replica on both devices after first-use copy")
#endif
           << ",\"regulator_replica_policy\":\"retain source on device1 and copy to device0 on first use\""
           << ",\"parameter_replica_policy\":\"device0 owner; device1 replica initialized on first use and refreshed on update\"} ";
    output << ",\"workload_identity\":{\"algorithm\":\"fnv1a64-little-endian-v1\",\"canonical_f32\":"
           << quote(hash_hex(workload.identity_hash()))
           << ",\"x_fp16\":" << quote(hash_hex(workload.quantized_x_hash())) << '}';
    output << ",\"preparation\":";
    write_timing(output, prep);
    output << ",\"summary\":";
    write_samples(output, samples, cold_start, warmups);
    if (correctness) {
        output << ",\"correctness_values\":{\"output\":"; write_array(output, correctness->output);
        output << ",\"dx\":"; write_array(output, correctness->dx);
        output << ",\"dz\":"; write_array(output, correctness->dz);
        output << ",\"dtheta\":"; write_array(output, correctness->dtheta);
        output << ",\"dphi\":"; write_array(output, correctness->dphi);
        output << ",\"dtau\":"; write_array(output, correctness->dtau);
        output << ",\"theta_after_update\":"; write_array(output, correctness->theta_after_update);
        output << ",\"post_update_output\":"; write_array(output, next->output);
        output << '}';
    }
    output << "}\n";
    if (!output) throw std::runtime_error("failed writing benchmark result");
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        auto workload = make_workload(options.case_name);
        Runner baseline(workload, options.device0, options.device1);
        std::vector<CycleResult> measured;
        std::vector<CycleResult> warmups;
        CycleResult cold_start;
        CycleResult correctness;
        CycleResult next;
        const CycleResult* cold_start_ptr = nullptr;
        const CycleResult* correctness_ptr = nullptr;
        const CycleResult* next_ptr = nullptr;
        if (options.mode == "correctness") {
            correctness = baseline.cycle(options.learning_rate, true, true);
            next = baseline.forward_only(true);
            correctness_ptr = &correctness;
            next_ptr = &next;
        } else {
            cold_start = baseline.cycle(options.learning_rate, true, false);
            cold_start_ptr = &cold_start;
            warmups.reserve(options.warmup);
            for (std::size_t warmup = 0; warmup < options.warmup; ++warmup)
                warmups.push_back(baseline.cycle(options.learning_rate, true, false));
            measured.reserve(options.samples);
            for (std::size_t sample = 0; sample < options.samples; ++sample)
                measured.push_back(baseline.cycle(options.learning_rate, true, false));
        }
        write_result(options, workload, baseline, measured, cold_start_ptr,
                     warmups, correctness_ptr, next_ptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native_data_benchmark_error: " << error.what() << '\n';
        return 2;
    }
}
