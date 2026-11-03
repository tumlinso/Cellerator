// Explicit fork of native_runner.cu SHA256 330270f691bfbce0adb1b48015736a9ff6a954166a4ca05977d41e64e58a41c7; see PROVENANCE.json.
#include <Cellerator/layout/factorized_interaction.hh>
#include "../wire_input.hh"
#include "../controls/minimal_direct_layout.hh"
#include "../candidates/direct_unique_output/direct_unique_output.hh"
#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/modeling/cuda/factorized_interaction.hh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace layout = cellerator::layout;
namespace memory_cuda = cellerator::memory::cuda;
namespace interaction = cellerator::modeling::cuda;

namespace controls = cellerator::bench::factorized_interactions::controls;
namespace candidates = cellerator::bench::factorized_interactions::candidates;
namespace {

// Candidate identity is set once after parsing and reported alongside the legacy
// backend field, which remains compatible with the fixed validator.
std::string requested_candidate;
std::string actual_implementation;
std::string structural_layout;

constexpr std::array<char, 8> input_magic{{'C','F','I','N','T','0','0','1'}};
constexpr std::array<char, 8> output_magic{{'C','F','O','U','T','0','0','1'}};
constexpr float accumulate_seed = 0.125f;
constexpr std::uint32_t max_timing_iterations = 256;

struct Header {
    std::uint64_t cells = 0, channels = 0, inputs = 0, contexts = 0;
    std::uint64_t outputs = 0, alpha_rows = 0, biases = 0, gammas = 0;
    std::uint64_t mechanisms = 0, terms = 0;
    std::uint32_t activation_type = 0, parameter_type = 0, output_type = 0;
};

struct HostCase {
    Header header;
    std::vector<layout::FactorizedInteractionMechanism> mechanisms;
    std::vector<std::byte> x, z, alpha, bias, gamma, dy;
};

struct OutputArrays {
    memory_cuda::PinnedBuffer y;
    memory_cuda::PinnedBuffer dx, dz, dalpha, dbias, dgamma;
};

class Cursor {
public:
    explicit Cursor(const std::vector<std::byte>& bytes) : bytes_(bytes) {}

    std::uint32_t u32() {
        const auto* ptr = take(4);
        std::uint32_t value = 0;
        for (unsigned i = 0; i < 4; ++i)
            value |= static_cast<std::uint32_t>(std::to_integer<unsigned char>(ptr[i])) << (8U * i);
        return value;
    }

    std::uint64_t u64() {
        const auto* ptr = take(8);
        std::uint64_t value = 0;
        for (unsigned i = 0; i < 8; ++i)
            value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(ptr[i])) << (8U * i);
        return value;
    }

    std::array<char, 8> magic() {
        const auto* ptr = take(8);
        std::array<char, 8> value{};
        std::memcpy(value.data(), ptr, value.size());
        return value;
    }

    std::vector<std::byte> bytes(std::size_t count) {
        const auto* ptr = take(count);
        return {ptr, ptr + count};
    }

    bool done() const noexcept { return offset_ == bytes_.size(); }

private:
    const std::byte* take(std::size_t count) {
        if (count > bytes_.size() - offset_)
            throw std::runtime_error("truncated factorized-interaction case file");
        const auto* ptr = bytes_.data() + offset_;
        offset_ += count;
        return ptr;
    }

    const std::vector<std::byte>& bytes_;
    std::size_t offset_ = 0;
};

std::size_t checked_product(std::initializer_list<std::uint64_t> factors,
                            std::size_t width) {
    std::size_t result = width;
    for (const auto factor : factors) {
        if (factor == 0) return 0;
        if (factor > std::numeric_limits<std::size_t>::max() / result)
            throw std::overflow_error("factorized-interaction wire extent overflows size_t");
        result *= static_cast<std::size_t>(factor);
    }
    return result;
}

std::size_t type_width(std::uint32_t code) {
    if (code == 1) return 2;
    if (code == 2) return 4;
    throw std::runtime_error("unknown factorized-interaction storage dtype code");
}

HostCase read_case(const std::vector<std::string>& paths) {
    auto bytes = factorized_wire::read_input_bytes(paths);
    Cursor cursor(bytes);
    if (cursor.magic() != input_magic)
        throw std::runtime_error("case input magic/version mismatch");
    HostCase result;
    auto& h = result.header;
    h.cells = cursor.u64(); h.channels = cursor.u64(); h.inputs = cursor.u64();
    h.contexts = cursor.u64(); h.outputs = cursor.u64(); h.alpha_rows = cursor.u64();
    h.biases = cursor.u64(); h.gammas = cursor.u64(); h.mechanisms = cursor.u64();
    h.terms = cursor.u64();
    h.activation_type = cursor.u32(); h.parameter_type = cursor.u32();
    h.output_type = cursor.u32();
    const std::size_t activation_width = type_width(h.activation_type);
    const std::size_t parameter_width = type_width(h.parameter_type);
    (void)type_width(h.output_type);

    result.mechanisms.reserve(static_cast<std::size_t>(h.mechanisms));
    std::uint64_t total_terms = 0;
    for (std::uint64_t i = 0; i < h.mechanisms; ++i) {
        layout::FactorizedInteractionMechanism mechanism;
        const std::uint64_t wire_logical_ordinal = cursor.u64();
        mechanism.target = cursor.u64();
        mechanism.alpha = cursor.u64();
        mechanism.bias = cursor.u64();
        mechanism.arity = cursor.u32();
        const std::uint32_t term_count = cursor.u32();
        if (wire_logical_ordinal != i || mechanism.arity < 2 || mechanism.arity > 4)
            throw std::runtime_error("invalid case logical ID or arity");
        if (wire_logical_ordinal == UINT64_MAX)
            throw std::runtime_error("case logical ID cannot be mapped to a stable prep ID");
        // The wire format deliberately uses zero-based record ordinals. The
        // host preparation API reserves zero and requires unique nonzero IDs,
        // so map only after validating the wire ordinal and keep the protocol
        // and independent oracle unchanged.
        mechanism.logical_id = wire_logical_ordinal + 1;
        for (auto& input_id : mechanism.inputs) input_id = cursor.u64();
        mechanism.terms.reserve(term_count);
        for (std::uint32_t term = 0; term < term_count; ++term) {
            mechanism.terms.push_back({cursor.u64(), cursor.u64()});
        }
        total_terms += term_count;
        result.mechanisms.push_back(std::move(mechanism));
    }
    if (total_terms != h.terms)
        throw std::runtime_error("case term total does not match its header");

    result.x = cursor.bytes(checked_product({h.inputs, h.channels, h.cells},
                                            activation_width));
    result.z = cursor.bytes(checked_product({h.contexts, h.cells}, activation_width));
    result.alpha = cursor.bytes(checked_product({h.alpha_rows, h.channels},
                                                parameter_width));
    result.bias = cursor.bytes(checked_product({h.biases}, parameter_width));
    result.gamma = cursor.bytes(checked_product({h.gammas}, parameter_width));
    result.dy = cursor.bytes(checked_product({h.outputs, h.channels, h.cells}, 4));
    if (!cursor.done()) throw std::runtime_error("case file contains trailing bytes");
    return result;
}

void append_u32(std::ofstream& output, std::uint32_t value) {
    std::array<unsigned char, 4> bytes{};
    for (unsigned i = 0; i < 4; ++i) bytes[i] = static_cast<unsigned char>(value >> (8U * i));
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void append_u64(std::ofstream& output, std::uint64_t value) {
    std::array<unsigned char, 8> bytes{};
    for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<unsigned char>(value >> (8U * i));
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void append_bytes(std::ofstream& output, const void* data, std::size_t bytes) {
    if (bytes != 0) output.write(static_cast<const char*>(data), bytes);
}

void write_result(const std::string& path, const Header& h, const void* y,
                  const OutputArrays& gradients, const OutputArrays* repeat_vjp) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open result output: " + path);
    output.write(output_magic.data(), output_magic.size());
    for (const auto value : {h.cells, h.channels, h.inputs, h.contexts, h.outputs,
                             h.alpha_rows, h.biases, h.gammas, h.mechanisms, h.terms})
        append_u64(output, value);
    append_u32(output, h.output_type);
    append_u32(output, 2); // FP32 gradient storage
    append_u32(output, repeat_vjp ? 2 : 1); // number of private VJP sets
    append_bytes(output, y, checked_product({h.outputs, h.channels, h.cells},
                                            type_width(h.output_type)));
    append_bytes(output, gradients.dx.data(), gradients.dx.bytes());
    append_bytes(output, gradients.dz.data(), gradients.dz.bytes());
    append_bytes(output, gradients.dalpha.data(), gradients.dalpha.bytes());
    append_bytes(output, gradients.dbias.data(), gradients.dbias.bytes());
    append_bytes(output, gradients.dgamma.data(), gradients.dgamma.bytes());
    if (repeat_vjp) {
        append_bytes(output, repeat_vjp->dx.data(), repeat_vjp->dx.bytes());
        append_bytes(output, repeat_vjp->dz.data(), repeat_vjp->dz.bytes());
        append_bytes(output, repeat_vjp->dalpha.data(), repeat_vjp->dalpha.bytes());
        append_bytes(output, repeat_vjp->dbias.data(), repeat_vjp->dbias.bytes());
        append_bytes(output, repeat_vjp->dgamma.data(), repeat_vjp->dgamma.bytes());
    }
    if (!output) throw std::runtime_error("failed while writing native result");
}

void upload(memory_cuda::DeviceBuffer& target, const std::vector<std::byte>& source,
            memory_cuda::Stream& stream) {
    if (source.empty()) return;
    memory_cuda::PinnedBuffer staging(source.size());
    std::memcpy(staging.data(), source.data(), source.size());
    target.upload_async(staging.data(), source.size(), stream);
    stream.synchronize(); // staging may be released only after its upload completes
}

void seed_gradient(memory_cuda::DeviceBuffer& target, std::size_t bytes,
                   memory_cuda::Stream& stream) {
    if (bytes == 0) return;
    if (bytes % sizeof(float) != 0)
        throw std::logic_error("gradient buffer is not FP32 aligned");
    std::vector<float> initial(bytes / sizeof(float), accumulate_seed);
    memory_cuda::PinnedBuffer staging(bytes);
    std::memcpy(staging.data(), initial.data(), bytes);
    target.upload_async(staging.data(), bytes, stream);
    stream.synchronize();
}

void download(const memory_cuda::DeviceBuffer& source, memory_cuda::PinnedBuffer& target,
              std::size_t bytes, memory_cuda::Stream& stream) {
    if (bytes == 0) return;
    source.download_async(target.data(), bytes, stream);
}

memory_cuda::DType dtype(std::uint32_t code) {
    return code == 1 ? memory_cuda::DType::float16 : memory_cuda::DType::float32;
}

std::string uuid_string(const cudaUUID_t& uuid);
double percentile(std::vector<double> values, double fraction);
const char* backend_name(interaction::FactorizedInteractionBackend backend);
void write_metrics(const std::string& path, const std::string& route,
                   const std::string& gradient_mode, const char* job_id,
                   interaction::FactorizedInteractionBackend backend,
                   const std::string& uuid, const cudaDeviceProp& properties,
                   const Header& header,
                   const interaction::FactorizedInteractionWorkspace& workspace,
                   std::size_t layout_bytes, std::size_t persistent_value_bytes,
                   std::size_t gradient_bytes, double host_prepare_ms,
                   double layout_upload_ms, double activation_upload_ms,
                   double parameter_upload_ms, double allocation_ms,
                   double first_forward_vjp_event_ms, double first_forward_vjp_wall_ms,
                   double warm_batch_wall_ms,
                   const std::vector<double>& parameter_refresh_ms, const std::vector<double>& forward_ms,
                   const std::vector<double>& backward_ms,
                   const std::vector<double>& step_ms);
void write_runner_metadata(const std::string& output_path,
                           const std::string& route,
                           const std::string& gradient_mode,
                           const char* job_id,
                           const std::string& uuid,
                           interaction::FactorizedInteractionBackend backend,
                           const layout::FactorizedInteractionView& plan,
                           const Header& header,
                           const interaction::FactorizedInteractionWorkspace& workspace,
                           std::size_t layout_bytes);


cudaError_t candidate_forward(const layout::FactorizedInteractionView& p, interaction::FactorizedInteractionProblem problem,
    interaction::FactorizedInteractionPrecision precision, interaction::FactorizedInteractionValues values,
    memory_cuda::DeviceView y, memory_cuda::DeviceView tape, memory_cuda::DeviceView scratch,
    cudaStream_t stream, interaction::FactorizedInteractionRoute route) {
    if (requested_candidate == "direct_unique_output")
        return candidates::factorized_interaction_unique_output_forward(p, problem, precision, values, y, tape, scratch, stream);
    return interaction::factorized_interaction_forward(p, problem, precision, values, y, tape, scratch, stream, route);
}
cudaError_t candidate_backward(const layout::FactorizedInteractionView& p, interaction::FactorizedInteractionProblem problem,
    interaction::FactorizedInteractionPrecision precision, interaction::FactorizedInteractionValues values,
    memory_cuda::ConstDeviceView dy, memory_cuda::ConstDeviceView tape, interaction::FactorizedInteractionGradients gradients,
    memory_cuda::DeviceView scratch, interaction::FactorizedInteractionGradientMode mode,
    cudaStream_t stream, interaction::FactorizedInteractionRoute route) {
    if (requested_candidate == "direct_unique_output")
        return candidates::factorized_interaction_unique_output_backward(p, problem, precision, values, dy, tape, gradients, scratch, mode, stream);
    return interaction::factorized_interaction_backward(p, problem, precision, values, dy, tape, gradients, scratch, mode, stream, route);
}
int run(const std::vector<std::string>& input_paths, const std::string& output_path,
        const std::string& route_name, const std::string& gradient_mode_name,
        int device, std::uint32_t iterations, const std::string& metrics_path,
        const char* scheduler_job_id, const std::string& expected_uuid,
        bool repeat_vjp) {
    if (!scheduler_job_id || !*scheduler_job_id)
        throw std::runtime_error("CUDA execution requires an Autobench worker Scheduler job context");
    if (iterations > max_timing_iterations)
        throw std::invalid_argument("timing iteration count exceeds the bounded limit");
    if (iterations != 0 && metrics_path.empty())
        throw std::invalid_argument("timing iterations require --metrics output");
    HostCase host = read_case(input_paths);
    const auto route = route_name == "packet" ? interaction::FactorizedInteractionRoute::packet
                                                : interaction::FactorizedInteractionRoute::direct;
    const auto gradient_mode = gradient_mode_name == "overwrite"
        ? interaction::FactorizedInteractionGradientMode::overwrite
        : interaction::FactorizedInteractionGradientMode::accumulate;
    if (route_name != "packet" && route_name != "direct")
        throw std::invalid_argument("route must be packet or direct");
    if (gradient_mode_name != "overwrite" && gradient_mode_name != "accumulate")
        throw std::invalid_argument("gradient mode must be overwrite or accumulate");

    memory_cuda::DeviceGuard guard(device);
    cudaDeviceProp properties{};
    memory_cuda::check(cudaGetDeviceProperties(&properties, device),
                       "query visible device properties");
    const auto actual_uuid = uuid_string(properties.uuid);
    if (expected_uuid.empty())
        throw std::invalid_argument("CUDA runner requires an explicit Scheduler UUID");
    if (actual_uuid != expected_uuid)
        throw std::runtime_error("visible CUDA UUID differs from the explicitly selected Scheduler UUID");
    memory_cuda::Stream stream(device);
    layout::FactorizedInteractionExtents extents{
        host.header.inputs, host.header.contexts, host.header.outputs,
        host.header.alpha_rows, host.header.biases, host.header.gammas};
    const auto cells = host.header.cells;
    const auto channels = host.header.channels;
    const interaction::FactorizedInteractionProblem problem{cells, channels};
    const auto preparation_begin = std::chrono::steady_clock::now();
    std::unique_ptr<controls::MinimalDirectHostLayout> minimal_host;
    std::unique_ptr<layout::FactorizedInteractionLayout> full_host;
    if (requested_candidate == "minimal_direct" || requested_candidate == "direct_unique_output") {
        minimal_host = std::make_unique<controls::MinimalDirectHostLayout>(host.mechanisms, extents);
        if (!minimal_host->direct_only_applicable(problem)) minimal_host.reset();
    }
    if (!minimal_host) {
        full_host = std::make_unique<layout::FactorizedInteractionLayout>(
            layout::prepare_factorized_interactions(host.mechanisms, extents));
    }
    structural_layout = minimal_host ? "canonical_direct" : "full_packet";
    const auto host_view = minimal_host ? minimal_host->view() : full_host->view();
    const auto preparation_end = std::chrono::steady_clock::now();
    const auto layout_upload_begin = std::chrono::steady_clock::now();
    std::unique_ptr<controls::MinimalDirectDeviceLayout> minimal_device;
    std::unique_ptr<interaction::FactorizedInteractionDeviceLayout> full_device;
    if (minimal_host)
        minimal_device = std::make_unique<controls::MinimalDirectDeviceLayout>(*minimal_host, problem, device, stream);
    else
        full_device = std::make_unique<interaction::FactorizedInteractionDeviceLayout>(*full_host, device, stream);
    stream.synchronize();
    const auto layout_upload_end = std::chrono::steady_clock::now();
    const auto& device_view = minimal_device ? minimal_device->view() : full_device->view();
    const auto layout_bytes = minimal_device ? minimal_device->bytes() : full_device->bytes();
    interaction::FactorizedInteractionBackend selected_backend{};
    if (requested_candidate == "direct_unique_output") {
        candidates::DirectUniqueOutputBackend selected{};
        memory_cuda::check(candidates::factorized_interaction_unique_output_backend(host_view, problem, &selected), "unique backend query");
        selected_backend = selected == candidates::DirectUniqueOutputBackend::stable
            ? interaction::FactorizedInteractionBackend::stable : interaction::FactorizedInteractionBackend::direct;
        actual_implementation = selected == candidates::DirectUniqueOutputBackend::direct_unique_output
            ? "direct_unique_output" : (selected == candidates::DirectUniqueOutputBackend::stable ? "stable" : "direct");
    } else {
        memory_cuda::check(interaction::factorized_interaction_backend(host_view, problem, route, &selected_backend), "factorized backend query");
        actual_implementation = backend_name(selected_backend);
    }
    interaction::FactorizedInteractionWorkspace workspace{};
    memory_cuda::check(requested_candidate == "direct_unique_output"
        ? candidates::factorized_interaction_unique_output_workspace(device_view, problem, &workspace)
        : interaction::factorized_interaction_workspace(device_view, problem, route, &workspace), "factorized workspace query");

    const auto value_allocation_begin = std::chrono::steady_clock::now();
    memory_cuda::DeviceBuffer x(host.x.size(), device), z(host.z.size(), device);
    memory_cuda::DeviceBuffer alpha(host.alpha.size(), device), bias(host.bias.size(), device);
    memory_cuda::DeviceBuffer gamma(host.gamma.size(), device), dy(host.dy.size(), device);
    const auto value_allocation_end = std::chrono::steady_clock::now();
    const auto activation_upload_begin = std::chrono::steady_clock::now();
    upload(x, host.x, stream); upload(z, host.z, stream); upload(dy, host.dy, stream);
    const auto activation_upload_end = std::chrono::steady_clock::now();
    const auto parameter_upload_begin = std::chrono::steady_clock::now();
    upload(alpha, host.alpha, stream); upload(bias, host.bias, stream);
    upload(gamma, host.gamma, stream);
    const auto parameter_upload_end = std::chrono::steady_clock::now();

    const auto output_bytes = checked_product({cells, channels, host.header.outputs},
                                              type_width(host.header.output_type));
    const auto dx_bytes = checked_product({host.header.inputs, channels, cells}, 4);
    const auto dz_bytes = checked_product({host.header.contexts, cells}, 4);
    const auto dalpha_bytes = checked_product({host.header.alpha_rows, channels}, 4);
    const auto dbias_bytes = checked_product({host.header.biases}, 4);
    const auto dgamma_bytes = checked_product({host.header.gammas}, 4);
    const auto output_allocation_begin = std::chrono::steady_clock::now();
    memory_cuda::DeviceBuffer output(output_bytes, device);
    memory_cuda::DeviceBuffer tape(workspace.tape_bytes, device);
    memory_cuda::DeviceBuffer forward_scratch(workspace.forward_bytes, device);
    memory_cuda::DeviceBuffer backward_scratch(workspace.backward_bytes, device);
    memory_cuda::DeviceBuffer dx(dx_bytes, device), dz(dz_bytes, device);
    memory_cuda::DeviceBuffer dalpha(dalpha_bytes, device), dbias(dbias_bytes, device);
    memory_cuda::DeviceBuffer dgamma(dgamma_bytes, device);
    seed_gradient(dx, dx_bytes, stream); seed_gradient(dz, dz_bytes, stream);
    seed_gradient(dalpha, dalpha_bytes, stream); seed_gradient(dbias, dbias_bytes, stream);
    seed_gradient(dgamma, dgamma_bytes, stream);
    if (output_bytes != 0)
        memory_cuda::check(cudaMemsetAsync(output.data(), 0xa5, output_bytes, stream.get()),
                           "seed output buffer");
    stream.synchronize();
    const auto output_allocation_end = std::chrono::steady_clock::now();

    const interaction::FactorizedInteractionPrecision precision{
        dtype(host.header.activation_type), dtype(host.header.parameter_type),
        dtype(host.header.output_type)};
    const interaction::FactorizedInteractionValues values{
        x.view(), z.view(), alpha.view(), bias.view(), gamma.view()};
    std::unique_ptr<memory_cuda::Event> first_begin, first_end;
    if (iterations) {
        first_begin = std::make_unique<memory_cuda::Event>(device, true);
        first_end = std::make_unique<memory_cuda::Event>(device, true);
    }
    const auto first_wall_begin = std::chrono::steady_clock::now();
    if (first_begin) first_begin->record(stream);
    memory_cuda::check(candidate_forward(
        device_view, problem, precision, values, output.view(), tape.view(),
        forward_scratch.view(), stream.get(), route), "factorized forward");
    memory_cuda::check(candidate_backward(
        device_view, problem, precision, values, dy.view(), tape.view(),
        {dx.view(), dz.view(), dalpha.view(), dbias.view(), dgamma.view()},
        backward_scratch.view(), gradient_mode, stream.get(), route),
        "factorized backward");
    if (first_end) { first_end->record(stream); stream.synchronize(); }
    const auto first_wall_end = std::chrono::steady_clock::now();
    const double first_forward_vjp_event_ms = first_begin ? first_begin->elapsed_ms(*first_end) : 0.0;
    const double first_forward_vjp_wall_ms = first_begin
        ? std::chrono::duration<double, std::milli>(first_wall_end - first_wall_begin).count() : 0.0;

    memory_cuda::DeviceBuffer dx_repeat(repeat_vjp ? dx_bytes : 0, device);
    memory_cuda::DeviceBuffer dz_repeat(repeat_vjp ? dz_bytes : 0, device);
    memory_cuda::DeviceBuffer dalpha_repeat(repeat_vjp ? dalpha_bytes : 0, device);
    memory_cuda::DeviceBuffer dbias_repeat(repeat_vjp ? dbias_bytes : 0, device);
    memory_cuda::DeviceBuffer dgamma_repeat(repeat_vjp ? dgamma_bytes : 0, device);
    memory_cuda::DeviceBuffer backward_scratch_repeat(
        repeat_vjp ? workspace.backward_bytes : 0, device);
    if (repeat_vjp) {
        seed_gradient(dx_repeat, dx_bytes, stream); seed_gradient(dz_repeat, dz_bytes, stream);
        seed_gradient(dalpha_repeat, dalpha_bytes, stream);
        seed_gradient(dbias_repeat, dbias_bytes, stream);
        seed_gradient(dgamma_repeat, dgamma_bytes, stream);
        memory_cuda::check(candidate_backward(
            device_view, problem, precision, values, dy.view(), tape.view(),
            {dx_repeat.view(), dz_repeat.view(), dalpha_repeat.view(),
             dbias_repeat.view(), dgamma_repeat.view()},
            backward_scratch_repeat.view(), gradient_mode, stream.get(), route),
            "repeated factorized backward");
    }

    OutputArrays host_output{
        memory_cuda::PinnedBuffer(output_bytes), memory_cuda::PinnedBuffer(dx_bytes),
        memory_cuda::PinnedBuffer(dz_bytes), memory_cuda::PinnedBuffer(dalpha_bytes),
        memory_cuda::PinnedBuffer(dbias_bytes), memory_cuda::PinnedBuffer(dgamma_bytes)};
    std::unique_ptr<OutputArrays> repeated_output;
    if (repeat_vjp) {
        repeated_output = std::make_unique<OutputArrays>(OutputArrays{
            memory_cuda::PinnedBuffer(0), memory_cuda::PinnedBuffer(dx_bytes),
            memory_cuda::PinnedBuffer(dz_bytes), memory_cuda::PinnedBuffer(dalpha_bytes),
            memory_cuda::PinnedBuffer(dbias_bytes), memory_cuda::PinnedBuffer(dgamma_bytes)});
    }
    download(output, host_output.y, output_bytes, stream);
    download(dx, host_output.dx, dx_bytes, stream);
    download(dz, host_output.dz, dz_bytes, stream);
    download(dalpha, host_output.dalpha, dalpha_bytes, stream);
    download(dbias, host_output.dbias, dbias_bytes, stream);
    download(dgamma, host_output.dgamma, dgamma_bytes, stream);
    if (repeat_vjp) {
        download(dx_repeat, repeated_output->dx, dx_bytes, stream);
        download(dz_repeat, repeated_output->dz, dz_bytes, stream);
        download(dalpha_repeat, repeated_output->dalpha, dalpha_bytes, stream);
        download(dbias_repeat, repeated_output->dbias, dbias_bytes, stream);
        download(dgamma_repeat, repeated_output->dgamma, dgamma_bytes, stream);
    }
    stream.synchronize();
    write_result(output_path, host.header, host_output.y.data(), host_output,
                 repeated_output.get());

    const auto host_prepare_ms = std::chrono::duration<double, std::milli>(
        preparation_end - preparation_begin).count();
    const auto layout_upload_ms = std::chrono::duration<double, std::milli>(
        layout_upload_end - layout_upload_begin).count();
    const auto activation_upload_ms = std::chrono::duration<double, std::milli>(
        activation_upload_end - activation_upload_begin).count();
    const auto parameter_upload_ms = std::chrono::duration<double, std::milli>(
        parameter_upload_end - parameter_upload_begin).count();
    const auto allocation_ms = std::chrono::duration<double, std::milli>(
        value_allocation_end - value_allocation_begin + output_allocation_end
        - output_allocation_begin).count();
    std::vector<double> parameter_refresh_ms;
    for (std::uint32_t i = 0; i < iterations; ++i) {
        const auto refresh_begin = std::chrono::steady_clock::now();
        upload(alpha, host.alpha, stream); upload(bias, host.bias, stream);
        upload(gamma, host.gamma, stream);
        const auto refresh_end = std::chrono::steady_clock::now();
        parameter_refresh_ms.push_back(std::chrono::duration<double, std::milli>(refresh_end - refresh_begin).count());
    }

    std::vector<double> forward_ms, backward_ms, step_ms;
    double warm_batch_wall_ms = 0.0;
    if (iterations != 0) {
        for (int i = 0; i < 3; ++i) {
            memory_cuda::check(candidate_forward(
                device_view, problem, precision, values, output.view(), tape.view(),
                forward_scratch.view(), stream.get(), route), "factorized warmup forward");
            memory_cuda::check(candidate_backward(
                device_view, problem, precision, values, dy.view(), tape.view(),
                {dx.view(), dz.view(), dalpha.view(), dbias.view(), dgamma.view()},
                backward_scratch.view(), gradient_mode, stream.get(), route),
                "factorized warmup backward");
        }
        stream.synchronize();
        std::vector<memory_cuda::Event> forward_begin, forward_end;
        std::vector<memory_cuda::Event> backward_begin, backward_end;
        std::vector<memory_cuda::Event> step_begin, step_end;
        forward_begin.reserve(iterations); forward_end.reserve(iterations);
        backward_begin.reserve(iterations); backward_end.reserve(iterations);
        step_begin.reserve(iterations); step_end.reserve(iterations);
        for (std::uint32_t i = 0; i < iterations; ++i) {
            forward_begin.emplace_back(device, true); forward_end.emplace_back(device, true);
            backward_begin.emplace_back(device, true); backward_end.emplace_back(device, true);
            step_begin.emplace_back(device, true); step_end.emplace_back(device, true);
        }
        const auto warm_batch_begin = std::chrono::steady_clock::now();
        for (std::uint32_t i = 0; i < iterations; ++i) {
            memory_cuda::check(candidate_forward(
                device_view, problem, precision, values, output.view(), tape.view(),
                forward_scratch.view(), stream.get(), route), "factorized primary forward");
            memory_cuda::check(candidate_backward(
                device_view, problem, precision, values, dy.view(), tape.view(),
                {dx.view(), dz.view(), dalpha.view(), dbias.view(), dgamma.view()},
                backward_scratch.view(), gradient_mode, stream.get(), route), "factorized primary backward");
        }
        stream.synchronize();
        const auto warm_batch_end = std::chrono::steady_clock::now();
        warm_batch_wall_ms = std::chrono::duration<double, std::milli>(warm_batch_end - warm_batch_begin).count();
        for (std::uint32_t i = 0; i < iterations; ++i) {
            step_begin[i].record(stream);
            forward_begin[i].record(stream);
            memory_cuda::check(candidate_forward(
                device_view, problem, precision, values, output.view(), tape.view(),
                forward_scratch.view(), stream.get(), route), "factorized timed forward");
            forward_end[i].record(stream);
            backward_begin[i].record(stream);
            memory_cuda::check(candidate_backward(
                device_view, problem, precision, values, dy.view(), tape.view(),
                {dx.view(), dz.view(), dalpha.view(), dbias.view(), dgamma.view()},
                backward_scratch.view(), gradient_mode, stream.get(), route),
                "factorized timed backward");
            backward_end[i].record(stream);
            step_end[i].record(stream);
        }
        stream.synchronize();
        forward_ms.reserve(iterations); backward_ms.reserve(iterations); step_ms.reserve(iterations);
        for (std::uint32_t i = 0; i < iterations; ++i) {
            forward_ms.push_back(forward_begin[i].elapsed_ms(forward_end[i]));
            backward_ms.push_back(backward_begin[i].elapsed_ms(backward_end[i]));
            step_ms.push_back(step_begin[i].elapsed_ms(step_end[i]));
        }
    }

    const auto device_uuid = uuid_string(properties.uuid);
    std::cerr << "visible_device=" << device << " uuid=" << device_uuid
              << " sm=" << properties.major << '.' << properties.minor
              << " host_prepare_ms=" << host_prepare_ms
              << " layout_bytes=" << layout_bytes
              << " tape_bytes=" << workspace.tape_bytes
              << " forward_scratch_bytes=" << workspace.forward_bytes
              << " backward_scratch_bytes=" << workspace.backward_bytes << '\n';
    if (iterations != 0) {
        const std::size_t persistent_value_bytes = host.x.size() + host.z.size()
            + host.alpha.size() + host.bias.size() + host.gamma.size() + host.dy.size();
        const std::size_t gradient_bytes = dx_bytes + dz_bytes + dalpha_bytes
            + dbias_bytes + dgamma_bytes;
        write_metrics(metrics_path, route_name, gradient_mode_name, scheduler_job_id,
                      selected_backend, device_uuid, properties, host.header, workspace,
                      layout_bytes, persistent_value_bytes, gradient_bytes,
                      host_prepare_ms, layout_upload_ms, activation_upload_ms,
                      parameter_upload_ms, allocation_ms, first_forward_vjp_event_ms, first_forward_vjp_wall_ms, warm_batch_wall_ms, parameter_refresh_ms,
                      forward_ms, backward_ms, step_ms);
    }
    write_runner_metadata(output_path, route_name, gradient_mode_name,
                          scheduler_job_id, device_uuid, selected_backend,
                          host_view, host.header, workspace,
                          layout_bytes);
    return 0;
}

std::string argument(int argc, char** argv, const std::string& key,
                     const std::string& fallback = {}) {
    for (int i = 1; i + 1 < argc; ++i)
        if (argv[i] == key) return argv[i + 1];
    return fallback;
}

bool has_flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i)
        if (argv[i] == key) return true;
    return false;
}

std::string uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream value;
    value << "GPU-" << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < sizeof(uuid.bytes); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) value << '-';
        value << std::setw(2) << static_cast<unsigned>(
            static_cast<unsigned char>(uuid.bytes[i]));
    }
    return value.str();
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

const char* backend_name(interaction::FactorizedInteractionBackend backend) {
    switch (backend) {
    case interaction::FactorizedInteractionBackend::packet: return "packet";
    case interaction::FactorizedInteractionBackend::direct: return "direct";
    case interaction::FactorizedInteractionBackend::stable: return "stable";
    }
    return "invalid";
}

void write_runner_metadata(const std::string& output_path,
                           const std::string& route,
                           const std::string& gradient_mode,
                           const char* job_id,
                           const std::string& uuid,
                           interaction::FactorizedInteractionBackend backend,
                           const layout::FactorizedInteractionView& plan,
                           const Header& header,
                           const interaction::FactorizedInteractionWorkspace& workspace,
                           std::size_t layout_bytes) {
    const std::string path = output_path + ".metadata.json";
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open runner metadata output: " + path);
    output << std::setprecision(9)
           << "{\n"
           << "  \"schema\": \"cellerator.factorized_interactions.runner.v1\",\n"
           << "  \"scheduler_job_id\": \"" << (job_id ? job_id : "") << "\",\n"
           << "  \"requested_route\": \"" << route << "\",\n"
           << "  \"selected_backend\": \"" << backend_name(backend) << "\",\n"
           << "  \"candidate\": \"" << requested_candidate << "\",\n"
           << "  \"actual_implementation\": \"" << actual_implementation << "\",\n"
           << "  \"structural_layout\": \"" << structural_layout << "\",\n"
           << "  \"gradient_mode\": \"" << gradient_mode << "\",\n"
           << "  \"device_uuid\": \"" << uuid << "\",\n"
           << "  \"cells\": " << header.cells << ",\n"
           << "  \"channels\": " << header.channels << ",\n"
           << "  \"mechanisms\": " << header.mechanisms << ",\n"
           << "  \"outputs\": " << header.outputs << ",\n"
           << "  \"terms\": " << header.terms << ",\n"
           << "  \"structural_maxima\": {\n"
           << "    \"output_mechanisms\": " << plan.max_output_mechanisms << ",\n"
           << "    \"input_roles\": " << plan.max_input_roles << ",\n"
           << "    \"context_terms\": " << plan.max_context_terms << ",\n"
           << "    \"alpha_mechanisms\": " << plan.max_alpha_mechanisms << ",\n"
           << "    \"bias_mechanisms\": " << plan.max_bias_mechanisms << ",\n"
           << "    \"gamma_terms\": " << plan.max_gamma_terms << ",\n"
           << "    \"mechanism_terms\": " << plan.max_mechanism_terms << "\n"
           << "  },\n"
           << "  \"memory_bytes\": {\n"
           << "    \"persistent_layout\": " << layout_bytes << ",\n"
           << "    \"tape\": " << workspace.tape_bytes << ",\n"
           << "    \"forward_scratch\": " << workspace.forward_bytes << ",\n"
           << "    \"backward_scratch\": " << workspace.backward_bytes << "\n"
           << "  }\n"
           << "}\n";
    if (!output) throw std::runtime_error("failed while writing runner metadata");
}

void write_metrics(const std::string& path, const std::string& route,
                   const std::string& gradient_mode, const char* job_id,
                   interaction::FactorizedInteractionBackend backend,
                   const std::string& uuid, const cudaDeviceProp& properties,
                   const Header& header,
                   const interaction::FactorizedInteractionWorkspace& workspace,
                   std::size_t layout_bytes, std::size_t persistent_value_bytes,
                   std::size_t gradient_bytes, double host_prepare_ms,
                   double layout_upload_ms, double activation_upload_ms,
                   double parameter_upload_ms, double allocation_ms,
                   double first_forward_vjp_event_ms, double first_forward_vjp_wall_ms,
                   double warm_batch_wall_ms,
                   const std::vector<double>& parameter_refresh_ms, const std::vector<double>& forward_ms,
                   const std::vector<double>& backward_ms,
                   const std::vector<double>& step_ms) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open timing metrics output: " + path);
    const auto vector_json = [](const std::vector<double>& v) {
        std::ostringstream o; o << std::setprecision(9) << "[";
        for (std::size_t i = 0; i < v.size(); ++i) { if (i) o << ","; o << v[i]; }
        o << "]"; return o.str();
    };
    const auto setup_ms = host_prepare_ms + layout_upload_ms + activation_upload_ms
        + parameter_upload_ms + allocation_ms;
    const auto resident_step_p50 = percentile(step_ms, 0.50);
    output << std::setprecision(9)
           << "{\n"
           << "  \"schema\": \"cellerator.factorized_interactions.metrics.v1\",\n"
           << "  \"scheduler_job_id\": \"" << (job_id ? job_id : "") << "\",\n"
           << "  \"route\": \"" << route << "\",\n"
           << "  \"selected_backend\": \"" << backend_name(backend) << "\",\n"
           << "  \"candidate\": \"" << requested_candidate << "\",\n"
           << "  \"actual_implementation\": \"" << actual_implementation << "\",\n"
           << "  \"structural_layout\": \"" << structural_layout << "\",\n"
           << "  \"gradient_mode\": \"" << gradient_mode << "\",\n"
           << "  \"device_uuid\": \"" << uuid << "\",\n"
           << "  \"compute_capability\": \"" << properties.major << '.'
           << properties.minor << "\",\n"
           << "  \"cells\": " << header.cells << ",\n"
           << "  \"channels\": " << header.channels << ",\n"
           << "  \"mechanisms\": " << header.mechanisms << ",\n"
           << "  \"iterations\": " << forward_ms.size() << ",\n"
           << "  \"phase_ms\": {\n"
           << "    \"host_and_device_setup_plus_initial_conversion\": " << setup_ms << ",\n"
           << "    \"host_preparation\": " << host_prepare_ms << ",\n"
           << "    \"device_layout_upload\": " << layout_upload_ms << ",\n"
           << "    \"activation_and_cotangent_conversion\": " << activation_upload_ms << ",\n"
           << "    \"parameter_initialization\": " << parameter_upload_ms << ",\n"
           << "    \"device_buffer_allocation_and_seed\": " << allocation_ms << ",\n"
           << "    \"parameter_refresh\": " << percentile(parameter_refresh_ms, 0.50) << ",\n"
           << "    \"first_forward_vjp_event\": " << first_forward_vjp_event_ms << ",\n"
           << "    \"first_forward_vjp_host_wall\": " << first_forward_vjp_wall_ms << ",\n"
           << "    \"warm_forward_vjp_batch_host_wall\": " << warm_batch_wall_ms << ",\n"
           << "    \"warm_forward_vjp_batch_host_wall_mean\": " << warm_batch_wall_ms / forward_ms.size() << ",\n"
           << "    \"forward_p50\": " << percentile(forward_ms, 0.50) << ",\n"
           << "    \"backward_and_gradient_reductions_p50\": "
           << percentile(backward_ms, 0.50) << ",\n"
           << "    \"resident_forward_backward_p50\": " << resident_step_p50 << ",\n"
           << "    \"estimated_setup_plus_warm_forward_vjp_p50\": "
           << setup_ms + resident_step_p50 << ",\n"
           << "    \"estimated_warm_forward_vjp_plus_parameter_refresh_p50\": "
           << percentile(parameter_refresh_ms, 0.50) + resident_step_p50 << "\n"
           << "  },\n"
           << "  \"timing_scope\": \"forward and VJP including tape and reductions; refresh reuploads identical parameters including pinned staging allocation, memcpy and three serial H2D completions; no optimizer; combined values are analytical estimates\",\n"
           << "  \"samples_ms\": {\n"
           << "    \"forward\": " << vector_json(forward_ms) << ",\n"
           << "    \"backward\": " << vector_json(backward_ms) << ",\n"
           << "    \"forward_vjp\": " << vector_json(step_ms) << ",\n"
           << "    \"parameter_refresh\": " << vector_json(parameter_refresh_ms) << "\n"
           << "  },\n"
           << "  \"phase_p95_ms\": {\n"
           << "    \"forward\": " << percentile(forward_ms, 0.95) << ",\n"
           << "    \"backward_and_gradient_reductions\": "
           << percentile(backward_ms, 0.95) << ",\n"
           << "    \"resident_forward_backward\": " << percentile(step_ms, 0.95) << "\n"
           << "  },\n"
           << "  \"reduction_accounting\": {\n"
           << "    \"output_reduction_in_forward\": true,\n"
           << "    \"parameter_and_context_reductions_in_backward\": true,\n"
           << "    \"separately_timed\": false\n"
           << "  },\n"
           << "  \"memory_bytes\": {\n"
           << "    \"persistent_layout\": " << layout_bytes << ",\n"
           << "    \"resident_values_and_cotangent\": " << persistent_value_bytes << ",\n"
           << "    \"gradient_destinations\": " << gradient_bytes << ",\n"
           << "    \"tape\": " << workspace.tape_bytes << ",\n"
           << "    \"forward_scratch\": " << workspace.forward_bytes << ",\n"
           << "    \"backward_scratch\": " << workspace.backward_bytes << "\n"
           << "  }\n"
           << "}\n";
    if (!output) throw std::runtime_error("failed while writing timing metrics");
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto input = factorized_wire::input_paths(argc, argv);
        const auto output = argument(argc, argv, "--output");
        const auto route = argument(argc, argv, "--route");
        requested_candidate = argument(argc, argv, "--candidate");
        if (requested_candidate != "native_direct" && requested_candidate != "packet" && requested_candidate != "minimal_direct" && requested_candidate != "direct_unique_output")
            throw std::invalid_argument("candidate must be native_direct, packet, minimal_direct, or direct_unique_output");
        if ((requested_candidate == "packet") != (route == "packet"))
            throw std::invalid_argument("candidate and route do not match");
        const auto gradient_mode = argument(argc, argv, "--gradient-mode", "overwrite");
        const auto device_text = argument(argc, argv, "--device", "0");
        const auto iteration_text = argument(argc, argv, "--iterations", "0");
        const auto metrics = argument(argc, argv, "--metrics");
        const auto expected_uuid = argument(argc, argv, "--expected-uuid");
        const bool repeat_vjp = has_flag(argc, argv, "--repeat-vjp");
        if (input.empty() || output.empty() || route.empty()) {
            std::cerr << "usage: candidate_runner --candidate NAME --input CASE [--input-shard PART ...] --output RESULT "
                         "--route packet|direct [--gradient-mode overwrite|accumulate] "
                         "[--repeat-vjp] [--device VISIBLE_DEVICE] "
                         "--expected-uuid GPU-... [--iterations N --metrics FILE]\n";
            return 2;
        }
        const int device = std::stoi(device_text);
        const auto iterations = static_cast<std::uint32_t>(std::stoul(iteration_text));
        return run(input, output, route, gradient_mode, device, iterations, metrics,
                   std::getenv("AUTOBENCH_SCHEDULER_JOB_ID"), expected_uuid, repeat_vjp);
    } catch (const std::exception& error) {
        std::cerr << "native_runner: " << error.what() << '\n';
        return 1;
    }
}
