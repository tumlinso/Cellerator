// Wire/parser/output fork of the existing native runner; see PROVENANCE.json.
#include <Cellerator/layout/factorized_interaction.hh>
#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/modeling/cuda/factorized_interaction.hh>
#include "../../wire_input.hh"
#include "circuit_kernels.cuh"

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

namespace {

constexpr std::array<char, 8> input_magic{{'C','F','I','N','T','0','0','1'}};
constexpr std::array<char, 8> output_magic{{'C','F','O','U','T','0','0','1'}};
constexpr float accumulate_seed = 0.125f;

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
    const auto bytes = factorized_wire::read_input_bytes(paths);
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

using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

const char* environment(const char* name) {
    const auto* value = std::getenv(name);
    return value ? value : "";
}

std::string uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream result;
    result << "GPU-" << std::hex << std::setfill('0');
    for (unsigned i = 0; i < sizeof(uuid.bytes); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) result << '-';
        result << std::setw(2) << unsigned(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return result.str();
}

bool validate_motif(const HostCase& host) {
    const auto& h = host.header;
    if ((h.cells != 64 && h.cells != 1024) || h.channels != 32 || h.inputs != 10 ||
        h.contexts != 2 || h.outputs != 1 || h.alpha_rows != 4 || h.biases != 3 ||
        h.gammas != 4 || h.mechanisms != 8 || h.terms != 16 ||
        h.activation_type != 1 || h.parameter_type != 1 || h.output_type != 2)
        throw std::invalid_argument("unsupported shape/dtype: this is only the fixed circuit probe");
    const bool repeated = host.mechanisms[0].inputs[2] == 0;
    for (std::size_t m = 0; m < 8; ++m) {
        const auto& record = host.mechanisms[m];
        const auto tail = repeated && m == 0 ? 0 : 2 + m;
        if (record.logical_id != m + 1 || record.target != 0 || record.alpha != m % 4 ||
            record.bias != m % 3 || record.arity != 3 ||
            record.inputs != std::array<std::uint64_t, 4>{0, 1, tail, 0} ||
            record.terms.size() != 2 || record.terms[0].context != 0 ||
            record.terms[0].gamma != m % 4 || record.terms[1].context != 1 ||
            record.terms[1].gamma != (m + 1) % 4)
            throw std::invalid_argument("unsupported topology: mechanism differs from frozen motif");
    }
    for (const auto* values : {&host.x, &host.z, &host.alpha, &host.bias, &host.gamma}) {
        for (std::size_t i = 0; i < values->size(); i += 2) {
            std::uint16_t bits;
            std::memcpy(&bits, values->data() + i, 2);
            if ((bits & 0x7c00u) == 0x7c00u)
                throw std::invalid_argument("prototype requires finite FP16 primals/parameters");
        }
    }
    for (std::size_t i = 0; i < host.dy.size(); i += 4) {
        float value;
        std::memcpy(&value, host.dy.data() + i, 4);
        if (!std::isfinite(value)) throw std::invalid_argument("prototype requires finite FP32 dy");
    }
    return repeated;
}

struct InputOwner {
    memory_cuda::DeviceBuffer x, z, alpha, bias, gamma, dy;
    double allocation_ms = 0, activation_upload_ms = 0, parameter_upload_ms = 0;
    InputOwner(const HostCase& host, int device, memory_cuda::Stream& stream)
        : x(host.x.size(), device), z(host.z.size(), device),
          alpha(host.alpha.size(), device), bias(host.bias.size(), device),
          gamma(host.gamma.size(), device), dy(host.dy.size(), device) {
        const auto activation_begin = Clock::now();
        upload(x, host.x, stream); upload(z, host.z, stream); upload(dy, host.dy, stream);
        activation_upload_ms = milliseconds(activation_begin, Clock::now());
        const auto parameter_begin = Clock::now();
        upload(alpha, host.alpha, stream); upload(bias, host.bias, stream);
        upload(gamma, host.gamma, stream);
        parameter_upload_ms = milliseconds(parameter_begin, Clock::now());
    }
    interaction::FactorizedInteractionValues values() const {
        return {x.view(), z.view(), alpha.view(), bias.view(), gamma.view()};
    }
    shared_product_circuit::Primals primals() const {
        return {static_cast<const __half*>(x.data()), static_cast<const __half*>(z.data()),
                static_cast<const __half*>(alpha.data()), static_cast<const __half*>(bias.data()),
                static_cast<const __half*>(gamma.data()), static_cast<const float*>(dy.data())};
    }
    std::size_t bytes() const {
        return x.bytes() + z.bytes() + alpha.bytes() + bias.bytes() + gamma.bytes() + dy.bytes();
    }
};

struct GradientOwner {
    memory_cuda::DeviceBuffer dx, dz, dalpha, dbias, dgamma, backward;
    GradientOwner(int cells, int device, std::size_t backward_bytes,
                  memory_cuda::Stream& stream)
        : dx(std::size_t(10) * 32 * cells * 4, device), dz(std::size_t(2) * cells * 4, device),
          dalpha(4 * 32 * 4, device), dbias(3 * 4, device), dgamma(4 * 4, device),
          backward(backward_bytes, device) {
        for (auto* buffer : {&dx, &dz, &dalpha, &dbias, &dgamma})
            seed_gradient(*buffer, buffer->bytes(), stream);
    }
    interaction::FactorizedInteractionGradients views() {
        return {dx.view(), dz.view(), dalpha.view(), dbias.view(), dgamma.view()};
    }
    std::size_t bytes() const {
        return dx.bytes() + dz.bytes() + dalpha.bytes() + dbias.bytes() + dgamma.bytes();
    }
};

struct Arm {
    std::string name, backend;
    int cells, device;
    bool repeated;
    interaction::FactorizedInteractionWorkspace workspace{};
    std::unique_ptr<interaction::FactorizedInteractionDeviceLayout> layout;
    memory_cuda::DeviceBuffer output, tape, forward_scratch;
    std::unique_ptr<GradientOwner> gradient;
    double preparation_ms = 0, layout_upload_ms = 0, allocation_ms = 0;

    Arm(std::string candidate, const HostCase& host, bool repeated_tail,
        int selected_device, memory_cuda::Stream& stream)
        : name(std::move(candidate)), cells(static_cast<int>(host.header.cells)),
          device(selected_device), repeated(repeated_tail) {
        if (name == "native_direct") {
            const auto preparation_begin = Clock::now();
            const layout::FactorizedInteractionExtents extents{10, 2, 1, 4, 3, 4};
            auto prepared = layout::prepare_factorized_interactions(host.mechanisms, extents);
            layout::validate_factorized_interactions(prepared);
            preparation_ms = milliseconds(preparation_begin, Clock::now());
            const auto upload_begin = Clock::now();
            layout = std::make_unique<interaction::FactorizedInteractionDeviceLayout>(
                prepared, device, stream);
            stream.synchronize();
            layout_upload_ms = milliseconds(upload_begin, Clock::now());
            interaction::FactorizedInteractionBackend actual{};
            memory_cuda::check(interaction::factorized_interaction_backend(
                prepared.view(), {static_cast<std::uint64_t>(cells), 32},
                interaction::FactorizedInteractionRoute::direct, &actual), "native backend");
            backend = actual == interaction::FactorizedInteractionBackend::stable ? "stable" :
                      actual == interaction::FactorizedInteractionBackend::direct ? "direct" : "packet";
            memory_cuda::check(interaction::factorized_interaction_workspace(
                layout->view(), {static_cast<std::uint64_t>(cells), 32},
                interaction::FactorizedInteractionRoute::direct, &workspace), "native workspace");
        } else {
            backend = name == "circuit" ? "shared_product_circuit" : "literal_unfactored";
            workspace.tape_bytes = std::size_t(2) * 8 * cells * sizeof(float);
            workspace.forward_bytes = 0;
            workspace.backward_bytes = std::size_t(4 * 32 + 8 * 32 + 8) * cells *
                                       sizeof(shared_product_circuit::Accurate);
        }
        const auto allocation_begin = Clock::now();
        output = memory_cuda::DeviceBuffer(std::size_t(32) * cells * 4, device);
        tape = memory_cuda::DeviceBuffer(workspace.tape_bytes, device);
        forward_scratch = memory_cuda::DeviceBuffer(workspace.forward_bytes, device);
        gradient = std::make_unique<GradientOwner>(cells, device, workspace.backward_bytes, stream);
        memory_cuda::check(cudaMemsetAsync(output.data(), 0xa5, output.bytes(), stream.get()),
                           "poison output");
        stream.synchronize();
        allocation_ms = milliseconds(allocation_begin, Clock::now());
    }

    shared_product_circuit::Scratch scratch(GradientOwner& owner) {
        auto* partials = static_cast<shared_product_circuit::Accurate*>(owner.backward.data());
        return {static_cast<float*>(tape.data()), partials,
                partials + std::size_t(4) * 32 * cells,
                partials + std::size_t(4 + 8) * 32 * cells};
    }
    shared_product_circuit::Destinations destinations(GradientOwner& owner) {
        return {static_cast<float*>(output.data()), static_cast<float*>(owner.dx.data()),
                static_cast<float*>(owner.dz.data()), static_cast<float*>(owner.dalpha.data()),
                static_cast<float*>(owner.dbias.data()), static_cast<float*>(owner.dgamma.data())};
    }
    void forward(const InputOwner& input, memory_cuda::Stream& stream) {
        cudaError_t result;
        if (name == "native_direct") {
            result = interaction::factorized_interaction_forward(
                layout->view(), {static_cast<std::uint64_t>(cells), 32},
                {memory_cuda::DType::float16, memory_cuda::DType::float16, memory_cuda::DType::float32},
                input.values(), output.view(), tape.view(), forward_scratch.view(), stream.get(),
                interaction::FactorizedInteractionRoute::direct);
        } else if (name == "circuit") {
            result = shared_product_circuit::launch_forward<true>(
                input.primals(), cells, repeated, scratch(*gradient), destinations(*gradient), stream.get());
        } else {
            result = shared_product_circuit::launch_forward<false>(
                input.primals(), cells, repeated, scratch(*gradient), destinations(*gradient), stream.get());
        }
        memory_cuda::check(result, "prototype forward");
    }
    void backward(const InputOwner& input, GradientOwner& owner, bool accumulate,
                  memory_cuda::Stream& stream) {
        cudaError_t result;
        if (name == "native_direct") {
            result = interaction::factorized_interaction_backward(
                layout->view(), {static_cast<std::uint64_t>(cells), 32},
                {memory_cuda::DType::float16, memory_cuda::DType::float16, memory_cuda::DType::float32},
                input.values(), input.dy.view(), tape.view(), owner.views(), owner.backward.view(),
                accumulate ? interaction::FactorizedInteractionGradientMode::accumulate
                           : interaction::FactorizedInteractionGradientMode::overwrite,
                stream.get(), interaction::FactorizedInteractionRoute::direct);
        } else if (name == "circuit") {
            result = shared_product_circuit::launch_backward<true>(
                input.primals(), cells, repeated, scratch(owner), destinations(owner), accumulate, stream.get());
        } else {
            result = shared_product_circuit::launch_backward<false>(
                input.primals(), cells, repeated, scratch(owner), destinations(owner), accumulate, stream.get());
        }
        memory_cuda::check(result, "prototype backward");
    }
    void step(const InputOwner& input, memory_cuda::Stream& stream) {
        forward(input, stream);
        backward(input, *gradient, false, stream);
    }
};

void memory_json(std::ostream& out, const Arm& arm, const InputOwner& input,
                 bool repeat = false) {
    out << "{\"persistent_layout\":" << (arm.layout ? arm.layout->bytes() : 0)
        << ",\"resident_values_and_cotangent\":" << input.bytes()
        << ",\"output\":" << arm.output.bytes()
        << ",\"gradient_destinations\":" << arm.gradient->bytes()
        << ",\"tape\":" << arm.workspace.tape_bytes
        << ",\"forward_scratch\":" << arm.workspace.forward_bytes
        << ",\"backward_scratch\":" << arm.workspace.backward_bytes
        << ",\"repeat_gradient_and_scratch\":"
        << (repeat ? arm.gradient->bytes() + arm.workspace.backward_bytes : 0) << '}';
}

void setup_json(std::ostream& out, const Arm& arm, const InputOwner& input) {
    out << "{\"host_preparation\":" << arm.preparation_ms
        << ",\"device_layout_upload\":" << arm.layout_upload_ms
        << ",\"arm_allocation_and_seed\":" << arm.allocation_ms
        << ",\"shared_value_allocation\":" << input.allocation_ms
        << ",\"shared_activation_and_cotangent_upload\":" << input.activation_upload_ms
        << ",\"shared_parameter_upload\":" << input.parameter_upload_ms << '}';
}

void identity_json(std::ostream& out, const std::string& uuid) {
    out << "\"scheduler_job_id\":\"" << environment("AUTOBENCH_SCHEDULER_JOB_ID")
        << "\",\"scheduler_attempt_id\":\"" << environment("AUTOBENCH_SCHEDULER_ATTEMPT_ID")
        << "\",\"device_uuid\":\"" << uuid << '"';
}

void metadata(const std::string& path, const Arm& arm, const InputOwner& input,
              const std::string& uuid, bool accumulate, bool repeat) {
    std::ofstream out(path + ".metadata.json");
    if (!out) throw std::runtime_error("cannot open prototype metadata");
    out << std::setprecision(17)
        << "{\"schema\":\"cellerator.shared_product_circuit.runner.v1\",";
    identity_json(out, uuid);
    out << ",\"requested_candidate\":\"" << arm.name
        << "\",\"selected_backend\":\"" << arm.backend
        << "\",\"gradient_mode\":\"" << (accumulate ? "accumulate" : "overwrite")
        << "\",\"gradient_seed\":" << (accumulate ? accumulate_seed : 0.0f)
        << ",\"repeat_vjp\":" << (repeat ? "true" : "false")
        << ",\"cells\":" << arm.cells
        << ",\"channels\":32,\"mechanisms\":8,\"terms\":16,\"repeated_tail\":"
        << (arm.repeated ? "true" : "false") << ",\"memory_bytes\":";
    memory_json(out, arm, input, repeat);
    out << ",\"setup_ms\":";
    setup_json(out, arm, input);
    out << "}\n";
    if (!out) throw std::runtime_error("failed writing prototype metadata");
}

OutputArrays download_owner(const Arm& arm, const GradientOwner& owner,
                           memory_cuda::Stream& stream, bool include_y) {
    OutputArrays host{memory_cuda::PinnedBuffer(include_y ? arm.output.bytes() : 0),
                     memory_cuda::PinnedBuffer(owner.dx.bytes()), memory_cuda::PinnedBuffer(owner.dz.bytes()),
                     memory_cuda::PinnedBuffer(owner.dalpha.bytes()), memory_cuda::PinnedBuffer(owner.dbias.bytes()),
                     memory_cuda::PinnedBuffer(owner.dgamma.bytes())};
    if (include_y) download(arm.output, host.y, arm.output.bytes(), stream);
    download(owner.dx, host.dx, owner.dx.bytes(), stream);
    download(owner.dz, host.dz, owner.dz.bytes(), stream);
    download(owner.dalpha, host.dalpha, owner.dalpha.bytes(), stream);
    download(owner.dbias, host.dbias, owner.dbias.bytes(), stream);
    download(owner.dgamma, host.dgamma, owner.dgamma.bytes(), stream);
    stream.synchronize();
    return host;
}

void write_arm(const std::string& path, const HostCase& host, const Arm& arm,
               const InputOwner& input, const std::string& uuid, bool accumulate,
               GradientOwner* repeat, memory_cuda::Stream& stream) {
    auto primary = download_owner(arm, *arm.gradient, stream, true);
    std::unique_ptr<OutputArrays> secondary;
    if (repeat) secondary = std::make_unique<OutputArrays>(download_owner(arm, *repeat, stream, false));
    write_result(path, host.header, primary.y.data(), primary, secondary.get());
    metadata(path, arm, input, uuid, accumulate, repeat != nullptr);
}

struct Observation {
    int block, position;
    Arm* arm;
    double elapsed;
};
struct Diagnostics {
    Arm* arm;
    std::vector<double> forward, backward, full;
};
void vector_json(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

int run(const std::vector<std::string>& paths, const std::string& output_path,
        const std::string& candidate, const std::string& mode, bool repeat,
        const std::string& metrics_path, const std::string& expected_uuid, int device) {
    if (!*environment("AUTOBENCH_SCHEDULER_JOB_ID") ||
        !*environment("AUTOBENCH_SCHEDULER_ATTEMPT_ID"))
        throw std::runtime_error("CUDA execution requires the admitted Autobench job/attempt context");
    if (mode != "overwrite" && mode != "accumulate")
        throw std::invalid_argument("gradient mode must be overwrite or accumulate");
    if (candidate != "circuit" && candidate != "unfactored" &&
        candidate != "native_direct" && candidate != "paired")
        throw std::invalid_argument("unknown prototype candidate");
    HostCase host = read_case(paths);
    const bool repeated = validate_motif(host);
    const bool paired = candidate == "paired";
    if (paired && (host.header.cells != 1024 || repeated || mode != "overwrite" ||
                   repeat || metrics_path.empty()))
        throw std::invalid_argument("paired mode requires C1024 canonical motif, overwrite, metrics and one VJP");
    memory_cuda::DeviceGuard guard(device);
    cudaDeviceProp properties{};
    memory_cuda::check(cudaGetDeviceProperties(&properties, device), "query CUDA UUID");
    const auto uuid = uuid_string(properties.uuid);
    if (expected_uuid.empty() || uuid != expected_uuid)
        throw std::runtime_error("visible device UUID differs from explicit admitted UUID");
    if (properties.major != 7 || properties.minor != 0)
        throw std::runtime_error("this bounded prototype targets SM70 only");
    memory_cuda::Stream stream(device);
    const auto input_begin = Clock::now();
    InputOwner input(host, device, stream);
    input.allocation_ms = milliseconds(input_begin, Clock::now()) -
                          input.activation_upload_ms - input.parameter_upload_ms;
    std::vector<std::unique_ptr<Arm>> arms;
    for (const auto& name : (paired ? std::vector<std::string>{"unfactored", "circuit", "native_direct"}
                                   : std::vector<std::string>{candidate}))
        arms.push_back(std::make_unique<Arm>(name, host, repeated, device, stream));
    if (!paired) {
        auto& arm = *arms.front();
        arm.forward(input, stream);
        arm.backward(input, *arm.gradient, mode == "accumulate", stream);
        std::unique_ptr<GradientOwner> second;
        if (repeat) {
            second = std::make_unique<GradientOwner>(arm.cells, device, arm.workspace.backward_bytes, stream);
            arm.backward(input, *second, mode == "accumulate", stream);
        }
        write_arm(output_path, host, arm, input, uuid, mode == "accumulate", second.get(), stream);
        return 0;
    }

    constexpr int warmups = 10, blocks = 7, steps = 30;
    for (auto& arm : arms) {
        for (int i = 0; i < warmups; ++i) arm->step(input, stream);
        stream.synchronize();
    }
    std::vector<Observation> observations;
    std::vector<std::array<Arm*, 3>> orders;
    for (int block = 0; block < blocks; ++block) {
        Arm* first = arms[block % 2 == 0 ? 0 : 1].get();
        Arm* second = arms[block % 2 == 0 ? 1 : 0].get();
        const int native_position = block % 3;
        std::array<Arm*, 3> order{};
        int pair_index = 0;
        for (int position = 0; position < 3; ++position)
            order[position] = position == native_position ? arms[2].get()
                             : pair_index++ == 0 ? first : second;
        orders.push_back(order);
        for (int position = 0; position < 3; ++position) {
            auto& arm = *order[position];
            // No events, allocation, upload, output download or file writes in
            // this host-wall interval. Each preceding arm ended with sync.
            const auto begin = Clock::now();
            for (int i = 0; i < steps; ++i) arm.step(input, stream);
            stream.synchronize();
            observations.push_back({block, position, &arm, milliseconds(begin, Clock::now())});
        }
    }
    std::vector<Diagnostics> diagnostics;
    for (auto& arm : arms) {
        Diagnostics row{arm.get(), {}, {}, {}};
        memory_cuda::Event begin(device, true), middle(device, true), end(device, true);
        for (int sample = 0; sample < 5; ++sample) {
            begin.record(stream);
            arm->forward(input, stream);
            middle.record(stream);
            arm->backward(input, *arm->gradient, false, stream);
            end.record(stream);
            stream.synchronize();
            row.forward.push_back(begin.elapsed_ms(middle));
            row.backward.push_back(middle.elapsed_ms(end));
            row.full.push_back(begin.elapsed_ms(end));
        }
        diagnostics.push_back(std::move(row));
    }
    for (auto& arm : arms)
        write_arm(output_path + "." + arm->name + ".cfout", host, *arm, input,
                  uuid, false, nullptr, stream);
    std::ofstream out(metrics_path);
    if (!out) throw std::runtime_error("cannot open paired metrics");
    out << std::setprecision(17)
        << "{\"schema\":\"cellerator.shared_product_circuit.paired.v1\",";
    identity_json(out, uuid);
    out << ",\"cells\":1024,\"channels\":32,\"mechanisms\":8,\"terms\":16,"
           "\"gradient_mode\":\"overwrite\",\"warmup_iterations_per_arm\":10,"
           "\"block_count\":7,\"iterations_per_block\":30,\"actual_order\":[";
    for (std::size_t b = 0; b < orders.size(); ++b) {
        if (b) out << ',';
        out << '[';
        for (unsigned p = 0; p < 3; ++p) {
            if (p) out << ',';
            out << '"' << orders[b][p]->name << '"';
        }
        out << ']';
    }
    out << "],\"measurements\":[";
    for (std::size_t i = 0; i < observations.size(); ++i) {
        if (i) out << ',';
        const auto& row = observations[i];
        out << "{\"block\":" << row.block << ",\"position\":" << row.position
            << ",\"candidate\":\"" << row.arm->name << "\",\"selected_backend\":\""
            << row.arm->backend << "\",\"host_block_ms\":" << row.elapsed
            << ",\"iterations\":30,\"host_mean_ms\":" << row.elapsed / steps << '}';
    }
    out << "],\"diagnostics\":[";
    for (std::size_t i = 0; i < diagnostics.size(); ++i) {
        if (i) out << ',';
        const auto& row = diagnostics[i];
        out << "{\"candidate\":\"" << row.arm->name << "\",\"selected_backend\":\""
            << row.arm->backend << "\",\"forward_ms\":";
        vector_json(out, row.forward); out << ",\"backward_ms\":";
        vector_json(out, row.backward); out << ",\"full_ms\":";
        vector_json(out, row.full); out << '}';
    }
    out << "],\"arms\":[";
    for (std::size_t i = 0; i < arms.size(); ++i) {
        if (i) out << ',';
        const auto& arm = *arms[i];
        // Paths are intentionally omitted here: the driver owns the selected
        // prefix and hashes the three fixed-name artifacts independently.
        out << "{\"candidate\":\"" << arm.name << "\",\"selected_backend\":\""
            << arm.backend << "\",\"memory_bytes\":";
        memory_json(out, arm, input); out << ",\"setup_ms\":";
        setup_json(out, arm, input); out << '}';
    }
    out << "]}\n";
    if (!out) throw std::runtime_error("failed writing paired metrics");
    return 0;
}

std::string argument(int argc, char** argv, const std::string& key,
                     const std::string& fallback = {}) {
    for (int i = 1; i + 1 < argc; ++i) if (argv[i] == key) return argv[i + 1];
    return fallback;
}
bool flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) if (argv[i] == key) return true;
    return false;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto paths = factorized_wire::input_paths(argc, argv);
        const auto output = argument(argc, argv, "--output");
        const auto candidate = argument(argc, argv, "--candidate");
        if (paths.empty() || output.empty() || candidate.empty())
            throw std::invalid_argument("usage: circuit_runner --input CASE --output RESULT "
                "--candidate unfactored|circuit|native_direct|paired --expected-uuid GPU-... "
                "[--gradient-mode overwrite|accumulate] [--repeat-vjp] [--metrics FILE]");
        if (std::stod(argument(argc, argv, "--seed-gradient", "0.125")) != accumulate_seed)
            throw std::invalid_argument("the fixed accumulation seed is 0.125");
        return run(paths, output, candidate, argument(argc, argv, "--gradient-mode", "overwrite"),
                   flag(argc, argv, "--repeat-vjp"), argument(argc, argv, "--metrics"),
                   argument(argc, argv, "--expected-uuid"),
                   std::stoi(argument(argc, argv, "--device", "0")));
    } catch (const std::exception& error) {
        std::cerr << "shared_product_circuit_runner: " << error.what() << '\n';
        return 1;
    }
}

