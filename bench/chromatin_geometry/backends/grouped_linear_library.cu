#include "grouped_linear_library.hh"
#include "consumer_fixture.hh"
#include "../reference/reference.hh"

#include <Cellerator/memory/cuda/resources.hh>
#include <cublas_v2.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cgs::backends {
namespace {
namespace mc = ::cellerator::memory::cuda;
using Incidence = cellerator::ops::mechanism::OutputIncidence;

struct DeviceGroup {
    std::uint64_t argument_begin;
    std::uint64_t output_begin;
    std::uint64_t parameter_index;
    std::uint64_t condition_index;
    std::uint64_t group_weight_begin;
};

struct DeviceGate {
    const float* probabilities;
    const std::uint8_t* admitted;
    const std::uint32_t* primal_words;
    const std::uint32_t* derivative_words;
    std::uint64_t conditions;
    std::uint64_t words_per_condition;
    GateProfile profile;
};

std::size_t checked_product(std::uint64_t a, std::uint64_t b,
                            const char* what) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument(std::string(what) + " size overflows");
    return static_cast<std::size_t>(a * b);
}

std::size_t dtype_bytes(DType dtype) noexcept {
    return dtype == DType::fp16 ? sizeof(__half) : sizeof(float);
}

void* offset_pointer(void* base, std::uint64_t elements, DType dtype) {
    return static_cast<char*>(base) + checked_product(elements, dtype_bytes(dtype),
                                                       "linear buffer offset");
}

const void* offset_pointer(const void* base, std::uint64_t elements,
                           DType dtype) {
    return static_cast<const char*>(base) +
           checked_product(elements, dtype_bytes(dtype), "linear buffer offset");
}

class DeviceArray final {
public:
    DeviceArray() = default;
    DeviceArray(std::size_t bytes, int device) : bytes_(bytes), device_(device) {
        if (!bytes_) return;
        mc::DeviceGuard guard(device_);
        mc::check(cudaMalloc(&data_, bytes_), "allocate library route metadata");
    }
    ~DeviceArray() noexcept { release(); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&& other) noexcept { swap(other); }
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            release();
            swap(other);
        }
        return *this;
    }
    void upload(const void* source, std::size_t bytes) {
        if (bytes > bytes_) throw std::invalid_argument("library metadata exceeds allocation");
        if (!bytes) return;
        mc::DeviceGuard guard(device_);
        mc::check(cudaMemcpy(data_, source, bytes, cudaMemcpyHostToDevice),
                  "copy library route metadata");
    }
    void* data() noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }
private:
    void release() noexcept {
        if (!data_) return;
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = have_previous && previous != device_ &&
                             cudaSetDevice(device_) == cudaSuccess;
        (void)cudaFree(data_);
        if (changed) (void)cudaSetDevice(previous);
        data_ = nullptr;
        bytes_ = 0;
    }
    void swap(DeviceArray& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(bytes_, other.bytes_);
        std::swap(device_, other.device_);
    }
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
    int device_ = 0;
};

__device__ __forceinline__ bool plane_bit(const std::uint32_t* words,
                                          std::uint64_t condition,
                                          std::uint64_t cell,
                                          std::uint64_t stride) {
    if (!words) return false;
    const auto word = words[condition * stride + cell / 32];
    return ((word >> (cell % 32)) & 1u) != 0;
}

__device__ __forceinline__ bool primal_active(const DeviceGate& gate,
                                               std::uint64_t cell,
                                               std::uint64_t condition) {
    if (gate.admitted)
        return gate.admitted[cell * gate.conditions + condition] != 0;
    return plane_bit(gate.primal_words, condition, cell,
                     gate.words_per_condition);
}

__device__ __forceinline__ bool derivative_active(const DeviceGate& gate,
                                                   std::uint64_t cell,
                                                   std::uint64_t condition) {
    if (gate.derivative_words)
        return plane_bit(gate.derivative_words, condition, cell,
                         gate.words_per_condition);
    return primal_active(gate, cell, condition);
}

__device__ __forceinline__ float gate_scale(const DeviceGate& gate,
                                             std::uint64_t cell,
                                             std::uint64_t condition,
                                             bool admitted) {
    if (!admitted) return 0.0f;
    if (gate.profile == GateProfile::ste)
        return gate.probabilities[cell * gate.conditions + condition];
    return 1.0f;
}

__device__ __forceinline__ float load_value(const void* ptr,
                                             std::uint64_t index,
                                             DType dtype) {
    if (dtype == DType::fp16)
        return __half2float(static_cast<const __half*>(ptr)[index]);
    return static_cast<const float*>(ptr)[index];
}

__device__ __forceinline__ void store_value(void* ptr, std::uint64_t index,
                                             float value, DType dtype) {
    if (dtype == DType::fp16)
        static_cast<__half*>(ptr)[index] = __float2half_rn(value);
    else
        static_cast<float*>(ptr)[index] = value;
}

cudaDataType_t cublas_dtype(DType dtype) noexcept {
    return dtype == DType::fp16 ? CUDA_R_16F : CUDA_R_32F;
}

cublasStatus_t gemm(cublasHandle_t handle, DType dtype,
                    cublasOperation_t trans_a, cublasOperation_t trans_b,
                    int m, int n, int k, const void* a, int lda,
                    const void* b, int ldb, void* c, int ldc, float beta) {
    const float alpha = 1.0f;
    return cublasGemmEx(handle, trans_a, trans_b, m, n, k,
                        &alpha, a, cublas_dtype(dtype), lda,
                        b, cublas_dtype(dtype), ldb,
                        &beta, c, cublas_dtype(dtype), ldc,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

__global__ void pack_inputs(const DeviceGroup* groups, const std::uint64_t* args,
                            std::uint64_t group_count, std::uint64_t batch,
                            std::uint64_t input_features,
                            std::uint64_t coordinates, DeviceGate gate,
                            const void* x, DType dtype,
                            bool derivative_pass, void* packed,
                            std::uint32_t* status) {
    const std::uint64_t total = group_count * batch * input_features;
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const auto local = i % input_features;
    const auto row = i / input_features;
    const auto cell = row % batch;
    const auto group_index = row / batch;
    const auto& group = groups[group_index];
    const bool admitted = primal_active(gate, cell, group.condition_index);
    const float q = gate_scale(gate, cell, group.condition_index, admitted);
    const bool derivative = gate.profile == GateProfile::ste &&
                            derivative_active(gate, cell, group.condition_index);
    const bool read = derivative_pass ? derivative : (admitted && q != 0.0f);
    float value = 0.0f;
    if (read) {
        const auto coordinate = args[group.argument_begin + local];
        value = load_value(x, cell * coordinates + coordinate, dtype);
        if (!isfinite(value)) atomicOr(status, status_nonfinite);
    }
    store_value(packed, i, value, dtype);
}

__global__ void mark_group_activity(const DeviceGroup* groups,
                                    std::uint64_t group_count,
                                    std::uint64_t batch,
                                    DeviceGate gate, bool include_derivative,
                                    std::uint8_t* flags) {
    const auto group_index = static_cast<std::uint64_t>(blockIdx.x);
    if (group_index >= group_count) return;
    __shared__ unsigned primal;
    __shared__ unsigned derivative;
    if (threadIdx.x == 0) { primal = 0; derivative = 0; }
    __syncthreads();
    const auto& group = groups[group_index];
    for (std::uint64_t cell = threadIdx.x; cell < batch; cell += blockDim.x) {
        const bool admitted = primal_active(gate, cell, group.condition_index);
        const float q = gate_scale(gate, cell, group.condition_index, admitted);
        if (admitted && q != 0.0f) atomicExch(&primal, 1u);
        if (include_derivative && gate.profile == GateProfile::ste &&
            derivative_active(gate, cell, group.condition_index))
            atomicExch(&derivative, 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        flags[group_index * 2] = static_cast<std::uint8_t>(primal);
        flags[group_index * 2 + 1] = static_cast<std::uint8_t>(derivative);
    }
}

// The output kernel uses the actual canonical coordinate extent to make
// incidence indexing explicit.
__global__ void publish_forward_canonical(const DeviceGroup* groups,
                                          const Incidence* outputs,
                                          std::uint64_t group_count,
                                          std::uint64_t batch,
                                          std::uint64_t output_features,
                                          std::uint64_t coordinates,
                                          DeviceGate gate, DType dtype,
                                          const void* local_output, float* y,
                                          std::uint32_t* status) {
    const std::uint64_t total = group_count * batch * output_features;
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const auto output = i % output_features;
    const auto row = i / output_features;
    const auto cell = row % batch;
    const auto group_index = row / batch;
    const auto& group = groups[group_index];
    if (!primal_active(gate, cell, group.condition_index)) return;
    const float q = gate_scale(gate, cell, group.condition_index, true);
    if (q == 0.0f) return;
    const float f = load_value(local_output, i, dtype);
    const auto incidence = outputs[group.output_begin + output];
    if (incidence.coordinate >= coordinates) {
        atomicOr(status, status_invalid_descriptor);
        return;
    }
    if (!isfinite(q) || !isfinite(f) || !isfinite(incidence.scale))
        atomicOr(status, status_nonfinite);
    atomicAdd(y + cell * coordinates + incidence.coordinate,
              (q * f) * incidence.scale);
}

__global__ void build_output_gradient(const DeviceGroup* groups,
                                      const Incidence* outputs,
                                      std::uint64_t group_count,
                                      std::uint64_t batch,
                                      std::uint64_t output_features,
                                      std::uint64_t coordinates,
                                      DeviceGate gate, const float* dy,
                                      DType dtype, void* local_dy,
                                      std::uint32_t* status) {
    const std::uint64_t total = group_count * batch * output_features;
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const auto output = i % output_features;
    const auto row = i / output_features;
    const auto cell = row % batch;
    const auto group_index = row / batch;
    const auto& group = groups[group_index];
    const bool admitted = primal_active(gate, cell, group.condition_index);
    const float q = gate_scale(gate, cell, group.condition_index, admitted);
    float value = 0.0f;
    if (admitted && q != 0.0f) {
        const auto incidence = outputs[group.output_begin + output];
        if (incidence.coordinate >= coordinates) {
            atomicOr(status, status_invalid_descriptor);
        } else {
            const float upstream = dy[cell * coordinates + incidence.coordinate];
            value = (upstream * incidence.scale) * q;
            if (!isfinite(upstream) || !isfinite(incidence.scale) || !isfinite(q))
                atomicOr(status, status_nonfinite);
        }
    }
    // This is the local dF storage boundary used by the existing operator.
    store_value(local_dy, i, value, dtype);
}

__global__ void publish_backward(const DeviceGroup* groups,
                                 const std::uint64_t* args,
                                 std::uint64_t group_count,
                                 std::uint64_t batch,
                                 std::uint64_t input_features,
                                 std::uint64_t coordinates,
                                 std::uint64_t argument_count,
                                 DeviceGate gate,
                                 DType dtype, const void* local_dx,
                                 float* dx, void* dx_group,
                                 std::uint32_t* status) {
    const std::uint64_t local_cells = group_count * batch;
    const std::uint64_t total_dx = local_cells * input_features;
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total_dx) return;
    const auto local_input = i % input_features;
    const auto row = i / input_features;
    const auto cell = row % batch;
    const auto group_index = row / batch;
    const auto& group = groups[group_index];
    const bool admitted = primal_active(gate, cell, group.condition_index);
    const float q = gate_scale(gate, cell, group.condition_index, admitted);
    float local_value = 0.0f;
    if (admitted && q != 0.0f) {
        local_value = load_value(local_dx, i, dtype);
        const auto coordinate = args[group.argument_begin + local_input];
        if (coordinate >= coordinates) {
            atomicOr(status, status_invalid_descriptor);
        } else {
            atomicAdd(dx + cell * coordinates + coordinate, local_value);
        }
    }
    store_value(dx_group,
                cell * argument_count + group.argument_begin + local_input,
                local_value, dtype);
}

__global__ void accumulate_gate_gradient(const DeviceGroup* groups,
                                         const Incidence* outputs,
                                         std::uint64_t group_count,
                                         std::uint64_t batch,
                                         std::uint64_t output_features,
                                         std::uint64_t coordinates,
                                         std::uint64_t conditions,
                                         DeviceGate gate, const float* dy,
                                         DType dtype, const void* local_output,
                                         float* dq,
                                         std::uint32_t* status) {
    const std::uint64_t total = group_count * batch;
    const auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total || gate.profile != GateProfile::ste) return;
    const auto cell = i % batch;
    const auto group_index = i / batch;
    const auto& group = groups[group_index];
    if (!derivative_active(gate, cell, group.condition_index)) return;
    float local_dq = 0.0f;
    for (std::uint64_t o = 0; o < output_features; ++o) {
        const auto incidence = outputs[group.output_begin + o];
        if (incidence.coordinate >= coordinates) {
            atomicOr(status, status_invalid_descriptor);
            continue;
        }
        const float upstream = dy[cell * coordinates + incidence.coordinate];
        const float f = load_value(local_output,
                                   (group_index * batch + cell) * output_features + o,
                                   dtype);
        local_dq += (upstream * incidence.scale) * f;
        if (!isfinite(upstream) || !isfinite(f) || !isfinite(incidence.scale))
            atomicOr(status, status_nonfinite);
    }
    if (group.condition_index >= conditions) {
        atomicOr(status, status_invalid_descriptor);
    } else {
        atomicAdd(dq + cell * conditions + group.condition_index, local_dq);
    }
}

__global__ void aggregate_weight_gradients(const DeviceGroup* groups,
                                           const std::uint64_t* parameter_offsets,
                                           const std::uint8_t* activity,
                                           std::uint64_t group_count,
                                           std::uint64_t matrix_size,
                                           const void* group_dweights,
                                           DType dtype, float* dweights) {
    const auto total = group_count * matrix_size;
    const auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const auto group_index = i / matrix_size;
    if (!activity[group_index * 2]) return;
    const auto local_index = i % matrix_size;
    const auto& group = groups[group_index];
    atomicAdd(dweights + parameter_offsets[group.parameter_index] + local_index,
              load_value(group_dweights, group.group_weight_begin + local_index, dtype));
}

__global__ void store_status(std::uint32_t* status, std::uint32_t value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *status = value;
}

bool same_shape(const LinearLibraryLayout& l, const ProblemView& p,
                const void* expected_weights) noexcept {
    return p.batch == l.batch &&
           p.topology.group_count == l.groups.size() &&
           p.topology.coordinates == l.coordinates &&
           p.topology.conditions == l.conditions &&
           p.topology.parameters == l.parameter_count &&
           p.topology.argument_count == l.argument_count &&
           p.weight_count == l.weight_count &&
           p.group_weight_count == l.group_weight_count &&
           p.weights == expected_weights;
}

} // namespace

struct GroupedLinearLibraryControl::Impl {
    LinearLibraryLayout layout;
    DType dtype;
    LinearLibraryArm arm;
    const void* weights;
    std::uint64_t input_features = 0;
    std::uint64_t output_features = 0;
    std::size_t scratch_bytes = 0;
    std::size_t internal_device_bytes = 0;
    std::size_t pinned_host_bytes = 0;
    GateRevisions forward_revisions{};
    const void* forward_x = nullptr;
    cudaStream_t forward_stream = nullptr;
    bool have_forward = false;
    DeviceArray device_groups;
    DeviceArray device_arguments;
    DeviceArray device_outputs;
    DeviceArray device_parameter_offsets;
    DeviceArray device_activity;
    DeviceArray device_batch_pointers;
    mc::PinnedBuffer host_activity;
    mc::PinnedBuffer host_batch_pointers;
    cublasHandle_t handle = nullptr;

    Impl(LinearLibraryLayout value, const ProblemView& problem, DType type,
         LinearLibraryArm execution_arm)
        : layout(std::move(value)), dtype(type), arm(execution_arm), weights(problem.weights) {
        if (layout.device < 0 ||
            layout.parameter_weight_offsets.size() != layout.parameter_count ||
            layout.group_weight_offsets.size() != layout.groups.size() + 1)
            throw std::invalid_argument("invalid fixed-B grouped-linear layout");
        if (!same_shape(layout, problem, weights))
            throw std::invalid_argument("library layout and problem shape disagree");
        if (problem.dtype != dtype)
            throw std::invalid_argument("library dtype differs from the problem dtype");
        if (arm != LinearLibraryArm::per_group && arm != LinearLibraryArm::batched)
            throw std::invalid_argument("unknown grouped-linear library arm");
        if (layout.groups.empty()) {
            if (layout.argument_count || layout.group_weight_count ||
                layout.group_weight_offsets.front() != 0)
                throw std::invalid_argument("empty library layout has nonempty local maps");
            return;
        }
        if (layout.batch == 0) return;
        if (layout.batch && (layout.coordinates == 0 || layout.conditions == 0 ||
                             layout.parameter_count == 0 || !weights ||
                             layout.argument_count == 0 || layout.weight_count == 0 ||
                             layout.group_weight_count == 0))
            throw std::invalid_argument("invalid nonempty fixed-B grouped-linear layout");
        if (layout.batch > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("fixed batch exceeds cuBLAS integer dimensions");
        if (arm==LinearLibraryArm::batched && layout.groups.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("batched expert count exceeds cuBLAS integer dimensions");

        input_features = layout.groups.front().arguments.size();
        output_features = layout.groups.front().outputs.size();
        if (input_features == 0 || output_features == 0 ||
            input_features > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
            output_features > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("unsupported grouped-linear I/O shape");
        const auto one_matrix = checked_product(input_features, output_features,
                                                "expert weight matrix");
        const auto element_width = dtype_bytes(dtype);
        if (layout.group_weight_offsets.front() != 0)
            throw std::invalid_argument("group weight offsets must begin at zero");
        std::vector<std::pair<std::uint64_t, std::uint64_t>> local_ranges;
        std::vector<DeviceGroup> records;
        std::vector<std::uint64_t> arguments;
        std::vector<Incidence> outputs;
        records.reserve(layout.groups.size());
        for (std::size_t g = 0; g < layout.groups.size(); ++g) {
            const auto& source = layout.groups[g];
            if (source.arguments.size() != input_features ||
                source.outputs.size() != output_features ||
                source.parameter_index >= layout.parameter_count ||
                source.condition_index >= layout.conditions)
                throw std::invalid_argument("library route requires one uniform I/O shape");
            for (auto coordinate : source.arguments)
                if (coordinate >= layout.coordinates)
                    throw std::invalid_argument("library participant is outside coordinates");
            for (const auto& incidence : source.outputs)
                if (incidence.coordinate >= layout.coordinates)
                    throw std::invalid_argument("library output is outside coordinates");
            if (arguments.size() > layout.argument_count ||
                source.arguments.size() > layout.argument_count - arguments.size())
                throw std::invalid_argument("library argument map exceeds its declared extent");
            const auto arg_begin = arguments.size();
            const auto output_begin = outputs.size();
            arguments.insert(arguments.end(), source.arguments.begin(), source.arguments.end());
            outputs.insert(outputs.end(), source.outputs.begin(), source.outputs.end());

            const auto local_begin = layout.group_weight_offsets[g];
            const auto local_end = layout.group_weight_offsets[g + 1];
            if (local_end < local_begin || local_end - local_begin != one_matrix ||
                local_end > layout.group_weight_count)
                throw std::invalid_argument("group dW span does not match its matrix");
            local_ranges.emplace_back(local_begin, local_end);
            records.push_back({static_cast<std::uint64_t>(arg_begin),
                               static_cast<std::uint64_t>(output_begin),
                               source.parameter_index, source.condition_index,
                               local_begin});
            const auto parameter_begin = layout.parameter_weight_offsets[source.parameter_index];
            if (parameter_begin > layout.weight_count ||
                one_matrix > layout.weight_count - parameter_begin)
                throw std::invalid_argument("expert weight span exceeds its allocation");
        }
        if (arguments.size() != layout.argument_count ||
            layout.group_weight_offsets.back() != layout.group_weight_count)
            throw std::invalid_argument("library local maps do not cover declared extents");
        std::sort(local_ranges.begin(), local_ranges.end());
        for (std::size_t i = 1; i < local_ranges.size(); ++i)
            if (local_ranges[i].first < local_ranges[i - 1].second)
                throw std::invalid_argument("distinct local dW outputs overlap");
        for (const auto parameter_begin : layout.parameter_weight_offsets)
            if (parameter_begin > layout.weight_count ||
                one_matrix > layout.weight_count - parameter_begin)
                throw std::invalid_argument("parameter weight span exceeds its allocation");

        const auto input_elements = checked_product(
            checked_product(layout.batch, layout.groups.size(), "packed batch"),
            input_features, "packed input");
        const auto output_elements = checked_product(
            checked_product(layout.batch, layout.groups.size(), "packed batch"),
            output_features, "packed output");
        scratch_bytes = 0;
        auto add_slice = [&](std::size_t bytes) {
            const std::size_t alignment = 256;
            if (scratch_bytes > std::numeric_limits<std::size_t>::max() - alignment + 1)
                throw std::invalid_argument("library workspace size overflows");
            scratch_bytes = (scratch_bytes + alignment - 1) & ~(alignment - 1);
            if (bytes > std::numeric_limits<std::size_t>::max() - scratch_bytes)
                throw std::invalid_argument("library workspace size overflows");
            scratch_bytes += bytes;
        };
        const auto input_bytes = mc::checked_bytes(input_elements, element_width);
        const auto output_bytes = mc::checked_bytes(output_elements, element_width);
        add_slice(input_bytes);  // forward P inputs; reused for backward D F
        add_slice(input_bytes);  // parameter-gradient inputs: P only
        add_slice(output_bytes); // rounded local F, recomputed for D in backward
        add_slice(output_bytes); // rounded local dF
        add_slice(input_bytes);  // rounded local dX
        if (scratch_bytes > std::numeric_limits<std::size_t>::max() - 255)
            throw std::invalid_argument("library workspace size overflows");
        // workspace_slices aligns against the caller's actual address, which
        // may begin up to 255 bytes past the nominal base.
        scratch_bytes += 255;

        device_groups = DeviceArray(mc::checked_bytes(records.size(), sizeof(DeviceGroup)), layout.device);
        device_arguments = DeviceArray(mc::checked_bytes(arguments.size(), sizeof(std::uint64_t)), layout.device);
        device_outputs = DeviceArray(mc::checked_bytes(outputs.size(), sizeof(Incidence)), layout.device);
        device_parameter_offsets = DeviceArray(
            mc::checked_bytes(layout.parameter_weight_offsets.size(), sizeof(std::uint64_t)),
            layout.device);
        device_groups.upload(records.data(), mc::checked_bytes(records.size(), sizeof(DeviceGroup)));
        device_arguments.upload(arguments.data(), mc::checked_bytes(arguments.size(), sizeof(std::uint64_t)));
        device_outputs.upload(outputs.data(), mc::checked_bytes(outputs.size(), sizeof(Incidence)));
        device_parameter_offsets.upload(layout.parameter_weight_offsets.data(),
                                        mc::checked_bytes(layout.parameter_weight_offsets.size(),
                                                          sizeof(std::uint64_t)));
        device_activity = DeviceArray(mc::checked_bytes(layout.groups.size() * 2,
                                                         sizeof(std::uint8_t)),
                                      layout.device);
        host_activity = mc::PinnedBuffer(mc::checked_bytes(layout.groups.size() * 2,
                                                           sizeof(std::uint8_t)));
        if (arm==LinearLibraryArm::batched) {
            const auto pointer_count=checked_product(layout.groups.size(),9,"batched pointer capacity");
            const auto pointer_bytes=mc::checked_bytes(pointer_count,sizeof(void*));
            device_batch_pointers=DeviceArray(pointer_bytes,layout.device);
            host_batch_pointers=mc::PinnedBuffer(pointer_bytes);
        }
        const DeviceArray* device_allocations[] = {
            &device_groups, &device_arguments, &device_outputs,
            &device_parameter_offsets, &device_activity, &device_batch_pointers};
        for (const auto* allocation : device_allocations) {
            if (allocation->bytes() > std::numeric_limits<std::size_t>::max() - internal_device_bytes)
                throw std::invalid_argument("library device allocation accounting overflows");
            internal_device_bytes += allocation->bytes();
        }
        pinned_host_bytes = host_activity.bytes();
        if (host_batch_pointers.bytes() > std::numeric_limits<std::size_t>::max() - pinned_host_bytes)
            throw std::invalid_argument("library pinned-host allocation accounting overflows");
        pinned_host_bytes += host_batch_pointers.bytes();
        mc::DeviceGuard guard(layout.device);
        const auto create_status = cublasCreate(&handle);
        if (create_status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("create adapted library-route cuBLAS handle failed");
        const auto math_mode = dtype == DType::fp16 ? CUBLAS_TENSOR_OP_MATH
                                                     : CUBLAS_DEFAULT_MATH;
        const auto mode_status = cublasSetMathMode(handle, math_mode);
        if (mode_status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle);
            handle = nullptr;
            throw std::runtime_error("set adapted library-route cuBLAS math mode failed");
        }
    }

    ~Impl() noexcept {
        if (handle) {
            int previous = -1;
            const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
            const bool changed = have_previous && previous != layout.device &&
                                 cudaSetDevice(layout.device) == cudaSuccess;
            (void)cublasDestroy(handle);
            if (changed) (void)cudaSetDevice(previous);
        }
    }
};

namespace {
struct WorkspaceSlices {
    void* forward_x;
    void* backward_x;
    void* local_f;
    void* local_df;
    void* local_dx;
};

template<class ImplT>
cudaError_t stage_activity(ImplT* impl, const GateView& gate,
                           cudaStream_t stream, bool include_derivative) {
    const auto count = impl->layout.groups.size();
    if (count > std::numeric_limits<unsigned>::max())
        return cudaErrorInvalidConfiguration;
    auto* device_flags = static_cast<std::uint8_t*>(impl->device_activity.data());
    auto* host_flags = static_cast<std::uint8_t*>(impl->host_activity.data());
    auto error = cudaMemsetAsync(device_flags, 0, count * 2, stream);
    if (error != cudaSuccess) return error;
    DeviceGate dg{gate.probabilities, gate.admitted, gate.primal_words,
                  gate.derivative_words, impl->layout.conditions,
                  gate.words_per_condition, gate.profile};
    mark_group_activity<<<static_cast<unsigned>(count), 128, 0, stream>>>(
        static_cast<const DeviceGroup*>(impl->device_groups.data()), count,
        impl->layout.batch, dg, include_derivative, device_flags);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    error = cudaMemcpyAsync(host_flags, device_flags, count * 2,
                            cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return error;
    // The fixed-B cuBLAS path can omit a wholly inactive expert only after
    // reading its gate summary. This synchronization and transfer are part of
    // the measured control path by design.
    auto borrowed = mc::Stream::borrow(stream, impl->layout.device);
    borrowed.synchronize();
    return cudaSuccess;
}

template<class ImplT>
cudaError_t bind_handle(ImplT* impl, cudaStream_t stream) {
    const auto status = cublasSetStream(impl->handle, stream);
    return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

template<class ImplT>
cudaError_t run_group_forward(ImplT* impl, std::size_t group,
                              const void* packed_x, void* packed_f) {
    const auto& l = impl->layout;
    const auto row_x = checked_product(
        checked_product(group, l.batch, "group input row"),
        impl->input_features, "group input offset");
    const auto row_f = checked_product(
        checked_product(group, l.batch, "group output row"),
        impl->output_features, "group output offset");
    const auto& descriptor = l.groups[group];
    const void* w = offset_pointer(impl->weights,
        l.parameter_weight_offsets[descriptor.parameter_index], impl->dtype);
    const auto status = gemm(impl->handle, impl->dtype, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(impl->output_features), static_cast<int>(l.batch),
        static_cast<int>(impl->input_features), w,
        static_cast<int>(impl->input_features),
        offset_pointer(packed_x, row_x, impl->dtype),
        static_cast<int>(impl->input_features),
        offset_pointer(packed_f, row_f, impl->dtype),
        static_cast<int>(impl->output_features), 0.0f);
    return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

template<class ImplT>
cudaError_t run_group_backward(ImplT* impl, std::size_t group,
                               const void* packed_x, const void* packed_df,
                               void* packed_dx, void* group_dw) {
    const auto& l = impl->layout;
    const auto row_x = checked_product(
        checked_product(group, l.batch, "group input row"),
        impl->input_features, "group input offset");
    const auto row_f = checked_product(
        checked_product(group, l.batch, "group output row"),
        impl->output_features, "group output offset");
    const auto& descriptor = l.groups[group];
    const void* w = offset_pointer(impl->weights,
        l.parameter_weight_offsets[descriptor.parameter_index], impl->dtype);
    auto* dx = offset_pointer(packed_dx, row_x, impl->dtype);
    auto* dw = offset_pointer(group_dw, l.group_weight_offsets[group], impl->dtype);
    auto status = gemm(impl->handle, impl->dtype, CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(impl->input_features), static_cast<int>(l.batch),
        static_cast<int>(impl->output_features), w,
        static_cast<int>(impl->input_features),
        offset_pointer(packed_df, row_f, impl->dtype),
        static_cast<int>(impl->output_features), dx,
        static_cast<int>(impl->input_features), 0.0f);
    if (status != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    status = gemm(impl->handle, impl->dtype, CUBLAS_OP_N, CUBLAS_OP_T,
        static_cast<int>(impl->input_features), static_cast<int>(impl->output_features),
        static_cast<int>(l.batch), offset_pointer(packed_x, row_x, impl->dtype),
        static_cast<int>(impl->input_features),
        offset_pointer(packed_df, row_f, impl->dtype),
        static_cast<int>(impl->output_features), dw,
        static_cast<int>(impl->input_features), 0.0f);
    return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

template<class ImplT>
std::vector<std::size_t> active_groups(ImplT* impl,std::size_t flag){
    const auto* active=static_cast<const std::uint8_t*>(impl->host_activity.data());
    std::vector<std::size_t> groups;groups.reserve(impl->layout.groups.size());
    for(std::size_t g=0;g<impl->layout.groups.size();++g)if(active[g*2+flag])groups.push_back(g);
    return groups;
}

template<class ImplT>
std::size_t fill_batch_arguments(ImplT* impl,const std::vector<std::size_t>& groups,
    std::size_t first_pointer,const void* a_data,std::uint64_t a_group_stride,
    const void* b_data,std::uint64_t b_group_stride,void* c_data,
    std::uint64_t c_group_stride,bool c_is_group_weight){
    const auto capacity=impl->layout.groups.size();
    auto* base=static_cast<char*>(impl->host_batch_pointers.data());
    auto** a=reinterpret_cast<const void**>(base+first_pointer*sizeof(void*));
    auto** b=reinterpret_cast<const void**>(base+(first_pointer+groups.size())*sizeof(void*));
    auto** c=reinterpret_cast<void**>(base+(first_pointer+2*groups.size())*sizeof(void*));
    const auto matrix=checked_product(impl->input_features,impl->output_features,"batched W stride");
    for(std::size_t at=0;at<groups.size();++at){
        const auto g=groups[at];const auto& descriptor=impl->layout.groups[g];
        const void* weight=offset_pointer(impl->weights,impl->layout.parameter_weight_offsets[descriptor.parameter_index],impl->dtype);
        a[at]=a_group_stride?offset_pointer(a_data,checked_product(g,a_group_stride,"batched A offset"),impl->dtype):weight;
        b[at]=offset_pointer(b_data,checked_product(g,b_group_stride,"batched B offset"),impl->dtype);
        c[at]=c_is_group_weight?offset_pointer(c_data,impl->layout.group_weight_offsets[g],impl->dtype):
            offset_pointer(c_data,checked_product(g,c_group_stride,"batched C offset"),impl->dtype);
    }
    return first_pointer+3*groups.size();
}

template<class ImplT>
cudaError_t copy_batch_arguments(ImplT* impl,std::size_t pointer_count,cudaStream_t stream){
    if (!pointer_count) return cudaSuccess;
    const auto bytes=mc::checked_bytes(pointer_count,sizeof(void*));
    return cudaMemcpyAsync(impl->device_batch_pointers.data(),impl->host_batch_pointers.data(),
                           bytes,cudaMemcpyHostToDevice,stream);
}

template<class ImplT>
cublasStatus_t batched_gemm(ImplT* impl,cublasOperation_t trans_a,
    cublasOperation_t trans_b,int m,int n,int k,const void* const a[],int lda,
    const void* const b[],int ldb,void* const c[],int ldc,int count){
    const float alpha=1.f,beta=0.f;
    return cublasGemmBatchedEx(impl->handle,trans_a,trans_b,m,n,k,&alpha,a,
        cublas_dtype(impl->dtype),lda,b,cublas_dtype(impl->dtype),ldb,&beta,c,
        cublas_dtype(impl->dtype),ldc,count,CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

WorkspaceSlices workspace_slices(void* base, const LinearLibraryLayout& l,
                                 std::uint64_t input_features,
                                 std::uint64_t output_features, DType dtype) {
    const auto input_elements = checked_product(
        checked_product(l.batch, l.groups.size(), "packed batch"),
        input_features, "packed input");
    const auto output_elements = checked_product(
        checked_product(l.batch, l.groups.size(), "packed batch"),
        output_features, "packed output");
    const auto input_bytes = mc::checked_bytes(input_elements, dtype_bytes(dtype));
    const auto output_bytes = mc::checked_bytes(output_elements, dtype_bytes(dtype));
    auto* cursor = static_cast<char*>(base);
    auto take = [&](std::size_t bytes) -> void* {
        const auto address = reinterpret_cast<std::uintptr_t>(cursor);
        const auto aligned = (address + 255u) & ~std::uintptr_t{255u};
        cursor = reinterpret_cast<char*>(aligned + bytes);
        return reinterpret_cast<void*>(aligned);
    };
    return {take(input_bytes), take(input_bytes), take(output_bytes),
            take(output_bytes), take(input_bytes)};
}

template<class ImplT>
cudaError_t check_call(ImplT* impl,
                       const ProblemView& problem, const GateView& gate,
                       ResultView result, WorkspaceView workspace,
                       cudaStream_t stream, bool backward) noexcept {
    if (!impl || problem.include_baseline || problem.dtype != impl->dtype ||
        !same_shape(impl->layout, problem, impl->weights))
        return cudaErrorInvalidValue;
    if (impl->layout.groups.empty() || impl->layout.batch == 0)
        return cudaSuccess;
    if (gate.revisions.structure_epoch != impl->layout.structure_epoch ||
        gate.revisions.mapping_revision != impl->layout.mapping_revision) {
        if (workspace.status)
            store_status<<<1, 1, 0, stream>>>(workspace.status, status_stale);
        return cudaErrorInvalidValue;
    }
    const auto words_required = impl->layout.batch / 32 +
                                (impl->layout.batch % 32 != 0);
    if (!workspace.data || workspace.bytes < impl->scratch_bytes ||
        !workspace.status || !problem.x || !result.y ||
        !gate.admitted && !gate.primal_words ||
        ((!gate.admitted || gate.profile == GateProfile::ste) &&
         gate.primal_words && gate.words_per_condition < words_required) ||
        (gate.profile == GateProfile::ste && !gate.probabilities) ||
        (gate.profile == GateProfile::ste && !gate.derivative_words) ||
        (gate.profile == GateProfile::ste &&
         gate.words_per_condition < words_required) ||
        (backward && (!result.dx || !result.dweights ||
                      (gate.profile == GateProfile::ste && !result.dq) ||
                      !result.dx_group || !result.dweights_group)))
        return cudaErrorInvalidValue;
    return cudaSuccess;
}
} // namespace

GroupedLinearLibraryControl::GroupedLinearLibraryControl(
    LinearLibraryLayout layout, const ProblemView& problem, DType dtype,
    LinearLibraryArm arm)
    : impl_(std::make_unique<Impl>(std::move(layout), problem, dtype,arm)) {}

GroupedLinearLibraryControl::~GroupedLinearLibraryControl() = default;

std::size_t GroupedLinearLibraryControl::workspace_bytes() const noexcept {
    return impl_ ? impl_->scratch_bytes : 0;
}
std::size_t GroupedLinearLibraryControl::device_bytes() const noexcept {
    return impl_ ? impl_->internal_device_bytes : 0;
}
std::size_t GroupedLinearLibraryControl::pinned_host_bytes() const noexcept {
    return impl_ ? impl_->pinned_host_bytes : 0;
}

const LinearLibraryLayout& GroupedLinearLibraryControl::layout() const noexcept {
    return impl_->layout;
}
LinearLibraryArm GroupedLinearLibraryControl::arm()const noexcept{return impl_->arm;}

cudaError_t GroupedLinearLibraryControl::forward(
    const ProblemView& problem, const GateView& gate, ResultView result,
    WorkspaceView workspace, cudaStream_t stream) noexcept {
    if (!impl_) return cudaErrorInvalidValue;
    const auto check = check_call(impl_.get(), problem, gate, result, workspace,
                                  stream, false);
    if (check != cudaSuccess) return check;
    const auto& l = impl_->layout;
    if (l.groups.empty() || l.batch == 0) return cudaSuccess;
    auto error = stage_activity(impl_.get(), gate, stream, false);
    if (error != cudaSuccess) return error;
    const auto slices = workspace_slices(workspace.data, l, impl_->input_features,
                                          impl_->output_features, impl_->dtype);
    DeviceGate dg{gate.probabilities, gate.admitted, gate.primal_words,
                  gate.derivative_words, l.conditions, gate.words_per_condition,
                  gate.profile};
    const auto total = checked_product(
        checked_product(l.batch, l.groups.size(), "packed batch"),
        impl_->input_features, "packed input");
    const auto blocks64 = (total + 255) / 256;
    if (blocks64 > std::numeric_limits<unsigned>::max()) return cudaErrorInvalidConfiguration;
    pack_inputs<<<static_cast<unsigned>(blocks64), 256, 0, stream>>>(
        static_cast<const DeviceGroup*>(impl_->device_groups.data()),
        static_cast<const std::uint64_t*>(impl_->device_arguments.data()),
        l.groups.size(), l.batch, impl_->input_features, l.coordinates, dg,
        problem.x, impl_->dtype, false, slices.forward_x, workspace.status);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    error = bind_handle(impl_.get(), stream);
    if (error != cudaSuccess) return error;
    if(impl_->arm==LinearLibraryArm::per_group){
        const auto* active = static_cast<const std::uint8_t*>(impl_->host_activity.data());
        for (std::size_t g = 0; g < l.groups.size(); ++g)
            if (active[g * 2]) {
                error = run_group_forward(impl_.get(), g, slices.forward_x, slices.local_f);
                if (error != cudaSuccess) return error;
            }
    }else{
        const auto selected=active_groups(impl_.get(),0);
        if(!selected.empty()){
            const auto x_stride=checked_product(l.batch,impl_->input_features,"batched X stride");
            const auto f_stride=checked_product(l.batch,impl_->output_features,"batched F stride");
            const auto pointer_count=fill_batch_arguments(impl_.get(),selected,0,nullptr,0,slices.forward_x,x_stride,
                                 slices.local_f,f_stride,false);
            error=copy_batch_arguments(impl_.get(),pointer_count,stream);if(error!=cudaSuccess)return error;
            auto* pointers=static_cast<char*>(impl_->device_batch_pointers.data());
            const auto stride=selected.size()*sizeof(void*);
            const auto* a=reinterpret_cast<const void* const*>(pointers);
            const auto* b=reinterpret_cast<const void* const*>(pointers+stride);
            auto* c=reinterpret_cast<void* const*>(pointers+2*stride);
            const auto status=batched_gemm(impl_.get(),CUBLAS_OP_T,CUBLAS_OP_N,
                static_cast<int>(impl_->output_features),static_cast<int>(l.batch),
                static_cast<int>(impl_->input_features),a,static_cast<int>(impl_->input_features),
                b,static_cast<int>(impl_->input_features),c,static_cast<int>(impl_->output_features),
                static_cast<int>(selected.size()));
            if(status!=CUBLAS_STATUS_SUCCESS)return cudaErrorUnknown;
        }
    }
    const auto out_total = checked_product(
        checked_product(l.batch, l.groups.size(), "packed batch"),
        impl_->output_features, "packed output");
    const auto out_blocks = (out_total + 255) / 256;
    if (out_blocks > std::numeric_limits<unsigned>::max()) return cudaErrorInvalidConfiguration;
    publish_forward_canonical<<<static_cast<unsigned>(out_blocks), 256, 0, stream>>>(
        static_cast<const DeviceGroup*>(impl_->device_groups.data()),
        static_cast<const Incidence*>(impl_->device_outputs.data()), l.groups.size(),
        l.batch, impl_->output_features, l.coordinates, dg, impl_->dtype,
        slices.local_f, result.y, workspace.status);
    error = cudaGetLastError();
    if (error == cudaSuccess) {
        impl_->forward_revisions = gate.revisions;
        impl_->forward_x = problem.x;
        impl_->forward_stream = stream;
        impl_->have_forward = true;
    }
    return error;
}

cudaError_t GroupedLinearLibraryControl::backward(
    const ProblemView& problem, const GateView& gate, const float* dy,
    ResultView result, WorkspaceView workspace, cudaStream_t stream) noexcept {
    if (!impl_) return cudaErrorInvalidValue;
    const auto check = check_call(impl_.get(), problem, gate, result, workspace,
                                  stream, true);
    if (check != cudaSuccess) return check;
    const auto& l = impl_->layout;
    if (l.groups.empty() || l.batch == 0) return cudaSuccess;
    if (!dy) return cudaErrorInvalidValue;
    if (!impl_->have_forward || impl_->forward_x != problem.x ||
        impl_->forward_stream != stream ||
        !same_revisions(impl_->forward_revisions, gate.revisions)) {
        store_status<<<1, 1, 0, stream>>>(workspace.status, status_stale);
        const auto status_error = cudaGetLastError();
        return status_error == cudaSuccess ? cudaErrorInvalidValue : status_error;
    }
    const auto slices = workspace_slices(workspace.data, l, impl_->input_features,
                                          impl_->output_features, impl_->dtype);
    DeviceGate dg{gate.probabilities, gate.admitted, gate.primal_words,
                  gate.derivative_words, l.conditions, gate.words_per_condition,
                  gate.profile};
    auto error = stage_activity(impl_.get(), gate, stream, true);
    if (error != cudaSuccess) return error;
    const auto input_total = checked_product(
        checked_product(l.batch, l.groups.size(), "packed batch"),
        impl_->input_features, "packed input");
    const auto output_total = checked_product(
        checked_product(l.batch, l.groups.size(), "packed batch"),
        impl_->output_features, "packed output");
    auto blocks = (input_total + 255) / 256;
    if (blocks > std::numeric_limits<unsigned>::max()) return cudaErrorInvalidConfiguration;
    pack_inputs<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
        static_cast<const DeviceGroup*>(impl_->device_groups.data()),
        static_cast<const std::uint64_t*>(impl_->device_arguments.data()),
        l.groups.size(), l.batch, impl_->input_features, l.coordinates, dg,
        problem.x, impl_->dtype, false, slices.backward_x, workspace.status);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    // Recompute F for the complete STE derivative set in backward. Dormant
    // D\P inputs are intentionally absent from the forward reads above.
    pack_inputs<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
        static_cast<const DeviceGroup*>(impl_->device_groups.data()),
        static_cast<const std::uint64_t*>(impl_->device_arguments.data()),
        l.groups.size(), l.batch, impl_->input_features, l.coordinates, dg,
        problem.x, impl_->dtype, true, slices.forward_x, workspace.status);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    blocks = (output_total + 255) / 256;
    if (blocks > std::numeric_limits<unsigned>::max()) return cudaErrorInvalidConfiguration;
    build_output_gradient<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
        static_cast<const DeviceGroup*>(impl_->device_groups.data()),
        static_cast<const Incidence*>(impl_->device_outputs.data()), l.groups.size(),
        l.batch, impl_->output_features, l.coordinates, dg, dy, impl_->dtype,
        slices.local_df, workspace.status);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;

    error = bind_handle(impl_.get(), stream);
    if (error != cudaSuccess) return error;
    if(impl_->arm==LinearLibraryArm::per_group){
        const auto one_matrix = checked_product(impl_->input_features,
                                                impl_->output_features,
                                                "expert weight matrix");
        const auto* active = static_cast<const std::uint8_t*>(impl_->host_activity.data());
        for (std::size_t g = 0; g < l.groups.size(); ++g) {
            if (active[g * 2]) {
                error = run_group_backward(impl_.get(), g, slices.backward_x,
                                           slices.local_df, slices.local_dx,
                                           result.dweights_group);
                if (error != cudaSuccess) return error;
            } else {
                const auto local_bytes = mc::checked_bytes(one_matrix,
                                                           dtype_bytes(impl_->dtype));
                error = cudaMemsetAsync(offset_pointer(result.dweights_group,
                                                       l.group_weight_offsets[g], impl_->dtype),
                                        0, local_bytes, stream);
                if (error != cudaSuccess) return error;
            }
            if (active[g * 2 + 1]) {
                error = run_group_forward(impl_.get(), g, slices.forward_x, slices.local_f);
                if (error != cudaSuccess) return error;
            }
        }
    }else{
        const auto primal=active_groups(impl_.get(),0);
        const auto derivative=active_groups(impl_.get(),1);
        const auto local_weight_bytes=mc::checked_bytes(l.group_weight_count,dtype_bytes(impl_->dtype));
        error=cudaMemsetAsync(result.dweights_group,0,local_weight_bytes,stream);if(error!=cudaSuccess)return error;
        const auto x_stride=checked_product(l.batch,impl_->input_features,"batched X stride");
        const auto f_stride=checked_product(l.batch,impl_->output_features,"batched F stride");
        const auto capacity=l.groups.size();std::size_t pointer_count=0;
        std::size_t dx_pointer=0,dw_pointer=0,f_pointer=0;
        if(!primal.empty()){
            dx_pointer=pointer_count;
            pointer_count=fill_batch_arguments(impl_.get(),primal,pointer_count,nullptr,0,slices.local_df,f_stride,
                                 slices.local_dx,x_stride,false);
            dw_pointer=pointer_count;
            pointer_count=fill_batch_arguments(impl_.get(),primal,pointer_count,slices.backward_x,x_stride,slices.local_df,f_stride,
                                 result.dweights_group,0,true);
        }
        if(!derivative.empty()){
            f_pointer=pointer_count;
            pointer_count=fill_batch_arguments(impl_.get(),derivative,pointer_count,nullptr,0,slices.forward_x,x_stride,
                                 slices.local_f,f_stride,false);
        }
        error=copy_batch_arguments(impl_.get(),pointer_count,stream);if(error!=cudaSuccess)return error;
        auto* device=static_cast<char*>(impl_->device_batch_pointers.data());
        if(!primal.empty()){
            const auto dx_a=reinterpret_cast<const void* const*>(device+dx_pointer*sizeof(void*));
            const auto dx_b=reinterpret_cast<const void* const*>(device+(dx_pointer+primal.size())*sizeof(void*));
            auto* dx_c=reinterpret_cast<void* const*>(device+(dx_pointer+2*primal.size())*sizeof(void*));
            auto status=batched_gemm(impl_.get(),CUBLAS_OP_N,CUBLAS_OP_N,
                static_cast<int>(impl_->input_features),static_cast<int>(l.batch),
                static_cast<int>(impl_->output_features),dx_a,
                static_cast<int>(impl_->input_features),dx_b,
                static_cast<int>(impl_->output_features),dx_c,
                static_cast<int>(impl_->input_features),static_cast<int>(primal.size()));
            if(status!=CUBLAS_STATUS_SUCCESS)return cudaErrorUnknown;
            const auto dw_a=reinterpret_cast<const void* const*>(device+dw_pointer*sizeof(void*));
            const auto dw_b=reinterpret_cast<const void* const*>(device+(dw_pointer+primal.size())*sizeof(void*));
            auto* dw_c=reinterpret_cast<void* const*>(device+(dw_pointer+2*primal.size())*sizeof(void*));
            status=batched_gemm(impl_.get(),CUBLAS_OP_N,CUBLAS_OP_T,
                static_cast<int>(impl_->input_features),static_cast<int>(impl_->output_features),
                static_cast<int>(l.batch),dw_a,
                static_cast<int>(impl_->input_features),dw_b,
                static_cast<int>(impl_->output_features),dw_c,
                static_cast<int>(impl_->input_features),static_cast<int>(primal.size()));
            if(status!=CUBLAS_STATUS_SUCCESS)return cudaErrorUnknown;
        }
        if(!derivative.empty()){
            const auto f_a=reinterpret_cast<const void* const*>(device+f_pointer*sizeof(void*));
            const auto f_b=reinterpret_cast<const void* const*>(device+(f_pointer+derivative.size())*sizeof(void*));
            auto* f_c=reinterpret_cast<void* const*>(device+(f_pointer+2*derivative.size())*sizeof(void*));
            const auto status=batched_gemm(impl_.get(),CUBLAS_OP_T,CUBLAS_OP_N,
                static_cast<int>(impl_->output_features),static_cast<int>(l.batch),
                static_cast<int>(impl_->input_features),f_a,
                static_cast<int>(impl_->input_features),f_b,
                static_cast<int>(impl_->input_features),f_c,
                static_cast<int>(impl_->output_features),static_cast<int>(derivative.size()));
            if(status!=CUBLAS_STATUS_SUCCESS)return cudaErrorUnknown;
        }
    }

    const auto publish_total = input_total;
    blocks = (publish_total + 255) / 256;
    if (blocks > std::numeric_limits<unsigned>::max()) return cudaErrorInvalidConfiguration;
    publish_backward<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
        static_cast<const DeviceGroup*>(impl_->device_groups.data()),
        static_cast<const std::uint64_t*>(impl_->device_arguments.data()),
        l.groups.size(), l.batch, impl_->input_features, l.coordinates,
        l.argument_count, dg, impl_->dtype, slices.local_dx,
        result.dx, result.dx_group, workspace.status);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    if (gate.profile == GateProfile::ste) {
        const auto gate_total = checked_product(l.groups.size(), l.batch,
                                                "grouped gate gradients");
        blocks = (gate_total + 255) / 256;
        if (blocks > std::numeric_limits<unsigned>::max())
            return cudaErrorInvalidConfiguration;
        accumulate_gate_gradient<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
            static_cast<const DeviceGroup*>(impl_->device_groups.data()),
            static_cast<const Incidence*>(impl_->device_outputs.data()),
            l.groups.size(), l.batch, impl_->output_features, l.coordinates,
            l.conditions, dg, dy, impl_->dtype, slices.local_f, result.dq,
            workspace.status);
        error = cudaGetLastError();
        if (error != cudaSuccess) return error;
    }
    const auto matrix_size = checked_product(impl_->input_features,
                                             impl_->output_features,
                                             "expert weight matrix");
    const auto weight_total = checked_product(l.groups.size(), matrix_size,
                                              "grouped weight gradients");
    blocks = (weight_total + 255) / 256;
    if (blocks > std::numeric_limits<unsigned>::max()) return cudaErrorInvalidConfiguration;
    aggregate_weight_gradients<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
        static_cast<const DeviceGroup*>(impl_->device_groups.data()),
        static_cast<const std::uint64_t*>(impl_->device_parameter_offsets.data()),
        static_cast<const std::uint8_t*>(impl_->device_activity.data()),
        l.groups.size(), matrix_size, result.dweights_group, impl_->dtype,
        result.dweights);
    return cudaGetLastError();
}

LinearLibraryLayout make_linear_library_layout(
    const ConsumerHostFixture& fixture, int device) {
    if (fixture.body != BodyKind::grouped_linear)
        throw std::invalid_argument("library layout requires a grouped-linear fixture");
    LinearLibraryLayout result;
    result.batch = fixture.batch;
    result.coordinates = fixture.coordinates;
    result.conditions = fixture.conditions;
    result.argument_count = fixture.arguments.size();
    result.weight_count = fixture.weights32.size();
    result.group_weight_count = fixture.group_weight_offsets.empty()
        ? 0 : fixture.group_weight_offsets.back();
    result.parameter_count = fixture.parameters;
    result.structure_epoch = fixture.revisions.structure_epoch;
    result.mapping_revision = fixture.revisions.mapping_revision;
    result.device = device;
    result.parameter_weight_offsets = fixture.weight_offsets;
    result.group_weight_offsets = fixture.group_weight_offsets;
    result.groups.reserve(fixture.groups.size());
    for (const auto& source : fixture.groups) {
        LinearLibraryGroup group;
        group.parameter_index = source.parameter_index;
        group.condition_index = source.condition_index;
        group.logical_id = source.logical_id;
        group.arguments.assign(
            fixture.arguments.begin() + static_cast<std::ptrdiff_t>(source.argument_begin),
            fixture.arguments.begin() + static_cast<std::ptrdiff_t>(source.argument_begin +
                                                                      source.argument_count));
        group.outputs.assign(
            fixture.outputs.begin() + static_cast<std::ptrdiff_t>(source.output_begin),
            fixture.outputs.begin() + static_cast<std::ptrdiff_t>(source.output_begin +
                                                                     source.output_count));
        result.groups.push_back(std::move(group));
    }
    return result;
}

namespace {
void test_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(action) + ": " +
                                 cudaGetErrorString(status));
}
struct TestAllocation {
    void* data = nullptr;
    explicit TestAllocation(std::size_t bytes) {
        test_cuda(cudaMalloc(&data, bytes), "allocate library test memory");
    }
    ~TestAllocation() { if (data) (void)cudaFree(data); }
    TestAllocation(const TestAllocation&) = delete;
    TestAllocation& operator=(const TestAllocation&) = delete;
};
} // namespace

bool run_grouped_linear_library_tests(cudaStream_t stream, std::string* error) {
    try {
        for (const auto dtype : {DType::fp16, DType::fp32}) {
            auto spec = workload_manifest().front();
            spec.cells = 35;
            spec.features = 96;
            spec.groups = 8;
            auto host = make_consumer_fixture(materialize_workload(spec),
                BodyKind::grouped_linear, dtype, GateProfile::ste);
            host.conditions=1;for(auto& group:host.groups)group.condition_index=0;
            const auto words = (host.batch + 31) / 32;
            host.admitted.assign(host.batch, 0);
            host.primal.assign(words, 0);
            host.derivative.assign(words, 0);
            host.probabilities.assign(host.batch, 0.25f);
            for (std::uint64_t cell = 0; cell < host.batch; ++cell) {
                const bool admitted = (cell % 2) == 0;
                host.admitted[cell] = admitted ? 1 : 0;
                if (admitted) {
                    host.primal[cell / 32] |= std::uint32_t{1} << (cell % 32);
                    host.probabilities[cell] = 0.75f;
                }
                // Every cell remains in the declared STE derivative set.
                host.derivative[cell / 32] |= std::uint32_t{1} << (cell % 32);
            }
            ++host.revisions.accessibility_generation;
            ++host.revisions.primal_generation;
            ++host.revisions.derivative_generation;
            const auto reference_host = host;
            const auto hp = reference_host.problem();
            const auto hg = reference_host.gates();
            const auto reference_f = reference::grouped_linear_forward(hp, hg);
            const auto reference_b = reference::grouped_linear_backward(
                hp, hg, reference_host.dy.data(), reference_f);

            // Poison every feature in dormant rows. The forward pass must not
            // inspect them; after it is queued, restore ordinary values so
            // the backward D pass can form the declared dormant surrogate.
            if (dtype == DType::fp16) {
                for (std::uint64_t cell = 1; cell < host.batch; cell += 2)
                    std::fill(host.x16.begin() + cell * host.coordinates,
                              host.x16.begin() + (cell + 1) * host.coordinates,
                              std::uint16_t{0x7e00u});
            } else {
                const auto nan = std::numeric_limits<float>::quiet_NaN();
                for (std::uint64_t cell = 1; cell < host.batch; cell += 2)
                    std::fill(host.x32.begin() + cell * host.coordinates,
                              host.x32.begin() + (cell + 1) * host.coordinates, nan);
            }
            ConsumerDeviceFixture device(host, stream);
            auto problem = device.problem();
            const auto gates = device.gates();
            int device_index = 0;
            test_cuda(cudaGetDevice(&device_index), "get scheduler-selected device");
            const auto layout = make_linear_library_layout(host, device_index);
            for(const auto arm:{LinearLibraryArm::per_group,LinearLibraryArm::batched}){
            if(dtype==DType::fp16)test_cuda(cudaMemcpyAsync(const_cast<void*>(problem.x),host.x16.data(),host.x16.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice,stream),"restore dormant poison before library arm");
            else test_cuda(cudaMemcpyAsync(const_cast<void*>(problem.x),host.x32.data(),host.x32.size()*sizeof(float),cudaMemcpyHostToDevice,stream),"restore dormant poison before library arm");
            GroupedLinearLibraryControl control(layout, problem, dtype,arm);
            TestAllocation work(control.workspace_bytes());
            TestAllocation status(sizeof(std::uint32_t));
            auto results = device.results();
            device.clear_results(stream);
            test_cuda(cudaMemsetAsync(status.data, 0, sizeof(std::uint32_t), stream),
                      "clear library status");
            WorkspaceView workspace{work.data, control.workspace_bytes(),
                                    static_cast<std::uint32_t*>(status.data)};
            test_cuda(control.forward(problem, gates, results, workspace, stream),
                      "library forward with dormant poison");
            test_cuda(cudaStreamSynchronize(stream),
                      "finish poisoned-forward read check");
            std::uint32_t forward_status = 0;
            test_cuda(cudaMemcpy(&forward_status, status.data, sizeof(forward_status),
                                 cudaMemcpyDeviceToHost), "read poisoned-forward status");
            if (forward_status)
                throw std::runtime_error("forward inspected a poisoned D\\P participant or failed: " +
                                         std::to_string(forward_status));
            if (dtype == DType::fp16)
                test_cuda(cudaMemcpyAsync(const_cast<void*>(problem.x), reference_host.x16.data(),
                    reference_host.x16.size() * sizeof(std::uint16_t),
                    cudaMemcpyHostToDevice, stream), "restore FP16 D inputs");
            else
                test_cuda(cudaMemcpyAsync(const_cast<void*>(problem.x), reference_host.x32.data(),
                    reference_host.x32.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream), "restore FP32 D inputs");
            test_cuda(control.backward(problem, gates, device.dy(), results,
                                       workspace, stream),
                      "library backward D recomputation");
            test_cuda(cudaStreamSynchronize(stream), "finish library correctness probe");
            std::uint32_t status_value = 0;
            test_cuda(cudaMemcpy(&status_value, status.data, sizeof(status_value),
                                 cudaMemcpyDeviceToHost), "read library status");
            if (status_value)
                throw std::runtime_error("library status=" +
                                         std::to_string(status_value));
            const auto actual = device.download(stream);
            const auto bounds=reference::grouped_linear_validation_bounds(hp,hg,reference_host.dy.data(),reference_f,reference_b);
            const auto bounded=[&](const std::vector<float>& value,const std::vector<float>& expected,const std::vector<float>& bound,const char* label){
                if(value.size()!=expected.size()||bound.size()!=expected.size())throw std::runtime_error(std::string(label)+" bound size mismatch");
                for(std::size_t j=0;j<value.size();++j)if(!reference::within_absolute_bound(value[j],expected[j],bound[j]))
                    throw std::runtime_error(std::string(label)+" compound bound mismatch at "+std::to_string(j));
            };
            bounded(actual.y,reference_f.y,bounds.y,"library y");
            bounded(actual.dx,reference_b.dx,bounds.dx,"library dx");
            bounded(actual.dq,reference_b.dq,bounds.dq,"library dormant dq");
            bounded(actual.dweights,reference_b.dweights,bounds.dweights,"library shared dweights");
            bounded(actual.dx_group,reference_b.group_dx,bounds.group_dx,"library local dx");
            bounded(actual.dweights_group,reference_b.group_dw,bounds.group_dw,"library local dw");
            }
        }
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

} // namespace cgs::backends
