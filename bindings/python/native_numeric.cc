#include "native_numeric.hh"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Cellerator/compute/operation/native_numeric/device_linear.hh>
#include <Cellerator/compute/operation/prepared_relation.hh>
#include <Cellerator/compute/operation/relation_update.hh>
#include <Cellerator/runtime/device_buffer.cuh>
#include <dlpack/dlpack.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace relation = cellerator::compute::relation;
namespace numeric = cellerator::compute::native_numeric;
namespace execution = cellerator::execution;

namespace cellerator::bindings::python {
namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void relation_check(relation::status status, const char* operation) {
    if (!status)
        throw std::runtime_error(std::string(operation) + ": " +
            (status.message ? status.message : "native relation operation failed"));
}

class DeviceScope {
public:
    explicit DeviceScope(int device) {
        cuda_check(cudaGetDevice(&previous_), "cudaGetDevice");
        changed_ = previous_ != device;
        if (changed_) cuda_check(cudaSetDevice(device), "cudaSetDevice");
    }
    ~DeviceScope() { if (changed_) cudaSetDevice(previous_); }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
private:
    int previous_ = 0;
    bool changed_ = false;
};

std::uint64_t checked_product(const std::vector<std::int64_t>& shape) {
    std::uint64_t count = 1;
    for (auto dim : shape) {
        if (dim < 0) throw py::value_error("shape dimensions must be nonnegative");
        const auto extent = static_cast<std::uint64_t>(dim);
        if (extent && count > std::numeric_limits<std::uint64_t>::max() / extent)
            throw std::overflow_error("buffer element extent overflow");
        count *= extent;
    }
    return count;
}

std::vector<std::int64_t> parse_shape(const py::sequence& input) {
    std::vector<std::int64_t> out;
    out.reserve(static_cast<std::size_t>(py::len(input)));
    for (const auto item : input) {
        if (PyBool_Check(item.ptr()) || !PyLong_Check(item.ptr()))
            throw py::type_error("shape dimensions must be integers");
        const auto value = PyLong_AsLongLong(item.ptr());
        if (value == -1 && PyErr_Occurred()) throw py::error_already_set();
        out.push_back(static_cast<std::int64_t>(value));
    }
    if (out.empty()) throw py::value_error("buffer shape must have at least one dimension");
    return out;
}

py::array strict_float_array(const py::handle& value, const char* name, int ndim) {
    if (!py::isinstance<py::array>(value))
        throw py::type_error(std::string(name) + " must be a NumPy array");
    auto array = py::reinterpret_borrow<py::array>(value);
    if (!array.dtype().is(py::dtype::of<float>()) || array.ndim() != ndim ||
        !(array.flags() & py::array::c_style) ||
        reinterpret_cast<std::uintptr_t>(array.data()) % alignof(float) != 0)
        throw py::value_error(std::string(name) + " must be aligned, contiguous float32 with rank " +
                              std::to_string(ndim));
    return array;
}

py::array strict_u64_array(const py::handle& value, const char* name) {
    if (!py::isinstance<py::array>(value))
        throw py::type_error(std::string(name) + " must be a NumPy array");
    auto array = py::reinterpret_borrow<py::array>(value);
    if (!array.dtype().is(py::dtype::of<std::uint64_t>()) || array.ndim() != 1 ||
        !(array.flags() & py::array::c_style) ||
        reinterpret_cast<std::uintptr_t>(array.data()) % alignof(std::uint64_t) != 0)
        throw py::value_error(std::string(name) + " must be contiguous rank-1 uint64");
    return array;
}

class Stream final {
public:
    explicit Stream(int device) : device_(device) {
        int count = 0;
        cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (device < 0 || device >= count) throw py::value_error("CUDA device ordinal is out of range");
        DeviceScope scope(device_);
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        owned_ = true;
    }
    Stream(int device, std::uintptr_t handle, py::object owner)
        : device_(device), stream_(reinterpret_cast<cudaStream_t>(handle)), owner_(std::move(owner)) {
        int count = 0;
        cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (device < 0 || device >= count) throw py::value_error("CUDA device ordinal is out of range");
        if (owner_.is_none()) throw py::value_error("borrowed CUDA stream requires an owner");
        DeviceScope scope(device_);
        unsigned flags = 0;
        const auto status = cudaStreamGetFlags(stream_, &flags);
        if (status != cudaSuccess) {
            cudaGetLastError();
            throw py::value_error("borrowed CUDA stream is not valid on the declared device");
        }
    }
    ~Stream() {
        if (owned_ && stream_) {
            int previous = 0;
            if (cudaGetDevice(&previous) != cudaSuccess) return;
            const bool changed = previous != device_;
            if (changed && cudaSetDevice(device_) != cudaSuccess) return;
            cudaStreamDestroy(stream_);
            if (changed) cudaSetDevice(previous);
        }
    }
    int device() const noexcept { return device_; }
    cudaStream_t native() const noexcept { return stream_; }
    void synchronize() const {
        DeviceScope scope(device_);
        cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }
    void wait_on(const Stream& other) const {
        if (device_ != other.device_) throw py::value_error("streams must use the same CUDA device");
        DeviceScope scope(device_);
        cudaEvent_t event = nullptr;
        cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags");
        try {
            cuda_check(cudaEventRecord(event, other.native()), "cudaEventRecord(wait_on)");
            cuda_check(cudaStreamWaitEvent(stream_, event, 0), "cudaStreamWaitEvent");
        } catch (...) {
            cudaEventDestroy(event);
            throw;
        }
        cuda_check(cudaEventDestroy(event), "cudaEventDestroy(wait_on)");
    }
private:
    int device_ = 0;
    cudaStream_t stream_ = nullptr;
    bool owned_ = false;
    py::object owner_ = py::none();
};

class Buffer final {
public:
    Buffer(std::vector<std::int64_t> shape, std::shared_ptr<Stream> stream)
        : shape_(std::move(shape)), stream_(require_stream(std::move(stream))),
          count_(checked_product(shape_)) {
        if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(float))
            throw std::overflow_error("buffer byte extent overflow");
        nbytes_ = static_cast<std::size_t>(count_) * sizeof(float);
        DeviceScope scope(stream_->device());
        storage_ = cellerator::runtime::allocate_device_buffer<float>(static_cast<std::size_t>(count_));
        data_ = storage_.data;
        owned_ = true;
    }

    static std::shared_ptr<Buffer> borrow(std::uintptr_t pointer,
        std::vector<std::int64_t> shape, std::uint64_t capacity_bytes,
        std::shared_ptr<Stream> stream, py::object owner) {
        auto out = std::shared_ptr<Buffer>(new Buffer(BorrowTag{}, std::move(shape),
            std::move(stream), pointer, capacity_bytes, std::move(owner)));
        return out;
    }

    const std::vector<std::int64_t>& shape() const noexcept { return shape_; }
    int device() const noexcept { return stream_->device(); }
    std::size_t nbytes() const noexcept { return nbytes_; }
    std::uint64_t elements() const noexcept { return count_; }
    void* data() const noexcept { return data_; }
    cudaStream_t native_stream() const noexcept { return stream_->native(); }
    const std::shared_ptr<Stream>& stream() const noexcept { return stream_; }
    bool owned() const noexcept { return owned_; }
    const std::shared_ptr<void>& native_owner() const noexcept { return storage_.owner; }

    void upload(py::array value) {
        auto array = strict_float_array(value, "upload input", static_cast<int>(shape_.size()));
        if (!matches_shape(array, shape_)) throw py::value_error("upload array shape differs from buffer shape");
        DeviceScope scope(device());
        if (nbytes_) {
            cuda_check(cudaMemcpyAsync(data_, array.data(), nbytes_, cudaMemcpyHostToDevice, native_stream()),
                       "cudaMemcpyAsync(H2D)");
            cuda_check(cudaStreamSynchronize(native_stream()), "cudaStreamSynchronize(upload)");
        }
    }

    py::array_t<float> download() const {
        py::array::ShapeContainer shape(shape_.begin(), shape_.end());
        py::array_t<float> out(shape);
        DeviceScope scope(device());
        if (nbytes_) {
            cuda_check(cudaMemcpyAsync(out.mutable_data(), data_, nbytes_, cudaMemcpyDeviceToHost, native_stream()),
                       "cudaMemcpyAsync(D2H)");
            cuda_check(cudaStreamSynchronize(native_stream()), "cudaStreamSynchronize(download)");
        }
        return out;
    }

private:
    struct BorrowTag {};
    Buffer(BorrowTag, std::vector<std::int64_t> shape, std::shared_ptr<Stream> stream,
           std::uintptr_t pointer, std::uint64_t capacity_bytes, py::object owner)
        : shape_(std::move(shape)), stream_(require_stream(std::move(stream))),
          count_(checked_product(shape_)), owner_(std::move(owner)),
          data_(reinterpret_cast<void*>(pointer)) {
        if (owner_.is_none()) throw py::value_error("borrowed buffer requires an owner");
        if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(float))
            throw std::overflow_error("buffer byte extent overflow");
        nbytes_ = static_cast<std::size_t>(count_) * sizeof(float);
        if (capacity_bytes < nbytes_) throw py::value_error("borrowed buffer capacity is too small");
        if (nbytes_ && (!data_ || pointer % alignof(float)))
            throw py::value_error("borrowed buffer pointer must be nonnull and float-aligned");
        DeviceScope scope(device());
        validate_device_pointer(data_, nbytes_, device());
    }

    static std::shared_ptr<Stream> require_stream(std::shared_ptr<Stream> stream) {
        if (!stream) throw py::value_error("stream is required");
        return stream;
    }
    static bool matches_shape(const py::array& array, const std::vector<std::int64_t>& shape) {
        if (array.ndim() != static_cast<py::ssize_t>(shape.size())) return false;
        for (std::size_t i = 0; i < shape.size(); ++i)
            if (array.shape(static_cast<py::ssize_t>(i)) != shape[i]) return false;
        return true;
    }
    static void validate_device_pointer(const void* pointer, std::size_t bytes, int device) {
        if (!bytes) return;
        cudaPointerAttributes attributes{};
        const auto status = cudaPointerGetAttributes(&attributes, pointer);
        if (status != cudaSuccess) {
            cudaGetLastError();
            throw py::value_error("borrowed pointer is not CUDA device storage");
        }
#if CUDART_VERSION >= 10000
        const bool device_memory = attributes.type == cudaMemoryTypeDevice ||
                                   attributes.type == cudaMemoryTypeManaged;
#else
        const bool device_memory = attributes.memoryType == cudaMemoryTypeDevice || attributes.isManaged;
#endif
        if (!device_memory || attributes.device != device)
            throw py::value_error("borrowed pointer belongs to a different CUDA device");
        const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
        if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
            throw std::overflow_error("borrowed pointer extent overflow");
    }

    std::vector<std::int64_t> shape_;
    std::shared_ptr<Stream> stream_;
    std::uint64_t count_ = 0;
    std::size_t nbytes_ = 0;
    cellerator::runtime::device_buffer<float> storage_{};
    bool owned_ = false;
    py::object owner_ = py::none();
    void* data_ = nullptr;
};

void check_context(const Buffer& buffer, const Stream& stream, const char* name) {
    if (buffer.device() != stream.device() || buffer.native_stream() != stream.native())
        throw py::value_error(std::string(name) + " must use the operation device and stream");
}

bool overlaps(const Buffer& left, const Buffer& right) {
    if (!left.nbytes() || !right.nbytes()) return false;
    const auto a = reinterpret_cast<std::uintptr_t>(left.data());
    const auto b = reinterpret_cast<std::uintptr_t>(right.data());
    if (a > std::numeric_limits<std::uintptr_t>::max() - left.nbytes() ||
        b > std::numeric_limits<std::uintptr_t>::max() - right.nbytes())
        throw std::overflow_error("buffer address extent overflow");
    return a < b + right.nbytes() && b < a + left.nbytes();
}

numeric::resident_vector resident(Buffer& buffer) {
    return {buffer.data(), buffer.elements(), numeric::device_representation::f32,
            buffer.device(), {0}};
}

struct DLPackContext {
    std::shared_ptr<void> owner;
    std::vector<std::int64_t> shape;
    DLManagedTensor tensor{};
};

void delete_dlpack_tensor(DLManagedTensor* tensor) {
    if (!tensor) return;
    auto* context = static_cast<DLPackContext*>(tensor->manager_ctx);
    delete context;
}

py::capsule export_dlpack(const Buffer& buffer, py::object stream_arg) {
    if (!buffer.owned()) throw py::value_error("DLPack export currently requires a native-owned buffer");
    DeviceScope scope(buffer.device());
    if (!stream_arg.is_none()) {
        const auto consumer_stream = py::cast<std::int64_t>(stream_arg);
        if (consumer_stream < -1)
            throw py::value_error("DLPack consumer stream must be -1 or a CUDA stream handle");
        auto target = consumer_stream == -1 ? nullptr :
            reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(consumer_stream));
        if (consumer_stream != -1 && target != buffer.native_stream()) {
            cudaEvent_t ready = nullptr;
            cuda_check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "cudaEventCreateWithFlags(DLPack)");
            try {
                cuda_check(cudaEventRecord(ready, buffer.native_stream()), "cudaEventRecord(DLPack)");
                cuda_check(cudaStreamWaitEvent(target, ready, 0), "cudaStreamWaitEvent(DLPack)");
            } catch (...) {
                cudaEventDestroy(ready);
                throw;
            }
            cuda_check(cudaEventDestroy(ready), "cudaEventDestroy(DLPack)");
        }
    }

    auto* context = new DLPackContext;
    context->owner = buffer.native_owner();
    context->shape = buffer.shape();
    context->tensor.dl_tensor.data = buffer.data();
    context->tensor.dl_tensor.device = {kDLCUDA, buffer.device()};
    context->tensor.dl_tensor.ndim = static_cast<std::int32_t>(context->shape.size());
    context->tensor.dl_tensor.dtype = {kDLFloat, 32, 1};
    context->tensor.dl_tensor.shape = context->shape.data();
    context->tensor.dl_tensor.strides = nullptr;
    context->tensor.dl_tensor.byte_offset = 0;
    context->tensor.manager_ctx = context;
    context->tensor.deleter = delete_dlpack_tensor;
    return py::capsule(&context->tensor, "dltensor", [](PyObject* capsule) {
        if (PyCapsule_IsValid(capsule, "dltensor")) {
            auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
            if (managed && managed->deleter) managed->deleter(managed);
        }
    });
}

std::atomic<std::uint64_t> next_identity{1};

execution::persistent_axis_identity make_axis(std::uint64_t id) {
    execution::persistent_axis_identity axis{};
    axis.header.schema_version = execution::biological_abi_version;
    axis.header.kind = execution::serialized_record_kind::persistent_axis_identity;
    axis.header.byte_count = sizeof(axis);
    axis.domain = {0x43454c4c45524154ULL, id};
    axis.order = {0x5245534944454e54ULL, id};
    axis.geometry = {0x4353525f50595448ULL, id};
    axis.partition = {0x53494e474c455f44ULL, id};
    return axis;
}

class PreparedCsr final {
public:
    PreparedCsr(py::array row_offsets, py::array source_indices,
        std::shared_ptr<Buffer> weights, std::uint64_t source_count,
        std::uint32_t feature_width, std::shared_ptr<Stream> stream)
        : weights_(std::move(weights)), stream_(require_stream(std::move(stream))),
          source_count_(source_count), feature_width_(feature_width) {
        if (!weights_) throw py::value_error("weights buffer is required");
        if (!source_count_ || !feature_width_) throw py::value_error("source_count and feature_width must be positive");
        if (source_count_ > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::overflow_error("source_count exceeds Python shape range");
        if (weights_->shape().size() != 1) throw py::value_error("weights must be a rank-1 edge buffer");
        check_context(*weights_, *stream_, "weights");
        auto offsets = strict_u64_array(row_offsets, "indptr");
        auto indices = strict_u64_array(source_indices, "indices");
        const auto* offset_data = static_cast<const std::uint64_t*>(offsets.data());
        const auto* index_data = static_cast<const std::uint64_t*>(indices.data());
        if (offsets.size() < 1 || indices.size() != static_cast<py::ssize_t>(weights_->elements()))
            throw py::value_error("CSR row offsets, indices, and edge weights disagree");
        destination_count_ = static_cast<std::uint64_t>(offsets.size() - 1);
        edge_count_ = static_cast<std::uint64_t>(indices.size());
        if (weights_->elements() != edge_count_ || offset_data[0] != 0 ||
            offset_data[offsets.size() - 1] != edge_count_)
            throw py::value_error("CSR offsets must span the edge index array");
        std::vector<std::uint32_t> rows(static_cast<std::size_t>(offsets.size()));
        std::vector<std::uint32_t> cols(static_cast<std::size_t>(indices.size()));
        for (py::ssize_t i = 0; i < offsets.size(); ++i) {
            if (offset_data[i] > edge_count_ || offset_data[i] > std::numeric_limits<std::uint32_t>::max() ||
                (i && offset_data[i] < offset_data[i - 1]))
                throw py::value_error("CSR offsets must be monotone uint32-range edge offsets");
            rows[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(offset_data[i]);
        }
        for (py::ssize_t i = 0; i < indices.size(); ++i) {
            if (index_data[i] >= source_count_ || index_data[i] > std::numeric_limits<std::uint32_t>::max())
                throw py::value_error("CSR source index is outside source_count or uint32 range");
            cols[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(index_data[i]);
        }

        const auto id = next_identity.fetch_add(1, std::memory_order_relaxed);
        topology_.identity = {0x505954484f4e4353ULL, id};
        topology_.epoch = {1};
        topology_.source = {make_axis(id), source_count_};
        topology_.destination = {make_axis(id + 0x100000000ULL), destination_count_};
        topology_.logical_edge_order = {0x4353525f4c4f4749ULL, id};
        topology_.edge_count = edge_count_;
        operation_.topology = topology_;
        operation_.direction = relation::orientation::forward;
        operation_.dense_width = feature_width_;
        operation_.arithmetic.relation_storage = execution::numeric_type::f32;
        operation_.arithmetic.input_storage = execution::numeric_type::f32;
        operation_.arithmetic.multiply = execution::numeric_type::f32;
        operation_.arithmetic.accumulation = execution::numeric_type::f32;
        operation_.arithmetic.output_storage = execution::numeric_type::f32;
        operation_.arithmetic.permit_fma = true;
        operation_.arithmetic.permit_reassociation = true;
        relation_check(relation::validate(operation_), "validate prepared CSR descriptor");
        auto transpose = operation_;
        transpose.direction = relation::orientation::transpose;
        relation_check(relation::validate(transpose), "validate prepared CSR transpose descriptor");
        DeviceScope scope(stream_->device());
        relation_check(relation::prepare_relation_pair(operation_, transpose,
            {rows.data(), static_cast<std::uint64_t>(rows.size()), cols.data(),
             static_cast<std::uint64_t>(cols.size())},
            {stream_->device(), 0, false}, stream_->native(), &handle_), "prepare paired CSR");
    }

    ~PreparedCsr() = default;
    PreparedCsr(const PreparedCsr&) = delete;
    PreparedCsr& operator=(const PreparedCsr&) = delete;

    void apply_into(const std::shared_ptr<Buffer>& input, const std::shared_ptr<Buffer>& output) {
        ensure_open();
        if (!input || !output) throw py::value_error("input and output buffers are required");
        check_context(*input, *stream_, "input");
        check_context(*output, *stream_, "output");
        const std::vector<std::int64_t> input_shape{static_cast<std::int64_t>(source_count_),
                                                    static_cast<std::int64_t>(feature_width_)};
        const std::vector<std::int64_t> output_shape{static_cast<std::int64_t>(destination_count_),
                                                     static_cast<std::int64_t>(feature_width_)};
        if (input->shape() != input_shape || output->shape() != output_shape)
            throw py::value_error("prepared CSR input/output shape mismatch");
        if (overlaps(*input, *output)) throw py::value_error("prepared CSR input and output must be disjoint");
        relation::device_state_view in{input->data(), input->elements(), topology_.source, stream_->device(),
                                       execution::numeric_type::f32};
        relation::device_result_view out{output->data(), output->elements(), topology_.destination,
                                         stream_->device(), execution::numeric_type::f32};
        DeviceScope scope(stream_->device());
        relation_check(relation::enqueue(*handle_, operation_, in, out, {generation_}, stream_->native()),
                       "enqueue prepared CSR");
    }

    void publish_values(const std::shared_ptr<Buffer>& values) {
        ensure_open();
        if (!values) throw py::value_error("weights buffer is required");
        check_context(*values, *stream_, "weights");
        if (values->shape().size() != 1 || values->elements() != edge_count_)
            throw py::value_error("weights must be a rank-1 buffer matching the CSR edge count");
        if (generation_ == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("prepared CSR value generation overflow");
        reap_value_leases();
        const auto next = generation_ + 1;
        relation::device_f32_values_binding binding{static_cast<const float*>(values->data()), edge_count_,
            topology_.identity, topology_.epoch, topology_.logical_edge_order, {next}, stream_->device()};
        DeviceScope scope(stream_->device());
        cudaEvent_t ready = nullptr;
        cuda_check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming),
                   "cudaEventCreateWithFlags(value lease)");
        try {
            value_leases_.emplace_back(values, ready);
        } catch (...) {
            cudaEventDestroy(ready);
            throw;
        }
        const auto status = relation::publish_f32_values(*handle_, binding, stream_->native());
        if (!status) {
            // A failure may be reported after a gather was submitted. Retain
            // the owner and poison the wrapper because the native generation
            // can no longer be inferred safely from the returned status.
            poisoned_ = true;
            if (cudaEventRecord(ready, stream_->native()) != cudaSuccess)
                value_leases_.back().second = nullptr;
            relation_check(status, "publish prepared CSR values");
        }
        generation_ = next;
        weights_ = values;
        const auto record_status = cudaEventRecord(ready, stream_->native());
        if (record_status != cudaSuccess) {
            value_leases_.back().second = nullptr; // Keep the owner through explicit close.
            poisoned_ = true;
            cuda_check(record_status, "cudaEventRecord(value lease)");
        }
    }

    void initialize_values() { publish_values(weights_); }

    std::uint64_t generation() const { ensure_open(); return generation_; }
    std::uint64_t prepared_bytes() const {
        ensure_open();
        relation::preparation_report report{};
        relation_check(relation::inspect(*handle_, &report), "inspect prepared CSR");
        if (report.instance_value_bytes >
            std::numeric_limits<std::uint64_t>::max() - report.shared_structural_bytes)
            throw std::overflow_error("prepared CSR byte count overflow");
        return report.shared_structural_bytes + report.instance_value_bytes;
    }
    std::size_t pending_value_leases() {
        ensure_open();
        reap_value_leases();
        return value_leases_.size();
    }
    void close() {
        if (closed_) return;
        DeviceScope scope(stream_->device());
        cuda_check(cudaStreamSynchronize(stream_->native()), "cudaStreamSynchronize(close prepared CSR)");
        if (handle_)
            relation_check(relation::close_relation_pair(&handle_), "close prepared CSR");
        weights_.reset();
        clear_value_leases();
        closed_ = true;
    }
private:
    static std::shared_ptr<Stream> require_stream(std::shared_ptr<Stream> stream) {
        if (!stream) throw py::value_error("stream is required");
        return stream;
    }
    void ensure_open() const {
        if (!handle_) throw py::value_error("prepared CSR is closed");
        if (poisoned_) throw py::value_error("prepared CSR is poisoned; close it before reuse");
    }
    void reap_value_leases() {
        DeviceScope scope(stream_->device());
        auto it = value_leases_.begin();
        while (it != value_leases_.end()) {
            if (!it->second) { ++it; continue; }
            const auto status = cudaEventQuery(it->second);
            if (status == cudaSuccess) {
                cudaEventDestroy(it->second);
                it = value_leases_.erase(it);
            } else {
                if (status != cudaErrorNotReady) cudaGetLastError();
                ++it;
            }
        }
    }
    void clear_value_leases() noexcept {
        for (auto& lease : value_leases_)
            if (lease.second) cudaEventDestroy(lease.second);
        value_leases_.clear();
    }
    std::shared_ptr<Buffer> weights_;
    std::vector<std::pair<std::shared_ptr<Buffer>, cudaEvent_t>> value_leases_;
    std::shared_ptr<Stream> stream_;
    std::uint64_t source_count_ = 0, destination_count_ = 0, edge_count_ = 0;
    std::uint32_t feature_width_ = 0;
    std::uint64_t generation_ = 0;
    bool poisoned_ = false;
    bool closed_ = false;
    relation::topology_descriptor topology_{};
    relation::operation_descriptor operation_{};
    relation::prepared_relation_pair* handle_ = nullptr;
};

std::shared_ptr<PreparedCsr> make_prepared_csr(py::array row_offsets,
    py::array source_indices, std::shared_ptr<Buffer> weights,
    std::uint64_t source_count, std::uint32_t feature_width,
    std::shared_ptr<Stream> stream) {
    auto* owner = new PreparedCsr(std::move(row_offsets), std::move(source_indices),
        std::move(weights), source_count, feature_width, std::move(stream));
    auto value = std::shared_ptr<PreparedCsr>(owner, [](PreparedCsr* value) noexcept {
        try {
            value->close();
            delete value;
        } catch (...) {
            // A failed completion fence leaves the CUDA owner, stream, weights,
            // and queued-value leases allocated rather than freeing live state.
        }
    });
    value->initialize_values();
    return value;
}

class Event final {
public:
    explicit Event(std::shared_ptr<Stream> stream) : stream_(std::move(stream)) { record(); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept
        : stream_(std::move(other.stream_)), event_(std::exchange(other.event_, nullptr)),
          recorded_(std::exchange(other.recorded_, false)) {}
    ~Event() {
        if (!event_ || !stream_) return;
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        const bool changed = previous != stream_->device();
        if (changed && cudaSetDevice(stream_->device()) != cudaSuccess) return;
        cudaEventDestroy(event_);
        if (changed) cudaSetDevice(previous);
    }
    void record() {
        if (!stream_) throw py::value_error("event requires a stream");
        DeviceScope scope(stream_->device());
        const bool created = !event_;
        if (created) cuda_check(cudaEventCreate(&event_), "cudaEventCreate");
        const auto status = cudaEventRecord(event_, stream_->native());
        if (status != cudaSuccess && created) {
            cudaEventDestroy(event_);
            event_ = nullptr;
        }
        cuda_check(status, "cudaEventRecord");
        recorded_ = true;
    }
    void synchronize() const {
        if (!recorded_) throw py::value_error("event has not been recorded");
        DeviceScope scope(stream_->device());
        cuda_check(cudaEventSynchronize(event_), "cudaEventSynchronize");
    }
    float elapsed_ms(const Event& other) const {
        if (!recorded_ || !other.recorded_) throw py::value_error("both events must be recorded");
        if (stream_->device() != other.stream_->device()) throw py::value_error("events must use the same CUDA device");
        synchronize();
        other.synchronize();
        DeviceScope scope(stream_->device());
        float value = 0.0f;
        cuda_check(cudaEventElapsedTime(&value, event_, other.event_), "cudaEventElapsedTime");
        return value;
    }
    static Event record_on(std::shared_ptr<Stream> stream) { return Event(std::move(stream)); }
private:
    std::shared_ptr<Stream> stream_;
    cudaEvent_t event_ = nullptr;
    bool recorded_ = false;
};

void validate_numeric_buffers(Buffer& a, Buffer& b, Buffer& out, const Stream& stream) {
    check_context(a, stream, "left input");
    check_context(b, stream, "right input");
    check_context(out, stream, "output");
    if (a.elements() != b.elements() || a.elements() != out.elements())
        throw py::value_error("numeric buffers must have equal element counts");
    if (a.shape() != b.shape() || a.shape() != out.shape())
        throw py::value_error("numeric buffers must have equal shapes");
    if (overlaps(out, a) || overlaps(out, b))
        throw py::value_error("numeric output must be disjoint from both inputs");
}

} // namespace

void bind_resident_cuda(py::module_& module) {
    py::class_<Stream, std::shared_ptr<Stream>>(module, "Stream")
        .def(py::init<int>(), py::arg("device") = 0)
        .def_static("borrow", [](int device, std::uintptr_t handle, py::object owner) {
            return std::make_shared<Stream>(device, handle, std::move(owner));
        }, py::arg("device"), py::arg("handle"), py::arg("owner"))
        .def_property_readonly("device", &Stream::device)
        .def_property_readonly("handle", [](const Stream& stream) {
            return reinterpret_cast<std::uintptr_t>(stream.native());
        })
        .def("synchronize", &Stream::synchronize)
        .def("wait_on", &Stream::wait_on, py::arg("other"));

    py::class_<Buffer, std::shared_ptr<Buffer>>(module, "Buffer")
        .def(py::init<std::vector<std::int64_t>, std::shared_ptr<Stream>>(),
             py::arg("shape"), py::arg("stream"))
        .def_static("borrow", &Buffer::borrow, py::arg("ptr"), py::arg("shape"),
                    py::arg("capacity_bytes"), py::arg("stream"), py::arg("owner"),
                    "Borrow aligned CUDA FP32 storage. capacity_bytes is the caller's asserted "
                    "capacity; owner, view and stream must remain alive through queued work completion.")
        .def_property_readonly("shape", [](const Buffer& buffer) {
            py::tuple result(buffer.shape().size());
            for (std::size_t i = 0; i < buffer.shape().size(); ++i)
                result[i] = py::int_(buffer.shape()[i]);
            return result;
        })
        .def_property_readonly("device", &Buffer::device)
        .def_property_readonly("nbytes", &Buffer::nbytes)
        .def_property_readonly("is_native_owned", &Buffer::owned)
        .def("data_ptr", [](const Buffer& buffer) { return reinterpret_cast<std::uintptr_t>(buffer.data()); })
        .def("upload", &Buffer::upload, py::arg("array"))
        .def("download", &Buffer::download)
        .def("__dlpack_device__", [](const Buffer& buffer) {
            return py::make_tuple(static_cast<int>(kDLCUDA), buffer.device());
        })
        .def("__dlpack__", &export_dlpack, py::arg("stream") = py::none());

    py::class_<Event>(module, "Event")
        .def(py::init<std::shared_ptr<Stream>>(), py::arg("stream"))
        .def_static("record", &Event::record_on, py::arg("stream"))
        .def("synchronize", &Event::synchronize)
        .def("elapsed_ms", &Event::elapsed_ms, py::arg("other"));

    py::class_<PreparedCsr, std::shared_ptr<PreparedCsr>>(module, "PreparedCsr")
        .def(py::init(&make_prepared_csr),
             py::arg("indptr"), py::arg("indices"), py::arg("weights"),
             py::arg("source_count"), py::arg("feature_width"), py::arg("stream"))
        .def("apply_into", &PreparedCsr::apply_into, py::arg("input"), py::arg("output"))
        .def("publish_values", &PreparedCsr::publish_values, py::arg("weights"))
        .def_property_readonly("generation", &PreparedCsr::generation)
        .def_property_readonly("prepared_bytes", &PreparedCsr::prepared_bytes)
        .def_property_readonly("pending_value_leases", &PreparedCsr::pending_value_leases)
        .def("close", &PreparedCsr::close);

    module.def("multiply_into", [](const std::shared_ptr<Buffer>& a,
        const std::shared_ptr<Buffer>& b, const std::shared_ptr<Buffer>& out,
        const std::shared_ptr<Stream>& stream) {
        if (!a || !b || !out || !stream) throw py::value_error("buffers and stream are required");
        validate_numeric_buffers(*a, *b, *out, *stream);
        DeviceScope scope(stream->device());
        auto output = resident(*out);
        cuda_check(numeric::enqueue_elementwise_multiply(resident(*a), resident(*b), output,
                                                         stream->native()), "enqueue multiply");
    }, py::arg("a"), py::arg("b"), py::arg("out"), py::arg("stream"));

    module.def("axpby_into", [](float alpha, const std::shared_ptr<Buffer>& a, float beta,
        const std::shared_ptr<Buffer>& b, const std::shared_ptr<Buffer>& out,
        const std::shared_ptr<Stream>& stream) {
        if (!a || !b || !out || !stream) throw py::value_error("buffers and stream are required");
        validate_numeric_buffers(*a, *b, *out, *stream);
        DeviceScope scope(stream->device());
        auto output = resident(*out);
        cuda_check(numeric::enqueue_axpby(alpha, resident(*a), beta, resident(*b), output,
                                          stream->native()), "enqueue axpby");
    }, py::arg("alpha"), py::arg("a"), py::arg("beta"), py::arg("b"), py::arg("out"), py::arg("stream"));
}

} // namespace cellerator::bindings::python
