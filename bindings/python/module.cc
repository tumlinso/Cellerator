#include <Cellerator/bindings/mechanism_handle.hh>
#include <Cellerator/compute/operation/product2/c_api.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#ifdef CELLERATOR_HAS_INDEXED_MECHANISM
#include <cuda_runtime_api.h>
#endif

namespace py = pybind11;
namespace ix = cellerator::compute::operation::indexed;
namespace {

using ProductContext = std::unique_ptr<ce_product2_context, decltype(&ce_product2_destroy)>;

void check_product(ce_product2_status status, const char* operation) {
    if (status == CE_PRODUCT2_INVALID_INDEX)
        throw py::value_error(std::string("native product2 ") + operation + " rejected packet index");
    if (status == CE_PRODUCT2_INVALID_ARGUMENT || status == CE_PRODUCT2_ALIAS)
        throw py::value_error(std::string("native product2 ") + operation + " rejected array binding");
    if (status == CE_PRODUCT2_OVERFLOW)
        throw std::overflow_error(std::string("native product2 ") + operation + " extent overflow");
    if (status != CE_PRODUCT2_SUCCESS)
        throw std::runtime_error(std::string("native product2 ") + operation +
                                 " rejected binding (status " + std::to_string(status) + ")");
}

template<class T>
py::array_t<T> vector_arg(const py::handle& value, const char* name) {
    if (!py::isinstance<py::array>(value))
        throw py::type_error(std::string(name) + " must be a NumPy array");
    auto array = py::reinterpret_borrow<py::array>(value);
    if (!array.dtype().is(py::dtype::of<T>()) || array.ndim() != 1 ||
        !(array.flags() & py::array::c_style) ||
        reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) != 0)
        throw py::value_error(std::string(name) + " must be a contiguous rank-1 NumPy " +
                              (std::is_same_v<T, float> ? "aligned float32" : "aligned int64") + " array");
    return py::reinterpret_borrow<py::array_t<T>>(value);
}

void check_product_vectors(const py::array_t<float>& x, const py::array_t<float>& k,
                           const py::array_t<std::int64_t>& a,
                           const py::array_t<std::int64_t>& b) {
    if (k.size() != a.size() || k.size() != b.size())
        throw py::value_error("k, a and b must have equal packet extents");
    for (py::ssize_t i = 0; i < a.size(); ++i)
        if (a.data()[i] < 0 || b.data()[i] < 0 ||
            static_cast<std::uint64_t>(a.data()[i]) >= static_cast<std::uint64_t>(x.size()) ||
            static_cast<std::uint64_t>(b.data()[i]) >= static_cast<std::uint64_t>(x.size()))
            throw py::value_error("packet index outside x extent");
}

ce_product2_binding bind_product(const py::array_t<float>& x,
                                 const py::array_t<float>& k,
                                 py::array_t<float>* y,
                                 const py::array_t<float>* g,
                                 const py::array_t<float>* dx,
                                 const py::array_t<float>* dk,
                                 py::array_t<float>* dy,
                                 py::array_t<float>* gx,
                                 py::array_t<float>* gk) {
    ce_product2_binding v{};
    v.x = x.data(); v.x_count = static_cast<std::uint64_t>(x.size());
    v.k = k.data(); v.k_count = static_cast<std::uint64_t>(k.size());
    if (y) { v.y = y->mutable_data(); v.y_count = static_cast<std::uint64_t>(y->size()); }
    if (g) { v.g = g->data(); v.g_count = static_cast<std::uint64_t>(g->size()); }
    if (dx) { v.dx = dx->data(); v.dx_count = static_cast<std::uint64_t>(dx->size()); }
    if (dk) { v.dk = dk->data(); v.dk_count = static_cast<std::uint64_t>(dk->size()); }
    if (dy) { v.dy = dy->mutable_data(); v.dy_count = static_cast<std::uint64_t>(dy->size()); }
    if (gx) { v.gx = gx->mutable_data(); v.gx_count = static_cast<std::uint64_t>(gx->size()); }
    if (gk) { v.gk = gk->mutable_data(); v.gk_count = static_cast<std::uint64_t>(gk->size()); }
    return v;
}

ProductContext make_product(const py::array_t<float>& x,
                            const py::array_t<float>& k,
                            const py::array_t<std::int64_t>& a,
                            const py::array_t<std::int64_t>& b,
                            std::uint64_t generation) {
    check_product_vectors(x, k, a, b);
    ce_product2_context* raw = nullptr;
    check_product(ce_product2_create(static_cast<std::uint64_t>(x.size()),
        static_cast<std::uint64_t>(k.size()), a.data(), static_cast<std::uint64_t>(a.size()),
        b.data(), static_cast<std::uint64_t>(b.size()), generation, &raw), "prepare");
    return ProductContext(raw, ce_product2_destroy);
}

class PreparedProduct2 final {
public:
    PreparedProduct2(std::uint64_t input_count,
                     py::array_t<std::int64_t> a,
                     py::array_t<std::int64_t> b,
                     std::uint64_t generation)
        : a_(std::move(a)), b_(std::move(b)), generation_(generation), input_count_(input_count) {
        if (a_.ndim() != 1 || b_.ndim() != 1 ||
            !(a_.flags() & py::array::c_style) || !(b_.flags() & py::array::c_style))
            throw py::value_error("a and b must be contiguous rank-1 int64 arrays");
        if (a_.size() != b_.size()) throw py::value_error("a and b extents must match");
        ce_product2_context* raw = nullptr;
        check_product(ce_product2_create(input_count, static_cast<std::uint64_t>(a_.size()),
            a_.data(), static_cast<std::uint64_t>(a_.size()), b_.data(),
            static_cast<std::uint64_t>(b_.size()), generation_, &raw), "prepare");
        context_.reset(raw);
    }

    py::array_t<float> forward(py::array x, py::array k) const {
        auto xf = vector_arg<float>(x, "x"); auto kf = vector_arg<float>(k, "k");
        validate_extents(xf, kf);
        py::array_t<float> y(a_.size());
        auto v = bind_product(xf, kf, &y, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        v.expected_structure_generation = v.current_structure_generation = generation_;
        check_product(ce_product2_forward(context_.get(), &v), "forward");
        return y;
    }

    py::tuple vjp(py::array x, py::array k, py::array g) const {
        auto xf = vector_arg<float>(x, "x"); auto kf = vector_arg<float>(k, "k");
        auto gf = vector_arg<float>(g, "g");
        validate_extents(xf, kf);
        if (gf.size() != a_.size()) throw py::value_error("cotangent extent differs from packet extent");
        py::array_t<float> gx(xf.size()), gk(kf.size());
        auto v = bind_product(xf, kf, nullptr, &gf, nullptr, nullptr, nullptr, &gx, &gk);
        v.expected_structure_generation = v.current_structure_generation = generation_;
        check_product(ce_product2_vjp(context_.get(), &v), "vjp");
        return py::make_tuple(std::move(gx), std::move(gk));
    }

    py::array_t<float> jvp(py::array x, py::array k, py::array dx, py::array dk) const {
        auto xf = vector_arg<float>(x, "x"); auto kf = vector_arg<float>(k, "k");
        auto dxf = vector_arg<float>(dx, "dx"); auto dkf = vector_arg<float>(dk, "dk");
        validate_extents(xf, kf);
        if (dxf.size() != xf.size() || dkf.size() != kf.size())
            throw py::value_error("JVP directions must match x and k extents");
        py::array_t<float> dy(a_.size());
        auto v = bind_product(xf, kf, nullptr, nullptr, &dxf, &dkf, &dy, nullptr, nullptr);
        v.expected_structure_generation = v.current_structure_generation = generation_;
        check_product(ce_product2_jvp(context_.get(), &v), "jvp");
        return dy;
    }

    std::uint64_t input_count() const { return context_ ? input_count_ : 0; }
    std::uint64_t packet_count() const { return static_cast<std::uint64_t>(a_.size()); }
    std::uint64_t structure_generation() const { return generation_; }

private:
    void validate_extents(const py::array_t<float>& x, const py::array_t<float>& k) const {
        if (static_cast<std::uint64_t>(x.size()) != input_count_)
            throw py::value_error("x extent differs from prepared input extent");
        if (static_cast<std::uint64_t>(k.size()) != static_cast<std::uint64_t>(a_.size()))
            throw py::value_error("k extent differs from prepared packet extent");
    }
    py::array_t<std::int64_t> a_, b_;
    std::uint64_t generation_ = 0;
    std::uint64_t input_count_ = 0;
    ProductContext context_{nullptr, ce_product2_destroy};
};

std::uint64_t bits(std::int64_t word) { return static_cast<std::uint64_t>(word); }
ix::identity decode_id(const std::vector<std::int64_t>& words, std::size_t index) {
    if (words.size() < 2 * (index + 1)) throw std::invalid_argument("identity word vector has an incomplete pair");
    return {bits(words[2 * index]), bits(words[2 * index + 1])};
}
cellerator::execution::persistent_axis_identity decode_axis(
    const std::vector<std::int64_t>& words, std::size_t offset) {
    if (words.size() < offset + 8) throw std::invalid_argument("axis identity needs eight words");
    cellerator::execution::persistent_axis_identity axis{};
    axis.header.schema_version = cellerator::execution::biological_abi_version;
    axis.header.kind = cellerator::execution::serialized_record_kind::persistent_axis_identity;
    axis.header.byte_count = sizeof(axis);
    axis.domain = {bits(words[offset]), bits(words[offset + 1])};
    axis.order = {bits(words[offset + 2]), bits(words[offset + 3])};
    axis.geometry = {bits(words[offset + 4]), bits(words[offset + 5])};
    axis.partition = {bits(words[offset + 6]), bits(words[offset + 7])};
    return axis;
}

ix::mechanism_declaration decode_mechanism(
    const std::vector<std::int64_t>& axis_words,
    const std::vector<std::int64_t>& axis_extents,
    const std::vector<std::int64_t>& coefficient_id_words,
    const std::vector<std::int64_t>& mechanism_id_words,
    const std::vector<std::int64_t>& arg_offsets,
    const std::vector<std::int64_t>& arg_slots,
    const std::vector<std::int64_t>& arg_role_words,
    const std::vector<std::int64_t>& arg_axes,
    const std::vector<std::int64_t>& arg_indices,
    const std::vector<std::int64_t>& coefficient_bindings,
    const std::vector<std::int64_t>& output_offsets,
    const std::vector<std::int64_t>& output_slots,
    const std::vector<std::int64_t>& output_role_words,
    const std::vector<std::int64_t>& output_axes,
    const std::vector<std::int64_t>& output_indices,
    const std::vector<std::int64_t>& output_assembly_words,
    const std::vector<double>& output_scales,
    std::int64_t max_batch, std::int64_t max_live_forwards,
    std::int64_t precision) {
    if (axis_words.size() != 24 || axis_extents.size() != 3)
        throw std::invalid_argument("axis declaration must contain 24 identity and 3 extent words");
    if (max_batch <= 0 || max_live_forwards <= 0)
        throw std::invalid_argument("mechanism capacities must be positive");
    if (static_cast<std::uint64_t>(max_live_forwards) >
        std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("max_live_forwards must fit uint32");
    if (precision != 0 && precision != 1) throw std::invalid_argument("precision must be 0 (f32) or 1 (mixed_f16)");
    if (std::any_of(axis_extents.begin(), axis_extents.end(), [](auto n) { return n <= 0; }))
        throw std::invalid_argument("axis extents must be positive");
    if (coefficient_id_words.size() != 2 * static_cast<std::size_t>(axis_extents[2]))
        throw std::invalid_argument("coefficient logical IDs must contain one low/high pair per coefficient");
    if (mechanism_id_words.empty() || mechanism_id_words.size() % 2)
        throw std::invalid_argument("mechanism IDs must be nonempty low/high pairs");
    const auto count = mechanism_id_words.size() / 2;
    if (arg_offsets.size() != count + 1 || output_offsets.size() != count + 1 ||
        coefficient_bindings.size() != count)
        throw std::invalid_argument("incidence offsets and coefficient bindings disagree with mechanism count");
    if (arg_slots.size() != arg_axes.size() || arg_slots.size() != arg_indices.size() ||
        arg_role_words.size() != arg_slots.size() * 2 ||
        output_slots.size() != output_axes.size() || output_slots.size() != output_indices.size() ||
        output_role_words.size() != output_slots.size() * 2 ||
        output_assembly_words.size() != output_slots.size() * 2 || output_scales.size() != output_slots.size())
        throw std::invalid_argument("incidence arrays disagree");
    if (arg_offsets.front() != 0 || arg_offsets.back() != static_cast<std::int64_t>(arg_slots.size()) ||
        output_offsets.front() != 0 || output_offsets.back() != static_cast<std::int64_t>(output_slots.size()))
        throw std::invalid_argument("incidence offsets do not span flattened entries");

    ix::mechanism_declaration declaration;
    declaration.input = {decode_axis(axis_words, 0), static_cast<std::uint64_t>(axis_extents[0])};
    declaration.output = {decode_axis(axis_words, 8), static_cast<std::uint64_t>(axis_extents[1])};
    declaration.coefficients = {decode_axis(axis_words, 16), static_cast<std::uint64_t>(axis_extents[2])};
    declaration.max_batch = static_cast<std::uint64_t>(max_batch);
    declaration.max_live_forwards = static_cast<std::uint32_t>(max_live_forwards);
    declaration.precision = precision == 0 ? ix::training_precision::f32 : ix::training_precision::mixed_f16;
    for (std::size_t i = 0; i < static_cast<std::size_t>(axis_extents[2]); ++i)
        declaration.coefficient_ids.push_back(decode_id(coefficient_id_words, i));
    for (std::size_t m = 0; m < count; ++m) {
        if (arg_offsets[m] < 0 || arg_offsets[m] > arg_offsets[m + 1] ||
            output_offsets[m] < 0 || output_offsets[m] > output_offsets[m + 1])
            throw std::invalid_argument("incidence offsets must be monotonic");
        ix::product_mechanism item;
        item.instance = decode_id(mechanism_id_words, m);
        if (coefficient_bindings[m] < 0) throw std::invalid_argument("coefficient binding must be nonnegative");
        item.coefficient = static_cast<std::uint64_t>(coefficient_bindings[m]);
        for (auto i = arg_offsets[m]; i < arg_offsets[m + 1]; ++i) {
            const auto n = static_cast<std::size_t>(i);
            if (arg_slots[n] < 0 || arg_axes[n] < 0 || arg_indices[n] < 0)
                throw std::invalid_argument("argument incidence values must be nonnegative");
            item.arguments.push_back({static_cast<std::uint64_t>(arg_slots[n]), decode_id(arg_role_words, n),
                static_cast<std::uint64_t>(arg_axes[n]), static_cast<std::uint64_t>(arg_indices[n])});
        }
        for (auto i = output_offsets[m]; i < output_offsets[m + 1]; ++i) {
            const auto n = static_cast<std::size_t>(i);
            if (output_slots[n] < 0 || output_axes[n] < 0 || output_indices[n] < 0)
                throw std::invalid_argument("output incidence values must be nonnegative");
            ix::output_index destination{};
            destination.slot = static_cast<std::uint64_t>(output_slots[n]);
            destination.role = decode_id(output_role_words, n);
            destination.axis = static_cast<std::uint64_t>(output_axes[n]);
            destination.index = static_cast<std::uint64_t>(output_indices[n]);
            destination.assembly_owner = decode_id(output_assembly_words, n);
            destination.effect.update = cellerator::execution::output_update_kind::accumulate;
            destination.effect.requires_initialized_destination = true;
            item.outputs.push_back({destination, static_cast<float>(output_scales[n])});
        }
        declaration.mechanisms.push_back(std::move(item));
    }
    return declaration;
}

std::vector<float> float_vector(const py::array& input, const char* name) {
    auto values = vector_arg<float>(input, name);
    return {values.data(), values.data() + values.size()};
}

struct PythonTape {
    std::shared_ptr<ix::prepared_mechanism_program> program;
    std::shared_ptr<ix::mechanism_tape> tape;
    std::atomic<bool> consumed{false};
    int device = 0;
};

#ifdef CELLERATOR_HAS_INDEXED_MECHANISM
void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + operation + ": " + cudaGetErrorString(status));
}
struct DeviceScope {
    int previous = 0;
    explicit DeviceScope(int device) {
        cuda_check(cudaGetDevice(&previous), "get device");
        if (previous != device) cuda_check(cudaSetDevice(device), "set device");
    }
    ~DeviceScope() { (void)cudaSetDevice(previous); }
};
struct DeviceBuffer {
    void* value = nullptr;
    explicit DeviceBuffer(std::size_t bytes) { if (bytes) cuda_check(cudaMalloc(&value, bytes), "allocate"); }
    ~DeviceBuffer() { if (value) (void)cudaFree(value); }
};
void copy_to_device(void* dst, const void* src, std::size_t bytes, cudaStream_t stream) {
    if (bytes) cuda_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), "upload");
}
void copy_to_host(void* dst, const void* src, std::size_t bytes, cudaStream_t stream) {
    if (bytes) cuda_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream), "download");
}
py::tuple native_forward(const std::shared_ptr<cellerator::bindings::MechanismHandle>& handle,
    py::array input, std::uintptr_t stream_address) {
    if (!py::isinstance<py::array>(input) || input.ndim() != 2 || !(input.flags() & py::array::c_style))
        throw py::value_error("input must be a contiguous rank-2 NumPy float32 or float16 array");
    const bool half = input.dtype().kind() == 'e' && input.itemsize() == 2 &&
        input.dtype().attr("isnative").cast<bool>();
    if (!half && !(input.dtype().is(py::dtype::of<float>()) && input.itemsize() == 4))
        throw py::value_error("input must be a contiguous rank-2 NumPy float32 or float16 array");
    const auto input_alignment = half ? alignof(std::uint16_t) : alignof(float);
    if (reinterpret_cast<std::uintptr_t>(input.data()) % input_alignment != 0)
        throw py::value_error("input must be aligned for its NumPy dtype");
    const auto& declaration = handle->declaration();
    const auto batch = static_cast<std::uint64_t>(input.shape(0));
    const auto width = static_cast<std::uint64_t>(input.shape(1));
    if (!batch || width != declaration.input.extent || batch > declaration.max_batch)
        throw py::value_error("input batch or width does not match prepared mechanism capacity");
    if (declaration.precision == ix::training_precision::f32 && half)
        throw py::value_error("FP32 mechanism requires FP32 input");
    if (half && declaration.precision != ix::training_precision::mixed_f16)
        throw py::value_error("FP16 input requires mixed_f16 mechanism precision");
    const auto stream = reinterpret_cast<cudaStream_t>(stream_address);
    DeviceScope device(handle->owner()->device());
    const std::size_t input_bytes = static_cast<std::size_t>(batch * width) * (half ? 2 : 4);
    const std::size_t output_bytes = static_cast<std::size_t>(batch * declaration.output.extent) * sizeof(float);
    DeviceBuffer d_input(input_bytes), d_output(output_bytes);
    copy_to_device(d_input.value, input.data(), input_bytes, stream);
    auto output = py::array_t<float>({static_cast<py::ssize_t>(batch),
                                      static_cast<py::ssize_t>(declaration.output.extent)});
    auto tape = handle->program()->forward(d_input.value, half, batch, declaration.input.identity,
        static_cast<float*>(d_output.value), stream);
    copy_to_host(output.mutable_data(), d_output.value, output_bytes, stream);
    cuda_check(cudaStreamSynchronize(stream), "synchronize forward");
    auto result = std::make_shared<PythonTape>();
    result->program = handle->program(); result->tape = std::move(tape);
    result->device = handle->owner()->device();
    return py::make_tuple(std::move(output), std::move(result));
}
py::tuple native_backward(const std::shared_ptr<PythonTape>& tape, py::array cotangent,
                          std::uintptr_t stream_address) {
    if (!cotangent.dtype().is(py::dtype::of<float>()) || cotangent.ndim() != 2 ||
        !(cotangent.flags() & py::array::c_style))
        throw py::value_error("output_gradient must be a contiguous rank-2 NumPy float32 array");
    if (reinterpret_cast<std::uintptr_t>(cotangent.data()) % alignof(float) != 0)
        throw py::value_error("output_gradient must be aligned float32");
    auto g = py::reinterpret_borrow<py::array_t<float>>(cotangent);
    if (!tape->tape) throw std::runtime_error("mechanism tape has already been consumed");
    if (static_cast<std::uint64_t>(cotangent.shape(0)) != tape->tape->batch() ||
        static_cast<std::uint64_t>(cotangent.shape(1)) != tape->program->declaration().output.extent)
        throw py::value_error("output gradient shape does not match saved mechanism output");
    bool expected = false;
    if (!tape->consumed.compare_exchange_strong(expected, true))
        throw std::runtime_error("mechanism tape supports one backward only");
    const auto& declaration = tape->program->declaration();
    const auto batch = tape->tape->batch();
    const auto stream = reinterpret_cast<cudaStream_t>(stream_address);
    DeviceScope device(tape->device);
    const std::size_t gbytes = static_cast<std::size_t>(g.size()) * sizeof(float);
    const std::size_t dxbytes = static_cast<std::size_t>(batch * declaration.input.extent) * sizeof(float);
    const std::size_t dkbytes = static_cast<std::size_t>(declaration.coefficients.extent) * sizeof(float);
    DeviceBuffer d_g(gbytes), d_dx(dxbytes), d_dk(dkbytes);
    copy_to_device(d_g.value, g.data(), gbytes, stream);
    tape->program->backward(*tape->tape, static_cast<const float*>(d_g.value),
        static_cast<float*>(d_dx.value), static_cast<float*>(d_dk.value), stream);
    py::array_t<float> dx({static_cast<py::ssize_t>(batch),
                           static_cast<py::ssize_t>(declaration.input.extent)});
    py::array_t<float> dk(declaration.coefficients.extent);
    copy_to_host(dx.mutable_data(), d_dx.value, dxbytes, stream);
    copy_to_host(dk.mutable_data(), d_dk.value, dkbytes, stream);
    cuda_check(cudaStreamSynchronize(stream), "synchronize backward");
    tape->tape.reset();
    return py::make_tuple(std::move(dx), std::move(dk));
}
#endif

} // namespace

namespace cellerator::bindings {
} // namespace cellerator::bindings

PYBIND11_MODULE(_native, m) {
    m.doc() = "Thin Python bindings to Cellerator native numerical operations";
    m.def("mechanisms_available", [] {
#ifdef CELLERATOR_HAS_INDEXED_MECHANISM
        return true;
#else
        return false;
#endif
    });

    py::class_<PreparedProduct2, std::shared_ptr<PreparedProduct2>>(m, "PreparedProduct2")
        .def("forward", &PreparedProduct2::forward)
        .def("vjp", &PreparedProduct2::vjp)
        .def("jvp", &PreparedProduct2::jvp)
        .def_property_readonly("input_count", &PreparedProduct2::input_count)
        .def_property_readonly("packet_count", &PreparedProduct2::packet_count)
        .def_property_readonly("structure_generation", &PreparedProduct2::structure_generation);

    m.def("prepare_product2", [](py::array a, py::array b, std::uint64_t input_count,
                                   std::uint64_t generation) {
        auto aa = vector_arg<std::int64_t>(a, "a");
        auto bb = vector_arg<std::int64_t>(b, "b");
        return std::make_shared<PreparedProduct2>(input_count, std::move(aa), std::move(bb), generation);
    }, py::arg("a"), py::arg("b"), py::arg("input_count"), py::arg("structure_generation") = 0);

    m.def("product2_forward", [](py::array x, py::array k, py::array a, py::array b) {
        auto xf = vector_arg<float>(x, "x"); auto kf = vector_arg<float>(k, "k");
        auto aa = vector_arg<std::int64_t>(a, "a"); auto bb = vector_arg<std::int64_t>(b, "b");
        auto owner = make_product(xf, kf, aa, bb, 0);
        py::array_t<float> y(aa.size());
        auto binding = bind_product(xf, kf, &y, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        check_product(ce_product2_forward(owner.get(), &binding), "forward"); return y;
    });
    m.def("product2_vjp", [](py::array x, py::array k, py::array a, py::array b, py::array g) {
        auto xf = vector_arg<float>(x, "x"); auto kf = vector_arg<float>(k, "k");
        auto aa = vector_arg<std::int64_t>(a, "a"); auto bb = vector_arg<std::int64_t>(b, "b");
        auto gf = vector_arg<float>(g, "g"); auto owner = make_product(xf, kf, aa, bb, 0);
        if (gf.size() != kf.size()) throw py::value_error("cotangent extent differs from packet extent");
        py::array_t<float> gx(xf.size()), gk(kf.size());
        auto binding = bind_product(xf, kf, nullptr, &gf, nullptr, nullptr, nullptr, &gx, &gk);
        check_product(ce_product2_vjp(owner.get(), &binding), "vjp"); return py::make_tuple(gx, gk);
    });
    m.def("product2_jvp", [](py::array x, py::array k, py::array a, py::array b,
                              py::array dx, py::array dk) {
        auto xf = vector_arg<float>(x, "x"); auto kf = vector_arg<float>(k, "k");
        auto aa = vector_arg<std::int64_t>(a, "a"); auto bb = vector_arg<std::int64_t>(b, "b");
        auto dxf = vector_arg<float>(dx, "dx"); auto dkf = vector_arg<float>(dk, "dk");
        auto owner = make_product(xf, kf, aa, bb, 0);
        if (dxf.size() != xf.size() || dkf.size() != kf.size())
            throw py::value_error("JVP directions must match x and k extents");
        py::array_t<float> dy(kf.size());
        auto binding = bind_product(xf, kf, nullptr, nullptr, &dxf, &dkf, &dy, nullptr, nullptr);
        check_product(ce_product2_jvp(owner.get(), &binding), "jvp"); return dy;
    });

#ifdef CELLERATOR_HAS_INDEXED_MECHANISM
    using Handle = cellerator::bindings::MechanismHandle;
    py::class_<PythonTape, std::shared_ptr<PythonTape>>(m, "MechanismTape")
        .def("backward", &native_backward, py::arg("output_gradient"), py::arg("stream") = 0)
        .def_property_readonly("batch", [](const PythonTape& tape) { return tape.tape ? tape.tape->batch() : 0; })
        .def_property_readonly("consumed", [](const PythonTape& tape) { return tape.consumed.load(); });
    py::class_<Handle, std::shared_ptr<Handle>>(m, "MechanismHandle", py::dynamic_attr())
        .def(py::init([](py::array initial,
            const std::vector<std::int64_t>& axis_words,
            const std::vector<std::int64_t>& axis_extents,
            const std::vector<std::int64_t>& coefficient_id_words,
            const std::vector<std::int64_t>& mechanism_id_words,
            const std::vector<std::int64_t>& arg_offsets,
            const std::vector<std::int64_t>& arg_slots,
            const std::vector<std::int64_t>& arg_role_words,
            const std::vector<std::int64_t>& arg_axes,
            const std::vector<std::int64_t>& arg_indices,
            const std::vector<std::int64_t>& coefficient_bindings,
            const std::vector<std::int64_t>& output_offsets,
            const std::vector<std::int64_t>& output_slots,
            const std::vector<std::int64_t>& output_role_words,
            const std::vector<std::int64_t>& output_axes,
            const std::vector<std::int64_t>& output_indices,
            const std::vector<std::int64_t>& output_assembly_words,
            const std::vector<double>& output_scales,
            std::int64_t max_batch, std::int64_t max_live_forwards,
            std::int64_t precision, int device) {
            auto coefficients = vector_arg<float>(initial, "initial_coefficients");
            auto declaration = decode_mechanism(axis_words, axis_extents,
                coefficient_id_words, mechanism_id_words, arg_offsets, arg_slots,
                arg_role_words, arg_axes, arg_indices, coefficient_bindings,
                output_offsets, output_slots, output_role_words, output_axes,
                output_indices, output_assembly_words, output_scales,
                max_batch, max_live_forwards, precision);
            return std::make_shared<Handle>(std::move(declaration),
                std::span<const float>(coefficients.data(), coefficients.size()), device);
        }), py::arg("initial_coefficients"), py::arg("axis_words"), py::arg("axis_extents"),
            py::arg("coefficient_id_words"), py::arg("mechanism_id_words"),
            py::arg("arg_offsets"), py::arg("arg_slots"), py::arg("arg_role_words"),
            py::arg("arg_axes"), py::arg("arg_indices"), py::arg("coefficient_bindings"),
            py::arg("output_offsets"), py::arg("output_slots"), py::arg("output_role_words"),
            py::arg("output_axes"), py::arg("output_indices"), py::arg("output_assembly_words"),
            py::arg("output_scales"), py::arg("max_batch"), py::arg("max_live_forwards") = 8,
            py::arg("precision") = 0, py::arg("device") = 0)
        .def_property_readonly("device", [](const Handle& h) { return h.owner()->device(); })
        .def_property_readonly("generation", [](const Handle& h) { return h.owner()->generation(); })
        .def_property_readonly("poisoned", [](const Handle& h) { return h.owner()->poisoned(); })
        .def_property_readonly("coefficient_count", [](const Handle& h) { return h.owner()->size(); })
        .def_property_readonly("max_batch", [](const Handle& h) { return h.declaration().max_batch; })
        .def_property_readonly("max_live_forwards", [](const Handle& h) { return h.declaration().max_live_forwards; })
        .def_property_readonly("input_extent", [](const Handle& h) { return h.declaration().input.extent; })
        .def_property_readonly("output_extent", [](const Handle& h) { return h.declaration().output.extent; })
        .def_property_readonly("reserved_bytes", [](const Handle& h) { return h.program()->reserved_bytes(); })
        .def_property_readonly("precision", [](const Handle& h) {
            return h.declaration().precision == ix::training_precision::f32 ? "f32" : "mixed_f16";
        })
        .def("owns_storage", &Handle::owns_storage, py::arg("address"), py::arg("bytes"))
        .def("snapshot", [](const Handle& h, std::uintptr_t stream) {
            const auto values = h.snapshot(reinterpret_cast<void*>(stream));
            py::array_t<float> out(values.size());
            std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(float));
            return out;
        }, py::arg("stream") = 0)
        .def("restore", [](Handle& h, py::array values, std::uintptr_t stream) {
            auto v = vector_arg<float>(values, "values");
            h.restore(std::span<const float>(v.data(), v.size()), reinterpret_cast<void*>(stream));
        }, py::arg("values"), py::arg("stream") = 0)
        .def("preflight_write", &Handle::preflight_write)
        .def("begin_write", [](Handle& h, std::uintptr_t stream) { h.begin_write(reinterpret_cast<void*>(stream)); }, py::arg("stream") = 0)
        .def("publish_write", [](Handle& h, std::uintptr_t stream) { h.publish_write(reinterpret_cast<void*>(stream)); }, py::arg("stream") = 0)
        .def("poison", &Handle::poison)
        .def("forward", &native_forward, py::arg("input"), py::arg("stream") = 0);
#endif
}
