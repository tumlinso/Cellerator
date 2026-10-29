#include <Cellerator/modeling/cuda/plans.hh>
#include <Cellerator/modeling/cuda/grouped_linear.hh>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py = pybind11;
using namespace cellerator::modeling::cuda;

// A Python transfer ticket keeps its buffer/host/stream arguments alive via
// keep_alive policies and waits before those references can be released.
class PinnedTransfer final {
public:
    explicit PinnedTransfer(int device) : event_(device) {}
    ~PinnedTransfer() noexcept {
        if (event_.recorded()) { try { event_.synchronize(); } catch (...) {} }
    }
    void record(const Stream& stream) { event_.record(stream); }
    void wait() const { event_.synchronize(); }
    bool ready() const { return event_.ready(); }
private:
    Event event_;
};

PYBIND11_MODULE(_modeling_cuda, m) {
    m.doc() = "Explicit CUDA allocations, streams, sparse dispatch and state quantization; no Torch dependency.";
    py::register_exception<Error>(m, "CudaError");
    py::enum_<DType>(m, "DType").value("float16", DType::float16).value("float32", DType::float32);
    py::class_<DeviceView>(m, "DeviceView")
        .def(py::init([](std::uintptr_t ptr, std::size_t bytes, int device) {
            if (device < 0 || (bytes && !ptr)) throw std::invalid_argument("invalid borrowed device view");
            return DeviceView{reinterpret_cast<void*>(ptr), bytes, device};
        }), py::arg("pointer"), py::arg("nbytes"), py::arg("device"))
        .def_property_readonly("pointer", [](DeviceView v) { return reinterpret_cast<std::uintptr_t>(v.data); })
        .def_readonly("nbytes", &DeviceView::bytes).def_readonly("device", &DeviceView::device)
        .def("subview", &DeviceView::subview, py::keep_alive<0, 1>());
    py::class_<Stream>(m, "Stream")
        .def(py::init<int, bool>(), py::arg("device") = 0, py::arg("nonblocking") = true)
        .def_static("borrow", [](std::uintptr_t pointer, int device) { return Stream::borrow(reinterpret_cast<cudaStream_t>(pointer), device); }, py::arg("pointer"), py::arg("device"))
        .def_property_readonly("pointer", [](const Stream& s) { return reinterpret_cast<std::uintptr_t>(s.get()); })
        .def_property_readonly("device", &Stream::device)
        .def_property_readonly("owns_handle", &Stream::owns_handle)
        .def("synchronize", &Stream::synchronize, py::call_guard<py::gil_scoped_release>())
        .def("ready", &Stream::ready).def("wait", &Stream::wait);
    py::class_<Event>(m, "Event")
        .def(py::init<int, bool>(), py::arg("device") = 0, py::arg("timing") = false)
        .def("record", &Event::record).def("ready", &Event::ready)
        .def("synchronize", &Event::synchronize, py::call_guard<py::gil_scoped_release>())
        .def("elapsed_ms", &Event::elapsed_ms);
    py::class_<PinnedBuffer>(m, "PinnedBuffer", py::buffer_protocol())
        .def(py::init<std::size_t>(), py::arg("nbytes"))
        .def_property_readonly("nbytes", &PinnedBuffer::bytes)
        .def_buffer([](PinnedBuffer& b) { return py::buffer_info(b.data(), 1, py::format_descriptor<unsigned char>::format(), 1, {b.bytes()}, {1}); });
    py::class_<PinnedTransfer>(m, "PinnedTransfer")
        .def("wait", &PinnedTransfer::wait, py::call_guard<py::gil_scoped_release>())
        .def("ready", &PinnedTransfer::ready);
    py::class_<DeviceBuffer>(m, "DeviceBuffer")
        .def(py::init<std::size_t, int>(), py::arg("nbytes"), py::arg("device") = 0)
        .def_property_readonly("nbytes", &DeviceBuffer::bytes)
        .def_property_readonly("device", &DeviceBuffer::device)
        .def("view", [](DeviceBuffer& b) { return b.view(); }, py::keep_alive<0, 1>())
        .def("zero_async", &DeviceBuffer::zero_async)
        .def("copy_from_async", [](DeviceBuffer& b, DeviceView v, const Stream& s) { b.copy_from_async(v, s); })
        .def("upload", [](DeviceBuffer& b, const py::array& array, const Stream& stream, std::size_t offset) {
            if (!(array.flags() & py::array::c_style) || array.dtype().has_fields() || array.dtype().kind() == 'O')
                throw std::invalid_argument("upload requires a contiguous plain numeric NumPy array");
            py::gil_scoped_release release;
            b.upload_async(array.data(), static_cast<std::size_t>(array.nbytes()), stream, offset); stream.synchronize();
        }, py::arg("array"), py::arg("stream"), py::arg("offset") = 0)
        .def("download", [](const DeviceBuffer& b, py::dtype dtype, const std::vector<py::ssize_t>& shape, const Stream& stream) {
            std::size_t n = 1;
            for (auto d : shape) { if (d < 0) throw std::invalid_argument("negative shape"); n = checked_bytes(n, static_cast<std::size_t>(d)); }
            if (dtype.has_fields() || dtype.kind() == 'O') throw std::invalid_argument("download requires a plain numeric dtype");
            const auto bytes = checked_bytes(n, static_cast<std::size_t>(dtype.itemsize()));
            if (bytes != b.bytes()) throw std::invalid_argument("download shape and dtype must exactly cover buffer");
            py::array result(dtype, shape);
            { py::gil_scoped_release release; b.download_async(result.mutable_data(), bytes, stream); stream.synchronize(); }
            return result;
        }, py::arg("dtype"), py::arg("shape"), py::arg("stream"))
        .def("upload_pinned_async", [](DeviceBuffer& b, const PinnedBuffer& host, const Stream& stream) {
            auto ticket = std::make_unique<PinnedTransfer>(stream.device());
            try { b.upload_async(host.data(), host.bytes(), stream); ticket->record(stream); }
            catch (...) { try { stream.synchronize(); } catch (...) {} throw; }
            return ticket;
        }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>(), py::keep_alive<0, 3>())
        .def("download_pinned_async", [](const DeviceBuffer& b, PinnedBuffer& host, const Stream& stream) {
            auto ticket = std::make_unique<PinnedTransfer>(stream.device());
            try { b.download_async(host.data(), host.bytes(), stream); ticket->record(stream); }
            catch (...) { try { stream.synchronize(); } catch (...) {} throw; }
            return ticket;
        }, py::keep_alive<0, 1>(), py::keep_alive<0, 2>(), py::keep_alive<0, 3>());
    py::class_<DispatchPlan>(m, "DispatchPlan")
        .def(py::init<std::int64_t, std::vector<std::int64_t>, std::vector<float>, std::vector<std::int64_t>, std::vector<std::int64_t>, std::vector<std::int64_t>, const Stream&>(),
             py::arg("origin_count"), py::arg("selection_origin"), py::arg("weights"), py::arg("origin_offsets"), py::arg("origin_dispatch"), py::arg("expert_offsets"), py::arg("stream"))
        .def_property_readonly("origin_count", &DispatchPlan::origin_count)
        .def_property_readonly("selection_count", &DispatchPlan::selection_count)
        .def_property_readonly("expert_offsets", &DispatchPlan::expert_offsets)
        .def_property_readonly("device", &DispatchPlan::device)
        .def("gather", [](const DispatchPlan& p, DeviceView input, DeviceView output, std::int64_t f, DType t, const Stream& s) { p.gather(input, output, f, t, s); })
        .def("combine", [](const DispatchPlan& p, DeviceView input, DeviceView output, std::int64_t f, DType t, const Stream& s) { p.combine(input, output, f, t, s); })
        .def("gather_backward", [](const DispatchPlan& p, DeviceView input, DeviceView output, std::int64_t f, DType t, const Stream& s) { p.gather_backward(input, output, f, t, s); })
        .def("combine_backward", [](const DispatchPlan& p, DeviceView grad, DeviceView expert, DeviceView grad_expert, DeviceView grad_weight, std::int64_t f, DType t, const Stream& s) { p.combine_backward(grad, expert, grad_expert, grad_weight, f, t, s); });
    py::class_<QuantizationProfile, std::shared_ptr<QuantizationProfile>>(m, "QuantizationProfile")
        .def(py::init<std::vector<std::uint32_t>, std::vector<float>, std::vector<std::uint32_t>, std::uint32_t, const Stream&>(),
             py::arg("offsets"), py::arg("levels"), py::arg("feature_map"), py::arg("bits"), py::arg("stream"))
        .def_property_readonly("device", &QuantizationProfile::device)
        .def_property_readonly("feature_count", &QuantizationProfile::feature_count)
        .def_property_readonly("bits", &QuantizationProfile::bits);
    py::class_<QuantizedState>(m, "QuantizedState")
        .def(py::init<std::shared_ptr<const QuantizationProfile>, std::uint32_t, std::uint32_t>(), py::arg("profile"), py::arg("rows"), py::arg("override_capacity") = 0)
        .def("encode", [](QuantizedState& q, DeviceView v, DType t, py::object mask, const Stream& s) { q.encode(v, t, mask.is_none() ? ConstDeviceView{} : static_cast<ConstDeviceView>(mask.cast<DeviceView>()), s); })
        .def("decode", &QuantizedState::decode)
        .def("status", &QuantizedState::status, py::call_guard<py::gil_scoped_release>())
        .def("override_count", &QuantizedState::override_count, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("rows", &QuantizedState::rows)
        .def_property_readonly("features", &QuantizedState::features)
        .def_property_readonly("packed_bytes", &QuantizedState::packed_bytes)
        .def_property_readonly("override_capacity", &QuantizedState::override_capacity)
        .def_property_readonly("device", &QuantizedState::device);
    py::enum_<GradientMode>(m, "GradientMode")
        .value("overwrite", GradientMode::overwrite).value("accumulate", GradientMode::accumulate);
    py::class_<GroupedLinear>(m, "GroupedLinear")
        .def(py::init([](std::int64_t inputs, std::int64_t outputs,
                         std::vector<std::int64_t> offsets,
                         const std::vector<DeviceView>& weights, DType dtype, int device) {
            std::vector<ConstDeviceView> borrowed;
            borrowed.reserve(weights.size());
            for (auto view : weights) borrowed.push_back(view);
            return std::make_unique<GroupedLinear>(inputs, outputs, std::move(offsets), std::move(borrowed), dtype, device);
        }), py::arg("input_features"), py::arg("output_features"), py::arg("row_offsets"),
            py::arg("weights"), py::arg("dtype"), py::arg("device"), py::keep_alive<1, 5>())
        .def_property_readonly("row_count", &GroupedLinear::row_count)
        .def_property_readonly("expert_count", &GroupedLinear::expert_count)
        .def_property_readonly("input_features", &GroupedLinear::input_features)
        .def_property_readonly("output_features", &GroupedLinear::output_features)
        .def_property_readonly("device", &GroupedLinear::device)
        .def_property_readonly("dtype", &GroupedLinear::dtype)
        .def("forward", [](const GroupedLinear& op, DeviceView input, DeviceView output, const Stream& stream) {
            op.forward(input, output, stream);
        })
        .def("backward", [](const GroupedLinear& op, DeviceView input, DeviceView output_gradient,
                            DeviceView input_gradient, const std::vector<DeviceView>& weight_gradients,
                            GradientMode input_mode, GradientMode weight_mode, const Stream& stream) {
            op.backward(input, output_gradient, input_gradient, weight_gradients, input_mode, weight_mode, stream);
        });

}
