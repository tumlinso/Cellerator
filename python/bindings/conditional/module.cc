#include <Cellerator/execution/conditional/dispatch.hh>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <map>
#include <memory>

namespace py = pybind11;
namespace c = cellerator::execution::conditional;
namespace {
using Record = std::tuple<std::uint64_t, std::uint64_t, std::uint32_t,
                          std::uint64_t, std::uint64_t, double>;

template<class T>
std::pair<std::vector<T>, std::size_t> read(const py::array& array, std::size_t rows) {
    if (!array.dtype().is(py::dtype::of<T>())) throw py::type_error("expert output dtype must match its input dtype");
    if (!(array.flags() & py::array::c_style)) throw py::value_error("conditional arrays must be C-contiguous");
    if (array.ndim() != 1 && array.ndim() != 2) throw py::value_error("conditional arrays must have rank one or two");
    if (static_cast<std::size_t>(array.shape(0)) != rows) throw py::value_error("conditional array row count mismatch");
    const auto features = array.ndim() == 1 ? 1u : static_cast<std::size_t>(array.shape(1));
    std::vector<T> values(c::checked_size(rows, features));
    if (!values.empty()) std::memcpy(values.data(), array.data(), values.size() * sizeof(T));
    return {std::move(values), features};
}

template<class T>
py::array array(const T* values, std::size_t rows, std::size_t features, bool flat) {
    std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(rows)};
    if (!flat) shape.push_back(static_cast<py::ssize_t>(features));
    py::array_t<T> result(shape);
    const auto count = c::checked_size(rows, features);
    if (count) std::memcpy(result.mutable_data(), values, count * sizeof(T));
    return result;
}

enum class Operation { gather, combine, transpose, execute, execute_occurrences };

template<class T>
py::array run(const c::DispatchPlan& plan, const py::array& input, Operation operation,
              std::size_t output_features, const py::object& callback) {
    const auto rows = operation == Operation::combine ? plan.size() : plan.origin_count();
    auto [values, features] = read<T>(input, rows);
    std::vector<T> result;
    std::size_t result_rows;
    if (operation == Operation::combine || operation == Operation::transpose) {
        py::gil_scoped_release release;
        result = operation == Operation::combine
            ? c::combine<T, T, T>(plan, values, features)
            : c::transpose_combine<T, T, T>(plan, values, features);
        result_rows = operation == Operation::combine ? plan.origin_count() : plan.size();
    } else {
        std::map<c::Origin, std::size_t> origin_slots;
        for (std::size_t i = 0; i < plan.origin_count(); ++i) origin_slots.emplace(plan.origins()[i], i);
        const auto provider = [&](const c::Origin& origin, T* row, std::size_t width) {
            std::copy_n(values.begin() + origin_slots.at(origin) * width, width, row);
        };
        if (operation == Operation::gather) {
            py::gil_scoped_release release;
            result = c::gather_inputs<T>(plan, features, provider);
            result_rows = plan.size();
        } else {
            if (!PyCallable_Check(callback.ptr())) throw py::type_error("expert callback must be callable");
            const auto invoke = [&](const c::ExpertKey& expert, const c::InputBatch<T>& batch, c::OutputBatch<T>& output) {
                    auto inputs = array(batch.data, batch.rows, batch.features, false);
                    std::vector<c::Origin> origins(batch.origins, batch.origins + batch.rows);
                    std::vector<std::size_t> selections(batch.selection_indices, batch.selection_indices + batch.rows);
                    const auto returned = callback(expert, inputs, origins, selections);
                    if (!py::isinstance<py::array>(returned)) throw py::type_error("expert callback must return a NumPy array");
                    auto [outputs, width] = read<T>(py::cast<py::array>(returned), batch.rows);
                    if (width != output.features) throw py::value_error("expert output feature count mismatch");
                    std::copy(outputs.begin(), outputs.end(), output.data);
                };
            if (operation == Operation::execute_occurrences) {
                // Preserve every module output before weighting or reduction.
                // The execution-engine owner performs the sole combination.
                std::vector<T> outputs(c::checked_size(plan.size(), output_features));
                const auto gathered = c::gather_inputs<T>(plan, features, provider);
                for (std::size_t expert = 0; expert < plan.experts().size(); ++expert) {
                    const auto begin = plan.expert_offsets()[expert];
                    const auto count = plan.expert_offsets()[expert + 1] - begin;
                    const c::InputBatch<T> batch{gathered.data() + begin * features,
                        plan.dispatch_origins().data() + begin, count, features,
                        plan.dispatch_to_selection().data() + begin};
                    c::OutputBatch<T> output{outputs.data() + begin * output_features, count, output_features};
                    invoke(plan.experts()[expert], batch, output);
                }
                result.resize(outputs.size());
                for (std::size_t dispatch = 0; dispatch < plan.size(); ++dispatch)
                    std::copy_n(outputs.begin() + dispatch * output_features, output_features,
                        result.begin() + plan.dispatch_to_selection()[dispatch] * output_features);
                result_rows = plan.size();
            } else {
                result = c::execute<T, T, T, T>(plan, features, output_features, provider, invoke);
                result_rows = plan.origin_count();
            }
            features = output_features;
        }
    }
    // execute always returns a matrix: output width can differ from input width.
    return array(result.data(), result_rows, features,
                 operation != Operation::execute && operation != Operation::execute_occurrences && input.ndim() == 1);
}

py::array dispatch(const c::DispatchPlan& plan, const py::array& input, Operation operation,
                   std::size_t output_features = 1, const py::object& callback = py::none()) {
    if (!(input.flags() & py::array::c_style)) throw py::value_error("conditional arrays must be C-contiguous");
    if (input.dtype().is(py::dtype::of<float>())) return run<float>(plan, input, operation, output_features, callback);
    if (input.dtype().is(py::dtype::of<double>())) return run<double>(plan, input, operation, output_features, callback);
    if (input.dtype().is(py::dtype("float16"))) {
        // Explicit host interoperability: FP16 storage is exactly expanded to
        // FP32; callback, multiply, accumulation and output are FP32.
        auto promoted = py::cast<py::array>(input.attr("astype")(py::dtype::of<float>()));
        return run<float>(plan, promoted, operation, output_features, callback);
    }
    throw py::type_error("conditional arrays require float16, float32 or float64 storage");
}
} // namespace

PYBIND11_MODULE(_conditional_native, module) {
    module.doc() = "CPU expert-major sparse dispatch; logical module keys and origins survive batching.";
    module.attr("backend") = "cpu";
    module.attr("float16_accumulation") = "float32";
    module.attr("float16_output") = "float32";
    py::class_<c::ExpertKey>(module, "ExpertKey")
        .def(py::init([](std::uint64_t id, std::uint64_t revision) { return c::ExpertKey{id, revision}; }),
             py::arg("expert_id"), py::arg("revision") = 0)
        .def_readonly("expert_id", &c::ExpertKey::expert_id)
        .def_readonly("revision", &c::ExpertKey::revision)
        .def("__eq__", [](const c::ExpertKey& left, const c::ExpertKey& right) { return left == right; });
    py::class_<c::Origin>(module, "Origin")
        .def(py::init([](std::uint32_t rank, std::uint64_t instance, std::uint64_t row) { return c::Origin{rank, instance, row}; }),
             py::arg("rank"), py::arg("instance"), py::arg("row"))
        .def_readonly("rank", &c::Origin::rank)
        .def_readonly("instance", &c::Origin::instance)
        .def_readonly("row", &c::Origin::row)
        .def("__eq__", [](const c::Origin& left, const c::Origin& right) { return left == right; });
    py::class_<c::Selection>(module, "Selection")
        .def(py::init([](c::ExpertKey expert, c::Origin origin, double weight) { return c::Selection{expert, origin, weight}; }),
             py::arg("expert"), py::arg("origin"), py::arg("weight") = 1.0)
        .def_readonly("expert", &c::Selection::expert)
        .def_readonly("origin", &c::Selection::origin)
        .def_readonly("weight", &c::Selection::weight);
    py::class_<c::DispatchPlan>(module, "DispatchPlan")
        .def(py::init<std::vector<c::Selection>>(), py::arg("selections"))
        .def(py::init([](const std::vector<Record>& records) {
            std::vector<c::Selection> selected;
            selected.reserve(records.size());
            for (const auto& record : records)
                selected.push_back({{std::get<0>(record), std::get<1>(record)},
                                    {std::get<2>(record), std::get<3>(record), std::get<4>(record)}, std::get<5>(record)});
            return std::make_unique<c::DispatchPlan>(std::move(selected));
        }), py::arg("selections"))
        .def("__len__", &c::DispatchPlan::size)
        .def_property_readonly("origin_count", &c::DispatchPlan::origin_count)
        .def_property_readonly("selections", &c::DispatchPlan::selections)
        .def_property_readonly("experts", &c::DispatchPlan::experts)
        .def_property_readonly("expert_offsets", &c::DispatchPlan::expert_offsets)
        .def_property_readonly("dispatch_to_selection", &c::DispatchPlan::dispatch_to_selection)
        .def_property_readonly("selection_to_dispatch", &c::DispatchPlan::selection_to_dispatch)
        .def_property_readonly("origins", &c::DispatchPlan::origins)
        .def_property_readonly("dispatch_origins", &c::DispatchPlan::dispatch_origins)
        .def_property_readonly("dispatch_origin_slots", &c::DispatchPlan::dispatch_origin_slots)
        .def_property_readonly("dispatch_weights", &c::DispatchPlan::dispatch_weights)
        .def_property_readonly("selection_to_origin", &c::DispatchPlan::selection_to_origin)
        .def_property_readonly("origin_offsets", &c::DispatchPlan::origin_offsets)
        .def_property_readonly("origin_dispatch_indices", &c::DispatchPlan::origin_dispatch_indices);
    module.def("gather_inputs", [](const c::DispatchPlan& plan, const py::array& values) {
        return dispatch(plan, values, Operation::gather);
    }, py::arg("plan"), py::arg("origin_values"));
    module.def("combine", [](const c::DispatchPlan& plan, const py::array& values) {
        return dispatch(plan, values, Operation::combine);
    }, py::arg("plan"), py::arg("dispatch_values"));
    module.def("transpose_combine", [](const c::DispatchPlan& plan, const py::array& values) {
        return dispatch(plan, values, Operation::transpose);
    }, py::arg("plan"), py::arg("origin_gradients"));
    module.def("execute", [](const c::DispatchPlan& plan, const py::array& values,
                              std::size_t output_features, const py::object& callback) {
        return dispatch(plan, values, Operation::execute, output_features, callback);
    }, py::arg("plan"), py::arg("origin_values"), py::arg("output_features"), py::arg("callback"));
    module.def("execute_occurrences", [](const c::DispatchPlan& plan, const py::array& values,
                                          std::size_t output_features, const py::object& callback) {
        return dispatch(plan, values, Operation::execute_occurrences, output_features, callback);
    }, py::arg("plan"), py::arg("origin_values"), py::arg("output_features"), py::arg("callback"));
}
