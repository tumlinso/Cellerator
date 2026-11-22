#include <Cellerator/geometry/dependency/explicit_pattern.hh>
#include <Cellerator/geometry/dependency/pattern_owner.hh>
#include <Cellerator/execution/dependency/incidence.hh>
#include <Cellerator/execution/dependency/workset.hh>
#include <Cellerator/containers/value_shape.hh>
#include <Cellerator/containers/posting_index.hh>
#include <Cellerator/ops/incidence.hh>
#include <Cellerator/layout/entity_layout.hh>
#include <Cellerator/modeling/value_field.hh>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <tuple>

namespace py = pybind11;
namespace m = cellerator::modeling;
namespace l = cellerator::layout;
namespace g = cellerator::geometry::dependency;
namespace e = cellerator::execution;
namespace oi = cellerator::ops::incidence;
namespace {
using EncodedParticipant = std::pair<std::uint32_t, std::uint64_t>;
using EncodedEdge = std::pair<std::uint64_t, std::vector<EncodedParticipant>>;

std::vector<g::CompactParticipant> participants(const std::vector<EncodedParticipant>& encoded) {
    std::vector<g::CompactParticipant> result;
    result.reserve(encoded.size());
    for (const auto& item : encoded) result.push_back({item.first, 0, item.second});
    return result;
}

std::vector<cellerator::containers::PostingTerm> terms(const std::vector<EncodedParticipant>& encoded) {
    std::vector<cellerator::containers::PostingTerm> result;
    result.reserve(encoded.size());
    for (const auto& item : encoded) result.push_back({item.first, 0, item.second});
    return result;
}

std::vector<std::uint64_t> edge_ids(const g::ExplicitPattern& relation) {
    std::vector<std::uint64_t> result;
    result.reserve(relation.size());
    for (const auto& edge : relation.edges()) result.push_back(edge.logical_edge_id);
    return result;
}

oi::JointReduction reduction(const std::string& name) {
    if (name == "sum") return oi::JointReduction::sum;
    if (name == "mean") return oi::JointReduction::mean;
    if (name == "product") return oi::JointReduction::product;
    if (name == "min" || name == "minimum") return oi::JointReduction::minimum;
    if (name == "max" || name == "maximum") return oi::JointReduction::maximum;
    throw py::value_error("reduction must be sum, mean, product, min, or max");
}

template<class T>
std::pair<std::vector<T>, std::size_t> read_values(const py::array& input, std::size_t rows) {
    if (!input.dtype().is(py::dtype::of<T>())) throw py::type_error("unexpected numerical dtype");
    if (!(input.flags() & py::array::c_style)) throw py::value_error("values must be C-contiguous");
    if (input.ndim() != 1 && input.ndim() != 2) throw py::value_error("values must have one or two dimensions");
    if (static_cast<std::size_t>(input.shape(0)) != rows) throw py::value_error("value row count does not match plan");
    const auto features = input.ndim() == 1 ? 1u : static_cast<std::size_t>(input.shape(1));
    const auto count = cellerator::containers::checked_value_count(rows, features);
    std::vector<T> values(count);
    if (count) std::memcpy(values.data(), input.data(), count * sizeof(T));
    return {std::move(values), features};
}

template<class T>
py::array output_array(const std::vector<T>& values, std::size_t rows, std::size_t features, bool flat) {
    std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(rows)};
    if (!flat) shape.push_back(static_cast<py::ssize_t>(features));
    py::array_t<T> output(shape);
    if (!values.empty()) std::memcpy(output.mutable_data(), values.data(), values.size() * sizeof(T));
    return output;
}

enum class Operation { gather, reduce, scatter, reduce_scatter };

template<class T>
py::array execute(const e::PreparedIncidence& plan, const py::array& input, Operation operation,
                  oi::JointReduction reduce, std::optional<std::uint32_t> input_role,
                  std::optional<std::uint32_t> output_role) {
    auto [values, features] = read_values<T>(input, operation == Operation::scatter
                                                 ? plan.edge_count() : plan.layout().size());
    std::vector<T> result;
    std::size_t rows = 0;
    {
        py::gil_scoped_release release;
        switch (operation) {
        case Operation::gather:
            result = oi::gather(plan, values, features); rows = plan.incidence_count(); break;
        case Operation::reduce:
            result = oi::joint_reduce<T, T, T>(plan, values, features, reduce, input_role);
            rows = plan.edge_count(); break;
        case Operation::scatter:
            result = oi::scatter_add<T, T, T>(plan, values, features, output_role);
            rows = plan.layout().size(); break;
        case Operation::reduce_scatter:
            result = oi::joint_reduce_scatter<T, T, T>(plan, values, features, reduce, input_role, output_role);
            rows = plan.layout().size(); break;
        }
    }
    return output_array(result, rows, features, input.ndim() == 1);
}

py::array dispatch(const e::PreparedIncidence& plan, const py::array& input, Operation operation,
                   oi::JointReduction reduce = oi::JointReduction::sum,
                   std::optional<std::uint32_t> input_role = std::nullopt,
                   std::optional<std::uint32_t> output_role = std::nullopt) {
    if (input.dtype().is(py::dtype::of<float>()))
        return execute<float>(plan, input, operation, reduce, input_role, output_role);
    if (input.dtype().is(py::dtype::of<double>()))
        return execute<double>(plan, input, operation, reduce, input_role, output_role);
    throw py::type_error("native CPU numerical arrays require explicit float32 or float64 dtype");
}

template<class T>
void bind_field(py::module_& module, const char* name) {
    using Field = m::ValueField<T>;
    py::class_<Field>(module, name)
        .def(py::init([](l::EntityLayout layout, const py::array& input) {
            auto [values, features] = read_values<T>(input, layout.size());
            return Field(std::move(layout), features, std::move(values));
        }), py::arg("layout"), py::arg("values"))
        .def_property_readonly("generation", [](const Field& field) { return field.generation().value; })
        .def_property_readonly("features", &Field::features)
        .def_property_readonly("layout", &Field::layout, py::return_value_policy::reference_internal)
        .def_property_readonly("values", [](const Field& field) {
            return output_array(field.values(), field.layout().size(), field.features(), false);
        })
        .def("replace", [](Field& field, const py::array& input) {
            auto [values, features] = read_values<T>(input, field.layout().size());
            if (features != field.features()) throw py::value_error("field feature shape mismatch");
            field.replace(std::move(values));
        })
        .def("update", [](Field& field, const std::vector<std::uint64_t>& ids, const py::array& input) {
            auto [values, features] = read_values<T>(input, ids.size());
            if (features != field.features()) throw py::value_error("field feature shape mismatch");
            field.update(ids, values);
        }, py::arg("entity_ids"), py::arg("values"));
}
} // namespace

PYBIND11_MODULE(_modeling_native, module) {
    module.doc() = "Host sparse modeling primitives; no Torch or CUDA initialization.";
    module.attr("backend") = "cpu";
    module.attr("storage_dtypes") = py::make_tuple("float32", "float64");
    py::class_<g::ExplicitPattern, std::shared_ptr<g::ExplicitPattern>>(module, "SparseRelation")
        .def(py::init([](const std::vector<EncodedEdge>& encoded, std::uint64_t relation,
                         std::uint64_t structure, std::uint64_t epoch) {
            std::vector<g::PatternEdge> edges;
            for (const auto& edge : encoded) edges.push_back({edge.first, participants(edge.second)});
            return std::make_shared<g::ExplicitPattern>(std::move(edges), relation, structure, epoch);
        }), py::arg("edges"), py::arg("relation_id") = 1, py::arg("structure_id") = 1,
            py::arg("structure_epoch") = 1)
        .def("__len__", &g::ExplicitPattern::size)
        .def_property_readonly("relation_id", &g::ExplicitPattern::pattern_id)
        .def_property_readonly("structure_id", &g::ExplicitPattern::structure_id)
        .def_property_readonly("structure_epoch", &g::ExplicitPattern::structure_epoch)
        .def_property_readonly("edge_ids", &edge_ids)
        .def_property_readonly("edges", [](const g::ExplicitPattern& relation) {
            std::vector<EncodedEdge> result;
            for (const auto& edge : relation.edges()) {
                std::vector<EncodedParticipant> encoded;
                for (std::size_t i = 0; i < edge.participant_count; ++i) {
                    const auto& participant = relation.participants()[edge.participant_begin + i];
                    encoded.emplace_back(participant.role, participant.entity_id);
                }
                result.emplace_back(edge.logical_edge_id, std::move(encoded));
            }
            return result;
        })
        .def("query", [](const g::ExplicitPattern& relation, const std::vector<EncodedParticipant>& encoded, bool all) {
            std::vector<std::uint64_t> ids;
            for (auto ordinal : relation.query(terms(encoded), all)) ids.push_back(relation.edges()[ordinal].logical_edge_id);
            return ids;
        }, py::arg("terms"), py::arg("require_all") = false);
    py::class_<g::PatternOwner>(module, "DynamicRelation")
        .def(py::init<std::shared_ptr<const g::ExplicitPattern>>(), py::arg("initial"))
        .def("stage_add", [](g::PatternOwner& relation, std::uint64_t id,
                              const std::vector<EncodedParticipant>& encoded) { relation.stage_add(id, participants(encoded)); },
            py::arg("edge_id"), py::arg("participants"))
        .def("stage_remove", &g::PatternOwner::stage_remove, py::arg("edge_id"))
        .def("publish", &g::PatternOwner::publish, py::arg("next_epoch") = 0)
        .def_property_readonly("snapshot", &g::PatternOwner::snapshot)
        .def_property_readonly("pending_additions", &g::PatternOwner::pending_additions)
        .def_property_readonly("pending_removals", &g::PatternOwner::pending_removals);
    py::class_<e::ActiveWorklist>(module, "ActiveWorklist")
        .def(py::init<std::shared_ptr<const g::ExplicitPattern>, std::vector<std::size_t>, std::uint64_t>(),
             py::arg("relation"), py::arg("ordinals"), py::arg("activity_generation") = 0)
        .def("__len__", &e::ActiveWorklist::size)
        .def_property_readonly("edge_ids", &e::ActiveWorklist::edge_ids)
        .def_property_readonly("ordinals", &e::ActiveWorklist::ordinals)
        .def_property_readonly("relation", &e::ActiveWorklist::pattern)
        .def_property_readonly("activity_generation", &e::ActiveWorklist::activity_generation);
    py::class_<e::ActivityState>(module, "ActivityState")
        .def(py::init<std::shared_ptr<const g::ExplicitPattern>, bool>(), py::arg("relation"), py::arg("initially_active") = true)
        .def("set_active", &e::ActivityState::set_active, py::arg("edge_ids"), py::arg("active"))
        .def("worklist", &e::ActivityState::worklist)
        .def("rebase", &e::ActivityState::rebase, py::arg("replacement"), py::arg("new_edges_active") = true)
        .def("query", [](const e::ActivityState& state, const std::vector<EncodedParticipant>& encoded, bool all) {
            return state.query(terms(encoded), all);
        }, py::arg("terms"), py::arg("require_all") = false)
        .def_property_readonly("generation", &e::ActivityState::generation);
    py::class_<l::EntityLayout>(module, "EntityLayout")
        .def(py::init([](std::vector<std::uint64_t> ids, std::uint64_t domain, std::uint64_t order) {
            return l::EntityLayout(std::move(ids), {domain, 0}, {order, 0});
        }), py::arg("entity_ids"), py::arg("domain_id") = 1, py::arg("order_id") = 1)
        .def("__len__", &l::EntityLayout::size)
        .def("slot", &l::EntityLayout::slot)
        .def("permutation_to", &l::EntityLayout::permutation_to)
        .def_property_readonly("entity_ids", &l::EntityLayout::entity_ids)
        .def_property_readonly("domain_id", [](const l::EntityLayout& layout) { return layout.domain().low; })
        .def_property_readonly("order_id", [](const l::EntityLayout& layout) { return layout.order().low; });
    py::class_<e::PreparedIncidence>(module, "IncidencePlan")
        .def(py::init<e::ActiveWorklist, l::EntityLayout>(), py::arg("worklist"), py::arg("layout"))
        .def_property_readonly("edge_ids", [](const e::PreparedIncidence& plan) { return plan.worklist().edge_ids(); })
        .def_property_readonly("offsets", &e::PreparedIncidence::offsets)
        .def_property_readonly("slots", &e::PreparedIncidence::slots)
        .def_property_readonly("roles", &e::PreparedIncidence::roles)
        .def_property_readonly("edge_count", &e::PreparedIncidence::edge_count)
        .def_property_readonly("incidence_count", &e::PreparedIncidence::incidence_count)
        .def_property_readonly("structure_epoch", [](const e::PreparedIncidence& plan) { return plan.worklist().pattern()->structure_epoch(); })
        .def_property_readonly("activity_generation", [](const e::PreparedIncidence& plan) { return plan.worklist().activity_generation(); })
        .def_property_readonly("layout", &e::PreparedIncidence::layout, py::return_value_policy::reference_internal);
    module.def("gather", [](const e::PreparedIncidence& plan, const py::array& values) {
        return dispatch(plan, values, Operation::gather);
    }, py::arg("plan"), py::arg("values"));
    module.def("joint_reduce", [](const e::PreparedIncidence& plan, const py::array& values,
                                  const std::string& operation, std::optional<std::uint32_t> role) {
        return dispatch(plan, values, Operation::reduce, reduction(operation), role);
    }, py::arg("plan"), py::arg("values"), py::arg("reduction") = "sum", py::arg("role") = py::none());
    module.def("scatter_add", [](const e::PreparedIncidence& plan, const py::array& values,
                                 std::optional<std::uint32_t> role) {
        return dispatch(plan, values, Operation::scatter, oi::JointReduction::sum, std::nullopt, role);
    }, py::arg("plan"), py::arg("edge_values"), py::arg("role") = py::none());
    module.def("joint_reduce_scatter", [](const e::PreparedIncidence& plan, const py::array& values,
                                          const std::string& operation, std::optional<std::uint32_t> input_role,
                                          std::optional<std::uint32_t> output_role) {
        return dispatch(plan, values, Operation::reduce_scatter, reduction(operation), input_role, output_role);
    }, py::arg("plan"), py::arg("values"), py::arg("reduction") = "sum",
        py::arg("input_role") = py::none(), py::arg("output_role") = py::none());
    bind_field<float>(module, "Float32Field");
    bind_field<double>(module, "Float64Field");
}
