#include "indexing_bindings.hh"

#include <Cellerator/compute/gating/condition_program.hh>
#include <Cellerator/compute/gating/derivative_sampling.hh>
#include <Cellerator/compute/indexing/reverse_postings.hh>
#include <Cellerator/compute/indexing/temporal_occurrences.hh>
#include <Cellerator/compute/support/proposal_discovery.hh>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace ce = cellerator::compute;
namespace ce_support = cellerator::compute::support;
namespace ce_indexing = cellerator::compute::indexing;
namespace ce_gating = cellerator::compute::gating;
namespace ce_bindings = cellerator::bindings::sparse_support;

namespace {

template<class T>
py::array_t<T> require_vector(py::handle value, const char *name) {
    if (!py::isinstance<py::array>(value))
        throw py::type_error(std::string(name) + " must be a NumPy array");
    auto array = py::reinterpret_borrow<py::array>(value);
    if (!array.dtype().is(py::dtype::of<T>()) || array.ndim() != 1
        || !(array.flags() & py::array::c_style)
        || reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) != 0u)
        throw py::value_error(std::string(name) + " has an incompatible dtype or layout");
    return py::reinterpret_borrow<py::array_t<T>>(value);
}

template<class T>
py::array_t<T> vector_array(const std::vector<T> &values) {
    py::array_t<T> result(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}

template<class T>
std::vector<T> copy_values(const py::array_t<T> &array) {
    if (array.size() == 0) return {};
    return std::vector<T>(array.data(), array.data() + array.size());
}

std::int64_t signed_id(std::uint64_t id) {
    if (id > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("logical identity does not fit the Python int64 surface");
    return static_cast<std::int64_t>(id);
}

void require_adaptive(ce_support::adaptive_support_status_v1 status,
                      const char *operation) {
    if (status != ce_support::adaptive_support_status_v1::success)
        throw std::runtime_error(std::string("native support ") + operation
            + " failed (status " + std::to_string(static_cast<unsigned>(status)) + ")");
}

py::dict relation_dict(const ce_support::logical_support_view_v1 &view) {
    std::vector<std::int64_t> ids;
    std::vector<std::int64_t> offsets;
    std::vector<std::int32_t> roles;
    std::vector<std::int64_t> entities;
    ids.reserve(view.edge_count);
    offsets.reserve(view.edge_count + 1u);
    roles.reserve(view.participant_count);
    entities.reserve(view.participant_count);
    offsets.push_back(0);
    for (std::uint64_t i = 0u; i < view.edge_count; ++i) {
        const auto &edge = view.edges[i];
        ids.push_back(signed_id(edge.logical_edge_id));
        for (std::uint32_t j = 0u; j < edge.participant_count; ++j) {
            const auto &participant = view.participants[edge.participant_begin + j];
            if (participant.role > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
                throw std::overflow_error("participant role does not fit Python int32");
            roles.push_back(static_cast<std::int32_t>(participant.role));
            entities.push_back(signed_id(participant.entity_id));
        }
        offsets.push_back(static_cast<std::int64_t>(entities.size()));
    }
    py::dict result;
    result["relation_id"] = signed_id(view.relation_id);
    result["structure_id"] = signed_id(view.structure_id);
    result["structure_epoch"] = view.structure_epoch;
    result["edge_ids"] = vector_array(ids);
    result["participant_offsets"] = vector_array(offsets);
    result["participant_roles"] = vector_array(roles);
    result["participant_entity_ids"] = vector_array(entities);
    return result;
}

py::dict support_snapshot_dict(
        const ce_bindings::owned_support_snapshot_v1& snapshot) {
    py::dict result;
    result["structure_epoch"] = snapshot.structure_epoch;
    result["base"] = relation_dict(snapshot.base.view());
    result["delta"] = relation_dict(snapshot.delta.view());
    std::vector<std::int64_t> revoked_ids;
    revoked_ids.reserve(snapshot.revoked_edge_ids.size());
    for (const auto id : snapshot.revoked_edge_ids)
        revoked_ids.push_back(signed_id(id));
    result["revoked_edge_ids"] = vector_array(revoked_ids);
    std::vector<std::int64_t> evidence_ids;
    std::vector<std::uint64_t> evidence_rounds, evidence_generations;
    std::vector<float> evidence_scores;
    const auto& records = snapshot.promotion_records;
    evidence_ids.reserve(records.size());
    evidence_rounds.reserve(records.size());
    evidence_generations.reserve(records.size());
    evidence_scores.reserve(records.size());
    for (const auto& record : records) {
        evidence_ids.push_back(signed_id(record.logical_edge_id));
        evidence_rounds.push_back(record.discovery_round_id);
        evidence_generations.push_back(record.scorer_generation);
        evidence_scores.push_back(record.proposal_score);
    }
    result["promotion_evidence_edge_ids"] = vector_array(evidence_ids);
    result["promotion_evidence_round_ids"] = vector_array(evidence_rounds);
    result["promotion_evidence_scorer_generations"] = vector_array(evidence_generations);
    result["promotion_evidence_scores"] = vector_array(evidence_scores);
    return result;
}

class NativeReverseIndex final {
public:
    explicit NativeReverseIndex(std::shared_ptr<ce_bindings::native_support_handle_v1> handle)
        : handle_(std::move(handle)) {
        if (!handle_) throw py::value_error("support handle must not be null");
        auto owned = handle_->snapshot_copy();
        auto base_view = owned.base.view();
        auto delta_view = owned.delta.view();
        ce_support::support_snapshot_pair_v1 snapshot{
            &base_view, &delta_view, owned.structure_epoch};
        requirements_ = ce_indexing::query_reverse_posting_requirements_v1(snapshot);
        record_scratch_.resize(requirements_.record_capacity);
        entries_.resize(requirements_.maximum_unique_entries);
        const auto status = ce_indexing::build_reverse_postings_v1(snapshot,
            record_scratch_.data(), record_scratch_.size(), entries_.data(),
            entries_.size(), &index_);
        if (status != ce_indexing::posting_status_v1::success)
            throw std::runtime_error("native reverse posting preparation failed");
    }

    py::dict query(py::array roles_arg, py::array entities_arg, bool all,
                   std::uint64_t output_capacity) const {
        ensure_current();
        auto roles = require_vector<std::int32_t>(roles_arg, "roles");
        auto entities = require_vector<std::int64_t>(entities_arg, "entity_ids");
        if (roles.size() != entities.size())
            throw py::value_error("roles and entity_ids must have equal lengths");
        std::vector<ce_indexing::posting_term_v1> terms(
            static_cast<std::size_t>(roles.size()));
        for (py::ssize_t i = 0; i < roles.size(); ++i) {
            if (roles.data()[i] < 0 || entities.data()[i] <= 0)
                throw py::value_error("posting query roles and IDs must be nonnegative/positive");
            terms[static_cast<std::size_t>(i)] = {
                static_cast<std::uint32_t>(roles.data()[i]), 0u,
                static_cast<std::uint64_t>(entities.data()[i])};
        }
        std::uint64_t scratch_count = 0u;
        for (const auto &term : terms) {
            const auto range = posting_range(term);
            const auto count = range.second - range.first;
            if (scratch_count > std::numeric_limits<std::uint64_t>::max() - count)
                throw std::overflow_error("posting query workspace extent overflow");
            scratch_count += count;
        }
        std::vector<ce_indexing::indexed_work_item_v1> scratch(scratch_count);
        std::vector<ce_indexing::indexed_work_item_v1> output(output_capacity);
        ce_indexing::candidate_output_v1 output_view{
            output.empty() ? nullptr : output.data(), output_capacity, 0u, 0u};
        const auto status = ce_indexing::query_reverse_postings_v1(index_,
            terms.data(), terms.size(), all ? ce_indexing::posting_match_v1::all
                : ce_indexing::posting_match_v1::any,
            {scratch.empty() ? nullptr : scratch.data(), scratch.size()}, &output_view);
        if (status == ce_indexing::posting_status_v1::output_overflow)
            throw std::overflow_error("posting result needs "
                + std::to_string(output_view.required) + " work items");
        if (status != ce_indexing::posting_status_v1::success)
            throw py::value_error("native posting query rejected its input or workspace");
        output.resize(output_view.written);
        std::vector<std::int64_t> ids;
        std::vector<std::int64_t> ordinals;
        ids.reserve(output.size());
        ordinals.reserve(output.size());
        for (const auto &item : output) {
            ids.push_back(signed_id(item.logical_edge_id));
            if (item.logical_edge_ordinal > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()))
                throw std::overflow_error("logical edge ordinal does not fit Python int64");
            ordinals.push_back(static_cast<std::int64_t>(item.logical_edge_ordinal));
        }
        py::dict result;
        result["logical_edge_ids"] = vector_array(ids);
        result["edge_ordinals"] = vector_array(ordinals);
        result["structure_epoch"] = index_.structure_epoch;
        result["required"] = output_view.required;
        return result;
    }

    std::uint64_t structure_epoch() const noexcept { return index_.structure_epoch; }

private:
    std::pair<std::uint64_t, std::uint64_t> posting_range(
        const ce_indexing::posting_term_v1 &term) const noexcept {
        std::uint64_t first = 0u, last = index_.entry_count;
        while (first < last) {
            const auto middle = first + (last - first) / 2u;
            const auto &entry = entries_[middle];
            if (entry.role < term.role
                || (entry.role == term.role && entry.entity_id < term.entity_id))
                first = middle + 1u;
            else last = middle;
        }
        if (first == index_.entry_count || entries_[first].role != term.role
            || entries_[first].entity_id != term.entity_id)
            return {0u, 0u};
        auto end = first;
        while (end < index_.entry_count && entries_[end].role == term.role
            && entries_[end].entity_id == term.entity_id) ++end;
        return {first, end};
    }

    void ensure_current() const {
        if (handle_->structure_epoch() != index_.structure_epoch)
            throw py::value_error("reverse posting index is stale after support publication");
    }

    std::shared_ptr<ce_bindings::native_support_handle_v1> handle_;
    ce_indexing::reverse_posting_requirements_v1 requirements_{};
    std::vector<ce_indexing::reverse_posting_record_v1> record_scratch_;
    std::vector<ce_indexing::reverse_posting_record_v1> entries_;
    ce_indexing::reverse_posting_view_v1 index_{};
};

class NativeTemporalOccurrenceRing final {
public:
    NativeTemporalOccurrenceRing(std::uint64_t cell_id, std::uint64_t entity_id,
                                 std::uint64_t capacity)
        : storage_(capacity) {
        const auto status = ce_indexing::initialize_temporal_event_ring_v1(
            &ring_, cell_id, entity_id,
            storage_.empty() ? nullptr : storage_.data(), storage_.size());
        if (status != ce_indexing::occurrence_status_v1::success)
            throw py::value_error("invalid temporal ring identity or capacity");
    }

    std::uint64_t append(std::uint64_t occurrence_id, std::uint64_t ordinal,
                         py::object timestamp) {
        const auto signed_max = static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max());
        if (occurrence_id > signed_max || ordinal > signed_max)
            throw std::overflow_error(
                "temporal occurrence identity and ordinal must fit Python int64");
        ce_indexing::temporal_occurrence_v1 event{};
        event.cell_id = ring_.cell_id;
        event.entity_id = ring_.entity_id;
        event.occurrence_id = occurrence_id;
        event.ordinal = ordinal;
        if (!timestamp.is_none()) {
            event.timestamp = timestamp.cast<std::int64_t>();
            event.flags = ce_indexing::occurrence_has_timestamp_v1;
        }
        std::uint64_t evicted = 0u;
        const auto status = ce_indexing::append_temporal_occurrence_v1(
            &ring_, event, ring_.generation, ring_.generation + 1u, &evicted);
        if (status == ce_indexing::occurrence_status_v1::ring_capacity_exhausted)
            throw std::overflow_error("temporal ring capacity is zero");
        if (status != ce_indexing::occurrence_status_v1::success)
            throw py::value_error("temporal occurrence append rejected its identity or ordinal");
        return evicted;
    }

    py::dict query(py::object min_ordinal, py::object max_ordinal,
                   py::object min_timestamp, py::object max_timestamp) const {
        ce_indexing::occurrence_window_v1 window{};
        window.use_ordinal_window = !min_ordinal.is_none() || !max_ordinal.is_none();
        if (window.use_ordinal_window) {
            window.minimum_ordinal = min_ordinal.is_none() ? 0u : min_ordinal.cast<std::uint64_t>();
            window.maximum_ordinal = max_ordinal.is_none()
                ? std::numeric_limits<std::uint64_t>::max()
                : max_ordinal.cast<std::uint64_t>();
        }
        window.use_timestamp_window = !min_timestamp.is_none() || !max_timestamp.is_none();
        if (window.use_timestamp_window) {
            window.minimum_timestamp = min_timestamp.is_none()
                ? std::numeric_limits<std::int64_t>::min()
                : min_timestamp.cast<std::int64_t>();
            window.maximum_timestamp = max_timestamp.is_none()
                ? std::numeric_limits<std::int64_t>::max()
                : max_timestamp.cast<std::int64_t>();
        }
        std::vector<ce_indexing::temporal_occurrence_v1> output(ring_.size);
        ce_indexing::occurrence_query_result_v1 result{};
        const auto status = ce_indexing::query_temporal_occurrences_v1(ring_, window,
            output.empty() ? nullptr : output.data(), output.size(), &result);
        if (status != ce_indexing::occurrence_status_v1::success)
            throw py::value_error("temporal occurrence query rejected its window");
        std::vector<std::int64_t> occurrence_ids, ordinals, timestamps;
        std::vector<std::uint32_t> flags;
        occurrence_ids.reserve(result.written);
        ordinals.reserve(result.written);
        timestamps.reserve(result.written);
        flags.reserve(result.written);
        for (std::uint64_t i = 0u; i < result.written; ++i) {
            occurrence_ids.push_back(signed_id(output[i].occurrence_id));
            ordinals.push_back(static_cast<std::int64_t>(output[i].ordinal));
            timestamps.push_back(output[i].timestamp);
            flags.push_back(output[i].flags);
        }
        py::dict value;
        value["occurrence_ids"] = vector_array(occurrence_ids);
        value["ordinals"] = vector_array(ordinals);
        value["timestamps"] = vector_array(timestamps);
        value["flags"] = vector_array(flags);
        value["history_truncated"] = result.history_truncated;
        return value;
    }

private:
    std::vector<ce_indexing::temporal_occurrence_v1> storage_;
    ce_indexing::temporal_event_ring_v1 ring_{};
};

} // namespace

namespace {

py::dict build_native_condition_worksets(
    std::uint64_t structure_epoch, std::uint64_t edge_count,
    std::uint64_t state_generation,
    py::array opcodes_arg, py::array expected_arg,
    py::array first_child_arg, py::array child_count_arg,
    py::array predicate_index_arg, py::array binding_ordinals_arg,
    py::array binding_roots_arg, py::array predicates_arg,
    py::array primal_offsets_arg, py::array primal_ids_arg,
    py::array primal_ordinals_arg, py::array derivative_offsets_arg,
    py::array derivative_ids_arg, py::array derivative_ordinals_arg,
    std::uint64_t primal_capacity, std::uint64_t derivative_capacity) {
    auto opcodes = require_vector<std::uint8_t>(opcodes_arg, "condition_opcodes");
    auto expected = require_vector<std::uint8_t>(expected_arg, "condition_expected");
    auto first_child = require_vector<std::uint32_t>(first_child_arg, "condition_first_child");
    auto child_count = require_vector<std::uint32_t>(child_count_arg, "condition_child_count");
    auto predicate_index = require_vector<std::uint32_t>(predicate_index_arg,
                                                         "condition_predicate_index");
    auto binding_ordinals = require_vector<std::uint64_t>(binding_ordinals_arg,
                                                          "binding_edge_ordinals");
    auto binding_roots = require_vector<std::uint32_t>(binding_roots_arg,
                                                       "binding_root_nodes");
    auto predicates = py::reinterpret_borrow<py::array>(predicates_arg);
    if (!predicates.dtype().is(py::dtype::of<std::uint8_t>())
        || predicates.ndim() != 2 || !(predicates.flags() & py::array::c_style))
        throw py::value_error("predicate_values must be a contiguous rank-2 uint8 matrix");
    const auto cells = static_cast<std::uint64_t>(predicates.shape(0));
    const auto predicate_count = static_cast<std::uint64_t>(predicates.shape(1));
    if (predicate_count > std::numeric_limits<std::uint32_t>::max()
        || (predicate_count != 0u
            && cells > std::numeric_limits<std::uint64_t>::max() / predicate_count))
        throw std::overflow_error("predicate matrix extent is unsupported");
    const auto node_count = static_cast<std::uint64_t>(opcodes.size());
    if (expected.size() != opcodes.size() || first_child.size() != opcodes.size()
        || child_count.size() != opcodes.size() || predicate_index.size() != opcodes.size()
        || binding_ordinals.size() != binding_roots.size())
        throw py::value_error("condition node/binding arrays have inconsistent lengths");
    std::vector<ce_gating::condition_node_v1> nodes(node_count);
    for (std::uint64_t i = 0u; i < node_count; ++i) {
        nodes[i] = {static_cast<ce_gating::condition_opcode_v1>(opcodes.data()[i]),
            expected.data()[i], 0u, first_child.data()[i], child_count.data()[i],
            predicate_index.data()[i], 0u};
    }
    std::vector<ce_gating::edge_condition_binding_v1> bindings(binding_ordinals.size());
    for (py::ssize_t i = 0; i < binding_ordinals.size(); ++i)
        bindings[static_cast<std::size_t>(i)] = {
            binding_ordinals.data()[i], binding_roots.data()[i], 0u};
    const ce_gating::condition_program_view_v1 program{structure_epoch,
        nodes.empty() ? nullptr : nodes.data(), node_count,
        bindings.empty() ? nullptr : bindings.data(), bindings.size(),
        edge_count, static_cast<std::uint32_t>(predicate_count), 0u};
    auto predicate_values = reinterpret_cast<const std::uint8_t *>(predicates.data());
    const auto words_per_cell = edge_count / 64u + (edge_count % 64u != 0u);
    if (words_per_cell != 0u
        && cells > std::numeric_limits<std::uint64_t>::max() / words_per_cell)
        throw std::overflow_error("condition bitmap extent overflow");
    std::vector<std::uint64_t> bitmaps(words_per_cell * cells);
    std::vector<std::uint8_t> node_scratch(node_count);
    const auto bitmap_status = ce_gating::evaluate_condition_bitmaps_v1(program,
        cells, predicate_values, cells * predicate_count,
        bitmaps.empty() ? nullptr : bitmaps.data(), bitmaps.size(),
        node_scratch.empty() ? nullptr : node_scratch.data(), node_scratch.size());
    if (bitmap_status != ce_gating::condition_status_v1::success)
        throw py::value_error("native condition program rejected its inputs");

    auto primal_offsets_array = require_vector<std::int64_t>(primal_offsets_arg,
                                                              "primal_offsets");
    auto primal_ids_array = require_vector<std::int64_t>(primal_ids_arg, "primal_edge_ids");
    auto primal_ordinals_array = require_vector<std::int64_t>(primal_ordinals_arg,
                                                              "primal_edge_ordinals");
    auto derivative_offsets_array = require_vector<std::int64_t>(derivative_offsets_arg,
                                                                  "derivative_offsets");
    auto derivative_ids_array = require_vector<std::int64_t>(derivative_ids_arg,
                                                              "derivative_edge_ids");
    auto derivative_ordinals_array = require_vector<std::int64_t>(derivative_ordinals_arg,
                                                                  "derivative_edge_ordinals");
    auto make_candidates = [cells](const py::array_t<std::int64_t> &offsets,
                                   const py::array_t<std::int64_t> &ids,
                                   const py::array_t<std::int64_t> &ordinals,
                                   std::vector<std::uint64_t> *owned_offsets,
                                   std::vector<ce_indexing::indexed_work_item_v1> *owned_items) {
        if (offsets.size() != static_cast<py::ssize_t>(cells + 1u)
            || ids.size() != ordinals.size())
            throw py::value_error("candidate CSR arrays have inconsistent lengths");
        owned_offsets->resize(static_cast<std::size_t>(offsets.size()));
        owned_items->resize(static_cast<std::size_t>(ids.size()));
        for (py::ssize_t i = 0; i < offsets.size(); ++i) {
            if (offsets.data()[i] < 0)
                throw py::value_error("candidate offsets must be nonnegative");
            (*owned_offsets)[static_cast<std::size_t>(i)] =
                static_cast<std::uint64_t>(offsets.data()[i]);
        }
        for (py::ssize_t i = 0; i < ids.size(); ++i) {
            if (ids.data()[i] <= 0 || ordinals.data()[i] < 0)
                throw py::value_error("candidate IDs and ordinals are invalid");
            (*owned_items)[static_cast<std::size_t>(i)] = {
                static_cast<std::uint64_t>(ids.data()[i]),
                static_cast<std::uint64_t>(ordinals.data()[i])};
        }
        return ce_gating::per_cell_candidate_view_v1{
            owned_offsets->data(), owned_items->data(), owned_items->size()};
    };
    std::vector<std::uint64_t> p_offsets, d_offsets;
    std::vector<ce_indexing::indexed_work_item_v1> p_items, d_items;
    const auto p_view = make_candidates(primal_offsets_array, primal_ids_array,
        primal_ordinals_array, &p_offsets, &p_items);
    const auto d_view = make_candidates(derivative_offsets_array,
        derivative_ids_array, derivative_ordinals_array, &d_offsets, &d_items);
    std::vector<std::uint64_t> p_out_offsets(cells + 1u), d_out_offsets(cells + 1u);
    std::vector<ce_indexing::indexed_work_item_v1> p_out(primal_capacity),
        d_out(derivative_capacity);
    ce_gating::gate_workset_pair_v1 output{
        {p_out_offsets.data(), p_out_offsets.size(),
         p_out.empty() ? nullptr : p_out.data(), p_out.size(), 0u, 0u},
        {d_out_offsets.data(), d_out_offsets.size(),
         d_out.empty() ? nullptr : d_out.data(), d_out.size(), 0u, 0u}};
    const auto workset_status = ce_gating::build_condition_worksets_v1(program,
        cells, state_generation, bitmaps.empty() ? nullptr : bitmaps.data(), bitmaps.size(),
        p_view, d_view, &output);
    if (workset_status == ce_gating::condition_status_v1::primal_workset_overflow)
        throw std::overflow_error("primal candidate capacity requires "
            + std::to_string(output.primal.required) + " items");
    if (workset_status == ce_gating::condition_status_v1::derivative_workset_overflow)
        throw std::overflow_error("derivative candidate capacity requires "
            + std::to_string(output.derivative.required) + " items");
    if (workset_status != ce_gating::condition_status_v1::success)
        throw py::value_error("native gate workset builder rejected its inputs");
    p_out.resize(output.primal.written);
    d_out.resize(output.derivative.written);
    auto extract_ids = [](const auto &items) {
        std::vector<std::int64_t> ids;
        ids.reserve(items.size());
        for (const auto &item : items) ids.push_back(signed_id(item.logical_edge_id));
        return ids;
    };
    auto extract_ordinals = [](const auto &items) {
        std::vector<std::int64_t> ordinals;
        ordinals.reserve(items.size());
        for (const auto &item : items) {
            if (item.logical_edge_ordinal > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()))
                throw std::overflow_error("logical edge ordinal does not fit Python int64");
            ordinals.push_back(static_cast<std::int64_t>(item.logical_edge_ordinal));
        }
        return ordinals;
    };
    auto signed_offsets = [](const std::vector<std::uint64_t> &offsets) {
        std::vector<std::int64_t> values;
        values.reserve(offsets.size());
        for (const auto offset : offsets) {
            if (offset > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()))
                throw std::overflow_error("workset offset does not fit Python int64");
            values.push_back(static_cast<std::int64_t>(offset));
        }
        return values;
    };
    py::dict result;
    result["condition_bitmap_words"] = vector_array(bitmaps);
    result["words_per_cell"] = words_per_cell;
    result["primal_offsets"] = vector_array(signed_offsets(p_out_offsets));
    result["primal_edge_ids"] = vector_array(extract_ids(p_out));
    result["primal_edge_ordinals"] = vector_array(extract_ordinals(p_out));
    result["derivative_offsets"] = vector_array(signed_offsets(d_out_offsets));
    result["derivative_edge_ids"] = vector_array(extract_ids(d_out));
    result["derivative_edge_ordinals"] = vector_array(extract_ordinals(d_out));
    result["structure_epoch"] = structure_epoch;
    result["state_generation"] = state_generation;
    return result;
}

py::dict build_native_derivative_sample(
        py::array admitted_offsets_arg, py::array admitted_ids_arg,
        py::array admitted_ordinals_arg, py::array dormant_offsets_arg,
        py::array dormant_ids_arg, py::array dormant_ordinals_arg,
        py::array budgets_arg, std::uint64_t window_counter,
        std::uint64_t seed, std::uint64_t structure_epoch,
        std::uint64_t state_generation) {
    auto admitted_offsets_array = require_vector<std::int64_t>(
        admitted_offsets_arg, "admitted_offsets");
    auto admitted_ids_array = require_vector<std::int64_t>(
        admitted_ids_arg, "admitted_ids");
    auto admitted_ordinals_array = require_vector<std::int64_t>(
        admitted_ordinals_arg, "admitted_ordinals");
    auto dormant_offsets_array = require_vector<std::int64_t>(
        dormant_offsets_arg, "dormant_offsets");
    auto dormant_ids_array = require_vector<std::int64_t>(
        dormant_ids_arg, "dormant_ids");
    auto dormant_ordinals_array = require_vector<std::int64_t>(
        dormant_ordinals_arg, "dormant_ordinals");
    auto budgets_array = require_vector<std::int64_t>(budgets_arg,
                                                       "per_cell_budgets");
    if (admitted_ids_array.size() != admitted_ordinals_array.size()
        || dormant_ids_array.size() != dormant_ordinals_array.size()
        || admitted_offsets_array.size() < 2
        || dormant_offsets_array.size() != admitted_offsets_array.size()
        || budgets_array.size() + 1 != admitted_offsets_array.size())
        throw py::value_error("candidate CSR arrays must have matching cell and item counts");

    const auto cell_count = static_cast<std::uint64_t>(budgets_array.size());
    auto copy_offsets = [](const py::array_t<std::int64_t>& input,
                           const char* name) {
        std::vector<std::uint64_t> result(static_cast<std::size_t>(input.size()));
        for (py::ssize_t i = 0; i < input.size(); ++i) {
            if (input.data()[i] < 0)
                throw py::value_error(std::string(name) + " must be nonnegative");
            result[static_cast<std::size_t>(i)] =
                static_cast<std::uint64_t>(input.data()[i]);
        }
        return result;
    };
    auto copy_items = [](const py::array_t<std::int64_t>& ids,
                         const py::array_t<std::int64_t>& ordinals,
                         const char* name) {
        std::vector<ce_indexing::indexed_work_item_v1> result(
            static_cast<std::size_t>(ids.size()));
        for (py::ssize_t i = 0; i < ids.size(); ++i) {
            if (ids.data()[i] < 0 || ordinals.data()[i] < 0)
                throw py::value_error(std::string(name) + " IDs and ordinals must be nonnegative");
            result[static_cast<std::size_t>(i)] = {
                static_cast<std::uint64_t>(ids.data()[i]),
                static_cast<std::uint64_t>(ordinals.data()[i])};
        }
        return result;
    };
    auto admitted_offsets = copy_offsets(admitted_offsets_array, "admitted_offsets");
    auto dormant_offsets = copy_offsets(dormant_offsets_array, "dormant_offsets");
    auto admitted_items = copy_items(admitted_ids_array,
        admitted_ordinals_array, "admitted");
    auto dormant_items = copy_items(dormant_ids_array,
        dormant_ordinals_array, "dormant");
    std::vector<std::uint64_t> budgets(static_cast<std::size_t>(cell_count));
    for (py::ssize_t i = 0; i < budgets_array.size(); ++i) {
        if (budgets_array.data()[i] < 0)
            throw py::value_error("per_cell_budgets must be nonnegative");
        budgets[static_cast<std::size_t>(i)] =
            static_cast<std::uint64_t>(budgets_array.data()[i]);
    }
    if (dormant_items.size() > std::numeric_limits<std::size_t>::max()
            - admitted_items.size())
        throw std::overflow_error("derivative output capacity overflows host size");
    const auto capacity = admitted_items.size() + dormant_items.size();
    std::vector<std::uint64_t> output_offsets(
        static_cast<std::size_t>(cell_count) + 1u);
    std::vector<ce_indexing::indexed_work_item_v1> output_items(capacity);
    std::vector<std::uint64_t> eligible(static_cast<std::size_t>(cell_count));
    std::vector<std::uint64_t> selected(static_cast<std::size_t>(cell_count));
    std::vector<std::uint64_t> remaining(static_cast<std::size_t>(cell_count));
    ce_gating::derivative_sample_request_v1 request{};
    request.structure_epoch = structure_epoch;
    request.state_generation = state_generation;
    request.window_counter = window_counter;
    request.seed = seed;
    request.cell_count = cell_count;
    request.admitted = {admitted_offsets.data(),
        admitted_items.empty() ? nullptr : admitted_items.data(), cell_count,
        admitted_items.size()};
    request.dormant = {dormant_offsets.data(),
        dormant_items.empty() ? nullptr : dormant_items.data(), cell_count,
        dormant_items.size()};
    request.dormant_budget_by_cell = budgets.data();
    ce_gating::derivative_sample_output_v1 output{};
    output.cell_offsets = output_offsets.data();
    output.cell_offsets_capacity = output_offsets.size();
    output.items = output_items.empty() ? nullptr : output_items.data();
    output.item_capacity = output_items.size();
    output.eligible_dormant_by_cell = eligible.data();
    output.selected_dormant_by_cell = selected.data();
    output.remaining_dormant_by_cell = remaining.data();
    output.cell_statistics_capacity = cell_count;
    const auto status = ce_gating::build_derivative_sample_v1(request, &output);
    if (status != ce_gating::derivative_sample_status_v1::success)
        throw py::value_error("native derivative candidate sampling failed (status "
            + std::to_string(static_cast<unsigned>(status)) + ")");
    auto signed_values = [](const std::uint64_t* values, std::uint64_t count,
                            const char* name) {
        std::vector<std::int64_t> result(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0u; i < count; ++i) {
            if (values[i] > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()))
                throw std::overflow_error(std::string(name)
                    + " does not fit the Python int64 surface");
            result[static_cast<std::size_t>(i)] =
                static_cast<std::int64_t>(values[i]);
        }
        return result;
    };
    std::vector<std::int64_t> item_ids, item_ordinals;
    item_ids.reserve(static_cast<std::size_t>(output.written));
    item_ordinals.reserve(static_cast<std::size_t>(output.written));
    for (std::uint64_t i = 0u; i < output.written; ++i) {
        item_ids.push_back(signed_id(output_items[i].logical_edge_id));
        if (output_items[i].logical_edge_ordinal > static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()))
            throw std::overflow_error("edge ordinal does not fit the Python int64 surface");
        item_ordinals.push_back(static_cast<std::int64_t>(
            output_items[i].logical_edge_ordinal));
    }
    py::dict result;
    result["derivative_offsets"] = vector_array(
        signed_values(output_offsets.data(), output_offsets.size(), "offset"));
    result["derivative_edge_ids"] = vector_array(item_ids);
    result["derivative_edge_ordinals"] = vector_array(item_ordinals);
    result["eligible_dormant_counts"] = vector_array(
        signed_values(eligible.data(), eligible.size(), "eligible count"));
    result["selected_dormant_counts"] = vector_array(
        signed_values(selected.data(), selected.size(), "selected count"));
    result["remaining_dormant_counts"] = vector_array(
        signed_values(remaining.data(), remaining.size(), "remaining count"));
    result["eligible_dormant_count"] = output.eligible_dormant_count;
    result["selected_dormant_count"] = output.selected_dormant_count;
    result["remaining_dormant_count"] = output.remaining_dormant_count;
    result["structure_epoch"] = output.structure_epoch;
    result["state_generation"] = output.state_generation;
    result["window_counter"] = window_counter;
    result["seed"] = seed;
    result["sampling_policy"] = "biased_cyclic_v1";
    return result;
}

py::dict propose_native_support(
    py::array entity_ids_arg, py::array mechanism_ids_arg,
    py::array posting_seeds_arg, py::array coactivations_arg,
    std::uint32_t participant_count, std::uint64_t seed,
    std::uint64_t proposal_budget, std::uint64_t attempt_budget,
    py::object known_mechanisms_arg, py::object known_participants_arg) {
    if (participant_count < 1u || participant_count > 3u)
        throw py::value_error("participant_count must be between one and three");
    auto entity_ids = require_vector<std::int64_t>(entity_ids_arg, "entity_ids");
    auto mechanism_ids = require_vector<std::int64_t>(mechanism_ids_arg, "mechanism_ids");
    auto posting_seeds = require_vector<std::int64_t>(posting_seeds_arg, "posting_seed_entity_ids");
    auto coactivations = py::reinterpret_borrow<py::array>(coactivations_arg);
    if (!coactivations.dtype().is(py::dtype::of<std::int64_t>())
        || coactivations.ndim() != 2 || coactivations.shape(1) != 2
        || !(coactivations.flags() & py::array::c_style))
        throw py::value_error("recent_coactivations must be a contiguous [N,2] int64 matrix");
    std::vector<std::uint64_t> entities, mechanisms, posting_ids;
    entities.reserve(entity_ids.size());
    mechanisms.reserve(mechanism_ids.size());
    posting_ids.reserve(posting_seeds.size());
    for (py::ssize_t i = 0; i < entity_ids.size(); ++i) {
        const auto id = entity_ids.data()[i];
        if (id <= 0) throw py::value_error("entity IDs must be positive");
        entities.push_back(static_cast<std::uint64_t>(id));
    }
    for (py::ssize_t i = 0; i < mechanism_ids.size(); ++i) {
        const auto id = mechanism_ids.data()[i];
        if (id <= 0) throw py::value_error("mechanism IDs must be positive");
        mechanisms.push_back(static_cast<std::uint64_t>(id));
    }
    for (py::ssize_t i = 0; i < posting_seeds.size(); ++i) {
        const auto id = posting_seeds.data()[i];
        if (id <= 0) throw py::value_error("posting seed IDs must be positive");
        posting_ids.push_back(static_cast<std::uint64_t>(id));
    }
    std::vector<ce_support::coactivation_pair_v1> pairs(
        static_cast<std::size_t>(coactivations.shape(0)));
    const auto *pair_data = static_cast<const std::int64_t *>(coactivations.data());
    for (std::size_t i = 0u; i < pairs.size(); ++i) {
        if (pair_data[2u * i] <= 0 || pair_data[2u * i + 1u] <= 0)
            throw py::value_error("coactivation IDs must be positive");
        pairs[i] = {static_cast<std::uint64_t>(pair_data[2u * i]),
                    static_cast<std::uint64_t>(pair_data[2u * i + 1u])};
    }
    std::vector<std::int64_t> known_mechanism_ids;
    std::vector<std::int64_t> known_participant_ids;
    if (!known_mechanisms_arg.is_none() || !known_participants_arg.is_none()) {
        if (known_mechanisms_arg.is_none() || known_participants_arg.is_none())
            throw py::value_error("known mechanism and participant arrays must be supplied together");
        auto known_mechanisms = require_vector<std::int64_t>(known_mechanisms_arg,
                                                              "known_mechanism_ids");
        auto known_participants = py::reinterpret_borrow<py::array>(known_participants_arg);
        if (!known_participants.dtype().is(py::dtype::of<std::int64_t>())
            || known_participants.ndim() != 2 || known_participants.shape(0) != known_mechanisms.size()
            || known_participants.shape(1) != 3
            || !(known_participants.flags() & py::array::c_style))
            throw py::value_error("known_participant_ids must be an int64 [known,3] matrix");
        known_mechanism_ids = copy_values(known_mechanisms);
        if (known_participants.shape(0) != 0) {
            const auto *known_data = static_cast<const std::int64_t *>(known_participants.data());
            known_participant_ids.assign(known_data,
                known_data + known_participants.shape(0) * 3);
        }
    }
    std::vector<ce_support::support_proposal_v1> known_routes(
        known_mechanism_ids.size());
    for (std::size_t i = 0u; i < known_routes.size(); ++i) {
        if (known_mechanism_ids[i] <= 0)
            throw py::value_error("known mechanism IDs must be positive");
        auto &route = known_routes[i];
        route.mechanism_id = static_cast<std::uint64_t>(known_mechanism_ids[i]);
        route.participant_count = participant_count;
        for (std::uint32_t slot = 0u; slot < participant_count; ++slot) {
            const auto entity = known_participant_ids[i * 3u + slot];
            if (entity <= 0) throw py::value_error("known participant IDs must be positive");
            route.participant_ids[slot] = static_cast<std::uint64_t>(entity);
        }
    }
    const ce_support::proposal_discovery_input_v1 input{
        entities.data(), entities.size(), mechanisms.data(), mechanisms.size(),
        known_routes.data(), known_routes.size(),
        posting_ids.data(), posting_ids.size(),
        pairs.data(), pairs.size(), participant_count, 0u};
    std::vector<ce_support::support_proposal_v1> proposals(proposal_budget);
    ce_support::proposal_discovery_result_v1 discovery{};
    const auto status = ce_support::propose_support_candidates_v1(input, seed,
        proposal_budget, attempt_budget,
        proposals.empty() ? nullptr : proposals.data(), proposals.size(), &discovery);
    if (status != ce_support::proposal_status_v1::success)
        throw py::value_error("native bounded proposal generator rejected its inputs");
    proposals.resize(discovery.written);
    std::vector<std::int64_t> proposed_mechanisms;
    std::vector<std::int64_t> participants;
    proposed_mechanisms.reserve(proposals.size());
    participants.reserve(proposals.size() * 3u);
    for (const auto &proposal : proposals) {
        proposed_mechanisms.push_back(signed_id(proposal.mechanism_id));
        for (std::uint32_t i = 0u; i < 3u; ++i)
            participants.push_back(i < proposal.participant_count
                ? signed_id(proposal.participant_ids[i]) : -1);
    }
    py::dict result;
    result["mechanism_ids"] = vector_array(proposed_mechanisms);
    py::array_t<std::int64_t> participant_matrix(
        {static_cast<py::ssize_t>(proposals.size()), py::ssize_t{3}});
    if (!participants.empty())
        std::copy(participants.begin(), participants.end(), participant_matrix.mutable_data());
    result["participant_ids"] = std::move(participant_matrix);
    result["participant_count"] = participant_count;
    result["attempts"] = discovery.attempts;
    result["candidate_space_upper_bound"] = discovery.candidate_space_upper_bound;
    result["candidate_space_upper_bound_saturated"] =
        discovery.candidate_space_upper_bound_saturated;
    result["missed_candidate_lower_bound"] = discovery.missed_candidate_lower_bound;
    result["missed_candidate_count_exact"] = discovery.missed_candidate_count_exact;
    result["sampled_coverage_lower_bound"] = discovery.sampled_coverage_lower_bound;
    result["next_seed"] = discovery.next_seed;
    return result;
}

} // namespace

ce_bindings::native_support_handle_v1::native_support_handle_v1(
    const ce_support::logical_support_view_v1 &base,
    std::uint64_t delta_relation_id) {
    require_adaptive(manager_.initialize(base, delta_relation_id), "initialize");
}

ce_bindings::owned_support_snapshot_v1
ce_bindings::native_support_handle_v1::snapshot_copy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_copy_locked();
}

ce_bindings::owned_support_snapshot_v1
ce_bindings::native_support_handle_v1::snapshot_copy_locked() const {
    ce_support::support_snapshot_pair_v1 pair{};
    require_adaptive(manager_.snapshot(&pair), "snapshot");
    auto copy_relation = [](const ce_support::logical_support_view_v1 &view) {
        ce_bindings::owned_logical_support_v1 result;
        result.schema_version = view.schema_version;
        result.relation_id = view.relation_id;
        result.structure_id = view.structure_id;
        result.structure_epoch = view.structure_epoch;
        if (view.edge_count != 0u)
            result.edges.assign(view.edges, view.edges + view.edge_count);
        if (view.participant_count != 0u)
            result.participants.assign(view.participants,
                                       view.participants + view.participant_count);
        return result;
    };
    ce_bindings::owned_support_snapshot_v1 result;
    result.base = copy_relation(*pair.base);
    result.delta = copy_relation(*pair.delta);
    result.promotion_records = manager_.promotion_records();
    result.revoked_edge_ids = manager_.revoked_edge_ids();
    result.structure_epoch = pair.snapshot_structure_epoch;
    return result;
}

ce_support::adaptive_support_status_v1
ce_bindings::native_support_handle_v1::publish_pending_snapshot(
    std::uint64_t epoch, ce_bindings::owned_support_snapshot_v1* output) {
    if (output == nullptr) return ce_support::adaptive_support_status_v1::invalid_support;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto status = manager_.publish_pending(epoch);
    if (status != ce_support::adaptive_support_status_v1::success) return status;
    *output = snapshot_copy_locked();
    return ce_support::adaptive_support_status_v1::success;
}

void bind_sparse_support_indexing(py::module_ &module) {
    py::class_<ce_bindings::native_support_handle_v1,
               std::shared_ptr<ce_bindings::native_support_handle_v1>>(
        module, "NativeSupportHandle")
        .def(py::init([](py::array edge_ids_arg, py::array offsets_arg,
                         py::array roles_arg, py::array entities_arg,
                         std::uint64_t relation_id, std::uint64_t structure_id,
                         std::uint64_t structure_epoch,
                         std::uint64_t delta_relation_id) {
            auto edge_ids = require_vector<std::int64_t>(edge_ids_arg, "edge_ids");
            auto offsets = require_vector<std::int64_t>(offsets_arg, "participant_offsets");
            auto roles = require_vector<std::int32_t>(roles_arg, "participant_roles");
            auto entities = require_vector<std::int64_t>(entities_arg,
                                                         "participant_entity_ids");
            if (offsets.size() != edge_ids.size() + 1 || offsets.size() == 0
                || offsets.data()[0] != 0 || offsets.data()[offsets.size() - 1]
                    != entities.size() || roles.size() != entities.size())
                throw py::value_error("support CSR offsets and participant arrays disagree");
            std::vector<ce_support::logical_support_edge_v1> edges(
                static_cast<std::size_t>(edge_ids.size()));
            std::vector<ce_support::logical_support_participant_v1> participants(
                static_cast<std::size_t>(entities.size()));
            for (py::ssize_t i = 0; i < edge_ids.size(); ++i) {
                if (edge_ids.data()[i] <= 0 || offsets.data()[i] < 0
                    || offsets.data()[i + 1] < offsets.data()[i]
                    || offsets.data()[i + 1] - offsets.data()[i]
                        > std::numeric_limits<std::uint32_t>::max())
                    throw py::value_error("support edge IDs or participant offsets are invalid");
                edges[static_cast<std::size_t>(i)] = {
                    static_cast<std::uint64_t>(edge_ids.data()[i]),
                    static_cast<std::uint64_t>(offsets.data()[i]),
                    static_cast<std::uint32_t>(offsets.data()[i + 1] - offsets.data()[i]), 0u};
            }
            for (py::ssize_t i = 0; i < entities.size(); ++i) {
                if (roles.data()[i] < 0 || entities.data()[i] <= 0)
                    throw py::value_error("participant roles and entity IDs must be valid");
                participants[static_cast<std::size_t>(i)] = {
                    static_cast<std::uint32_t>(roles.data()[i]), 0u,
                    static_cast<std::uint64_t>(entities.data()[i])};
            }
            const ce_support::logical_support_view_v1 view{
                ce_support::logical_support_schema_version_v1, 0u, relation_id,
                structure_id, structure_epoch,
                edges.empty() ? nullptr : edges.data(), edges.size(),
                participants.empty() ? nullptr : participants.data(), participants.size()};
            return std::make_shared<ce_bindings::native_support_handle_v1>(
                view, delta_relation_id);
        }), py::arg("edge_ids"), py::arg("participant_offsets"),
            py::arg("participant_roles"), py::arg("participant_entity_ids"),
            py::arg("relation_id"), py::arg("structure_id"),
            py::arg("structure_epoch"), py::arg("delta_relation_id") = 0u)
        .def("stage_promotion", [](ce_bindings::native_support_handle_v1 &handle,
                                    std::uint64_t edge_id, py::array roles_arg,
                                    py::array entities_arg,
                                    std::uint64_t discovery_round_id,
                                    std::uint64_t scorer_generation,
                                    float proposal_score) {
            auto roles = require_vector<std::int32_t>(roles_arg, "participant_roles");
            auto entities = require_vector<std::int64_t>(entities_arg,
                                                         "participant_entity_ids");
            if (roles.size() != entities.size()
                || roles.size() > std::numeric_limits<std::uint32_t>::max())
                throw py::value_error("promotion participant arrays disagree");
            std::vector<ce_support::logical_support_participant_v1> participants(
                static_cast<std::size_t>(roles.size()));
            for (py::ssize_t i = 0; i < roles.size(); ++i) {
                if (roles.data()[i] < 0 || entities.data()[i] <= 0)
                    throw py::value_error("promotion participant identity is invalid");
                participants[static_cast<std::size_t>(i)] = {
                    static_cast<std::uint32_t>(roles.data()[i]), 0u,
                    static_cast<std::uint64_t>(entities.data()[i])};
            }
            require_adaptive(handle.stage_promotion({edge_id,
                participants.empty() ? nullptr : participants.data(),
                static_cast<std::uint32_t>(participants.size()), 0u,
                discovery_round_id, scorer_generation, proposal_score}),
                "stage promotion");
        }, py::arg("logical_edge_id"), py::arg("participant_roles"),
            py::arg("participant_entity_ids"), py::arg("discovery_round_id") = 0u,
            py::arg("scorer_generation") = 0u, py::arg("proposal_score") = 0.0f)
        .def("stage_revocation", [](ce_bindings::native_support_handle_v1 &handle,
                                     std::uint64_t edge_id) {
            require_adaptive(handle.stage_revocation(edge_id), "stage revocation");
        })
        .def("stage_base_promotion", [](ce_bindings::native_support_handle_v1 &handle,
                                         std::uint64_t edge_id) {
            require_adaptive(handle.stage_base_promotion(edge_id),
                             "stage base promotion");
        })
        .def("publish_pending", [](ce_bindings::native_support_handle_v1 &handle,
                                    std::uint64_t epoch) {
            require_adaptive(handle.publish_pending(epoch), "publish support revision");
            return handle.structure_epoch();
        })
        .def("publish_pending_snapshot", [](ce_bindings::native_support_handle_v1 &handle,
                                              std::uint64_t epoch) {
            ce_bindings::owned_support_snapshot_v1 snapshot;
            require_adaptive(handle.publish_pending_snapshot(epoch, &snapshot),
                             "publish support revision and copy snapshot");
            return support_snapshot_dict(snapshot);
        }, py::arg("next_epoch"))
        .def("pin_forward_tape", [](ce_bindings::native_support_handle_v1 &handle,
                                     std::uint64_t epoch) {
            require_adaptive(handle.pin_forward_tape(epoch), "pin forward tape");
        })
        .def("retire_forward_tape", [](ce_bindings::native_support_handle_v1 &handle,
                                       std::uint64_t epoch) {
            require_adaptive(handle.retire_forward_tape(epoch), "retire forward tape");
        })
        .def("snapshot", [](const ce_bindings::native_support_handle_v1 &handle) {
            return support_snapshot_dict(handle.snapshot_copy());
        })
        .def_property_readonly("structure_epoch",
            &ce_bindings::native_support_handle_v1::structure_epoch);

    py::class_<NativeReverseIndex>(module, "NativeReverseIndex")
        .def(py::init<std::shared_ptr<ce_bindings::native_support_handle_v1>>(),
             py::arg("support"))
        .def("query_any", [](const NativeReverseIndex &index, py::array roles,
                              py::array entities, std::uint64_t capacity) {
            return index.query(roles, entities, false, capacity);
        }, py::arg("roles"), py::arg("entity_ids"), py::arg("capacity"))
        .def("query_all", [](const NativeReverseIndex &index, py::array roles,
                              py::array entities, std::uint64_t capacity) {
            return index.query(roles, entities, true, capacity);
        }, py::arg("roles"), py::arg("entity_ids"), py::arg("capacity"))
        .def_property_readonly("structure_epoch", &NativeReverseIndex::structure_epoch);

    py::class_<NativeTemporalOccurrenceRing>(module, "NativeTemporalOccurrenceRing")
        .def(py::init<std::uint64_t, std::uint64_t, std::uint64_t>(),
             py::arg("cell_id"), py::arg("entity_id"), py::arg("capacity"))
        .def("append", &NativeTemporalOccurrenceRing::append,
             py::arg("occurrence_id"), py::arg("ordinal"),
             py::arg("timestamp") = py::none())
        .def("query_window", &NativeTemporalOccurrenceRing::query,
             py::arg("minimum_ordinal") = py::none(),
             py::arg("maximum_ordinal") = py::none(),
             py::arg("minimum_timestamp") = py::none(),
             py::arg("maximum_timestamp") = py::none());

    module.def("build_native_condition_worksets", &build_native_condition_worksets,
        py::arg("structure_epoch"), py::arg("edge_count"), py::arg("state_generation"),
        py::arg("condition_opcodes"), py::arg("condition_expected"),
        py::arg("condition_first_child"), py::arg("condition_child_count"),
        py::arg("condition_predicate_index"), py::arg("binding_edge_ordinals"),
        py::arg("binding_root_nodes"), py::arg("predicate_values"),
        py::arg("primal_offsets"), py::arg("primal_edge_ids"),
        py::arg("primal_edge_ordinals"), py::arg("derivative_offsets"),
        py::arg("derivative_edge_ids"), py::arg("derivative_edge_ordinals"),
        py::arg("primal_capacity"), py::arg("derivative_capacity"));
    module.def("build_native_derivative_sample", &build_native_derivative_sample,
        py::arg("admitted_offsets"), py::arg("admitted_ids"),
        py::arg("admitted_ordinals"), py::arg("dormant_offsets"),
        py::arg("dormant_ids"), py::arg("dormant_ordinals"),
        py::arg("per_cell_budgets"), py::arg("window_counter"), py::arg("seed"),
        py::arg("structure_epoch") = 1u, py::arg("state_generation") = 1u);
    module.def("propose_native_support", &propose_native_support,
        py::arg("entity_ids"), py::arg("mechanism_ids"),
        py::arg("posting_seed_entity_ids"), py::arg("recent_coactivations"),
        py::arg("participant_count"), py::arg("seed"),
        py::arg("proposal_budget"), py::arg("attempt_budget"),
        py::arg("known_mechanism_ids") = py::none(),
        py::arg("known_participant_ids") = py::none());
}
