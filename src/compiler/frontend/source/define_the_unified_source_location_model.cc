#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <algorithm>
#include <limits>
#include <utility>

namespace Cellerator::compiler::frontend::source {
namespace {

std::vector<std::uint64_t> line_starts(std::string_view bytes) {
    std::vector<std::uint64_t> result{0};
    for (std::uint64_t offset = 0; offset < bytes.size(); ++offset) {
        if (bytes[offset] == '\n') {
            result.push_back(offset + 1);
        }
    }
    return result;
}

bool contains(source_span_v1 span, source_location_v1 location) noexcept {
    return span.valid() && span.begin.space == location.space &&
           span.begin.byte_offset <= location.byte_offset &&
           location.byte_offset <= span.end.byte_offset;
}

} // namespace

source_space_id_v1 source_map_v1::add_space(source_space_kind_v1 kind,
                                            std::string stable_name,
                                            std::string bytes,
                                            std::optional<source_space_id_v1> parent) {
    if (spaces_.size() >= std::numeric_limits<source_space_id_v1>::max() - 1U) {
        return invalid_source_space_v1;
    }
    if (parent && find_space(*parent) == nullptr) {
        return invalid_source_space_v1;
    }
    const auto id = static_cast<source_space_id_v1>(spaces_.size() + 1U);
    auto starts = line_starts(bytes);
    spaces_.push_back(source_space_v1{id, kind, std::move(stable_name),
                                      std::move(bytes), parent, std::move(starts)});
    return id;
}

const source_space_v1* source_map_v1::find_space(source_space_id_v1 id) const noexcept {
    if (id == invalid_source_space_v1 || id > spaces_.size()) {
        return nullptr;
    }
    return &spaces_[id - 1U];
}

std::optional<line_column_v1> source_map_v1::line_column(source_location_v1 location) const noexcept {
    const auto* space = find_space(location.space);
    if (space == nullptr || location.byte_offset > space->bytes.size()) {
        return std::nullopt;
    }
    const auto upper = std::upper_bound(space->line_starts.begin(),
                                        space->line_starts.end(), location.byte_offset);
    const auto index = static_cast<std::size_t>(upper - space->line_starts.begin() - 1);
    return line_column_v1{index + 1U, location.byte_offset - space->line_starts[index]};
}

std::optional<source_location_v1> source_map_v1::location(source_space_id_v1 id,
                                                          line_column_v1 position) const noexcept {
    const auto* space = find_space(id);
    if (space == nullptr || position.line == 0 || position.line > space->line_starts.size()) {
        return std::nullopt;
    }
    const auto line_index = static_cast<std::size_t>(position.line - 1U);
    const auto start = space->line_starts[line_index];
    const auto limit = line_index + 1U < space->line_starts.size()
                           ? space->line_starts[line_index + 1U]
                           : static_cast<std::uint64_t>(space->bytes.size()) + 1U;
    const auto offset = start + position.byte_column;
    if (offset >= limit || offset > space->bytes.size()) {
        return std::nullopt;
    }
    return source_location_v1{id, offset};
}

bool source_map_v1::add_mapping(source_mapping_edge_v1 edge) {
    const auto* derived = find_space(edge.derived.begin.space);
    const auto* origin = find_space(edge.origin.begin.space);
    if (!edge.reversible() || derived == nullptr || origin == nullptr ||
        edge.derived.end.byte_offset > derived->bytes.size() ||
        edge.origin.end.byte_offset > origin->bytes.size()) {
        return false;
    }
    mappings_.push_back(edge);
    return true;
}

std::optional<source_location_v1> source_map_v1::map_to_origin(source_location_v1 location) const noexcept {
    for (auto it = mappings_.rbegin(); it != mappings_.rend(); ++it) {
        if (contains(it->derived, location)) {
            return source_location_v1{it->origin.begin.space,
                                      it->origin.begin.byte_offset +
                                          location.byte_offset - it->derived.begin.byte_offset};
        }
    }
    return std::nullopt;
}

std::optional<source_location_v1> source_map_v1::map_to_derived(source_location_v1 location) const noexcept {
    for (auto it = mappings_.rbegin(); it != mappings_.rend(); ++it) {
        if (contains(it->origin, location)) {
            return source_location_v1{it->derived.begin.space,
                                      it->derived.begin.byte_offset +
                                          location.byte_offset - it->origin.begin.byte_offset};
        }
    }
    return std::nullopt;
}

} // namespace Cellerator::compiler::frontend::source
