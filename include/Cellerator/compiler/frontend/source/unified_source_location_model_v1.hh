#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::source {

using source_space_id_v1 = std::uint32_t;
inline constexpr source_space_id_v1 invalid_source_space_v1 = 0;

enum class source_space_kind_v1 : std::uint8_t {
    physical_file = 1,
    include_instance,
    macro_expansion,
    transformed_buffer,
    ceir_node,
    backend_output,
};

struct source_location_v1 {
    source_space_id_v1 space = invalid_source_space_v1;
    std::uint64_t byte_offset = 0;

    friend constexpr bool operator==(source_location_v1 lhs,
                                     source_location_v1 rhs) noexcept {
        return lhs.space == rhs.space && lhs.byte_offset == rhs.byte_offset;
    }
};

struct source_span_v1 {
    source_location_v1 begin{};
    source_location_v1 end{};

    [[nodiscard]] constexpr bool valid() const noexcept {
        return begin.space != invalid_source_space_v1 && begin.space == end.space &&
               begin.byte_offset <= end.byte_offset;
    }

    [[nodiscard]] constexpr std::uint64_t size_bytes() const noexcept {
        return valid() ? end.byte_offset - begin.byte_offset : 0;
    }
};

// Lines are one-based. byte_column is zero-based and deliberately counts bytes:
// unlike a display column, it remains exactly reversible for UTF-8 and tabs.
struct line_column_v1 {
    std::uint64_t line = 1;
    std::uint64_t byte_column = 0;
};

struct source_space_v1 {
    source_space_id_v1 id = invalid_source_space_v1;
    source_space_kind_v1 kind = source_space_kind_v1::physical_file;
    std::string stable_name;
    std::string bytes;
    std::optional<source_space_id_v1> parent;
    std::vector<std::uint64_t> line_starts;
};

enum class mapping_edge_kind_v1 : std::uint8_t {
    include_expansion = 1,
    macro_spelling,
    macro_expansion,
    source_transform,
    ceir_provenance,
    backend_provenance,
};

struct source_mapping_edge_v1 {
    source_span_v1 derived{};
    source_span_v1 origin{};
    mapping_edge_kind_v1 kind = mapping_edge_kind_v1::source_transform;

    [[nodiscard]] constexpr bool reversible() const noexcept {
        return derived.valid() && origin.valid() &&
               derived.size_bytes() == origin.size_bytes();
    }
};

class source_map_v1 {
  public:
    [[nodiscard]] source_space_id_v1 add_space(source_space_kind_v1 kind,
                                               std::string stable_name,
                                               std::string bytes,
                                               std::optional<source_space_id_v1> parent = std::nullopt);

    [[nodiscard]] const source_space_v1* find_space(source_space_id_v1 id) const noexcept;
    [[nodiscard]] std::optional<line_column_v1> line_column(source_location_v1 location) const noexcept;
    [[nodiscard]] std::optional<source_location_v1> location(source_space_id_v1 space,
                                                             line_column_v1 position) const noexcept;

    // Only equal-length edges are admitted. This makes every byte mapping exact
    // and reversible; lossy transformations must publish smaller exact edges.
    [[nodiscard]] bool add_mapping(source_mapping_edge_v1 edge);
    [[nodiscard]] std::optional<source_location_v1> map_to_origin(source_location_v1 location) const noexcept;
    [[nodiscard]] std::optional<source_location_v1> map_to_derived(source_location_v1 location) const noexcept;

    [[nodiscard]] const std::vector<source_mapping_edge_v1>& mappings() const noexcept {
        return mappings_;
    }

  private:
    std::vector<source_space_v1> spaces_;
    std::vector<source_mapping_edge_v1> mappings_;
};

} // namespace Cellerator::compiler::frontend::source
