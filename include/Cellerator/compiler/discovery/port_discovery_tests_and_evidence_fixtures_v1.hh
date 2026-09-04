#pragma once

#include <cstddef>
#include <string_view>

namespace Cellerator::compiler::discovery {

struct migrated_fixture_family_v1 {
    std::string_view source_path;
    std::string_view source_tree_sha256;
    std::string_view focused_gate;
    std::string_view intentional_change;
    std::size_t source_file_count;
};

[[nodiscard]] const migrated_fixture_family_v1* migrated_fixture_inventory_v1(
    std::size_t* count) noexcept;

[[nodiscard]] bool valid_migrated_fixture_family_v1(
    const migrated_fixture_family_v1& family) noexcept;

[[nodiscard]] std::size_t migrated_fixture_source_file_count_v1() noexcept;

}  // namespace Cellerator::compiler::discovery
