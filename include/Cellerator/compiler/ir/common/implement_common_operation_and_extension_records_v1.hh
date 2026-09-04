#pragma once

#include <Cellerator/compiler/ir/common/implement_regions_blocks_values_and_use_def_chains_v1.hh>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cellerator::compiler::ir {

enum class validation_mode : std::uint8_t { strict, compatible, opaque };
enum class effect_kind : std::uint8_t { read, write, allocate, synchronize, communicate };
struct source_provenance {
    std::string file;
    std::uint32_t begin{};
    std::uint32_t end{};
    std::uint64_t stable_source_id{};
};
struct named_attribute { std::string name; std::string canonical_value; };
struct opaque_extension { std::string namespace_name; std::vector<std::uint8_t> payload; };
struct common_operation {
    std::string namespace_name;
    std::string operation_name;
    std::vector<value_handle> operands;
    std::vector<value_handle> results;
    std::vector<region_handle> regions;
    std::vector<named_attribute> attributes;
    std::vector<effect_kind> effects;
    source_provenance provenance;
    validation_mode mode{validation_mode::strict};
    std::vector<opaque_extension> unknown_extensions;
};

enum class operation_validation : std::uint8_t {
    ok, missing_namespace, missing_name, duplicate_attribute, invalid_extension
};
operation_validation validate_common_operation(const common_operation &operation) noexcept;
std::string qualified_operation_name(const common_operation &operation);

} // namespace cellerator::compiler::ir
