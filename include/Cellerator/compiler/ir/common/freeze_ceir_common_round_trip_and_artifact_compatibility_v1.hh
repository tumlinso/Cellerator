#pragma once

#include <Cellerator/compiler/ir/common/implement_common_operation_and_extension_records_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::ir {

struct round_trip_report {
    bool text_stable{};
    bool binary_valid{};
    bool binary_payload_equal{};
    bool unknown_extensions_preserved{};
    bool standalone_resumed{};
    bool source_inline_parsed{};
    std::uint64_t canonical_hash{};
    std::string diagnostic;
};

round_trip_report verify_common_round_trip(
    const std::vector<common_operation> &operations);

} // namespace cellerator::compiler::ir
