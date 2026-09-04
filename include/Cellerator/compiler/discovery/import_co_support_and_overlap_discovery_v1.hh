#pragma once

#include <Cellerator/compiler/discovery/import_support_signature_discovery_v1.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::discovery {

struct co_support_proposal_v1 {
    std::uint64_t first_source = 0;
    std::uint64_t second_source = 0;
    std::uint64_t observed_together = 0;
    std::uint64_t first_prevalence = 0;
    std::uint64_t second_prevalence = 0;
    std::uint64_t null_numerator = 0;
    std::uint64_t null_denominator = 0;
};

struct co_support_discovery_v1 {
    std::vector<std::uint64_t> source_prevalence;
    std::vector<std::uint64_t> destination_convergence;
    std::vector<co_support_proposal_v1> proposals;
    std::uint64_t enumerated_pairs = 0;
};

enum class co_support_status_v1 : std::uint8_t {
    success = 0,
    invalid_relation,
    invalid_source,
    invalid_config,
};

[[nodiscard]] co_support_status_v1 discover_co_support_and_overlap_v1(
    support_relation_view_v1 relation,
    std::uint64_t source_count,
    std::uint32_t top_l,
    co_support_discovery_v1* output) noexcept;

}  // namespace Cellerator::compiler::discovery
