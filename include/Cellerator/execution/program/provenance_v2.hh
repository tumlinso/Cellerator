#pragma once
#include <Cellerator/execution/program/program_v2.h>
#include <Cellerator/execution/identity.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>
namespace cellerator::execution::program {
// Existing semantic IDs identify numerical operations and source definitions.
// Neither stage nor candidate IDs establish molecular or biological identity.
struct operation_origin_v2 {
    compute::operation::v2::stable_id operation{}, source{};
    execution::structure_epoch source_epoch{};
    execution::value_generation values{};
};
struct stage_origins_v2 {
    std::uint64_t stage_id=0, candidate_id=0;
    const operation_origin_v2* origins=nullptr;
    std::uint64_t count=0;
};
// Borrowed immutable preparation metadata: one row per actual prepared stage.
// Multiple stages may refer to one origin and a fused stage to several origins.
struct program_origins_v2 {
    const prepared_program_v2* program=nullptr;
    const stage_origins_v2* stages=nullptr;
    std::uint64_t count=0;
};
bool valid_origins_v2(const program_origins_v2&) noexcept;
// Cold set comparison includes source epoch and value generation, ignoring only
// repeated origin references caused by decomposition. No runtime hash registry.
bool same_origins_v2(const program_origins_v2&,const program_origins_v2&) noexcept;
} // namespace cellerator::execution::program
