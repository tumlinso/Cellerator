#pragma once
#include <Cellerator/execution/lowering_resumption/resumption_v1.hh>
#include <cstdint>
namespace cellerator::compiler::diagnostics::v1 {
enum class generation_kind:std::uint8_t{structure=0,value,support,order,profile};
struct generation_change{generation_kind kind=generation_kind::structure;std::uint64_t before=0,after=0,statement=0;};
struct staleness_explanation{bool changed=false;std::uint32_t stale_artifact_mask=0;cellerator::execution::lowering_resumption::lowering_stage_v1 resume_from=cellerator::execution::lowering_resumption::lowering_stage_v1::canonical_source;};
[[nodiscard]] staleness_explanation explain_staleness(const generation_change&) noexcept;
}
