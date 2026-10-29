#pragma once
#include <cstdint>
namespace Cellerator::compiler::migration {
enum class superatom_disposition_v1:std::uint8_t{promoted=1,evaluated_not_promoted,invalid};
struct superatom_promotion_evidence_v1{std::uint64_t derivation_identity=0,exact_member_count=0,baseline_total_ns=0,composed_total_ns=0,deconstruction_digest=0;bool independently_verified=false;};
[[nodiscard]] constexpr superatom_disposition_v1 evaluate_superatom(superatom_promotion_evidence_v1 e)noexcept{
 if(!e.derivation_identity||!e.exact_member_count||!e.deconstruction_digest||!e.independently_verified)return superatom_disposition_v1::invalid;
 return e.composed_total_ns<e.baseline_total_ns?superatom_disposition_v1::promoted:superatom_disposition_v1::evaluated_not_promoted;
}
[[nodiscard]] constexpr bool is_storage_shard(superatom_promotion_evidence_v1)noexcept{return false;}
} // namespace Cellerator::compiler::migration
