#pragma once
#include <Cellerator/compiler/reflection/freeze_the_compile_time_ir_handle_model_v1.hh>
#include <cstdint>
#include <vector>
namespace cellerator::compiler::reflection::v1 {
enum class inline_identity_action_v1:std::uint8_t{one_to_one=1,split,fuse,clone,forced_reuse};
struct inline_identity_request_v1{inline_identity_action_v1 action=inline_identity_action_v1::one_to_one;std::vector<ir_handle_v1>sources;std::uint32_t output_count=1;bool semantically_equivalent=false,unsafe_acknowledged=false;};
struct inline_identity_result_v1{std::vector<ir_handle_v1>outputs;std::vector<ir_handle_v1>cold_lineage;bool unsafe_reuse=false;};
enum class inline_identity_status_v1:std::uint8_t{valid=0,missing_source,invalid_output_count,equivalence_required,unsafe_not_acknowledged};
[[nodiscard]] inline_identity_status_v1 derive_inline_identities_v1(const inline_identity_request_v1&,inline_identity_result_v1*)noexcept;
}
