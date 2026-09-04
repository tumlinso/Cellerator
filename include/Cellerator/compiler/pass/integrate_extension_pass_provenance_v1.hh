#pragma once

#include <Cellerator/compiler/pass/freeze_the_pass_pipeline_stage_taxonomy_v1.hh>
#include <Cellerator/compiler/pass/implement_pass_manager_and_analysis_invalidation_v1.hh>
#include <Cellerator/compiler/pass/implement_transform_sandbox_policy_as_opt_in_not_authori_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::pass::v1 {

struct extension_pass_provenance_record_v1 {
    std::string extension_identity;
    std::string binary_or_source_hash;
    pipeline_stage_v1 pipeline_location{};
    std::vector<std::uint64_t> input_operation_ids;
    std::vector<std::uint64_t> output_operation_ids;
    analysis_set_v1 invalidated_analyses = 0;
    std::vector<std::string> diagnostics;
    transform_execution_mode_v1 trust_mode =
        transform_execution_mode_v1::trusted_in_process;
};

class cold_extension_provenance_v1 {
public:
    [[nodiscard]] bool append(extension_pass_provenance_record_v1 record);
    [[nodiscard]] std::vector<const extension_pass_provenance_record_v1*>
    trace_operation(std::uint64_t operation_id) const;
    [[nodiscard]] const std::vector<extension_pass_provenance_record_v1>& records()
        const noexcept { return records_; }

private:
    std::vector<extension_pass_provenance_record_v1> records_;
};

}  // namespace cellerator::compiler::pass::v1
