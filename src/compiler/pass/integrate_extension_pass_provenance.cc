#include <Cellerator/compiler/pass/integrate_extension_pass_provenance_v1.hh>

#include <algorithm>
#include <utility>

namespace cellerator::compiler::pass::v1 {

bool cold_extension_provenance_v1::append(
    extension_pass_provenance_record_v1 record) {
    if (record.extension_identity.empty() || record.binary_or_source_hash.empty()
        || !valid_pipeline_stage_v1(record.pipeline_location)
        || record.input_operation_ids.empty() || record.output_operation_ids.empty()) {
        return false;
    }
    records_.push_back(std::move(record));
    return true;
}

std::vector<const extension_pass_provenance_record_v1*>
cold_extension_provenance_v1::trace_operation(std::uint64_t operation_id) const {
    std::vector<const extension_pass_provenance_record_v1*> trace;
    std::vector<std::uint64_t> frontier{operation_id};
    for (auto iterator = records_.rbegin(); iterator != records_.rend(); ++iterator) {
        const bool produces = std::any_of(iterator->output_operation_ids.begin(),
            iterator->output_operation_ids.end(), [&](std::uint64_t output) {
                return std::find(frontier.begin(), frontier.end(), output) != frontier.end();
            });
        if (!produces) continue;
        trace.push_back(&*iterator);
        frontier.insert(frontier.end(), iterator->input_operation_ids.begin(),
            iterator->input_operation_ids.end());
    }
    std::reverse(trace.begin(), trace.end());
    return trace;
}

}  // namespace cellerator::compiler::pass::v1
