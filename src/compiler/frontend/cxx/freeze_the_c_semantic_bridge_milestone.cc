#include <Cellerator/compiler/frontend/cxx/freeze_the_c_semantic_bridge_milestone_v1.hh>

#include <utility>

namespace Cellerator::compiler::frontend::cxx {

cxx_semantic_bridge_milestone_status_v1 freeze_cxx_semantic_bridge_milestone_v1(
    const cxx_semantic_bridge_milestone_request_v1& request,
    cxx_semantic_bridge_milestone_v1* milestone) noexcept {
    if (milestone == nullptr) {
        return cxx_semantic_bridge_milestone_status_v1::null_output;
    }
    *milestone = {};
    if (request.schema_version != cxx_semantic_bridge_milestone_schema_version_v1) {
        return cxx_semantic_bridge_milestone_status_v1::schema_mismatch;
    }
    if (request.adapter == nullptr ||
        validate_upstream_clang_adapter_v1(*request.adapter) !=
            upstream_clang_adapter_status_v1::success) {
        return cxx_semantic_bridge_milestone_status_v1::invalid_adapter;
    }

    source_capture_binding_result_v1 bindings;
    if (bind_source_captures_v1(
            source_capture_binding_schema_version_v1,
            *request.adapter,
            {request.activated_placeholder},
            &bindings) != source_capture_binding_status_v1::success ||
        bindings.captures.size() != 1) {
        return cxx_semantic_bridge_milestone_status_v1::placeholder_resolution_failed;
    }

    std::vector<cxx_type_record_v1> types;
    if (extract_cxx_types_v1(
            cxx_type_extraction_schema_version_v1,
            *request.adapter,
            bindings.captures,
            &types) != cxx_type_extraction_status_v1::success ||
        types.size() != 1) {
        return cxx_semantic_bridge_milestone_status_v1::numeric_type_resolution_failed;
    }

    std::vector<biological_template_operation_v1> operations;
    if (instantiate_biological_template_operations_v1(
            biological_template_operation_schema_version_v1,
            *request.adapter,
            request.biological_template_name,
            &operations) != biological_template_operation_status_v1::success) {
        return cxx_semantic_bridge_milestone_status_v1::template_resolution_failed;
    }

    std::vector<constexpr_value_v1> constants;
    if (import_constexpr_values_v1(
            constexpr_import_schema_version_v1,
            *request.adapter,
            request.constants,
            &constants) != constexpr_import_status_v1::success) {
        return cxx_semantic_bridge_milestone_status_v1::constexpr_resolution_failed;
    }

    milestone->llvm_major = request.adapter->llvm_major;
    milestone->placeholder = std::move(bindings.captures.front());
    milestone->numeric_type = std::move(types.front());
    milestone->operations = std::move(operations);
    milestone->constants = std::move(constants);
    return cxx_semantic_bridge_milestone_status_v1::success;
}

}  // namespace Cellerator::compiler::frontend::cxx
