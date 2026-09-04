#include <Cellerator/compiler/ir/realization/implement_realization_ir_validators_and_referees_v1.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::compiler::ir::realization::v1 {
namespace {

constexpr std::uint32_t phase_bit(realization_validation_phase_v1 phase) noexcept {
    return 1u << (static_cast<std::uint32_t>(phase) - 1u);
}

realization_validation_receipt_v1 failure(
    realization_validation_status_v1 status,
    realization_validation_phase_v1 phase,
    std::uint32_t phases_run,
    std::string detail) {
    return {status, phase, phases_run, 0u, false, std::move(detail)};
}

bool valid_tolerance(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

} // namespace

realization_validation_status_v1 run_weighted_row_referee_v1(
    const weighted_row_referee_case_v1& test_case,
    std::string* error) noexcept {
    if (test_case.row_offsets.empty() || test_case.row_offsets.front() != 0u ||
        test_case.row_offsets.back() != test_case.column_indices.size() ||
        test_case.weights.size() != test_case.column_indices.size() ||
        test_case.expected.size() + 1u != test_case.row_offsets.size() ||
        !valid_tolerance(test_case.absolute_tolerance) ||
        !valid_tolerance(test_case.relative_tolerance)) {
        if (error != nullptr) *error = "malformed weighted-row referee case";
        return realization_validation_status_v1::referee_mismatch;
    }
    for (std::size_t row = 0; row < test_case.expected.size(); ++row) {
        if (test_case.row_offsets[row] > test_case.row_offsets[row + 1u]) {
            if (error != nullptr) *error = "row offsets are not monotonic";
            return realization_validation_status_v1::referee_mismatch;
        }
        long double observed = 0.0L;
        for (std::uint64_t index = test_case.row_offsets[row];
             index < test_case.row_offsets[row + 1u]; ++index) {
            const auto column = test_case.column_indices[index];
            if (column >= test_case.dense_input.size()) {
                if (error != nullptr) *error = "referee column is out of range";
                return realization_validation_status_v1::referee_mismatch;
            }
            observed += static_cast<long double>(test_case.weights[index]) *
                static_cast<long double>(test_case.dense_input[column]);
        }
        const auto expected = static_cast<long double>(test_case.expected[row]);
        const auto tolerance = static_cast<long double>(test_case.absolute_tolerance) +
            static_cast<long double>(test_case.relative_tolerance) * std::abs(expected);
        if (!std::isfinite(static_cast<double>(observed)) ||
            std::abs(observed - expected) > tolerance) {
            if (error != nullptr) *error = "weighted-row referee result differs";
            return realization_validation_status_v1::referee_mismatch;
        }
    }
    if (error != nullptr) error->clear();
    return realization_validation_status_v1::valid;
}

realization_validation_receipt_v1 validate_realization_ir_v1(
    const realization_validation_request_v1& request) noexcept {
    std::uint32_t phases_run = phase_bit(realization_validation_phase_v1::structural);
    std::string detail;
    const auto document = parse_realization_text_v1(request.serialized_ir, &detail);
    if (!document) {
        return failure(realization_validation_status_v1::structurally_invalid,
            realization_validation_phase_v1::structural, phases_run, std::move(detail));
    }

    if (request.mode == realization_validation_mode_v1::unchecked) {
        realization_validation_receipt_v1 receipt;
        receipt.status = request.allow_unsafe_continuation
            ? realization_validation_status_v1::unsafe_continuation
            : realization_validation_status_v1::semantically_invalid;
        receipt.failed_phase = realization_validation_phase_v1::semantic;
        receipt.phases_run = phases_run;
        receipt.phases_skipped = phase_bit(realization_validation_phase_v1::semantic) |
            phase_bit(realization_validation_phase_v1::exact_coverage) |
            phase_bit(realization_validation_phase_v1::resource_capability) |
            phase_bit(realization_validation_phase_v1::host_referee);
        receipt.unsafe_continuation_used = request.allow_unsafe_continuation;
        receipt.detail = request.allow_unsafe_continuation
            ? "semantic and execution checks explicitly bypassed"
            : "unchecked validation requires explicit unsafe continuation";
        return receipt;
    }

    phases_run |= phase_bit(realization_validation_phase_v1::semantic);
    if (request.module == nullptr ||
        validate_realization_module_v1(*request.module, &detail) !=
            realization_module_status_v1::valid ||
        !(document->module == request.module->identity)) {
        if (detail.empty()) detail = "serialized and semantic module identities differ";
        return failure(realization_validation_status_v1::semantically_invalid,
            realization_validation_phase_v1::semantic, phases_run, std::move(detail));
    }

    phases_run |= phase_bit(realization_validation_phase_v1::exact_coverage);
    if (request.exact_cover == nullptr ||
        validate_exact_cover_v1(*request.exact_cover, &detail) != exact_cover_status_v1::exact) {
        if (detail.empty()) detail = "exact cover is required";
        return failure(realization_validation_status_v1::inexact_coverage,
            realization_validation_phase_v1::exact_coverage, phases_run, std::move(detail));
    }

    phases_run |= phase_bit(realization_validation_phase_v1::resource_capability);
    const bool missing_resource_contract = request.capability == nullptr ||
        request.requirement == nullptr || request.memory_requirements == nullptr ||
        request.memory_accounting == nullptr || request.bindings == nullptr ||
        request.stage_graph == nullptr || request.launch_graph == nullptr;
    if (missing_resource_contract ||
        satisfies_target_requirement_v1(*request.capability, *request.requirement, &detail) !=
            capability_status_v1::compatible ||
        compare_memory_requirements_v1(*request.memory_requirements,
            *request.memory_accounting, &detail) != memory_requirement_status_v1::valid ||
        validate_symbolic_binding_table_v1(*request.bindings) !=
            symbolic_binding_status_v1::valid ||
        validate_prepared_stage_graph_v1(*request.stage_graph, &detail) !=
            stage_graph_status_v1::valid ||
        validate_launch_dependency_graph_v1(*request.launch_graph, &detail) !=
            launch_dependency_status_v1::valid) {
        if (detail.empty()) detail = "resource contracts are missing or incompatible";
        return failure(realization_validation_status_v1::resource_incompatible,
            realization_validation_phase_v1::resource_capability, phases_run, std::move(detail));
    }

    realization_validation_receipt_v1 receipt;
    receipt.phases_run = phases_run;
    if (request.mode == realization_validation_mode_v1::verified) {
        phases_run |= phase_bit(realization_validation_phase_v1::host_referee);
        receipt.phases_run = phases_run;
        if (request.referee == nullptr ||
            run_weighted_row_referee_v1(*request.referee, &detail) !=
                realization_validation_status_v1::valid) {
            if (detail.empty()) detail = "verified validation requires host referee evidence";
            return failure(realization_validation_status_v1::referee_mismatch,
                realization_validation_phase_v1::host_referee, phases_run, std::move(detail));
        }
    } else {
        receipt.phases_skipped = phase_bit(realization_validation_phase_v1::host_referee);
    }
    receipt.status = realization_validation_status_v1::valid;
    receipt.failed_phase = realization_validation_phase_v1::structural;
    return receipt;
}

} // namespace cellerator::compiler::ir::realization::v1
