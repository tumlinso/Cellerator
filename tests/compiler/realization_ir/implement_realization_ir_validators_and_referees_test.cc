#include <Cellerator/compiler/ir/realization/implement_realization_ir_validators_and_referees_v1.hh>

#include <cassert>

using namespace cellerator::compiler::ir::realization::v1;

namespace {

realization_validation_request_v1 valid_request() {
    static realization_module_v1 module{1u, {1u, 1u}, "module",
        {{{2u, 1u}, "host", "reference"}},
        {{{3u, 1u}, {2u, 1u}, realization_object_kind_v1::function, "forward",
            {{4u, 1u}, {4u, 2u}, {4u, 3u}}}}};
    static exact_cover_v1 cover{{5u, 1u}, {5u, 2u}, 2u,
        {{0u, {6u, 1u}, {7u, 1u}, {}, {}, {}, 0u},
         {1u, {6u, 2u}, {7u, 2u}, {}, {}, {}, 1u}}};
    static target_capability_v1 capability{{8u, 1u}, architecture_class_v1::host,
        {}, {"scalar"}, collective_scope_v1::none, memory_host_v1, numeric_f64_v1,
        false, "c++17", "native", "reference"};
    static target_requirement_v1 requirement{architecture_class_v1::host, {},
        {"scalar"}, collective_scope_v1::none, memory_host_v1, numeric_f64_v1,
        false, "c++17", "native", "reference"};
    static std::vector<memory_requirement_v1> memory{{{9u, 1u},
        allocation_class_v1::transient, allocation_owner_v1::caller,
        address_space_class_v1::host, plane_lifetime_v1::invocation, 64u, 64u, 16u}};
    static session_memory_accounting_v1 accounting{1024u, 0u, 0u, 0u, 1024u};
    static symbolic_binding_table_v1 bindings{{10u, 1u},
        {{{10u, 2u}, binding_slot_kind_v1::input, {}, 64u, 16u}}};
    static prepared_stage_graph_v1 stages{{11u, 1u},
        {{{11u, 2u}, {11u, 3u}, {10u, 2u}, prepared_stage_kind_v1::host_stub,
            {}, {{9u, 1u}}, {{12u, 1u}, order_class_v1::logical},
            {{12u, 1u}, order_class_v1::logical}, 1u, 1u, 1u,
            {0u, 2u, 0u, 2u}, 0u, 64u}}};
    static launch_dependency_graph_v1 launches{1u, {}, 0u};
    static weighted_row_referee_case_v1 referee{{0u, 2u, 3u}, {0u, 1u, 1u},
        {2.0, 3.0, 4.0}, {5.0, 7.0}, {31.0, 28.0}, 0.0, 0.0};

    realization_text_document_v1 document{{1u, 1u}, {{"stage", {11u, 2u}, "forward"}}};
    return {realization_validation_mode_v1::verified, false,
        print_realization_text_v1(document), &module, &cover, &capability, &requirement,
        &memory, &accounting, &bindings, &stages, &launches, &referee};
}

} // namespace

int main() {
    auto request = valid_request();
    auto receipt = validate_realization_ir_v1(request);
    assert(receipt.status == realization_validation_status_v1::valid);
    assert(receipt.phases_run == 0x1fu);

    request.mode = realization_validation_mode_v1::checked;
    request.referee = nullptr;
    receipt = validate_realization_ir_v1(request);
    assert(receipt.status == realization_validation_status_v1::valid);
    assert(receipt.phases_run == 0x0fu && receipt.phases_skipped == 0x10u);

    request.mode = realization_validation_mode_v1::unchecked;
    receipt = validate_realization_ir_v1(request);
    assert(receipt.status == realization_validation_status_v1::semantically_invalid);
    request.allow_unsafe_continuation = true;
    receipt = validate_realization_ir_v1(request);
    assert(receipt.status == realization_validation_status_v1::unsafe_continuation);
    assert(receipt.unsafe_continuation_used);

    request = valid_request();
    request.serialized_ir = "bad header";
    assert(validate_realization_ir_v1(request).status ==
        realization_validation_status_v1::structurally_invalid);

    request = valid_request();
    auto bad_referee = *request.referee;
    bad_referee.expected[1] = 29.0;
    request.referee = &bad_referee;
    assert(validate_realization_ir_v1(request).status ==
        realization_validation_status_v1::referee_mismatch);
}
