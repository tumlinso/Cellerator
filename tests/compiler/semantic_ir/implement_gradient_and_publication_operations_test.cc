#include <Cellerator/compiler/ir/semantic/implement_gradient_and_publication_operations_v1.hh>

#include <array>
#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;
using cellerator::execution::numeric_type;

int main() {
    const semantic_identity_v1 source{1, 2};
    const semantic_identity_v1 destination{3, 4};
    gradient_publication_program_ir_v1 semantic;
    semantic.program_identity = 10;
    semantic.structure_identity = {11, 12};
    semantic.structure_epoch = 2;
    semantic.prepared_generation = 3;
    semantic.numerical = {numeric_type::f32, numeric_type::f32,
                          numeric_type::f32, numeric_type::f32};
    semantic.stages = {
        {20, gradient_publication_operation_ir_v1::forward, source, destination, 3, 0, false},
        {21, gradient_publication_operation_ir_v1::transpose, destination, source, 3, 0, false},
        {22, gradient_publication_operation_ir_v1::value_gradient, source, destination, 3, 0, false},
        {23, gradient_publication_operation_ir_v1::canonicalize, destination, destination, 3, 0, true},
        {24, gradient_publication_operation_ir_v1::publish_generation, destination, destination, 3, 4, false},
    };
    semantic.update_policy = {30, 31, true};
    assert(validate_gradient_publication_program_ir_v1(semantic) ==
           gradient_publication_status_ir_v1::success);

    std::array<cellerator::execution::training_v2::training_stage_v2, 5> stages{};
    for (std::size_t index = 0; index < stages.size(); ++index) {
        stages[index].stage_identity = semantic.stages[index].identity;
        stages[index].kind = lower_gradient_publication_stage_kind_v1(
            semantic.stages[index].kind);
    }
    cellerator::execution::training_v2::training_program_v2 training;
    training.stage_count = stages.size();
    training.stages = stages.data();
    training.program_identity = semantic.program_identity;
    training.epoch = {semantic.structure_epoch};
    training.prepared_generation = {semantic.prepared_generation};
    assert(compare_gradient_program_with_training_v2(semantic, training) ==
           gradient_publication_status_ir_v1::success);

    semantic.stages[3].explicit_order_transform = false;
    assert(validate_gradient_publication_program_ir_v1(semantic) ==
           gradient_publication_status_ir_v1::invalid_canonicalization);
    semantic.stages[3].explicit_order_transform = true;
    semantic.update_policy.owned_by_caller = false;
    assert(validate_gradient_publication_program_ir_v1(semantic) ==
           gradient_publication_status_ir_v1::update_policy_not_caller_owned);

    std::cout << "gradient_closure=complete publication=3_to_4 caller_policy=separate\n";
}
