#pragma once
#include <Cellerator/compiler/frontend/source/implement_file_local_dialect_state_v1.hh>
#include <Cellerator/compiler/frontend/source/register_the_pragma_cellerator_preprocessor_contract_v1.hh>
namespace Cellerator::compiler::frontend::source {
using dialect_state = file_local_dialect_stack_v1;
using dialect_pragma_result = pragma_result_v1;
} // namespace Cellerator::compiler::frontend::source
