#include <Cellerator/compiler/backend/nvcc/deliver_the_first_nvcc_object_milestone_v1.hh>

namespace cellerator::compiler::backend::nvcc::v1 {

std::optional<nvcc_object_milestone> make_first_nvcc_object_milestone(
    const nvcc_object_milestone_request& request) {
    if (request.profile == 0 || request.planning_ir == 0 ||
        request.realization_ir == 0 || request.compute_capability < 50 ||
        request.relation_kernel.kind != cuda_entity_kind::kernel ||
        request.relation_kernel.stable_name.empty() ||
        request.relation_kernel.declaration.empty()) {
        return std::nullopt;
    }

    nvcc_object_milestone result;
    result.cuda_source = request.relation_kernel.declaration;
    result.compile_arguments = {
        "-std=c++17",
        "-arch=sm_" + std::to_string(request.compute_capability),
        "--compile",
    };
    result.link_arguments = {
        "-arch=sm_" + std::to_string(request.compute_capability),
        "-lcudart",
    };
    return result;
}

} // namespace cellerator::compiler::backend::nvcc::v1
