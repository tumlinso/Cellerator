#include <Cellerator/compiler/tooling/deliver_celleratord_c_parity_milestone_v1.hh>

namespace Cellerator::compiler::tooling {

celleratord_cpp_parity_status_v1 validate_celleratord_cpp_parity_v1(
    const celleratord_cpp_parity_receipt_v1& receipt) noexcept {
    if (receipt.executable.empty())
        return celleratord_cpp_parity_status_v1::executable_missing;
    if (receipt.resource_directory.empty())
        return celleratord_cpp_parity_status_v1::resource_directory_missing;
    if (receipt.ordinary_cpp_documents == 0 || receipt.cellerator_documents == 0)
        return celleratord_cpp_parity_status_v1::mixed_workspace_missing;
    if (!receipt.ordinary_cpp_diagnostics || !receipt.ordinary_cpp_navigation ||
        !receipt.ordinary_cpp_completion)
        return celleratord_cpp_parity_status_v1::cpp_feature_missing;
    if (!receipt.cellerator_syntax_diagnostics)
        return celleratord_cpp_parity_status_v1::cellerator_diagnostics_missing;
    if (!receipt.host_only)
        return celleratord_cpp_parity_status_v1::cuda_required;
    if (!receipt.worker_process_stopped)
        return celleratord_cpp_parity_status_v1::process_leaked;
    return celleratord_cpp_parity_status_v1::valid;
}

}  // namespace Cellerator::compiler::tooling
