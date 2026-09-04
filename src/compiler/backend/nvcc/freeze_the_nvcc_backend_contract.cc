#include <Cellerator/compiler/backend/nvcc/freeze_the_nvcc_backend_contract_v1.hh>

namespace cellerator::compiler::backend::nvcc::v1 {

contract_status validate_job(const compilation_job& job) noexcept {
    if (job.generated_input.empty() || job.output.empty()) return contract_status::invalid_job;
    const bool device = job.kind == job_kind::device_compile || job.kind == job_kind::device_link;
    if (job.kind < job_kind::device_compile || job.kind > job_kind::host_link)
        return contract_status::invalid_job;
    if (!job.input_is_generated && !job.pure_cuda_fallthrough)
        return contract_status::cellerator_source_reaches_nvcc;
    if (device && job.target_architectures.empty()) return contract_status::missing_architecture;
    for (const auto architecture : job.target_architectures)
        if (architecture < 50u || architecture > 999u) return contract_status::invalid_architecture;
    if (job.input_is_generated && job.source_map.empty()) return contract_status::missing_source_map;
    if (job.pure_cuda_fallthrough && (job.input_is_generated || job.kind != job_kind::device_compile))
        return contract_status::invalid_fallthrough;
    for (const auto& entry : job.source_map)
        if (entry.generated_path.empty() || entry.cellerator_path.empty() ||
            entry.generated_line == 0u || entry.cellerator_line == 0u)
            return contract_status::missing_source_map;
    return contract_status::ok;
}

}  // namespace cellerator::compiler::backend::nvcc::v1
