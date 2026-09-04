#include <Cellerator/compiler/backend/freeze_the_cpu_backend_thin_waist_v1.hh>

namespace cellerator::compiler::backend::v1 {

const cpu_backend_thin_waist_receipt_v1&
freeze_cpu_backend_thin_waist_v1() noexcept {
    static constexpr cpu_backend_thin_waist_receipt_v1 receipt{
        backend_thin_waist_version_v1,
        cpu::v1::cpu_backend_contract_version_v1,
        true, true, true, true, true};
    return receipt;
}

}  // namespace cellerator::compiler::backend::v1
