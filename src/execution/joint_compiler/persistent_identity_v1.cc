#include <Cellerator/execution/joint_compiler/persistent_identity_v1.hh>

namespace cellerator::execution::joint_compiler {

persistent_identity_validation_result_v1 validate_persistent_identity_v1(
    persistent_identity_v1 identity) noexcept {
    if (identity.producer_namespace == 0u)
        return {
            persistent_identity_validation_code_v1::missing_producer_namespace};
    if (identity.local_identity == 0u)
        return {persistent_identity_validation_code_v1::missing_local_identity};
    return {};
}

persistent_identity_validation_result_v1
validate_persistent_identity_record_v1(
    const persistent_identity_record_v1 &record) noexcept {
    if (record.schema_version != persistent_identity_schema_version_v1)
        return {persistent_identity_validation_code_v1::unsupported_schema};
    if (record.record_bytes != sizeof(persistent_identity_record_v1))
        return {persistent_identity_validation_code_v1::invalid_record_bytes};
    return validate_persistent_identity_v1(record.identity);
}

}  // namespace cellerator::execution::joint_compiler
