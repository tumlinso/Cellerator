#include <Cellerator/compiler/ir/realization/implement_projection_contracts_v1.hh>

#include <cassert>

using namespace cellerator::compiler::ir::realization::v1;

int main() {
    cellpack::persistence::execution_projection_entry_v1 entry{};
    entry.identity_low = 7u;
    entry.identity_high = 8u;
    entry.kind = cellpack::persistence::execution_projection_kind::native_row_masked;
    entry.schema_version = 1u;
    entry.flags = cellpack::persistence::projection_forward_capable |
        cellpack::persistence::projection_transpose_capable;
    auto projection = import_cpe2_projection_v1(entry, {1u, 1u}, {1u, 2u});
    projection.forward_value_map = {{0u, 2u}, {1u, 0u}, {2u, 1u}};
    projection.transpose_value_map = {{0u, 1u}, {1u, 2u}, {2u, 0u}};
    assert(projection.kind == projection_kind_v1::row_masked);
    assert(validate_projection_contract_v1(projection) == projection_status_v1::valid);

    projection.capability_identity = {9u, 1u};
    target_capability_v1 capability;
    capability.identity = {9u, 1u};
    capability.memory_interfaces = memory_device_global_v1;
    capability.numeric_support = numeric_f32_v1;
    capability.toolchain = "cuda";
    capability.runtime = "cuda";
    capability.backend = "native";
    assert(validate_projection_contract_v1(projection, &capability) ==
        projection_status_v1::valid);
    capability.identity = {9u, 2u};
    assert(validate_projection_contract_v1(projection, &capability) ==
        projection_status_v1::capability_mismatch);
}
