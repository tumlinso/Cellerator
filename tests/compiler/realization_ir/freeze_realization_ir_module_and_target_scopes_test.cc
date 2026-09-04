#include <Cellerator/compiler/ir/realization/freeze_realization_ir_module_and_target_scopes_v1.hh>

#include <array>
#include <cassert>
#include <string>

using namespace cellerator::compiler::ir::realization::v1;

int main() {
    const target_scope_v1 sm70{{1u, 1u}, "cuda.sm70", "pbmc.forward"};
    const target_scope_v1 host{{1u, 2u}, "host.x86_64", "pbmc.reference"};
    const realization_lineage_v1 lineage{{2u, 1u}, {2u, 2u}, {2u, 3u}};

    realization_module_v1 module;
    module.identity = {3u, 1u};
    module.name = "pbmc_relation_apply";
    module.targets = {sm70, host};
    module.objects = {
        {{4u, 1u}, sm70.identity, realization_object_kind_v1::kernel,
            "relation_apply_sm70", lineage},
        {{4u, 2u}, sm70.identity, realization_object_kind_v1::host_stub,
            "launch_relation_apply_sm70", lineage},
        {{4u, 3u}, sm70.identity, realization_object_kind_v1::data_artifact,
            "packed_projection", lineage},
        {{4u, 4u}, sm70.identity, realization_object_kind_v1::stage,
            "forward", lineage},
        {{4u, 5u}, sm70.identity, realization_object_kind_v1::binding,
            "mutable_values", lineage},
        {{4u, 6u}, sm70.identity, realization_object_kind_v1::native_fragment,
            "row_masked_tile", lineage},
        {{4u, 7u}, host.identity, realization_object_kind_v1::function,
            "relation_apply_reference", lineage},
    };

    std::string error;
    assert(validate_realization_module_v1(module, &error) ==
        realization_module_status_v1::valid);
    assert(error.empty());

    // The owning representation round-trips through ordinary C++ value
    // semantics without runtime objects or pointer identity.
    const realization_module_v1 copied = module;
    assert(equivalent_realization_module_v1(module, copied));

    auto invalid = module;
    invalid.objects.front().target_scope = {9u, 9u};
    assert(validate_realization_module_v1(invalid) ==
        realization_module_status_v1::unknown_target);

    invalid = module;
    invalid.targets.push_back(sm70);
    assert(validate_realization_module_v1(invalid) ==
        realization_module_status_v1::duplicate_target);

    invalid = module;
    invalid.objects.front().lineage.planning_identity = {};
    assert(validate_realization_module_v1(invalid) ==
        realization_module_status_v1::missing_lineage);
}
