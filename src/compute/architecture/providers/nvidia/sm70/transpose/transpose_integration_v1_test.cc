#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_integration_v1.hh>

#include <cstdlib>

using namespace cellerator::compute::architecture::providers::nvidia::sm70;
namespace operation = cellerator::compute::operation;
namespace planner = cellerator::planner;

namespace { void require(bool value) { if (!value) std::abort(); } }

int main() {
    operation::catalog_v3::candidate_descriptor_v3 candidates[2]{};
    operation::catalog_v3::candidate_stage_v3 catalog_stages[2]{};
    planner::resource::candidate_resource_manifest_v1 manifests[2]{};
    planner::resource::planning_stage_v1 planning_stages[2]{};
    planner::resource::stage_resource_receipt_v1 receipts[2]{};
    transpose::transpose_integration_profile_v1 profile{};
    profile.device_class_id = 70u;
    profile.projection_type_id = 11u;
    profile.capability_id = 12u;
    profile.projection = {13u, 14u};
    profile.geometry = {15u, 16u};
    profile.logical_edge_count = 31u;
    profile.physical_work_items = 32u;
    profile.padded_work_items = 1u;
    profile.residual_edge_count = 15u;
    profile.owner_count = 7u;
    profile.relation_bytes = 1024u;
    profile.dense_input_bytes = 2048u;
    profile.output_bytes = 512u;
    profile.value_pack_bytes = 128u;
    profile.persistent_bytes = 4096u;
    profile.transient_bytes = 256u;
    profile.sparse_kernel_ns = 20.0;
    profile.mma_kernel_ns = 10.0;
    transpose::transpose_integration_view_v1 view{};
    require(transpose::build_transpose_integration_v1(profile,
        {candidates, catalog_stages, manifests, planning_stages, receipts, 2u},
        &view) == transpose::transpose_status_v1::success);
    require(operation::catalog_v3::validate_candidate_catalog_v3(view.catalog)
        == operation::catalog_v3::catalog_status::success);
    for (std::uint64_t index = 0u; index < view.resource_count; ++index) {
        require(static_cast<bool>(planner::resource::
            validate_candidate_resource_manifest_v1(view.resources[index])));
        planner::phase_costs costs{};
        require(static_cast<bool>(planner::resource::compute_manifest_phase_costs_v1(
            view.resources[index], &costs)));
        require(costs.kernel_ns > 0.0 && costs.persistent_bytes == 4096u);
    }
    require(view.catalog.candidates[0].identity.classification
        == operation::catalog_v3::candidate_class::experimental);
    require(view.catalog.candidates[1].identity.classification
        == operation::catalog_v3::candidate_class::production);
    return 0;
}
