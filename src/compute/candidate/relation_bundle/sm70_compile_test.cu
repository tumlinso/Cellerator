#include "Cellerator/compute/operation/relation_bundle/catalog.hh"
#include "Cellerator/compute/operation/relation_bundle/moments.hh"
#include "Cellerator/compute/operation/relation_chain/candidates.hh"
#include "Cellerator/compute/operation/relation_chain/hierarchy.hh"

extern "C" __global__ void cellerator_relation_bundle_stage_identity_sm70(
    unsigned long long* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[0] = cellerator::compute::relation_bundle::candidate_catalog[0].stage_id;
    }
}

extern "C" __global__ void cellerator_relation_chain_stage_identity_sm70(
    unsigned long long* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[0] = cellerator::compute::relation_bundle::candidate_catalog[3].stage_id;
    }
}
