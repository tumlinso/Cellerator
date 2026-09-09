#pragma once
#include <Cellerator/compute/operation/native_foundation_contract.hh>
#include <cassert>
using namespace cellerator;
namespace nf = compute::operation::nf1;

execution::persistent_axis_identity axis() {
    return {{execution::biological_abi_version, execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
}
compute::operation::v2::numerical_policy numeric() {
    auto f = execution::numeric_type::f32;
    return {f, f, f, f, f, f};
}
