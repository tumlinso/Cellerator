// Host adapter test: include the owning TU to inspect its deliberately private seam.
#include "../../../src/compute/operation/prepared_relation.cu"
#include "test_require.hh"
#include <iostream>
namespace ce = cellerator::compute::relation;
namespace ex = cellerator::execution;
int main() {
    ce::operation_descriptor op{};
    op.topology.identity={0xfedcba9876543210ULL,0x123456789abcdef0ULL};
    op.topology.epoch={0x100000001ULL};
    op.topology.logical_edge_order={0x8123456789abcdefULL,0xfedcba9876543210ULL};
    op.topology.edge_count=7;
    auto axis=[](unsigned base) {
        ce::axis_descriptor a{}; a.extent=base;
        a.identity.header={ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(a.identity)};
        a.identity.domain={base,0x100000000ULL+base}; a.identity.order={base+1,0x200000000ULL+base};
        a.identity.geometry={base+2,0x300000000ULL+base}; a.identity.partition={base+3,0x400000000ULL+base};
        return a;
    };
    op.topology.source=axis(4);op.topology.destination=axis(9);
    for(auto direction:{ce::orientation::forward,ce::orientation::transpose}) {
        op.direction=direction;ce::native_contract contract{};
        SPINE_REQUIRE(ce::adapt(op,contract));SPINE_REQUIRE(ce::equivalent(op,contract.semantic));
        SPINE_REQUIRE(ex::same_identity(contract.structures.structures[0].persistent,op.topology.identity));
        SPINE_REQUIRE(contract.structures.structures[0].epoch.value==op.topology.epoch.value);
        SPINE_REQUIRE(contract.numeric.sparse_storage==ex::numeric_type::f16);
        SPINE_REQUIRE(contract.numeric.dense_storage==ex::numeric_type::f32);
        SPINE_REQUIRE(!ex::same_axis_identity(contract.source,contract.destination));
    }
    ce::native_contract out{};
    op.dense_width=2;SPINE_REQUIRE(ce::validate(op));SPINE_REQUIRE(ce::adapt(op,out).code==ce::status_code::unsupported_width);
    op.dense_width=1;op.arithmetic.permit_fma=false;SPINE_REQUIRE(ce::adapt(op,out).code==ce::status_code::unsupported_numeric_policy);
    op.arithmetic.permit_fma=true;op.update=ce::output_update::accumulate;
    SPINE_REQUIRE(ce::validate(op));SPINE_REQUIRE(ce::adapt(op,out).code==ce::status_code::unsupported_semantics);
    std::cout<<"adapter identities, orientation, numeric separation and capability rejection passed\n";
}
