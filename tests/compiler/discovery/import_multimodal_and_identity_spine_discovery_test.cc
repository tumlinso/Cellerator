#include <Cellerator/compiler/discovery/import_multimodal_and_identity_spine_discovery_v1.hh>

#include <cassert>
#include <vector>

using namespace Cellerator::compiler::discovery;

namespace {

persistent_atom_identity_v1 id(std::uint64_t value) {
    return {1, value};
}

}  // namespace

int main() {
    multimodal_identity_spine_v1 spine{
        id(1), id(2), id(3), id(4), id(5), 7,
        {
            {id(10), id(3), id(4), id(5), id(20), id(21), id(22), {}, 2,
             modality_kind_v1::transcriptome},
            {id(11), id(6), id(7), id(8), id(30), id(31), id(32), id(9), 4,
             modality_kind_v1::chromatin},
        },
    };
    std::vector<modality_overlay_v1> overlays{
        {id(10), id(20), id(21), id(22), id(40), id(50), 2},
        {id(11), id(30), id(31), id(32), id(41), id(51), 4},
    };
    cross_modal_relation_proposal_v1 first{
        id(101), id(70), id(80), id(10), id(20), id(21), id(90),
        id(11), id(30), id(31), id(91), 3, 4, false};
    auto second = first;
    second.proposal_identity = id(100);
    second.source_entity_identity = id(92);
    std::vector<cross_modal_relation_proposal_v1> proposals;
    assert(discover_multimodal_identity_spine_v1(
               spine, overlays, {first, second}, &proposals) ==
           multimodal_discovery_status_v1::success);
    assert(proposals.size() == 2);
    assert(proposals[0].proposal_identity == id(100));
    assert(!authorizes_execution(proposals[0]));

    auto shape_only = spine;
    shape_only.modalities[0].observation_order_identity = id(500);
    assert(discover_multimodal_identity_spine_v1(
               shape_only, overlays, {}, &proposals) ==
           multimodal_discovery_status_v1::subject_identity_mismatch);

    auto wrong_axis = first;
    wrong_axis.destination_axis_identity = id(999);
    assert(discover_multimodal_identity_spine_v1(
               spine, overlays, {wrong_axis}, &proposals) ==
           multimodal_discovery_status_v1::endpoint_identity_mismatch);

    auto stale = overlays;
    stale[1].value_generation = 3;
    assert(discover_multimodal_identity_spine_v1(spine, stale, {}, &proposals) ==
           multimodal_discovery_status_v1::stale_value_generation);

    second.proposal_identity = first.proposal_identity;
    assert(discover_multimodal_identity_spine_v1(
               spine, overlays, {first, second}, &proposals) ==
           multimodal_discovery_status_v1::duplicate_proposal);
}
