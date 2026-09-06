#pragma once
#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/gradient_cover.hh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/relation_gradient.cuh>
namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient {
struct hybrid_gradient;
struct hybrid_report {
    std::uint64_t tile_count=0, residual_count=0;
    std::uint64_t persistent_bytes=0, scratch_bytes=0, preparation_byte_bound=0;
    std::uint64_t pack_refreshes=0, pack_launches=0, wmma_launches=0;
    std::uint64_t extraction_launches=0, residual_launches=0, sparse_launches=0;
};
// Borrows the pair's immutable physical-order support and whole-operand scratch.
// Owns only cover metadata, gathered panels and score scratch. Cold preparation
// validates host edges against their physical index, uploads and drains stream.
contract::status_v1 prepare_hybrid_gradient(const contract::edge_ref_v1 *host_edges,
    contract::support_view_v1 device_support, const tile_hint *hints,
    std::uint64_t hint_count, std::uint64_t byte_limit, cudaStream_t stream,
    hybrid_gradient **out, std::uint64_t scratch_byte_limit = ~std::uint64_t{0}) noexcept;
// False invokes the existing complete sparse route (either numeric profile).
// True requires half-rounded semantics and a nonempty exact rectangular cover.
// The caller supplies the same immutable support, owner stream and N16 scratch.
contract::status_v1 enqueue_hybrid_gradient(hybrid_gradient&,
    const relation_gradient_request&, bool use_hybrid) noexcept;
hybrid_report inspect_hybrid_gradient(const hybrid_gradient&) noexcept;
void destroy_hybrid_gradient(hybrid_gradient*) noexcept;
bool overlaps_hybrid_storage(const hybrid_gradient&,const void*,std::uint64_t) noexcept;
// W05 implements this bounded choice; force never changes numeric eligibility.
enum class gradient_choice { automatic, force_sparse, force_hybrid };
struct gradient_selection { bool use_hybrid=false; const char *reason=nullptr; };
contract::status_v1 select_gradient_route(bool half_rounded, gradient_choice,
    std::uint64_t tile_count, gradient_selection&) noexcept;
}
