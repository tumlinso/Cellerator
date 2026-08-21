#include <Baseplane/seq/dna2_views.hh>
#include <Cellerator/execution/biological_abi.hh>

#include <type_traits>

namespace bp = baseplane::seq;
namespace ce = cellerator::execution;

ce::bit_plane_view bind_planes(
    bp::dna2_planes32_stream_view source,
    const std::uint32_t *validity,
    ce::axis_identity coordinate_axis,
    std::uint32_t base_count) {
    return ce::bit_plane_view{
        coordinate_axis,
        source.lo_words,
        source.hi_words,
        validity,
        {ce::residency_kind::host, {}, -1, 0u},
        static_cast<std::uint32_t>(source.n_words),
        base_count};
}

int main() {
    static_assert(std::is_trivially_copyable<bp::dna2_planes32_stream_view>::value,
        "Baseplane plane view must remain POD");
    static_assert(std::is_trivially_copyable<ce::bit_plane_view>::value,
        "Cellerator bit plane view must remain POD");
    return 0;
}
