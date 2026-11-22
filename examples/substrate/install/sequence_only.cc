#include <Cellerator/layout/sequence/dna2.hh>
#include <Cellerator/ops/sequence/dna2.hh>

namespace layout = cellerator::layout::sequence;
namespace seq = cellerator::ops::sequence;

int main() {
    cellerator::tensor::sequence::PackedView packed{};
    uint64_t words[1]{}, validity[1]{};
    if (layout::pack_ascii("ACCAC", 5, words, 1, validity, 1, &packed)
        != layout::Status::ok) return 1;

    seq::motif32_exact motif{};
    if (!seq::dna2_normalize_motif32_exact(4, 2, 0, 1, &motif)) return 2;
    uint64_t count = 0;
    if (seq::count_motif(packed, motif, &count) != seq::Status::ok) return 3;
    return count == 2 ? 0 : 4;
}
