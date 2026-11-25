#include <string_view>

#ifdef CELLERATOR_ANNOTATION_TEST_MODULE
import cellerator.annotation;
#else
#include <Cellerator/biology/gene_catalog.hh>
#endif

namespace bio = cellerator::biology;
using namespace std::literals;

bool equal(bio::AnnotationText value, std::string_view text) {
    return value.size == text.size() &&
        (!value.size || std::string_view(value.data, value.size) == text);
}

int main() {
    const auto *reference = bio::builtin_gene_catalog();
    if (!reference || reference != bio::builtin_gene_catalog()) return 1;
    if (reference->assembly != "fixture-assembly"sv ||
        reference->release != "fixture-v1"sv || reference->source != "reference.gff3"sv) return 2;
    const auto [genes, count] = reference->catalog;
    if (count != 2 || !equal(genes[0].id, "gene:A.1") || !equal(genes[1].id, "gene:B.2") ||
        !equal(genes[0].symbol, "A\"quote")) return 3;
    if (genes[1].alias_count != 2 || !equal(genes[1].aliases[0], "alt,B") ||
        !equal(genes[1].aliases[1], "shared")) return 4;
    if (reference->subset_count != 2) return 5;
    const auto &reversed = reference->subsets[0], &single = reference->subsets[1];
    if (reversed.name != "reversed"sv || reversed.count != 2 ||
        reversed.indices[0] != 1 || reversed.indices[1] != 0) return 6;
    if (single.name != "single"sv || single.count != 1 || single.indices[0] != 0) return 7;
    return 0;
}
