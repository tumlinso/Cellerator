#include <Cellerator/biology/gene_catalog.hh>

namespace cellerator::biology {

#ifdef CELLERATOR_HAS_BUILTIN_GENE_CATALOG
namespace {
#include "gene_catalog.gen.inc"
}
#endif

const BuiltinGeneCatalog *builtin_gene_catalog() noexcept {
#ifdef CELLERATOR_HAS_BUILTIN_GENE_CATALOG
    return &builtin_catalog;
#else
    return nullptr;
#endif
}

} // namespace cellerator::biology
