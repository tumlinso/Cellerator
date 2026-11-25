#pragma once

#include <Cellerator/biology/gene_annotation.hh>

namespace cellerator::biology {

struct GeneSubset {
    const char *name;
    const uint32_t *indices; // positions in catalog.records, in dictionary order
    size_t count;
};

struct BuiltinGeneCatalog {
    const char *assembly;
    const char *release;
    const char *source;
    GeneCatalog catalog; // sorted by exact canonical gene ID
    const GeneSubset *subsets;
    size_t subset_count;
};

// Static storage for the lifetime of the process. Null when no release was built.
const BuiltinGeneCatalog *builtin_gene_catalog() noexcept;

} // namespace cellerator::biology
