#pragma once

#include <stddef.h>
#include <stdint.h>

namespace cellerator::biology {

enum class GeneAnnotationFormat : uint8_t { gtf, gff3 };

struct AnnotationText {
    const char *data;
    size_t size;
};

struct GeneAnnotationRecord {
    AnnotationText id, symbol;
    const AnnotationText *aliases;
    size_t alias_count;
};

struct GeneCatalog {
    const GeneAnnotationRecord *records;
    size_t count;
};

struct GeneAnnotationError {
    const char *message = nullptr;
    size_t line = 0;
    bool io = false;
};

class GeneAnnotation {
    void *storage_ = nullptr;
    const GeneAnnotationRecord *records_ = nullptr;
    size_t count_ = 0, capacity_ = 0;

    friend GeneAnnotationError read_gene_annotation(
        const char *, GeneAnnotationFormat, GeneAnnotation &, const char *) noexcept;

public:
    GeneAnnotation() = default;
    ~GeneAnnotation();
    GeneAnnotation(const GeneAnnotation &) = delete;
    GeneAnnotation &operator=(const GeneAnnotation &) = delete;

    const GeneAnnotationRecord *records() const noexcept { return records_; }
    size_t count() const noexcept { return count_; }
    // Borrowed until this annotation is replaced or destroyed.
    GeneCatalog catalog() const noexcept { return {records_, count_}; }
};

// Load a regular text file and grow catalog storage when necessary. Success
// replaces the catalog. Existing record pointers remain valid on failure.
GeneAnnotationError read_gene_annotation(const char *path, GeneAnnotationFormat format,
    GeneAnnotation &out, const char *feature_type = "gene") noexcept;

} // namespace cellerator::biology
