#pragma once

#include "gzip_seek.hh"

namespace cellerator::rds::detail {

enum class IndexStatus {
    ok, invalid_argument, io_error, unsupported, invalid_format, invalid_index, stale_index
};

struct FileIdentity {
    uint64_t device, inode, bytes, modified_sec, modified_nsec, changed_sec, changed_nsec;
};

struct IndexHeader {
    uint64_t tag, version, index_bytes;
    FileIdentity source;
    uint64_t decoded, checkpoint_count, offsets_at, names_at;
    uint64_t row_indices, values, rows, columns, entries, name_count[2];
    uint64_t checksum;
};

struct IndexedFile {
    int fd;
    const IndexHeader *header;
    const GzipCheckpoint *checkpoints;
    const int32_t *offsets;
    const uint64_t *names[2];
};

IndexStatus create_index(const char *source, const char *path, uint64_t checkpoint_spacing_bytes, GzipStats&) noexcept;
IndexStatus open_index(const char *source, const char *path, IndexedFile *&) noexcept;
void close_index(IndexedFile *) noexcept;
const GzipCheckpoint& checkpoint_at(const IndexedFile&, uint64_t decoded) noexcept;

} // namespace cellerator::rds::detail
