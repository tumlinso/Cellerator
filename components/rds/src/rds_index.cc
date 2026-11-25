#include "rds_index.hh"
#include "dgcmatrix_rds.hh"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cellerator::rds::detail {
namespace {

constexpr uint64_t index_tag = 0x3158444953445243ULL;
constexpr uint64_t index_version = 1;
constexpr size_t index_read_bytes = 256*1024;
static_assert(sizeof(GzipCheckpoint) == 32792);
static_assert(sizeof(IndexHeader) == 176);
static_assert(sizeof(off_t) >= sizeof(int64_t) && sizeof(z_off_t) >= sizeof(int64_t));

struct Fd {
    int fd;
    ~Fd() { if (fd >= 0) close(fd); }
    int release() noexcept { int result = fd; fd = -1; return result; }
};

struct Spool {
    FILE *file = tmpfile();
    ~Spool() { if (file) fclose(file); }
};

struct TemporaryIndex {
    char *path = nullptr;
    int fd = -1;
    ~TemporaryIndex() {
        if (fd >= 0) {
            close(fd);
            unlink(path);
        }
        free(path);
    }

    bool create(const char *final_path) noexcept {
        constexpr char suffix[] = ".tmp.XXXXXX";
        size_t length = __builtin_strlen(final_path);
        if (length > SIZE_MAX - sizeof(suffix)) return false;
        path = static_cast<char*>(malloc(length + sizeof(suffix)));
        if (!path) return false;
        __builtin_memcpy(path, final_path, length);
        __builtin_memcpy(path + length, suffix, sizeof(suffix));
        fd = mkstemp(path);
        return fd >= 0;
    }
};

struct Mapping {
    void *data;
    size_t bytes;
    ~Mapping() { if (data != MAP_FAILED) munmap(data, bytes); }
    void release() noexcept { data = MAP_FAILED; }
};

bool identity(int fd, FileIdentity& id) noexcept {
    struct stat info{};
    if (fstat(fd, &info) || !S_ISREG(info.st_mode) || info.st_size < 0) return false;
    id = {uint64_t(info.st_dev), uint64_t(info.st_ino), uint64_t(info.st_size),
          uint64_t(info.st_mtim.tv_sec), uint64_t(info.st_mtim.tv_nsec),
          uint64_t(info.st_ctim.tv_sec), uint64_t(info.st_ctim.tv_nsec)};
    return true;
}

bool same_file(const FileIdentity& a, const FileIdentity& b) noexcept {
    return a.device == b.device && a.inode == b.inode && a.bytes == b.bytes &&
           a.modified_sec == b.modified_sec && a.modified_nsec == b.modified_nsec &&
           a.changed_sec == b.changed_sec && a.changed_nsec == b.changed_nsec;
}

bool write_all(int fd, const void *bytes, size_t size) noexcept {
    auto *data = static_cast<const unsigned char*>(bytes);
    while (size) {
        ssize_t n = write(fd, data, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        data += n;
        size -= size_t(n);
    }
    return true;
}

struct IndexWriter {
    int fd;
    FILE *offsets, *names;
    uint64_t checkpoints = 0, offset_count = 0, name_counts[2]{};
    uLong crc = 0;
};

bool write_payload(IndexWriter& writer, const void *data, size_t bytes) noexcept {
    if (!write_all(writer.fd, data, bytes)) return false;
    writer.crc = crc32_z(writer.crc, static_cast<const Bytef*>(data), bytes);
    return true;
}

bool copy_spool(FILE *from, IndexWriter& writer, uint64_t bytes) noexcept {
    if (!bytes) return true;
    if (fseek(from, 0, SEEK_SET)) return false;
    unsigned char buffer[64*1024];
    while (bytes) {
        size_t count = bytes < sizeof(buffer) ? size_t(bytes) : sizeof(buffer);
        if (fread(buffer, 1, count, from) != count || !write_payload(writer, buffer, count)) return false;
        bytes -= count;
    }
    return true;
}

bool write_checkpoint(void *opaque, const GzipCheckpoint& checkpoint) noexcept {
    auto& writer = *static_cast<IndexWriter*>(opaque);
    if (!write_payload(writer, &checkpoint, sizeof(checkpoint))) return false;
    ++writer.checkpoints;
    return true;
}

bool spool_offsets(void *opaque, const int32_t *values, size_t count) noexcept {
    auto& writer = *static_cast<IndexWriter*>(opaque);
    if (fwrite(values, sizeof(*values), count, writer.offsets) != count) return false;
    writer.offset_count += count;
    return true;
}

bool spool_name(void *opaque, unsigned axis, uint64_t offset) noexcept {
    auto& writer = *static_cast<IndexWriter*>(opaque);
    if (fwrite(&offset, sizeof(offset), 1, writer.names) != 1) return false;
    ++writer.name_counts[axis];
    return true;
}

bool scan_take(void *opaque, void *out, size_t bytes) noexcept {
    size_t got;
    return static_cast<GzipScan*>(opaque)->read(out, bytes, got) && got == bytes;
}

IndexStatus scan_status(XdrStatus status) noexcept {
    if (status == XdrStatus::unsupported) return IndexStatus::unsupported;
    if (status == XdrStatus::io_error) return IndexStatus::io_error;
    return IndexStatus::invalid_format;
}

uint64_t aligned_names(uint64_t offsets_at, uint64_t offset_count) noexcept {
    return (offsets_at + offset_count*sizeof(int32_t) + 7) & ~uint64_t(7);
}

uLong header_checksum(const IndexHeader& h) noexcept {
    return crc32_z(0, reinterpret_cast<const Bytef*>(&h), offsetof(IndexHeader, checksum));
}

bool read_at(int fd, uint64_t at, void *out, size_t bytes) noexcept {
    auto *data = static_cast<unsigned char*>(out);
    while (bytes) {
        ssize_t n = pread(fd, data, bytes, off_t(at));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        at += uint64_t(n);
        data += n;
        bytes -= size_t(n);
    }
    return true;
}

bool read_payload(int fd, uint64_t& at, uLong& crc, void *out, size_t bytes) noexcept {
    if (!read_at(fd, at, out, bytes)) return false;
    crc = crc32_z(crc, static_cast<Bytef*>(out), bytes);
    at += bytes;
    return true;
}

bool valid_layout(const IndexHeader& h, uint64_t bytes) noexcept {
    if (h.tag != index_tag || h.version != index_version || h.index_bytes != bytes ||
        !h.checkpoint_count || h.checkpoint_count > (bytes - sizeof(h))/sizeof(GzipCheckpoint) ||
        h.rows > INT32_MAX || h.columns > INT32_MAX || h.entries > INT32_MAX) return false;
    if (h.offsets_at != sizeof(h) + h.checkpoint_count*sizeof(GzipCheckpoint)) return false;
    const uint64_t offsets_size = (h.columns + 1)*sizeof(int32_t);
    if (offsets_size > bytes - h.offsets_at ||
        h.names_at != aligned_names(h.offsets_at, h.columns + 1) || h.names_at > bytes)
        return false;

    const uint64_t extents[] = {h.rows, h.columns};
    for (unsigned axis = 0; axis < 2; ++axis) {
        uint64_t maximum = (extents[axis] + name_checkpoint_stride - 1)/name_checkpoint_stride;
        if (h.name_count[axis] && h.name_count[axis] != maximum) return false;
    }
    if (bytes - h.names_at != (h.name_count[0] + h.name_count[1])*sizeof(uint64_t) ||
        h.row_indices > h.decoded || h.values > h.decoded ||
        h.entries*sizeof(int32_t) > h.decoded - h.row_indices ||
        h.entries*sizeof(double) > h.decoded - h.values) return false;

    return true;
}

IndexStatus validate_payload(int fd, const IndexHeader& h) noexcept {
    uint64_t at = sizeof(h);
    uLong crc = header_checksum(h);
    {
        GzipCheckpoint points[index_read_bytes/sizeof(GzipCheckpoint)];
        constexpr size_t capacity = sizeof(points)/sizeof(*points);
        uint64_t decoded = 0, compressed = 0;
        for (uint64_t i = 0; i < h.checkpoint_count;) {
            uint64_t left = h.checkpoint_count - i;
            size_t n = left < capacity ? size_t(left) : capacity;
            if (!read_payload(fd, at, crc, points, n*sizeof(*points))) return IndexStatus::io_error;
            for (size_t j = 0; j < n; ++j, ++i) {
                const auto& point = points[j];
                if (point.decoded > h.decoded || point.compressed >= h.source.bytes ||
                    point.bits > 7 || (point.bits && !point.compressed) ||
                    point.history_size > gzip_history_bytes || point.history_size > point.decoded ||
                    (i ? point.decoded <= decoded || point.compressed < compressed : point.decoded != 0))
                    return IndexStatus::invalid_index;
                decoded = point.decoded;
                compressed = point.compressed;
            }
        }
    }
    {
        int32_t offsets[index_read_bytes/sizeof(int32_t)];
        constexpr size_t capacity = sizeof(offsets)/sizeof(*offsets);
        int32_t previous = 0;
        const uint64_t count = h.columns + 1;
        for (uint64_t i = 0; i < count;) {
            uint64_t left = count - i;
            size_t n = left < capacity ? size_t(left) : capacity;
            if (!read_payload(fd, at, crc, offsets, n*sizeof(*offsets))) return IndexStatus::io_error;
            if (!i && offsets[0] != 0) return IndexStatus::invalid_index;
            bool ordered = offsets[0] >= previous;
            for (size_t j = 1; j < n; ++j) ordered &= offsets[j] >= offsets[j-1];
            if (!ordered) return IndexStatus::invalid_index;
            previous = offsets[n-1];
            i += n;
        }
        if (previous != int32_t(h.entries)) return IndexStatus::invalid_index;
        int32_t padding;
        if (at != h.names_at && !read_payload(fd, at, crc, &padding, sizeof(padding)))
            return IndexStatus::io_error;
    }
    {
        uint64_t names[index_read_bytes/sizeof(uint64_t)];
        constexpr size_t capacity = sizeof(names)/sizeof(*names);
        for (unsigned axis = 0; axis < 2; ++axis) {
            uint64_t previous = 0;
            for (uint64_t i = 0; i < h.name_count[axis];) {
                uint64_t left = h.name_count[axis] - i;
                size_t n = left < capacity ? size_t(left) : capacity;
                if (!read_payload(fd, at, crc, names, n*sizeof(*names))) return IndexStatus::io_error;
                for (size_t j = 0; j < n; ++j, ++i) {
                    if (names[j] >= h.decoded || (i && names[j] <= previous)) return IndexStatus::invalid_index;
                    previous = names[j];
                }
            }
        }
    }
    return crc == h.checksum ? IndexStatus::ok : IndexStatus::invalid_index;
}

} // namespace

IndexStatus create_index(const char *source, const char *path, uint64_t checkpoint_spacing_bytes,
                         GzipStats& stats) noexcept {
    stats = {};
    if (!source || !*source || !path || !*path || !checkpoint_spacing_bytes)
        return IndexStatus::invalid_argument;

    Fd input{open(source, O_RDONLY | O_CLOEXEC | O_NONBLOCK)};
    IndexHeader header{};
    if (input.fd < 0 || !identity(input.fd, header.source)) return IndexStatus::io_error;
    if (header.source.bytes < 2) return IndexStatus::invalid_format;
    unsigned char magic[2];
    if (pread(input.fd, magic, sizeof(magic), 0) != sizeof(magic)) return IndexStatus::io_error;
    if (magic[0] != 0x1f || magic[1] != 0x8b) return IndexStatus::unsupported;

    TemporaryIndex output;
    if (!output.create(path) || !write_all(output.fd, &header, sizeof(header)))
        return IndexStatus::io_error;
    Spool offsets, names;
    if (!offsets.file || !names.file) return IndexStatus::io_error;
    IndexWriter writer{output.fd, offsets.file, names.file};

    GzipScan gzip;
    if (!gzip.initialize(input.fd, checkpoint_spacing_bytes, write_checkpoint, &writer))
        return IndexStatus::io_error;
    XdrInput xdr{&gzip, scan_take, 0};
    DgcMatrixLayout layout{};
    auto status = scan_dgcmatrix(xdr, {&writer, spool_offsets, spool_name}, layout);
    if (status != XdrStatus::ok) {
        stats = gzip.stats();
        return scan_status(status);
    }
    bool complete = gzip.finish();
    stats = gzip.stats();
    if (!complete || !writer.checkpoints || writer.offset_count != uint64_t(layout.columns) + 1)
        return IndexStatus::invalid_format;

    FileIdentity current{};
    if (!identity(input.fd, current)) return IndexStatus::io_error;
    if (!same_file(header.source, current)) return IndexStatus::stale_index;
    header.tag = index_tag;
    header.version = index_version;
    header.decoded = xdr.position;
    header.checkpoint_count = writer.checkpoints;
    header.offsets_at = sizeof(header) + writer.checkpoints*sizeof(GzipCheckpoint);
    header.names_at = aligned_names(header.offsets_at, writer.offset_count);
    header.row_indices = layout.row_index_offset;
    header.values = layout.value_offset;
    header.rows = layout.rows;
    header.columns = layout.columns;
    header.entries = layout.entries;
    header.name_count[0] = writer.name_counts[0];
    header.name_count[1] = writer.name_counts[1];
    uint64_t offsets_bytes = writer.offset_count*sizeof(int32_t);
    uint64_t names_bytes = (writer.name_counts[0] + writer.name_counts[1])*sizeof(uint64_t);
    header.index_bytes = header.names_at + names_bytes;
    const uint64_t padding = 0;
    if (header.index_bytes > SIZE_MAX || header.index_bytes > INT64_MAX ||
        !copy_spool(offsets.file, writer, offsets_bytes) ||
        !write_payload(writer, &padding, size_t(header.names_at - header.offsets_at - offsets_bytes)) ||
        !copy_spool(names.file, writer, names_bytes))
        return IndexStatus::io_error;

    header.checksum = crc32_combine(header_checksum(header), writer.crc, z_off_t(header.index_bytes - sizeof(header)));
    if (lseek(output.fd, 0, SEEK_SET) != 0 || !write_all(output.fd, &header, sizeof(header)) ||
        fsync(output.fd) || link(output.path, path)) return IndexStatus::io_error;
    return IndexStatus::ok;
}

IndexStatus open_index(const char *source, const char *path, IndexedFile *& result) noexcept {
    if (!source || !*source || !path || !*path) return IndexStatus::invalid_argument;
    Fd input{open(source, O_RDONLY | O_CLOEXEC | O_NONBLOCK)};
    Fd index{open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)};
    FileIdentity source_id{}, index_id{};
    if (input.fd < 0 || index.fd < 0 || !identity(input.fd, source_id) ||
        !identity(index.fd, index_id)) return IndexStatus::io_error;
    if (index_id.bytes < sizeof(IndexHeader) || index_id.bytes > SIZE_MAX)
        return IndexStatus::invalid_index;

    IndexHeader checked;
    if (!read_at(index.fd, 0, &checked, sizeof(checked))) return IndexStatus::io_error;
    if (!valid_layout(checked, index_id.bytes)) return IndexStatus::invalid_index;
    if (!same_file(checked.source, source_id)) return IndexStatus::stale_index;
    auto status = validate_payload(index.fd, checked);
    if (status != IndexStatus::ok) return status;
    FileIdentity current{};
    if (!identity(index.fd, current)) return IndexStatus::io_error;
    if (!same_file(index_id, current)) return IndexStatus::invalid_index;

    const size_t size = size_t(index_id.bytes);
    Mapping mapping{mmap(nullptr, size, PROT_READ, MAP_PRIVATE, index.fd, 0), size};
    if (mapping.data == MAP_FAILED) return IndexStatus::io_error;
    const auto *header = static_cast<const IndexHeader*>(mapping.data);
    const auto *bytes = static_cast<const unsigned char*>(mapping.data);
    const auto *checkpoints = reinterpret_cast<const GzipCheckpoint*>(bytes + sizeof(*header));
    const auto *offsets = reinterpret_cast<const int32_t*>(bytes + checked.offsets_at);
    const auto *names = reinterpret_cast<const uint64_t*>(bytes + checked.names_at);

    auto *file = static_cast<IndexedFile*>(malloc(sizeof(IndexedFile)));
    if (!file) return IndexStatus::io_error;
    *file = {input.release(), header, checkpoints, offsets, {names, names + checked.name_count[0]}};
    mapping.release();
    result = file;
    return IndexStatus::ok;
}

void close_index(IndexedFile *file) noexcept {
    if (!file) return;
    munmap(const_cast<IndexHeader*>(file->header), size_t(file->header->index_bytes));
    close(file->fd);
    free(file);
}

const GzipCheckpoint& checkpoint_at(const IndexedFile& file, uint64_t decoded) noexcept {
    uint64_t first = 0, end = file.header->checkpoint_count;
    while (end - first > 1) {
        uint64_t middle = first + (end - first)/2;
        if (file.checkpoints[middle].decoded <= decoded) first = middle;
        else end = middle;
    }
    return file.checkpoints[first];
}

} // namespace cellerator::rds::detail
