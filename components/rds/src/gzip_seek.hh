#pragma once

#include <stddef.h>
#include <stdint.h>
#include <zlib.h>

namespace cellerator::rds::detail {

inline constexpr uint32_t gzip_history_bytes = 32768;
inline constexpr size_t gzip_io_bytes = 256*1024;

struct GzipCheckpoint {
    uint64_t decoded, compressed;
    uint32_t bits, history_size;
    unsigned char history[gzip_history_bytes];
};

struct GzipStats {
    uint64_t compressed_bytes = 0, decoded_bytes = 0;
};

using EmitGzipCheckpoint = bool (*)(void*, const GzipCheckpoint&) noexcept;

class GzipScan {
public:
    GzipScan() noexcept = default;
    ~GzipScan() noexcept;
    GzipScan(const GzipScan&) = delete;
    GzipScan& operator=(const GzipScan&) = delete;

    // The descriptor is borrowed. The scan accepts exactly one gzip member.
    bool initialize(int fd, uint64_t checkpoint_spacing_bytes, EmitGzipCheckpoint emit, void *context) noexcept;
    // A null destination skips bytes. A short successful read denotes verified EOF.
    bool read(void *out, size_t count, size_t &got) noexcept;
    // Check that the caller consumed precisely the entire decoded member.
    bool finish() noexcept;
    GzipStats stats() const noexcept { return stats_; }

private:
    bool fill() noexcept;

    z_stream stream_{};
    int fd_ = -1;
    EmitGzipCheckpoint emit_ = nullptr;
    void *context_ = nullptr;
    uint64_t input_end_ = 0, last_checkpoint_ = 0, checkpoint_spacing_bytes_ = 0;
    GzipStats stats_{};
    size_t next_ = 0, size_ = 0;
    bool indexed_ = false, ended_ = false, failed_ = false;
    unsigned char input_[gzip_io_bytes], output_[gzip_io_bytes];
};

class GzipCursor {
public:
    GzipCursor() noexcept = default;
    ~GzipCursor() noexcept;
    GzipCursor(const GzipCursor&) = delete;
    GzipCursor& operator=(const GzipCursor&) = delete;

    // The descriptor and discard storage are borrowed. A worker uses cursors serially.
    bool initialize(int fd, unsigned char (&discard)[gzip_io_bytes]) noexcept;
    void rebind(int fd) noexcept { fd_ = fd; started_ = false; stats_ = {}; }
    bool read_at(const GzipCheckpoint&, uint64_t offset, void *out, size_t count) noexcept;
    GzipStats stats() const noexcept { return stats_; }

private:
    bool restart(const GzipCheckpoint&) noexcept;
    bool decode(unsigned char *out, size_t count) noexcept;

    z_stream stream_{};
    int fd_ = -1;
    uint64_t input_end_ = 0, decoded_ = 0;
    GzipStats stats_{};
    bool started_ = false, ended_ = false;
    unsigned char *discard_ = nullptr;
    unsigned char input_[gzip_io_bytes];
};

} // namespace cellerator::rds::detail
