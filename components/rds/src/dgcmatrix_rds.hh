#pragma once

#include <stddef.h>
#include <stdint.h>

namespace cellerator::rds::detail {

struct XdrInput {
    void *state;
    // Take exactly count bytes. A null destination discards them.
    bool (*take)(void *, void *destination, size_t count) noexcept;
    uint64_t position;
};

enum class XdrStatus { ok, io_error, unsupported, invalid };
enum class XdrNameEncoding { native, utf8, latin1, bytes };

struct XdrStringHeader {
    int32_t length; // -1 is NA, zero is an empty string
    XdrNameEncoding encoding;
};

struct DgcMatrixLayout {
    uint64_t row_index_offset, value_offset;
    uint32_t rows, columns, entries;
    bool has_names[2];
};

inline constexpr unsigned name_checkpoint_stride = 256;

struct DgcMatrixSink {
    void *state;
    bool (*column_offsets)(void *, const int32_t *, size_t count) noexcept;
    // Positions name the start of the serialized CHARSXP, in source order.
    bool (*name_checkpoint)(void *, unsigned axis, uint64_t position) noexcept;
};

XdrStatus read_xdr_bytes(XdrInput &, void *destination, size_t count) noexcept;
XdrStatus read_xdr_string_header(XdrInput &, XdrStringHeader &) noexcept;
// Leaves the input after the object. The gzip layer verifies end of input and checksum.
XdrStatus scan_dgcmatrix(XdrInput &, const DgcMatrixSink &, DgcMatrixLayout &) noexcept;

} // namespace cellerator::rds::detail
