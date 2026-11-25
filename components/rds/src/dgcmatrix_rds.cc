#include "dgcmatrix_rds.hh"

namespace cellerator::rds::detail {

XdrStatus read_xdr_bytes(XdrInput &in, void *out, size_t n) noexcept {
    if (n > UINT64_MAX - in.position) return XdrStatus::invalid;
    if (n && !in.take(in.state, out, n)) return XdrStatus::io_error;
    in.position += n;
    return XdrStatus::ok;
}

namespace {

using S = XdrStatus;

// Codes in R's XDR serialization stream, not the in-memory SEXPREC layout.
enum : uint32_t {
    symbol = 1, pairlist = 2, chars = 9, integer = 13, real = 14, strings = 16,
    list = 19, s4 = 25, nil = 254, reference = 255, type_mask = 255,
    object = 1u << 8, has_attributes = 1u << 9, has_tag = 1u << 10,
    bytes = 1u << 13, latin1 = 1u << 14, utf8 = 1u << 15,
    s4_object = 1u << 16, ascii = 1u << 18
};

uint32_t from_xdr(uint32_t word) noexcept {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap32(word);
#else
    return word;
#endif
}

S read_words(XdrInput &in, uint32_t *words, size_t n) noexcept {
    auto status = read_xdr_bytes(in, words, n*sizeof(*words));
    if (status == S::ok)
        for (size_t i = 0; i < n; ++i) words[i] = from_xdr(words[i]);
    return status;
}

S skip(XdrInput &in, uint64_t n) noexcept {
    while (n) {
        size_t count = n > SIZE_MAX ? SIZE_MAX : size_t(n);
        auto status = read_xdr_bytes(in, nullptr, count);
        if (status != S::ok) return status;
        n -= count;
    }
    return S::ok;
}

S read_length(XdrInput &in, uint32_t first, uint32_t &n) noexcept {
    if (first <= INT32_MAX) {
        n = first;
        return S::ok;
    }
    if (first != UINT32_MAX) return S::invalid;
    uint32_t parts[2];
    auto status = read_words(in, parts, 2);
    if (status != S::ok) return status;
    if (parts[0]) return S::unsupported;
    n = parts[1];
    return S::ok;
}

S read_vector(XdrInput &in, uint32_t expected, uint32_t &n) noexcept {
    uint32_t header[2];
    auto status = read_words(in, header, 2);
    if (status != S::ok) return status;
    if (header[0] != expected) return S::unsupported;
    return read_length(in, header[1], n);
}

enum class Slot : unsigned { i, p, dim, dimnames, x, factors, class_, package, end };

struct Symbols {
    Slot slots[8];
    unsigned count;
};

constexpr const char *slot_names[] = {
    "i", "p", "Dim", "Dimnames", "x", "factors", "class", "package"
};
constexpr int32_t slot_lengths[] = {1, 1, 3, 8, 1, 7, 5, 7};

S read_tagged_slot(XdrInput &in, Symbols &refs, Slot &slot) noexcept {
    uint32_t flags;
    auto status = read_words(in, &flags, 1);
    if (status != S::ok) return status;
    if (flags == nil) {
        slot = Slot::end;
        return S::ok;
    }
    if (flags != (pairlist | has_tag)) return S::unsupported;
    status = read_words(in, &flags, 1);
    if (status != S::ok) return status;
    if ((flags & type_mask) == reference) {
        uint32_t index = flags >> 8;
        if (!index) {
            status = read_words(in, &index, 1);
            if (status != S::ok) return status;
        }
        if (!index || index > refs.count) return S::invalid;
        slot = refs.slots[index - 1];
        return S::ok;
    }
    if (flags != symbol || refs.count == sizeof(slot_lengths)/sizeof(*slot_lengths))
        return S::unsupported;

    XdrStringHeader name;
    status = read_xdr_string_header(in, name);
    if (status != S::ok) return status;
    char text[8];
    if (name.length <= 0 || size_t(name.length) > sizeof(text)) return S::unsupported;
    status = read_xdr_bytes(in, text, size_t(name.length));
    if (status != S::ok) return status;
    for (unsigned i = 0; i < sizeof(slot_lengths)/sizeof(*slot_lengths); ++i) {
        if (name.length == slot_lengths[i] &&
            !__builtin_memcmp(text, slot_names[i], size_t(name.length))) {
            slot = Slot(i);
            refs.slots[refs.count++] = slot;
            return S::ok;
        }
    }
    return S::unsupported;
}

S read_literal(XdrInput &in, const char *expected, int32_t n) noexcept {
    XdrStringHeader header;
    auto status = read_xdr_string_header(in, header);
    if (status != S::ok) return status;
    char text[9];
    if (header.length != n || size_t(n) > sizeof(text)) return S::unsupported;
    status = read_xdr_bytes(in, text, size_t(n));
    if (status != S::ok) return status;
    return __builtin_memcmp(text, expected, size_t(n)) ? S::unsupported : S::ok;
}

S read_class(XdrInput &in, Symbols &refs) noexcept {
    uint32_t n;
    auto status = read_vector(in, strings | has_attributes, n);
    if (status != S::ok) return status;
    if (n != 1) return S::unsupported;
    status = read_literal(in, "dgCMatrix", 9);
    if (status != S::ok) return status;
    Slot slot;
    status = read_tagged_slot(in, refs, slot);
    if (status != S::ok) return status;
    if (slot != Slot::package) return S::unsupported;
    status = read_vector(in, strings, n);
    if (status != S::ok) return status;
    if (n != 1) return S::unsupported;
    status = read_literal(in, "Matrix", 6);
    if (status != S::ok) return status;
    status = read_tagged_slot(in, refs, slot);
    if (status != S::ok) return status;
    return slot == Slot::end ? S::ok : S::unsupported;
}

S read_offsets(XdrInput &in, const DgcMatrixSink &sink, uint32_t n,
               uint32_t &terminal) noexcept {
    if (!n || n > uint32_t(INT32_MAX) + 1u) return S::invalid;
    int32_t p[4096];
    uint32_t previous = 0;
    for (uint32_t i = 0; i < n;) {
        uint32_t count = n - i;
        if (count > sizeof(p)/sizeof(*p)) count = sizeof(p)/sizeof(*p);
        auto status = read_xdr_bytes(in, p, count*sizeof(*p));
        if (status != S::ok) return status;
        for (uint32_t j = 0; j < count; ++j) {
            uint32_t next = from_xdr(uint32_t(p[j]));
            if (next > INT32_MAX || next < previous || (!(i + j) && next)) return S::invalid;
            p[j] = int32_t(next);
            previous = next;
        }
        if (!sink.column_offsets(sink.state, p, count)) return S::io_error;
        i += count;
    }
    terminal = previous;
    return S::ok;
}

S read_dimnames(XdrInput &in, const DgcMatrixSink &sink, DgcMatrixLayout &layout,
                uint32_t (&counts)[2]) noexcept {
    uint32_t n;
    auto status = read_vector(in, list, n);
    if (status != S::ok) return status;
    if (n != 2) return S::invalid;
    for (unsigned axis = 0; axis < 2; ++axis) {
        uint32_t flags, encoded_length;
        status = read_words(in, &flags, 1);
        if (status != S::ok) return status;
        if (flags == nil) continue;
        if (flags != strings) return S::unsupported;
        status = read_words(in, &encoded_length, 1);
        if (status != S::ok) return status;
        status = read_length(in, encoded_length, counts[axis]);
        if (status != S::ok) return status;
        if (counts[axis] > INT32_MAX) return S::invalid;
        layout.has_names[axis] = true;
        for (uint32_t i = 0; i < counts[axis]; ++i) {
            uint64_t position = in.position;
            XdrStringHeader name;
            status = read_xdr_string_header(in, name);
            if (status != S::ok) return status;
            if (i % name_checkpoint_stride == 0 && !sink.name_checkpoint(sink.state, axis, position))
                return S::io_error;
            if (name.length > 0) {
                status = read_xdr_bytes(in, nullptr, size_t(name.length));
                if (status != S::ok) return status;
            }
        }
    }
    return S::ok;
}

} // namespace

XdrStatus read_xdr_string_header(XdrInput &in, XdrStringHeader &out) noexcept {
    uint32_t header[2];
    auto status = read_words(in, header, 2);
    if (status != S::ok) return status;
    uint32_t flags = header[0];
    if ((flags & ~(ascii | bytes | latin1 | utf8)) != chars) return S::unsupported;
    if (header[1] > INT32_MAX && header[1] != UINT32_MAX) return S::invalid;
    auto encoding = flags & bytes ? XdrNameEncoding::bytes :
                    flags & latin1 ? XdrNameEncoding::latin1 :
                    flags & utf8 ? XdrNameEncoding::utf8 : XdrNameEncoding::native;
    out = {header[1] == UINT32_MAX ? -1 : int32_t(header[1]), encoding};
    return S::ok;
}

XdrStatus scan_dgcmatrix(XdrInput &in, const DgcMatrixSink &sink, DgcMatrixLayout &out) noexcept {
    char format[2];
    auto status = read_xdr_bytes(in, format, 2);
    if (status != S::ok) return status;
    if (format[0] != 'X' || format[1] != '\n') return S::unsupported;

    uint32_t header[4];
    status = read_words(in, header, 4);
    if (status != S::ok) return status;
    if (header[0] != 3) return S::unsupported;
    if (header[3] > INT32_MAX) return S::invalid;
    status = skip(in, header[3]); // Source native-encoding label.
    if (status != S::ok) return status;
    uint32_t flags;
    status = read_words(in, &flags, 1);
    if (status != S::ok) return status;
    if (flags != (s4 | object | has_attributes | s4_object)) return S::unsupported;

    DgcMatrixLayout layout{};
    Symbols refs{};
    uint32_t seen = 0, ni = 0, np = 0, nx = 0, terminal = 0, name_counts[2]{};
    for (;;) {
        Slot slot;
        status = read_tagged_slot(in, refs, slot);
        if (status != S::ok) return status;
        if (slot == Slot::end) break;
        if (slot == Slot::package) return S::unsupported;
        uint32_t bit = 1u << unsigned(slot);
        if (seen & bit) return S::invalid;
        seen |= bit;

        switch (slot) {
        case Slot::i:
            status = read_vector(in, integer, ni);
            if (status != S::ok) return status;
            if (ni > INT32_MAX) return S::invalid;
            layout.row_index_offset = in.position;
            status = skip(in, uint64_t(ni)*4);
            break;
        case Slot::p:
            status = read_vector(in, integer, np);
            if (status == S::ok) status = read_offsets(in, sink, np, terminal);
            break;
        case Slot::dim: {
            uint32_t n, dims[2];
            status = read_vector(in, integer, n);
            if (status != S::ok) return status;
            if (n != 2) return S::invalid;
            status = read_words(in, dims, 2);
            if (status != S::ok) return status;
            if (dims[0] > INT32_MAX || dims[1] > INT32_MAX) return S::invalid;
            layout.rows = dims[0];
            layout.columns = dims[1];
            break;
        }
        case Slot::dimnames:
            status = read_dimnames(in, sink, layout, name_counts);
            break;
        case Slot::x:
            status = read_vector(in, real, nx);
            if (status != S::ok) return status;
            if (nx > INT32_MAX) return S::invalid;
            layout.value_offset = in.position;
            status = skip(in, uint64_t(nx)*8);
            break;
        case Slot::factors: {
            uint32_t n;
            status = read_vector(in, list, n);
            if (status == S::ok && n) return S::unsupported;
            break;
        }
        case Slot::class_:
            status = read_class(in, refs);
            break;
        default:
            return S::unsupported;
        }
        if (status != S::ok) return status;
    }

    constexpr uint32_t required = (1u << unsigned(Slot::package)) - 1u;
    if (seen != required || np != layout.columns + 1u || ni != nx || ni != terminal ||
        (layout.has_names[0] && name_counts[0] != layout.rows) ||
        (layout.has_names[1] && name_counts[1] != layout.columns) ||
        (ni && !layout.rows)) return S::invalid;
    layout.entries = ni;
    out = layout;
    return S::ok;
}

} // namespace cellerator::rds::detail
