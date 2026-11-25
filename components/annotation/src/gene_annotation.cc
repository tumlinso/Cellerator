#include <Cellerator/biology/gene_annotation.hh>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cellerator::biology {
namespace {

using Text = AnnotationText;

Text literal(const char *text) noexcept {
    return {text, __builtin_strlen(text)};
}

bool equal(Text a, Text b) noexcept {
    return a.size == b.size && (!a.size || !__builtin_memcmp(a.data, b.data, a.size));
}

bool space(char c) noexcept { return c == ' ' || (c >= '\t' && c <= '\r'); }

Text trim(Text v) noexcept {
    while (v.size && space(*v.data)) { ++v.data; --v.size; }
    while (v.size && space(v.data[v.size-1])) --v.size;
    return v;
}

Text split(Text &text, char separator) noexcept {
    const char *data = text.data;
    size_t n = 0;
    while (n < text.size && data[n] != separator) ++n;
    const size_t used = n + (n < text.size);
    text = {data + used, text.size - used};
    return {data, n};
}

struct Attribute {
    Text key, value;
    bool quoted;
};

bool attribute(Text &text, bool gff, Attribute &out) noexcept {
    size_t n = 0;
    bool quoted = false;
    for (; n < text.size; ++n) {
        const char c = text.data[n];
        if (!gff && quoted && c == '\\' && n+1 < text.size) { ++n; continue; }
        if (!gff && c == '"') quoted = !quoted;
        if (c == ';' && !quoted) break;
    }
    Text item{text.data, n};
    const size_t used = n + (n < text.size);
    text = {text.data + used, text.size - used};
    if (!gff) item = trim(item);
    out = {};
    if (!item.size || (gff && equal(item, literal(".")))) return true;
    size_t k = 0;
    while (k < item.size && (gff ? item.data[k] != '=' : !space(item.data[k]))) ++k;
    if (k == item.size || quoted) return false;
    out.key = trim({item.data, k});
    out.value = {item.data + k+1, item.size - k-1};
    if (!gff) out.value = trim(out.value);
    if (!out.key.size || (!gff && !out.value.size)) return false;
    if (!gff && *out.value.data == '"') {
        if (out.value.size < 2 || out.value.data[out.value.size-1] != '"') return false;
        out.quoted = true;
        ++out.value.data;
        out.value.size -= 2;
    }
    return true;
}

int hex(char c) noexcept {
    return c >= '0' && c <= '9' ? c-'0'
         : c >= 'a' && c <= 'f' ? c-'a'+10
         : c >= 'A' && c <= 'F' ? c-'A'+10 : -1;
}

int next_byte(Text &text, bool gff, bool escaped) noexcept {
    const char c = *text.data++;
    --text.size;
    if (escaped && c == '\\' && text.size) {
        --text.size;
        return static_cast<unsigned char>(*text.data++);
    }
    if (!gff || c != '%') return static_cast<unsigned char>(c);
    if (text.size < 2) return -1;
    const int a = hex(text.data[0]), b = hex(text.data[1]);
    text.data += 2;
    text.size -= 2;
    return a < 0 || b < 0 ? -1 : 16*a+b;
}

bool same_value(Attribute a, Attribute b, bool gff) noexcept {
    if (gff || (!a.quoted && !b.quoted)) return equal(a.value, b.value);
    while (a.value.size && b.value.size)
        if (next_byte(a.value, false, a.quoted) != next_byte(b.value, false, b.quoted)) return false;
    return !a.value.size && !b.value.size;
}

struct Catalog {
    GeneAnnotationRecord *records = nullptr;
    Text *aliases = nullptr;
    char *bytes = nullptr;
    size_t record_count = 0, alias_count = 0, byte_count = 0;

    bool append(Text text, bool gff, bool escaped, bool strip, Text &value) noexcept {
        size_t n = 0, first = 0, last = 0;
        bool seen = false;
        while (text.size) {
            const int c = next_byte(text, gff, escaped);
            if (c < 0) return false;
            if (!strip || !space(char(c))) {
                if (!seen) first = n;
                seen = true;
                last = n+1;
            }
            // The sizing pass reserves space for each decoded byte, including
            // surrounding whitespace, so the fill pass never needs scratch.
            if (bytes) bytes[byte_count+n] = char(c);
            ++n;
        }
        if (n > SIZE_MAX - byte_count) return false;
        value = {bytes ? bytes + byte_count+first : nullptr, last-first};
        byte_count += n;
        return true;
    }
};

const char *record(Text text, bool gff, Catalog &catalog) noexcept {
    const char *gtf_keys[]{"gene_id", "gene_name", "gene_synonym", "gene_alias"};
    const char *gff_keys[]{"ID", "Name", "Alias"};
    Attribute selected[4]{};
    const char *begin = text.data;
    while (text.size) {
        const Text start = text;
        Attribute item;
        if (!attribute(text, gff, item)) return "malformed annotation attribute";
        if (!item.key.size) continue;
        if (gff || !equal(item.key, literal("tag"))) {
            Text previous{begin, size_t(start.data - begin)};
            while (previous.size) {
                Attribute existing;
                if (!attribute(previous, gff, existing)) return "malformed annotation attribute";
                if (equal(item.key, existing.key) && !same_value(existing, item, gff))
                    return "conflicting duplicate annotation attribute";
            }
        }
        for (size_t i = 0; i < (gff ? 3u : 4u); ++i) {
            if (!equal(item.key, literal(gff ? gff_keys[i] : gtf_keys[i]))) continue;
            selected[i] = item;
        }
    }
    if (!selected[0].value.size) return "selected annotation feature is missing its gene identity";
    GeneAnnotationRecord out{};
    if (!catalog.append(selected[0].value, gff, selected[0].quoted, false, out.id))
        return "invalid percent escape in GFF3 attribute";
    out.aliases = catalog.aliases ? catalog.aliases + catalog.alias_count : nullptr;
    for (size_t i = 1; i < (gff ? 3u : 4u); ++i) {
        const Attribute a = selected[i];
        Text remaining = a.value;
        while (remaining.size) {
            Text part;
            if (a.quoted) {
                const char *begin = remaining.data;
                const char *end = begin;
                while (remaining.size) {
                    end = remaining.data;
                    if (next_byte(remaining, false, true) == '|') break;
                    end = remaining.data;
                }
                part = {begin, size_t(end - begin)};
            } else part = split(remaining, gff && i == 2 ? ',' : '|');
            Text value;
            if (!catalog.append(part, gff, a.quoted, true, value))
                return "invalid percent escape in GFF3 attribute";
            if (!value.size) continue;
            if (!out.symbol.size) out.symbol = value;
            else {
                if (catalog.aliases) catalog.aliases[catalog.alias_count] = value;
                ++catalog.alias_count;
                ++out.alias_count;
            }
        }
    }
    if (catalog.records) catalog.records[catalog.record_count] = out;
    ++catalog.record_count;
    return nullptr;
}

GeneAnnotationError parse(Text input, bool gff, Text feature, Catalog &catalog) noexcept {
    for (size_t line = 1; input.size; ++line) {
        Text row = split(input, '\n');
        if (row.size && row.data[row.size-1] == '\r') --row.size;
        if (gff && equal(row, literal("##FASTA"))) break;
        if (!row.size || *row.data == '#') continue;
        Text kind;
        for (int column = 0; column < 8; ++column) {
            const size_t size = row.size;
            const Text field = split(row, '\t');
            if (field.size == size) return {"malformed annotation row", line};
            if (column == 2) kind = field;
        }
        if (equal(kind, feature))
            if (const char *error = record(row, gff, catalog)) return {error, line};
    }
    return catalog.record_count ? GeneAnnotationError{} :
        GeneAnnotationError{"annotation contains no selected features"};
}

} // namespace

GeneAnnotation::~GeneAnnotation() { free(storage_); }

GeneAnnotationError read_gene_annotation(const char *path, GeneAnnotationFormat format,
    GeneAnnotation &out, const char *feature_type) noexcept {
    if (!path || !feature_type || !*feature_type)
        return {"annotation path and feature type are required"};
    if (format != GeneAnnotationFormat::gtf && format != GeneAnnotationFormat::gff3)
        return {"invalid annotation format"};
    const size_t path_size = __builtin_strlen(path);
    if (path_size >= 3 && equal({path + path_size-3, 3}, literal(".gz")))
        return {"compressed annotations are not supported; provide a plain GTF or GFF3 file"};
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return {"unable to open gene annotation", 0, true};
    struct stat info{};
    if (fstat(fd, &info) || !S_ISREG(info.st_mode) || info.st_size < 0 ||
        uintmax_t(info.st_size) > SIZE_MAX) {
        close(fd);
        return {"unable to read gene annotation", 0, true};
    }
    const size_t size = size_t(info.st_size);
    char *data = size ? static_cast<char *>(malloc(size)) : nullptr;
    if (size && !data) {
        close(fd);
        return {"unable to provision annotation input", 0, true};
    }
    for (size_t offset = 0; offset < size;) {
        const size_t remaining = size - offset, chunk = 1u << 20;
        const ssize_t n = read(fd, data + offset, remaining < chunk ? remaining : chunk);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            close(fd);
            free(data);
            return {"unable to read gene annotation", 0, true};
        }
        offset += size_t(n);
    }
    close(fd);

    const Text input{data, size};
    const Text feature = literal(feature_type);
    const bool gff = format == GeneAnnotationFormat::gff3;
    Catalog count{};
    GeneAnnotationError error = parse(input, gff, feature, count);
    if (!error.message) {
        size_t records, aliases, total;
        if (__builtin_mul_overflow(count.record_count, sizeof(GeneAnnotationRecord), &records) ||
            __builtin_mul_overflow(count.alias_count, sizeof(Text), &aliases) ||
            __builtin_add_overflow(records, aliases, &total) ||
            __builtin_add_overflow(total, count.byte_count, &total)) {
            error = {"annotation storage size overflow", 0, true};
        } else if (void *storage = total <= out.capacity_ ? out.storage_ : malloc(total)) {
            char *base = static_cast<char *>(storage);
            Catalog catalog{static_cast<GeneAnnotationRecord *>(storage),
                reinterpret_cast<Text *>(base + records), base + records+aliases};
            error = parse(input, gff, feature, catalog);
            if (!error.message) {
                if (storage != out.storage_) {
                    free(out.storage_);
                    out.storage_ = storage;
                    out.capacity_ = total;
                }
                out.records_ = catalog.records;
                out.count_ = catalog.record_count;
            } else if (storage != out.storage_) free(storage);
        } else error = {"unable to provision annotation storage", 0, true};
    }
    free(data);
    return error;
}

} // namespace cellerator::biology
