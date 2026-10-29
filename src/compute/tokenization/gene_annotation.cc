#include <Cellerator/compute/tokenization/gene_vocabulary.hh>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cellerator::tokenization {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

std::string trim(std::string value) {
    auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::vector<std::string> split(const std::string& value, char separator) {
    std::vector<std::string> result;
    std::istringstream input(value);
    std::string part;
    while (std::getline(input, part, separator)) result.push_back(part);
    if (!value.empty() && value.back() == separator) result.emplace_back();
    return result;
}

std::string percent_decode(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '%') {
            result.push_back(value[i]);
            continue;
        }
        require(i + 2 < value.size(), "invalid percent escape in GFF3 attribute");
        const auto high = hex(value[i + 1]);
        const auto low = hex(value[i + 2]);
        require(high >= 0 && low >= 0, "invalid percent escape in GFF3 attribute");
        result.push_back(static_cast<char>(high * 16 + low));
        i += 2;
    }
    return result;
}

std::unordered_map<std::string, std::string> parse_gtf_attributes(
    const std::string& text) {
    std::unordered_map<std::string, std::string> result;
    std::size_t begin = 0;
    while (begin < text.size()) {
        auto end = begin;
        bool quoted = false;
        bool escaped = false;
        for (; end < text.size(); ++end) {
            const char c = text[end];
            if (escaped) { escaped = false; continue; }
            if (quoted && c == '\\') { escaped = true; continue; }
            if (c == '"') { quoted = !quoted; continue; }
            if (c == ';' && !quoted) break;
        }
        auto item = trim(text.substr(begin, end - begin));
        begin = end == text.size() ? end : end + 1;
        if (item.empty()) continue;
        const auto separator = item.find_first_of(" \t");
        require(separator != std::string::npos,
                "malformed GTF attribute (expected key and quoted value)");
        auto key = trim(item.substr(0, separator));
        auto value = trim(item.substr(separator + 1));
        require(!key.empty() && !value.empty(), "malformed GTF attribute");
        std::string decoded;
        if (value.front() == '"') {
            require(value.size() >= 2 && value.back() == '"',
                    "malformed quoted GTF attribute");
            decoded.reserve(value.size() - 2);
            for (std::size_t i = 1; i + 1 < value.size(); ++i) {
                if (value[i] == '\\' && i + 2 < value.size()) ++i;
                decoded.push_back(value[i]);
            }
        } else {
            decoded = std::move(value);
        }
        const auto [it, inserted] = result.emplace(key, decoded);
        require(inserted || key == "tag" || it->second == decoded,
                "conflicting duplicate GTF attribute");
    }
    return result;
}

std::unordered_map<std::string, std::string> parse_gff3_attributes(
    const std::string& text) {
    std::unordered_map<std::string, std::string> result;
    for (const auto& raw : split(text, ';')) {
        if (raw.empty() || raw == ".") continue;
        const auto separator = raw.find('=');
        require(separator != std::string::npos,
                "malformed GFF3 attribute (expected key=value)");
        auto key = trim(raw.substr(0, separator));
        auto value = raw.substr(separator + 1);
        require(!key.empty(), "empty GFF3 attribute key");
        const auto [it, inserted] = result.emplace(key, std::move(value));
        require(inserted || it->second == raw.substr(separator + 1),
                "conflicting duplicate GFF3 attribute");
    }
    return result;
}

std::vector<std::string> annotation_aliases(
    const std::unordered_map<std::string, std::string>& attributes,
    GeneAnnotationFormat format) {
    std::vector<std::string> aliases;
    auto append = [&](const char* key, bool comma_separated = false) {
        auto found = attributes.find(key);
        if (found == attributes.end()) return;
        for (auto value : split(found->second, comma_separated ? ',' : '|')) {
            if (format == GeneAnnotationFormat::gff3) value = percent_decode(value);
            value = trim(value);
            if (!value.empty()) aliases.push_back(std::move(value));
        }
    };
    if (format == GeneAnnotationFormat::gtf) {
        append("gene_name");
        append("gene_synonym");
        append("gene_alias");
    } else {
        append("Name");
        append("Alias", true);
    }
    return aliases;
}

GeneRecord parse_annotation_record(const std::string& attributes,
                                   GeneAnnotationFormat format) {
    auto parsed = format == GeneAnnotationFormat::gtf
        ? parse_gtf_attributes(attributes) : parse_gff3_attributes(attributes);
    const char* identity_key = format == GeneAnnotationFormat::gtf ? "gene_id" : "ID";
    auto identity = parsed.find(identity_key);
    require(identity != parsed.end() && !identity->second.empty(),
            "selected annotation feature is missing its gene identity");
    GeneRecord record;
    record.gene_id = format == GeneAnnotationFormat::gff3
        ? percent_decode(identity->second) : identity->second;
    auto aliases = annotation_aliases(parsed, format);
    if (!aliases.empty()) {
        record.symbol = aliases.front();
        aliases.erase(aliases.begin());
    }
    record.aliases = std::move(aliases);
    return record;
}

std::vector<std::string> annotation_fields(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (fields.size() < 8) {
        const auto end = line.find('\t', begin);
        if (end == std::string::npos) return {};
        fields.push_back(line.substr(begin, end - begin));
        begin = end + 1;
    }
    fields.push_back(line.substr(begin));
    return fields;
}


} // namespace

GeneVocabulary GeneVocabulary::from_annotation(
    const std::string& path, GeneAnnotationFormat format,
    const GeneAnnotationOptions& options) {
    require(!options.feature_type.empty(), "annotation feature type must not be empty");
    require(path.size() < 3 || path.substr(path.size() - 3) != ".gz",
            "compressed annotations are not supported; provide a plain GTF or GFF3 file");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("unable to open gene annotation: " + path);
    auto metadata = options.metadata;
    if (metadata.source.empty()) metadata.source = path;
    std::vector<GeneRecord> records;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (format == GeneAnnotationFormat::gff3 && line == "##FASTA") break;
        if (line.empty() || line.front() == '#') continue;
        auto fields = annotation_fields(line);
        if (fields.size() != 9)
            throw std::invalid_argument("malformed annotation row at line "
                                        + std::to_string(line_number));
        if (fields[2] != options.feature_type) continue;
        try {
            records.push_back(parse_annotation_record(fields[8], format));
        } catch (const std::exception& error) {
            throw std::invalid_argument("annotation line " + std::to_string(line_number)
                                        + ": " + error.what());
        }
    }
    if (input.bad()) throw std::runtime_error("failed while reading gene annotation: " + path);
    if (records.empty())
        throw std::invalid_argument("annotation contains no selected features: "
                                    + options.feature_type);
    return from_records(std::move(records), std::move(metadata),
                        options.unknown_policy, options.unknown_token_id);
}

} // namespace cellerator::tokenization
