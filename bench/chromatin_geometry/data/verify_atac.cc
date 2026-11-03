#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr std::uint64_t expected_cells = 12016;
constexpr std::uint64_t expected_peaks = 63751;
constexpr std::uint64_t expected_words = 376;
constexpr std::size_t header_size = 72;
constexpr const char* expected_source_sha =
    "0668e2bf9ff521a8e5e7aa37717f3780bad84081f26f888df16a9398d5020e76";

using Bytes = std::vector<std::uint8_t>;
using Digest = std::array<unsigned char, 32>;

Bytes read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    return Bytes(std::istreambuf_iterator<char>(input), {});
}

Digest sha256(const std::uint8_t* data, std::size_t size) {
    Digest digest{};
    unsigned digest_size = 0;
    if (EVP_Digest(data, size, digest.data(), &digest_size,
                   EVP_sha256(), nullptr) != 1 || digest_size != digest.size())
        throw std::runtime_error("SHA-256 failed");
    return digest;
}

std::string hex(const Digest& digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2u);
    for (unsigned char byte : digest) {
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 15u]);
    }
    return result;
}

std::uint64_t read_le(const Bytes& bytes, std::size_t offset,
                      std::size_t width) {
    if (width > 8 || offset > bytes.size() || width > bytes.size() - offset)
        throw std::runtime_error("binary field exceeds replay size");
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < width; ++i)
        value |= static_cast<std::uint64_t>(bytes[offset + i]) << (i * 8u);
    return value;
}

std::uint64_t fnv64(const std::uint8_t* data, std::size_t size) noexcept {
    std::uint64_t value = 14695981039346656037ull;
    for (std::size_t i = 0; i < size; ++i) {
        value ^= data[i];
        value *= 1099511628211ull;
    }
    return value;
}

std::string field(const std::string& manifest, const std::string& key) {
    const auto marker = "\"" + key + "\": \"";
    const auto begin = manifest.find(marker);
    if (begin == std::string::npos)
        throw std::runtime_error("manifest is missing string field " + key);
    const auto value_begin = begin + marker.size();
    const auto end = manifest.find('"', value_begin);
    if (end == std::string::npos)
        throw std::runtime_error("manifest has unterminated string field " + key);
    return manifest.substr(value_begin, end - value_begin);
}

std::uint64_t numeric_field(const std::string& manifest,
                            const std::string& key) {
    const auto marker = "\"" + key + "\": ";
    const auto begin = manifest.find(marker);
    if (begin == std::string::npos)
        throw std::runtime_error("manifest is missing numeric field " + key);
    const auto value_begin = begin + marker.size();
    std::size_t used = 0;
    const auto value = std::stoull(manifest.substr(value_begin), &used);
    if (used == 0) throw std::runtime_error("manifest has invalid number " + key);
    return value;
}

std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    for (;;) {
        const auto end = line.find('\t', begin);
        fields.push_back(line.substr(begin, end == std::string::npos
            ? std::string::npos : end - begin));
        if (end == std::string::npos) return fields;
        begin = end + 1u;
    }
}

std::vector<std::string> lines(const Bytes& bytes) {
    std::string content(bytes.begin(), bytes.end());
    std::vector<std::string> result;
    std::size_t begin = 0;
    while (begin < content.size()) {
        const auto end = content.find('\n', begin);
        result.push_back(content.substr(begin, end == std::string::npos
            ? std::string::npos : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1u;
    }
    return result;
}

void verify_sidecars(const Bytes& cells, const Bytes& features) {
    const auto cell_lines = lines(cells);
    const auto feature_lines = lines(features);
    if (cell_lines.size() != expected_cells + 1u ||
        cell_lines.front() != "cell_index\tbarcode")
        throw std::runtime_error("cell sidecar row count/header mismatch");
    if (feature_lines.size() != expected_peaks + 1u ||
        feature_lines.front() !=
            "peak_index\tsource_feature_index\tid\tname\tchromosome\tstart\tend")
        throw std::runtime_error("feature sidecar row count/header mismatch");

    std::unordered_set<std::string> barcodes;
    barcodes.reserve(expected_cells);
    for (std::size_t i = 1; i < cell_lines.size(); ++i) {
        const auto columns = split_tsv(cell_lines[i]);
        if (columns.size() != 2 || columns[0] != std::to_string(i - 1u) ||
            columns[1].empty() || !barcodes.insert(columns[1]).second)
            throw std::runtime_error("cell sidecar contains invalid or duplicate identity");
    }

    std::unordered_set<std::string> peak_ids;
    peak_ids.reserve(expected_peaks);
    for (std::size_t i = 1; i < feature_lines.size(); ++i) {
        const auto columns = split_tsv(feature_lines[i]);
        if (columns.size() != 7 || columns[0] != std::to_string(i - 1u) ||
            columns[2].empty() || columns[4].empty() ||
            !peak_ids.insert(columns[2]).second)
            throw std::runtime_error("feature sidecar contains invalid or duplicate identity");
        std::size_t start_used = 0, end_used = 0;
        const auto start = std::stoull(columns[5], &start_used);
        const auto end = std::stoull(columns[6], &end_used);
        if (start_used != columns[5].size() || end_used != columns[6].size() || end < start)
            throw std::runtime_error("feature coordinate sidecar is malformed");
    }
}

void verify(const std::filesystem::path& directory) {
    const auto payload = read_bytes(directory / "accessibility.cgatac");
    const auto cells = read_bytes(directory / "cells.tsv");
    const auto features = read_bytes(directory / "features.tsv");
    const auto manifest_bytes = read_bytes(directory / "manifest.json");
    const std::string manifest(manifest_bytes.begin(), manifest_bytes.end());
    const std::array<std::uint8_t, 8> magic{{'C','G','A','T','A','C','0','1'}};
    if (payload.size() != header_size + expected_peaks * expected_words * 4u ||
        !std::equal(magic.begin(), magic.end(), payload.begin()))
        throw std::runtime_error("CGATAC01 magic or payload size mismatch");
    if (read_le(payload, 8, 4) != 1u || read_le(payload, 12, 4) != 0x01020304u ||
        read_le(payload, 16, 8) != expected_cells ||
        read_le(payload, 24, 8) != expected_peaks ||
        read_le(payload, 32, 8) != expected_words)
        throw std::runtime_error("CGATAC01 version, endian tag, or extents mismatch");
    Digest embedded{};
    std::copy(payload.begin() + 40, payload.begin() + 72, embedded.begin());
    if (hex(embedded) != expected_source_sha ||
        field(manifest, "source_sha256") != expected_source_sha)
        throw std::runtime_error("replay source identity does not match the pinned download");
    if (field(manifest, "format") != "CGATAC01" ||
        field(manifest, "cellshard_revision") != "bbd84ebd43c7bf1f1cc0168fd38c3689b0279730" ||
        numeric_field(manifest, "cells") != expected_cells ||
        numeric_field(manifest, "peaks") != expected_peaks ||
        numeric_field(manifest, "words_per_peak") != expected_words)
        throw std::runtime_error("manifest format, dependency revision, or extents mismatch");
    const std::uint64_t replay_fnv = fnv64(payload.data() + header_size,
                                           payload.size() - header_size);
    if (replay_fnv != numeric_field(manifest, "payload_fnv64"))
        throw std::runtime_error("CGATAC01 payload FNV64 mismatch");
    std::uint64_t positive_observations = 0;
    for (std::uint64_t peak = 0; peak < expected_peaks; ++peak) {
        const auto offset = header_size + (peak * expected_words + expected_words - 1u) * 4u;
        const auto tail = read_le(payload, static_cast<std::size_t>(offset), 4);
        if ((tail & 0xffff0000u) != 0u)
            throw std::runtime_error("CGATAC01 tail padding bits are set");
    }
    for (std::uint64_t i = header_size; i < payload.size(); i += 4u)
        positive_observations += __builtin_popcount(static_cast<std::uint32_t>(
            read_le(payload, static_cast<std::size_t>(i), 4)));
    if (positive_observations == 0)
        throw std::runtime_error("CGATAC01 contains no positive observations");

    verify_sidecars(cells, features);
    const auto replay_sha = hex(sha256(payload.data(), payload.size()));
    const auto cells_sha = hex(sha256(cells.data(), cells.size()));
    const auto features_sha = hex(sha256(features.data(), features.size()));
    Bytes identities;
    identities.reserve(cells.size() + features.size());
    identities.insert(identities.end(), cells.begin(), cells.end());
    identities.insert(identities.end(), features.begin(), features.end());
    const auto identities_sha = hex(sha256(identities.data(), identities.size()));
    Bytes wrapper;
    wrapper.reserve(payload.size() + identities.size());
    wrapper.insert(wrapper.end(), payload.begin(), payload.end());
    wrapper.insert(wrapper.end(), cells.begin(), cells.end());
    wrapper.insert(wrapper.end(), features.begin(), features.end());
    const auto wrapper_sha = hex(sha256(wrapper.data(), wrapper.size()));
    if (field(manifest, "replay_sha256") != replay_sha ||
        field(manifest, "cells_sha256") != cells_sha ||
        field(manifest, "features_sha256") != features_sha ||
        field(manifest, "identity_sidecars_sha256") != identities_sha ||
        field(manifest, "wrapper_sha256") != wrapper_sha)
        throw std::runtime_error("CGATAC01, sidecar, identity, or wrapper SHA256 mismatch");
    if (manifest.find("\"observed_positive_only\": true") == std::string::npos ||
        manifest.find("\"zero_means_biological_exclusion\": false") == std::string::npos)
        throw std::runtime_error("ATAC observation semantics are missing from manifest");
    std::cout << "CGATAC01 verified: cells=" << expected_cells
              << " peaks=" << expected_peaks
              << " positive_observations=" << positive_observations
              << " replay_sha256=" << replay_sha
              << " wrapper_sha256=" << wrapper_sha << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: cgs_verify_atac <prepared-replay-directory>\n";
        return 2;
    }
    try {
        verify(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cgs_verify_atac: " << error.what() << '\n';
        return 1;
    }
}
