#include <CellShard/ingest/internal/tenx_h5/tenx_h5_reader.cuh>
#include <Cellerator/tensor/base/compressed.cuh>

#include <cuda_fp16.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace tenx = cellshard::ingest::tenx_h5;
namespace common = cellshard::ingest::common;

namespace {

constexpr const char* source_url =
    "https://cf.10xgenomics.com/samples/cell-arc/1.0.0/pbmc_unsorted_10k/"
    "pbmc_unsorted_10k_filtered_feature_bc_matrix.h5";
constexpr const char* dataset_page =
    "https://www.10xgenomics.com/datasets/"
    "pbmc-from-a-healthy-donor-no-cell-sorting-10-k-1-standard-1-0-0";
constexpr const char* license_name = "CC BY 4.0";
constexpr const char* expected_source_sha256 =
    "0668e2bf9ff521a8e5e7aa37717f3780bad84081f26f888df16a9398d5020e76";
constexpr std::uint64_t expected_cells = 12016;
constexpr std::uint64_t expected_peaks = 63751;

using Digest = std::array<unsigned char, 32>;

Digest sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open source file for SHA-256");
    EVP_MD_CTX* raw = EVP_MD_CTX_new();
    if (raw == nullptr) throw std::runtime_error("cannot allocate SHA-256 context");
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(raw, EVP_MD_CTX_free);
    if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("cannot initialize SHA-256");
    std::array<char, 1u << 20> chunk{};
    while (input) {
        input.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const auto count = input.gcount();
        if (count > 0 && EVP_DigestUpdate(context.get(), chunk.data(),
                                          static_cast<std::size_t>(count)) != 1)
            throw std::runtime_error("cannot hash source bytes");
    }
    if (!input.eof()) throw std::runtime_error("error reading source file");
    Digest digest{};
    unsigned length = 0;
    if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1 ||
        length != digest.size())
        throw std::runtime_error("cannot finish SHA-256");
    return digest;
}

Digest sha256_bytes(const std::vector<std::uint8_t>& bytes) {
    Digest digest{};
    unsigned length = 0;
    if (EVP_Digest(bytes.data(), bytes.size(), digest.data(), &length,
                   EVP_sha256(), nullptr) != 1 || length != digest.size())
        throw std::runtime_error("cannot compute SHA-256");
    return digest;
}

std::string hex(const Digest& digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : digest) out << std::setw(2) << unsigned(byte);
    return out.str();
}

std::string csv(const std::string& value) {
    if (value.find_first_of("\t\r\n\"") == std::string::npos)
        return value;
    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') out.push_back('"');
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

std::string json(const std::string& value) {
    std::string out = "\"";
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') {
            out.push_back('\\');
            out.push_back(static_cast<char>(ch));
        } else if (ch == '\n') {
            out += "\\n";
        } else if (ch == '\r') {
            out += "\\r";
        } else if (ch == '\t') {
            out += "\\t";
        } else if (ch < 0x20u) {
            char encoded[7]{};
            std::snprintf(encoded, sizeof(encoded), "\\u%04x", unsigned(ch));
            out += encoded;
        } else {
            out.push_back(static_cast<char>(ch));
        }
    }
    out.push_back('"');
    return out;
}

std::uint64_t fnv64(const std::vector<std::uint8_t>& bytes) noexcept {
    std::uint64_t value = 14695981039346656037ull;
    for (std::uint8_t byte : bytes) {
        value ^= byte;
        value *= 1099511628211ull;
    }
    return value;
}

template<class Integer>
void append_le(std::vector<std::uint8_t>& bytes, Integer value) {
    static_assert(std::is_unsigned_v<Integer>);
    for (std::size_t i = 0; i < sizeof(Integer); ++i)
        bytes.push_back(static_cast<std::uint8_t>(value >> (8u * i)));
}

void append_bytes(std::vector<std::uint8_t>& bytes,
                  const unsigned char* data, std::size_t size) {
    bytes.insert(bytes.end(), data, data + size);
}

void write_file(const std::filesystem::path& path,
                const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("failed writing " + path.string());
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) throw std::runtime_error("failed writing " + path.string());
}

struct Peak final {
    std::uint64_t source_index = 0;
    std::string id;
    std::string name;
    std::string chromosome;
    std::uint64_t start = 0;
    std::uint64_t end = 0;
};

bool parse_coordinate(const std::string& id, std::string* chromosome,
                      std::uint64_t* start, std::uint64_t* end) {
    const auto colon = id.rfind(':');
    const auto dash = id.rfind('-');
    if (colon == std::string::npos || dash == std::string::npos || dash <= colon + 1 ||
        dash + 1 >= id.size())
        return false;
    const auto number = [](const std::string& text, std::size_t first,
                           std::size_t last, std::uint64_t* out) {
        if (first >= last) return false;
        std::uint64_t value = 0;
        for (std::size_t i = first; i < last; ++i) {
            const char ch = text[i];
            if (ch < '0' || ch > '9') return false;
            const auto digit = static_cast<std::uint64_t>(ch - '0');
            if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10u)
                return false;
            value = value * 10u + digit;
        }
        *out = value;
        return true;
    };
    if (!number(id, colon + 1, dash, start) ||
        !number(id, dash + 1, id.size(), end) || *end < *start)
        return false;
    *chromosome = id.substr(0, colon);
    return !chromosome->empty();
}

void validate_raw_counts(const std::filesystem::path& path,
                         std::uint64_t expected_nnz) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("CellShard count validation cannot open HDF5 source");
    hid_t matrix = H5Gopen2(file, "/matrix", H5P_DEFAULT);
    if (matrix < 0) {
        H5Fclose(file);
        throw std::runtime_error("10x source does not contain /matrix");
    }
    hid_t dataset = H5Dopen2(matrix, "data", H5P_DEFAULT);
    if (dataset < 0) {
        H5Gclose(matrix);
        H5Fclose(file);
        throw std::runtime_error("10x source does not contain /matrix/data");
    }
    hsize_t length = 0;
    const bool length_ok = cellshard::ingest::h5ad::dataset_length(dataset, &length);
    H5Dclose(dataset);
    if (!length_ok || length != expected_nnz) {
        H5Gclose(matrix);
        H5Fclose(file);
        throw std::runtime_error("10x count vector length does not match CellShard metadata");
    }
    constexpr std::uint64_t chunk_elements = 1u << 20u;
    std::vector<double> values;
    bool valid = true;
    for (std::uint64_t begin = 0; begin < expected_nnz; begin += chunk_elements) {
        const auto count = std::min<std::uint64_t>(chunk_elements, expected_nnz - begin);
        if (!cellshard::ingest::h5ad::read_dataset_double_range(
                matrix, "data", begin, count, &values) || values.size() != count) {
            H5Gclose(matrix);
            H5Fclose(file);
            throw std::runtime_error("failed to read raw 10x count values");
        }
        for (double value : values) {
            if (!std::isfinite(value) || value < 0.0 ||
                std::floor(value) != value) {
                valid = false;
                break;
            }
        }
        if (!valid) break;
    }
    H5Gclose(matrix);
    H5Fclose(file);
    if (!valid)
        throw std::runtime_error(
            "ATAC replay accepts only finite nonnegative integer counts");
}

std::vector<std::uint8_t> make_cell_sidecar(const std::vector<std::string>& barcodes) {
    std::ostringstream out;
    out << "cell_index\tbarcode\n";
    for (std::size_t i = 0; i < barcodes.size(); ++i)
        out << i << '\t' << csv(barcodes[i]) << '\n';
    const auto text = out.str();
    return {text.begin(), text.end()};
}

std::vector<std::uint8_t> make_peak_sidecar(const std::vector<Peak>& peaks) {
    std::ostringstream out;
    out << "peak_index\tsource_feature_index\tid\tname\tchromosome\tstart\tend\n";
    for (std::size_t i = 0; i < peaks.size(); ++i) {
        const auto& peak = peaks[i];
        out << i << '\t' << peak.source_index << '\t' << csv(peak.id) << '\t'
            << csv(peak.name) << '\t' << csv(peak.chromosome) << '\t'
            << peak.start << '\t' << peak.end << '\n';
    }
    const auto text = out.str();
    return {text.begin(), text.end()};
}

void prepare(const std::filesystem::path& source,
             const std::filesystem::path& output_dir,
             const std::string& cellshard_revision) {
    if (cellshard_revision.size() != 40u ||
        !std::all_of(cellshard_revision.begin(), cellshard_revision.end(),
                     [](unsigned char ch) { return std::isxdigit(ch) != 0; }))
        throw std::runtime_error("CellShard revision must be a 40-character commit ID");
    const Digest source_digest = sha256_file(source);
    if (hex(source_digest) != expected_source_sha256)
        throw std::runtime_error("source SHA-256 does not match the pinned official 10x input");
    tenx::matrix_info info{};
    std::string error;
    if (!tenx::probe_matrix(source.c_str(), &info, &error) || !info.available)
        throw std::runtime_error("CellShard could not read 10x matrix: " + error);
    if (info.rows != expected_cells || info.cols != 100352ul)
        throw std::runtime_error("dataset matrix shape differs from the pinned PBMC ARC source");

    common::barcode_table barcode_table{};
    common::init(&barcode_table);
    common::feature_table feature_table{};
    common::init(&feature_table);
    if (!tenx::load_barcodes(source.c_str(), &barcode_table, &error))
        throw std::runtime_error("CellShard barcode ingest failed: " + error);
    if (!tenx::load_feature_table(source.c_str(), &feature_table, &error))
        throw std::runtime_error("CellShard feature ingest failed: " + error);
    if (common::count(&barcode_table) != info.rows ||
        common::count(&feature_table) != info.cols)
        throw std::runtime_error("CellShard metadata columns do not match sparse matrix shape");

    std::vector<std::string> barcodes;
    barcodes.reserve(info.rows);
    std::unordered_set<std::string> unique_barcodes;
    for (unsigned int i = 0; i < common::count(&barcode_table); ++i) {
        const char* value = common::get(&barcode_table, i);
        if (value == nullptr || *value == '\0' || !unique_barcodes.emplace(value).second)
            throw std::runtime_error("10x cell barcode is empty or duplicated");
        barcodes.emplace_back(value);
    }

    std::vector<Peak> peaks;
    peaks.reserve(expected_peaks);
    std::unordered_set<std::string> unique_peak_ids;
    std::vector<std::uint64_t> feature_to_peak(
        static_cast<std::size_t>(info.cols), std::numeric_limits<std::uint64_t>::max());
    for (unsigned int i = 0; i < common::count(&feature_table); ++i) {
        const char* type = common::type(&feature_table, i);
        if (type == nullptr || std::strcmp(type, "Peaks") != 0) continue;
        const char* id = common::id(&feature_table, i);
        const char* name = common::name(&feature_table, i);
        if (id == nullptr || name == nullptr || *id == '\0' || *name == '\0')
            throw std::runtime_error("ATAC peak metadata is empty");
        if (!unique_peak_ids.emplace(id).second)
            throw std::runtime_error("ATAC peak ID is duplicated");
        Peak peak{};
        peak.source_index = i;
        peak.id = id;
        peak.name = name;
        if (!parse_coordinate(peak.id, &peak.chromosome, &peak.start, &peak.end))
            throw std::runtime_error("peak ID does not retain a parseable genomic interval: " + peak.id);
        feature_to_peak[i] = peaks.size();
        peaks.push_back(std::move(peak));
    }
    if (peaks.size() != expected_peaks)
        throw std::runtime_error("filtered Peaks feature count differs from pinned dataset identity");

    validate_raw_counts(source, info.nnz);
    cellshard::ingest::mtx::header header{};
    unsigned long* row_nnz = nullptr;
    if (!tenx::scan_row_nnz(source.c_str(), &header, &row_nnz, &error))
        throw std::runtime_error("CellShard sparse matrix scan failed: " + error);
    std::unique_ptr<unsigned long, decltype(&std::free)> row_nnz_owner(row_nnz, std::free);
    std::vector<unsigned long> row_offsets{0ul, static_cast<unsigned long>(info.rows)};
    std::vector<unsigned long> part_nnz{static_cast<unsigned long>(info.nnz)};
    cellshard::sharded<cellerator::tensor::base::compressed> matrix{};
    cellshard::init(&matrix);
    if (!tenx::load_part_window_compressed(source.c_str(), &header,
            row_offsets.data(), part_nnz.data(), 1ul, 0ul, 1ul, &matrix, &error))
        throw std::runtime_error("CellShard sparse cell-row ingest failed: " + error);
    std::unique_ptr<cellshard::sharded<cellerator::tensor::base::compressed>,
                    void(*)(cellshard::sharded<cellerator::tensor::base::compressed>*)>
        matrix_owner(&matrix, [](auto* value) { cellshard::clear(value); });
    auto* cells = matrix.parts[0];
    const std::uint64_t words_per_peak = (info.rows + 31u) / 32u;
    if (peaks.size() > std::numeric_limits<std::size_t>::max() / words_per_peak)
        throw std::overflow_error("ATAC feature-plane extent overflows");
    std::vector<std::uint32_t> planes(
        peaks.size() * static_cast<std::size_t>(words_per_peak), 0u);
    for (std::uint64_t cell = 0; cell < info.rows; ++cell) {
        const auto begin = cells->majorPtr[cell];
        const auto end = cells->majorPtr[cell + 1u];
        for (auto entry = begin; entry < end; ++entry) {
            const std::uint64_t feature = cells->minorIdx[entry];
            const std::uint64_t peak = feature_to_peak[feature];
            if (peak == std::numeric_limits<std::uint64_t>::max()) continue;
            const float count = static_cast<float>(cells->val[entry]);
            if (count <= 0.0f) continue;
            const auto word = static_cast<std::size_t>(
                peak * words_per_peak + cell / 32u);
            planes[word] |= std::uint32_t{1} << (cell & 31u);
        }
    }

    std::vector<std::uint8_t> payload;
    payload.reserve(64u + planes.size() * sizeof(std::uint32_t));
    const std::array<std::uint8_t, 8> magic{{'C','G','A','T','A','C','0','1'}};
    payload.insert(payload.end(), magic.begin(), magic.end());
    append_le(payload, std::uint32_t{1});
    append_le(payload, std::uint32_t{0x01020304});
    append_le(payload, static_cast<std::uint64_t>(info.rows));
    append_le(payload, static_cast<std::uint64_t>(peaks.size()));
    append_le(payload, words_per_peak);
    append_bytes(payload, source_digest.data(), source_digest.size());
    const std::size_t header_bytes = payload.size();
    for (std::uint32_t word : planes) append_le(payload, word);

    const auto cells_sidecar = make_cell_sidecar(barcodes);
    const auto peaks_sidecar = make_peak_sidecar(peaks);
    std::vector<std::uint8_t> wrapper;
    wrapper.reserve(payload.size() + cells_sidecar.size() + peaks_sidecar.size());
    wrapper.insert(wrapper.end(), payload.begin(), payload.end());
    wrapper.insert(wrapper.end(), cells_sidecar.begin(), cells_sidecar.end());
    wrapper.insert(wrapper.end(), peaks_sidecar.begin(), peaks_sidecar.end());
    const Digest replay_digest = sha256_bytes(payload);
    const Digest cells_digest = sha256_bytes(cells_sidecar);
    const Digest peaks_digest = sha256_bytes(peaks_sidecar);
    const Digest wrapper_digest = sha256_bytes(wrapper);
    const Digest identities_digest = [&] {
        std::vector<std::uint8_t> ids;
        ids.reserve(cells_sidecar.size() + peaks_sidecar.size());
        ids.insert(ids.end(), cells_sidecar.begin(), cells_sidecar.end());
        ids.insert(ids.end(), peaks_sidecar.begin(), peaks_sidecar.end());
        return sha256_bytes(ids);
    }();
    const std::uint64_t payload_fnv = [&] {
        std::vector<std::uint8_t> bytes;
        bytes.reserve(planes.size() * sizeof(std::uint32_t));
        for (std::size_t i = header_bytes; i < payload.size(); ++i)
            bytes.push_back(payload[i]);
        return fnv64(bytes);
    }();

    std::filesystem::create_directories(output_dir);
    write_file(output_dir / "accessibility.cgatac", payload);
    write_file(output_dir / "cells.tsv", cells_sidecar);
    write_file(output_dir / "features.tsv", peaks_sidecar);
    std::ostringstream manifest;
    manifest << "{\n"
             << "  \"format\": \"CGATAC01\",\n"
             << "  \"format_version\": 1,\n"
             << "  \"byte_order\": \"little-endian\",\n"
             << "  \"dataset\": \"PBMC from a healthy donor, no cell sorting, 10k\",\n"
             << "  \"dataset_release\": \"10x ARC 1.0.0\",\n"
             << "  \"dataset_page\": " << json(dataset_page) << ",\n"
             << "  \"source_url\": " << json(source_url) << ",\n"
             << "  \"license\": " << json(license_name) << ",\n"
             << "  \"source_sha256\": " << json(hex(source_digest)) << ",\n"
             << "  \"cellshard_revision\": " << json(cellshard_revision) << ",\n"
             << "  \"preparation_compiler\": "
             << json("nvcc " + std::to_string(__CUDACC_VER_MAJOR__) + "." +
                    std::to_string(__CUDACC_VER_MINOR__) + "." +
                    std::to_string(__CUDACC_VER_BUILD__)) << ",\n"
             << "  \"hdf5_version\": "
             << json(std::to_string(H5_VERS_MAJOR) + "." +
                    std::to_string(H5_VERS_MINOR) + "." +
                    std::to_string(H5_VERS_RELEASE)) << ",\n"
             << "  \"cells\": " << info.rows << ",\n"
             << "  \"peaks\": " << peaks.size() << ",\n"
             << "  \"words_per_peak\": " << words_per_peak << ",\n"
             << "  \"payload_layout\": \"peak-major uint32[peak,ceil(cells/32)]\",\n"
             << "  \"observed_positive_only\": true,\n"
             << "  \"zero_means_biological_exclusion\": false,\n"
             << "  \"payload_fnv64\": " << payload_fnv << ",\n"
             << "  \"replay_sha256\": " << json(hex(replay_digest)) << ",\n"
             << "  \"cells_sha256\": " << json(hex(cells_digest)) << ",\n"
             << "  \"features_sha256\": " << json(hex(peaks_digest)) << ",\n"
             << "  \"identity_sidecars_sha256\": " << json(hex(identities_digest)) << ",\n"
             << "  \"wrapper_sha256\": " << json(hex(wrapper_digest)) << "\n"
             << "}\n";
    write_text(output_dir / "manifest.json", manifest.str());
    std::cout << "native CellShard ATAC replay prepared: cells=" << info.rows
              << " peaks=" << peaks.size() << " bytes=" << payload.size()
              << " source_sha256=" << hex(source_digest)
              << " replay_sha256=" << hex(replay_digest)
              << " wrapper_sha256=" << hex(wrapper_digest) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: cgs_prepare_atac <10x-filtered-feature-matrix.h5> "
                     "<output-dir> <CellShard-commit-id>\n";
        return 2;
    }
    try {
        prepare(argv[1], argv[2], argv[3]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cgs_prepare_atac: " << error.what() << '\n';
        return 1;
    }
}
