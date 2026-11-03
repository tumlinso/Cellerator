// Independent host-only replay check against the original 10x CSC HDF5 data.
// This reads the raw HDF5 matrix with HDF5 C APIs and does not call CellShard
// or converter code. Compile with h5c++ -std=c++17 ... -lcrypto.
#include <hdf5.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
constexpr std::uint64_t kCells = 12016;
constexpr std::uint64_t kFeatures = 100352;
constexpr std::uint64_t kPeaks = 63751;
constexpr std::uint64_t kWords = (kCells + 31) / 32;
constexpr std::size_t kHeader = 72;
constexpr const char* kSourceSha =
    "0668e2bf9ff521a8e5e7aa37717f3780bad84081f26f888df16a9398d5020e76";

struct H5 final {
    hid_t id = -1;
    herr_t (*close)(hid_t) = nullptr;
    H5(hid_t value, herr_t (*closer)(hid_t)) : id(value), close(closer) {
        if (id < 0) throw std::runtime_error("HDF5 open failed");
    }
    ~H5() { if (id >= 0) close(id); }
    H5(const H5&) = delete;
    H5& operator=(const H5&) = delete;
};

H5 dataset(hid_t file, const char* name) {
    return H5(H5Dopen2(file, name, H5P_DEFAULT), H5Dclose);
}

std::uint64_t extent_1d(hid_t dset) {
    H5 space(H5Dget_space(dset), H5Sclose);
    if (H5Sget_simple_extent_ndims(space.id) != 1)
        throw std::runtime_error("expected a one-dimensional HDF5 dataset");
    hsize_t dim = 0;
    if (H5Sget_simple_extent_dims(space.id, &dim, nullptr) < 0)
        throw std::runtime_error("cannot read HDF5 extent");
    return dim;
}

template<class T>
std::vector<T> read_all(hid_t dset, hid_t memory_type, std::uint64_t count) {
    std::vector<T> values(static_cast<std::size_t>(count));
    if (H5Dread(dset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                values.data()) < 0)
        throw std::runtime_error("HDF5 read failed");
    return values;
}

template<class T>
std::vector<T> read_range(hid_t dset, hid_t memory_type,
                          std::uint64_t start, std::uint64_t count) {
    std::vector<T> values(static_cast<std::size_t>(count));
    H5 file_space(H5Dget_space(dset), H5Sclose);
    const hsize_t offset = start, extent = count;
    if (H5Sselect_hyperslab(file_space.id, H5S_SELECT_SET, &offset, nullptr,
                            &extent, nullptr) < 0)
        throw std::runtime_error("HDF5 slice selection failed");
    H5 memory_space(H5Screate_simple(1, &extent, nullptr), H5Sclose);
    if (H5Dread(dset, memory_type, memory_space.id, file_space.id, H5P_DEFAULT,
                values.data()) < 0)
        throw std::runtime_error("HDF5 slice read failed");
    return values;
}

std::vector<std::string> read_fixed_strings(hid_t dset, std::uint64_t count) {
    H5 type(H5Dget_type(dset), H5Tclose);
    if (H5Tget_class(type.id) != H5T_STRING || H5Tis_variable_str(type.id) > 0)
        throw std::runtime_error("expected fixed-width HDF5 strings");
    const auto width = H5Tget_size(type.id);
    std::vector<char> raw(static_cast<std::size_t>(count) * width, '\0');
    if (H5Dread(dset, type.id, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.data()) < 0)
        throw std::runtime_error("HDF5 string read failed");
    std::vector<std::string> result;
    result.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        const char* begin = raw.data() + static_cast<std::size_t>(i) * width;
        std::size_t size = 0;
        while (size < width && begin[size] != '\0') ++size;
        result.emplace_back(begin, size);
    }
    return result;
}

std::vector<std::vector<std::string>> read_tsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read replay sidecar: " + path);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string> fields;
        std::size_t begin = 0;
        for (;;) {
            const auto end = line.find('\t', begin);
            fields.emplace_back(line.substr(begin, end == std::string::npos
                ? std::string::npos : end - begin));
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        rows.push_back(std::move(fields));
    }
    return rows;
}

std::uint64_t decimal(const std::string& text) {
    if (text.empty()) throw std::runtime_error("empty coordinate");
    std::uint64_t value = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9') throw std::runtime_error("invalid coordinate");
        const auto digit = static_cast<unsigned>(ch - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
            throw std::runtime_error("coordinate overflow");
        value = value * 10 + digit;
    }
    return value;
}

std::string sha256(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot hash original HDF5 source");
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context) throw std::runtime_error("SHA-256 allocation failed");
    if (EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("SHA-256 initialization failed");
    std::array<char, 1u << 20> buffer{};
    while (in) {
        in.read(buffer.data(), buffer.size());
        const auto n = in.gcount();
        if (n && EVP_DigestUpdate(context, buffer.data(), n) != 1)
            throw std::runtime_error("SHA-256 update failed");
    }
    std::array<unsigned char, 32> digest{};
    unsigned length = 0;
    const int ok = EVP_DigestFinal_ex(context, digest.data(), &length);
    EVP_MD_CTX_free(context);
    if (ok != 1 || length != digest.size()) throw std::runtime_error("SHA-256 final failed");
    static constexpr char hex[] = "0123456789abcdef";
    std::string text;
    text.reserve(64);
    for (auto byte : digest) { text.push_back(hex[byte >> 4]); text.push_back(hex[byte & 15]); }
    return text;
}

std::uint64_t read_le(const std::vector<std::uint8_t>& data, std::size_t at,
                      unsigned width) {
    if (width > 8 || at > data.size() || width > data.size() - at)
        throw std::runtime_error("binary replay header is truncated");
    std::uint64_t value = 0;
    for (unsigned i = 0; i < width; ++i)
        value |= std::uint64_t(data[at + i]) << (8 * i);
    return value;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read replay payload");
    return {std::istreambuf_iterator<char>(in), {}};
}

void verify(const std::string& source, const std::string& replay_dir) {
    if (sha256(source) != kSourceSha)
        throw std::runtime_error("original source SHA-256 differs from pinned dataset");
    const auto cells = read_tsv(replay_dir + "/cells.tsv");
    const auto peaks = read_tsv(replay_dir + "/features.tsv");
    if (cells.size() != kCells + 1 || cells[0] != std::vector<std::string>{"cell_index", "barcode"} ||
        peaks.size() != kPeaks + 1 || peaks[0] != std::vector<std::string>{
            "peak_index", "source_feature_index", "id", "name", "chromosome", "start", "end"})
        throw std::runtime_error("replay sidecar headers or lengths differ from expected dataset");

    H5 file(H5Fopen(source.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    auto barcodes_d = dataset(file.id, "/matrix/barcodes");
    auto indptr_d = dataset(file.id, "/matrix/indptr");
    auto indices_d = dataset(file.id, "/matrix/indices");
    auto values_d = dataset(file.id, "/matrix/data");
    auto ids_d = dataset(file.id, "/matrix/features/id");
    auto names_d = dataset(file.id, "/matrix/features/name");
    auto types_d = dataset(file.id, "/matrix/features/feature_type");
    if (extent_1d(barcodes_d.id) != kCells || extent_1d(indptr_d.id) != kCells + 1 ||
        extent_1d(ids_d.id) != kFeatures || extent_1d(names_d.id) != kFeatures ||
        extent_1d(types_d.id) != kFeatures)
        throw std::runtime_error("source HDF5 dimensions differ from pinned dataset");
    const auto source_barcodes = read_fixed_strings(barcodes_d.id, kCells);
    const auto source_ids = read_fixed_strings(ids_d.id, kFeatures);
    const auto source_names = read_fixed_strings(names_d.id, kFeatures);
    const auto source_types = read_fixed_strings(types_d.id, kFeatures);
    if (cells[0].size() != 2 || peaks[0].size() != 7)
        throw std::runtime_error("invalid sidecar header columns");
    std::unordered_set<std::string> unique_barcodes, unique_peaks;
    for (std::uint64_t c = 0; c < kCells; ++c) {
        if (cells[c + 1].size() != 2 || cells[c + 1][0] != std::to_string(c) ||
            cells[c + 1][1] != source_barcodes[c] || !unique_barcodes.insert(source_barcodes[c]).second)
            throw std::runtime_error("cell identity/order does not match original 10x source");
    }
    std::vector<std::int32_t> source_to_peak(kFeatures, -1);
    std::uint64_t expected_peak = 0;
    for (std::uint64_t feature = 0; feature < kFeatures; ++feature) {
        if (source_types[feature] != "Peaks") continue;
        if (expected_peak >= kPeaks) throw std::runtime_error("too many source Peaks features");
        const auto& row = peaks[++expected_peak];
        if (row.size() != 7 || row[0] != std::to_string(expected_peak - 1) ||
            decimal(row[1]) != feature || row[2] != source_ids[feature] ||
            row[3] != source_names[feature] || !unique_peaks.insert(row[2]).second)
            throw std::runtime_error("peak identity/order does not match original 10x source");
        const auto colon = row[2].rfind(':');
        const auto dash = row[2].rfind('-');
        if (colon == std::string::npos || dash <= colon + 1 ||
            row[4] != row[2].substr(0, colon) ||
            decimal(row[5]) != decimal(row[2].substr(colon + 1, dash - colon - 1)) ||
            decimal(row[6]) != decimal(row[2].substr(dash + 1)))
            throw std::runtime_error("peak coordinates do not match original feature IDs");
        source_to_peak[feature] = static_cast<std::int32_t>(expected_peak - 1);
    }
    if (expected_peak != kPeaks) throw std::runtime_error("source peak count differs");

    const auto indptr = read_all<std::uint64_t>(indptr_d.id, H5T_NATIVE_UINT64, kCells + 1);
    if (indptr.front() != 0 || indptr.back() != extent_1d(values_d.id) ||
        extent_1d(indices_d.id) != indptr.back())
        throw std::runtime_error("CSC sparse index extents are inconsistent");
    const auto payload = read_file(replay_dir + "/accessibility.cgatac");
    if (payload.size() != kHeader + kPeaks * kWords * 4 ||
        std::memcmp(payload.data(), "CGATAC01", 8) != 0 ||
        read_le(payload, 16, 8) != kCells || read_le(payload, 24, 8) != kPeaks ||
        read_le(payload, 32, 8) != kWords)
        throw std::runtime_error("CGATAC01 header shape mismatch");
    for (std::size_t i = 0; i < 32; ++i) {
        const auto expected_byte = static_cast<std::uint8_t>(
            std::stoul(std::string(kSourceSha + i * 2, 2), nullptr, 16));
        if (payload[40 + i] != expected_byte)
            throw std::runtime_error("CGATAC01 header does not bind the pinned source");
    }

    std::uint64_t checked_nonzero = 0;
    for (std::uint64_t word = 0; word < kWords; ++word) {
        const std::uint64_t first_cell = word * 32;
        const std::uint64_t end_cell = std::min(kCells, first_cell + 32);
        std::vector<std::uint32_t> expected(kPeaks, 0);
        const std::uint64_t begin = indptr[first_cell];
        const std::uint64_t end = indptr[end_cell];
        const auto indices = read_range<std::uint64_t>(indices_d.id, H5T_NATIVE_UINT64,
                                                        begin, end - begin);
        const auto values = read_range<std::int64_t>(values_d.id, H5T_NATIVE_INT64,
                                                      begin, end - begin);
        std::uint64_t cursor = 0;
        for (auto cell = first_cell; cell < end_cell; ++cell) {
            for (; cursor < indptr[cell + 1] - begin; ++cursor) {
                const auto source_feature = indices[static_cast<std::size_t>(cursor)];
                const auto count = values[static_cast<std::size_t>(cursor)];
                if (source_feature >= kFeatures || count < 0)
                    throw std::runtime_error("source CSC index/count is invalid");
                const auto peak = source_to_peak[source_feature];
                if (peak >= 0 && count > 0) {
                    expected[static_cast<std::size_t>(peak)] |=
                        std::uint32_t{1} << (cell & 31u);
                    ++checked_nonzero;
                }
            }
        }
        for (std::uint64_t peak = 0; peak < kPeaks; ++peak) {
            const auto offset = kHeader + (peak * kWords + word) * 4;
            if (read_le(payload, static_cast<std::size_t>(offset), 4) != expected[peak])
                throw std::runtime_error("replay bit differs from raw 10x matrix at peak " +
                                         std::to_string(peak) + ", cell word " +
                                         std::to_string(word));
        }
    }
    if ((kCells & 31u) != 0) {
        const auto padding_mask = ~((std::uint32_t{1} << (kCells & 31u)) - 1u);
        for (std::uint64_t peak = 0; peak < kPeaks; ++peak) {
            const auto offset = kHeader + (peak * kWords + kWords - 1) * 4;
            if ((read_le(payload, static_cast<std::size_t>(offset), 4) & padding_mask) != 0)
                throw std::runtime_error("replay has nonzero tail padding");
        }
    }
    std::cout << "Independent raw HDF5 replay match: cells=" << kCells
              << " peaks=" << kPeaks << " positive matrix observations="
              << checked_nonzero << " words=" << (kPeaks * kWords) << '\n';
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: verify_replay_source <original-10x.h5> <replay-dir>\n";
        return 2;
    }
    try { verify(argv[1], argv[2]); return 0; }
    catch (const std::exception& error) {
        std::cerr << "independent raw replay check: " << error.what() << '\n';
        return 1;
    }
}
