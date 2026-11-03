#include "data/replay.hh"
#include <openssl/evp.h>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cgs {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t le(const unsigned char* p, int n) {
    std::uint64_t value = 0;
    for (int i = 0; i < n; ++i)
        value |= std::uint64_t(p[i]) << (8 * i);
    return value;
}

} // namespace

void apply_atac_replay(WorkloadFixture& fixture, const std::string& path) {
    apply_atac_replay(fixture, path, nullptr);
}

void apply_atac_replay(WorkloadFixture& f, const std::string& path,
                       ReplayPreparationTimes* timings) {
    const auto operational_begin = Clock::now();
    std::uint64_t bytes_hashed = 0;
    std::uint64_t bytes_read = 0;

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open replay payload");

    EVP_MD_CTX* raw = EVP_MD_CTX_new();
    if (!raw) throw std::runtime_error("SHA context allocation failed");
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(raw, EVP_MD_CTX_free);
    if (EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("SHA init failed");

    std::array<char, 65536> chunk{};
    while (in) {
        in.read(chunk.data(), chunk.size());
        const auto count = in.gcount();
        if (count > 0) {
            if (EVP_DigestUpdate(ctx.get(), chunk.data(),
                                 static_cast<std::size_t>(count)) != 1)
                throw std::runtime_error("SHA update failed");
            bytes_hashed += static_cast<std::uint64_t>(count);
        }
    }
    if (!in.eof()) throw std::runtime_error("replay read failure");

    std::array<unsigned char, 32> hash{};
    unsigned size = 0;
    if (EVP_DigestFinal_ex(ctx.get(), hash.data(), &size) != 1 || size != 32)
        throw std::runtime_error("SHA final failed");
    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (auto byte : hash) hex << std::setw(2) << unsigned(byte);
    if (hex.str() != "45de0da98a8dc6fa19f33eacc52223bc5c3190e956f769f7ff4b253e1d92bad3")
        throw std::runtime_error("replay payload SHA-256 mismatch");

    in.clear();
    in.seekg(0);
    std::array<unsigned char, 72> header{};
    in.read(reinterpret_cast<char*>(header.data()), header.size());
    if (!in || std::memcmp(header.data(), "CGATAC01", 8) ||
        le(header.data() + 8, 4) != 1 || le(header.data() + 12, 4) != 0x01020304u)
        throw std::runtime_error("invalid replay header");
    bytes_read += header.size();

    const auto cells = le(header.data() + 16, 8);
    const auto peaks = le(header.data() + 24, 8);
    const auto words = le(header.data() + 32, 8);
    if (cells != 12016 || peaks != 63751 || words != (cells + 31) / 32 ||
        f.spec.cells > cells || f.spec.features > peaks)
        throw std::runtime_error("replay dimensions incompatible");

    std::vector<unsigned char> row(words * 4);
    for (std::uint64_t peak = 0; peak < f.spec.features; ++peak) {
        in.seekg(72 + peak * words * 4);
        in.read(reinterpret_cast<char*>(row.data()), row.size());
        if (!in) throw std::runtime_error("truncated replay plane");
        bytes_read += row.size();
        for (std::uint64_t cell = 0; cell < f.spec.cells; ++cell)
            f.accessibility[cell * f.spec.features + peak] =
                (le(row.data() + 4 * (cell / 32), 4) >> (cell % 32)) & 1u;
    }

    f.accessibility_after = f.accessibility;
    for (std::uint64_t cell = 0; cell < f.spec.cells; ++cell)
        for (auto i = f.changed_input_offsets[cell]; i < f.changed_input_offsets[cell + 1]; ++i)
            f.accessibility_after[cell * f.spec.features + f.changed_input_features[i]] ^= 1u;

    const auto operational_done = Clock::now();
    const auto reference_begin = operational_done;

    f.root_decision_changes = 0;
    const auto& bank = f.programs.front();
    for (std::uint64_t cell = 0; cell < f.spec.cells; ++cell) {
        const auto* before = f.accessibility.data() + cell * f.spec.features;
        const auto* after = f.accessibility_after.data() + cell * f.spec.features;
        const auto before_nodes = evaluate_condition_nodes(bank, before, f.spec.features);
        const auto after_nodes = evaluate_condition_nodes(bank, after, f.spec.features);
        for (std::size_t edge = 0; edge < f.edge_roots.size(); ++edge) {
            const auto idx = cell * f.edge_roots.size() + edge;
            const auto root = f.edge_roots[edge];
            f.predicates_before[idx] = before_nodes[root];
            f.predicates_after[idx] = after_nodes[root];
            f.root_decision_changes += f.predicates_before[idx] != f.predicates_after[idx];
        }
    }

    if (timings) {
        const auto reference_done = Clock::now();
        ReplayPreparationTimes measured;
        measured.operational_input_load_ms =
            std::chrono::duration<double, std::milli>(operational_done - operational_begin).count();
        measured.reference_recompute_ms =
            std::chrono::duration<double, std::milli>(reference_done - reference_begin).count();
        measured.payload_bytes_hashed = bytes_hashed;
        measured.payload_bytes_read = bytes_read;
        *timings = measured;
    }
}
}
