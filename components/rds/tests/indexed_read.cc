#include <Cellerator/execution/streaming/source.hh>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>
#include "../../../tests/execution/streaming/source_pack_fixture.hh"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>
#include <string_view>
#include <unistd.h>

import cellerator.rds;

namespace {
using namespace cellerator::rds;
namespace memory = cellerator::memory;
namespace streaming = cellerator::execution::streaming;
namespace fs = std::filesystem;
constexpr memory::placement host{memory::domain::host};

void check(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "%s\n", message);
    std::abort();
}

bool read(const RdsFile& file) {
    ReadContext context;
    if (context.prepare_transfer(file) != Status::ok) return false;

    const ColumnRange ranges[]{{2, 2}, {0, 3}, {2, 1}, {4, 1}};
    PreparedRead prepared;
    if (file.prepare_read(ranges, 4, prepared) != Status::ok) return false;
    const auto req = prepared.requirements();
    if (req.rows != 4 || req.columns != 7 || req.entries != 6 || req.offsets != 8) return false;

    double values[6]{};
    int32_t indices[6]{};
    int32_t offsets[8]{};
    CscView result{};
    if (context.read_into(prepared, {values, 6, host}, {indices, 6, host},
                          {offsets, 8, host}, result) != Status::ok) return false;
    const auto request_order_bytes = context.stats().decoded_bytes;
    const double expected_values[]{2.5, -7, 1.25, 0, 2.5, 2.5};
    const int32_t expected_indices[]{1, 2, 0, 3, 1, 1};
    const int32_t expected_offsets[]{0, 1, 2, 4, 4, 5, 6, 6};
    if (result.values != values || result.structure.minor_idx != indices ||
        result.structure.major_ptr != offsets ||
        result.structure.axis != cellerator::tensor::base::CompressedAxis::column ||
        !std::equal(values, values + 6, expected_values) ||
        !std::equal(indices, indices + 6, expected_indices) ||
        !std::equal(offsets, offsets + 8, expected_offsets)) return false;

    values[0] = -19;
    indices[0] = -19;
    offsets[0] = -19;
    if (context.read_into(prepared, {values, 5, host}, {indices, 6, host},
                          {offsets, 8, host}, result) != Status::insufficient_capacity ||
        values[0] != -19 || indices[0] != -19 || offsets[0] != -19) return false;

    ReadContext ordered;
    ReadRun runs[4];
    if (ordered.prepare_transfer(file, TransferMode::numeric) != Status::ok ||
        file.prepare_read(ranges, 4, {runs, 3, host}, prepared) != Status::insufficient_capacity ||
        file.prepare_read(ranges, 4, {runs, 4, host}, prepared) != Status::ok) return false;
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto before = ordered.stats().decoded_bytes;
        if (ordered.read_into(prepared, {values, 6, host}, {indices, 6, host},
                              {offsets, 8, host}, result) != Status::ok ||
            !std::equal(values, values + 6, expected_values) ||
            !std::equal(indices, indices + 6, expected_indices) ||
            !std::equal(offsets, offsets + 8, expected_offsets) ||
            ordered.stats().decoded_bytes - before >= request_order_bytes) return false;
    }
    size_t name_bytes = 0;
    if (ordered.read_names_into(Axis::features, {0, 4}, {}, {}, name_bytes) != Status::unsupported ||
        ordered.prepare_transfer(file, TransferMode::numeric_and_names) != Status::ok ||
        ordered.read_names_into(Axis::features, {0, 4}, {}, {}, name_bytes) != Status::ok ||
        name_bytes != 15) return false;

    const ColumnRange empty_column{1, 1};
    if (file.prepare_read(&empty_column, 1, prepared) != Status::ok ||
        context.read_into(prepared, {nullptr, 0, host}, {nullptr, 0, host},
                          {offsets, 2, host}, result) != Status::ok ||
        offsets[0] != 0 || offsets[1] != 0 || result.structure.nnz != 0) return false;
    if (file.prepare_read(nullptr, 0, prepared) != Status::ok ||
        context.read_into(prepared, {nullptr, 0, host}, {nullptr, 0, host},
                          {offsets, 1, host}, result) != Status::ok ||
        offsets[0] != 0 || result.structure.cols != 0) return false;
    name_bytes = 0;
    AxisName names[4]{};
    char bytes[15]{};
    if (context.read_names_into(Axis::features, {0, 4}, {}, {}, name_bytes) != Status::ok ||
        name_bytes != 15 ||
        context.read_names_into(Axis::features, {0, 4}, {names, 4, host},
                                {bytes, 15, host}, name_bytes) != Status::ok ||
        !names[0].text || std::string_view(names[0].text, names[0].size) != "gene-a" ||
        !names[1].text || std::string_view(names[1].text, names[1].size) != "gene-b" ||
        names[2].text || names[2].size || !names[3].text || names[3].size ||
        names[3].text[0] != '\0') return false;
    if (context.read_names_into(Axis::cells, {3, 2}, {names, 2, host},
                                {bytes, 15, host}, name_bytes) != Status::ok ||
        name_bytes != 14 ||
        !names[0].text || std::string_view(names[0].text, names[0].size) != "cell-d" ||
        !names[1].text || std::string_view(names[1].text, names[1].size) != "cell-e") return false;
    return true;
}

bool fetch(const RdsFile& file) {
    using memory::status;
    ReadContext context;
    if (context.prepare_transfer(file, TransferMode::numeric) != Status::ok) return false;
    FetchReader reader{file, context, 17};
    const auto source = reader.source();
    if (source.info.cells != 5 || source.info.features != 4 || source.info.feature_axis != 17 ||
        source.info.value != streaming::ScalarType::f64 || source.info.index != streaming::ScalarType::i32 ||
        source.info.offset != streaming::ScalarType::i32 || !source.cell_lengths ||
        !source.requirements || !source.read_into) return false;

    const streaming::CellRange cells{1, 3};
    uint64_t lengths[3]{}, nnz = 0;
    if (source.cell_lengths(source.context, cells, lengths, 2) != status::capacity_exceeded ||
        source.cell_lengths(source.context, cells, lengths, 3) != status::success ||
        lengths[0] != 0 || lengths[1] != 1 || lengths[2] != 1 ||
        source.requirements(source.context, cells, &nnz) != status::success || nnz != 2 ||
        source.requirements(source.context, {UINT64_MAX, 1}, &nnz) != status::invalid_argument ||
        context.stats().compressed_bytes || context.stats().decoded_bytes) return false;

    double values[2]{-9, -9};
    int32_t indices[2]{-9, -9}, offsets[4]{-9, -9, -9, -9};
    const streaming::SparseBuffers out{values, indices, offsets, sizeof(values), sizeof(indices), sizeof(offsets)};
    auto short_out = out; short_out.offset_bytes -= sizeof(int32_t);
    if (source.read_into(source.context, cells, short_out, nullptr) != status::capacity_exceeded ||
        values[0] != -9 || indices[0] != -9 || offsets[0] != -9) return false;

    streaming::ReadStats stats;
    const double expected_values[]{2.5, -7};
    const int32_t expected_indices[]{1, 2}, expected_offsets[]{0, 0, 1, 2};
    return source.read_into(source.context, cells, out, &stats) == status::success &&
        values[0] == 2.5 && values[1] == -7 && indices[0] == 1 && indices[1] == 2 &&
        offsets[0] == 0 && offsets[1] == 0 && offsets[2] == 1 && offsets[3] == 2 &&
        stats.decoded_bytes != 0 && cellerator::test::check_source_packet(
            source, cells, expected_values, expected_indices, expected_offsets);
}
}

int main(int argc, char** argv) {
    check(argc == 2, "fixture path required");
    char path[]{"/tmp/cellerator-rds-test.XXXXXX"};
    check(mkdtemp(path) != nullptr, "temporary directory");
    const fs::path dir(path);
    const auto index = dir / "fixture.index";
    const auto copy = dir / "stale.rds";
    const auto corrupt = dir / "corrupt.index";

    IoStats stats{};
    check(build_index(argv[1], index.c_str(), stats) == Status::ok, "index build");
    check(stats.decoded_bytes > 0 && stats.compressed_bytes > 0, "index build stats");
    RdsFile file;
    check(file.open(argv[1], index.c_str()) == Status::ok, "indexed open");
    const auto info = file.info();
    check(info.rows == 4 && info.columns == 5 && info.entries == 4 &&
          info.has_names[0] && info.has_names[1], "fixture metadata");

    bool other_result = false;
    std::thread other([&] { other_result = read(file); });
    const bool local_result = read(file);
    other.join();
    check(local_result && other_result, "independent worker reads");
    check(read(file), "repeat read");
    check(fetch(file), "source adapter lengths, typed payload and read statistics");

    RdsFile rejected;
    fs::copy_file(argv[1], copy);
    check(rejected.open(copy.c_str(), index.c_str()) == Status::stale_index, "stale index");
    fs::copy_file(index, corrupt);
    {
        constexpr auto mode = std::ios::binary | std::ios::in | std::ios::out;
        std::fstream bytes(corrupt, mode);
        bytes.seekg(-1, std::ios::end);
        char last;
        check(bool(bytes.get(last)), "read index byte");
        bytes.seekp(-1, std::ios::end);
        check(bool(bytes.put(static_cast<char>(last ^ 1))), "corrupt index byte");
    }
    check(rejected.open(argv[1], corrupt.c_str()) == Status::invalid_index, "index checksum");
    check(file.open(argv[1], corrupt.c_str()) == Status::invalid_index &&
          read(file), "failed reopen preserves session");
    fs::resize_file(corrupt, 3);
    check(rejected.open(argv[1], corrupt.c_str()) == Status::invalid_index, "truncated index");

    ReadContext context;
    PreparedRead original, reopened;
    const ColumnRange column{0, 1};
    double values[2]{};
    int32_t indices[2]{}, offsets[2]{};
    CscView result{};
    auto load = [&](const PreparedRead& prepared) {
        return context.read_into(prepared, {values, 2, host}, {indices, 2, host},
                                 {offsets, 2, host}, result);
    };
    check(context.prepare_transfer(file) == Status::ok &&
          file.prepare_read(&column, 1, original) == Status::ok &&
          load(original) == Status::ok, "read before reopen");
    FetchReader old_reader{file, context, 17};
    const auto old_source = old_reader.source();
    file.close();
    check(load(original) == Status::invalid_argument, "closed session");
    check(file.open(argv[1], index.c_str()) == Status::ok &&
          context.prepare_transfer(file) == Status::ok &&
          file.prepare_read(&column, 1, reopened) == Status::ok &&
          load(original) == Status::invalid_argument &&
          load(reopened) == Status::ok && values[0] == 1.25, "rebind after reopen");
    uint64_t old_entries = 42;
    check(old_source.requirements(old_source.context, {0, 1}, &old_entries) == memory::status::invalid_state &&
          old_entries == 42, "source adapter preserves file generation");
    fs::remove_all(dir);
}
