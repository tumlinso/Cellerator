#include "../../../bench/benchmark_mutex.hh"

#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <latch>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

import cellerator.rds;

namespace {
namespace rds = cellerator::rds;
namespace memory = cellerator::memory;
using Clock = std::chrono::steady_clock;
using I32 = std::int32_t;
using U32 = std::uint32_t;
using U64 = std::uint64_t;
constexpr std::size_t sweep_cells = 4096;
constexpr std::size_t scatter_cells = 512;
constexpr std::size_t scatter_batches = 64;
constexpr memory::placement host{memory::domain::host};

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void check(rds::Status status, const char* operation) {
    if (status == rds::Status::ok) return;
    throw std::runtime_error(std::string(operation) + " failed with status " +
                             std::to_string(static_cast<int>(status)));
}

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct SourcePages {
    U64 total = 0, resident = 0, device = 0, inode = 0, bytes = 0;
};

SourcePages source_pages(const char* path, std::string_view mode) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    check(fd >= 0, "could not open source for page cache inspection");
    struct stat info{};
    if (fstat(fd, &info) != 0 || info.st_size <= 0) {
        close(fd);
        throw std::runtime_error("could not stat source");
    }
    try {
        const auto bytes = static_cast<std::size_t>(info.st_size);
        const auto page_size = sysconf(_SC_PAGESIZE);
        check(page_size > 0, "could not determine page size");
        const std::size_t pages = (bytes - 1) / page_size + 1;
        std::vector<unsigned char> resident(pages);
        if (mode == "cold") {
            check(posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0,
                  "file-local page cache eviction request failed");
        } else if (mode == "warm") {
            std::vector<char> scratch(1024 * 1024);
            for (;;) {
                const auto got = read(fd, scratch.data(), scratch.size());
                if (got < 0 && errno == EINTR) continue;
                check(got >= 0, "source warm-up read failed");
                if (got == 0) break;
            }
        } else {
            check(mode == "as-is", "expected cold, warm or as-is cache policy");
        }

        void* mapping = mmap(nullptr, bytes, PROT_NONE, MAP_SHARED, fd, 0);
        check(mapping != MAP_FAILED, "source map for mincore failed");
        const auto result = mincore(mapping, bytes, resident.data());
        munmap(mapping, bytes);
        check(result == 0, "mincore failed");
        const auto hot = static_cast<U64>(std::count_if(
            resident.begin(), resident.end(), [](unsigned char page) { return page & 1; }));
        close(fd);
        return {pages, hot, U64(info.st_dev), U64(info.st_ino), bytes};
    } catch (...) {
        close(fd);
        throw;
    }
}

bool verified(SourcePages pages, std::string_view mode) {
    if (mode == "cold") return pages.resident * 100 <= pages.total;
    if (mode == "warm") return pages.resident * 100 >= pages.total * 99;
    return false;
}

std::vector<int> worker_cpus(std::size_t workers) {
    cpu_set_t allowed;
    check(sched_getaffinity(0, sizeof allowed, &allowed) == 0, "could not read CPU allowance");
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE && cpus.size() < workers; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
    }
    check(cpus.size() == workers, "too few schedulable CPUs");
    return cpus;
}

void pin(int cpu) {
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);
    check(sched_setaffinity(0, sizeof one, &one) == 0, "could not pin worker");
}

struct Workload {
    std::vector<rds::ColumnRange> ranges;
    std::vector<rds::PreparedRead> plans;
    std::size_t entries = 0, offsets = 0, cells = 0;
    U64 total_bytes = 0;
};

Workload prepare(const rds::RdsFile& file, bool scatter) {
    Workload work;
    const auto cells = file.info().columns;
    if (scatter) {
        const auto full = cells / scatter_cells;
        check(full >= scatter_batches, "too few full scatter ranges");
        std::vector<U32> blocks(full);
        std::iota(blocks.begin(), blocks.end(), 0);
        std::mt19937_64 random(0x1020304050607080);
        std::shuffle(blocks.begin(), blocks.end(), random);
        for (std::size_t i = 0; i < scatter_batches; ++i)
            work.ranges.push_back({U32(blocks[i] * scatter_cells), scatter_cells});
    } else {
        for (U64 i = 0; i < cells; i += sweep_cells)
            work.ranges.push_back({U32(i), U32(std::min<U64>(sweep_cells, cells - i))});
    }
    work.plans.resize(work.ranges.size());
    for (std::size_t i = 0; i < work.plans.size(); ++i) {
        check(file.prepare_read(&work.ranges[i], 1, work.plans[i]), "prepare_read");
        const auto req = work.plans[i].requirements();
        work.entries = std::max(work.entries, req.entries);
        work.offsets = std::max(work.offsets, req.offsets);
        work.cells += req.columns;
        work.total_bytes += U64(req.entries) * (sizeof(double) + sizeof(I32)) + U64(req.offsets) * sizeof(I32);
    }
    return work;
}

struct Worker {
    rds::ReadContext context;
    std::vector<double> values;
    std::vector<I32> indices, offsets;
    std::exception_ptr failure;
    rds::IoStats stats{};

    Worker(const rds::RdsFile& file, std::size_t entries, std::size_t offset_count)
        : values(entries), indices(entries), offsets(offset_count) {
        check(context.prepare_transfer(file, rds::TransferMode::numeric), "prepare_transfer");
    }

    rds::CscView transfer(const rds::PreparedRead& plan) {
        rds::CscView view{};
        check(context.read_into(plan, {values.data(), values.size(), host},
                                {indices.data(), indices.size(), host},
                                {offsets.data(), offsets.size(), host}, view), "read_into");
        const auto req = plan.requirements();
        check(view.structure.rows == req.rows && view.structure.cols == req.columns &&
              view.structure.nnz == req.entries && view.values == values.data() &&
              offsets.front() == 0 && std::size_t(offsets[req.columns]) == req.entries,
              "returned CSC shape or endpoint mismatch");
        return view;
    }

    void run(const Workload& work, std::size_t rank, std::size_t workers) {
        const auto first = work.plans.size() * rank / workers;
        const auto end = work.plans.size() * (rank + 1) / workers;
        for (auto i = first; i < end; ++i) transfer(work.plans[i]);
        stats = context.stats();
    }
};

struct Measurement {
    SourcePages pages;
    rds::IoStats io{};
    U64 index_bytes = 0, useful_bytes = 0, cells = 0, batches = 0;
    U64 output_capacity_bytes_per_worker = 0;
    double setup_seconds = 0, cache_seconds = 0, elapsed_seconds = 0;
    std::vector<int> cpus;
};

Measurement read_batches(const char* source, const char* index, std::string_view mode,
                         bool scatter, std::size_t workers) {
    Measurement result;
    const auto setup = Clock::now();
    rds::RdsFile file;
    check(file.open(source, index), "open");
    const auto workload = prepare(file, scatter);
    result.cpus = worker_cpus(workers);
    result.index_bytes = file.info().index_bytes;
    result.useful_bytes = workload.total_bytes;
    result.cells = workload.cells;
    result.batches = workload.plans.size();
    result.output_capacity_bytes_per_worker = U64(workload.entries) * (sizeof(double) + sizeof(I32)) +
                                              U64(workload.offsets) * sizeof(I32);

    std::vector<std::unique_ptr<Worker>> context;
    for (std::size_t i = 0; i < workers; ++i)
        context.push_back(std::make_unique<Worker>(file, workload.entries, workload.offsets));
    std::latch ready(workers);
    std::latch start(1);
    std::vector<std::jthread> threads;
    try {
        for (std::size_t i = 0; i < workers; ++i) {
            threads.emplace_back([&, i] {
                auto& reader = *context[i];
                try { pin(result.cpus[i]); }
                catch (...) { reader.failure = std::current_exception(); }
                ready.count_down();
                start.wait();
                if (reader.failure) return;
                try { reader.run(workload, i, workers); }
                catch (...) { reader.failure = std::current_exception(); }
            });
        }
        ready.wait();
        result.setup_seconds = elapsed(setup);
        const auto cache = Clock::now();
        result.pages = source_pages(source, mode);
        result.cache_seconds = elapsed(cache);
    } catch (...) {
        start.count_down();
        throw;
    }

    const auto timer = Clock::now();
    start.count_down();
    for (auto& thread : threads) thread.join();
    result.elapsed_seconds = elapsed(timer);
    for (const auto& reader : context) {
        if (reader->failure) std::rethrow_exception(reader->failure);
        result.io.compressed_bytes += reader->stats.compressed_bytes;
        result.io.decoded_bytes += reader->stats.decoded_bytes;
    }
    return result;
}

struct Entry {
    U32 row, column;
    double value;
};

Entry parse_entry(std::string_view line) {
    const char* p = line.data();
    const char* end = p + line.size();
    auto parse = [&](auto& number) {
        while (p < end && (*p == ' ' || *p == '\t')) ++p;
        const auto read = std::from_chars(p, end, number);
        check(read.ec == std::errc{}, "invalid Matrix Market integer");
        p = read.ptr;
    };
    U32 row, column;
    std::int64_t value;
    parse(row);
    parse(column);
    parse(value);
    check(row && column, "Matrix Market coordinates must be one-based");
    return {row - 1, column - 1, double(value)};
}

class ColumnOrderedMtx {
public:
    ColumnOrderedMtx(const char* path, const rds::FileInfo& info) : input_(path) {
        check(input_.is_open() && bool(std::getline(input_, line_)) &&
              line_ == "%%MatrixMarket matrix coordinate integer general", "unexpected Matrix Market header");
        do { check(bool(std::getline(input_, line_)), "missing Matrix Market dimensions"); }
        while (!line_.empty() && line_.front() == '%');
        unsigned rows, cols;
        unsigned long long entries;
        check(std::sscanf(line_.c_str(), "%u %u %llu", &rows, &cols, &entries) == 3 &&
              rows == info.rows && cols == info.columns && entries == info.entries,
              "Matrix Market dimensions or entry count differ");
        first_ = input_.tellg();
        input_.seekg(0, std::ios::end);
        end_ = input_.tellg();
        check(first_ >= 0 && end_ > first_, "could not determine Matrix Market data extent");
    }

    void verify(rds::ColumnRange range, const rds::CscView& view, U32 destination_col) {
        std::streamoff low = first_, high = end_;
        while (low < high) {
            const auto middle = low + (high - low) / 2;
            seek_line(middle);
            if (std::getline(input_, line_) && parse_entry(line_).column < range.start) low = middle + 1;
            else high = middle;
        }
        seek_line(low);
        auto entry_index = view.structure.major_ptr[destination_col];
        const U64 limit = U64(range.start) + range.count;
        U64 source_col = range.start;
        while (std::getline(input_, line_)) {
            const auto entry = parse_entry(line_);
            if (entry.column >= limit) break;
            check(entry.column >= source_col, "oracle Matrix Market entries are not column ordered");
            while (source_col < entry.column) {
                ++source_col;
                ++destination_col;
                check(view.structure.major_ptr[destination_col] == entry_index, "column offset differs");
            }
            check(entry_index < view.structure.major_ptr[destination_col + 1] &&
                  view.structure.minor_idx[entry_index] == I32(entry.row) &&
                  view.values[entry_index] == entry.value, "entry differs from Matrix Market");
            ++entry_index;
        }
        while (source_col < limit) {
            ++source_col;
            ++destination_col;
            check(view.structure.major_ptr[destination_col] == entry_index, "final column offset differs");
        }
    }

private:
    void seek_line(std::streamoff position) {
        input_.clear();
        if (position == first_) input_.seekg(position);
        else {
            input_.seekg(position - 1);
            input_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }

    std::ifstream input_;
    std::string line_;
    std::streamoff first_ = 0, end_ = 0;
};

Measurement verify_against_mtx(const char* source, const char* index, const char* mtx) {
    Measurement result;
    const auto timer = Clock::now();
    rds::RdsFile file;
    check(file.open(source, index), "open");
    const auto info = file.info();
    check(info.columns >= scatter_cells * 3, "too few cells for real-file verification plan");
    const auto scattered = prepare(file, true);
    const rds::ColumnRange ranges[]{{U32(info.columns - scatter_cells), scatter_cells},
                                    {0, scatter_cells},
                                    {info.columns / 2, scatter_cells},
                                    {0, scatter_cells}};
    rds::PreparedRead plan{};
    rds::ReadRun runs[std::size(ranges)];
    check(file.prepare_read(ranges, std::size(ranges), {runs, std::size(runs), host}, plan), "prepare_read");
    const auto req = plan.requirements();
    Worker reader(file, std::max(req.entries, scattered.entries), std::max(req.offsets, scattered.offsets));
    ColumnOrderedMtx reference(mtx, info);
    for (std::size_t i = 0; i < scattered.plans.size(); ++i)
        reference.verify(scattered.ranges[i], reader.transfer(scattered.plans[i]), 0);
    const auto view = reader.transfer(plan);
    U32 column = 0;
    for (const auto range : ranges) {
        reference.verify(range, view, column);
        column += range.count;
    }
    result.elapsed_seconds = elapsed(timer);
    result.pages = source_pages(source, "as-is");
    result.io = reader.context.stats();
    result.index_bytes = info.index_bytes;
    result.cells = scattered.cells + req.columns;
    result.batches = scattered.plans.size() + 1;
    result.useful_bytes = scattered.total_bytes +
        U64(req.entries) * (sizeof(double) + sizeof(I32)) + U64(req.offsets) * sizeof(I32);
    return result;
}

Measurement read_index(const char* source, const char* index, std::string_view mode, bool build) {
    Measurement result;
    result.cpus = worker_cpus(1);
    pin(result.cpus.front());
    const auto cache = Clock::now();
    result.pages = source_pages(source, mode);
    result.cache_seconds = elapsed(cache);
    rds::RdsFile file;
    const auto timer = Clock::now();
    if (build) check(rds::build_index(source, index, result.io), "build_index");
    else check(file.open(source, index), "open");
    result.elapsed_seconds = elapsed(timer);
    if (build) {
        struct stat st{};
        check(stat(index, &st) == 0, "could not stat new index");
        result.index_bytes = st.st_size;
    } else result.index_bytes = file.info().index_bytes;
    return result;
}

void report(const char* operation, const char* cache, const Measurement& result) {
    struct rusage usage{};
    check(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
    const auto& pages = result.pages;
    std::printf("{\"status\":\"ok\",\"operation\":\"%s\",\"cache_request\":\"%s\","
                "\"cache_verified\":%s,\"source_device\":%llu,\"source_inode\":%llu,"
                "\"source_bytes\":%llu,\"pages_total\":%llu,\"pages_resident_before\":%llu,"
                "\"index_bytes\":%llu,\"cells\":%llu,\"batches\":%llu,"
                "\"output_capacity_bytes_per_worker\":%llu,\"useful_bytes\":%llu,"
                "\"compressed_bytes\":%llu,\"decoded_bytes\":%llu,\"setup_seconds\":%.9f,"
                "\"cache_seconds\":%.9f,\"seconds\":%.9f,\"useful_mib_per_second\":%.3f,"
                "\"peak_rss_kib\":%ld,\"cpus\":[",
                operation, cache, verified(pages, cache) ? "true" : "false",
                (unsigned long long)pages.device, (unsigned long long)pages.inode,
                (unsigned long long)pages.bytes, (unsigned long long)pages.total,
                (unsigned long long)pages.resident, (unsigned long long)result.index_bytes,
                (unsigned long long)result.cells, (unsigned long long)result.batches,
                (unsigned long long)result.output_capacity_bytes_per_worker,
                (unsigned long long)result.useful_bytes, (unsigned long long)result.io.compressed_bytes,
                (unsigned long long)result.io.decoded_bytes, result.setup_seconds, result.cache_seconds,
                result.elapsed_seconds, result.useful_bytes / (result.elapsed_seconds * 1048576),
                usage.ru_maxrss);
    for (std::size_t i = 0; i < result.cpus.size(); ++i)
        std::printf("%s%d", i ? "," : "", result.cpus[i]);
    std::puts("]}");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 6) {
            std::fprintf(stderr, "usage: %s <build|reopen|sweep|scatter|verify> SOURCE INDEX "
                                 "<cold|warm|as-is> [1|4|ORACLE_MTX]\n", argv[0]);
            return 2;
        }
        const std::string_view operation(argv[1]), cache(argv[4]);
        check(cache == "cold" || cache == "warm" || cache == "as-is", "unknown cache policy");
        if (operation == "verify") {
            check(argc == 6 && cache == "as-is", "verify needs ORACLE_MTX and as-is cache policy");
            report(argv[1], argv[4], verify_against_mtx(argv[2], argv[3], argv[5]));
            return 0;
        }
        cellerator::bench::benchmark_mutex_guard mutex("cellerator-rds-access");
        if (operation == "build" || operation == "reopen") {
            check(argc == 5, "build/reopen take no trailing argument");
            report(argv[1], argv[4], read_index(argv[2], argv[3], cache, operation == "build"));
        } else if (operation == "sweep" || operation == "scatter") {
            check(argc == 6 && (std::strcmp(argv[5], "1") == 0 || std::strcmp(argv[5], "4") == 0),
                  "sweep/scatter require 1 or 4 workers");
            report(argv[1], argv[4], read_batches(argv[2], argv[3], cache, operation == "scatter", argv[5][0] - '0'));
        } else throw std::runtime_error("unknown operation");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rds access benchmark: %s\n", e.what());
        std::puts("{\"status\":\"failed\"}");
        return 1;
    }
}
