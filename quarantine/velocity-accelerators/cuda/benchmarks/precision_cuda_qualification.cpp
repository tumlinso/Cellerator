#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <sys/resource.h>
#include <utility>
#include <variant>
#include <vector>

import velocity.cuda;

namespace {

using Clock = std::chrono::steady_clock;
namespace ac = anndata::cuda;
namespace vc = velocity::cuda;

constexpr std::uint64_t max_fixture_bytes = 512ull * 1024 * 1024;
constexpr std::uint64_t max_accounted_bytes = 1536ull * 1024 * 1024;
constexpr std::size_t max_workspace_bytes = 64ull * 1024 * 1024;
constexpr std::size_t max_cells = 8192, max_genes = 8192;
constexpr std::uint64_t max_elements = 4096ull * 2048;
constexpr std::size_t sink_slots = 3;
constexpr std::size_t allocation_slots = 4096;

constinit std::atomic<bool> count_host_allocations{false};
constinit std::atomic<std::uint64_t> host_allocation_calls{0}, host_allocation_bytes{0};

void* allocate(std::size_t bytes, std::size_t alignment) {
    void* p = nullptr;
    for (;;) {
        if (alignment <= alignof(std::max_align_t)) p = std::malloc(bytes ? bytes : 1);
        else {
            const auto size = bytes ? bytes : alignment;
            if (size > std::numeric_limits<std::size_t>::max() - (alignment - 1)) throw std::bad_alloc();
            const auto rounded = (size + alignment - 1) & ~(alignment - 1);
            p = std::aligned_alloc(alignment, rounded);
        }
        if (p) break;
        auto handler = std::get_new_handler();
        if (!handler) throw std::bad_alloc();
        handler();
    }
    if (count_host_allocations.load(std::memory_order_relaxed)) {
        host_allocation_calls.fetch_add(1, std::memory_order_relaxed);
        host_allocation_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    return p;
}

struct DeviceAllocation {
    void* pointer = nullptr;
    std::size_t bytes = 0;
};
constinit std::array<DeviceAllocation, allocation_slots> device_allocations{};
constinit std::atomic_flag device_allocation_lock = ATOMIC_FLAG_INIT;
constinit std::atomic<bool> count_device_allocations{false};
constinit std::atomic<std::uint64_t> device_malloc_calls{0}, device_malloc_bytes{0};
constinit std::atomic<std::uint64_t> device_free_calls{0}, device_free_bytes{0};
constinit std::atomic<std::uint64_t> device_live_bytes{0}, device_peak_bytes{0};
constinit std::atomic<std::uint64_t> device_tracking_overflows{0};

void lock_device_allocations() noexcept {
    while (device_allocation_lock.test_and_set(std::memory_order_acquire)) {}
}
void unlock_device_allocations() noexcept { device_allocation_lock.clear(std::memory_order_release); }

void track_device_allocation(void* pointer, std::size_t bytes) noexcept {
    if (!count_device_allocations.load(std::memory_order_relaxed)) return;
    device_malloc_calls.fetch_add(1, std::memory_order_relaxed);
    device_malloc_bytes.fetch_add(bytes, std::memory_order_relaxed);
    lock_device_allocations();
    DeviceAllocation* empty = nullptr;
    for (auto& item : device_allocations) {
        if (item.pointer == pointer) { item.bytes = bytes; empty = nullptr; break; }
        if (!item.pointer && !empty) empty = &item;
    }
    if (empty) {
        empty->pointer = pointer;
        empty->bytes = bytes;
        const auto live = device_live_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        auto peak = device_peak_bytes.load(std::memory_order_relaxed);
        while (live > peak && !device_peak_bytes.compare_exchange_weak(peak, live,
               std::memory_order_relaxed, std::memory_order_relaxed)) {}
    } else device_tracking_overflows.fetch_add(1, std::memory_order_relaxed);
    unlock_device_allocations();
}

void track_device_free(void* pointer) noexcept {
    if (!count_device_allocations.load(std::memory_order_relaxed)) return;
    device_free_calls.fetch_add(1, std::memory_order_relaxed);
    lock_device_allocations();
    for (auto& item : device_allocations) if (item.pointer == pointer) {
        device_free_bytes.fetch_add(item.bytes, std::memory_order_relaxed);
        device_live_bytes.fetch_sub(item.bytes, std::memory_order_relaxed);
        item = {};
        break;
    }
    unlock_device_allocations();
}

} // namespace

// ELF symbol interposition observes cudaMalloc/cudaFree calls made by the
// executable and linked velocity/anndata objects. The runtime itself is called
// through RTLD_NEXT. Unsupported linkers or hidden shared-library references
// are visible in the emitted coverage counters and zero-allocation status.
extern "C" cudaError_t cudaMalloc(void** pointer, std::size_t bytes) {
    using Function = cudaError_t (*)(void**, std::size_t);
    static auto real = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "cudaMalloc"));
    if (!real) return cudaErrorUnknown;
    auto status = real(pointer, bytes);
    if (status == cudaSuccess) track_device_allocation(pointer ? *pointer : nullptr, bytes);
    return status;
}

extern "C" cudaError_t cudaFree(void* pointer) {
    using Function = cudaError_t (*)(void*);
    static auto real = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "cudaFree"));
    if (!real) return cudaErrorUnknown;
    auto status = real(pointer);
    if (status == cudaSuccess && pointer) track_device_free(pointer);
    return status;
}

void* operator new(std::size_t n) { return allocate(n, alignof(std::max_align_t)); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocate(n, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new[](n); } catch (...) { return nullptr; }
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return ::operator new(n, a); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return ::operator new[](n, a); } catch (...) { return nullptr; }
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }

namespace {

struct Csr {
    std::size_t rows = 0, columns = 0;
    std::vector<std::int64_t> offsets;
    std::vector<std::int32_t> indices;
    std::vector<float> values;
    std::size_t bytes() const {
        return offsets.capacity()*sizeof(std::int64_t)
            + indices.capacity()*sizeof(std::int32_t) + values.capacity()*sizeof(float);
    }
};

struct Fixture {
    Csr spliced, unspliced, weights;
    std::uint64_t file_bytes = 0;
};

std::uint64_t add_bytes(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::runtime_error("fixture byte count overflows");
    return a + b;
}
std::uint64_t mul_bytes(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::runtime_error("fixture byte count overflows");
    return a*b;
}
std::size_t checked_size(std::uint64_t n, const char* what) {
    if (n > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error(std::string(what) + " exceeds host size_t");
    return static_cast<std::size_t>(n);
}
void read_exact(std::ifstream& file, void* output, std::size_t bytes) {
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("fixture section exceeds stream limits");
    if (bytes && !file.read(static_cast<char*>(output), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("truncated canonical CSR fixture");
}

void validate_csr(const Csr& matrix, bool normalized_rows) {
    if (!matrix.rows || !matrix.columns || matrix.offsets.size() != matrix.rows + 1
        || matrix.indices.size() != matrix.values.size() || matrix.offsets.front() != 0
        || matrix.offsets.back() != static_cast<std::int64_t>(matrix.values.size()))
        throw std::runtime_error("invalid CSR dimensions or offsets");
    for (std::size_t i = 0; i < matrix.rows; ++i) {
        if (matrix.offsets[i] < 0 || matrix.offsets[i] > matrix.offsets[i + 1]
            || matrix.offsets[i + 1] > static_cast<std::int64_t>(matrix.values.size()))
            throw std::runtime_error("CSR offsets are not monotone");
        std::int32_t previous = -1;
        double row_sum = 0;
        for (auto p = static_cast<std::size_t>(matrix.offsets[i]);
             p < static_cast<std::size_t>(matrix.offsets[i + 1]); ++p) {
            const auto column = matrix.indices[p];
            const auto value = matrix.values[p];
            if (column < 0 || static_cast<std::size_t>(column) >= matrix.columns || column <= previous
                || !std::isfinite(value) || value <= 0)
                throw std::runtime_error("CSR entries must be finite, positive, sorted, and unique");
            previous = column;
            row_sum += value;
        }
        if (normalized_rows && (matrix.offsets[i] == matrix.offsets[i + 1]
            || std::abs(row_sum - 1) > 2e-5))
            throw std::runtime_error("graph rows must be nonempty and normalized");
    }
}

Csr read_matrix(std::ifstream& file, std::size_t rows, std::size_t columns,
                std::uint64_t nnz, bool normalized_rows) {
    Csr matrix;
    matrix.rows = rows;
    matrix.columns = columns;
    const auto count = checked_size(nnz, "nnz");
    if (rows == std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("row count overflows offsets");
    matrix.offsets.resize(rows + 1);
    matrix.indices.resize(count);
    matrix.values.resize(count);
    read_exact(file, matrix.offsets.data(), matrix.offsets.size()*sizeof(std::int64_t));
    read_exact(file, matrix.indices.data(), matrix.indices.size()*sizeof(std::int32_t));
    read_exact(file, matrix.values.data(), matrix.values.size()*sizeof(float));
    validate_csr(matrix, normalized_rows);
    return matrix;
}

Fixture load_fixture(const std::filesystem::path& path) {
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error("canonical fixture requires a little-endian host");
    if (!std::numeric_limits<float>::is_iec559 || sizeof(float) != 4)
        throw std::runtime_error("canonical fixture requires IEEE-754 binary32");
    std::error_code error;
    const auto file_bytes = std::filesystem::file_size(path, error);
    if (error) throw std::runtime_error("cannot stat input fixture: " + error.message());
    if (file_bytes < 40 || file_bytes > max_fixture_bytes)
        throw std::runtime_error("fixture must be between 40 bytes and 512 MiB");
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open input fixture");
    std::uint64_t header[5]{};
    read_exact(file, header, sizeof header);
    const auto rows = checked_size(header[0], "N");
    const auto columns = checked_size(header[1], "G");
    if (!rows || !columns || rows > max_cells || columns > max_genes
        || mul_bytes(rows, columns) > max_elements
        || rows > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())
        || columns > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error("fixture dimensions exceed the bounded 8192-per-axis and 8388608-element probe");
    std::uint64_t expected = 40;
    const auto offsets = mul_bytes(rows + 1, sizeof(std::int64_t));
    for (auto nnz : {header[2], header[3], header[4]})
        expected = add_bytes(expected, add_bytes(offsets,
            mul_bytes(nnz, sizeof(std::int32_t) + sizeof(float))));
    if (expected != file_bytes) throw std::runtime_error("fixture byte count does not match its header");
    Fixture out;
    out.file_bytes = file_bytes;
    out.spliced = read_matrix(file, rows, columns, header[2], false);
    out.unspliced = read_matrix(file, rows, columns, header[3], false);
    out.weights = read_matrix(file, rows, rows, header[4], true);
    if (file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("fixture has trailing bytes");
    return out;
}

enum class Profile { half, allhalf, fp32, fp64, legacy };
enum class Method { steady, stochastic };
enum class Storage { float16, float32, float64 };
struct Config {
    std::filesystem::path input, output_json;
    Profile profile = Profile::half;
    Method method = Method::stochastic;
    std::size_t block = 64;
};

Profile parse_profile(std::string_view name) {
    if (name == "half") return Profile::half;
    if (name == "allhalf") return Profile::allhalf;
    if (name == "fp32") return Profile::fp32;
    if (name == "fp64") return Profile::fp64;
    if (name == "legacy") return Profile::legacy;
    throw std::runtime_error("profile must be half, allhalf, fp32, fp64, or legacy");
}
Method parse_method(std::string_view name) {
    if (name == "steady") return Method::steady;
    if (name == "stochastic") return Method::stochastic;
    throw std::runtime_error("method must be steady or stochastic");
}
Storage storage(Profile profile, bool weights) {
    if (profile == Profile::allhalf) return Storage::float16;
    if (profile == Profile::half && !weights) return Storage::float16;
    if (profile == Profile::fp64) return Storage::float64;
    return Storage::float32;
}
velocity::ComputePrecision compute(Profile profile) {
    return profile == Profile::fp64 || profile == Profile::legacy
        ? velocity::ComputePrecision::float64 : velocity::ComputePrecision::float32;
}
const char* profile_name(Profile profile) {
    switch (profile) {
    case Profile::half: return "half";
    case Profile::allhalf: return "allhalf";
    case Profile::fp32: return "fp32";
    case Profile::fp64: return "fp64";
    case Profile::legacy: return "legacy";
    }
    return "invalid";
}
const char* storage_name(Storage type) {
    switch (type) {
    case Storage::float16: return "float16";
    case Storage::float32: return "float32";
    case Storage::float64: return "float64";
    }
    return "invalid";
}
std::size_t storage_size(Storage type) {
    return type == Storage::float16 ? sizeof(anndata::Float16)
        : type == Storage::float32 ? sizeof(float) : sizeof(double);
}
ac::ScalarType scalar_type(Storage type) {
    return type == Storage::float16 ? ac::ScalarType::float16
        : type == Storage::float32 ? ac::ScalarType::float32 : ac::ScalarType::float64;
}
std::size_t compute_size(velocity::ComputePrecision precision) {
    return precision == velocity::ComputePrecision::float32 ? sizeof(float) : sizeof(double);
}
ac::ScalarType compute_scalar_type(velocity::ComputePrecision precision) {
    return precision == velocity::ComputePrecision::float32
        ? ac::ScalarType::float32 : ac::ScalarType::float64;
}
ac::ScalarType output_scalar_type(Profile profile) {
    return profile == Profile::legacy ? ac::ScalarType::float32
        : compute_scalar_type(compute(profile));
}
std::size_t output_size(Profile profile) {
    return profile == Profile::legacy ? sizeof(float) : compute_size(compute(profile));
}
const char* compute_name(velocity::ComputePrecision precision) {
    return precision == velocity::ComputePrecision::float32 ? "float32" : "float64";
}

void help() {
    std::cout << "usage: velocity-precision-cuda-qualification --input FILE "
                 "--profile half|allhalf|fp32|fp64|legacy --method steady|stochastic "
                 "[--block 16|32|64] [--output-json FILE]\n";
}
Config parse_args(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option = argv[i];
        if (option == "--help" || option == "-h") { help(); std::exit(0); }
        if (++i == argc) throw std::runtime_error("option requires a value");
        const std::string_view value = argv[i];
        if (option == "--input") config.input = value;
        else if (option == "--profile") config.profile = parse_profile(value);
        else if (option == "--method") config.method = parse_method(value);
        else if (option == "--output-json") config.output_json = value;
        else if (option == "--block") {
            const auto [end, error] = std::from_chars(value.data(), value.data()+value.size(), config.block);
            if (error != std::errc{} || end != value.data()+value.size())
                throw std::runtime_error("invalid block size");
        } else throw std::runtime_error("unknown option: " + std::string(option));
    }
    if (config.input.empty() || (config.block != 16 && config.block != 32 && config.block != 64))
        throw std::runtime_error("input, profile, method, and block 16, 32, or 64 are required");
    return config;
}

struct AllocationStats { std::uint64_t calls = 0, bytes = 0; };
AllocationStats host_allocations() {
    return {host_allocation_calls.load(std::memory_order_relaxed),
            host_allocation_bytes.load(std::memory_order_relaxed)};
}
struct DeviceStats {
    std::uint64_t malloc_calls = 0, malloc_bytes = 0;
    std::uint64_t free_calls = 0, free_bytes = 0;
    std::uint64_t live_bytes = 0, peak_bytes = 0, overflowed_records = 0;
};
DeviceStats device_stats() {
    return {device_malloc_calls.load(std::memory_order_relaxed), device_malloc_bytes.load(std::memory_order_relaxed),
        device_free_calls.load(std::memory_order_relaxed), device_free_bytes.load(std::memory_order_relaxed),
        device_live_bytes.load(std::memory_order_relaxed), device_peak_bytes.load(std::memory_order_relaxed),
        device_tracking_overflows.load(std::memory_order_relaxed)};
}
DeviceStats difference(DeviceStats a, DeviceStats b) {
    return {b.malloc_calls-a.malloc_calls, b.malloc_bytes-a.malloc_bytes,
        b.free_calls-a.free_calls, b.free_bytes-a.free_bytes, b.live_bytes, b.peak_bytes,
        b.overflowed_records-a.overflowed_records};
}
AllocationStats difference(AllocationStats a, AllocationStats b) {
    return {b.calls-a.calls, b.bytes-a.bytes};
}

struct Memory { std::uint64_t rss = 0, high_water = 0, rusage_peak = 0; };
Memory memory() {
    std::ifstream status("/proc/self/status");
    Memory out;
    std::string line;
    while (std::getline(status, line)) {
        auto read_kib = [&](std::string_view key) {
            if (line.compare(0, key.size(), key) != 0) return std::uint64_t{};
            const char* first = line.data() + key.size();
            while (first != line.data()+line.size() && (*first == ' ' || *first == '\t')) ++first;
            std::uint64_t value = 0;
            const auto [last, error] = std::from_chars(first, line.data()+line.size(), value);
            return error == std::errc{} && last != first ? value*1024 : std::uint64_t{};
        };
        if (line.starts_with("VmRSS:")) out.rss = read_kib("VmRSS:");
        else if (line.starts_with("VmHWM:")) out.high_water = read_kib("VmHWM:");
    }
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage failed");
    if (usage.ru_maxrss > 0) out.rusage_peak = std::uint64_t(usage.ru_maxrss)*1024;
    return out;
}
double seconds(Clock::time_point first, Clock::time_point last) {
    return std::chrono::duration<double>(last-first).count();
}
void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void event_create(cudaEvent_t& event) { cuda_check(cudaEventCreate(&event), "cudaEventCreate"); }
double event_seconds(cudaEvent_t first, cudaEvent_t last) {
    float milliseconds = 0;
    cuda_check(cudaEventElapsedTime(&milliseconds, first, last), "cudaEventElapsedTime");
    return double(milliseconds)*1e-3;
}

struct StreamOwner {
    cudaStream_t native = nullptr;
    int device = -1;
    StreamOwner() {
        cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        cuda_check(cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }
    ~StreamOwner() {
        if (native) { cudaStreamSynchronize(native); cudaStreamDestroy(native); }
    }
    ac::Stream view() const noexcept { return {reinterpret_cast<void*>(native), device}; }
};

template<class T> struct TypedValues { std::vector<T> data; };
using PackedValues = std::variant<TypedValues<anndata::Float16>, TypedValues<float>, TypedValues<double>>;
struct PackedCsr {
    PackedValues values;
    std::size_t bytes = 0;
};
template<class T> PackedCsr pack_values(const Csr& source) {
    TypedValues<T> values;
    values.data.reserve(source.values.size());
    for (auto x : source.values) {
        if constexpr (std::is_same_v<T, anndata::Float16>) values.data.push_back(anndata::Float16::from_float(x));
        else values.data.push_back(T(x));
    }
    const auto bytes = values.data.capacity()*sizeof(T);
    return {std::move(values), bytes};
}
PackedCsr pack_values(const Csr& source, Storage type) {
    if (type == Storage::float16) return pack_values<anndata::Float16>(source);
    if (type == Storage::float32) return pack_values<float>(source);
    return pack_values<double>(source);
}
template<class F> decltype(auto) visit_values(PackedValues& values, F&& function) {
    return std::visit([&](auto& typed) -> decltype(auto) { return function(typed.data); }, values);
}

struct DeviceCsr {
    ac::DeviceBuffer values, indices, offsets;
    vc::MatrixView matrix;
    std::size_t bytes = 0;
};

DeviceCsr allocate_matrix(const Csr& source, Storage type, int device) {
    DeviceCsr out;
    const auto value_bytes = source.values.size()*storage_size(type);
    const auto index_bytes = source.indices.size()*sizeof(std::int32_t);
    const auto offset_bytes = source.offsets.size()*sizeof(std::int64_t);
    out.values = ac::DeviceBuffer(value_bytes, device);
    out.indices = ac::DeviceBuffer(index_bytes, device);
    out.offsets = ac::DeviceBuffer(offset_bytes, device);
    out.bytes = value_bytes+index_bytes+offset_bytes;
    out.matrix = ac::SparseView{out.values.span(), out.indices.span(), out.offsets.span(), source.rows,
        source.columns, source.values.size(), scalar_type(type), ac::ScalarType::int32,
        ac::ScalarType::int64, ac::SparseLayout::csr};
    return out;
}
void upload_matrix(DeviceCsr& out, const Csr& source, PackedCsr& packed, ac::Stream stream) {
    const auto index_bytes = source.indices.size()*sizeof(std::int32_t);
    const auto offset_bytes = source.offsets.size()*sizeof(std::int64_t);
    visit_values(packed.values, [&](auto& values) {
        using T = typename std::remove_reference_t<decltype(values)>::value_type;
        if (!values.empty()) ac::copy_to_device(out.values.span(), 0, values.data(), values.size()*sizeof(T), stream);
    });
    if (index_bytes) ac::copy_to_device(out.indices.span(), 0, source.indices.data(), index_bytes, stream);
    if (offset_bytes) ac::copy_to_device(out.offsets.span(), 0, source.offsets.data(), offset_bytes, stream);
}

struct DeviceOutput {
    ac::DeviceBuffer owner;
    ac::DenseView view;
    DeviceOutput(std::size_t rows, std::size_t columns, ac::ScalarType type, int device)
        : owner(rows*columns*ac::scalar_size(type), device),
          view{owner.span(), 0, rows, columns, 1, std::ptrdiff_t(rows), type} {}
};

struct Checksum {
    std::uint64_t values = 0;
    double sum = 0, absolute_sum = 0;
    bool finite = true;
    void add(double value) {
        finite = finite && std::isfinite(value);
        sum += value;
        absolute_sum += std::abs(value);
        ++values;
    }
};

struct HostSlot {
    void* velocity = nullptr;
    void* residual = nullptr;
    std::size_t capacity_bytes = 0;
    std::size_t first = 0, count = 0;
    cudaEvent_t begin = nullptr, end = nullptr;
    bool pending = false;
    ~HostSlot() {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (velocity) cudaFreeHost(velocity);
        if (residual) cudaFreeHost(residual);
    }
};

struct SinkState {
    std::array<HostSlot, sink_slots> slots;
    Checksum velocity, residual;
    std::size_t rows = 0, genes = 0, next_gene = 0, next_block = 0;
    std::size_t output_width = 0;
    bool stochastic = false;
    ac::ScalarType output_type = ac::ScalarType::float32;
    std::uint64_t d2h_bytes = 0;
    double d2h_event_seconds = 0;

    SinkState(std::size_t n, std::size_t g, std::size_t block, std::size_t width,
              bool is_stochastic, ac::ScalarType type)
        : rows(n), genes(g), output_width(width), stochastic(is_stochastic), output_type(type) {
        const auto bytes = rows*block*output_width;
        for (auto& slot : slots) {
            slot.capacity_bytes = bytes;
            cuda_check(cudaHostAlloc(&slot.velocity, bytes, cudaHostAllocDefault), "cudaHostAlloc velocity sink");
            if (stochastic)
                cuda_check(cudaHostAlloc(&slot.residual, bytes, cudaHostAllocDefault), "cudaHostAlloc residual sink");
            event_create(slot.begin);
            event_create(slot.end);
        }
    }

    template<class T> void checksum(const void* values, std::size_t count, Checksum& sum) {
        const auto* data = static_cast<const T*>(values);
        for (std::size_t i = 0; i < count; ++i) sum.add(double(data[i]));
    }
    void drain(HostSlot& slot) {
        if (!slot.pending) return;
        cuda_check(cudaEventSynchronize(slot.end), "cudaEventSynchronize output block");
        d2h_event_seconds += event_seconds(slot.begin, slot.end);
        const auto count = rows*slot.count;
        if (output_type == ac::ScalarType::float32) {
            checksum<float>(slot.velocity, count, velocity);
            if (stochastic) checksum<float>(slot.residual, count, residual);
        } else {
            checksum<double>(slot.velocity, count, velocity);
            if (stochastic) checksum<double>(slot.residual, count, residual);
        }
        slot.pending = false;
    }
};

void sink_write(void* context, const vc::OutputBlock& block, ac::Stream stream) {
    auto& state = *static_cast<SinkState*>(context);
    if (block.first != state.next_gene || !block.count || block.count > state.genes-block.first)
        throw std::runtime_error("output sink received an out-of-order gene block");
    auto check_view = [&](ac::DenseView view) {
        if (!view.values.data || view.rows != state.rows || view.columns != block.count
            || view.dtype != state.output_type || view.row_stride != 1
            || view.column_stride != std::ptrdiff_t(state.rows))
            throw std::runtime_error("output sink received an unexpected block-local view");
    };
    check_view(block.outputs.velocity);
    if (state.stochastic) check_view(block.outputs.second_residual);
    auto& slot = state.slots[state.next_block % sink_slots];
    state.drain(slot);
    slot.first = block.first;
    slot.count = block.count;
    const auto bytes = state.rows*block.count*state.output_width;
    const auto native = reinterpret_cast<cudaStream_t>(stream.native);
    cuda_check(cudaEventRecord(slot.begin, native), "cudaEventRecord output start");
    const auto* velocity = static_cast<const char*>(block.outputs.velocity.values.data)
        + block.outputs.velocity.origin;
    cuda_check(cudaMemcpyAsync(slot.velocity, velocity, bytes, cudaMemcpyDeviceToHost, native),
               "cudaMemcpyAsync velocity block");
    if (state.stochastic) {
        const auto* residual = static_cast<const char*>(block.outputs.second_residual.values.data)
            + block.outputs.second_residual.origin;
        cuda_check(cudaMemcpyAsync(slot.residual, residual, bytes, cudaMemcpyDeviceToHost, native),
                   "cudaMemcpyAsync second residual block");
    }
    cuda_check(cudaEventRecord(slot.end, native), "cudaEventRecord output end");
    slot.pending = true;
    state.d2h_bytes += bytes*(state.stochastic ? 2 : 1);
    state.next_gene += block.count;
    ++state.next_block;
}

void drain_sink(SinkState& sink) {
    for (auto& slot : sink.slots) sink.drain(slot);
    if (sink.next_gene != sink.genes || sink.velocity.values != sink.rows*sink.genes
        || !sink.velocity.finite || !std::isfinite(sink.velocity.sum)
        || !std::isfinite(sink.velocity.absolute_sum))
        throw std::runtime_error("velocity sink did not cover every gene with finite output");
    if (sink.stochastic && (sink.residual.values != sink.rows*sink.genes || !sink.residual.finite
        || !std::isfinite(sink.residual.sum) || !std::isfinite(sink.residual.absolute_sum)))
        throw std::runtime_error("stochastic residual sink did not cover every gene with finite output");
}

struct DeviceCase {
    int device = -1;
    std::string device_name, device_uuid;
    int compute_major = 0, compute_minor = 0, runtime_version = 0, driver_version = 0;
    std::uint64_t input_device_bytes = 0, prepared_bytes = 0, workspace_bytes = 0;
    std::uint64_t result_bytes = 0, output_device_bytes = 0, h2d_bytes = 0;
    std::uint64_t d2h_bytes = 0, fit_result_bytes = 0;
    std::uint64_t rows = 0, genes = 0, input_file_bytes = 0;
    std::uint64_t selected_genes = 0, successful_genes = 0;
    std::array<std::uint64_t, 7> status_counts{};
    Checksum velocity, residual;
    double gamma_sum = 0, offset_sum = 0, offset2_sum = 0;
    double first_scale_sum = 0, second_scale_sum = 0, objective_sum = 0;
    double load_s = 0, packing_s = 0, input_allocate_s = 0, h2d_s = 0;
    double h2d_host_s = 0;
    double preparation_s = 0, resident_device_s = 0, output_replay_wall_s = 0;
    double output_replay_device_s = 0, d2h_device_s = 0, end_to_end_s = 0;
    Memory before, loaded, packed, prepared, after, after_cleanup;
    AllocationStats load_alloc, packing_alloc, h2d_alloc, prep_alloc, execution_alloc, d2h_alloc, total_end_to_end_alloc;
    DeviceStats upload_device_alloc, prep_device_alloc, execution_device_alloc, total_device_alloc;
    DeviceStats resident_device_alloc;
    std::uint64_t reported_device_bytes = 0, pinned_host_bytes = 0, packed_host_bytes = 0;
    std::uint64_t output_bytes = 0;
    const char* output_dtype = "float32";
    bool selected_all = false, finite = false;
};

void copy_vector(ac::DenseView source, void* destination, std::size_t count,
                 std::size_t width, ac::Stream stream) {
    if (!source.values.data || source.rows != count || source.columns != 1
        || source.row_stride != 1)
        throw std::runtime_error("fit diagnostics must be contiguous device vectors");
    ac::copy_to_host(destination, source.values, source.origin, count*width, stream);
}

struct DiagnosticData {
    std::vector<std::byte> numeric;
    std::vector<std::uint8_t> selected, status;
    double gamma_sum = 0, offset_sum = 0, offset2_sum = 0;
    double first_scale_sum = 0, second_scale_sum = 0, objective_sum = 0;
    bool finite = true;
    std::size_t selected_count = 0, successful_count = 0;
    std::array<std::uint64_t, 7> status_counts{};
};

template<class T> double inspect_numeric(const std::byte* data, std::size_t i, bool& finite) {
    T value;
    std::memcpy(&value, data+i*sizeof(T), sizeof(T));
    finite = finite && std::isfinite(double(value));
    return double(value);
}

DiagnosticData download_diagnostics(vc::ResultView result, std::size_t genes,
                                    velocity::ComputePrecision precision, bool stochastic,
                                    ac::Stream stream, double& d2h_seconds,
                                    std::uint64_t& d2h_bytes) {
    DiagnosticData out;
    const auto width = compute_size(precision);
    const auto fields = stochastic ? 8u : 3u;
    out.numeric.resize(genes*fields*width);
    out.selected.resize(genes);
    out.status.resize(genes);
    std::array<ac::DenseView, 8> vectors{
        result.gamma, result.offset, result.r2, result.offset2,
        result.first_scale, result.second_scale, result.objective, result.initial_gamma};
    // Stochastic diagnostics reuse the first seven fields above; deterministic
    // diagnostics use the initial slope/intercept/R2 triples.
    if (!stochastic) vectors = {result.initial_gamma, result.initial_offset, result.r2,
        ac::DenseView{}, ac::DenseView{}, ac::DenseView{}, ac::DenseView{}, ac::DenseView{}};
    cudaEvent_t begin = nullptr, end = nullptr;
    event_create(begin); event_create(end);
    const auto native = reinterpret_cast<cudaStream_t>(stream.native);
    cuda_check(cudaEventRecord(begin, native), "cudaEventRecord diagnostics start");
    for (std::size_t i = 0; i < fields; ++i)
        copy_vector(vectors[i], out.numeric.data()+i*genes*width, genes, width, stream);
    copy_vector(result.selected, out.selected.data(), genes, sizeof(std::uint8_t), stream);
    copy_vector(result.status, out.status.data(), genes, sizeof(std::uint8_t), stream);
    cuda_check(cudaEventRecord(end, native), "cudaEventRecord diagnostics end");
    cuda_check(cudaEventSynchronize(end), "cudaEventSynchronize diagnostics");
    d2h_seconds += event_seconds(begin, end);
    cudaEventDestroy(begin); cudaEventDestroy(end);
    d2h_bytes += genes*fields*width + 2*genes;

    for (std::size_t gene = 0; gene < genes; ++gene) {
        const bool selected = out.selected[gene] != 0;
        out.selected_count += selected;
        out.successful_count += selected && out.status[gene] == 0;
        if (out.status[gene] < out.status_counts.size()) ++out.status_counts[out.status[gene]];
        auto get = [&](std::size_t field) {
            if (precision == velocity::ComputePrecision::float32)
                return inspect_numeric<float>(out.numeric.data()+field*genes*width, gene, out.finite);
            return inspect_numeric<double>(out.numeric.data()+field*genes*width, gene, out.finite);
        };
        if (stochastic) {
            out.gamma_sum += get(0); out.offset_sum += get(1); out.offset2_sum += get(3);
            out.first_scale_sum += get(4); out.second_scale_sum += get(5); out.objective_sum += get(6);
        } else {
            out.gamma_sum += get(0); out.offset_sum += get(1);
        }
    }
    out.finite = out.finite && std::isfinite(out.gamma_sum) && std::isfinite(out.offset_sum)
        && std::isfinite(out.offset2_sum) && std::isfinite(out.first_scale_sum)
        && std::isfinite(out.second_scale_sum) && std::isfinite(out.objective_sum);
    return out;
}

void set_device_metadata(DeviceCase& data, int device) {
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    data.device = device;
    data.device_name = properties.name;
    data.compute_major = properties.major;
    data.compute_minor = properties.minor;
    const auto& uuid = properties.uuid;
    char digits[33]{};
    for (std::size_t i = 0; i < sizeof uuid.bytes; ++i)
        std::snprintf(digits+2*i, sizeof(digits)-2*i, "%02x", static_cast<unsigned char>(uuid.bytes[i]));
    data.device_uuid = digits;
    cuda_check(cudaRuntimeGetVersion(&data.runtime_version), "cudaRuntimeGetVersion");
    cuda_check(cudaDriverGetVersion(&data.driver_version), "cudaDriverGetVersion");
}

std::string json_escape(std::string_view value) {
    std::string out;
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') { out.push_back('\\'); out.push_back(char(ch)); }
        else if (ch == '\n') out += "\\n";
        else if (ch == '\r') out += "\\r";
        else if (ch == '\t') out += "\\t";
        else if (ch < 0x20) out.push_back('?');
        else out.push_back(char(ch));
    }
    return out;
}

void run_method(const Config& config, Fixture& fixture, DeviceCase& data,
                Clock::time_point total_start) {
    const auto precision = compute(config.profile);
    const bool stochastic = config.method == Method::stochastic;
    const auto n = fixture.spliced.rows, g = fixture.spliced.columns;
    data.rows = n; data.genes = g; data.input_file_bytes = fixture.file_bytes;
    auto s_storage = storage(config.profile, false);
    auto u_storage = storage(config.profile, false);
    auto w_storage = storage(config.profile, true);
    data.input_device_bytes = 0;
    data.reported_device_bytes = 0;

    const auto packing_start = Clock::now();
    const auto packing_alloc_start = host_allocations();
    auto packed_s = pack_values(fixture.spliced, s_storage);
    auto packed_u = pack_values(fixture.unspliced, u_storage);
    auto packed_w = pack_values(fixture.weights, w_storage);
    data.packed_host_bytes = packed_s.bytes+packed_u.bytes+packed_w.bytes;
    data.packed = memory();
    const auto packing_end = Clock::now();
    data.packing_s = seconds(packing_start, packing_end);
    data.packing_alloc = difference(packing_alloc_start, host_allocations());

    StreamOwner stream_owner;
    const auto stream = stream_owner.view();
    data.device = stream.device;
    set_device_metadata(data, stream.device);
    count_device_allocations.store(true, std::memory_order_relaxed);

    const auto allocate_start = Clock::now();
    const auto upload_device_start = device_stats();
    auto device_s = allocate_matrix(fixture.spliced, s_storage, stream.device);
    auto device_u = allocate_matrix(fixture.unspliced, u_storage, stream.device);
    auto device_w = allocate_matrix(fixture.weights, w_storage, stream.device);
    data.input_device_bytes = device_s.bytes+device_u.bytes+device_w.bytes;
    const auto allocate_end = Clock::now();
    data.input_allocate_s = seconds(allocate_start, allocate_end);
    data.upload_device_alloc = difference(upload_device_start, device_stats());
    cudaEvent_t h2d_begin = nullptr, h2d_end = nullptr;
    event_create(h2d_begin); event_create(h2d_end);
    const auto upload_host_start = Clock::now();
    const auto upload_alloc_start = host_allocations();
    const auto native_stream = reinterpret_cast<cudaStream_t>(stream.native);
    cuda_check(cudaEventRecord(h2d_begin, native_stream), "H2D start event");
    upload_matrix(device_s, fixture.spliced, packed_s, stream);
    upload_matrix(device_u, fixture.unspliced, packed_u, stream);
    upload_matrix(device_w, fixture.weights, packed_w, stream);
    cuda_check(cudaEventRecord(h2d_end, native_stream), "H2D end event");
    cuda_check(cudaEventSynchronize(h2d_end), "cudaEventSynchronize H2D");
    const auto upload_host_end = Clock::now();
    data.h2d_s = event_seconds(h2d_begin, h2d_end);
    data.h2d_host_s = seconds(upload_host_start, upload_host_end);
    data.h2d_alloc = difference(upload_alloc_start, host_allocations());
    cudaEventDestroy(h2d_begin); cudaEventDestroy(h2d_end);
    data.h2d_bytes = data.input_device_bytes;

    vc::Input input;
    input.weights = device_w.matrix;
    input.spliced = device_s.matrix;
    input.unspliced = device_u.matrix;
    const auto prep_start = Clock::now();
    const auto prep_alloc_start = host_allocations();
    const auto prep_device_start = device_stats();
    vc::PreparedInput prepared(input, stream, anndata::Validation::none);
    data.prepared_bytes = prepared.storage_bytes();
    const auto output_type = output_scalar_type(config.profile);
    const auto output_width = output_size(config.profile);
    data.output_dtype = config.profile == Profile::legacy ? "float32" : compute_name(precision);
    data.output_bytes = n*g*output_width;
    const auto output_columns = std::min(config.block, g);

    if (stochastic) {
        velocity::StochasticRunOptions run;
        run.workspace_bytes = max_workspace_bytes;
        run.block_genes = config.block;
        run.compute_precision = precision;
        run.second_residual = true;
        vc::Workspace workspace(prepared, run);
        vc::FitResult result(prepared, precision, true);
        DeviceOutput velocity_output(n, output_columns, output_type, stream.device);
        DeviceOutput residual_output(n, output_columns, output_type, stream.device);
        data.workspace_bytes = workspace.scratch_bytes();
        data.result_bytes = result.storage_bytes();
        data.output_device_bytes = velocity_output.owner.size_bytes()+residual_output.owner.size_bytes();
        data.reported_device_bytes = data.input_device_bytes+data.prepared_bytes+data.workspace_bytes
            +data.result_bytes+data.output_device_bytes;
        if (data.reported_device_bytes > max_accounted_bytes)
            throw std::runtime_error("input/prepared/workspace/result/output storage exceeds 1.5 GiB");
        vc::Outputs outputs;
        outputs.velocity = velocity_output.view;
        outputs.second_residual = residual_output.view;
        outputs.block_local = true;
        velocity::StochasticOptions options;
        options.initialization.min_r2 = -100;
        options.initialization.min_ratio = -100;
        SinkState sink(n, g, workspace.block_genes(), output_width, true, output_type);
        data.pinned_host_bytes = sink.slots.size() * n * output_columns * output_width * 2;
        cudaEvent_t begin = nullptr, end = nullptr;
        event_create(begin); event_create(end);
        data.prepared = memory();
        data.preparation_s = seconds(prep_start, Clock::now());
        data.prep_alloc = difference(prep_alloc_start, host_allocations());
        data.prep_device_alloc = difference(prep_device_start, device_stats());
        const auto replay_start = Clock::now();
        const auto execution_alloc_start = host_allocations();
        const auto execution_device_start = device_stats();
        cuda_check(cudaEventRecord(begin, native_stream), "output replay start event");
        vc::reference::stochastic_into(prepared, workspace, result.view(), options, outputs, stream,
            {&sink, sink_write});
        cuda_check(cudaEventRecord(end, native_stream), "output replay end event");
        cuda_check(cudaEventSynchronize(end), "cudaEventSynchronize output replay");
        data.output_replay_device_s = event_seconds(begin, end);
        drain_sink(sink);
        data.output_replay_wall_s = seconds(replay_start, Clock::now());
        data.d2h_device_s = sink.d2h_event_seconds;
        data.execution_alloc = difference(execution_alloc_start, host_allocations());
        data.execution_device_alloc = difference(execution_device_start, device_stats());
        const auto d2h_alloc_start = host_allocations();
        auto diagnostic = download_diagnostics(result.view(), g, precision, true, stream,
                                                data.d2h_device_s, data.d2h_bytes);
        data.d2h_alloc = difference(d2h_alloc_start, host_allocations());
        data.d2h_bytes += sink.d2h_bytes;
        data.velocity = sink.velocity;
        data.residual = sink.residual;
        data.selected_genes = diagnostic.selected_count;
        data.successful_genes = diagnostic.successful_count;
        data.status_counts = diagnostic.status_counts;
        data.selected_all = diagnostic.selected_count == g;
        data.fit_result_bytes = result.storage_bytes();
        data.finite = diagnostic.finite && sink.velocity.finite && sink.residual.finite
            && std::isfinite(sink.velocity.sum) && std::isfinite(sink.residual.sum);
        if (!data.finite) throw std::runtime_error("nonfinite stochastic CUDA outputs or diagnostics");
        data.gamma_sum = diagnostic.gamma_sum;
        data.offset_sum = diagnostic.offset_sum;
        data.offset2_sum = diagnostic.offset2_sum;
        data.first_scale_sum = diagnostic.first_scale_sum;
        data.second_scale_sum = diagnostic.second_scale_sum;
        data.objective_sum = diagnostic.objective_sum;
        data.total_end_to_end_alloc = host_allocations();
        data.end_to_end_s = seconds(total_start, Clock::now());

        const auto resident_device_start = device_stats();
        cuda_check(cudaEventRecord(begin, native_stream), "resident start event");
        vc::reference::stochastic_into(prepared, workspace, result.view(), options, outputs, stream);
        cuda_check(cudaEventRecord(end, native_stream), "resident end event");
        cuda_check(cudaEventSynchronize(end), "cudaEventSynchronize resident");
        data.resident_device_s = event_seconds(begin, end);
        data.resident_device_alloc = difference(resident_device_start, device_stats());
        cudaEventDestroy(begin); cudaEventDestroy(end);
    } else {
        velocity::RunOptions run;
        run.workspace_bytes = max_workspace_bytes;
        run.block_genes = config.block;
        run.compute_precision = precision;
        vc::Workspace workspace(prepared, run);
        vc::FitResult result(prepared, precision);
        DeviceOutput velocity_output(n, output_columns, output_type, stream.device);
        data.workspace_bytes = workspace.scratch_bytes();
        data.result_bytes = result.storage_bytes();
        data.output_device_bytes = velocity_output.owner.size_bytes();
        data.reported_device_bytes = data.input_device_bytes+data.prepared_bytes+data.workspace_bytes
            +data.result_bytes+data.output_device_bytes;
        if (data.reported_device_bytes > max_accounted_bytes)
            throw std::runtime_error("input/prepared/workspace/result/output storage exceeds 1.5 GiB");
        vc::Outputs outputs;
        outputs.velocity = velocity_output.view;
        outputs.block_local = true;
        velocity::FitOptions options;
        options.min_r2 = -100;
        options.min_ratio = -100;
        SinkState sink(n, g, workspace.block_genes(), output_width, false, output_type);
        data.pinned_host_bytes = sink.slots.size() * n * output_columns * output_width;
        cudaEvent_t begin = nullptr, end = nullptr;
        event_create(begin); event_create(end);
        data.prepared = memory();
        data.preparation_s = seconds(prep_start, Clock::now());
        data.prep_alloc = difference(prep_alloc_start, host_allocations());
        data.prep_device_alloc = difference(prep_device_start, device_stats());
        const auto replay_start = Clock::now();
        const auto execution_alloc_start = host_allocations();
        const auto execution_device_start = device_stats();
        cuda_check(cudaEventRecord(begin, native_stream), "output replay start event");
        vc::reference::steady_state_into(prepared, workspace, result.view(), options, outputs, stream,
            {&sink, sink_write});
        cuda_check(cudaEventRecord(end, native_stream), "output replay end event");
        cuda_check(cudaEventSynchronize(end), "cudaEventSynchronize output replay");
        data.output_replay_device_s = event_seconds(begin, end);
        drain_sink(sink);
        data.output_replay_wall_s = seconds(replay_start, Clock::now());
        data.d2h_device_s = sink.d2h_event_seconds;
        data.execution_alloc = difference(execution_alloc_start, host_allocations());
        data.execution_device_alloc = difference(execution_device_start, device_stats());
        const auto d2h_alloc_start = host_allocations();
        auto diagnostic = download_diagnostics(result.view(), g, precision, false, stream,
                                                data.d2h_device_s, data.d2h_bytes);
        data.d2h_alloc = difference(d2h_alloc_start, host_allocations());
        data.d2h_bytes += sink.d2h_bytes;
        data.velocity = sink.velocity;
        data.selected_genes = diagnostic.selected_count;
        data.successful_genes = diagnostic.successful_count;
        data.status_counts = diagnostic.status_counts;
        data.selected_all = diagnostic.selected_count == g;
        data.fit_result_bytes = result.storage_bytes();
        data.finite = diagnostic.finite && sink.velocity.finite && std::isfinite(sink.velocity.sum);
        if (!data.finite) throw std::runtime_error("nonfinite steady-state CUDA outputs or diagnostics");
        data.gamma_sum = diagnostic.gamma_sum;
        data.offset_sum = diagnostic.offset_sum;
        data.total_end_to_end_alloc = host_allocations();
        data.end_to_end_s = seconds(total_start, Clock::now());

        const auto resident_device_start = device_stats();
        cuda_check(cudaEventRecord(begin, native_stream), "resident start event");
        vc::reference::steady_state_into(prepared, workspace, result.view(), options, outputs, stream);
        cuda_check(cudaEventRecord(end, native_stream), "resident end event");
        cuda_check(cudaEventSynchronize(end), "cudaEventSynchronize resident");
        data.resident_device_s = event_seconds(begin, end);
        data.resident_device_alloc = difference(resident_device_start, device_stats());
        cudaEventDestroy(begin); cudaEventDestroy(end);
    }

    data.after = memory();
}

// These are assigned after local DeviceBuffers and prepared/workspace/result
// owners are destroyed, so frees are included in the observed wrapper totals.
struct MeasuredRun { DeviceCase data; };

MeasuredRun measure(const Config& config, Clock::time_point total_start) {
    MeasuredRun measured;
    auto& data = measured.data;
    data.before = memory();
    const auto load_start = Clock::now();
    const auto load_alloc_start = host_allocations();
    auto fixture = load_fixture(config.input);
    const auto load_end = Clock::now();
    data.loaded = memory();
    data.load_s = seconds(load_start, load_end);
    data.load_alloc = difference(load_alloc_start, host_allocations());
    const auto g = fixture.spliced.columns, n = fixture.spliced.rows;
    data.execution_alloc = {};
    data.d2h_alloc = {};

    run_method(config, fixture, data, total_start);
    data.total_device_alloc = device_stats();
    data.after_cleanup = memory();
    return measured;
}

std::string make_json(const Config& config, const Fixture& fixture, const DeviceCase& data) {
    const auto n = fixture.spliced.rows, g = fixture.spliced.columns;
    const auto precision = compute(config.profile);
    const auto s_storage = storage(config.profile, false), w_storage = storage(config.profile, true);
    std::uint64_t max_rss = 0, max_hwm = 0, max_rusage = 0;
    for (const auto& m : {data.before, data.loaded, data.packed, data.prepared, data.after, data.after_cleanup}) {
        max_rss = std::max(max_rss, m.rss); max_hwm = std::max(max_hwm, m.high_water);
        max_rusage = std::max(max_rusage, m.rusage_peak);
    }
    const auto max_observed = std::max({max_rss, max_hwm, max_rusage});
    std::ostringstream json;
    json << std::setprecision(17)
        << "{\"schema\":\"velocity.precision-cuda-qualification.v1\","
        << "\"input\":\"" << json_escape(config.input.string()) << "\","
        << "\"input_file_bytes\":" << fixture.file_bytes << ","
        << "\"shape\":[" << n << ',' << g << "],"
        << "\"profile\":\"" << profile_name(config.profile) << "\","
        << "\"method\":\"" << (config.method == Method::steady ? "steady" : "stochastic") << "\","
        << "\"spliced_storage\":\"" << storage_name(s_storage) << "\","
        << "\"unspliced_storage\":\"" << storage_name(s_storage) << "\","
        << "\"weights_storage\":\"" << storage_name(w_storage) << "\","
        << "\"compute_precision\":\"" << compute_name(precision) << "\",\"output_storage\":\"" << data.output_dtype << "\","
        << "\"block_genes\":" << config.block << ",\"full_refine\":" << (data.selected_all && (config.method != Method::stochastic || data.successful_genes == fixture.spliced.columns) ? "true" : "false") << ","
        << "\"selected_genes\":" << data.selected_genes << ","
        << "\"successful_genes\":" << data.successful_genes << ","
        << "\"status_counts\":[";
    for (std::size_t i = 0; i < data.status_counts.size(); ++i) {
        if (i) json << ',';
        json << data.status_counts[i];
    }
    json << "],\"device\":{\"ordinal\":" << data.device << ",\"name\":\""
        << json_escape(data.device_name) << "\",\"uuid\":\"" << data.device_uuid
        << "\",\"compute_capability\":[" << data.compute_major << ',' << data.compute_minor
        << "],\"runtime_version\":" << data.runtime_version << ",\"driver_version\":" << data.driver_version << "},"
        << "\"device_storage_bytes\":{\"input\":" << data.input_device_bytes
        << ",\"prepared_host_and_device\":" << data.prepared_bytes << ",\"workspace\":" << data.workspace_bytes
        << ",\"fit_result\":" << data.result_bytes << ",\"block_outputs\":" << data.output_device_bytes
        << ",\"component_total_including_prepared_host_bytes\":" << data.reported_device_bytes
        << ",\"cudaMalloc_peak_live\":" << data.total_device_alloc.peak_bytes
        << ",\"cudaMalloc_live_at_exit\":" << data.total_device_alloc.live_bytes << "},"
        << "\"host_memory_bytes\":{\"rss_before\":" << data.before.rss
        << ",\"rss_after_load\":" << data.loaded.rss << ",\"rss_after_pack\":" << data.packed.rss
        << ",\"rss_after_preparation\":" << data.prepared.rss
        << ",\"max_observed_vmrss\":" << max_rss
        << ",\"max_observed_vmhwm\":" << max_hwm
        << ",\"max_observed_ru_maxrss\":" << max_rusage
        << ",\"max_observed_of_all_three\":" << max_observed
        << ",\"packed_values\":" << data.packed_host_bytes << ",\"pinned_output_staging\":" << data.pinned_host_bytes << "},"
        << "\"allocations\":{\"end_to_end_global_new_calls\":" << data.total_end_to_end_alloc.calls
        << ",\"end_to_end_global_new_bytes\":" << data.total_end_to_end_alloc.bytes
        << ",\"load_global_new_calls\":" << data.load_alloc.calls
        << ",\"load_global_new_bytes\":" << data.load_alloc.bytes
        << ",\"packing_global_new_calls\":" << data.packing_alloc.calls
        << ",\"packing_global_new_bytes\":" << data.packing_alloc.bytes
        << ",\"preparation_global_new_calls\":" << data.prep_alloc.calls
        << ",\"preparation_global_new_bytes\":" << data.prep_alloc.bytes
        << ",\"execution_global_new_calls\":" << data.execution_alloc.calls
        << ",\"execution_global_new_bytes\":" << data.execution_alloc.bytes
        << ",\"d2h_global_new_calls\":" << data.d2h_alloc.calls
        << ",\"d2h_global_new_bytes\":" << data.d2h_alloc.bytes
        << ",\"cudaMalloc_calls\":" << data.total_device_alloc.malloc_calls
        << ",\"cudaMalloc_requested_bytes\":" << data.total_device_alloc.malloc_bytes
        << ",\"cudaFree_calls\":" << data.total_device_alloc.free_calls
        << ",\"cudaFree_tracked_bytes\":" << data.total_device_alloc.free_bytes
        << ",\"cudaMalloc_wrap_overflows\":" << data.total_device_alloc.overflowed_records << "},"
        << "\"transfer_bytes\":{\"h2d\":" << data.h2d_bytes << ",\"d2h\":" << data.d2h_bytes << "},"
        << "\"phases_s\":{\"load_or_generate\":" << data.load_s
        << ",\"packing\":" << data.packing_s << ",\"input_device_allocate\":" << data.input_allocate_s
        << ",\"h2d_device_events\":" << data.h2d_s << ",\"h2d_host_wall\":" << data.h2d_host_s << ",\"preparation\":" << data.preparation_s
        << ",\"resident_compute_device_events\":" << data.resident_device_s
        << ",\"d2h_device_events\":" << data.d2h_device_s
        << ",\"output_replay_wall\":" << data.output_replay_wall_s
        << ",\"output_replay_device_events\":" << data.output_replay_device_s
        << ",\"end_to_end\":" << data.end_to_end_s << "},"
        << "\"checksums\":{\"velocity_sum\":" << data.velocity.sum
        << ",\"velocity_absolute_sum\":" << data.velocity.absolute_sum
        << ",\"second_residual_sum\":" << data.residual.sum
        << ",\"second_residual_absolute_sum\":" << data.residual.absolute_sum
        << ",\"gamma_sum\":" << data.gamma_sum << ",\"offset_sum\":" << data.offset_sum
        << ",\"offset2_sum\":" << data.offset2_sum << ",\"first_scale_sum\":" << data.first_scale_sum
        << ",\"second_scale_sum\":" << data.second_scale_sum
        << ",\"objective_sum\":" << data.objective_sum << "},"
        << "\"finite\":" << (data.finite ? "true" : "false")
        << ",\"memory_protocol\":\"Linux proc VmRSS and VmHWM and getrusage ru_maxrss sampled before load, after load, packing, preparation, execution, and cleanup; max of observed values includes the resident diagnostic; proc and rusage are independent kernel estimates\""
        << ",\"allocation_interception\":\"cudaMalloc/cudaFree ELF symbol interposition; async, managed, pitched, driver-internal and CUDA runtime-internal allocations are outside this counter\"}\n";
    return json.str();
}

int run(const Config& config) {
    const auto total_start = Clock::now();
    count_host_allocations.store(true, std::memory_order_relaxed);
    auto measured = measure(config, total_start);
    count_host_allocations.store(false, std::memory_order_relaxed);
    const auto json = make_json(config, [&]() -> const Fixture& {
        // The result needs only dimensions and input-file length after the
        // processing owners have been released. Re-read the small header here.
        static Fixture metadata;
        std::ifstream file(config.input, std::ios::binary);
        std::uint64_t header[5]{};
        read_exact(file, header, sizeof header);
        metadata.spliced.rows = checked_size(header[0], "N");
        metadata.spliced.columns = checked_size(header[1], "G");
        std::error_code error;
        metadata.file_bytes = std::filesystem::file_size(config.input, error);
        if (error) throw std::runtime_error("cannot restat fixture: " + error.message());
        return metadata;
    }(), measured.data);
    if (config.output_json.empty()) std::cout << json;
    else {
        std::ofstream output(config.output_json, std::ios::binary);
        if (!output || !(output << json)) throw std::runtime_error("cannot write JSON output");
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try { return run(parse_args(argc, argv)); }
    catch (const std::exception& error) {
        count_host_allocations.store(false, std::memory_order_relaxed);
        std::cerr << "velocity-precision-cuda-qualification: " << error.what() << '\n';
        return 2;
    }
}
