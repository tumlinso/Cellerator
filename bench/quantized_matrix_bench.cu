#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "benchmark_mutex.hh"
#include "../src/quantized/api.cuh"

namespace msq = ::cellerator::quantized;

namespace {

struct bench_fixture {
    int rows = 32768;
    int cols = 4096;
    int nnz_per_row = 32;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> gene_scales;
    std::vector<float> gene_offsets;
    std::vector<float> row_offsets;
    std::vector<float> values;
    std::vector<float> recovered;
};

struct bench_options {
    int host_iters = 20;
    int gpu_iters = 400;
    int bits_filter = 0;
    bool run_per_gene_affine = true;
    bool run_column_scale_row_offset = true;
};

struct device_workspace {
    int* d_row_ptr = nullptr;
    int* d_packed_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    float* d_scales = nullptr;
    float* d_offsets = nullptr;
    float* d_row_offsets = nullptr;
    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_packed_values = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

template<typename Metadata>
struct metadata_case;

template<>
struct metadata_case<msq::per_gene_affine<float>> {
    static constexpr const char* label() { return "per_gene_affine"; }

    static msq::per_gene_affine<float> make_host(const bench_fixture& fixture) {
        return msq::make_per_gene_affine(fixture.gene_scales.data(), fixture.gene_offsets.data());
    }

    static msq::per_gene_affine<float> make_device(const device_workspace& workspace) {
        return msq::make_per_gene_affine(workspace.d_scales, workspace.d_offsets);
    }

    static float make_value(const bench_fixture& fixture, int row, int gene, unsigned int code) {
        (void) row;
        return fixture.gene_offsets[static_cast<std::size_t>(gene)] +
               fixture.gene_scales[static_cast<std::size_t>(gene)] * static_cast<float>(code);
    }
};

template<>
struct metadata_case<msq::column_scale_row_offset<float>> {
    static constexpr const char* label() { return "column_scale_row_offset"; }

    static msq::column_scale_row_offset<float> make_host(const bench_fixture& fixture) {
        return msq::make_column_scale_row_offset(fixture.gene_scales.data(), fixture.row_offsets.data());
    }

    static msq::column_scale_row_offset<float> make_device(const device_workspace& workspace) {
        return msq::make_column_scale_row_offset(workspace.d_scales, workspace.d_row_offsets);
    }

    static float make_value(const bench_fixture& fixture, int row, int gene, unsigned int code) {
        return fixture.row_offsets[static_cast<std::size_t>(row)] +
               fixture.gene_scales[static_cast<std::size_t>(gene)] * static_cast<float>(code);
    }
};

static bool check_cuda(cudaError_t status, const char* what) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    return false;
}

static void destroy_workspace(device_workspace* workspace) {
    if (workspace == nullptr) {
        return;
    }
    if (workspace->start != nullptr) {
        cudaEventDestroy(workspace->start);
        workspace->start = nullptr;
    }
    if (workspace->stop != nullptr) {
        cudaEventDestroy(workspace->stop);
        workspace->stop = nullptr;
    }
    cudaFree(workspace->d_row_ptr);
    cudaFree(workspace->d_packed_row_ptr);
    cudaFree(workspace->d_col_idx);
    cudaFree(workspace->d_scales);
    cudaFree(workspace->d_offsets);
    cudaFree(workspace->d_row_offsets);
    cudaFree(workspace->d_input);
    cudaFree(workspace->d_output);
    cudaFree(workspace->d_packed_values);
    *workspace = {};
}

static bool init_workspace(const bench_fixture& fixture, device_workspace* workspace) {
    const std::size_t row_ptr_bytes = fixture.row_ptr.size() * sizeof(int);
    const std::size_t col_idx_bytes = fixture.col_idx.size() * sizeof(int);
    const std::size_t scales_bytes = fixture.gene_scales.size() * sizeof(float);
    const std::size_t offsets_bytes = fixture.gene_offsets.size() * sizeof(float);
    const std::size_t row_offsets_bytes = fixture.row_offsets.size() * sizeof(float);
    const std::size_t values_bytes = fixture.values.size() * sizeof(float);
    const std::size_t packed_bytes = fixture.values.size() * sizeof(unsigned char);
    bool ok = true;

    if (workspace == nullptr) {
        return false;
    }

    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&workspace->d_row_ptr), row_ptr_bytes), "cudaMalloc d_row_ptr") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&workspace->d_packed_row_ptr), row_ptr_bytes), "cudaMalloc d_packed_row_ptr") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&workspace->d_col_idx), col_idx_bytes), "cudaMalloc d_col_idx") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&workspace->d_scales), scales_bytes), "cudaMalloc d_scales") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&workspace->d_offsets), offsets_bytes), "cudaMalloc d_offsets") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&workspace->d_row_offsets), row_offsets_bytes), "cudaMalloc d_row_offsets") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&workspace->d_input), values_bytes), "cudaMalloc d_input") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&workspace->d_output), values_bytes), "cudaMalloc d_output") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&workspace->d_packed_values), packed_bytes), "cudaMalloc d_packed_values") && ok;
    ok = check_cuda(cudaEventCreate(&workspace->start), "cudaEventCreate start") && ok;
    ok = check_cuda(cudaEventCreate(&workspace->stop), "cudaEventCreate stop") && ok;
    if (!ok) {
        destroy_workspace(workspace);
        return false;
    }

    ok = check_cuda(cudaMemcpy(workspace->d_row_ptr, fixture.row_ptr.data(), row_ptr_bytes, cudaMemcpyHostToDevice),
                    "copy row_ptr") && ok;
    ok = check_cuda(cudaMemcpy(workspace->d_col_idx, fixture.col_idx.data(), col_idx_bytes, cudaMemcpyHostToDevice),
                    "copy col_idx") && ok;
    ok = check_cuda(cudaMemcpy(workspace->d_scales, fixture.gene_scales.data(), scales_bytes, cudaMemcpyHostToDevice),
                    "copy scales") && ok;
    ok = check_cuda(cudaMemcpy(workspace->d_offsets, fixture.gene_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice),
                    "copy offsets") && ok;
    ok = check_cuda(cudaMemcpy(workspace->d_row_offsets,
                               fixture.row_offsets.data(),
                               row_offsets_bytes,
                               cudaMemcpyHostToDevice),
                    "copy row_offsets") && ok;

    if (!ok) {
        destroy_workspace(workspace);
    }
    return ok;
}

static bench_fixture make_fixture() {
    bench_fixture fixture;

    fixture.row_ptr.resize(static_cast<std::size_t>(fixture.rows) + 1u, 0);
    fixture.col_idx.resize(static_cast<std::size_t>(fixture.rows * fixture.nnz_per_row), 0);
    fixture.gene_scales.resize(static_cast<std::size_t>(fixture.cols), 1.0f);
    fixture.gene_offsets.resize(static_cast<std::size_t>(fixture.cols), 0.0f);
    fixture.row_offsets.resize(static_cast<std::size_t>(fixture.rows), 0.0f);

    for (int col = 0; col < fixture.cols; ++col) {
        fixture.gene_scales[static_cast<std::size_t>(col)] = 0.125f + 0.015625f * static_cast<float>(col % 17);
        fixture.gene_offsets[static_cast<std::size_t>(col)] = 0.25f * static_cast<float>((col % 5) - 2);
    }
    for (int row = 0; row < fixture.rows; ++row) {
        fixture.row_offsets[static_cast<std::size_t>(row)] = 0.125f * static_cast<float>((row % 9) - 4);
        fixture.row_ptr[static_cast<std::size_t>(row)] = row * fixture.nnz_per_row;
        for (int lane = 0; lane < fixture.nnz_per_row; ++lane) {
            fixture.col_idx[static_cast<std::size_t>(row * fixture.nnz_per_row + lane)] =
                (row * 17 + lane * 19) % fixture.cols;
        }
    }

    fixture.row_ptr[static_cast<std::size_t>(fixture.rows)] = fixture.rows * fixture.nnz_per_row;
    fixture.values.resize(fixture.col_idx.size(), 0.0f);
    fixture.recovered.resize(fixture.col_idx.size(), 0.0f);
    return fixture;
}

static void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--host-iters N] [--gpu-iters N] [--bits {1|2|4|8|all}] "
                 "[--policy {all|per_gene_affine|column_scale_row_offset}]\n",
                 argv0);
}

static bool parse_options(int argc, char** argv, bench_options* options) {
    if (options == nullptr) {
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (std::strcmp(arg, "--host-iters") == 0) {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return false;
            }
            options->host_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--gpu-iters") == 0) {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return false;
            }
            options->gpu_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--bits") == 0) {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return false;
            }
            const char* value = argv[++i];
            if (std::strcmp(value, "all") == 0) {
                options->bits_filter = 0;
            } else {
                options->bits_filter = std::atoi(value);
                if (!msq::valid_bits(options->bits_filter)) {
                    print_usage(argv[0]);
                    return false;
                }
            }
        } else if (std::strcmp(arg, "--policy") == 0) {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return false;
            }
            const char* value = argv[++i];
            options->run_per_gene_affine = false;
            options->run_column_scale_row_offset = false;
            if (std::strcmp(value, "all") == 0) {
                options->run_per_gene_affine = true;
                options->run_column_scale_row_offset = true;
            } else if (std::strcmp(value, "per_gene_affine") == 0) {
                options->run_per_gene_affine = true;
            } else if (std::strcmp(value, "column_scale_row_offset") == 0) {
                options->run_column_scale_row_offset = true;
            } else {
                print_usage(argv[0]);
                return false;
            }
        } else {
            print_usage(argv[0]);
            return false;
        }
    }

    if (options->host_iters <= 0 || options->gpu_iters <= 0 ||
        (!options->run_per_gene_affine && !options->run_column_scale_row_offset)) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

template<int Bits, typename Metadata>
static bool run_case(const bench_fixture& base, const bench_options& options, device_workspace* workspace) {
    bench_fixture fixture = base;
    const int host_iters = options.host_iters;
    const int gpu_iters = options.gpu_iters;
    const unsigned int code_mask = static_cast<unsigned int>(msq::format_traits<Bits>::code_mask);
    const int nnz = fixture.row_ptr.back();
    const double processed_nnz = static_cast<double>(nnz) * 2.0;
    const Metadata host_metadata = metadata_case<Metadata>::make_host(fixture);
    std::vector<int> packed_row_ptr(static_cast<std::size_t>(fixture.rows) + 1u, 0);
    std::vector<unsigned char> packed_values;

    msq::build_packed_row_ptr<Bits>(fixture.row_ptr.data(), fixture.rows, packed_row_ptr.data());
    packed_values.assign(static_cast<std::size_t>(packed_row_ptr.back()), 0u);

    for (int row = 0; row < fixture.rows; ++row) {
        for (int lane = 0; lane < fixture.nnz_per_row; ++lane) {
            const int index = row * fixture.nnz_per_row + lane;
            const int gene = fixture.col_idx[static_cast<std::size_t>(index)];
            const unsigned int code = static_cast<unsigned int>((index + gene * 3 + row) % (code_mask + 1u));

            fixture.values[static_cast<std::size_t>(index)] =
                metadata_case<Metadata>::make_value(fixture, row, gene, code);
        }
    }

    auto host_matrix = msq::make_matrix<Bits>(
        fixture.rows,
        fixture.cols,
        nnz,
        0,
        fixture.row_ptr.data(),
        packed_row_ptr.data(),
        fixture.col_idx.data(),
        nullptr,
        packed_values.data(),
        host_metadata);

    {
        const auto host_begin = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < host_iters; ++iter) {
            msq::pack_nnz_values(&host_matrix, fixture.values.data());
            msq::unpack_nnz_values(&host_matrix, fixture.recovered.data());
        }

        const auto host_end = std::chrono::high_resolution_clock::now();
        const double avg_ms =
            std::chrono::duration<double, std::milli>(host_end - host_begin).count() / static_cast<double>(host_iters);
        const double mnnz_per_s = processed_nnz / avg_ms / 1.0e3;

        std::printf("policy=%s bits=%d host pack+unpack avg_ms=%.3f mnnz_per_s=%.2f packed_bytes=%zu\n",
                    metadata_case<Metadata>::label(),
                    Bits,
                    avg_ms,
                    mnnz_per_s,
                    packed_values.size());
    }

    if (workspace == nullptr) {
        std::printf("policy=%s bits=%d gpu=skipped no CUDA device\n", metadata_case<Metadata>::label(), Bits);
        return true;
    }

    {
        const std::size_t packed_row_ptr_bytes = packed_row_ptr.size() * sizeof(int);
        const std::size_t values_bytes = fixture.values.size() * sizeof(float);
        const Metadata device_metadata = metadata_case<Metadata>::make_device(*workspace);
        auto device_matrix = msq::make_matrix<Bits>(
            fixture.rows,
            fixture.cols,
            nnz,
            0,
            workspace->d_row_ptr,
            workspace->d_packed_row_ptr,
            workspace->d_col_idx,
            nullptr,
            workspace->d_packed_values,
            device_metadata);
        const auto full_block = msq::get_block(&host_matrix, 0);
        float elapsed_ms = 0.0f;
        bool ok = true;

        ok = check_cuda(cudaMemcpy(workspace->d_packed_row_ptr,
                                   packed_row_ptr.data(),
                                   packed_row_ptr_bytes,
                                   cudaMemcpyHostToDevice),
                        "copy packed_row_ptr") && ok;
        ok = check_cuda(cudaMemcpy(workspace->d_input,
                                   fixture.values.data(),
                                   values_bytes,
                                   cudaMemcpyHostToDevice),
                        "copy input") && ok;
        ok = check_cuda(cudaMemset(workspace->d_output, 0, values_bytes), "zero output") && ok;
        ok = check_cuda(cudaMemset(workspace->d_packed_values, 0, packed_values.size()), "zero packed") && ok;
        if (!ok) {
            return false;
        }

        ok = check_cuda(msq::launch_quantize_block_v100(&device_matrix, full_block, workspace->d_input), "warmup quantize") && ok;
        ok = check_cuda(msq::launch_dequantize_block_v100(&device_matrix, full_block, workspace->d_output), "warmup dequantize") && ok;
        ok = check_cuda(cudaDeviceSynchronize(), "warmup sync") && ok;
        if (!ok) {
            return false;
        }

        ok = check_cuda(cudaEventRecord(workspace->start), "event start") && ok;
        for (int iter = 0; iter < gpu_iters; ++iter) {
            ok = check_cuda(msq::launch_quantize_block_v100(&device_matrix, full_block, workspace->d_input), "bench quantize") && ok;
            ok = check_cuda(msq::launch_dequantize_block_v100(&device_matrix, full_block, workspace->d_output), "bench dequantize") && ok;
        }
        ok = check_cuda(cudaEventRecord(workspace->stop), "event stop") && ok;
        ok = check_cuda(cudaEventSynchronize(workspace->stop), "event sync") && ok;
        ok = check_cuda(cudaEventElapsedTime(&elapsed_ms, workspace->start, workspace->stop), "event elapsed") && ok;
        ok = check_cuda(cudaMemcpy(fixture.recovered.data(),
                                   workspace->d_output,
                                   values_bytes,
                                   cudaMemcpyDeviceToHost),
                        "copy recovered") && ok;
        if (!ok) {
            return false;
        }

        for (int i = 0; i < nnz; ++i) {
            if (std::fabs(fixture.values[static_cast<std::size_t>(i)] - fixture.recovered[static_cast<std::size_t>(i)]) >
                1.0e-6f) {
                std::fprintf(stderr,
                             "bench verification mismatch policy=%s bits=%d index=%d\n",
                             metadata_case<Metadata>::label(),
                             Bits,
                             i);
                return false;
            }
        }

        std::printf("policy=%s bits=%d gpu quantize+dequantize avg_ms=%.3f mnnz_per_s=%.2f packed_bytes=%zu\n",
                    metadata_case<Metadata>::label(),
                    Bits,
                    elapsed_ms / static_cast<float>(gpu_iters),
                    processed_nnz / static_cast<double>(elapsed_ms) / 1.0e3 * static_cast<double>(gpu_iters),
                    packed_values.size());
    }

    return true;
}

template<typename Metadata>
static bool run_policy_suite(const bench_fixture& fixture, const bench_options& options, device_workspace* workspace) {
    bool ok = true;

    for (int bits : {1, 2, 4, 8}) {
        if (options.bits_filter != 0 && options.bits_filter != bits) {
            continue;
        }
        ok = msq::dispatch_bits(bits, [&](auto bits_tag) {
                 constexpr int Bits = decltype(bits_tag)::value;
                 return run_case<Bits, Metadata>(fixture, options, workspace);
             }) &&
             ok;
    }
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    cellerator::bench::benchmark_mutex_guard benchmark_mutex("quantizedMatrixBench");
    bench_options options;
    bench_fixture fixture = make_fixture();
    device_workspace workspace;
    device_workspace* workspace_ptr = nullptr;
    int device_count = 0;
    bool ok = true;

    if (!parse_options(argc, argv, &options)) {
        return 1;
    }

    std::printf("cuda_mode=%s\n", cellerator::build::cuda_mode_name);

    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
        return 1;
    }
    if (device_count <= 0) {
        std::fprintf(stderr, "quantizedMatrixBench requires at least one visible CUDA device\n");
        return 1;
    }
    if (!init_workspace(fixture, &workspace)) {
        destroy_workspace(&workspace);
        return 1;
    }
    workspace_ptr = &workspace;

    if (options.run_per_gene_affine) {
        ok = run_policy_suite<msq::per_gene_affine<float>>(fixture, options, workspace_ptr) && ok;
    }
    if (options.run_column_scale_row_offset) {
        ok = run_policy_suite<msq::column_scale_row_offset<float>>(fixture, options, workspace_ptr) && ok;
    }

    destroy_workspace(&workspace);
    return ok ? 0 : 1;
}
