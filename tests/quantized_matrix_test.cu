#include <algorithm>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "../src/quantized/api.cuh"

namespace msq = ::cellerator::quantized;

namespace {

struct test_fixture {
    int rows = 4;
    int cols = 8;
    std::vector<int> row_ptr{0, 4, 7, 10, 12};
    std::vector<int> col_idx{0, 2, 5, 7, 1, 4, 6, 0, 3, 7, 2, 5};
    std::vector<float> gene_scales{0.25f, 0.50f, 0.75f, 1.00f, 1.25f, 1.50f, 1.75f, 2.00f};
    std::vector<float> gene_offsets{0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 0.25f, -0.25f, 2.0f};
    std::vector<float> row_offsets{0.0f, 1.5f, -0.75f, 0.25f};
    std::vector<int> packed_row_ptr;
    std::vector<int> block_row_ptr;
    std::vector<float> input_values;
    std::vector<float> unpacked_values;
    std::vector<unsigned char> packed_values;
};

static bool nearly_equal(float a, float b, float eps = 1.0e-6f) {
    return std::fabs(a - b) <= eps;
}

static bool check_cuda(cudaError_t status, const char* what) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    return false;
}

template<typename Metadata>
struct metadata_case;

template<>
struct metadata_case<msq::per_gene_affine<float>> {
    static constexpr const char* label() { return "per_gene_affine"; }

    static msq::per_gene_affine<float> make(const test_fixture& fixture) {
        return msq::make_per_gene_affine(fixture.gene_scales.data(), fixture.gene_offsets.data());
    }

    static float expected_value(const test_fixture& fixture, int row, int index, unsigned int code) {
        (void) row;
        const int gene = fixture.col_idx[static_cast<std::size_t>(index)];

        return fixture.gene_offsets[static_cast<std::size_t>(gene)] +
               fixture.gene_scales[static_cast<std::size_t>(gene)] * static_cast<float>(code);
    }
};

template<>
struct metadata_case<msq::column_scale_row_offset<float>> {
    static constexpr const char* label() { return "column_scale_row_offset"; }

    static msq::column_scale_row_offset<float> make(const test_fixture& fixture) {
        return msq::make_column_scale_row_offset(fixture.gene_scales.data(), fixture.row_offsets.data());
    }

    static float expected_value(const test_fixture& fixture, int row, int index, unsigned int code) {
        const int gene = fixture.col_idx[static_cast<std::size_t>(index)];

        return fixture.row_offsets[static_cast<std::size_t>(row)] +
               fixture.gene_scales[static_cast<std::size_t>(gene)] * static_cast<float>(code);
    }
};

template<int Bits, typename Metadata>
static void fill_fixture_inputs(test_fixture* fixture) {
    const unsigned int max_code = static_cast<unsigned int>(msq::format_traits<Bits>::code_mask);

    fixture->input_values.resize(fixture->col_idx.size());
    fixture->unpacked_values.assign(fixture->col_idx.size(), 0.0f);
    for (int row = 0; row < fixture->rows; ++row) {
        for (int index = fixture->row_ptr[static_cast<std::size_t>(row)];
             index < fixture->row_ptr[static_cast<std::size_t>(row + 1)];
             ++index) {
            const int gene = fixture->col_idx[static_cast<std::size_t>(index)];
            const unsigned int code =
                max_code == 0u ? 0u : static_cast<unsigned int>((index + row + gene) % (max_code + 1u));

            fixture->input_values[static_cast<std::size_t>(index)] =
                metadata_case<Metadata>::expected_value(*fixture, row, index, code);
        }
    }
}

template<int Bits, typename Metadata>
bool run_host_case() {
    test_fixture fixture;
    const int block_count = msq::block_count_for_rows(fixture.rows, 2);
    const Metadata metadata = metadata_case<Metadata>::make(fixture);

    fixture.packed_row_ptr.resize(static_cast<std::size_t>(fixture.rows) + 1u, 0);
    msq::build_packed_row_ptr<Bits>(fixture.row_ptr.data(), fixture.rows, fixture.packed_row_ptr.data());
    fixture.block_row_ptr.resize(static_cast<std::size_t>(block_count) + 1u, 0);
    msq::build_uniform_block_row_ptr(fixture.rows, 2, fixture.block_row_ptr.data());
    fixture.packed_values.assign(static_cast<std::size_t>(fixture.packed_row_ptr.back()), 0u);
    fill_fixture_inputs<Bits, Metadata>(&fixture);

    auto matrix = msq::make_matrix<Bits>(
        fixture.rows,
        fixture.cols,
        static_cast<int>(fixture.col_idx.size()),
        block_count,
        fixture.row_ptr.data(),
        fixture.packed_row_ptr.data(),
        fixture.col_idx.data(),
        fixture.block_row_ptr.data(),
        fixture.packed_values.data(),
        metadata);

    if (msq::pack_nnz_values(&matrix, fixture.input_values.data()) != 0) {
        return false;
    }
    if (msq::unpack_nnz_values(&matrix, fixture.unpacked_values.data()) != 0) {
        return false;
    }

    for (std::size_t i = 0; i < fixture.input_values.size(); ++i) {
        if (!nearly_equal(fixture.input_values[i], fixture.unpacked_values[i])) {
            std::fprintf(stderr,
                         "host mismatch policy=%s bits=%d index=%zu expected=%f got=%f\n",
                         metadata_case<Metadata>::label(),
                         Bits,
                         i,
                         fixture.input_values[i],
                         fixture.unpacked_values[i]);
            return false;
        }
    }

    for (int row = 0; row < fixture.rows; ++row) {
        for (int index = fixture.row_ptr[static_cast<std::size_t>(row)];
             index < fixture.row_ptr[static_cast<std::size_t>(row + 1)];
             ++index) {
            const float got = msq::get_value(&matrix, row, fixture.col_idx[static_cast<std::size_t>(index)]);

            if (!nearly_equal(got, fixture.input_values[static_cast<std::size_t>(index)])) {
                std::fprintf(stderr,
                             "value lookup mismatch policy=%s bits=%d row=%d index=%d expected=%f got=%f\n",
                             metadata_case<Metadata>::label(),
                             Bits,
                             row,
                             index,
                             fixture.input_values[static_cast<std::size_t>(index)],
                             got);
                return false;
            }
        }
    }

    return true;
}

template<int Bits, typename Metadata>
bool run_device_case() {
    test_fixture fixture;
    const int block_count = msq::block_count_for_rows(fixture.rows, 2);
    const Metadata host_metadata = metadata_case<Metadata>::make(fixture);
    int device_count = 0;
    int* d_row_ptr = nullptr;
    int* d_packed_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    int* d_block_row_ptr = nullptr;
    float* d_scales = nullptr;
    float* d_offsets = nullptr;
    float* d_row_offsets = nullptr;
    float* d_input = nullptr;
    float* d_output = nullptr;
    unsigned char* d_packed_values = nullptr;
    bool ok = true;
    std::vector<unsigned char> expected_packed;

    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
        return false;
    }
    if (device_count == 0) {
        return true;
    }

    fixture.packed_row_ptr.resize(static_cast<std::size_t>(fixture.rows) + 1u, 0);
    msq::build_packed_row_ptr<Bits>(fixture.row_ptr.data(), fixture.rows, fixture.packed_row_ptr.data());
    fixture.block_row_ptr.resize(static_cast<std::size_t>(block_count) + 1u, 0);
    msq::build_uniform_block_row_ptr(fixture.rows, 2, fixture.block_row_ptr.data());
    fixture.packed_values.assign(static_cast<std::size_t>(fixture.packed_row_ptr.back()), 0u);
    fill_fixture_inputs<Bits, Metadata>(&fixture);

    {
        auto host_matrix = msq::make_matrix<Bits>(
            fixture.rows,
            fixture.cols,
            static_cast<int>(fixture.col_idx.size()),
            block_count,
            fixture.row_ptr.data(),
            fixture.packed_row_ptr.data(),
            fixture.col_idx.data(),
            fixture.block_row_ptr.data(),
            fixture.packed_values.data(),
            host_metadata);

        if (msq::pack_nnz_values(&host_matrix, fixture.input_values.data()) != 0) {
            return false;
        }
        expected_packed = fixture.packed_values;
        std::fill(fixture.packed_values.begin(), fixture.packed_values.end(), 0u);
        fixture.unpacked_values.assign(fixture.input_values.size(), 0.0f);
    }

    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_row_ptr), fixture.row_ptr.size() * sizeof(int)), "cudaMalloc d_row_ptr") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_packed_row_ptr), fixture.packed_row_ptr.size() * sizeof(int)), "cudaMalloc d_packed_row_ptr") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_col_idx), fixture.col_idx.size() * sizeof(int)), "cudaMalloc d_col_idx") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_block_row_ptr), fixture.block_row_ptr.size() * sizeof(int)), "cudaMalloc d_block_row_ptr") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_scales), fixture.gene_scales.size() * sizeof(float)), "cudaMalloc d_scales") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_offsets), fixture.gene_offsets.size() * sizeof(float)), "cudaMalloc d_offsets") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_row_offsets), fixture.row_offsets.size() * sizeof(float)), "cudaMalloc d_row_offsets") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_input), fixture.input_values.size() * sizeof(float)), "cudaMalloc d_input") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_output), fixture.unpacked_values.size() * sizeof(float)), "cudaMalloc d_output") && ok;
    ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_packed_values), fixture.packed_values.size() * sizeof(unsigned char)), "cudaMalloc d_packed_values") && ok;
    if (!ok) {
        goto cleanup;
    }

    ok = check_cuda(cudaMemcpy(d_row_ptr, fixture.row_ptr.data(), fixture.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice),
                    "copy row_ptr") && ok;
    ok = check_cuda(cudaMemcpy(d_packed_row_ptr,
                               fixture.packed_row_ptr.data(),
                               fixture.packed_row_ptr.size() * sizeof(int),
                               cudaMemcpyHostToDevice),
                    "copy packed_row_ptr") && ok;
    ok = check_cuda(cudaMemcpy(d_col_idx, fixture.col_idx.data(), fixture.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice),
                    "copy col_idx") && ok;
    ok = check_cuda(cudaMemcpy(d_block_row_ptr,
                               fixture.block_row_ptr.data(),
                               fixture.block_row_ptr.size() * sizeof(int),
                               cudaMemcpyHostToDevice),
                    "copy block_row_ptr") && ok;
    ok = check_cuda(cudaMemcpy(d_scales,
                               fixture.gene_scales.data(),
                               fixture.gene_scales.size() * sizeof(float),
                               cudaMemcpyHostToDevice),
                    "copy scales") && ok;
    ok = check_cuda(cudaMemcpy(d_offsets,
                               fixture.gene_offsets.data(),
                               fixture.gene_offsets.size() * sizeof(float),
                               cudaMemcpyHostToDevice),
                    "copy offsets") && ok;
    ok = check_cuda(cudaMemcpy(d_row_offsets,
                               fixture.row_offsets.data(),
                               fixture.row_offsets.size() * sizeof(float),
                               cudaMemcpyHostToDevice),
                    "copy row_offsets") && ok;
    ok = check_cuda(cudaMemcpy(d_input,
                               fixture.input_values.data(),
                               fixture.input_values.size() * sizeof(float),
                               cudaMemcpyHostToDevice),
                    "copy input") && ok;
    ok = check_cuda(cudaMemset(d_output, 0, fixture.unpacked_values.size() * sizeof(float)), "zero output") && ok;
    ok = check_cuda(cudaMemset(d_packed_values, 0, fixture.packed_values.size() * sizeof(unsigned char)), "zero packed") && ok;
    if (!ok) {
        goto cleanup;
    }

    {
        const Metadata device_metadata = [](
            const test_fixture& host_fixture,
            float* scales,
            float* offsets,
            float* row_offsets) {
            if constexpr (std::is_same_v<Metadata, msq::per_gene_affine<float>>) {
                (void) row_offsets;
                return msq::make_per_gene_affine(scales, offsets);
            } else {
                (void) offsets;
                (void) host_fixture;
                return msq::make_column_scale_row_offset(scales, row_offsets);
            }
        }(fixture, d_scales, d_offsets, d_row_offsets);

        auto host_matrix = msq::make_matrix<Bits>(
            fixture.rows,
            fixture.cols,
            static_cast<int>(fixture.col_idx.size()),
            block_count,
            fixture.row_ptr.data(),
            fixture.packed_row_ptr.data(),
            fixture.col_idx.data(),
            fixture.block_row_ptr.data(),
            fixture.packed_values.data(),
            host_metadata);
        auto device_matrix = msq::make_matrix<Bits>(
            fixture.rows,
            fixture.cols,
            static_cast<int>(fixture.col_idx.size()),
            block_count,
            d_row_ptr,
            d_packed_row_ptr,
            d_col_idx,
            d_block_row_ptr,
            d_packed_values,
            device_metadata);

        for (int block = 0; block < block_count; ++block) {
            const auto block_view = msq::get_block(&host_matrix, block);

            ok = check_cuda(msq::launch_quantize_block_v100(&device_matrix, block_view, d_input + block_view.nnz_begin),
                            "launch quantize block") && ok;
        }
        ok = check_cuda(cudaDeviceSynchronize(), "sync after quantize") && ok;
        if (!ok) {
            goto cleanup;
        }

        ok = check_cuda(cudaMemcpy(fixture.packed_values.data(),
                                   d_packed_values,
                                   fixture.packed_values.size() * sizeof(unsigned char),
                                   cudaMemcpyDeviceToHost),
                        "copy packed back") && ok;
        if (!ok) {
            goto cleanup;
        }

        if (expected_packed != fixture.packed_values) {
            std::fprintf(stderr, "device packed-bytes mismatch policy=%s bits=%d\n",
                         metadata_case<Metadata>::label(),
                         Bits);
            ok = false;
            goto cleanup;
        }

        for (int block = 0; block < block_count; ++block) {
            const auto block_view = msq::get_block(&host_matrix, block);

            ok = check_cuda(msq::launch_dequantize_block_v100(&device_matrix, block_view, d_output + block_view.nnz_begin),
                            "launch dequantize block") && ok;
        }
        ok = check_cuda(cudaDeviceSynchronize(), "sync after dequantize") && ok;
        ok = check_cuda(cudaMemcpy(fixture.unpacked_values.data(),
                                   d_output,
                                   fixture.unpacked_values.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        "copy output back") && ok;
        if (!ok) {
            goto cleanup;
        }
    }

    for (std::size_t i = 0; i < fixture.input_values.size(); ++i) {
        if (!nearly_equal(fixture.input_values[i], fixture.unpacked_values[i])) {
            std::fprintf(stderr,
                         "device mismatch policy=%s bits=%d index=%zu expected=%f got=%f\n",
                         metadata_case<Metadata>::label(),
                         Bits,
                         i,
                         fixture.input_values[i],
                         fixture.unpacked_values[i]);
            ok = false;
            break;
        }
    }

cleanup:
    cudaFree(d_row_ptr);
    cudaFree(d_packed_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_block_row_ptr);
    cudaFree(d_scales);
    cudaFree(d_offsets);
    cudaFree(d_row_offsets);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_packed_values);
    return ok;
}

template<typename Metadata>
bool run_policy_suite() {
    bool ok = true;

    for (int bits : {1, 2, 4, 8}) {
        ok = msq::dispatch_bits(bits, [&](auto bits_tag) {
                 constexpr int Bits = decltype(bits_tag)::value;
                 return run_host_case<Bits, Metadata>() && run_device_case<Bits, Metadata>();
             }) &&
             ok;
    }
    return ok;
}

} // namespace

int main() {
    const bool ok =
        run_policy_suite<msq::per_gene_affine<float>>() &&
        run_policy_suite<msq::column_scale_row_offset<float>>();

    if (!ok) {
        return 1;
    }

    std::puts("quantized_matrix_test: all policy and format checks passed");
    return 0;
}
