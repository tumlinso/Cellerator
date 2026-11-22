#include <Cellerator/interop/torch/compressed.hh>

#include <ATen/ATen.h>

#include <string.h>

namespace ct = cellerator::interop::torch;
using cellerator::tensor::base::CompressedAxis;

template<class Function>
void require_rejection(Function function) {
    try {
        function();
    } catch (const c10::Error &) {
        return;
    }
    TORCH_CHECK(false, "expected incompatible compressed input to be rejected");
}

void csr_alias(at::Device device) {
    auto offsets = at::tensor({-1, 0, 2, 3}, at::kInt).to(device).narrow(0, 1, 3);
    auto minor = at::tensor({-1, 0, 3, 1}, at::kInt).to(device).narrow(0, 1, 3);
    auto storage = at::tensor({99.f, 1.f, 2.f, 3.f}, at::kFloat).to(device);
    auto values = storage.narrow(0, 1, 3);
    auto matrix = ct::wrap_compressed(CompressedAxis::row, 2, 4, offsets, minor, values);
    const auto view = ct::borrow_compressed<int32_t, float>(matrix);
    TORCH_CHECK(view.structure.axis == CompressedAxis::row && view.structure.rows == 2 &&
                view.structure.cols == 4 && view.structure.nnz == 3);
    TORCH_CHECK(view.structure.major_ptr == offsets.const_data_ptr() &&
                view.structure.minor_idx == minor.const_data_ptr() &&
                view.values == values.const_data_ptr());
    if (device.is_cpu()) static_cast<float *>(values.mutable_data_ptr())[1] = 5.f;
    offsets.reset(); minor.reset(); values.reset(); storage.reset();
    const auto retained = ct::borrow_compressed<int32_t, float>(matrix);
    TORCH_CHECK(retained.structure.major_ptr == view.structure.major_ptr &&
                retained.structure.minor_idx == view.structure.minor_idx && retained.values == view.values);
    if (device.is_cpu())
        TORCH_CHECK(retained.structure.major_ptr[2] == 3 && retained.structure.minor_idx[1] == 3 &&
                    retained.values[1] == 5.f);
    require_rejection([&] { ct::borrow_compressed<int64_t, float>(matrix); });
    require_rejection([&] { ct::borrow_compressed<int32_t, double>(matrix); });
}

void csc_alias(at::Device device) {
    auto offsets = at::tensor({0, 2, 3}, at::kLong).to(device);
    auto minor = at::tensor({0, 3, 1}, at::kLong).to(device);
    auto values = at::tensor({1.f, 2.f, 3.f}).to(at::kHalf).to(device);
    auto matrix = ct::wrap_compressed(CompressedAxis::column, 4, 2, offsets, minor, values);
    const auto view = ct::borrow_compressed_fp16<int64_t>(matrix);
    TORCH_CHECK(view.structure.axis == CompressedAxis::column && view.structure.rows == 4 &&
                view.structure.cols == 2 && view.structure.nnz == 3);
    TORCH_CHECK(view.structure.major_ptr == offsets.const_data_ptr() &&
                view.structure.minor_idx == minor.const_data_ptr() &&
                view.values == values.const_data_ptr());
    const auto edit = ct::borrow_mutable_compressed_fp16<int64_t>(matrix);
    TORCH_CHECK(edit.values == values.const_data_ptr());
    if (device.is_cpu()) {
        static_cast<at::Half *>(edit.values)[1] = at::Half(4.f);
        TORCH_CHECK(float(static_cast<const at::Half *>(values.const_data_ptr())[1]) == 4.f);
    }
    matrix.set_requires_grad(true);
    require_rejection([&] { ct::borrow_mutable_compressed_fp16<int64_t>(matrix); });
}

void empty_and_rejections() {
    auto offsets = at::tensor({0}, at::kInt);
    auto minor = at::empty({0}, at::kInt);
    auto values = at::empty({0}, at::kFloat);
    const auto empty = ct::wrap_compressed(CompressedAxis::row, 0, 4, offsets, minor, values);
    const auto view = ct::borrow_compressed<int32_t, float>(empty);
    TORCH_CHECK(view.structure.rows == 0 && view.structure.cols == 4 && view.structure.nnz == 0);
    TORCH_CHECK(view.structure.major_ptr == offsets.const_data_ptr() &&
                view.structure.minor_idx == minor.const_data_ptr() &&
                view.values == values.const_data_ptr());
    require_rejection([&] { ct::borrow_compressed_fp16<int32_t>(empty); });
    require_rejection([&] { ct::borrow_compressed<int32_t, float>(values); });
    require_rejection([&] { ct::wrap_compressed(CompressedAxis::row, 1, 4, offsets, minor, values); });
    require_rejection([&] { ct::wrap_compressed(CompressedAxis::row, 0, 4, offsets, minor.to(at::kLong), values); });
    require_rejection([&] { ct::wrap_compressed(CompressedAxis::row, 0, int64_t(INT32_MAX) + 1, offsets, minor, values); });

    auto row = at::tensor({0, 2}, at::kInt);
    auto cols = at::tensor({0, 1}, at::kInt);
    auto strided_values = at::ones({2, 2}).select(1, 0);
    TORCH_CHECK(!strided_values.is_contiguous());
    require_rejection([&] { ct::wrap_compressed(CompressedAxis::row, 1, 2, row, cols, strided_values); });
    auto hybrid = at::sparse_csr_tensor(row, cols, at::ones({2, 2}), {1, 2, 2},
                                        at::TensorOptions().layout(at::kSparseCsr));
    require_rejection([&] { ct::borrow_compressed<int32_t, float>(hybrid); });
    auto coords = at::tensor({0, 1, 0, 1}, at::kLong).view({2, 2});
    auto coo = at::sparse_coo_tensor(coords, at::ones({2}), {2, 2},
                                      at::TensorOptions().layout(at::kSparse));
    require_rejection([&] { ct::borrow_compressed<int64_t, float>(coo); });
}

int main(int argc, char **argv) {
    const bool cuda = argc == 2 && strcmp(argv[1], "--cuda") == 0;
    TORCH_CHECK(argc == 1 || cuda, "expected no arguments or --cuda");
    const auto device = cuda ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
    csr_alias(device);
    csc_alias(device);
    if (!cuda) empty_and_rejections();
}
