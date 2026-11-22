#include <Cellerator/interop/torch/coo.hh>

#include <ATen/ATen.h>

#include <string.h>

namespace ct = cellerator::interop::torch;

template<class Function>
void require_rejection(Function function) {
    try {
        function();
    } catch (const c10::Error &) {
        return;
    }
    TORCH_CHECK(false, "expected incompatible COO input to be rejected");
}

void preserve_alias(at::Device device) {
    auto coordinates = at::tensor({-1, 1, 0, 1, 1, 0, 2, 0, 1}, at::kLong)
                           .to(device).narrow(0, 1, 8).view({2, 4});
    auto values = at::tensor({-1.f, 2.f, 5.f, 3.f, 7.f}).to(device).narrow(0, 1, 4);
    auto matrix = ct::wrap_coo_entries(ct::CooEntryOrder::preserve, 2, 3, coordinates, values);
    TORCH_CHECK(!matrix.is_coalesced());
    const auto view = ct::borrow_coo_entries<float>(matrix);
    const auto *coord = static_cast<const int64_t *>(coordinates.const_data_ptr());
    TORCH_CHECK(view.rows == 2 && view.cols == 3 && view.nnz == 4);
    TORCH_CHECK(view.row_idx == coord && view.col_idx == coord + 4 && view.values == values.const_data_ptr());
    if (device.is_cpu()) {
        TORCH_CHECK(view.row_idx[0] == 1 && view.row_idx[1] == 0 && view.row_idx[2] == 1 &&
                    view.col_idx[0] == 0 && view.col_idx[1] == 2 && view.col_idx[2] == 0 &&
                    view.values[0] == 2.f && view.values[2] == 3.f);
        static_cast<float *>(values.mutable_data_ptr())[3] = 11.f;
    }
    auto expected = at::tensor({5.f, 5.f, device.is_cpu() ? 11.f : 7.f});
    coordinates.reset(); values.reset();
    const auto retained = ct::borrow_coo_entries<float>(matrix);
    TORCH_CHECK(retained.row_idx == view.row_idx && retained.col_idx == view.col_idx &&
                retained.values == view.values);
    require_rejection([&] { ct::borrow_coo_entries_fp16(matrix); });

    auto reduced = matrix.coalesce();
    TORCH_CHECK(reduced.is_coalesced() && reduced._nnz() == 3 && matrix._nnz() == 4);
    TORCH_CHECK(reduced.values().cpu().equal(expected));
}

void sorted_alias(at::Device device) {
    auto coordinates = at::tensor({0, 1, 2, 1}, at::kLong).to(device).view({2, 2});
    auto values = at::tensor({2.f, 3.f}).to(at::kHalf).to(device);
    auto matrix = ct::wrap_coo_entries(ct::CooEntryOrder::assume_sorted_unique, 2, 4, coordinates, values);
    const auto view = ct::borrow_coo_entries_fp16(matrix);
    TORCH_CHECK(matrix.is_coalesced() && view.nnz == 2 &&
                view.row_idx == coordinates.const_data_ptr() &&
                view.col_idx == static_cast<const int64_t *>(coordinates.const_data_ptr()) + 2 &&
                view.values == values.const_data_ptr());
}

void pack_coordinates(at::Device device) {
    auto rows = at::tensor({2, 1, 2}, at::kInt).to(device);
    auto cols = at::tensor({0, 4, 0}, at::kLong).to(device);
    auto packed = at::empty({2, 3}, at::TensorOptions().device(device).dtype(at::kLong));
    const void *storage = packed.const_data_ptr();
    ct::pack_coo_indices(packed, rows, cols);
    TORCH_CHECK(packed.const_data_ptr() == storage &&
                packed.cpu().equal(at::tensor({2, 1, 2, 0, 4, 0}, at::kLong).view({2, 3})));

    uint32_t input[] = {UINT32_MAX, 2u, 1u};
    auto unsigned_rows = at::from_blob(input, {3}, at::TensorOptions().dtype(at::kUInt32)).to(device);
    ct::pack_coo_indices(packed, unsigned_rows, cols);
    const auto cpu = packed.cpu();
    const auto *indices = static_cast<const int64_t *>(cpu.const_data_ptr());
    TORCH_CHECK(packed.const_data_ptr() == storage && indices[0] == UINT32_MAX &&
                indices[1] == 2 && indices[2] == 1 && indices[3] == 0 && indices[4] == 4 && indices[5] == 0);
    if (device.is_cuda())
        require_rejection([&] { ct::pack_coo_indices(packed, rows.cpu(), cols); });
    else {
        auto overlapping = at::from_blob(packed.mutable_data_ptr(), {3}, at::TensorOptions().dtype(at::kLong));
        require_rejection([&] { ct::pack_coo_indices(packed, rows, overlapping); });
    }
}

void empty_and_rejections() {
    auto empty_coords = at::empty({2, 0}, at::kLong);
    auto empty_values = at::empty({0}, at::kFloat);
    auto empty_idx = at::empty({0}, at::kInt);
    ct::pack_coo_indices(empty_coords, empty_idx, empty_idx);
    const auto empty = ct::wrap_coo_entries(ct::CooEntryOrder::preserve, 0, 5, empty_coords, empty_values);
    const auto view = ct::borrow_coo_entries<float>(empty);
    TORCH_CHECK(view.rows == 0 && view.cols == 5 && view.nnz == 0 &&
                view.row_idx == empty_coords.const_data_ptr() && view.col_idx == view.row_idx &&
                view.values == empty_values.const_data_ptr());

    auto indices = at::zeros({2, 3}, at::kLong);
    auto values = at::zeros({3}, at::kFloat);
    auto strided_indices = at::zeros({3, 2}, at::kLong).transpose(0, 1);
    auto strided_values = at::zeros({3, 2}, at::kFloat).select(1, 0);
    require_rejection([&] { ct::wrap_coo_entries(ct::CooEntryOrder::preserve, 2, 2, strided_indices, values); });
    require_rejection([&] { ct::wrap_coo_entries(ct::CooEntryOrder::preserve, 2, 2, indices, strided_values); });
    require_rejection([&] { ct::wrap_coo_entries(ct::CooEntryOrder::preserve, 2, 2, indices.to(at::kInt), values); });
    require_rejection([&] { ct::wrap_coo_entries(ct::CooEntryOrder::preserve, 2, 2, indices, empty_values); });
    const auto strided = at::sparse_coo_tensor(strided_indices, values, {2, 2}, values.options().layout(at::kSparse));
    require_rejection([&] { ct::borrow_coo_entries<float>(strided); });
    const auto hybrid = at::sparse_coo_tensor(at::zeros({1, 3}, at::kLong), at::zeros({3, 2}), {2, 2},
                                              values.options().layout(at::kSparse));
    require_rejection([&] { ct::borrow_coo_entries<float>(hybrid); });
    require_rejection([&] { ct::borrow_coo_entries<float>(values); });
}

int main(int argc, char **argv) {
    const bool cuda = argc == 2 && strcmp(argv[1], "--cuda") == 0;
    TORCH_CHECK(argc == 1 || cuda, "expected no arguments or --cuda");
    const auto device = cuda ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
    preserve_alias(device);
    sorted_alias(device);
    pack_coordinates(device);
    if (!cuda) empty_and_rejections();
}
