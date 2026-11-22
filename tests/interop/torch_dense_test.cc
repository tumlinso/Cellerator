#include <Cellerator/interop/torch/dense.hh>

#include <ATen/ATen.h>

#include <stdint.h>
#include <string.h>

namespace ct = cellerator::interop::torch;
namespace cm = cellerator::memory;

template<class Function>
void require_rejection(Function function) {
    try {
        function();
    } catch (const c10::Error &) {
        return;
    }
    TORCH_CHECK(false, "expected incompatible dense input to be rejected");
}

cm::placement placement(at::Device device) {
    return device.is_cuda() ? cm::placement{cm::domain::device, int16_t(device.index())}
                            : cm::placement{cm::domain::host};
}

void transpose_and_offset_alias(at::Device device) {
    auto owner = at::arange(30, at::TensorOptions().dtype(at::kFloat)).to(device).view({5, 6});
    auto padded = owner.narrow(0, 1, 3).narrow(1, 1, 4);
    const auto rows = ct::borrow_dense<float>(padded);
    TORCH_CHECK(rows.data == padded.const_data_ptr() && rows.data != owner.const_data_ptr() &&
                rows.rows == 3 && rows.columns == 4 && rows.row_stride == 6 && rows.column_stride == 1 &&
                rows.where == placement(device));

    auto transposed = padded.transpose(0, 1);
    const auto edit = ct::borrow_mutable_dense<float>(transposed);
    const auto columns = cm::as_const(edit);
    TORCH_CHECK(columns.rows == 4 && columns.columns == 3 && columns.data == rows.data &&
                columns.row_stride == 1 && columns.column_stride == 6 && columns.where == rows.where);
    auto wrapped = ct::wrap_dense(edit);
    TORCH_CHECK(wrapped.const_data_ptr() == rows.data && wrapped.device() == device &&
                wrapped.size(0) == 4 && wrapped.size(1) == 3 && wrapped.stride(0) == 1 && wrapped.stride(1) == 6);
    wrapped.select(0, 2).select(0, 1).fill_(101.f);
    TORCH_CHECK(owner.select(0, 2).select(0, 3).item<float>() == 101.f);
}

void native_wrap_and_empty(at::Device device) {
    float stack[24]{};
    at::Tensor backing;
    float *data = stack + 2;
    if (device.is_cuda()) {
        backing = at::zeros({24}, at::TensorOptions().dtype(at::kFloat).device(device));
        data = static_cast<float *>(backing.mutable_data_ptr()) + 2;
    }
    cm::matrix_view<float> native{data, 2, 3, 7, placement(device), 2};
    auto wrapped = ct::wrap_dense(native);
    TORCH_CHECK(wrapped.const_data_ptr() == data && wrapped.stride(0) == 7 && wrapped.stride(1) == 2);
    wrapped.select(0, 1).select(0, 2).fill_(17.f);
    if (device.is_cpu()) {
        TORCH_CHECK(stack[13] == 17.f);
    } else {
        TORCH_CHECK(backing.select(0, 13).item<float>() == 17.f);
    }

    auto empty = ct::wrap_dense(cm::matrix_view<float>{nullptr, 3, 0, 5, placement(device), 2});
    const auto view = ct::borrow_dense<float>(empty);
    TORCH_CHECK(view.rows == 3 && view.columns == 0 && view.row_stride == 5 &&
                view.column_stride == 2 && view.where == placement(device));
}

void fp16_alias(at::Device device) {
    auto owner = at::ones({3, 4}, at::kHalf).to(device);
    auto values = owner.narrow(1, 1, 2);
    const auto read = ct::borrow_dense<at::Half>(values);
    const auto edit = ct::borrow_mutable_dense<at::Half>(values);
    auto wrapped = ct::wrap_dense(edit);
    TORCH_CHECK(read.data == values.const_data_ptr() && edit.data == read.data &&
                read.row_stride == 4 && read.column_stride == 1 &&
                wrapped.scalar_type() == at::kHalf && wrapped.const_data_ptr() == read.data);
    wrapped.select(0, 2).select(0, 1).fill_(3.f);
    TORCH_CHECK(owner.select(0, 2).select(0, 2).item<float>() == 3.f);
}

void incompatible_layouts() {
    auto wrong_rank = at::ones({2, 3, 4});
    auto wrong_dtype = at::ones({2, 3}, at::kDouble);
    auto sparse = at::eye(3).to_sparse();
    auto meta = at::empty({2, 3}, at::TensorOptions().device(at::kMeta));
    auto conjugate = at::ones({2, 3}, at::kComplexFloat).conj();
    require_rejection([&] { ct::borrow_dense<float>(wrong_rank); });
    require_rejection([&] { ct::borrow_dense<float>(wrong_dtype); });
    require_rejection([&] { ct::borrow_dense<float>(sparse); });
    require_rejection([&] { ct::borrow_dense<float>(meta); });
    require_rejection([&] { ct::borrow_dense<c10::complex<float>>(conjugate); });

    alignas(float) unsigned char bytes[4*sizeof(float) + 1]{};
    auto unaligned = at::from_blob(bytes + 1, {2, 2}, at::TensorOptions().dtype(at::kFloat));
    require_rejection([&] { ct::borrow_dense<float>(unaligned); });

    auto broadcast = at::ones({1, 3}).expand({4, 3});
    auto overlapping = at::ones({8}).as_strided({3, 3}, {1, 2});
    auto strided = at::ones({8}).as_strided({2, 2}, {3, 2});
    TORCH_CHECK(ct::borrow_dense<float>(broadcast).row_stride == 0);
    TORCH_CHECK(ct::borrow_dense<float>(overlapping).column_stride == 2);
    TORCH_CHECK(ct::borrow_mutable_dense<float>(strided).row_stride == 3);
    require_rejection([&] { ct::borrow_mutable_dense<float>(broadcast); });
    require_rejection([&] { ct::borrow_mutable_dense<float>(overlapping); });
    strided.set_requires_grad(true);
    require_rejection([&] { ct::borrow_mutable_dense<float>(strided); });

    const cm::placement host{cm::domain::host};
    float data[8]{};
    require_rejection([&] { ct::wrap_dense(cm::matrix_view<float>{data, 2, 2, 1, host, 1}); });
    require_rejection([&] { ct::wrap_dense(cm::matrix_view<float>{nullptr, 2, 2, 3, host, 1}); });
    require_rejection([&] { ct::wrap_dense(cm::matrix_view<float>{data, 2, 2, 3, {}, 1}); });
    require_rejection([&] { ct::wrap_dense(cm::matrix_view<float>{data, 2, 2, size_t(INT64_MAX), host, 1}); });
}

int main(int argc, char **argv) {
    const bool cuda = argc == 2 && strcmp(argv[1], "--cuda") == 0;
    TORCH_CHECK(argc == 1 || cuda, "expected no arguments or --cuda");
    const auto device = cuda ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
    transpose_and_offset_alias(device);
    native_wrap_and_empty(device);
    fp16_alias(device);
    if (!cuda) incompatible_layouts();
}
