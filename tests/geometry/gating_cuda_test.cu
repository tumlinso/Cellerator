#include <Cellerator/geometry/gating_cuda.cuh>

#include <cuda_runtime_api.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class device_buffer {
public:
    device_buffer() = default;

    explicit device_buffer(std::size_t count) {
        reset(count);
    }

    ~device_buffer() {
        if (ptr_ != nullptr) cudaFree(ptr_);
    }

    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;

    void reset(std::size_t count) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0u;
        }
        count_ = count;
        if (count_ != 0u) {
            require_cuda(cudaMalloc(reinterpret_cast<void **>(&ptr_), count_ * sizeof(T)), "cudaMalloc");
        }
    }

    void copy_from_host(const T *src, std::size_t count) {
        require(count <= count_, "device copy exceeds allocation");
        if (count != 0u) {
            require_cuda(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        }
    }

    std::vector<T> copy_to_host(std::size_t count) const {
        require(count <= count_, "device copy exceeds allocation");
        std::vector<T> out(count);
        if (count != 0u) {
            require_cuda(cudaMemcpy(out.data(), ptr_, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        }
        return out;
    }

    void zero() {
        if (count_ != 0u) {
            require_cuda(cudaMemset(ptr_, 0, count_ * sizeof(T)), "cudaMemset");
        }
    }

    T *get() { return ptr_; }
    const T *get() const { return ptr_; }

private:
    T *ptr_ = nullptr;
    std::size_t count_ = 0u;
};

cellpack::static_plan build_fixture_plan() {
    constexpr cellpack::u32 residual_module = 9u;
    const cellpack::u32 feature_modules[] = {
        0u, 0u, 1u, 1u, residual_module
    };
    const cellpack::u32 row_offsets[] = {
        0u, 2u, 4u, 6u, 8u
    };
    const cellpack::u32 row_modules[] = {
        0u, 1u,
        1u, 0u,
        1u, residual_module,
        0u, residual_module
    };

    cellpack::feature_module_assignment_view features;
    features.feature_to_module = feature_modules;
    features.feature_count = 5u;
    features.residual_module_id = residual_module;

    cellpack::row_signature_view rows;
    rows.row_count = 4u;
    rows.row_offsets = row_offsets;
    rows.module_ids = row_modules;
    rows.entry_count = 8u;

    cellpack::planner_config config;
    config.residual_module_id = residual_module;
    config.min_primary_rows = 2u;

    cellpack::static_plan plan;
    cellpack::validation_result result = cellpack::build_static_plan(features, rows, config, &plan);
    require(static_cast<bool>(result), result.message);
    return plan;
}

cellpack::packed_coordinate_plan build_packed(cellpack::static_plan &plan) {
    const cellpack::u32 offsets[] = { 0u, 3u, 5u, 7u, 9u };
    const cellpack::u32 features[] = {
        0u, 2u, 4u,
        1u, 3u,
        2u, 4u,
        0u, 4u
    };
    const float values[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f,
        6.0f, 7.0f,
        8.0f, 9.0f
    };
    cellpack::csr_view csr;
    csr.row_count = 4u;
    csr.feature_count = 5u;
    csr.nnz_count = 9u;
    csr.row_offsets = offsets;
    csr.feature_ids = features;
    csr.values = values;

    cellpack::packed_coordinate_plan packed;
    cellpack::validation_result result = cellpack::build_packed_coordinate_plan(csr, plan, &packed);
    require(static_cast<bool>(result), result.message);
    return packed;
}

bool route_contains(const cellpack::route_mask &mask, cellpack::u32 region_id) {
    for (cellpack::u32 active : mask.region_ids) {
        if (active == region_id) return true;
    }
    return false;
}

std::vector<float> cpu_forward_reference(
    const cellpack::packed_coordinate_plan &packed,
    const cellpack::route_mask &mask,
    const std::vector<float> &x) {
    std::vector<float> y(packed.row_count, 0.0f);
    for (const cellpack::packed_coordinate &coordinate : packed.coordinates) {
        if (!route_contains(mask, coordinate.region_id)) continue;
        y[coordinate.original_row] += coordinate.value * x[coordinate.original_feature];
    }
    return y;
}

std::vector<float> cpu_backward_reference(
    const cellpack::packed_coordinate_plan &packed,
    const cellpack::route_mask &mask,
    const std::vector<float> &grad_y) {
    std::vector<float> grad_x(packed.feature_count, 0.0f);
    for (const cellpack::packed_coordinate &coordinate : packed.coordinates) {
        if (!route_contains(mask, coordinate.region_id)) continue;
        grad_x[coordinate.original_feature] += coordinate.value * grad_y[coordinate.original_row];
    }
    return grad_x;
}

void require_close(const std::vector<float> &actual, const std::vector<float> &expected, const char *message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1.0e-5f) {
            throw std::runtime_error(message);
        }
    }
}

} // namespace

int main() {
    cellpack::static_plan plan = build_fixture_plan();
    cellpack::packed_coordinate_plan packed = build_packed(plan);

    cellpack::compiled_coordinate_plan compiled;
    cellpack::validation_result result = cellpack::build_compiled_coordinate_plan(plan, packed, &compiled);
    require(static_cast<bool>(result), result.message);

    cellpack::route_mask all_regions;
    result = cellpack::build_oracle_route_mask(
        plan,
        cellpack::oracle_gating_scenario::all_regions,
        0u,
        &all_regions);
    require(static_cast<bool>(result), result.message);

    cellpack::route_mask skipped;
    result = cellpack::build_oracle_route_mask(
        plan,
        cellpack::oracle_gating_scenario::alternating_modules,
        0u,
        &skipped);
    require(static_cast<bool>(result), result.message);
    require(!skipped.region_ids.empty(), "skipped oracle mask is empty");

    cellpack::route_tape tape;
    result = cellpack::record_route_tape(cellpack::view_route_mask(skipped), &tape);
    require(static_cast<bool>(result), result.message);

    cellpack::route_tape wrong_tape;
    result = cellpack::record_route_tape(cellpack::view_route_mask(all_regions), &wrong_tape);
    require(static_cast<bool>(result), result.message);
    result = cellpack::validate_route_tape_for_replay(
        plan,
        cellpack::view_route_mask(skipped),
        cellpack::view_route_tape(wrong_tape));
    require(result.code == cellpack::validation_code::invalid_offsets, "wrong route tape was accepted before launch");

    device_buffer<cellpack::region_coordinate_span> d_spans(compiled.region_spans.size());
    device_buffer<cellpack::u32> d_rows(compiled.row_ids.size());
    device_buffer<cellpack::u32> d_features(compiled.feature_ids.size());
    device_buffer<float> d_values(compiled.values.size());
    d_spans.copy_from_host(compiled.region_spans.data(), compiled.region_spans.size());
    d_rows.copy_from_host(compiled.row_ids.data(), compiled.row_ids.size());
    d_features.copy_from_host(compiled.feature_ids.data(), compiled.feature_ids.size());
    d_values.copy_from_host(compiled.values.data(), compiled.values.size());

    cellpack::device_coordinate_plan_view device_plan;
    device_plan.row_count = compiled.row_count;
    device_plan.feature_count = compiled.feature_count;
    device_plan.region_span_count = static_cast<cellpack::u32>(compiled.region_spans.size());
    device_plan.coordinate_count = static_cast<cellpack::u32>(compiled.values.size());
    device_plan.region_spans = d_spans.get();
    device_plan.row_ids = d_rows.get();
    device_plan.feature_ids = d_features.get();
    device_plan.values = d_values.get();

    device_buffer<cellpack::u32> d_all_mask(all_regions.region_ids.size());
    device_buffer<cellpack::u32> d_skip_mask(skipped.region_ids.size());
    d_all_mask.copy_from_host(all_regions.region_ids.data(), all_regions.region_ids.size());
    d_skip_mask.copy_from_host(skipped.region_ids.data(), skipped.region_ids.size());

    std::vector<float> x = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    std::vector<float> grad_y = { 0.5f, 1.5f, 2.5f, 3.5f };
    device_buffer<float> d_x(x.size());
    device_buffer<float> d_y(plan.desc.row_count);
    device_buffer<float> d_grad_y(grad_y.size());
    device_buffer<float> d_grad_x(plan.desc.feature_count);
    d_x.copy_from_host(x.data(), x.size());
    d_grad_y.copy_from_host(grad_y.data(), grad_y.size());

    d_y.zero();
    cellpack::route_mask_view all_view;
    all_view.region_ids = d_all_mask.get();
    all_view.region_count = static_cast<cellpack::u32>(all_regions.region_ids.size());
    require_cuda(cellpack::launch_route_forward(device_plan, all_view, d_x.get(), d_y.get()), "launch all forward");
    require_cuda(cudaDeviceSynchronize(), "sync all forward");
    require_close(d_y.copy_to_host(plan.desc.row_count), cpu_forward_reference(packed, all_regions, x), "all-region forward mismatch");

    d_y.zero();
    cellpack::route_mask_view skip_view;
    skip_view.region_ids = d_skip_mask.get();
    skip_view.region_count = static_cast<cellpack::u32>(skipped.region_ids.size());
    require_cuda(cellpack::launch_route_forward(device_plan, skip_view, d_x.get(), d_y.get()), "launch skipped forward");
    require_cuda(cudaDeviceSynchronize(), "sync skipped forward");
    require_close(d_y.copy_to_host(plan.desc.row_count), cpu_forward_reference(packed, skipped, x), "oracle-skipped forward mismatch");

    d_grad_x.zero();
    cellpack::route_tape_view tape_view;
    tape_view.region_ids = d_skip_mask.get();
    tape_view.region_count = static_cast<cellpack::u32>(skipped.region_ids.size());
    require_cuda(cellpack::launch_route_backward_replay(device_plan, tape_view, d_grad_y.get(), d_grad_x.get()),
                 "launch skipped backward replay");
    require_cuda(cudaDeviceSynchronize(), "sync skipped backward replay");
    require_close(d_grad_x.copy_to_host(plan.desc.feature_count),
                  cpu_backward_reference(packed, skipped, grad_y),
                  "oracle-skipped backward replay mismatch");

    return 0;
}
