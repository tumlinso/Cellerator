#include "feature_major_tiles_fixture.hh"

#include <Cellerator/runtime/value_readiness.cuh>

#include <iostream>

namespace {

using namespace cellerator::test::feature_tiles;
namespace runtime = cellerator::runtime;

Support skewed() {
    Support support{609, 97, {0}, {}};
    for (uint32_t row = 0; row < support.rows; ++row) {
        // Leave an empty tile between populated tiles.
        if (row / 32 != 4 && row % 9 != 8) {
            // Deliberately unordered, one shared feature, two row-specific features.
            support.columns.push_back(49 + (row * 7) % 47);
            support.columns.push_back(0);
            support.columns.push_back(1 + (row * 13) % 43);
        }
        support.offsets.push_back(static_cast<uint32_t>(support.columns.size()));
    }
    return support;
}

void require_readiness(runtime::value_readiness_status status, const char* what) {
    require(status == runtime::value_readiness_status::success, what);
}

void check_training(const Support& support, FeatureMajorTileGrouping grouping,
                    bool expect_chunks) {
    const Stream producer, consumer;
    const HostTiles host(support, grouping);
    const DeviceTiles tiles(host, producer);
    require((tiles.index.chunk_count > 1) == expect_chunks, "unexpected reverse splitting");
    if (grouping == FeatureMajorTileGrouping::local_feature_hash && expect_chunks) {
        FeatureMajorTileStructureView host_structure{};
        FeatureMajorTilesIndexView host_index{};
        check(layout::bind_feature_major_tiles(host.required, host.indices.data(), host.indices.size(),
              host.chunks.data(), host.chunks.size(), &host_structure, &host_index), "bind host layout");
        require(host_index.logical_rows != nullptr, "missing row map");
        bool moved = false;
        for (uint32_t row = 0; row < support.rows; ++row)
            moved = moved || host_index.logical_rows[row] != row;
        require(moved, "grouped fixture did not exercise a row remap");
    }

    auto expected = parameters(support);
    DeviceArray<float> source(expected.x.size()), master(expected.x.size()),
        dx(expected.x.size()), exported_dx(expected.x.size()), w(expected.w.size()),
        dw(expected.w.size()), target(expected.target.size()), y(expected.target.size()),
        g(expected.target.size()), partials(matrix::feature_major_tiles_transpose_partial_count(tiles.index));
    DeviceArray<__half> execution(expected.x.size());
    const FeatureMajorTilesView<__half> writable{tiles.structure, execution.get()};
    const FeatureMajorTilesView<const __half> input{tiles.structure, execution.get()};
    source.upload(expected.x, producer);
    w.upload(expected.w, producer);
    target.upload(expected.target, producer);

    int device = 0;
    check(cudaGetDevice(&device), "current device");
    runtime::value_readiness_record readiness;
    require_readiness(runtime::initialize_value_readiness(&readiness, device), "initialize readiness");
    constexpr uint64_t epoch = 7;
    check(matrix::feature_major_tiles_initialize_master(
        writable, tiles.index, source.get(), master.get(), producer), "initialize master");
    require_readiness(runtime::publish_value_generation(&readiness, epoch, 1, producer, cudaSuccess),
                      "publish initial generation");
    require(runtime::wait_for_value_generation(readiness, epoch, 2, consumer, device) ==
            runtime::value_readiness_status::stale_generation, "accepted future generation");

    const auto execute = [&](uint64_t generation, bool first) {
        require_readiness(runtime::wait_for_value_generation(
            readiness, epoch, generation, consumer, device), "wait for value generation");
        check(matrix::feature_major_tiles_spmm(input, tiles.index, w.get(), y.get(), consumer), "forward");
        loss(y, target, g, consumer);
        if (first && expect_chunks)
            require(matrix::feature_major_tiles_spmm_transpose(input, tiles.index, g.get(), dw.get(),
                    partials.get(), partials.count() - 1, consumer) == cudaErrorInvalidValue,
                    "accepted insufficient transpose workspace");
        check(matrix::feature_major_tiles_spmm_transpose(input, tiles.index, g.get(), dw.get(),
              partials.get(), partials.count(), consumer), "weight gradient");
        check(matrix::feature_major_tiles_sampled_gradient(tiles.structure, tiles.index,
              g.get(), w.get(), dx.get(), consumer), "sampled gradient");
        check(matrix::feature_major_tiles_export_gradient(
              tiles.structure, tiles.index, dx.get(), exported_dx.get(), consumer), "export sparse gradient");
        auto ref = reference(support, expected);
        compare(y.read(consumer), ref.y, "forward");
        compare(g.read(consumer), ref.g, "loss gradient");
        compare(dw.read(consumer), ref.dw, "weight gradient");
        const auto sparse_gradient = exported_dx.read(consumer);
        compare(sparse_gradient, ref.dx, "sampled gradient");
        if (first && !sparse_gradient.empty()) {
            require(expected.x[0] == 0.0f, "fixture value must be zero");
            require(std::abs(sparse_gradient[0]) > 1e-6f, "stored zero lost its sampled gradient");
        }
        return ref;
    };

    for (uint64_t step = 1; step <= 3; ++step) {
        const auto ref = execute(step, step == 1);
        reference_sgd(expected, ref);
        // execute() waited for the consumer. The producer can reuse the execution slot.
        check(matrix::feature_major_tiles_sparse_sgd(
            writable, master.get(), dx.get(), learning_rate, producer), "sparse SGD and cast");
        update_w(w, dw, producer);
        require_readiness(runtime::publish_value_generation(
            &readiness, epoch, step + 1, producer, cudaSuccess), "publish learned generation");
        require(runtime::wait_for_value_generation(readiness, epoch, step, consumer, device) ==
                runtime::value_readiness_status::stale_generation, "accepted stale generation");
        compare(w.read(producer), expected.w, "updated weights");
        if (!expected.x.empty()) {
            FeatureMajorTileStructureView host_structure{};
            FeatureMajorTilesIndexView host_index{};
            check(layout::bind_feature_major_tiles(host.required, host.indices.data(), host.indices.size(),
                  host.chunks.data(), host.chunks.size(), &host_structure, &host_index), "bind master order");
            const auto updated_master = master.read(producer);
            std::vector<float> csr(updated_master.size());
            for (size_t p = 0; p < csr.size(); ++p)
                csr[host_index.source_value_positions[p]] = updated_master[p];
            compare(csr, expected.x, "updated sparse master");
        }
    }

    execute(4, false);
    std::vector<__half> refreshed(expected.x.size());
    for (size_t i = 0; i < refreshed.size(); ++i) {
        refreshed[i] = __float2half_rn(parameter(i, 11, 31, -0.004f));
        expected.x[i] = __half2float(refreshed[i]);
    }
    DeviceArray<__half> external(refreshed.size());
    external.upload(refreshed, producer);
    check(matrix::feature_major_tiles_refresh(writable, tiles.index, external.get(), producer), "CSR refresh");
    require_readiness(runtime::publish_value_generation(&readiness, epoch, 5, producer, cudaSuccess),
                      "publish refreshed generation");
    execute(5, false);
    require_readiness(runtime::clear_value_readiness(&readiness), "clear readiness");
}

} // namespace

int main() {
    try {
        const Support fixture = skewed();
        check_training(fixture, FeatureMajorTileGrouping::contiguous, true);
        check_training(fixture, FeatureMajorTileGrouping::local_feature_hash, true);
        check_training({0, 12, {0}, {}}, FeatureMajorTileGrouping::contiguous, false);
        check_training({3, 0, {0, 0, 0, 0}, {}}, FeatureMajorTileGrouping::local_feature_hash, false);
        check_training({3, 4, {0, 0, 0, 0}, {}}, FeatureMajorTileGrouping::contiguous, false);
        std::cout << "feature-major forward, backward, SGD, refresh and readiness passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
