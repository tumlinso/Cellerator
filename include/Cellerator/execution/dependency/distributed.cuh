#pragma once

#include <Cellerator/execution/dependency/distributed.hh>
#include <Cellerator/execution/dependency/packed_program.cuh>
#include <Cellerator/execution/dependency/transfer.cuh>
#include <Cellerator/ops/mechanism/gated_product.hh>
#include <Cellerator/ops/mechanism/gated_product_host.hh>
#include <Cellerator/tensor/block_masked/structure.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace cellerator::ops::mechanism { class PreparedGatedProductBinding; }

namespace cellerator::execution::dependency {

struct DistributedGatedProductOptions {
    std::shared_ptr<const tensor::block_masked::Structure> structure;
    std::shared_ptr<const ops::mechanism::PreparedGatedProductBinding> semantics;
    // Provider-supplied logical associations, checked against the foundation.
    std::span<const geometry::dependency::SourceDescriptor> source_bindings;
    std::uint64_t access_revision = 1;
    ops::mechanism::GatedProductTopologyView topology;
    // One slot (0 or 1) per group, copied by preparation. Both devices must
    // participate; group ownership does not alter logical argument/output IDs.
    std::span<const std::uint8_t> group_slots;
    std::array<int, 2> devices{0, 1};
    TransferMode transfer_mode = TransferMode::staged;
    std::size_t maximum_live_forwards = 2;
    float beta = 0.25f;
    // Explicit identities of the two prepared physical realizations. These
    // are separate from semantic logical orders and must both be nonzero.
    std::array<structure::order_id, 2> physical_orders{};
    std::array<PackedAtomIdentities, 2> atoms{};
    std::array<structure::structure_id, 2> physical_structures{};
    std::array<std::uint64_t, 2> structure_epochs{1, 1};
};

struct DistributedTransferStats {
    std::uint64_t condition_bytes = 0;
    std::uint64_t participant_bytes = 0;
    std::uint64_t parameter_bytes = 0;
    std::uint64_t output_bytes = 0;
    // Backward payload movement, including the cotangent host upload and all
    // device crossings. This is not solely network/inter-device traffic.
    std::uint64_t backward_bytes = 0;
    std::uint64_t replica_refresh_bytes = 0;
    // Known prepared allocation footprint, not a measured allocator peak.
    std::size_t prepared_device_bytes = 0;
    std::size_t prepared_staging_bytes = 0;
    bool peer_forward = false;
    bool peer_reverse = false;
    std::uint64_t omitted_participant_transfers = 0;
    std::uint64_t projection_refreshes = 0;
    std::uint64_t projected_participant_values = 0;
    std::uint64_t prepared_program_forwards = 0;
    std::uint64_t prepared_program_backwards = 0;
};

struct ParameterLearningRates {
    float theta = 0.0f;
    float phi = 0.0f;
    float tau = 0.0f;
};

namespace detail { struct DistributedProductState; }
class PreparedDistributedGatedProduct;

// Move-only continuing evaluation. It retains actual primal contents until
// backward completion or abandonment, including both device completion events.
// wait() retires completed input leases after backward; output/gradient slots
// remain reserved until this handle is destroyed. Host calls are serialized.
class DistributedProductTape final {
public:
    DistributedProductTape(DistributedProductTape&&) noexcept;
    DistributedProductTape& operator=(DistributedProductTape&&) noexcept;
    ~DistributedProductTape() noexcept;
    DistributedProductTape(const DistributedProductTape&) = delete;
    DistributedProductTape& operator=(const DistributedProductTape&) = delete;
    void wait();
    bool consumed() const;
    std::uint64_t parameter_generation() const;
private:
    friend class PreparedDistributedGatedProduct;
    DistributedProductTape(std::shared_ptr<detail::DistributedProductState>,
                           std::size_t slot, std::uint64_t incarnation);
    void release() noexcept;
    std::shared_ptr<detail::DistributedProductState> state_;
    std::size_t slot_ = 0;
    std::uint64_t incarnation_ = 0;
};

// A concrete prepared consumer of the native packed view, not a model base or
// a generic distributed graph. Preparation reserves all CUDA resources and
// bounded evaluation slots. Values use FP16 bits; products, outputs and VJPs
// use FP32. z is [batch,condition,2], parameters are three [parameter] spans.
// Canonical homes are x/output/parameters at devices[0] and regulators at
// devices[1]; group work runs on both. The other copies are explicit replicas
// or condition-valid packed projections, never independent canonical sources.
// Dense batch axes (z, cotangents and returned y/dx/dz) follow ascending
// Structure::canonical_rows() IDs. The IDs need not be contiguous or zero-based;
// their physical row positions are resolved once during preparation.
// This bounded packed consumer requires output-producing groups and nonempty
// participant coverage on each device. Input-free/outputless compositions can
// use ordinary local operations; the v1 atom adapter rejects unsupported scope.
class PreparedDistributedGatedProduct final {
public:
    PreparedDistributedGatedProduct(
        DistributedGatedProductOptions,
        std::span<const std::uint16_t> packed_values,
        std::span<const float> regulators,
        std::span<const float> theta,
        std::span<const float> phi,
        std::span<const float> tau);
    ~PreparedDistributedGatedProduct();
    PreparedDistributedGatedProduct(const PreparedDistributedGatedProduct&) = delete;
    PreparedDistributedGatedProduct& operator=(const PreparedDistributedGatedProduct&) = delete;
    PreparedDistributedGatedProduct(PreparedDistributedGatedProduct&&) noexcept;
    PreparedDistributedGatedProduct& operator=(PreparedDistributedGatedProduct&&) noexcept;

    DistributedProductTape forward();
    // Copies cotangents into prepared pinned storage before returning. A tape
    // is consumed once; malformed calls fail before consuming or publishing.
    void backward(DistributedProductTape&, std::span<const float> cotangents);
    std::vector<float> values(DistributedProductTape&) const;
    ops::mechanism::GatedProductHostBackward<float> gradients(DistributedProductTape&) const;

    // One canonical SGD update from this evaluation's reduced gradients.
    // Rejects unresolved forward readers, stale gradients and duplicate updates;
    // publishes the canonical generation and refreshes the replica explicitly.
    // The scalar convenience overload updates theta only. Explicit nonzero
    // gate rates opt phi/tau into the same update/publication.
    // Checks completed gradients and a staged proposal before publication;
    // numerical overflow leaves canonical values and generation unchanged.
    void update(DistributedProductTape&, float learning_rate);
    void update(DistributedProductTape&, ParameterLearningRates);
    std::array<std::vector<float>, 3> parameters() const;
    std::uint64_t parameter_generation() const;
    // Wait for update/replica and queued compute completion on both devices.
    void synchronize() const;

    void upload_values(std::span<const std::uint16_t>, std::uint64_t generation);
    void upload_regulators(std::span<const float>, std::uint64_t generation);
    DistributedTransferStats stats() const;
private:
    std::shared_ptr<detail::DistributedProductState> state_;
};

} // namespace cellerator::execution::dependency
