#pragma once

#include <Cellerator/compute/support/adaptive_support.hh>

#include <pybind11/pybind11.h>

#include <memory>
#include <mutex>
#include <vector>

namespace cellerator::bindings::sparse_support {

struct owned_logical_support_v1 {
    std::uint32_t schema_version = cellerator::compute::support::logical_support_schema_version_v1;
    std::uint64_t relation_id = 0u;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    std::vector<cellerator::compute::support::logical_support_edge_v1> edges;
    std::vector<cellerator::compute::support::logical_support_participant_v1> participants;

    cellerator::compute::support::logical_support_view_v1 view() const noexcept {
        return {schema_version, 0u, relation_id, structure_id, structure_epoch,
            edges.empty() ? nullptr : edges.data(), edges.size(),
            participants.empty() ? nullptr : participants.data(), participants.size()};
    }
};

struct owned_support_snapshot_v1 {
    owned_logical_support_v1 base;
    owned_logical_support_v1 delta;
    std::vector<cellerator::compute::support::support_promotion_record_v1>
        promotion_records;
    std::vector<std::uint64_t> revoked_edge_ids;
    std::uint64_t structure_epoch = 0u;
};

class native_support_handle_v1 final {
public:
    explicit native_support_handle_v1(
        const cellerator::compute::support::logical_support_view_v1 &base,
        std::uint64_t delta_relation_id = 0u);

    cellerator::compute::support::adaptive_support_status_v1 stage_promotion(
        const cellerator::compute::support::staged_support_edge_v1 &proposal) {
        std::lock_guard<std::mutex> lock(mutex_);
        return manager_.stage_promotion(proposal);
    }
    cellerator::compute::support::adaptive_support_status_v1 stage_revocation(
        std::uint64_t edge_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return manager_.stage_revocation(edge_id);
    }
    cellerator::compute::support::adaptive_support_status_v1 stage_base_promotion(
        std::uint64_t edge_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return manager_.stage_base_promotion(edge_id);
    }
    cellerator::compute::support::adaptive_support_status_v1 publish_pending(
        std::uint64_t epoch) {
        std::lock_guard<std::mutex> lock(mutex_);
        return manager_.publish_pending(epoch);
    }
    cellerator::compute::support::adaptive_support_status_v1 publish_pending_snapshot(
        std::uint64_t epoch, owned_support_snapshot_v1* output);
    cellerator::compute::support::adaptive_support_status_v1 restore_snapshot(
        const owned_support_snapshot_v1& snapshot);
    cellerator::compute::support::adaptive_support_status_v1 pin_forward_tape(
        std::uint64_t epoch) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return manager_.pin_forward_tape(epoch);
    }
    cellerator::compute::support::adaptive_support_status_v1 retire_forward_tape(
        std::uint64_t epoch) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return manager_.retire_forward_tape(epoch);
    }
    owned_support_snapshot_v1 snapshot_copy() const;
    std::uint64_t structure_epoch() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return manager_.structure_epoch();
    }
    std::vector<cellerator::compute::support::support_promotion_record_v1>
    promotion_records() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return manager_.promotion_records();
    }
    std::vector<std::uint64_t> revoked_edge_ids() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return manager_.revoked_edge_ids();
    }

private:
    owned_support_snapshot_v1 snapshot_copy_locked() const;
    cellerator::compute::support::adaptive_support_manager_v1 manager_;
    mutable std::mutex mutex_;
};

} // namespace cellerator::bindings::sparse_support

void bind_sparse_support_indexing(pybind11::module_ &module);
