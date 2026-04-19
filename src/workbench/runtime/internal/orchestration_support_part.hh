#pragma once

struct preprocess_partition_assignment {
    host_buffer<int> owner;
    host_buffer<unsigned int> slots;
};

struct preprocess_slot_state {
    int device = -1;
    unsigned int slot = 0u;
    unsigned long partitions_processed = 0ul;
    bool ok = false;
    cpre::device_workspace workspace;
    std::vector<issue> issues;

    preprocess_slot_state() {
        cpre::init(&workspace);
    }

    ~preprocess_slot_state() {
        cpre::clear(&workspace);
    }

    preprocess_slot_state(const preprocess_slot_state &) = delete;
    preprocess_slot_state &operator=(const preprocess_slot_state &) = delete;
};

inline void push_issue(std::vector<issue> *issues,
                       issue_severity severity,
                       const std::string &scope,
                       const std::string &message) {
    if (issues == nullptr) return;
    issues->push_back(issue{severity, scope, message});
}

inline bool check_cuda(cudaError_t status,
                       std::vector<issue> *issues,
                       const std::string &scope,
                       const std::string &label) {
    if (status == cudaSuccess) return true;
    push_issue(issues,
               issue_severity::error,
               scope,
               label + ": " + std::string(cudaGetErrorString(status)));
    return false;
}

inline std::string normalized_upper(std::string value) {
    for (char &ch : value) ch = (char) std::toupper((unsigned char) ch);
    return value;
}

inline host_buffer<unsigned int> select_preprocess_slots(const csd::local_context &ctx,
                                                         const preprocess_config &config) {
    host_buffer<unsigned int> slots;
    if (!config.use_all_devices) {
        if (config.device >= 0 && (unsigned int) config.device < ctx.device_count) {
            slots.push_back((unsigned int) config.device);
        }
        return slots;
    }
    const unsigned int count = std::min<unsigned int>(ctx.device_count, 4u);
    slots.reserve(count);
    for (unsigned int slot = 0u; slot < count; ++slot) slots.push_back(slot);
    return slots;
}

template<typename MatrixT>
inline preprocess_partition_assignment build_preprocess_partition_assignment(const MatrixT &matrix,
                                                                            const host_buffer<unsigned int> &slots) {
    preprocess_partition_assignment assignment;
    host_buffer<std::size_t> slot_bytes;
    struct weighted_part {
        unsigned long id;
        std::size_t bytes;
    };
    host_buffer<weighted_part> weighted;
    assignment.owner.assign_fill((std::size_t) matrix.num_partitions, -1);
    assignment.slots = slots;
    slot_bytes.assign_fill(slots.size(), 0u);
    weighted.reserve((std::size_t) matrix.num_partitions);
    for (unsigned long part_id = 0u; part_id < matrix.num_partitions; ++part_id) {
        weighted.push_back({ part_id, cs::partition_bytes(&matrix, part_id) });
    }
    std::sort(weighted.begin(),
              weighted.end(),
              [](const weighted_part &lhs, const weighted_part &rhs) {
                  if (lhs.bytes != rhs.bytes) return lhs.bytes > rhs.bytes;
                  return lhs.id < rhs.id;
              });
    for (const weighted_part &part : weighted) {
        std::size_t best = 0u;
        for (std::size_t slot = 1u; slot < slots.size(); ++slot) {
            if (slot_bytes[slot] < slot_bytes[best]) best = slot;
        }
        assignment.owner[(std::size_t) part.id] = (int) slots[best];
        slot_bytes[best] += part.bytes;
    }
    return assignment;
}

inline void append_issues(std::vector<issue> *dst, const std::vector<issue> &src) {
    if (dst == nullptr || src.empty()) return;
    dst->insert(dst->end(), src.begin(), src.end());
}

inline bool append_execution_layout_metadata(const ingest_plan &plan,
                                             std::vector<issue> *issues) {
    host_buffer<std::uint32_t> part_formats;
    host_buffer<std::uint32_t> part_block_sizes;
    host_buffer<std::uint32_t> part_bucket_counts;
    host_buffer<float> part_fill_ratios;
    host_buffer<std::uint64_t> part_execution_bytes;
    host_buffer<std::uint64_t> part_blocked_ell_bytes;
    host_buffer<std::uint64_t> part_bucketed_blocked_ell_bytes;
    host_buffer<std::uint32_t> shard_formats;
    host_buffer<std::uint32_t> shard_block_sizes;
    host_buffer<std::uint32_t> shard_bucketed_partition_counts;
    host_buffer<std::uint32_t> shard_bucketed_segment_counts;
    host_buffer<float> shard_fill_ratios;
    host_buffer<std::uint64_t> shard_execution_bytes;
    host_buffer<std::uint64_t> shard_bucketed_blocked_ell_bytes;
    host_buffer<std::uint32_t> shard_pair_ids;
    cellshard::dataset_execution_view execution = {};
    const std::uint32_t persisted_execution_format = cellshard::dataset_execution_format_bucketed_blocked_ell;
    std::uint32_t preferred_base_format = cellshard::dataset_execution_format_unknown;

    part_formats.reserve(plan.parts.size());
    part_block_sizes.reserve(plan.parts.size());
    part_bucket_counts.reserve(plan.parts.size());
    part_fill_ratios.reserve(plan.parts.size());
    part_execution_bytes.reserve(plan.parts.size());
    part_blocked_ell_bytes.reserve(plan.parts.size());
    part_bucketed_blocked_ell_bytes.reserve(plan.parts.size());
    for (const planned_part &part : plan.parts) {
        part_formats.push_back(persisted_execution_format);
        part_block_sizes.push_back(part.blocked_ell_block_size);
        part_bucket_counts.push_back(std::max<std::uint32_t>(1u, part.blocked_ell_bucket_count));
        part_fill_ratios.push_back((float) part.blocked_ell_fill_ratio);
        part_execution_bytes.push_back((std::uint64_t) (part.bucketed_blocked_ell_bytes != 0
            ? part.bucketed_blocked_ell_bytes
            : part.execution_bytes));
        part_blocked_ell_bytes.push_back((std::uint64_t) part.blocked_ell_bytes);
        part_bucketed_blocked_ell_bytes.push_back((std::uint64_t) (part.bucketed_blocked_ell_bytes != 0
            ? part.bucketed_blocked_ell_bytes
            : part.execution_bytes));
    }

    shard_formats.reserve(plan.shards.size());
    shard_block_sizes.reserve(plan.shards.size());
    shard_bucketed_partition_counts.reserve(plan.shards.size());
    shard_bucketed_segment_counts.reserve(plan.shards.size());
    shard_fill_ratios.reserve(plan.shards.size());
    shard_execution_bytes.reserve(plan.shards.size());
    shard_bucketed_blocked_ell_bytes.reserve(plan.shards.size());
    shard_pair_ids.reserve(plan.shards.size());
    for (const planned_shard &shard : plan.shards) {
        const std::uint32_t encoded = persisted_execution_format;
        shard_formats.push_back(encoded);
        shard_block_sizes.push_back(shard.blocked_ell_block_size);
        shard_bucketed_partition_counts.push_back(std::max<std::uint32_t>(1u, shard.bucketed_partition_count));
        shard_bucketed_segment_counts.push_back(std::max<std::uint32_t>(1u, shard.bucketed_segment_count));
        shard_fill_ratios.push_back((float) shard.blocked_ell_fill_ratio);
        shard_execution_bytes.push_back((std::uint64_t) (shard.bucketed_blocked_ell_bytes != 0
            ? shard.bucketed_blocked_ell_bytes
            : shard.execution_bytes));
        shard_bucketed_blocked_ell_bytes.push_back((std::uint64_t) (shard.bucketed_blocked_ell_bytes != 0
            ? shard.bucketed_blocked_ell_bytes
            : shard.execution_bytes));
        shard_pair_ids.push_back(shard.preferred_pair);
        if (preferred_base_format == cellshard::dataset_execution_format_unknown) {
            preferred_base_format = encoded;
        } else if (preferred_base_format != encoded) {
            preferred_base_format = cellshard::dataset_execution_format_mixed;
        }
    }
    if (preferred_base_format == cellshard::dataset_execution_format_unknown) {
        preferred_base_format = cellshard::dataset_execution_format_blocked_ell;
    }

    execution.partition_count = (std::uint32_t) part_formats.size();
    execution.partition_execution_formats = part_formats.empty() ? nullptr : part_formats.data();
    execution.partition_blocked_ell_block_sizes = part_block_sizes.empty() ? nullptr : part_block_sizes.data();
    execution.partition_blocked_ell_bucket_counts = part_bucket_counts.empty() ? nullptr : part_bucket_counts.data();
    execution.partition_blocked_ell_fill_ratios = part_fill_ratios.empty() ? nullptr : part_fill_ratios.data();
    execution.partition_execution_bytes = part_execution_bytes.empty() ? nullptr : part_execution_bytes.data();
    execution.partition_blocked_ell_bytes = part_blocked_ell_bytes.empty() ? nullptr : part_blocked_ell_bytes.data();
    execution.partition_bucketed_blocked_ell_bytes = part_bucketed_blocked_ell_bytes.empty() ? nullptr : part_bucketed_blocked_ell_bytes.data();
    execution.shard_count = (std::uint32_t) shard_formats.size();
    execution.shard_execution_formats = shard_formats.empty() ? nullptr : shard_formats.data();
    execution.shard_blocked_ell_block_sizes = shard_block_sizes.empty() ? nullptr : shard_block_sizes.data();
    execution.shard_bucketed_partition_counts = shard_bucketed_partition_counts.empty() ? nullptr : shard_bucketed_partition_counts.data();
    execution.shard_bucketed_segment_counts = shard_bucketed_segment_counts.empty() ? nullptr : shard_bucketed_segment_counts.data();
    execution.shard_blocked_ell_fill_ratios = shard_fill_ratios.empty() ? nullptr : shard_fill_ratios.data();
    execution.shard_execution_bytes = shard_execution_bytes.empty() ? nullptr : shard_execution_bytes.data();
    execution.shard_bucketed_blocked_ell_bytes = shard_bucketed_blocked_ell_bytes.empty() ? nullptr : shard_bucketed_blocked_ell_bytes.data();
    execution.shard_preferred_pair_ids = shard_pair_ids.empty() ? nullptr : shard_pair_ids.data();
    execution.preferred_base_format = preferred_base_format;

    if (!cellshard::append_dataset_execution_h5(plan.policy.output_path.c_str(), &execution)) {
        push_issue(issues, issue_severity::error, "execution", "failed to append execution layout metadata");
        return false;
    }
    {
        cellshard::dataset_runtime_service_view runtime_service{};
        cellshard::init(&runtime_service);
        runtime_service.service_mode = cellshard::dataset_runtime_service_mode_owner_hosted;
        runtime_service.live_write_mode = cellshard::dataset_live_write_mode_append_only;
        runtime_service.prefer_pack_delivery = 1u;
        runtime_service.remote_pack_delivery = 0u;
        runtime_service.single_reader_coordinator = 1u;
        runtime_service.maintenance_lock_blocks_overwrite = 1u;
        runtime_service.canonical_generation = 1u;
        runtime_service.execution_plan_generation = 1u;
        runtime_service.pack_generation = 1u;
        runtime_service.service_epoch = 1u;
        runtime_service.active_read_generation = 1u;
        runtime_service.staged_write_generation = 1u;
        if (!cellshard::append_dataset_runtime_service_h5(plan.policy.output_path.c_str(), &runtime_service)) {
            push_issue(issues, issue_severity::error, "runtime_service", "failed to append runtime service metadata");
            return false;
        }
    }
    return true;
}

inline host_buffer<unsigned char> build_gene_flags(const dataset_summary &summary,
                                                   const preprocess_config &config) {
    host_buffer<unsigned char> flags;
    flags.assign_fill(summary.cols, static_cast<unsigned char>(0u));
    if (!config.mark_mito_from_feature_names || config.mito_prefix.empty()) return flags;
    const std::string prefix = normalized_upper(config.mito_prefix);
    for (std::size_t i = 0; i < summary.feature_names.size() && i < flags.size(); ++i) {
        std::string name = normalized_upper(summary.feature_names[i]);
        if (name.rfind(prefix, 0) == 0) flags[i] = (unsigned char) cpre::gene_flag_mito;
    }
    return flags;
}
