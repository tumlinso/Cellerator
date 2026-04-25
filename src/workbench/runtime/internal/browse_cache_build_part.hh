#pragma once

inline bool build_browse_cache_multigpu(const std::string &path,
                                        const ingest_plan &plan,
                                        std::vector<issue> *issues) {
    cs::sharded<cs::sparse::blocked_ell> matrix;
    cs::shard_storage storage;
    csd::local_context ctx;
    csd::shard_map shard_map;
    browse_cache_owned owned;
    bool ok = false;

    cs::init(&matrix);
    cs::init(&storage);
    csd::init(&ctx);
    csd::init(&shard_map);

    if (plan.policy.cache_dir.empty()) {
        push_issue(issues, issue_severity::error, "browse", "explicit local cache_dir is required for browse cache generation");
        goto done;
    }

    if (!cs::load_header(path.c_str(), &matrix, &storage)) {
        push_issue(issues, issue_severity::error, "browse", "failed to reload dataset header for browse cache build");
        goto done;
    }
    if (!cs::bind_dataset_h5_cache(&storage, plan.policy.cache_dir.c_str())) {
        push_issue(issues, issue_severity::error, "browse", "failed to bind explicit local cache_dir for browse cache generation");
        goto done;
    }

    if (!check_cuda(csd::discover_local(&ctx, 1, cudaStreamNonBlocking), issues, "browse", "discover_local")
        || !check_cuda(csd::enable_peer_access(&ctx), issues, "browse", "enable_peer_access")) goto done;
    if (ctx.device_count < 4u) {
        push_issue(issues, issue_severity::error, "browse", "browse cache generation requires 4 visible GPUs");
        goto done;
    }
    if (!csd::assign_shards_by_bytes(&shard_map, &matrix, &ctx)) {
        push_issue(issues, issue_severity::error, "browse", "failed to assign shards across GPUs");
        goto done;
    }

    {
        host_buffer<int> shard_owner;
        host_buffer<gene_metric_partial> partials;
        host_buffer<std::thread> workers;
        shard_owner.assign_fill(matrix.num_shards, -1);
        partials.resize(ctx.device_count);
        workers.reserve(ctx.device_count);
        for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
            shard_owner[shard_id] = shard_map.device_slot[shard_id];
        }

        for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
            workers.push_back(std::thread([&, slot]() {
                (void) build_gene_metric_partials_blocked_ell(path,
                                                              plan.policy.cache_dir,
                                                              shard_owner,
                                                              slot,
                                                              ctx.device_ids[slot],
                                                              (unsigned int) matrix.cols,
                                                              partials.data() + slot,
                                                              issues);
            }));
        }
        for (std::thread &worker : workers) worker.join();

        owned.gene_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        owned.gene_detected.assign_fill((std::size_t) matrix.cols, 0.0f);
        owned.gene_sq_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        for (const gene_metric_partial &partial : partials) {
            if (!partial.ok) goto done;
            for (std::size_t gene = 0; gene < owned.gene_sum.size(); ++gene) {
                owned.gene_sum[gene] += partial.gene_sum[gene];
                owned.gene_detected[gene] += partial.gene_detected[gene];
                owned.gene_sq_sum[gene] += partial.gene_sq_sum[gene];
            }
        }
    }

    {
        host_buffer<std::pair<float, std::uint32_t>> ranked;
        ranked.reserve(owned.gene_sum.size());
        for (std::uint32_t gene = 0; gene < owned.gene_sum.size(); ++gene) {
            ranked.push_back(std::pair<float, std::uint32_t>{ owned.gene_sum[gene], gene });
        }
        std::partial_sort(ranked.begin(),
                          ranked.begin() + std::min<std::size_t>(plan.policy.browse_top_features, ranked.size()),
                          ranked.end(),
                          [](const auto &lhs, const auto &rhs) {
                              if (lhs.first != rhs.first) return lhs.first > rhs.first;
                              return lhs.second < rhs.second;
                          });
        const std::size_t count = std::min<std::size_t>(plan.policy.browse_top_features, ranked.size());
        owned.selected_feature_indices.assign_fill(count, 0u);
        for (std::size_t i = 0; i < count; ++i) owned.selected_feature_indices[i] = ranked[i].second;
    }

    if (owned.selected_feature_indices.empty()) {
        push_issue(issues, issue_severity::warning, "browse", "browse cache skipped because no features were selected");
        ok = true;
        goto done;
    }

    owned.dataset_feature_mean.assign_fill(plan.datasets.size() * owned.selected_feature_indices.size(), 0.0f);
    owned.shard_feature_mean.assign_fill((std::size_t) matrix.num_shards * owned.selected_feature_indices.size(), 0.0f);
    owned.partition_sample_row_offsets.assign_fill((std::size_t) matrix.num_partitions + 1u, 0u);
    for (unsigned long part_id = 0; part_id < matrix.num_partitions; ++part_id) {
        owned.partition_sample_row_offsets[part_id + 1u] =
            owned.partition_sample_row_offsets[part_id] + plan.policy.browse_sample_rows_per_partition;
    }
    owned.partition_sample_global_rows.assign_fill((std::size_t) matrix.num_partitions * plan.policy.browse_sample_rows_per_partition,
                                                   std::numeric_limits<std::uint64_t>::max());
    owned.partition_sample_values.assign_fill((std::size_t) matrix.num_partitions
                                              * plan.policy.browse_sample_rows_per_partition
                                              * owned.selected_feature_indices.size(),
                                              0.0f);

    {
        host_buffer<int> shard_owner;
        host_buffer<std::uint32_t> part_dataset_indices;
        host_buffer<selected_feature_partial> partials;
        host_buffer<std::thread> workers;
        host_buffer<std::uint32_t> source_to_dataset;
        shard_owner.assign_fill(matrix.num_shards, -1);
        part_dataset_indices.assign_fill(plan.parts.size(), 0u);
        partials.resize(ctx.device_count);
        workers.reserve(ctx.device_count);
        source_to_dataset.assign_fill(plan.sources.size(), std::numeric_limits<std::uint32_t>::max());
        for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
            shard_owner[shard_id] = shard_map.device_slot[shard_id];
        }
        for (std::size_t dataset_index = 0; dataset_index < plan.datasets.size(); ++dataset_index) {
            source_to_dataset[plan.datasets[dataset_index].source_index] = (std::uint32_t) dataset_index;
        }
        for (std::size_t part_index = 0; part_index < plan.parts.size(); ++part_index) {
            const std::size_t source_index = plan.parts[part_index].source_index;
            if (source_index < source_to_dataset.size() && source_to_dataset[source_index] != std::numeric_limits<std::uint32_t>::max()) {
                part_dataset_indices[part_index] = source_to_dataset[source_index];
            }
        }

        for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
            workers.push_back(std::thread([&, slot]() {
                partials[slot].ok = build_selected_feature_partials_blocked_ell(path,
                                                                                plan.policy.cache_dir,
                                                                                shard_owner,
                                                                                slot,
                                                                                ctx.device_ids[slot],
                                                                                part_dataset_indices,
                                                                                (unsigned int) plan.datasets.size(),
                                                                                (unsigned int) matrix.num_shards,
                                                                                owned.selected_feature_indices,
                                                                                plan.policy.browse_sample_rows_per_partition,
                                                                                &partials[slot].dataset_feature_sum,
                                                                                &partials[slot].shard_feature_sum,
                                                                                &owned.partition_sample_global_rows,
                                                                                &owned.partition_sample_values,
                                                                                issues);
            }));
        }
        for (std::thread &worker : workers) worker.join();

        for (const selected_feature_partial &partial : partials) {
            if (!partial.ok) goto done;
            if (partial.dataset_feature_sum.empty()) continue;
            for (std::size_t i = 0; i < owned.dataset_feature_mean.size(); ++i) {
                owned.dataset_feature_mean[i] += partial.dataset_feature_sum[i];
            }
        }
        for (const selected_feature_partial &partial : partials) {
            if (partial.shard_feature_sum.empty()) continue;
            for (std::size_t i = 0; i < owned.shard_feature_mean.size(); ++i) {
                owned.shard_feature_mean[i] += partial.shard_feature_sum[i];
            }
        }
    }

    for (std::size_t dataset_index = 0; dataset_index < plan.datasets.size(); ++dataset_index) {
        const double denom = plan.datasets[dataset_index].rows != 0 ? (double) plan.datasets[dataset_index].rows : 1.0;
        for (std::size_t feature = 0; feature < owned.selected_feature_indices.size(); ++feature) {
            owned.dataset_feature_mean[dataset_index * owned.selected_feature_indices.size() + feature] =
                (float) (owned.dataset_feature_mean[dataset_index * owned.selected_feature_indices.size() + feature] / denom);
        }
    }
    for (std::size_t shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
        const double denom = cs::rows_in_shard(&matrix, (unsigned long) shard_id) != 0
            ? (double) cs::rows_in_shard(&matrix, (unsigned long) shard_id)
            : 1.0;
        for (std::size_t feature = 0; feature < owned.selected_feature_indices.size(); ++feature) {
            owned.shard_feature_mean[shard_id * owned.selected_feature_indices.size() + feature] =
                (float) (owned.shard_feature_mean[shard_id * owned.selected_feature_indices.size() + feature] / denom);
        }
    }

    {
        cellshard::dataset_browse_cache_view view{};
        view.selected_feature_count = (std::uint32_t) owned.selected_feature_indices.size();
        view.selected_feature_indices = owned.selected_feature_indices.data();
        view.gene_sum = owned.gene_sum.data();
        view.gene_detected = owned.gene_detected.data();
        view.gene_sq_sum = owned.gene_sq_sum.data();
        view.dataset_count = (std::uint32_t) plan.datasets.size();
        view.dataset_feature_mean = owned.dataset_feature_mean.data();
        view.shard_count = (std::uint32_t) matrix.num_shards;
        view.shard_feature_mean = owned.shard_feature_mean.data();
        view.partition_count = (std::uint32_t) matrix.num_partitions;
        view.sample_rows_per_partition = plan.policy.browse_sample_rows_per_partition;
        view.partition_sample_row_offsets = owned.partition_sample_row_offsets.data();
        view.partition_sample_global_rows = owned.partition_sample_global_rows.data();
        view.partition_sample_values = owned.partition_sample_values.data();
        if (!cellshard::append_dataset_browse_cache_h5(path.c_str(), &view)) {
            push_issue(issues, issue_severity::error, "browse", "failed to append browse cache to dataset.csh5");
            goto done;
        }
    }

    ok = true;

done:
    csd::clear(&shard_map);
    csd::clear(&ctx);
    cs::clear(&storage);
    cs::clear(&matrix);
    return ok;
}

inline bool build_browse_cache_after_preprocess(const std::string &path,
                                                const dataset_summary &summary,
                                                const preprocess_config &config,
                                                std::vector<issue> *issues) {
    static constexpr unsigned int browse_top_features = 16u;
    static constexpr unsigned int browse_sample_rows_per_partition = 8u;
    cs::sharded<cs::sparse::blocked_ell> matrix;
    cs::shard_storage storage;
    csd::local_context ctx;
    csd::shard_map shard_map;
    browse_cache_owned owned;
    bool ok = false;

    cs::init(&matrix);
    cs::init(&storage);
    csd::init(&ctx);
    csd::init(&shard_map);

    if (config.cache_dir.empty()) {
        push_issue(issues, issue_severity::error, "browse", "explicit local cache_dir is required for browse cache generation");
        goto done;
    }
    if (!summary.ok) {
        push_issue(issues, issue_severity::error, "browse", "cannot rebuild browse cache for an unreadable dataset");
        goto done;
    }
    if (summary.matrix_format.find("sliced") != std::string::npos) {
        push_issue(issues, issue_severity::warning, "browse", "browse cache rebuild is skipped for sliced datasets");
        ok = true;
        goto done;
    }
    if (!cs::load_header(path.c_str(), &matrix, &storage)) {
        push_issue(issues, issue_severity::error, "browse", "failed to reload dataset header for browse cache build");
        goto done;
    }
    if (!cs::bind_dataset_h5_cache(&storage, config.cache_dir.c_str())) {
        push_issue(issues, issue_severity::error, "browse", "failed to bind explicit local cache_dir for browse cache generation");
        goto done;
    }
    if (!check_cuda(csd::discover_local(&ctx, 1, cudaStreamNonBlocking), issues, "browse", "discover_local")
        || !check_cuda(csd::enable_peer_access(&ctx), issues, "browse", "enable_peer_access")) goto done;
    if (ctx.device_count < 4u) {
        push_issue(issues, issue_severity::warning, "browse", "browse cache skipped because 4 visible GPUs are required");
        ok = true;
        goto done;
    }
    if (!csd::assign_shards_by_bytes(&shard_map, &matrix, &ctx)) {
        push_issue(issues, issue_severity::error, "browse", "failed to assign shards across GPUs");
        goto done;
    }

    {
        host_buffer<int> shard_owner;
        host_buffer<gene_metric_partial> partials;
        host_buffer<std::thread> workers;
        shard_owner.assign_fill(matrix.num_shards, -1);
        partials.resize(ctx.device_count);
        workers.reserve(ctx.device_count);
        for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
            shard_owner[shard_id] = shard_map.device_slot[shard_id];
        }
        for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
            workers.push_back(std::thread([&, slot]() {
                (void) build_gene_metric_partials_blocked_ell(path,
                                                              config.cache_dir,
                                                              shard_owner,
                                                              slot,
                                                              ctx.device_ids[slot],
                                                              (unsigned int) matrix.cols,
                                                              partials.data() + slot,
                                                              issues);
            }));
        }
        for (std::thread &worker : workers) worker.join();

        owned.gene_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        owned.gene_detected.assign_fill((std::size_t) matrix.cols, 0.0f);
        owned.gene_sq_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        for (const gene_metric_partial &partial : partials) {
            if (!partial.ok) goto done;
            for (std::size_t gene = 0; gene < owned.gene_sum.size(); ++gene) {
                owned.gene_sum[gene] += partial.gene_sum[gene];
                owned.gene_detected[gene] += partial.gene_detected[gene];
                owned.gene_sq_sum[gene] += partial.gene_sq_sum[gene];
            }
        }
    }

    {
        host_buffer<std::pair<float, std::uint32_t>> ranked;
        ranked.reserve(owned.gene_sum.size());
        for (std::uint32_t gene = 0; gene < owned.gene_sum.size(); ++gene) {
            ranked.push_back(std::pair<float, std::uint32_t>{ owned.gene_sum[gene], gene });
        }
        std::partial_sort(ranked.begin(),
                          ranked.begin() + std::min<std::size_t>(browse_top_features, ranked.size()),
                          ranked.end(),
                          [](const auto &lhs, const auto &rhs) {
                              if (lhs.first != rhs.first) return lhs.first > rhs.first;
                              return lhs.second < rhs.second;
                          });
        const std::size_t count = std::min<std::size_t>(browse_top_features, ranked.size());
        owned.selected_feature_indices.assign_fill(count, 0u);
        for (std::size_t i = 0; i < count; ++i) owned.selected_feature_indices[i] = ranked[i].second;
    }

    if (owned.selected_feature_indices.empty()) {
        push_issue(issues, issue_severity::warning, "browse", "browse cache skipped because no features were selected");
        ok = true;
        goto done;
    }

    owned.dataset_feature_mean.assign_fill(summary.datasets.size() * owned.selected_feature_indices.size(), 0.0f);
    owned.shard_feature_mean.assign_fill((std::size_t) matrix.num_shards * owned.selected_feature_indices.size(), 0.0f);
    owned.partition_sample_row_offsets.assign_fill((std::size_t) matrix.num_partitions + 1u, 0u);
    for (unsigned long part_id = 0; part_id < matrix.num_partitions; ++part_id) {
        owned.partition_sample_row_offsets[part_id + 1u] =
            owned.partition_sample_row_offsets[part_id] + browse_sample_rows_per_partition;
    }
    owned.partition_sample_global_rows.assign_fill((std::size_t) matrix.num_partitions * browse_sample_rows_per_partition,
                                                   std::numeric_limits<std::uint64_t>::max());
    owned.partition_sample_values.assign_fill((std::size_t) matrix.num_partitions
                                              * browse_sample_rows_per_partition
                                              * owned.selected_feature_indices.size(),
                                              0.0f);

    {
        host_buffer<int> shard_owner;
        host_buffer<std::uint32_t> part_dataset_indices;
        host_buffer<selected_feature_partial> partials;
        host_buffer<std::thread> workers;
        shard_owner.assign_fill(matrix.num_shards, -1);
        part_dataset_indices.assign_fill(matrix.num_partitions, 0u);
        partials.resize(ctx.device_count);
        workers.reserve(ctx.device_count);
        for (unsigned long shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
            shard_owner[shard_id] = shard_map.device_slot[shard_id];
        }
        for (std::size_t part_index = 0; part_index < summary.partitions.size() && part_index < part_dataset_indices.size(); ++part_index) {
            part_dataset_indices[part_index] = summary.partitions[part_index].dataset_id;
        }
        for (unsigned int slot = 0; slot < ctx.device_count; ++slot) {
            workers.push_back(std::thread([&, slot]() {
                partials[slot].ok = build_selected_feature_partials_blocked_ell(path,
                                                                                config.cache_dir,
                                                                                shard_owner,
                                                                                slot,
                                                                                ctx.device_ids[slot],
                                                                                part_dataset_indices,
                                                                                (unsigned int) summary.datasets.size(),
                                                                                (unsigned int) matrix.num_shards,
                                                                                owned.selected_feature_indices,
                                                                                browse_sample_rows_per_partition,
                                                                                &partials[slot].dataset_feature_sum,
                                                                                &partials[slot].shard_feature_sum,
                                                                                &owned.partition_sample_global_rows,
                                                                                &owned.partition_sample_values,
                                                                                issues);
            }));
        }
        for (std::thread &worker : workers) worker.join();

        for (const selected_feature_partial &partial : partials) {
            if (!partial.ok) goto done;
            if (partial.dataset_feature_sum.empty()) continue;
            for (std::size_t i = 0; i < owned.dataset_feature_mean.size(); ++i) {
                owned.dataset_feature_mean[i] += partial.dataset_feature_sum[i];
            }
        }
        for (const selected_feature_partial &partial : partials) {
            if (partial.shard_feature_sum.empty()) continue;
            for (std::size_t i = 0; i < owned.shard_feature_mean.size(); ++i) {
                owned.shard_feature_mean[i] += partial.shard_feature_sum[i];
            }
        }
    }

    for (std::size_t dataset_index = 0; dataset_index < summary.datasets.size(); ++dataset_index) {
        const double denom = summary.datasets[dataset_index].rows != 0 ? (double) summary.datasets[dataset_index].rows : 1.0;
        for (std::size_t feature = 0; feature < owned.selected_feature_indices.size(); ++feature) {
            owned.dataset_feature_mean[dataset_index * owned.selected_feature_indices.size() + feature] =
                (float) (owned.dataset_feature_mean[dataset_index * owned.selected_feature_indices.size() + feature] / denom);
        }
    }
    for (std::size_t shard_id = 0; shard_id < matrix.num_shards; ++shard_id) {
        const double denom = cs::rows_in_shard(&matrix, (unsigned long) shard_id) != 0
            ? (double) cs::rows_in_shard(&matrix, (unsigned long) shard_id)
            : 1.0;
        for (std::size_t feature = 0; feature < owned.selected_feature_indices.size(); ++feature) {
            owned.shard_feature_mean[shard_id * owned.selected_feature_indices.size() + feature] =
                (float) (owned.shard_feature_mean[shard_id * owned.selected_feature_indices.size() + feature] / denom);
        }
    }

    {
        cs::dataset_browse_cache_view view{};
        view.selected_feature_count = (std::uint32_t) owned.selected_feature_indices.size();
        view.selected_feature_indices = owned.selected_feature_indices.data();
        view.gene_sum = owned.gene_sum.data();
        view.gene_detected = owned.gene_detected.data();
        view.gene_sq_sum = owned.gene_sq_sum.data();
        view.dataset_count = (std::uint32_t) summary.datasets.size();
        view.dataset_feature_mean = owned.dataset_feature_mean.data();
        view.shard_count = (std::uint32_t) matrix.num_shards;
        view.shard_feature_mean = owned.shard_feature_mean.data();
        view.partition_count = (std::uint32_t) matrix.num_partitions;
        view.sample_rows_per_partition = browse_sample_rows_per_partition;
        view.partition_sample_row_offsets = owned.partition_sample_row_offsets.data();
        view.partition_sample_global_rows = owned.partition_sample_global_rows.data();
        view.partition_sample_values = owned.partition_sample_values.data();
        if (!cs::append_dataset_browse_cache_h5(path.c_str(), &view)) {
            push_issue(issues, issue_severity::error, "browse", "failed to append browse cache to dataset.csh5");
            goto done;
        }
    }

    ok = true;

done:
    csd::clear(&shard_map);
    csd::clear(&ctx);
    cs::clear(&storage);
    cs::clear(&matrix);
    return ok;
}
