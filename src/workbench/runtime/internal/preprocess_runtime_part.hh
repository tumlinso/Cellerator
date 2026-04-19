#pragma once

inline void fill_preprocess_summary_from_analysis(preprocess_summary *summary,
                                                  const preprocess_analysis_table &analysis) {
    if (summary == nullptr) return;
    summary->ok = analysis.ok;
    summary->device = analysis.device;
    summary->partitions_processed = analysis.partitions_processed;
    summary->rows = analysis.rows;
    summary->cols = analysis.cols;
    summary->nnz = analysis.nnz;
    summary->kept_cells = analysis.kept_cells;
    summary->kept_genes = analysis.kept_genes;
    summary->gene_sum_checksum = analysis.gene_sum_checksum;
    summary->issues = analysis.issues;
}

inline void finalize_gene_keep_mask(const host_buffer<float> &gene_sum,
                                    const host_buffer<float> &gene_sq_sum,
                                    const host_buffer<float> &gene_detected,
                                    float kept_cells,
                                    const cpre::gene_filter_params &gene_filter,
                                    host_buffer<unsigned char> *keep_genes) {
    if (keep_genes == nullptr) return;
    keep_genes->assign_fill(gene_sum.size(), static_cast<unsigned char>(0u));
    const float inv_cells = kept_cells > 0.0f ? (1.0f / kept_cells) : 0.0f;
    for (unsigned int gene = 0u; gene < (unsigned int) gene_sum.size(); ++gene) {
        const float mean = gene_sum[gene] * inv_cells;
        const float var = std::max(gene_sq_sum[gene] * inv_cells - mean * mean, 0.0f);
        (*keep_genes)[gene] =
            (unsigned char) (gene_sum[gene] >= gene_filter.min_sum
                             && gene_detected[gene] >= gene_filter.min_detected_cells
                             && var >= gene_filter.min_variance);
    }
}

inline preprocess_analysis_table analyze_blocked_dataset_preprocess(const std::string &path,
                                                                   const preprocess_config &config,
                                                                   const dataset_summary &dataset) {
    preprocess_analysis_table analysis;
    cs::sharded<cs::sparse::blocked_ell> matrix;
    cs::shard_storage storage;
    cellshard::bucketed_blocked_ell_partition exec_part;
    csv::partition_record<cs::sparse::sliced_ell> device_part;
    cpre::device_workspace workspace;
    host_buffer<unsigned char> gene_flags;
    host_buffer<unsigned char> host_keep_genes;
    host_buffer<float> host_gene_sum;
    host_buffer<float> host_gene_sq_sum;
    host_buffer<float> host_gene_detected;
    host_buffer<float> host_cell_total_counts;
    host_buffer<float> host_cell_mito_counts;
    host_buffer<float> host_cell_max_counts;
    host_buffer<unsigned int> host_cell_detected_genes;
    host_buffer<unsigned char> host_keep_cells;
    float kept_cells = 0.0f;
    const cpre::cell_filter_params cell_filter = {
        config.min_counts,
        config.min_genes,
        config.max_mito_fraction
    };
    const cpre::gene_filter_params gene_filter = {
        config.min_gene_sum,
        config.min_detected_cells,
        config.min_variance
    };

    analysis.path = path;
    analysis.matrix_format = dataset.matrix_format;
    cs::init(&matrix);
    cs::init(&storage);
    cellshard::init(&exec_part);
    csv::zero_record(&device_part);
    cpre::init(&workspace);

    if (config.cache_dir.empty()) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "explicit local cache_dir is required for preprocess analysis");
        return analysis;
    }

    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), &analysis.issues, "preprocess", "cudaGetDeviceCount")) goto done;
    if (device_count == 0) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "no CUDA devices are available");
        goto done;
    }
    if (config.device < 0 || config.device >= device_count) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "requested CUDA device is out of range");
        goto done;
    }

    if (!cs::load_header(path.c_str(), &matrix, &storage)) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to load blocked-ELL dataset header");
        goto done;
    }
    if (!cs::bind_dataset_h5_cache(&storage, config.cache_dir.c_str())) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to bind explicit local cache_dir");
        goto done;
    }
    if (!cpre::setup(&workspace, config.device, (cudaStream_t) 0)
        || !cpre::reserve(&workspace, max_partition_rows(matrix), (unsigned int) matrix.cols, max_partition_nnz(matrix))) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to set up preprocess workspace");
        goto done;
    }

    host_cell_total_counts.assign_fill((std::size_t) matrix.rows, 0.0f);
    host_cell_mito_counts.assign_fill((std::size_t) matrix.rows, 0.0f);
    host_cell_max_counts.assign_fill((std::size_t) matrix.rows, 0.0f);
    host_cell_detected_genes.assign_fill((std::size_t) matrix.rows, 0u);
    host_keep_cells.assign_fill((std::size_t) matrix.rows, static_cast<unsigned char>(0u));
    host_gene_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
    host_gene_sq_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
    host_gene_detected.assign_fill((std::size_t) matrix.cols, 0.0f);
    gene_flags = build_gene_flags(dataset, config);

    for (unsigned long part_id = 0; part_id < matrix.num_partitions; ++part_id) {
        host_buffer<unsigned char> exec_gene_flags;
        host_buffer<float> exec_gene_sum;
        host_buffer<float> exec_gene_sq_sum;
        host_buffer<float> exec_gene_detected;
        if (!fetch_execution_partition(&exec_part, &matrix, &storage, part_id, &analysis.issues, "preprocess", "fetch blocked execution partition")) goto done;
        exec_gene_flags.assign_fill((std::size_t) matrix.cols, static_cast<unsigned char>(0u));
        for (unsigned int exec_col = 0u; exec_col < (unsigned int) matrix.cols; ++exec_col) {
            const unsigned int canonical_col =
                exec_part.exec_to_canonical_cols != nullptr ? exec_part.exec_to_canonical_cols[exec_col] : exec_col;
            exec_gene_flags[exec_col] = canonical_col < gene_flags.size() ? gene_flags[canonical_col] : static_cast<unsigned char>(0u);
        }
        if (!cpre::upload_gene_flags(&workspace, (unsigned int) matrix.cols, exec_gene_flags.empty() ? nullptr : exec_gene_flags.data())
            || !cpre::zero_gene_metrics(&workspace, (unsigned int) matrix.cols)) {
            push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to initialize blocked preprocess workspace state");
            goto done;
        }

        for (std::uint32_t segment = 0u; segment < exec_part.segment_count; ++segment) {
            csv::sliced_ell_view part_view;
            cpre::part_preprocess_result part_result;
            owned_sliced_ell_host host;
            host_buffer<float> seg_total_counts;
            host_buffer<float> seg_mito_counts;
            host_buffer<float> seg_max_counts;
            host_buffer<unsigned int> seg_detected_genes;
            host_buffer<unsigned char> seg_keep_cells;
            if (!build_preprocess_sliced_segment(exec_part.segments + segment, 32u, config.device, &host.part)) {
                push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to build sliced preprocess bridge segment");
                goto done;
            }
            if (!check_cuda(cudaSetDevice(config.device), &analysis.issues, "preprocess", "cudaSetDevice blocked bridge upload")
                || !check_cuda(csv::upload(&host.part, &device_part), &analysis.issues, "preprocess", "upload blocked bridge segment")) {
                goto done;
            }
            if (!cpre::bind_uploaded_part_view(&part_view, &host.part, &device_part)) {
                push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to bind blocked bridge segment");
                goto done;
            }
            if (!cpre::preprocess_part_inplace(&part_view, &workspace, cell_filter, config.target_sum, &part_result)) {
                push_issue(&analysis.issues, issue_severity::error, "preprocess", "blocked preprocess kernel pass failed");
                goto done;
            }
            if (!check_cuda(cudaStreamSynchronize(workspace.stream), &analysis.issues, "preprocess", "cudaStreamSynchronize blocked part")) goto done;
            if (part_result.cell.rows != 0u) {
                const std::size_t row_count = (std::size_t) part_result.cell.rows;
                seg_total_counts.assign_fill(row_count, 0.0f);
                seg_mito_counts.assign_fill(row_count, 0.0f);
                seg_max_counts.assign_fill(row_count, 0.0f);
                seg_detected_genes.assign_fill(row_count, 0u);
                seg_keep_cells.assign_fill(row_count, static_cast<unsigned char>(0u));
                if (!check_cuda(cudaMemcpy(seg_total_counts.data(),
                                           part_result.cell.total_counts,
                                           row_count * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues, "preprocess", "cudaMemcpy blocked cell total counts")) goto done;
                if (!check_cuda(cudaMemcpy(seg_mito_counts.data(),
                                           part_result.cell.mito_counts,
                                           row_count * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues, "preprocess", "cudaMemcpy blocked cell mito counts")) goto done;
                if (!check_cuda(cudaMemcpy(seg_max_counts.data(),
                                           part_result.cell.max_counts,
                                           row_count * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues, "preprocess", "cudaMemcpy blocked cell max counts")) goto done;
                if (!check_cuda(cudaMemcpy(seg_detected_genes.data(),
                                           part_result.cell.detected_genes,
                                           row_count * sizeof(unsigned int),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues, "preprocess", "cudaMemcpy blocked cell detected genes")) goto done;
                if (!check_cuda(cudaMemcpy(seg_keep_cells.data(),
                                           part_result.cell.keep_cells,
                                           row_count * sizeof(unsigned char),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues, "preprocess", "cudaMemcpy blocked cell keep mask")) goto done;
                for (std::size_t local_row = 0; local_row < row_count; ++local_row) {
                    const std::size_t exec_row = (std::size_t) exec_part.segment_row_offsets[segment] + local_row;
                    const std::size_t canonical_row = (std::size_t) exec_part.exec_to_canonical_rows[exec_row];
                    const std::size_t dst_row = (std::size_t) matrix.partition_offsets[part_id] + canonical_row;
                    host_cell_total_counts[dst_row] = seg_total_counts[local_row];
                    host_cell_mito_counts[dst_row] = seg_mito_counts[local_row];
                    host_cell_max_counts[dst_row] = seg_max_counts[local_row];
                    host_cell_detected_genes[dst_row] = seg_detected_genes[local_row];
                    host_keep_cells[dst_row] = seg_keep_cells[local_row];
                }
            }
            if (!check_cuda(csv::release(&device_part), &analysis.issues, "preprocess", "release blocked bridge segment")) goto done;
        }

        exec_gene_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        exec_gene_sq_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        exec_gene_detected.assign_fill((std::size_t) matrix.cols, 0.0f);
        if (!check_cuda(cudaMemcpy(exec_gene_sum.data(),
                                   workspace.d_gene_sum,
                                   exec_gene_sum.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        &analysis.issues, "preprocess", "cudaMemcpy blocked gene sum")) goto done;
        if (!check_cuda(cudaMemcpy(exec_gene_sq_sum.data(),
                                   workspace.d_gene_sq_sum,
                                   exec_gene_sq_sum.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        &analysis.issues, "preprocess", "cudaMemcpy blocked gene sq sum")) goto done;
        if (!check_cuda(cudaMemcpy(exec_gene_detected.data(),
                                   workspace.d_gene_detected,
                                   exec_gene_detected.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        &analysis.issues, "preprocess", "cudaMemcpy blocked gene detected")) goto done;
        for (unsigned int exec_col = 0u; exec_col < (unsigned int) matrix.cols; ++exec_col) {
            const unsigned int canonical_col =
                exec_part.exec_to_canonical_cols != nullptr ? exec_part.exec_to_canonical_cols[exec_col] : exec_col;
            host_gene_sum[canonical_col] += exec_gene_sum[exec_col];
            host_gene_sq_sum[canonical_col] += exec_gene_sq_sum[exec_col];
            host_gene_detected[canonical_col] += exec_gene_detected[exec_col];
        }
        ++analysis.partitions_processed;
        cellshard::clear(&exec_part);
    }

    kept_cells = 0.0f;
    for (unsigned char keep : host_keep_cells) kept_cells += keep != 0u ? 1.0f : 0.0f;
    finalize_gene_keep_mask(host_gene_sum, host_gene_sq_sum, host_gene_detected, kept_cells, gene_filter, &host_keep_genes);
    analysis.ok = true;
    analysis.device = config.device;
    analysis.rows = matrix.rows;
    analysis.cols = matrix.cols;
    analysis.nnz = matrix.nnz;
    analysis.kept_cells = kept_cells;
    for (unsigned char keep : host_keep_genes) analysis.kept_genes += keep != 0u ? 1ul : 0ul;
    for (float value : host_gene_sum) analysis.gene_sum_checksum += (double) value;
    analysis.cell_total_counts.assign(host_cell_total_counts.begin(), host_cell_total_counts.end());
    analysis.cell_mito_counts.assign(host_cell_mito_counts.begin(), host_cell_mito_counts.end());
    analysis.cell_max_counts.assign(host_cell_max_counts.begin(), host_cell_max_counts.end());
    analysis.cell_detected_genes.assign(host_cell_detected_genes.begin(), host_cell_detected_genes.end());
    analysis.cell_keep.assign(host_keep_cells.begin(), host_keep_cells.end());
    analysis.gene_sum.assign(host_gene_sum.begin(), host_gene_sum.end());
    analysis.gene_sq_sum.assign(host_gene_sq_sum.begin(), host_gene_sq_sum.end());
    analysis.gene_detected_cells.assign(host_gene_detected.begin(), host_gene_detected.end());
    analysis.gene_keep.assign(host_keep_genes.begin(), host_keep_genes.end());
    analysis.gene_flags.assign(gene_flags.begin(), gene_flags.end());

done:
    if (device_part.view != nullptr) (void) csv::release(&device_part);
    cellshard::clear(&exec_part);
    cpre::clear(&workspace);
    cs::clear(&storage);
    cs::clear(&matrix);
    return analysis;
}

inline preprocess_analysis_table analyze_sliced_dataset_preprocess(const std::string &path,
                                                                  const preprocess_config &config,
                                                                  const dataset_summary &dataset) {
    preprocess_analysis_table analysis;
    cs::sharded<cs::sparse::sliced_ell> matrix;
    cs::shard_storage storage;
    cellshard::bucketed_sliced_ell_partition exec_part;
    cs::dataset_sliced_execution_device_partition_view staged_part;
    csv::partition_record<cs::sparse::sliced_ell> device_part;
    cpre::device_workspace workspace;
    host_buffer<unsigned char> gene_flags;
    host_buffer<unsigned char> host_keep_genes;
    host_buffer<float> host_gene_sum;
    host_buffer<float> host_gene_sq_sum;
    host_buffer<float> host_gene_detected;
    host_buffer<float> host_cell_total_counts;
    host_buffer<float> host_cell_mito_counts;
    host_buffer<float> host_cell_max_counts;
    host_buffer<unsigned int> host_cell_detected_genes;
    host_buffer<unsigned char> host_keep_cells;
    float kept_cells = 0.0f;
    const cpre::cell_filter_params cell_filter = {
        config.min_counts,
        config.min_genes,
        config.max_mito_fraction
    };
    const cpre::gene_filter_params gene_filter = {
        config.min_gene_sum,
        config.min_detected_cells,
        config.min_variance
    };

    analysis.path = path;
    analysis.matrix_format = dataset.matrix_format;
    cs::init(&matrix);
    cs::init(&storage);
    cellshard::init(&exec_part);
    cs::init(&staged_part);
    csv::zero_record(&device_part);
    cpre::init(&workspace);

    if (config.cache_dir.empty()) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "explicit local cache_dir is required for preprocess analysis");
        return analysis;
    }

    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), &analysis.issues, "preprocess", "cudaGetDeviceCount")) goto done;
    if (device_count == 0) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "no CUDA devices are available");
        goto done;
    }
    if (config.device < 0 || config.device >= device_count) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "requested CUDA device is out of range");
        goto done;
    }

    if (!cs::load_dataset_sliced_ell_h5_header(path.c_str(), &matrix, &storage)) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to load sliced-ELL dataset header");
        goto done;
    }
    if (!cs::bind_dataset_h5_cache(&storage, config.cache_dir.c_str())) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to bind explicit local cache_dir");
        goto done;
    }
    if (!cpre::setup(&workspace, config.device, (cudaStream_t) 0)
        || !cpre::reserve(&workspace, max_partition_rows(matrix), (unsigned int) matrix.cols, max_partition_nnz(matrix))) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to set up preprocess workspace");
        goto done;
    }

    host_cell_total_counts.assign_fill((std::size_t) matrix.rows, 0.0f);
    host_cell_mito_counts.assign_fill((std::size_t) matrix.rows, 0.0f);
    host_cell_max_counts.assign_fill((std::size_t) matrix.rows, 0.0f);
    host_cell_detected_genes.assign_fill((std::size_t) matrix.rows, 0u);
    host_keep_cells.assign_fill((std::size_t) matrix.rows, static_cast<unsigned char>(0u));
    host_gene_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
    host_gene_sq_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
    host_gene_detected.assign_fill((std::size_t) matrix.cols, 0.0f);
    gene_flags = build_gene_flags(dataset, config);

    for (unsigned long part_id = 0; part_id < matrix.num_partitions; ++part_id) {
        const cellshard::bucketed_sliced_ell_partition *active_exec_part = 0;
        const csv::partition_record<cs::sparse::sliced_ell> *active_device_segments = 0;
        host_buffer<float> exec_gene_sum;
        host_buffer<float> exec_gene_sq_sum;
        host_buffer<float> exec_gene_detected;
        if (!cpre::upload_gene_flags(&workspace, (unsigned int) matrix.cols, gene_flags.empty() ? nullptr : gene_flags.data())
            || !cpre::zero_gene_metrics(&workspace, (unsigned int) matrix.cols)) {
            push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to initialize sliced preprocess workspace state");
            goto done;
        }
        if (config.enable_sliced_device_cache) {
            if (!cs::acquire_dataset_sliced_ell_h5_execution_partition_device(&staged_part,
                                                                              &matrix,
                                                                              &storage,
                                                                              part_id,
                                                                              config.device,
                                                                              config.sliced_device_cache_bytes)) {
                push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to acquire cached sliced execution partition");
                goto done;
            }
            active_exec_part = staged_part.host_partition;
            active_device_segments = staged_part.device_segments;
        } else {
            if (!fetch_execution_partition(&exec_part, &matrix, &storage, part_id, &analysis.issues, "preprocess", "fetch sliced execution partition")) goto done;
            active_exec_part = &exec_part;
        }
        if (active_exec_part == nullptr) {
            push_issue(&analysis.issues, issue_severity::error, "preprocess", "missing sliced execution partition state");
            goto done;
        }
        for (std::uint32_t segment = 0u; segment < active_exec_part->segment_count; ++segment) {
            csv::sliced_ell_view part_view;
            cpre::part_preprocess_result part_result;
            host_buffer<float> seg_total_counts;
            host_buffer<float> seg_mito_counts;
            host_buffer<float> seg_max_counts;
            host_buffer<unsigned int> seg_detected_genes;
            host_buffer<unsigned char> seg_keep_cells;
            if (config.enable_sliced_device_cache) {
                if (active_device_segments == nullptr) {
                    push_issue(&analysis.issues, issue_severity::error, "preprocess", "cached sliced execution partition is missing device segments");
                    goto done;
                }
            } else if (!check_cuda(cudaSetDevice(config.device), &analysis.issues, "preprocess", "cudaSetDevice sliced upload")
                       || !check_cuda(csv::upload(active_exec_part->segments + segment, &device_part),
                                      &analysis.issues,
                                      "preprocess",
                                      "upload sliced execution segment")) {
                goto done;
            }
            if (!cpre::bind_uploaded_part_view(&part_view,
                                               active_exec_part->segments + segment,
                                               config.enable_sliced_device_cache ? (active_device_segments + segment) : &device_part)) {
                push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to bind uploaded sliced execution segment");
                goto done;
            }
            if (!cpre::preprocess_part_inplace(&part_view, &workspace, cell_filter, config.target_sum, &part_result)) {
                push_issue(&analysis.issues, issue_severity::error, "preprocess", "sliced preprocess kernel pass failed");
                goto done;
            }
            if (!check_cuda(cudaStreamSynchronize(workspace.stream), &analysis.issues, "preprocess", "cudaStreamSynchronize sliced part")) goto done;
            if (part_result.cell.rows != 0u) {
                const std::size_t row_count = (std::size_t) part_result.cell.rows;
                seg_total_counts.assign_fill(row_count, 0.0f);
                seg_mito_counts.assign_fill(row_count, 0.0f);
                seg_max_counts.assign_fill(row_count, 0.0f);
                seg_detected_genes.assign_fill(row_count, 0u);
                seg_keep_cells.assign_fill(row_count, static_cast<unsigned char>(0u));
                if (!check_cuda(cudaMemcpy(seg_total_counts.data(),
                                           part_result.cell.total_counts,
                                           row_count * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues, "preprocess", "cudaMemcpy sliced cell total counts")) goto done;
                if (!check_cuda(cudaMemcpy(seg_mito_counts.data(),
                                           part_result.cell.mito_counts,
                                           row_count * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues, "preprocess", "cudaMemcpy sliced cell mito counts")) goto done;
                if (!check_cuda(cudaMemcpy(seg_max_counts.data(),
                                           part_result.cell.max_counts,
                                           row_count * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues, "preprocess", "cudaMemcpy sliced cell max counts")) goto done;
                if (!check_cuda(cudaMemcpy(seg_detected_genes.data(),
                                           part_result.cell.detected_genes,
                                           row_count * sizeof(unsigned int),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues, "preprocess", "cudaMemcpy sliced cell detected genes")) goto done;
                if (!check_cuda(cudaMemcpy(seg_keep_cells.data(),
                                           part_result.cell.keep_cells,
                                           row_count * sizeof(unsigned char),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues, "preprocess", "cudaMemcpy sliced cell keep mask")) goto done;
                for (std::size_t local_row = 0; local_row < row_count; ++local_row) {
                    const std::size_t exec_row = (std::size_t) active_exec_part->segment_row_offsets[segment] + local_row;
                    const std::size_t canonical_row = (std::size_t) active_exec_part->exec_to_canonical_rows[exec_row];
                    const std::size_t dst_row = (std::size_t) matrix.partition_offsets[part_id] + canonical_row;
                    host_cell_total_counts[dst_row] = seg_total_counts[local_row];
                    host_cell_mito_counts[dst_row] = seg_mito_counts[local_row];
                    host_cell_max_counts[dst_row] = seg_max_counts[local_row];
                    host_cell_detected_genes[dst_row] = seg_detected_genes[local_row];
                    host_keep_cells[dst_row] = seg_keep_cells[local_row];
                }
            }
            if (!config.enable_sliced_device_cache
                && !check_cuda(csv::release(&device_part), &analysis.issues, "preprocess", "release sliced execution segment")) goto done;
        }

        exec_gene_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        exec_gene_sq_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        exec_gene_detected.assign_fill((std::size_t) matrix.cols, 0.0f);
        if (!check_cuda(cudaMemcpy(exec_gene_sum.data(),
                                   workspace.d_gene_sum,
                                   exec_gene_sum.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        &analysis.issues, "preprocess", "cudaMemcpy sliced gene sum")) goto done;
        if (!check_cuda(cudaMemcpy(exec_gene_sq_sum.data(),
                                   workspace.d_gene_sq_sum,
                                   exec_gene_sq_sum.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        &analysis.issues, "preprocess", "cudaMemcpy sliced gene sq sum")) goto done;
        if (!check_cuda(cudaMemcpy(exec_gene_detected.data(),
                                   workspace.d_gene_detected,
                                   exec_gene_detected.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        &analysis.issues, "preprocess", "cudaMemcpy sliced gene detected")) goto done;
        for (unsigned int gene = 0u; gene < (unsigned int) matrix.cols; ++gene) {
            host_gene_sum[gene] += exec_gene_sum[gene];
            host_gene_sq_sum[gene] += exec_gene_sq_sum[gene];
            host_gene_detected[gene] += exec_gene_detected[gene];
        }
        ++analysis.partitions_processed;
        if (config.enable_sliced_device_cache) {
            (void) cs::release_dataset_sliced_ell_h5_execution_partition_device(&staged_part);
        } else {
            cellshard::clear(&exec_part);
        }
    }

    kept_cells = 0.0f;
    for (unsigned char keep : host_keep_cells) kept_cells += keep != 0u ? 1.0f : 0.0f;
    finalize_gene_keep_mask(host_gene_sum, host_gene_sq_sum, host_gene_detected, kept_cells, gene_filter, &host_keep_genes);
    analysis.ok = true;
    analysis.device = config.device;
    analysis.rows = matrix.rows;
    analysis.cols = matrix.cols;
    analysis.nnz = matrix.nnz;
    analysis.kept_cells = kept_cells;
    for (unsigned char keep : host_keep_genes) analysis.kept_genes += keep != 0u ? 1ul : 0ul;
    for (float value : host_gene_sum) analysis.gene_sum_checksum += (double) value;
    analysis.cell_total_counts.assign(host_cell_total_counts.begin(), host_cell_total_counts.end());
    analysis.cell_mito_counts.assign(host_cell_mito_counts.begin(), host_cell_mito_counts.end());
    analysis.cell_max_counts.assign(host_cell_max_counts.begin(), host_cell_max_counts.end());
    analysis.cell_detected_genes.assign(host_cell_detected_genes.begin(), host_cell_detected_genes.end());
    analysis.cell_keep.assign(host_keep_cells.begin(), host_keep_cells.end());
    analysis.gene_sum.assign(host_gene_sum.begin(), host_gene_sum.end());
    analysis.gene_sq_sum.assign(host_gene_sq_sum.begin(), host_gene_sq_sum.end());
    analysis.gene_detected_cells.assign(host_gene_detected.begin(), host_gene_detected.end());
    analysis.gene_keep.assign(host_keep_genes.begin(), host_keep_genes.end());
    analysis.gene_flags.assign(gene_flags.begin(), gene_flags.end());

done:
    if (device_part.view != nullptr) (void) csv::release(&device_part);
    (void) cs::release_dataset_sliced_ell_h5_execution_partition_device(&staged_part);
    cellshard::clear(&exec_part);
    cpre::clear(&workspace);
    cs::clear(&storage);
    cs::clear(&matrix);
    return analysis;
}

inline preprocess_analysis_table analyze_sliced_dataset_preprocess_multigpu(const std::string &path,
                                                                            const preprocess_config &config,
                                                                            const dataset_summary &dataset) {
    preprocess_analysis_table analysis;
    cs::sharded<cs::sparse::sliced_ell> matrix;
    cs::shard_storage storage;
    csd::local_context ctx;
    host_buffer<unsigned char> gene_flags;
    host_buffer<unsigned char> host_keep_genes;
    host_buffer<float> host_gene_sum;
    host_buffer<float> host_gene_sq_sum;
    host_buffer<float> host_gene_detected;
    host_buffer<float> host_cell_total_counts;
    host_buffer<float> host_cell_mito_counts;
    host_buffer<float> host_cell_max_counts;
    host_buffer<unsigned int> host_cell_detected_genes;
    host_buffer<unsigned char> host_keep_cells;
    float kept_cells = 0.0f;
    const cpre::cell_filter_params cell_filter = {
        config.min_counts,
        config.min_genes,
        config.max_mito_fraction
    };
    const cpre::gene_filter_params gene_filter = {
        config.min_gene_sum,
        config.min_detected_cells,
        config.min_variance
    };

    analysis.path = path;
    analysis.matrix_format = dataset.matrix_format;
    cs::init(&matrix);
    cs::init(&storage);
    csd::init(&ctx);

    if (config.cache_dir.empty()) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "explicit local cache_dir is required for preprocess analysis");
        return analysis;
    }

    if (!check_cuda(csd::discover_local(&ctx, 1, cudaStreamNonBlocking),
                    &analysis.issues,
                    "preprocess",
                    "discover_local preprocess")
        || !check_cuda(csd::enable_peer_access(&ctx),
                       &analysis.issues,
                       "preprocess",
                       "enable_peer_access preprocess")) {
        goto done;
    }
    if (ctx.device_count == 0u) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "no CUDA devices are available");
        goto done;
    }
    if (!cs::load_dataset_sliced_ell_h5_header(path.c_str(), &matrix, &storage)) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to load sliced-ELL dataset header");
        goto done;
    }
    if (!cs::bind_dataset_h5_cache(&storage, config.cache_dir.c_str())) {
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "failed to bind explicit local cache_dir");
        goto done;
    }

    {
        const std::vector<unsigned int> slots = select_preprocess_slots(ctx, config);
        if (slots.size() < 2u || matrix.num_partitions < 2ul) {
            csd::clear(&ctx);
            cs::clear(&storage);
            cs::clear(&matrix);
            return analyze_sliced_dataset_preprocess(path, config, dataset);
        }

        const preprocess_partition_assignment assignment = build_preprocess_partition_assignment(matrix, slots);
        const unsigned int max_rows = max_partition_rows(matrix);
        const unsigned int max_nnz = max_partition_nnz(matrix);
        std::vector<std::unique_ptr<preprocess_slot_state>> states;
        std::vector<std::thread> workers;
        bool all_ok = true;
        bool reduced_on_device = false;

        host_cell_total_counts.assign_fill((std::size_t) matrix.rows, 0.0f);
        host_cell_mito_counts.assign_fill((std::size_t) matrix.rows, 0.0f);
        host_cell_max_counts.assign_fill((std::size_t) matrix.rows, 0.0f);
        host_cell_detected_genes.assign_fill((std::size_t) matrix.rows, 0u);
        host_keep_cells.assign_fill((std::size_t) matrix.rows, static_cast<unsigned char>(0u));
        host_gene_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        host_gene_sq_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
        host_gene_detected.assign_fill((std::size_t) matrix.cols, 0.0f);
        gene_flags = build_gene_flags(dataset, config);

        states.reserve(slots.size());
        for (unsigned int slot : slots) {
            std::unique_ptr<preprocess_slot_state> state(new preprocess_slot_state());
            state->slot = slot;
            state->device = ctx.device_ids[slot];
            if (!cpre::setup(&state->workspace,
                             state->device,
                             ctx.streams != nullptr ? ctx.streams[slot] : (cudaStream_t) 0)
                || !cpre::reserve(&state->workspace, max_rows, (unsigned int) matrix.cols, max_nnz)
                || !cpre::upload_gene_flags(&state->workspace,
                                            (unsigned int) matrix.cols,
                                            gene_flags.empty() ? nullptr : gene_flags.data())
                || !cpre::zero_gene_metrics(&state->workspace, (unsigned int) matrix.cols)) {
                append_issues(&analysis.issues, state->issues);
                push_issue(&analysis.issues,
                           issue_severity::error,
                           "preprocess",
                           "failed to initialize multi-GPU preprocess workspace state");
                goto done;
            }
            states.push_back(std::move(state));
        }

        workers.reserve(states.size());
        for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
            workers.emplace_back([&, state_index]() {
                preprocess_slot_state &state = *states[state_index];
                cs::sharded<cs::sparse::sliced_ell> worker_matrix;
                cs::shard_storage worker_storage;
                cellshard::bucketed_sliced_ell_partition exec_part;
                csv::partition_record<cs::sparse::sliced_ell> device_part;

                cs::init(&worker_matrix);
                cs::init(&worker_storage);
                cellshard::init(&exec_part);
                csv::zero_record(&device_part);

                if (!cs::load_dataset_sliced_ell_h5_header(path.c_str(), &worker_matrix, &worker_storage)) {
                    push_issue(&state.issues, issue_severity::error, "preprocess", "failed to reload sliced-ELL dataset header for worker");
                    goto worker_done;
                }
                if (!cs::bind_dataset_h5_cache(&worker_storage, config.cache_dir.c_str())) {
                    push_issue(&state.issues, issue_severity::error, "preprocess", "failed to bind explicit local cache_dir for worker");
                    goto worker_done;
                }

                for (unsigned long part_id = 0u; part_id < worker_matrix.num_partitions; ++part_id) {
                    if (assignment.owner[(std::size_t) part_id] != (int) state.slot) continue;
                    if (!fetch_execution_partition(&exec_part,
                                                   &worker_matrix,
                                                   &worker_storage,
                                                   part_id,
                                                   &state.issues,
                                                   "preprocess",
                                                   "fetch sliced execution partition")) {
                        goto worker_done;
                    }

                    for (std::uint32_t segment = 0u; segment < exec_part.segment_count; ++segment) {
                        csv::sliced_ell_view part_view;
                        cpre::part_preprocess_result part_result;
                        host_buffer<float> seg_total_counts;
                        host_buffer<float> seg_mito_counts;
                        host_buffer<float> seg_max_counts;
                        host_buffer<unsigned int> seg_detected_genes;
                        host_buffer<unsigned char> seg_keep_cells;

                        if (!check_cuda(cudaSetDevice(state.device),
                                        &state.issues,
                                        "preprocess",
                                        "cudaSetDevice sliced upload multigpu")
                            || !check_cuda(csv::upload(exec_part.segments + segment, &device_part),
                                           &state.issues,
                                           "preprocess",
                                           "upload sliced execution segment multigpu")) {
                            goto worker_done;
                        }
                        if (!cpre::bind_uploaded_part_view(&part_view, exec_part.segments + segment, &device_part)) {
                            push_issue(&state.issues,
                                       issue_severity::error,
                                       "preprocess",
                                       "failed to bind uploaded sliced execution segment");
                            goto worker_done;
                        }
                        if (!cpre::preprocess_part_inplace(&part_view,
                                                           &state.workspace,
                                                           cell_filter,
                                                           config.target_sum,
                                                           &part_result)) {
                            push_issue(&state.issues,
                                       issue_severity::error,
                                       "preprocess",
                                       "sliced preprocess kernel pass failed");
                            goto worker_done;
                        }
                        if (!check_cuda(cudaStreamSynchronize(state.workspace.stream),
                                        &state.issues,
                                        "preprocess",
                                        "cudaStreamSynchronize sliced part multigpu")) {
                            goto worker_done;
                        }

                        if (part_result.cell.rows != 0u) {
                            const std::size_t row_count = (std::size_t) part_result.cell.rows;
                            seg_total_counts.assign_fill(row_count, 0.0f);
                            seg_mito_counts.assign_fill(row_count, 0.0f);
                            seg_max_counts.assign_fill(row_count, 0.0f);
                            seg_detected_genes.assign_fill(row_count, 0u);
                            seg_keep_cells.assign_fill(row_count, static_cast<unsigned char>(0u));
                            if (!check_cuda(cudaMemcpy(seg_total_counts.data(),
                                                       part_result.cell.total_counts,
                                                       row_count * sizeof(float),
                                                       cudaMemcpyDeviceToHost),
                                            &state.issues,
                                            "preprocess",
                                            "cudaMemcpy sliced cell total counts multigpu")) goto worker_done;
                            if (!check_cuda(cudaMemcpy(seg_mito_counts.data(),
                                                       part_result.cell.mito_counts,
                                                       row_count * sizeof(float),
                                                       cudaMemcpyDeviceToHost),
                                            &state.issues,
                                            "preprocess",
                                            "cudaMemcpy sliced cell mito counts multigpu")) goto worker_done;
                            if (!check_cuda(cudaMemcpy(seg_max_counts.data(),
                                                       part_result.cell.max_counts,
                                                       row_count * sizeof(float),
                                                       cudaMemcpyDeviceToHost),
                                            &state.issues,
                                            "preprocess",
                                            "cudaMemcpy sliced cell max counts multigpu")) goto worker_done;
                            if (!check_cuda(cudaMemcpy(seg_detected_genes.data(),
                                                       part_result.cell.detected_genes,
                                                       row_count * sizeof(unsigned int),
                                                       cudaMemcpyDeviceToHost),
                                            &state.issues,
                                            "preprocess",
                                            "cudaMemcpy sliced cell detected genes multigpu")) goto worker_done;
                            if (!check_cuda(cudaMemcpy(seg_keep_cells.data(),
                                                       part_result.cell.keep_cells,
                                                       row_count * sizeof(unsigned char),
                                                       cudaMemcpyDeviceToHost),
                                            &state.issues,
                                            "preprocess",
                                            "cudaMemcpy sliced cell keep mask multigpu")) goto worker_done;
                            for (std::size_t local_row = 0; local_row < row_count; ++local_row) {
                                const std::size_t exec_row = (std::size_t) exec_part.segment_row_offsets[segment] + local_row;
                                const std::size_t canonical_row = (std::size_t) exec_part.exec_to_canonical_rows[exec_row];
                                const std::size_t dst_row = (std::size_t) worker_matrix.partition_offsets[part_id] + canonical_row;
                                host_cell_total_counts[dst_row] = seg_total_counts[local_row];
                                host_cell_mito_counts[dst_row] = seg_mito_counts[local_row];
                                host_cell_max_counts[dst_row] = seg_max_counts[local_row];
                                host_cell_detected_genes[dst_row] = seg_detected_genes[local_row];
                                host_keep_cells[dst_row] = seg_keep_cells[local_row];
                            }
                        }

                        if (!check_cuda(csv::release(&device_part),
                                        &state.issues,
                                        "preprocess",
                                        "release sliced execution segment multigpu")) {
                            goto worker_done;
                        }
                    }

                    cellshard::clear(&exec_part);
                    ++state.partitions_processed;
                }

                state.ok = true;

worker_done:
                if (device_part.view != nullptr) (void) csv::release(&device_part);
                cellshard::clear(&exec_part);
                cs::clear(&worker_storage);
                cs::clear(&worker_matrix);
            });
        }
        for (std::thread &worker : workers) worker.join();

        for (const std::unique_ptr<preprocess_slot_state> &state : states) {
            append_issues(&analysis.issues, state->issues);
            analysis.partitions_processed += state->partitions_processed;
            if (!state->ok) all_ok = false;
        }
        if (!all_ok) goto done;

#if CELLSHARD_HAS_NCCL
        {
            std::vector<const void *> sendbufs(states.size());
            std::vector<void *> recvbufs(states.size());
            for (std::size_t i = 0; i < states.size(); ++i) {
                sendbufs[i] = states[i]->workspace.d_gene_sum;
                recvbufs[i] = states[i]->workspace.d_gene_sum;
            }
            if (csd::local_allreduce(&ctx,
                                     slots.data(),
                                     (unsigned int) slots.size(),
                                     sendbufs.data(),
                                     recvbufs.data(),
                                     (std::size_t) matrix.cols,
                                     ncclFloat32,
                                     ncclSum) == ncclSuccess) {
                for (std::size_t i = 0; i < states.size(); ++i) {
                    sendbufs[i] = states[i]->workspace.d_gene_sq_sum;
                    recvbufs[i] = states[i]->workspace.d_gene_sq_sum;
                }
                if (csd::local_allreduce(&ctx,
                                         slots.data(),
                                         (unsigned int) slots.size(),
                                         sendbufs.data(),
                                         recvbufs.data(),
                                         (std::size_t) matrix.cols,
                                         ncclFloat32,
                                         ncclSum) == ncclSuccess) {
                    for (std::size_t i = 0; i < states.size(); ++i) {
                        sendbufs[i] = states[i]->workspace.d_gene_detected;
                        recvbufs[i] = states[i]->workspace.d_gene_detected;
                    }
                    if (csd::local_allreduce(&ctx,
                                             slots.data(),
                                             (unsigned int) slots.size(),
                                             sendbufs.data(),
                                             recvbufs.data(),
                                             (std::size_t) matrix.cols,
                                             ncclFloat32,
                                             ncclSum) == ncclSuccess) {
                        for (std::size_t i = 0; i < states.size(); ++i) {
                            sendbufs[i] = states[i]->workspace.d_active_rows;
                            recvbufs[i] = states[i]->workspace.d_active_rows;
                        }
                        if (csd::local_allreduce(&ctx,
                                                 slots.data(),
                                                 (unsigned int) slots.size(),
                                                 sendbufs.data(),
                                                 recvbufs.data(),
                                                 1u,
                                                 ncclFloat32,
                                                 ncclSum) == ncclSuccess
                            && check_cuda(csd::synchronize(&ctx),
                                          &analysis.issues,
                                          "preprocess",
                                          "synchronize preprocess nccl allreduce")
                            && check_cuda(cudaSetDevice(states.front()->device),
                                          &analysis.issues,
                                          "preprocess",
                                          "cudaSetDevice preprocess nccl root")
                            && check_cuda(cudaMemcpy(host_gene_sum.data(),
                                                     states.front()->workspace.d_gene_sum,
                                                     host_gene_sum.size() * sizeof(float),
                                                     cudaMemcpyDeviceToHost),
                                          &analysis.issues,
                                          "preprocess",
                                          "cudaMemcpy reduced sliced gene sum")
                            && check_cuda(cudaMemcpy(host_gene_sq_sum.data(),
                                                     states.front()->workspace.d_gene_sq_sum,
                                                     host_gene_sq_sum.size() * sizeof(float),
                                                     cudaMemcpyDeviceToHost),
                                          &analysis.issues,
                                          "preprocess",
                                          "cudaMemcpy reduced sliced gene sq sum")
                            && check_cuda(cudaMemcpy(host_gene_detected.data(),
                                                     states.front()->workspace.d_gene_detected,
                                                     host_gene_detected.size() * sizeof(float),
                                                     cudaMemcpyDeviceToHost),
                                          &analysis.issues,
                                          "preprocess",
                                          "cudaMemcpy reduced sliced gene detected")
                            && check_cuda(cudaMemcpy(&kept_cells,
                                                     states.front()->workspace.d_active_rows,
                                                     sizeof(float),
                                                     cudaMemcpyDeviceToHost),
                                          &analysis.issues,
                                          "preprocess",
                                          "cudaMemcpy reduced sliced active rows")) {
                            reduced_on_device = true;
                        }
                    }
                }
            }
        }
#endif

        if (!reduced_on_device) {
            host_buffer<float> slot_gene_sum;
            host_buffer<float> slot_gene_sq_sum;
            host_buffer<float> slot_gene_detected;
            float slot_kept_cells = 0.0f;
            push_issue(&analysis.issues,
                       issue_severity::warning,
                       "preprocess",
                       "falling back to host reduction for multi-GPU gene metrics");
            slot_gene_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
            slot_gene_sq_sum.assign_fill((std::size_t) matrix.cols, 0.0f);
            slot_gene_detected.assign_fill((std::size_t) matrix.cols, 0.0f);
            kept_cells = 0.0f;
            for (const std::unique_ptr<preprocess_slot_state> &state : states) {
                if (!check_cuda(cudaSetDevice(state->device),
                                &analysis.issues,
                                "preprocess",
                                "cudaSetDevice preprocess host reduce")) {
                    goto done;
                }
                if (!check_cuda(cudaMemcpy(slot_gene_sum.data(),
                                           state->workspace.d_gene_sum,
                                           slot_gene_sum.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues,
                                "preprocess",
                                "cudaMemcpy sliced gene sum host reduce")) goto done;
                if (!check_cuda(cudaMemcpy(slot_gene_sq_sum.data(),
                                           state->workspace.d_gene_sq_sum,
                                           slot_gene_sq_sum.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues,
                                "preprocess",
                                "cudaMemcpy sliced gene sq sum host reduce")) goto done;
                if (!check_cuda(cudaMemcpy(slot_gene_detected.data(),
                                           state->workspace.d_gene_detected,
                                           slot_gene_detected.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues,
                                "preprocess",
                                "cudaMemcpy sliced gene detected host reduce")) goto done;
                if (!check_cuda(cudaMemcpy(&slot_kept_cells,
                                           state->workspace.d_active_rows,
                                           sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                &analysis.issues,
                                "preprocess",
                                "cudaMemcpy sliced active rows host reduce")) goto done;
                kept_cells += slot_kept_cells;
                for (std::size_t gene = 0; gene < host_gene_sum.size(); ++gene) {
                    host_gene_sum[gene] += slot_gene_sum[gene];
                    host_gene_sq_sum[gene] += slot_gene_sq_sum[gene];
                    host_gene_detected[gene] += slot_gene_detected[gene];
                }
            }
        }
    }

    finalize_gene_keep_mask(host_gene_sum, host_gene_sq_sum, host_gene_detected, kept_cells, gene_filter, &host_keep_genes);
    analysis.ok = true;
    analysis.device = -1;
    analysis.rows = matrix.rows;
    analysis.cols = matrix.cols;
    analysis.nnz = matrix.nnz;
    analysis.kept_cells = kept_cells;
    for (unsigned char keep : host_keep_genes) analysis.kept_genes += keep != 0u ? 1ul : 0ul;
    for (float value : host_gene_sum) analysis.gene_sum_checksum += (double) value;
    analysis.cell_total_counts.assign(host_cell_total_counts.begin(), host_cell_total_counts.end());
    analysis.cell_mito_counts.assign(host_cell_mito_counts.begin(), host_cell_mito_counts.end());
    analysis.cell_max_counts.assign(host_cell_max_counts.begin(), host_cell_max_counts.end());
    analysis.cell_detected_genes.assign(host_cell_detected_genes.begin(), host_cell_detected_genes.end());
    analysis.cell_keep.assign(host_keep_cells.begin(), host_keep_cells.end());
    analysis.gene_sum.assign(host_gene_sum.begin(), host_gene_sum.end());
    analysis.gene_sq_sum.assign(host_gene_sq_sum.begin(), host_gene_sq_sum.end());
    analysis.gene_detected_cells.assign(host_gene_detected.begin(), host_gene_detected.end());
    analysis.gene_keep.assign(host_keep_genes.begin(), host_keep_genes.end());
    analysis.gene_flags.assign(gene_flags.begin(), gene_flags.end());

done:
    csd::clear(&ctx);
    cs::clear(&storage);
    cs::clear(&matrix);
    return analysis;
}

preprocess_analysis_table analyze_dataset_preprocess(const std::string &path, const preprocess_config &config) {
    preprocess_analysis_table analysis;
    dataset_summary dataset = summarize_dataset_csh5(path);

    if (!dataset.ok) {
        analysis.path = path;
        analysis.issues = dataset.issues;
        push_issue(&analysis.issues, issue_severity::error, "preprocess", "cannot preprocess an unreadable dataset.csh5");
        return analysis;
    }
    if (dataset.preprocess.available) {
        analysis.path = path;
        analysis.matrix_format = dataset.matrix_format;
        push_issue(&analysis.issues,
                   issue_severity::error,
                   "preprocess",
                   "dataset.csh5 already contains persisted preprocess metadata; start from a raw dataset file instead");
        return analysis;
    }
    if (dataset.matrix_format.find("sliced") != std::string::npos) {
        if (config.use_all_devices) return analyze_sliced_dataset_preprocess_multigpu(path, config, dataset);
        return analyze_sliced_dataset_preprocess(path, config, dataset);
    }
    return analyze_blocked_dataset_preprocess(path, config, dataset);
}

preprocess_persist_summary persist_preprocess_analysis(const std::string &path,
                                                       const preprocess_analysis_table &analysis,
                                                       const preprocess_config &config) {
    preprocess_persist_summary result;
    dataset_summary dataset = summarize_dataset_csh5(path);
    cs::dataset_preprocess_view preprocess = {};
    std::vector<std::pair<std::string, std::string>> dataset_attributes;
    host_buffer<float> host_cell_total_counts;
    host_buffer<float> host_cell_mito_counts;
    host_buffer<float> host_cell_max_counts;
    host_buffer<unsigned int> host_cell_detected_genes;
    host_buffer<unsigned char> host_keep_cells;
    host_buffer<float> host_gene_sum;
    host_buffer<float> host_gene_sq_sum;
    host_buffer<float> host_gene_detected;
    host_buffer<unsigned char> host_keep_genes;
    host_buffer<unsigned char> gene_flags;
    unsigned int mito_feature_count = 0u;

    fill_preprocess_summary_from_analysis(&result.summary, analysis);
    if (!analysis.ok) {
        push_issue(&result.summary.issues, issue_severity::error, "preprocess", "cannot persist preprocess results from a failed analysis pass");
        result.summary.ok = false;
        return result;
    }
    if (!dataset.ok) {
        result.summary.issues.insert(result.summary.issues.end(), dataset.issues.begin(), dataset.issues.end());
        push_issue(&result.summary.issues, issue_severity::error, "preprocess", "cannot persist preprocess metadata into an unreadable dataset.csh5");
        result.summary.ok = false;
        return result;
    }
    if (dataset.preprocess.available) {
        push_issue(&result.summary.issues,
                   issue_severity::error,
                   "preprocess",
                   "dataset.csh5 already contains persisted preprocess metadata; start from a raw dataset file instead");
        result.summary.ok = false;
        return result;
    }

    host_cell_total_counts.assign_copy(analysis.cell_total_counts.data(), analysis.cell_total_counts.size());
    host_cell_mito_counts.assign_copy(analysis.cell_mito_counts.data(), analysis.cell_mito_counts.size());
    host_cell_max_counts.assign_copy(analysis.cell_max_counts.data(), analysis.cell_max_counts.size());
    host_cell_detected_genes.assign_copy(analysis.cell_detected_genes.data(), analysis.cell_detected_genes.size());
    host_keep_cells.assign_copy(analysis.cell_keep.data(), analysis.cell_keep.size());
    host_gene_sum.assign_copy(analysis.gene_sum.data(), analysis.gene_sum.size());
    host_gene_sq_sum.assign_copy(analysis.gene_sq_sum.data(), analysis.gene_sq_sum.size());
    host_gene_detected.assign_copy(analysis.gene_detected_cells.data(), analysis.gene_detected_cells.size());
    host_keep_genes.assign_copy(analysis.gene_keep.data(), analysis.gene_keep.size());
    gene_flags.assign_copy(analysis.gene_flags.data(), analysis.gene_flags.size());
    for (unsigned char flag : gene_flags) mito_feature_count += (flag & cpre::gene_flag_mito) != 0u;

    preprocess.assay = "scrna";
    preprocess.matrix_orientation = "observations_by_features";
    preprocess.matrix_state = "raw_counts";
    preprocess.pipeline_scope = "qc_filter_metrics_from_normalized_log1p";
    preprocess.raw_matrix_name = "X";
    preprocess.active_matrix_name = "X";
    preprocess.feature_namespace = "unknown";
    preprocess.mito_prefix = config.mito_prefix.c_str();
    preprocess.raw_counts_available = 1u;
    preprocess.processed_matrix_available = 0u;
    preprocess.normalized_log1p_metrics = 1u;
    preprocess.hvg_available = 0u;
    preprocess.mark_mito_from_feature_names = config.mark_mito_from_feature_names ? 1u : 0u;
    preprocess.rows = analysis.rows;
    preprocess.cols = (std::uint32_t) analysis.cols;
    preprocess.nnz = analysis.nnz;
    preprocess.partitions_processed = (std::uint32_t) analysis.partitions_processed;
    preprocess.mito_feature_count = mito_feature_count;
    preprocess.target_sum = config.target_sum;
    preprocess.min_counts = config.min_counts;
    preprocess.min_genes = config.min_genes;
    preprocess.max_mito_fraction = config.max_mito_fraction;
    preprocess.min_gene_sum = config.min_gene_sum;
    preprocess.min_detected_cells = config.min_detected_cells;
    preprocess.min_variance = config.min_variance;
    preprocess.kept_cells = analysis.kept_cells;
    preprocess.kept_genes = (std::uint32_t) analysis.kept_genes;
    preprocess.gene_sum_checksum = analysis.gene_sum_checksum;
    preprocess.cell_total_counts = host_cell_total_counts.empty() ? nullptr : host_cell_total_counts.data();
    preprocess.cell_mito_counts = host_cell_mito_counts.empty() ? nullptr : host_cell_mito_counts.data();
    preprocess.cell_max_counts = host_cell_max_counts.empty() ? nullptr : host_cell_max_counts.data();
    preprocess.cell_detected_genes = host_cell_detected_genes.empty() ? nullptr : host_cell_detected_genes.data();
    preprocess.cell_keep = host_keep_cells.empty() ? nullptr : host_keep_cells.data();
    preprocess.gene_sum = host_gene_sum.empty() ? nullptr : host_gene_sum.data();
    preprocess.gene_sq_sum = host_gene_sq_sum.empty() ? nullptr : host_gene_sq_sum.data();
    preprocess.gene_detected_cells = host_gene_detected.empty() ? nullptr : host_gene_detected.data();
    preprocess.gene_keep = host_keep_genes.empty() ? nullptr : host_keep_genes.data();
    preprocess.gene_flags = gene_flags.empty() ? nullptr : gene_flags.data();

    {
        const auto persist_begin = std::chrono::steady_clock::now();
        if (!rewrite_observation_annotations_with_preprocess(path,
                                                             host_cell_total_counts,
                                                             host_cell_mito_counts,
                                                             host_cell_max_counts,
                                                             host_cell_detected_genes,
                                                             host_keep_cells,
                                                             &result.summary.issues)
            || !rewrite_feature_metadata_with_preprocess(path,
                                                         host_gene_sum,
                                                         host_gene_sq_sum,
                                                         host_gene_detected,
                                                         host_keep_genes,
                                                         gene_flags,
                                                         &result.summary.issues)) {
            result.summary.ok = false;
            return result;
        }

        dataset_attributes.push_back({"preprocess.assay", "scrna"});
        dataset_attributes.push_back({"preprocess.matrix_orientation", "observations_by_features"});
        dataset_attributes.push_back({"preprocess.matrix_state", "raw_counts"});
        dataset_attributes.push_back({"preprocess.pipeline_scope", "qc_filter_metrics_from_normalized_log1p"});
        dataset_attributes.push_back({"preprocess.active_matrix_name", "X"});
        dataset_attributes.push_back({"preprocess.raw_counts_available", "1"});
        dataset_attributes.push_back({"preprocess.processed_matrix_available", "0"});
        dataset_attributes.push_back({"preprocess.normalized_log1p_metrics", "1"});
        dataset_attributes.push_back({"preprocess.hvg_available", "0"});
        dataset_attributes.push_back({"preprocess.mark_mito_from_feature_names", config.mark_mito_from_feature_names ? "1" : "0"});
        dataset_attributes.push_back({"preprocess.mito_prefix", config.mito_prefix});
        dataset_attributes.push_back({"preprocess.partitions_processed", std::to_string(analysis.partitions_processed)});
        dataset_attributes.push_back({"preprocess.mito_feature_count", std::to_string(mito_feature_count)});
        dataset_attributes.push_back({"preprocess.target_sum", std::to_string(config.target_sum)});
        dataset_attributes.push_back({"preprocess.min_counts", std::to_string(config.min_counts)});
        dataset_attributes.push_back({"preprocess.min_genes", std::to_string(config.min_genes)});
        dataset_attributes.push_back({"preprocess.max_mito_fraction", std::to_string(config.max_mito_fraction)});
        dataset_attributes.push_back({"preprocess.min_gene_sum", std::to_string(config.min_gene_sum)});
        dataset_attributes.push_back({"preprocess.min_detected_cells", std::to_string(config.min_detected_cells)});
        dataset_attributes.push_back({"preprocess.min_variance", std::to_string(config.min_variance)});
        dataset_attributes.push_back({"preprocess.kept_cells", std::to_string(analysis.kept_cells)});
        dataset_attributes.push_back({"preprocess.kept_genes", std::to_string(analysis.kept_genes)});
        dataset_attributes.push_back({"preprocess.gene_sum_checksum", std::to_string(analysis.gene_sum_checksum)});
        {
            const std::vector<std::pair<std::string, std::string>> merged_attrs =
                merge_dataset_attribute_entries(path, dataset_attributes, &result.summary.issues);
            if (!write_dataset_attribute_strings(path, merged_attrs, &result.summary.issues)) {
                result.summary.ok = false;
                return result;
            }
        }
        if (!cs::append_dataset_preprocess_h5(path.c_str(), &preprocess)) {
            push_issue(&result.summary.issues, issue_severity::error, "preprocess", "failed to append persisted preprocess metadata");
            result.summary.ok = false;
            return result;
        }

        if (config.finalize_after_preprocess) {
            if (dataset.matrix_format.find("blocked") == std::string::npos) {
                push_issue(&result.summary.issues,
                           issue_severity::error,
                           "preprocess",
                           "finalize_after_preprocess currently requires a blocked-ELL dataset");
                result.summary.ok = false;
                return result;
            }

            observation_metadata_table current_obs = load_observation_metadata_table(path);
            feature_metadata_table current_feature = load_feature_metadata_table(path);
            owned_annotation_bundle filtered_obs_bundle;
            owned_annotation_bundle filtered_feature_bundle;
            owned_embedded_metadata_bundle filtered_embedded_bundle;
            owned_user_attribute_bundle final_attr_bundle;
            cs::dataset_embedded_metadata_view filtered_embedded_view{};
            cs::dataset_annotation_view filtered_obs_view{};
            cs::dataset_feature_metadata_view filtered_feature_view{};
            cs::dataset_user_attribute_view final_attr_view{};
            std::vector<std::pair<std::string, std::string>> final_attr_updates;
            std::vector<std::pair<std::string, std::string>> final_attr_entries;
            std::uint64_t final_rows = analysis.rows;
            std::uint64_t final_cols = analysis.cols;
            std::uint64_t final_nnz = analysis.nnz;

            if (current_obs.available) {
                if (!filter_observation_metadata_table(current_obs, host_keep_cells, &filtered_obs_bundle, &result.summary.issues)) {
                    result.summary.ok = false;
                    return result;
                }
            } else if (!current_obs.error.empty()) {
                push_issue(&result.summary.issues, issue_severity::warning, "preprocess", current_obs.error);
            }
            if (current_feature.available) {
                if (!filter_feature_metadata_table(current_feature, host_keep_genes, &filtered_feature_bundle, &result.summary.issues)) {
                    result.summary.ok = false;
                    return result;
                }
            } else if (!current_feature.error.empty()) {
                push_issue(&result.summary.issues, issue_severity::warning, "preprocess", current_feature.error);
            }
            if (!dataset.embedded_metadata.empty()
                && !filter_embedded_metadata_tables(path, dataset, host_keep_cells, &filtered_embedded_bundle, &result.summary.issues)) {
                result.summary.ok = false;
                return result;
            }

            final_attr_updates.push_back({"preprocess.processed_matrix_available", "1"});
            final_attr_entries = merge_dataset_attribute_entries(path, final_attr_updates, &result.summary.issues);
            if (!build_user_attribute_bundle(final_attr_entries, &final_attr_bundle, &result.summary.issues)) {
                result.summary.ok = false;
                return result;
            }
            filtered_embedded_view = filtered_embedded_bundle.view();
            filtered_obs_view.extent = (std::uint64_t) analysis.kept_cells;
            filtered_obs_view.cols = (std::uint32_t) filtered_obs_bundle.views.size();
            filtered_obs_view.columns = filtered_obs_bundle.views.empty() ? nullptr : filtered_obs_bundle.views.data();
            filtered_feature_view.cols = (std::uint64_t) analysis.kept_genes;
            filtered_feature_view.annotation_count = (std::uint32_t) filtered_feature_bundle.views.size();
            filtered_feature_view.annotations = filtered_feature_bundle.views.empty() ? nullptr : filtered_feature_bundle.views.data();
            final_attr_view = final_attr_bundle.view((std::uint32_t) final_attr_entries.size());

            if (!cs::finalize_preprocessed_blocked_ell_dataset_h5(path.c_str(),
                                                                  host_keep_cells.empty() ? nullptr : host_keep_cells.data(),
                                                                  host_keep_genes.empty() ? nullptr : host_keep_genes.data(),
                                                                  filtered_embedded_bundle.table_views.empty() ? nullptr : &filtered_embedded_view,
                                                                  filtered_obs_bundle.views.empty() ? nullptr : &filtered_obs_view,
                                                                  filtered_feature_bundle.views.empty() ? nullptr : &filtered_feature_view,
                                                                  &final_attr_view,
                                                                  &preprocess,
                                                                  config.working_root.empty() ? nullptr : config.working_root.c_str(),
                                                                  &final_rows,
                                                                  &final_cols,
                                                                  &final_nnz)) {
                push_issue(&result.summary.issues, issue_severity::error, "preprocess", "failed to finalize compacted blocked-ELL dataset");
                result.summary.ok = false;
                return result;
            }
            result.summary.rows = (unsigned long) final_rows;
            result.summary.cols = (unsigned long) final_cols;
            result.summary.nnz = (unsigned long) final_nnz;
        }
        const auto persist_end = std::chrono::steady_clock::now();
        result.persist_ms = std::chrono::duration<double, std::milli>(persist_end - persist_begin).count();
    }

    if (config.finalize_after_preprocess) {
        const auto browse_begin = std::chrono::steady_clock::now();
        dataset_summary finalized_dataset = summarize_dataset_csh5(path);
        if (!finalized_dataset.ok) {
            result.summary.issues.insert(result.summary.issues.end(), finalized_dataset.issues.begin(), finalized_dataset.issues.end());
            push_issue(&result.summary.issues, issue_severity::error, "preprocess", "failed to summarize finalized dataset");
            result.summary.ok = false;
            return result;
        }
        if (!build_browse_cache_after_preprocess(path, finalized_dataset, config, &result.summary.issues)) {
            result.summary.ok = false;
            return result;
        }
        const auto browse_end = std::chrono::steady_clock::now();
        result.browse_ms = std::chrono::duration<double, std::milli>(browse_end - browse_begin).count();
    }

    result.summary.ok = true;
    return result;
}

preprocess_summary run_preprocess_pass(const std::string &path, const preprocess_config &config) {
    const preprocess_analysis_table analysis = analyze_dataset_preprocess(path, config);
    if (!analysis.ok) {
        preprocess_summary summary;
        fill_preprocess_summary_from_analysis(&summary, analysis);
        summary.ok = false;
        return summary;
    }
    return persist_preprocess_analysis(path, analysis, config).summary;
}
