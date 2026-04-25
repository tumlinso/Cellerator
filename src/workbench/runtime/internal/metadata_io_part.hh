#pragma once

inline bool append_embedded_metadata_tables(const ingest_plan &plan,
                                            std::vector<issue> *issues) {
    std::vector<owned_metadata_table *> owned;
    std::vector<cellshard::dataset_metadata_table_view> views;
    std::vector<std::uint32_t> dataset_indices;
    std::vector<std::uint64_t> global_row_begin;
    std::vector<std::uint64_t> global_row_end;
    cellshard::dataset_embedded_metadata_view metadata_view{};
    bool ok = true;

    owned.reserve(plan.datasets.size());
    views.reserve(plan.datasets.size());
    dataset_indices.reserve(plan.datasets.size());
    global_row_begin.reserve(plan.datasets.size());
    global_row_end.reserve(plan.datasets.size());

    for (std::size_t i = 0; i < plan.datasets.size(); ++i) {
        const planned_dataset &dataset = plan.datasets[i];
        const source_entry &source = plan.sources[dataset.source_index];
        if (source.format != cseries::source_h5ad && source.metadata_path.empty()) continue;

        owned_metadata_table *owned_table = new owned_metadata_table();
        if (!load_source_metadata_table(source, owned_table, issues)) {
            delete owned_table;
            continue;
        }
        if (owned_table->table.num_rows != dataset.rows) {
            push_issue(issues,
                       issue_severity::error,
                       "metadata",
                       "metadata row count does not match barcodes for " + dataset.dataset_id);
            delete owned_table;
            ok = false;
            continue;
        }

        dataset_indices.push_back((std::uint32_t) i);
        global_row_begin.push_back((std::uint64_t) dataset.global_row_begin);
        global_row_end.push_back((std::uint64_t) dataset.global_row_end);
        views.push_back(cellshard::dataset_metadata_table_view{
            owned_table->table.num_rows,
            owned_table->table.num_cols,
            as_text_view(&owned_table->table.column_names),
            as_text_view(&owned_table->table.field_values),
            owned_table->table.row_offsets
        });
        owned.push_back(owned_table);
    }

    if (!ok) {
        for (owned_metadata_table *table : owned) delete table;
        return false;
    }

    metadata_view.count = (std::uint32_t) views.size();
    metadata_view.dataset_indices = dataset_indices.empty() ? nullptr : dataset_indices.data();
    metadata_view.global_row_begin = global_row_begin.empty() ? nullptr : global_row_begin.data();
    metadata_view.global_row_end = global_row_end.empty() ? nullptr : global_row_end.data();
    metadata_view.tables = views.empty() ? nullptr : views.data();

    if (!cellshard::append_dataset_embedded_metadata_h5(plan.policy.output_path.c_str(), &metadata_view)) {
        push_issue(issues, issue_severity::error, "metadata", "failed to append embedded metadata to dataset.csh5");
        ok = false;
    }

    for (owned_metadata_table *table : owned) delete table;
    return ok;
}

inline bool append_observation_metadata_table(const ingest_plan &plan,
                                             std::vector<issue> *issues) {
    std::vector<loaded_observation_metadata> dataset_metadata(plan.datasets.size());
    std::vector<std::string> text_column_names;
    std::unordered_map<std::string, std::size_t> text_column_indices;
    std::vector<std::unique_ptr<owned_observation_metadata_column>> columns;
    std::vector<cs::dataset_observation_metadata_column_view> views;
    cs::dataset_observation_metadata_view metadata_view{};
    bool saw_any_metadata = false;

    if (plan.total_rows > (unsigned long) std::numeric_limits<std::uint32_t>::max()) {
        push_issue(issues, issue_severity::error, "metadata", "typed observation metadata currently requires <= 2^32-1 rows");
        return false;
    }

    for (std::size_t i = 0; i < plan.datasets.size(); ++i) {
        const planned_dataset &dataset = plan.datasets[i];
        const source_entry &source = plan.sources[dataset.source_index];
        if (source.format != cseries::source_h5ad && source.metadata_path.empty()) continue;

        auto owned = std::make_unique<owned_metadata_table>();
        if (!load_source_metadata_table(source, owned.get(), issues)) {
            continue;
        }
        if (owned->table.num_rows != dataset.rows) {
            push_issue(issues,
                       issue_severity::error,
                       "metadata",
                       "metadata row count does not match barcodes for " + dataset.dataset_id);
            return false;
        }

        loaded_observation_metadata loaded;
        loaded.column_names = make_unique_metadata_column_names(owned->table);
        loaded.table = std::move(owned);
        for (const std::string &name : loaded.column_names) {
            if (text_column_indices.find(name) != text_column_indices.end()) continue;
            text_column_indices.emplace(name, text_column_names.size());
            text_column_names.push_back(name);
        }
        dataset_metadata[i] = std::move(loaded);
        saw_any_metadata = true;
    }

    if (!saw_any_metadata) return true;

    for (loaded_observation_metadata &loaded : dataset_metadata) {
        loaded.global_text_to_local.assign(text_column_names.size(), -1);
        for (std::size_t local = 0; local < loaded.column_names.size(); ++local) {
            const auto it = text_column_indices.find(loaded.column_names[local]);
            if (it != text_column_indices.end()) loaded.global_text_to_local[it->second] = (int) local;
        }
    }

    columns.reserve(text_column_names.size());
    for (const std::string &name : text_column_names) {
        auto column = std::make_unique<owned_observation_metadata_column>();
        column->name = name;
        column->type = cs::dataset_observation_metadata_type_text;
        column->text_values = std::make_unique<owned_observation_text_column>();
        columns.push_back(std::move(column));
    }

    for (std::size_t dataset_index = 0; dataset_index < plan.datasets.size(); ++dataset_index) {
        const planned_dataset &dataset = plan.datasets[dataset_index];
        const loaded_observation_metadata &loaded = dataset_metadata[dataset_index];
        for (unsigned long row = 0; row < dataset.rows; ++row) {
            for (std::size_t global_col = 0; global_col < text_column_names.size(); ++global_col) {
                const int local_col = global_col < loaded.global_text_to_local.size()
                    ? loaded.global_text_to_local[global_col]
                    : -1;
                const char *value = (loaded.table != nullptr && local_col >= 0)
                    ? ccommon::field(&loaded.table->table, (unsigned int) row, (unsigned int) local_col)
                    : "";
                if (!ccommon::append(&columns[global_col]->text_values->values,
                                     value != nullptr ? value : "",
                                     std::strlen(value != nullptr ? value : ""))) {
                    push_issue(issues, issue_severity::error, "metadata", "failed to build observation metadata text column");
                    return false;
                }
            }
        }
    }

    views.reserve(columns.size());
    for (const std::unique_ptr<owned_observation_metadata_column> &column : columns) {
        views.push_back(column->view());
    }

    metadata_view.rows = plan.total_rows;
    metadata_view.cols = (std::uint32_t) views.size();
    metadata_view.columns = views.empty() ? nullptr : views.data();

    if (!cs::append_dataset_observation_metadata_h5(plan.policy.output_path.c_str(), &metadata_view)) {
        push_issue(issues, issue_severity::error, "metadata", "failed to append typed observation metadata to dataset.csh5");
        return false;
    }

    return true;
}

inline bool load_dataset_feature_ids(const std::string &path,
                                     std::vector<std::string> *feature_ids,
                                     std::vector<issue> *issues) {
    hid_t file = (hid_t) -1;
    hid_t provenance = (hid_t) -1;
    hid_t group = (hid_t) -1;
    hid_t offsets_dset = (hid_t) -1;
    hid_t data_dset = (hid_t) -1;
    hid_t offsets_space = (hid_t) -1;
    hid_t data_space = (hid_t) -1;
    hsize_t offsets_dims[1] = {0u};
    hsize_t data_dims[1] = {0u};
    std::vector<std::uint32_t> offsets;
    std::vector<char> data;
    bool ok = false;

    if (feature_ids == nullptr) return false;
    feature_ids->clear();
    file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        push_issue(issues, issue_severity::error, "metadata", "failed to reopen dataset.csh5 provenance for feature metadata");
        return false;
    }
    provenance = H5Gopen2(file, "/provenance", H5P_DEFAULT);
    if (provenance < 0) {
        push_issue(issues, issue_severity::error, "metadata", "failed to read dataset.csh5 feature ids");
        goto done;
    }
    group = H5Gopen2(provenance, "feature_ids", H5P_DEFAULT);
    if (group < 0) {
        push_issue(issues, issue_severity::error, "metadata", "failed to open dataset.csh5 feature_ids text column");
        goto done;
    }
    offsets_dset = H5Dopen2(group, "offsets", H5P_DEFAULT);
    data_dset = H5Dopen2(group, "data", H5P_DEFAULT);
    if (offsets_dset < 0 || data_dset < 0) {
        push_issue(issues, issue_severity::error, "metadata", "failed to open dataset.csh5 feature_ids payload");
        goto done;
    }
    offsets_space = H5Dget_space(offsets_dset);
    data_space = H5Dget_space(data_dset);
    if (offsets_space < 0 || data_space < 0
        || H5Sget_simple_extent_ndims(offsets_space) != 1
        || H5Sget_simple_extent_ndims(data_space) != 1
        || H5Sget_simple_extent_dims(offsets_space, offsets_dims, nullptr) != 1
        || H5Sget_simple_extent_dims(data_space, data_dims, nullptr) != 1) {
        push_issue(issues, issue_severity::error, "metadata", "failed to inspect dataset.csh5 feature_ids payload");
        goto done;
    }
    offsets.assign((std::size_t) offsets_dims[0], 0u);
    data.assign((std::size_t) data_dims[0], '\0');
    if ((!offsets.empty() && H5Dread(offsets_dset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, offsets.data()) < 0)
        || (!data.empty() && H5Dread(data_dset, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0)) {
        push_issue(issues, issue_severity::error, "metadata", "failed to read dataset.csh5 feature_ids payload");
        goto done;
    }
    feature_ids->reserve(offsets.size() > 0u ? offsets.size() - 1u : 0u);
    for (std::size_t i = 0; i + 1u < offsets.size(); ++i) {
        const std::uint32_t begin = offsets[i];
        const std::uint32_t end = offsets[i + 1u];
        if (end <= begin || end > data.size()) feature_ids->push_back(std::string());
        else feature_ids->emplace_back(data.data() + begin);
    }
    ok = true;

done:
    if (data_space >= 0) H5Sclose(data_space);
    if (offsets_space >= 0) H5Sclose(offsets_space);
    if (data_dset >= 0) H5Dclose(data_dset);
    if (offsets_dset >= 0) H5Dclose(offsets_dset);
    if (group >= 0) H5Gclose(group);
    if (provenance >= 0) H5Gclose(provenance);
    if (file >= 0) H5Fclose(file);
    return ok;
}

inline bool append_text_column_values(ccommon::text_column *out, const std::vector<std::string> &values) {
    if (out == nullptr) return false;
    ccommon::clear(out);
    ccommon::init(out);
    for (const std::string &value : values) {
        if (!ccommon::append(out, value.c_str(), value.size())) return false;
    }
    return true;
}

inline std::unique_ptr<owned_observation_metadata_column> materialize_annotation_column(const staged_annotation_column &staged) {
    auto out = std::make_unique<owned_observation_metadata_column>();
    out->name = staged.name;
    out->type = staged.type;
    if (staged.type == cs::dataset_observation_metadata_type_text) {
        out->text_values = std::make_unique<owned_observation_text_column>();
        if (!append_text_column_values(&out->text_values->values, staged.text_values)) return nullptr;
    } else if (staged.type == cs::dataset_observation_metadata_type_float32) {
        out->float32_values = staged.float32_values;
    } else if (staged.type == cs::dataset_observation_metadata_type_uint8) {
        out->uint8_values = staged.uint8_values;
    } else {
        return nullptr;
    }
    return out;
}

inline bool append_feature_metadata_table(const ingest_plan &plan,
                                          std::vector<issue> *issues) {
    std::vector<std::string> global_feature_ids;
    std::unordered_map<std::string, std::size_t> global_feature_index;
    std::vector<staged_annotation_column> staged_columns;
    std::unordered_map<std::string, std::size_t> column_indices;
    std::vector<std::unique_ptr<owned_observation_metadata_column>> owned_columns;
    std::vector<cs::dataset_observation_metadata_column_view> column_views;
    cs::dataset_feature_metadata_view metadata_view{};
    bool saw_any = false;

    if (!load_dataset_feature_ids(plan.policy.output_path, &global_feature_ids, issues)) return false;
    for (std::size_t i = 0; i < global_feature_ids.size(); ++i) {
        if (global_feature_index.find(global_feature_ids[i]) == global_feature_index.end()) {
            global_feature_index.emplace(global_feature_ids[i], i);
        }
    }

    for (const planned_dataset &dataset : plan.datasets) {
        const source_entry &source = plan.sources[dataset.source_index];
        std::vector<cellerator::ingest::h5ad::observation_column> feature_columns;
        ccommon::feature_table features;
        unsigned long feature_rows = 0ul;
        std::string error;

        if (source.format != cseries::source_h5ad) continue;
        ccommon::init(&features);
        if (!cellerator::ingest::h5ad::load_feature_columns(source.matrix_path.c_str(),
                                                            source.matrix_source.c_str(),
                                                            &feature_columns,
                                                            &feature_rows,
                                                            &error)) {
            push_issue(issues, issue_severity::warning, "metadata",
                       error.empty() ? ("failed to load feature annotations for " + source.dataset_id) : error);
            ccommon::clear(&features);
            continue;
        }
        if (!cellerator::ingest::h5ad::load_feature_table(source.matrix_path.c_str(),
                                                          source.matrix_source.c_str(),
                                                          &features,
                                                          &error)) {
            for (auto &column : feature_columns) cellerator::ingest::h5ad::clear(&column);
            push_issue(issues, issue_severity::warning, "metadata",
                       error.empty() ? ("failed to load feature ids for " + source.dataset_id) : error);
            ccommon::clear(&features);
            continue;
        }
        if ((unsigned long) features.ids.count != feature_rows) {
            for (auto &column : feature_columns) cellerator::ingest::h5ad::clear(&column);
            ccommon::clear(&features);
            push_issue(issues, issue_severity::warning, "metadata", "feature annotation rows do not match feature ids for " + source.dataset_id);
            continue;
        }
        saw_any = saw_any || !feature_columns.empty();
        for (const auto &source_column : feature_columns) {
            const auto existing = column_indices.find(source_column.name);
            if (existing == column_indices.end()) {
                staged_annotation_column staged;
                staged.name = source_column.name;
                staged.type = source_column.type;
                if (staged.type == cs::dataset_observation_metadata_type_text) {
                    staged.text_values.assign(global_feature_ids.size(), std::string());
                } else if (staged.type == cs::dataset_observation_metadata_type_float32) {
                    staged.float32_values.assign(global_feature_ids.size(), std::numeric_limits<float>::quiet_NaN());
                } else if (staged.type == cs::dataset_observation_metadata_type_uint8) {
                    staged.uint8_values.assign(global_feature_ids.size(), 0u);
                } else {
                    continue;
                }
                column_indices.emplace(staged.name, staged_columns.size());
                staged_columns.push_back(std::move(staged));
            } else if (staged_columns[existing->second].type != source_column.type) {
                push_issue(issues, issue_severity::warning, "metadata",
                           "skipping conflicting feature annotation type for column " + source_column.name);
            }
        }

        for (unsigned int local_feature = 0; local_feature < features.ids.count; ++local_feature) {
            const char *feature_id = ccommon::get(&features.ids, local_feature);
            const auto global_it = global_feature_index.find(feature_id != nullptr ? feature_id : "");
            if (global_it == global_feature_index.end()) continue;
            const std::size_t global_index = global_it->second;
            for (const auto &source_column : feature_columns) {
                const auto staged_it = column_indices.find(source_column.name);
                if (staged_it == column_indices.end()) continue;
                staged_annotation_column &staged = staged_columns[staged_it->second];
                if (staged.type != source_column.type) continue;
                if (source_column.type == cs::dataset_observation_metadata_type_text) {
                    if (global_index < staged.text_values.size()) {
                        const char *value = local_feature < source_column.text_values.count
                            ? ccommon::get(&source_column.text_values, local_feature)
                            : "";
                        if (staged.text_values[global_index].empty()) staged.text_values[global_index] = value != nullptr ? value : "";
                    }
                } else if (source_column.type == cs::dataset_observation_metadata_type_float32) {
                    if (global_index < staged.float32_values.size() && local_feature < source_column.float32_values.size()
                        && !std::isfinite(staged.float32_values[global_index])) {
                        staged.float32_values[global_index] = source_column.float32_values[local_feature];
                    }
                } else if (source_column.type == cs::dataset_observation_metadata_type_uint8) {
                    if (global_index < staged.uint8_values.size() && local_feature < source_column.uint8_values.size()) {
                        staged.uint8_values[global_index] = source_column.uint8_values[local_feature];
                    }
                }
            }
        }

        for (auto &column : feature_columns) cellerator::ingest::h5ad::clear(&column);
        ccommon::clear(&features);
    }

    if (!saw_any || staged_columns.empty()) return true;

    owned_columns.reserve(staged_columns.size());
    for (const staged_annotation_column &staged : staged_columns) {
        std::unique_ptr<owned_observation_metadata_column> owned = materialize_annotation_column(staged);
        if (!owned) {
            push_issue(issues, issue_severity::error, "metadata", "failed to materialize feature annotation column " + staged.name);
            return false;
        }
        owned_columns.push_back(std::move(owned));
    }

    column_views.reserve(owned_columns.size());
    for (const auto &column : owned_columns) column_views.push_back(column->view());
    metadata_view.cols = (std::uint64_t) global_feature_ids.size();
    metadata_view.annotation_count = (std::uint32_t) column_views.size();
    metadata_view.annotations = column_views.empty() ? nullptr : column_views.data();
    if (!cs::append_dataset_feature_metadata_h5(plan.policy.output_path.c_str(), &metadata_view)) {
        push_issue(issues, issue_severity::error, "metadata", "failed to append feature metadata to dataset.csh5");
        return false;
    }
    return true;
}

inline bool remove_optional_group(const std::string &path,
                                  const char *group_path,
                                  std::vector<issue> *issues) {
    hid_t file = (hid_t) -1;
    bool ok = true;
    if (group_path == nullptr) return false;
    file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) {
        push_issue(issues, issue_severity::error, "preprocess", "failed to open dataset.csh5 for metadata rewrite");
        return false;
    }
    if (H5Lexists(file, group_path, H5P_DEFAULT) > 0 && H5Ldelete(file, group_path, H5P_DEFAULT) < 0) {
        push_issue(issues, issue_severity::error, "preprocess", std::string("failed to replace metadata group ") + group_path);
        ok = false;
    }
    H5Fclose(file);
    return ok;
}

inline bool rewrite_observation_annotations_with_preprocess(const std::string &path,
                                                            const host_buffer<float> &cell_total_counts,
                                                            const host_buffer<float> &cell_mito_counts,
                                                            const host_buffer<float> &cell_max_counts,
                                                            const host_buffer<unsigned int> &cell_detected_genes,
                                                            const host_buffer<unsigned char> &cell_keep,
                                                            std::vector<issue> *issues) {
    observation_metadata_table existing = load_observation_metadata_table(path);
    std::vector<std::unique_ptr<owned_observation_metadata_column>> owned_columns;
    std::vector<cs::dataset_observation_metadata_column_view> views;
    cs::dataset_annotation_view metadata_view{};
    const std::size_t rows = cell_total_counts.size();

    auto append_float_column = [&](const std::string &name, const float *values) {
        auto column = std::make_unique<owned_observation_metadata_column>();
        column->name = name;
        column->type = cs::dataset_observation_metadata_type_float32;
        column->float32_values.assign(values, values + rows);
        owned_columns.push_back(std::move(column));
    };

    auto append_uint8_column = [&](const std::string &name, const unsigned char *values) {
        auto column = std::make_unique<owned_observation_metadata_column>();
        column->name = name;
        column->type = cs::dataset_observation_metadata_type_uint8;
        column->uint8_values.assign(values, values + rows);
        owned_columns.push_back(std::move(column));
    };

    auto append_detected_column = [&]() {
        auto column = std::make_unique<owned_observation_metadata_column>();
        column->name = "preprocess_detected_genes";
        column->type = cs::dataset_observation_metadata_type_float32;
        column->float32_values.resize(rows, 0.0f);
        for (std::size_t i = 0; i < rows; ++i) column->float32_values[i] = (float) cell_detected_genes[i];
        owned_columns.push_back(std::move(column));
    };

    if (existing.available) {
        for (const observation_metadata_column &source : existing.columns) {
            staged_annotation_column staged;
            staged.name = source.name;
            staged.type = source.type;
            staged.text_values = source.text_values;
            staged.float32_values = source.float32_values;
            staged.uint8_values = source.uint8_values;
            std::unique_ptr<owned_observation_metadata_column> owned = materialize_annotation_column(staged);
            if (!owned) {
                push_issue(issues, issue_severity::error, "preprocess", "failed to preserve existing observation annotation " + source.name);
                return false;
            }
            owned_columns.push_back(std::move(owned));
        }
    } else if (!existing.error.empty() && existing.error.find("missing") == std::string::npos) {
        push_issue(issues, issue_severity::warning, "preprocess", existing.error);
    }

    append_float_column("preprocess_total_counts", cell_total_counts.data());
    append_float_column("preprocess_mito_counts", cell_mito_counts.data());
    append_float_column("preprocess_max_counts", cell_max_counts.data());
    append_detected_column();
    append_uint8_column("preprocess_keep", cell_keep.data());

    if (!remove_optional_group(path, "/observation_metadata", issues)) return false;
    views.reserve(owned_columns.size());
    for (const auto &column : owned_columns) views.push_back(column->view());
    metadata_view.extent = (std::uint64_t) rows;
    metadata_view.cols = (std::uint32_t) views.size();
    metadata_view.columns = views.empty() ? nullptr : views.data();
    if (!cs::append_dataset_observation_annotations_h5(path.c_str(), &metadata_view)) {
        push_issue(issues, issue_severity::error, "preprocess", "failed to rewrite observation annotations with preprocess columns");
        return false;
    }
    return true;
}

inline bool build_filtered_observation_metadata_bundle_with_preprocess(const observation_metadata_table &source,
                                                                       const host_buffer<float> &cell_total_counts,
                                                                       const host_buffer<float> &cell_mito_counts,
                                                                       const host_buffer<float> &cell_max_counts,
                                                                       const host_buffer<unsigned int> &cell_detected_genes,
                                                                       const host_buffer<unsigned char> &cell_keep,
                                                                       owned_annotation_bundle *bundle,
                                                                       std::vector<issue> *issues) {
    const std::size_t rows = cell_keep.size();
    if (bundle == nullptr) return false;

    for (const observation_metadata_column &column : source.columns) {
        auto owned = std::make_unique<owned_observation_metadata_column>();
        owned->name = column.name;
        owned->type = column.type;
        if (column.type == cs::dataset_observation_metadata_type_text) {
            owned->text_values = std::make_unique<owned_observation_text_column>();
            for (std::size_t row = 0; row < rows && row < column.text_values.size(); ++row) {
                if (cell_keep[row] == 0u) continue;
                const std::string &value = column.text_values[row];
                if (!ccommon::append(&owned->text_values->values, value.c_str(), value.size())) {
                    push_issue(issues, issue_severity::error, "preprocess", "failed to filter observation metadata text column");
                    return false;
                }
            }
        } else if (column.type == cs::dataset_observation_metadata_type_float32) {
            owned->float32_values.reserve(rows);
            for (std::size_t row = 0; row < rows && row < column.float32_values.size(); ++row) {
                if (cell_keep[row] != 0u) owned->float32_values.push_back(column.float32_values[row]);
            }
        } else if (column.type == cs::dataset_observation_metadata_type_uint8) {
            owned->uint8_values.reserve(rows);
            for (std::size_t row = 0; row < rows && row < column.uint8_values.size(); ++row) {
                if (cell_keep[row] != 0u) owned->uint8_values.push_back(column.uint8_values[row]);
            }
        } else {
            push_issue(issues, issue_severity::error, "preprocess", "unknown observation metadata column type");
            return false;
        }
        bundle->columns.push_back(std::move(owned));
    }

    auto append_float_column = [&](const std::string &name, const host_buffer<float> &values) {
        auto column = std::make_unique<owned_observation_metadata_column>();
        column->name = name;
        column->type = cs::dataset_observation_metadata_type_float32;
        column->float32_values.reserve(rows);
        for (std::size_t row = 0; row < rows && row < values.size(); ++row) {
            if (cell_keep[row] != 0u) column->float32_values.push_back(values[row]);
        }
        bundle->columns.push_back(std::move(column));
    };
    auto append_uint8_column = [&](const std::string &name, const host_buffer<unsigned char> &values) {
        auto column = std::make_unique<owned_observation_metadata_column>();
        column->name = name;
        column->type = cs::dataset_observation_metadata_type_uint8;
        column->uint8_values.reserve(rows);
        for (std::size_t row = 0; row < rows && row < values.size(); ++row) {
            if (cell_keep[row] != 0u) column->uint8_values.push_back(values[row]);
        }
        bundle->columns.push_back(std::move(column));
    };
    {
        auto column = std::make_unique<owned_observation_metadata_column>();
        column->name = "preprocess_detected_genes";
        column->type = cs::dataset_observation_metadata_type_float32;
        column->float32_values.reserve(rows);
        for (std::size_t row = 0; row < rows && row < cell_detected_genes.size(); ++row) {
            if (cell_keep[row] != 0u) column->float32_values.push_back((float) cell_detected_genes[row]);
        }
        bundle->columns.push_back(std::move(column));
    }
    append_float_column("preprocess_total_counts", cell_total_counts);
    append_float_column("preprocess_mito_counts", cell_mito_counts);
    append_float_column("preprocess_max_counts", cell_max_counts);
    append_uint8_column("preprocess_keep", cell_keep);

    bundle->views.reserve(bundle->columns.size());
    for (const auto &column : bundle->columns) bundle->views.push_back(column->view());
    return true;
}

inline bool rewrite_feature_metadata_with_preprocess(const std::string &path,
                                                     const host_buffer<float> &gene_sum,
                                                     const host_buffer<float> &gene_sq_sum,
                                                     const host_buffer<float> &gene_detected,
                                                     const host_buffer<unsigned char> &gene_keep,
                                                     const host_buffer<unsigned char> &gene_flags,
                                                     std::vector<issue> *issues) {
    feature_metadata_table existing = load_feature_metadata_table(path);
    std::vector<std::unique_ptr<owned_observation_metadata_column>> owned_columns;
    std::vector<cs::dataset_observation_metadata_column_view> views;
    cs::dataset_feature_metadata_view metadata_view{};
    const std::size_t cols = gene_sum.size();

    auto append_float_column = [&](const std::string &name, const float *values) {
        auto column = std::make_unique<owned_observation_metadata_column>();
        column->name = name;
        column->type = cs::dataset_observation_metadata_type_float32;
        column->float32_values.assign(values, values + cols);
        owned_columns.push_back(std::move(column));
    };

    auto append_uint8_column = [&](const std::string &name, const unsigned char *values) {
        auto column = std::make_unique<owned_observation_metadata_column>();
        column->name = name;
        column->type = cs::dataset_observation_metadata_type_uint8;
        column->uint8_values.assign(values, values + cols);
        owned_columns.push_back(std::move(column));
    };

    if (existing.available) {
        for (const feature_metadata_column &source : existing.columns) {
            staged_annotation_column staged;
            staged.name = source.name;
            staged.type = source.type;
            staged.text_values = source.text_values;
            staged.float32_values = source.float32_values;
            staged.uint8_values = source.uint8_values;
            std::unique_ptr<owned_observation_metadata_column> owned = materialize_annotation_column(staged);
            if (!owned) {
                push_issue(issues, issue_severity::error, "preprocess", "failed to preserve existing feature annotation " + source.name);
                return false;
            }
            owned_columns.push_back(std::move(owned));
        }
    } else if (!existing.error.empty() && existing.error.find("missing") == std::string::npos) {
        push_issue(issues, issue_severity::warning, "preprocess", existing.error);
    }

    append_float_column("preprocess_sum", gene_sum.data());
    append_float_column("preprocess_sq_sum", gene_sq_sum.data());
    append_float_column("preprocess_detected_cells", gene_detected.data());
    append_uint8_column("preprocess_keep", gene_keep.data());
    append_uint8_column("preprocess_flags", gene_flags.data());

    if (!remove_optional_group(path, "/feature_metadata", issues)) return false;
    views.reserve(owned_columns.size());
    for (const auto &column : owned_columns) views.push_back(column->view());
    metadata_view.cols = (std::uint64_t) cols;
    metadata_view.annotation_count = (std::uint32_t) views.size();
    metadata_view.annotations = views.empty() ? nullptr : views.data();
    if (!cs::append_dataset_feature_metadata_h5(path.c_str(), &metadata_view)) {
        push_issue(issues, issue_severity::error, "preprocess", "failed to rewrite feature metadata with preprocess columns");
        return false;
    }
    return true;
}

inline bool build_filtered_feature_metadata_bundle_with_preprocess(const feature_metadata_table &source,
                                                                  const host_buffer<float> &gene_sum,
                                                                  const host_buffer<float> &gene_sq_sum,
                                                                  const host_buffer<float> &gene_detected,
                                                                  const host_buffer<unsigned char> &gene_keep,
                                                                  const host_buffer<unsigned char> &gene_flags,
                                                                  owned_annotation_bundle *bundle,
                                                                  std::vector<issue> *issues) {
    const std::size_t cols = gene_keep.size();
    if (bundle == nullptr) return false;

    for (const feature_metadata_column &column : source.columns) {
        auto owned = std::make_unique<owned_observation_metadata_column>();
        owned->name = column.name;
        owned->type = column.type;
        if (column.type == cs::dataset_observation_metadata_type_text) {
            owned->text_values = std::make_unique<owned_observation_text_column>();
            for (std::size_t col = 0; col < cols && col < column.text_values.size(); ++col) {
                if (gene_keep[col] == 0u) continue;
                const std::string &value = column.text_values[col];
                if (!ccommon::append(&owned->text_values->values, value.c_str(), value.size())) {
                    push_issue(issues, issue_severity::error, "preprocess", "failed to filter feature metadata text column");
                    return false;
                }
            }
        } else if (column.type == cs::dataset_observation_metadata_type_float32) {
            owned->float32_values.reserve(cols);
            for (std::size_t col = 0; col < cols && col < column.float32_values.size(); ++col) {
                if (gene_keep[col] != 0u) owned->float32_values.push_back(column.float32_values[col]);
            }
        } else if (column.type == cs::dataset_observation_metadata_type_uint8) {
            owned->uint8_values.reserve(cols);
            for (std::size_t col = 0; col < cols && col < column.uint8_values.size(); ++col) {
                if (gene_keep[col] != 0u) owned->uint8_values.push_back(column.uint8_values[col]);
            }
        } else {
            push_issue(issues, issue_severity::error, "preprocess", "unknown feature metadata column type");
            return false;
        }
        bundle->columns.push_back(std::move(owned));
    }

    auto append_float_column = [&](const std::string &name, const host_buffer<float> &values) {
        auto column = std::make_unique<owned_observation_metadata_column>();
        column->name = name;
        column->type = cs::dataset_observation_metadata_type_float32;
        column->float32_values.reserve(cols);
        for (std::size_t col = 0; col < cols && col < values.size(); ++col) {
            if (gene_keep[col] != 0u) column->float32_values.push_back(values[col]);
        }
        bundle->columns.push_back(std::move(column));
    };
    auto append_uint8_column = [&](const std::string &name, const host_buffer<unsigned char> &values) {
        auto column = std::make_unique<owned_observation_metadata_column>();
        column->name = name;
        column->type = cs::dataset_observation_metadata_type_uint8;
        column->uint8_values.reserve(cols);
        for (std::size_t col = 0; col < cols && col < values.size(); ++col) {
            if (gene_keep[col] != 0u) column->uint8_values.push_back(values[col]);
        }
        bundle->columns.push_back(std::move(column));
    };

    append_float_column("preprocess_sum", gene_sum);
    append_float_column("preprocess_sq_sum", gene_sq_sum);
    append_float_column("preprocess_detected_cells", gene_detected);
    append_uint8_column("preprocess_keep", gene_keep);
    append_uint8_column("preprocess_flags", gene_flags);

    bundle->views.reserve(bundle->columns.size());
    for (const auto &column : bundle->columns) bundle->views.push_back(column->view());
    return true;
}

inline std::vector<std::pair<std::string, std::string>> merge_dataset_attribute_entries(
    const std::string &path,
    const std::vector<std::pair<std::string, std::string>> &updates,
    std::vector<issue> *issues) {
    dataset_attribute_table existing = load_dataset_attribute_table(path);
    std::vector<std::pair<std::string, std::string>> merged;
    std::unordered_map<std::string, std::size_t> positions;

    if (existing.available) {
        merged.reserve(existing.entries.size() + updates.size());
        for (const dataset_attribute_entry &entry : existing.entries) {
            positions.emplace(entry.key, merged.size());
            merged.push_back({entry.key, entry.value});
        }
    } else if (!existing.error.empty() && existing.error.find("missing") == std::string::npos) {
        push_issue(issues, issue_severity::warning, "preprocess", existing.error);
    }

    for (const auto &entry : updates) {
        const auto hit = positions.find(entry.first);
        if (hit == positions.end()) {
            positions.emplace(entry.first, merged.size());
            merged.push_back(entry);
        } else {
            merged[hit->second].second = entry.second;
        }
    }
    return merged;
}

inline bool build_user_attribute_bundle(const std::vector<std::pair<std::string, std::string>> &entries,
                                        owned_user_attribute_bundle *bundle,
                                        std::vector<issue> *issues) {
    if (bundle == nullptr) return false;
    for (const auto &entry : entries) {
        if (!ccommon::append(&bundle->keys, entry.first.c_str(), entry.first.size())
            || !ccommon::append(&bundle->values, entry.second.c_str(), entry.second.size())) {
            push_issue(issues, issue_severity::error, "preprocess", "failed to build merged dataset attribute payload");
            return false;
        }
    }
    return true;
}

inline bool filter_observation_metadata_table(const observation_metadata_table &source,
                                              const host_buffer<unsigned char> &keep_rows,
                                              owned_annotation_bundle *bundle,
                                              std::vector<issue> *issues) {
    if (bundle == nullptr) return false;
    for (const observation_metadata_column &column : source.columns) {
        auto owned = std::make_unique<owned_observation_metadata_column>();
        owned->name = column.name;
        owned->type = column.type;
        if (column.type == cs::dataset_observation_metadata_type_text) {
            owned->text_values = std::make_unique<owned_observation_text_column>();
            for (std::size_t row = 0; row < keep_rows.size() && row < column.text_values.size(); ++row) {
                if (keep_rows[row] == 0u) continue;
                const std::string &value = column.text_values[row];
                if (!ccommon::append(&owned->text_values->values, value.c_str(), value.size())) {
                    push_issue(issues, issue_severity::error, "preprocess", "failed to filter observation metadata text column");
                    return false;
                }
            }
        } else if (column.type == cs::dataset_observation_metadata_type_float32) {
            owned->float32_values.reserve(keep_rows.size());
            for (std::size_t row = 0; row < keep_rows.size() && row < column.float32_values.size(); ++row) {
                if (keep_rows[row] != 0u) owned->float32_values.push_back(column.float32_values[row]);
            }
        } else if (column.type == cs::dataset_observation_metadata_type_uint8) {
            owned->uint8_values.reserve(keep_rows.size());
            for (std::size_t row = 0; row < keep_rows.size() && row < column.uint8_values.size(); ++row) {
                if (keep_rows[row] != 0u) owned->uint8_values.push_back(column.uint8_values[row]);
            }
        } else {
            push_issue(issues, issue_severity::error, "preprocess", "unknown observation metadata column type");
            return false;
        }
        bundle->columns.push_back(std::move(owned));
    }
    bundle->views.reserve(bundle->columns.size());
    for (const auto &column : bundle->columns) bundle->views.push_back(column->view());
    return true;
}

inline bool filter_feature_metadata_table(const feature_metadata_table &source,
                                          const host_buffer<unsigned char> &keep_cols,
                                          owned_annotation_bundle *bundle,
                                          std::vector<issue> *issues) {
    if (bundle == nullptr) return false;
    for (const feature_metadata_column &column : source.columns) {
        auto owned = std::make_unique<owned_observation_metadata_column>();
        owned->name = column.name;
        owned->type = column.type;
        if (column.type == cs::dataset_observation_metadata_type_text) {
            owned->text_values = std::make_unique<owned_observation_text_column>();
            for (std::size_t col = 0; col < keep_cols.size() && col < column.text_values.size(); ++col) {
                if (keep_cols[col] == 0u) continue;
                const std::string &value = column.text_values[col];
                if (!ccommon::append(&owned->text_values->values, value.c_str(), value.size())) {
                    push_issue(issues, issue_severity::error, "preprocess", "failed to filter feature metadata text column");
                    return false;
                }
            }
        } else if (column.type == cs::dataset_observation_metadata_type_float32) {
            owned->float32_values.reserve(keep_cols.size());
            for (std::size_t col = 0; col < keep_cols.size() && col < column.float32_values.size(); ++col) {
                if (keep_cols[col] != 0u) owned->float32_values.push_back(column.float32_values[col]);
            }
        } else if (column.type == cs::dataset_observation_metadata_type_uint8) {
            owned->uint8_values.reserve(keep_cols.size());
            for (std::size_t col = 0; col < keep_cols.size() && col < column.uint8_values.size(); ++col) {
                if (keep_cols[col] != 0u) owned->uint8_values.push_back(column.uint8_values[col]);
            }
        } else {
            push_issue(issues, issue_severity::error, "preprocess", "unknown feature metadata column type");
            return false;
        }
        bundle->columns.push_back(std::move(owned));
    }
    bundle->views.reserve(bundle->columns.size());
    for (const auto &column : bundle->columns) bundle->views.push_back(column->view());
    return true;
}

inline bool filter_embedded_metadata_tables(const std::string &path,
                                            const dataset_summary &dataset,
                                            const host_buffer<unsigned char> &keep_rows,
                                            owned_embedded_metadata_bundle *bundle,
                                            std::vector<issue> *issues) {
    std::uint64_t row_cursor = 0u;
    if (bundle == nullptr) return false;
    bundle->tables.reserve(dataset.embedded_metadata.size());
    bundle->table_views.reserve(dataset.embedded_metadata.size());
    bundle->dataset_indices.reserve(dataset.embedded_metadata.size());
    bundle->global_row_begin.reserve(dataset.embedded_metadata.size());
    bundle->global_row_end.reserve(dataset.embedded_metadata.size());

    for (std::size_t table_index = 0; table_index < dataset.embedded_metadata.size(); ++table_index) {
        embedded_metadata_table table = load_embedded_metadata_table(path, table_index);
        auto owned = std::make_unique<owned_metadata_table>();
        std::vector<char *> header_fields;
        if (!table.available) {
            if (!table.error.empty()) push_issue(issues, issue_severity::warning, "preprocess", table.error);
            return false;
        }
        header_fields.reserve(table.column_names.size());
        for (std::string &name : table.column_names) header_fields.push_back(name.empty() ? const_cast<char *>("") : name.data());
        if (!header_fields.empty() && !ccommon::append_header(&owned->table, header_fields.data(), (unsigned int) header_fields.size())) {
            push_issue(issues, issue_severity::error, "preprocess", "failed to initialize filtered embedded metadata header");
            return false;
        }
        for (std::uint32_t row = 0; row < table.rows; ++row) {
            const std::uint64_t global_row = table.row_begin + row;
            std::vector<char *> row_fields;
            if (global_row >= keep_rows.size() || keep_rows[(std::size_t) global_row] == 0u) continue;
            row_fields.reserve(table.cols);
            for (std::uint32_t col = 0; col < table.cols; ++col) {
                const std::size_t field_index = (std::size_t) table.row_offsets[row] + col;
                const std::string &value = field_index < table.field_values.size() ? table.field_values[field_index] : std::string();
                row_fields.push_back(value.empty() ? const_cast<char *>("") : const_cast<char *>(value.c_str()));
            }
            if (!row_fields.empty() && !ccommon::append_row(&owned->table, row_fields.data(), (unsigned int) row_fields.size())) {
                push_issue(issues, issue_severity::error, "preprocess", "failed to append filtered embedded metadata row");
                return false;
            }
        }
        bundle->dataset_indices.push_back(table.dataset_index);
        bundle->global_row_begin.push_back(row_cursor);
        row_cursor += owned->table.num_rows;
        bundle->global_row_end.push_back(row_cursor);
        bundle->table_views.push_back(cs::dataset_metadata_table_view{
            owned->table.num_rows,
            owned->table.num_cols,
            as_text_view(&owned->table.column_names),
            as_text_view(&owned->table.field_values),
            owned->table.row_offsets
        });
        bundle->tables.push_back(std::move(owned));
    }
    return true;
}

inline bool write_dataset_attribute_strings(const std::string &path,
                                            const std::vector<std::pair<std::string, std::string>> &entries,
                                            std::vector<issue> *issues) {
    ccommon::text_column keys;
    ccommon::text_column values;
    cs::dataset_user_attribute_view attrs{};
    bool ok = false;

    ccommon::init(&keys);
    ccommon::init(&values);
    if (!remove_optional_group(path, "/dataset_attributes", issues)) goto done;
    for (const auto &entry : entries) {
        if (!ccommon::append(&keys, entry.first.c_str(), entry.first.size())
            || !ccommon::append(&values, entry.second.c_str(), entry.second.size())) {
            push_issue(issues, issue_severity::error, "preprocess", "failed to build dataset attribute payload");
            goto done;
        }
    }
    attrs.count = (std::uint32_t) entries.size();
    attrs.keys = as_text_view(&keys);
    attrs.values = as_text_view(&values);
    if (!cs::append_dataset_user_attributes_h5(path.c_str(), &attrs)) {
        push_issue(issues, issue_severity::error, "preprocess", "failed to append dataset attributes");
        goto done;
    }
    ok = true;

done:
    ccommon::clear(&values);
    ccommon::clear(&keys);
    return ok;
}
