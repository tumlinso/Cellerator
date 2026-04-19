#pragma once

struct normalized_day_value {
    std::string label;
    float numeric = std::numeric_limits<float>::quiet_NaN();
    std::uint8_t is_postnatal = 0u;
    bool has_numeric = false;
};

struct loaded_observation_metadata {
    std::unique_ptr<owned_metadata_table> table;
    std::vector<std::string> column_names;
    std::vector<int> global_text_to_local;
    int day_column = -1;
};

inline bool load_source_metadata_table(const source_entry &source,
                                       owned_metadata_table *owned,
                                       std::vector<issue> *issues) {
    std::string error;
    if (owned == nullptr) return false;
    if (source.format == cseries::source_h5ad) {
        if (!cellerator::ingest::h5ad::load_metadata_table(source.matrix_path.c_str(), &owned->table, &error)) {
            push_issue(issues,
                       issue_severity::warning,
                       "metadata",
                       error.empty() ? ("failed to load h5ad metadata for " + source.dataset_id) : error);
            return false;
        }
        return true;
    }
    if (source.metadata_path.empty()) return false;
    if (!ccommon::load_tsv(source.metadata_path.c_str(), &owned->table, 1)) {
        push_issue(issues, issue_severity::warning, "metadata", "failed to load metadata table for " + source.dataset_id);
        return false;
    }
    return true;
}

inline std::string trim_copy(std::string value) {
    std::size_t begin = 0u;
    std::size_t end = value.size();
    while (begin < end && std::isspace((unsigned char) value[begin])) ++begin;
    while (end > begin && std::isspace((unsigned char) value[end - 1u])) --end;
    return value.substr(begin, end - begin);
}

inline std::string lower_copy(std::string value) {
    for (char &ch : value) ch = (char) std::tolower((unsigned char) ch);
    return value;
}

inline std::string sanitize_metadata_column_name(const char *raw_name,
                                                 unsigned int column_index) {
    std::string name = trim_copy(raw_name != nullptr ? raw_name : "");
    if (!name.empty()) return name;
    if (column_index == 0u) return "row_id";
    return "column_" + std::to_string(column_index);
}

inline std::vector<std::string> make_unique_metadata_column_names(const ccommon::metadata_table &table) {
    std::unordered_map<std::string, unsigned int> counts;
    std::vector<std::string> names;
    names.reserve(table.num_cols);
    for (unsigned int col = 0; col < table.num_cols; ++col) {
        const std::string base = sanitize_metadata_column_name(ccommon::column_name(&table, col), col);
        const unsigned int seen = counts[base]++;
        if (seen == 0u) names.push_back(base);
        else names.push_back(base + "_" + std::to_string(seen + 1u));
    }
    return names;
}

inline int find_day_column_index(const std::vector<std::string> &column_names) {
    int stage_index = -1;
    for (std::size_t i = 0; i < column_names.size(); ++i) {
        const std::string name = lower_copy(column_names[i]);
        if (name == "day" || name == "embryonic_day_label") return (int) i;
        if (stage_index < 0 && name == "stage") stage_index = (int) i;
    }
    return stage_index;
}

inline normalized_day_value normalize_day_value(const char *raw_value) {
    normalized_day_value out;
    const std::string value = trim_copy(raw_value != nullptr ? raw_value : "");
    if (value.empty()) return out;

    out.label = value;
    if (value == "P0" || value == "p0") {
        out.is_postnatal = 1u;
        return out;
    }

    if (value.size() > 1u && (value[0] == 'E' || value[0] == 'e')) {
        char *end = nullptr;
        const float parsed = std::strtof(value.c_str() + 1u, &end);
        if (end != value.c_str() + 1u && end != nullptr && *end == '\0' && std::isfinite(parsed)) {
            out.numeric = parsed;
            out.has_numeric = true;
        }
    }
    return out;
}

struct browse_cache_owned {
    host_buffer<std::uint32_t> selected_feature_indices;
    host_buffer<float> gene_sum;
    host_buffer<float> gene_detected;
    host_buffer<float> gene_sq_sum;
    host_buffer<float> dataset_feature_mean;
    host_buffer<float> shard_feature_mean;
    host_buffer<std::uint32_t> partition_sample_row_offsets;
    host_buffer<std::uint64_t> partition_sample_global_rows;
    host_buffer<float> partition_sample_values;
};

struct gene_metric_partial {
    host_buffer<float> gene_sum;
    host_buffer<float> gene_detected;
    host_buffer<float> gene_sq_sum;
    float active_rows = 0.0f;
    bool ok = false;
};

struct selected_feature_partial {
    host_buffer<float> dataset_feature_sum;
    host_buffer<float> shard_feature_sum;
    bool ok = false;
};
