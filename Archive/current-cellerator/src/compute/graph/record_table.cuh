#pragma once

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cellerator::compute::graph {

struct EmbryoRowSpan {
    std::int32_t embryo_id = -1;
    std::uint32_t row_begin = 0;
    std::uint32_t row_end = 0;
};

struct TrajectoryRecordTable {
    std::uint32_t rows = 0;
    std::int32_t latent_dim = 0;
    std::vector<std::uint64_t> cell_index;
    std::vector<std::int32_t> embryo_id;
    std::vector<float> developmental_time;
    std::vector<__half> latent;

    void validate() const {
        if (latent_dim <= 0) throw std::invalid_argument("TrajectoryRecordTable.latent_dim must be > 0");
        if (cell_index.size() != rows || embryo_id.size() != rows || developmental_time.size() != rows) {
            throw std::invalid_argument("TrajectoryRecordTable vectors must match rows");
        }
        if (latent.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(latent_dim)) {
            throw std::invalid_argument("TrajectoryRecordTable.latent size must equal rows * latent_dim");
        }
    }

    void reserve(std::uint32_t row_capacity, std::int32_t next_latent_dim) {
        if (next_latent_dim <= 0) throw std::invalid_argument("reserve latent_dim must be > 0");
        if (rows != 0 && latent_dim != next_latent_dim) {
            throw std::invalid_argument("cannot change latent_dim after rows have been inserted");
        }
        latent_dim = next_latent_dim;
        cell_index.reserve(row_capacity);
        embryo_id.reserve(row_capacity);
        developmental_time.reserve(row_capacity);
        latent.reserve(static_cast<std::size_t>(row_capacity) * static_cast<std::size_t>(latent_dim));
    }

    void append(std::uint64_t next_cell_index,
                std::int32_t next_embryo_id,
                float next_time,
                const std::vector<float> &latent_row) {
        if (latent_dim == 0) {
            if (latent_row.empty()) throw std::invalid_argument("latent_row must not be empty");
            latent_dim = static_cast<std::int32_t>(latent_row.size());
        }
        if (static_cast<std::int32_t>(latent_row.size()) != latent_dim) {
            throw std::invalid_argument("latent_row dimension does not match latent_dim");
        }

        cell_index.push_back(next_cell_index);
        embryo_id.push_back(next_embryo_id);
        developmental_time.push_back(next_time);
        for (float value : latent_row) latent.push_back(__float2half_rn(value));
        rows = static_cast<std::uint32_t>(cell_index.size());
    }

    const __half *latent_row_ptr(std::uint32_t row) const {
        if (row >= rows) throw std::out_of_range("latent_row_ptr row out of range");
        return latent.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(latent_dim);
    }
};

inline void sort_record_table(TrajectoryRecordTable *table) {
    if (table == nullptr) throw std::invalid_argument("sort_record_table requires a table");
    table->validate();

    std::vector<std::uint32_t> order(table->rows);
    for (std::uint32_t row = 0; row < table->rows; ++row) order[static_cast<std::size_t>(row)] = row;
    std::stable_sort(order.begin(), order.end(), [&](std::uint32_t lhs, std::uint32_t rhs) {
        if (table->embryo_id[lhs] < table->embryo_id[rhs]) return true;
        if (table->embryo_id[lhs] > table->embryo_id[rhs]) return false;
        if (table->developmental_time[lhs] < table->developmental_time[rhs]) return true;
        if (table->developmental_time[lhs] > table->developmental_time[rhs]) return false;
        return table->cell_index[lhs] < table->cell_index[rhs];
    });

    std::vector<std::uint64_t> next_cell_index(table->rows);
    std::vector<std::int32_t> next_embryo_id(table->rows);
    std::vector<float> next_time(table->rows);
    std::vector<__half> next_latent(static_cast<std::size_t>(table->rows) * static_cast<std::size_t>(table->latent_dim));

    for (std::uint32_t row = 0; row < table->rows; ++row) {
        const std::uint32_t src = order[static_cast<std::size_t>(row)];
        next_cell_index[row] = table->cell_index[src];
        next_embryo_id[row] = table->embryo_id[src];
        next_time[row] = table->developmental_time[src];
        const __half *src_row = table->latent_row_ptr(src);
        __half *dst_row = next_latent.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(table->latent_dim);
        std::copy(src_row, src_row + table->latent_dim, dst_row);
    }

    table->cell_index = std::move(next_cell_index);
    table->embryo_id = std::move(next_embryo_id);
    table->developmental_time = std::move(next_time);
    table->latent = std::move(next_latent);
}

inline std::vector<EmbryoRowSpan> build_embryo_row_spans(const TrajectoryRecordTable &table) {
    table.validate();
    std::vector<EmbryoRowSpan> spans;
    if (table.rows == 0) return spans;

    std::uint32_t begin = 0;
    while (begin < table.rows) {
        const std::int32_t embryo = table.embryo_id[begin];
        std::uint32_t end = begin + 1;
        while (end < table.rows && table.embryo_id[end] == embryo) ++end;
        spans.push_back(EmbryoRowSpan{ embryo, begin, end });
        begin = end;
    }
    return spans;
}

} // namespace cellerator::compute::graph
