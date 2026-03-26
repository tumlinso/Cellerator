#pragma once

namespace matrix {

template<typename MatrixT>
struct sharded {
    typedef typename MatrixT::value_type value_type;
    typedef Index index_type;

    Index rows;
    Index cols;
    Index nnz;
    unsigned char format;

    Index num_parts;
    Index part_capacity;
    MatrixT **parts;
    Index *part_offsets;
    Index *part_rows;
    Index *part_nnz;
    Index *part_aux;

    Index num_shards;
    Index shard_capacity;
    Index *shard_offsets;
};

template<typename MatrixT> inline void destroy(MatrixT *m);
template<typename MatrixT> inline Index part_aux(const MatrixT *m);
template<typename MatrixT> inline std::size_t part_bytes(const sharded<MatrixT> *m, Index partId);

template<typename MatrixT>
inline void init(sharded<MatrixT> *m) {
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->format = format_none;
    m->num_parts = 0;
    m->part_capacity = 0;
    m->parts = 0;
    m->part_offsets = 0;
    m->part_rows = 0;
    m->part_nnz = 0;
    m->part_aux = 0;
    m->num_shards = 0;
    m->shard_capacity = 0;
    m->shard_offsets = 0;
}

template<typename MatrixT>
inline void clear(sharded<MatrixT> *m) {
    Index i = 0;
    if (m->parts != 0) {
        for (i = 0; i < m->num_parts; ++i) destroy(m->parts[i]);
    }
    std::free(m->parts);
    std::free(m->part_offsets);
    std::free(m->part_rows);
    std::free(m->part_nnz);
    std::free(m->part_aux);
    std::free(m->shard_offsets);
    init(m);
}

template<typename MatrixT>
inline int reserve_parts(sharded<MatrixT> *m, Index capacity) {
    MatrixT **newParts = 0;
    Index *newOffsets = 0;
    Index *newRows = 0;
    Index *newNnz = 0;
    Index *newAux = 0;

    if (capacity <= m->part_capacity) return 1;
    newParts = (MatrixT **) std::calloc((std::size_t) capacity, sizeof(MatrixT *));
    newOffsets = (Index *) std::calloc((std::size_t) (capacity + 1), sizeof(Index));
    newRows = (Index *) std::calloc((std::size_t) capacity, sizeof(Index));
    newNnz = (Index *) std::calloc((std::size_t) capacity, sizeof(Index));
    newAux = (Index *) std::calloc((std::size_t) capacity, sizeof(Index));
    if (newParts == 0 || newOffsets == 0 || newRows == 0 || newNnz == 0 || newAux == 0) {
        std::free(newParts);
        std::free(newOffsets);
        std::free(newRows);
        std::free(newNnz);
        std::free(newAux);
        return 0;
    }
    if (m->num_parts != 0) {
        std::memcpy(newParts, m->parts, (std::size_t) m->num_parts * sizeof(MatrixT *));
        std::memcpy(newOffsets, m->part_offsets, (std::size_t) (m->num_parts + 1) * sizeof(Index));
        std::memcpy(newRows, m->part_rows, (std::size_t) m->num_parts * sizeof(Index));
        std::memcpy(newNnz, m->part_nnz, (std::size_t) m->num_parts * sizeof(Index));
        std::memcpy(newAux, m->part_aux, (std::size_t) m->num_parts * sizeof(Index));
    }

    std::free(m->parts);
    std::free(m->part_offsets);
    std::free(m->part_rows);
    std::free(m->part_nnz);
    std::free(m->part_aux);
    m->parts = newParts;
    m->part_offsets = newOffsets;
    m->part_rows = newRows;
    m->part_nnz = newNnz;
    m->part_aux = newAux;
    m->part_capacity = capacity;
    return 1;
}

template<typename MatrixT>
inline int reserve_shards(sharded<MatrixT> *m, Index capacity) {
    Index *newOffsets = 0;

    if (capacity <= m->shard_capacity) return 1;
    newOffsets = (Index *) std::calloc((std::size_t) (capacity + 1), sizeof(Index));
    if (newOffsets == 0) return 0;
    if (m->shard_offsets != 0 && m->num_shards != 0) {
        std::memcpy(newOffsets, m->shard_offsets, (std::size_t) (m->num_shards + 1) * sizeof(Index));
    }
    std::free(m->shard_offsets);
    m->shard_offsets = newOffsets;
    m->shard_capacity = capacity;
    return 1;
}

template<typename MatrixT>
inline void rebuild_part_offsets(sharded<MatrixT> *m) {
    Index i = 0;

    m->rows = 0;
    m->nnz = 0;
    if (m->part_offsets == 0) return;
    m->part_offsets[0] = 0;
    for (i = 0; i < m->num_parts; ++i) {
        m->part_offsets[i + 1] = m->part_offsets[i] + m->part_rows[i];
        m->nnz += m->part_nnz[i];
    }
    m->rows = m->part_offsets[m->num_parts];
}

template<typename MatrixT>
inline int set_shards_to_parts(sharded<MatrixT> *m) {
    Index i = 0;
    if (!reserve_shards(m, m->num_parts)) return 0;
    m->num_shards = m->num_parts;
    for (i = 0; i <= m->num_parts; ++i) m->shard_offsets[i] = m->part_offsets[i];
    return 1;
}

template<typename MatrixT>
inline Index find_part(const sharded<MatrixT> *m, Index row) {
    if (m->part_offsets == 0 || m->num_parts == 0) return m->num_parts;
    return find_offset_span(row, m->part_offsets, m->num_parts);
}

template<typename MatrixT>
inline Index find_shard(const sharded<MatrixT> *m, Index row) {
    if (m->shard_offsets == 0 || m->num_shards == 0) return m->num_shards;
    return find_offset_span(row, m->shard_offsets, m->num_shards);
}

template<typename MatrixT>
inline const typename MatrixT::value_type *at(const sharded<MatrixT> *m, Index r, Index c) {
    Index partId = find_part(m, r);
    if (partId >= m->num_parts || m->parts[partId] == 0) return 0;
    return at(m->parts[partId], r - m->part_offsets[partId], c);
}

template<typename MatrixT>
inline typename MatrixT::value_type *at(sharded<MatrixT> *m, Index r, Index c) {
    Index partId = find_part(m, r);
    if (partId >= m->num_parts || m->parts[partId] == 0) return 0;
    return at(m->parts[partId], r - m->part_offsets[partId], c);
}

template<typename MatrixT>
inline int append_part(sharded<MatrixT> *m, MatrixT *part) {
    Index next = 0;

    if (m->num_parts == m->part_capacity) {
        next = m->part_capacity == 0 ? 4 : m->part_capacity << 1;
        if (!reserve_parts(m, next)) return 0;
    }
    m->parts[m->num_parts] = part;
    m->part_rows[m->num_parts] = part != 0 ? part->rows : 0;
    m->part_nnz[m->num_parts] = part != 0 ? part->nnz : 0;
    m->part_aux[m->num_parts] = part != 0 ? ::matrix::part_aux(part) : 0;
    ++m->num_parts;
    if (part != 0) {
        if (m->cols == 0) m->cols = part->cols;
        m->format = part->format;
    }
    rebuild_part_offsets(m);
    return set_shards_to_parts(m);
}

template<typename MatrixT>
inline int concatenate(sharded<MatrixT> *dst, sharded<MatrixT> *src) {
    Index i = 0;

    if (src->num_parts == 0) return 1;
    if (!reserve_parts(dst, dst->num_parts + src->num_parts)) return 0;
    for (i = 0; i < src->num_parts; ++i) {
        dst->parts[dst->num_parts + i] = src->parts[i];
        dst->part_rows[dst->num_parts + i] = src->part_rows[i];
        dst->part_nnz[dst->num_parts + i] = src->part_nnz[i];
        dst->part_aux[dst->num_parts + i] = src->part_aux[i];
        src->parts[i] = 0;
        src->part_rows[i] = 0;
        src->part_nnz[i] = 0;
        src->part_aux[i] = 0;
    }
    dst->num_parts += src->num_parts;
    src->num_parts = 0;
    if (dst->cols == 0) dst->cols = src->cols;
    if (dst->format == format_none) dst->format = src->format;
    rebuild_part_offsets(dst);
    set_shards_to_parts(dst);
    rebuild_part_offsets(src);
    src->rows = 0;
    src->nnz = 0;
    src->num_shards = 0;
    return 1;
}

template<typename MatrixT>
inline int set_equal_shards(sharded<MatrixT> *m, Index count) {
    Index base = 0;
    Index rem = 0;
    Index row = 0;
    Index i = 0;

    if (count == 0) {
        m->num_shards = 0;
        return 1;
    }
    if (!reserve_shards(m, count)) return 0;
    m->num_shards = count;
    base = m->rows / count;
    rem = m->rows % count;
    for (i = 0; i < count; ++i) {
        m->shard_offsets[i] = row;
        row += base + (i < rem ? 1 : 0);
    }
    m->shard_offsets[count] = m->rows;
    return 1;
}

template<typename MatrixT>
inline int reshard(sharded<MatrixT> *m, Index count, const Index *offsets) {
    Index i = 0;

    if (count == 0 || offsets == 0) {
        m->num_shards = 0;
        return 1;
    }
    if (offsets[0] != 0 || offsets[count] != m->rows) return 0;
    for (i = 0; i < count; ++i) {
        if (offsets[i] > offsets[i + 1]) return 0;
    }
    if (!reserve_shards(m, count)) return 0;
    m->num_shards = count;
    for (i = 0; i <= count; ++i) m->shard_offsets[i] = offsets[i];
    return 1;
}

template<typename MatrixT>
inline Index first_part_in_shard(const sharded<MatrixT> *m, Index shardId) {
    if (shardId >= m->num_shards || m->num_parts == 0) return m->num_parts;
    return find_part(m, m->shard_offsets[shardId]);
}

template<typename MatrixT>
inline Index last_part_in_shard(const sharded<MatrixT> *m, Index shardId) {
    Index rowEnd = 0;
    if (shardId >= m->num_shards) return m->num_parts;
    rowEnd = m->shard_offsets[shardId + 1];
    if (rowEnd == 0) return 0;
    return find_part(m, rowEnd - 1) + 1;
}

template<typename MatrixT>
inline int part_loaded(const sharded<MatrixT> *m, Index partId) {
    if (partId >= m->num_parts) return 0;
    return m->parts[partId] != 0;
}

template<typename MatrixT>
inline int shard_loaded(const sharded<MatrixT> *m, Index shardId) {
    Index begin = 0;
    Index end = 0;
    Index i = 0;

    if (shardId >= m->num_shards) return 0;
    begin = first_part_in_shard(m, shardId);
    end = last_part_in_shard(m, shardId);
    for (i = begin; i < end; ++i) {
        if (!part_loaded(m, i)) return 0;
    }
    return 1;
}

template<typename MatrixT>
inline std::size_t bytes(const sharded<MatrixT> *m) {
    Index i = 0;
    std::size_t total = sizeof(*m);
    total += (std::size_t) m->part_capacity * sizeof(MatrixT *);
    total += (std::size_t) (m->part_capacity + 1) * sizeof(Index);
    total += (std::size_t) m->part_capacity * sizeof(Index);
    total += (std::size_t) m->part_capacity * sizeof(Index);
    total += (std::size_t) m->part_capacity * sizeof(Index);
    total += (std::size_t) (m->shard_capacity + 1) * sizeof(Index);
    for (i = 0; i < m->num_parts; ++i) {
        total += part_bytes(m, i);
    }
    return total;
}

template<typename MatrixT>
inline int set_shards_by_part_bytes(sharded<MatrixT> *m, std::size_t max_bytes) {
    std::size_t used = 0;
    std::size_t partBytes = 0;
    Index shardCount = 0;
    Index i = 0;

    if (max_bytes == 0) return set_shards_to_parts(m);
    if (!reserve_shards(m, m->num_parts)) return 0;

    m->shard_offsets[0] = 0;
    shardCount = 0;
    used = 0;
    for (i = 0; i < m->num_parts; ++i) {
        partBytes = part_bytes(m, i);
        if (partBytes == 0) continue;
        if (used != 0 && used + partBytes > max_bytes) {
            ++shardCount;
            m->shard_offsets[shardCount] = m->part_offsets[i];
            used = 0;
        }
        used += partBytes;
    }

    if (m->num_parts != 0) {
        ++shardCount;
        m->shard_offsets[shardCount] = m->rows;
    }
    m->num_shards = shardCount;
    return 1;
}

} // namespace matrix
