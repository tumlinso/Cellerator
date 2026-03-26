#pragma once

#include <cstddef>
#include <cstdlib>

#include "matrix.hh"

namespace matrix {

struct shard_storage {
    Index capacity;
    char **paths;
};

namespace detail {

struct dense_load_result {
    header h;
    void *val;
};

struct csr_load_result {
    header h;
    Index *rowPtr;
    Index *colIdx;
    void *val;
};

struct coo_load_result {
    header h;
    Index *rowIdx;
    Index *colIdx;
    void *val;
};

struct dia_load_result {
    header h;
    Index num_diagonals;
    DiagIndex *offsets;
    void *val;
};

struct sharded_header_load_result {
    header h;
    Index num_parts;
    Index num_shards;
    Index *part_rows;
    Index *part_nnz;
    Index *part_aux;
    Index *shard_offsets;
};

void init(shard_storage *s);
void clear(shard_storage *s);
int reserve(shard_storage *s, Index capacity);
int bind(shard_storage *s, Index partId, const char *path);
int bind_sequential(shard_storage *s, Index count, const char *prefix);

int store_dense_raw(const char *filename, Index rows, Index cols, Index nnz, const void *val, std::size_t value_size);
int load_dense_raw(const char *filename, std::size_t value_size, dense_load_result *out);

int store_csr_raw(const char *filename, Index rows, Index cols, Index nnz, const Index *rowPtr, const Index *colIdx, const void *val, std::size_t value_size);
int load_csr_raw(const char *filename, std::size_t value_size, csr_load_result *out);

int store_coo_raw(const char *filename, Index rows, Index cols, Index nnz, const Index *rowIdx, const Index *colIdx, const void *val, std::size_t value_size);
int load_coo_raw(const char *filename, std::size_t value_size, coo_load_result *out);

int store_dia_raw(const char *filename, Index rows, Index cols, Index nnz, Index num_diagonals, const DiagIndex *offsets, const void *val, std::size_t value_size);
int load_dia_raw(const char *filename, std::size_t value_size, dia_load_result *out);

int store_sharded_header_raw(const char *filename,
                             unsigned char format,
                             Index rows,
                             Index cols,
                             Index nnz,
                             Index num_parts,
                             Index num_shards,
                             const Index *part_rows,
                             const Index *part_nnz,
                             const Index *part_aux,
                             const Index *shard_offsets);

int load_sharded_header_raw(const char *filename, sharded_header_load_result *out);

} // namespace detail

inline void init(shard_storage *s) {
    detail::init(s);
}

inline void clear(shard_storage *s) {
    detail::clear(s);
}

inline int reserve(shard_storage *s, Index capacity) {
    return detail::reserve(s, capacity);
}

inline int bind(shard_storage *s, Index partId, const char *path) {
    return detail::bind(s, partId, path);
}

inline int bind_sequential(shard_storage *s, Index count, const char *prefix) {
    return detail::bind_sequential(s, count, prefix);
}

template<typename ValueT>
inline int store(const char *filename, const dense<ValueT> *m) {
    return detail::store_dense_raw(filename, m->rows, m->cols, m->nnz, m->val, sizeof(ValueT));
}

template<typename ValueT>
inline int load(const char *filename, dense<ValueT> *m) {
    detail::dense_load_result tmp;

    tmp.val = 0;
    if (!detail::load_dense_raw(filename, sizeof(ValueT), &tmp)) return 0;
    clear(m);
    init(m, tmp.h.rows, tmp.h.cols);
    m->nnz = tmp.h.nnz;
    m->val = (ValueT *) tmp.val;
    return 1;
}

template<typename ValueT>
inline int store(const char *filename, const sparse::csr<ValueT> *m) {
    return detail::store_csr_raw(filename, m->rows, m->cols, m->nnz, m->rowPtr, m->colIdx, m->val, sizeof(ValueT));
}

template<typename ValueT>
inline int load(const char *filename, sparse::csr<ValueT> *m) {
    detail::csr_load_result tmp;

    tmp.rowPtr = 0;
    tmp.colIdx = 0;
    tmp.val = 0;
    if (!detail::load_csr_raw(filename, sizeof(ValueT), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->rowPtr = tmp.rowPtr;
    m->colIdx = tmp.colIdx;
    m->val = (ValueT *) tmp.val;
    return 1;
}

template<typename ValueT>
inline int store(const char *filename, const sparse::coo<ValueT> *m) {
    return detail::store_coo_raw(filename, m->rows, m->cols, m->nnz, m->rowIdx, m->colIdx, m->val, sizeof(ValueT));
}

template<typename ValueT>
inline int load(const char *filename, sparse::coo<ValueT> *m) {
    detail::coo_load_result tmp;

    tmp.rowIdx = 0;
    tmp.colIdx = 0;
    tmp.val = 0;
    if (!detail::load_coo_raw(filename, sizeof(ValueT), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->rowIdx = tmp.rowIdx;
    m->colIdx = tmp.colIdx;
    m->val = (ValueT *) tmp.val;
    return 1;
}

template<typename ValueT>
inline int store(const char *filename, const sparse::dia<ValueT> *m) {
    return detail::store_dia_raw(filename, m->rows, m->cols, m->nnz, m->num_diagonals, m->offsets, m->val, sizeof(ValueT));
}

template<typename ValueT>
inline int load(const char *filename, sparse::dia<ValueT> *m) {
    detail::dia_load_result tmp;

    tmp.num_diagonals = 0;
    tmp.offsets = 0;
    tmp.val = 0;
    if (!detail::load_dia_raw(filename, sizeof(ValueT), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->num_diagonals = tmp.num_diagonals;
    m->offsets = tmp.offsets;
    m->val = (ValueT *) tmp.val;
    return 1;
}

template<typename MatrixT>
inline int store(const char *filename, const sharded<MatrixT> *m, const shard_storage *s) {
    Index i = 0;

    if (s == 0 || s->capacity < m->num_parts) return 0;
    if (!detail::store_sharded_header_raw(filename,
                                          m->format,
                                          m->rows,
                                          m->cols,
                                          m->nnz,
                                          m->num_parts,
                                          m->num_shards,
                                          m->part_rows,
                                          m->part_nnz,
                                          m->part_aux,
                                          m->shard_offsets)) return 0;

    for (i = 0; i < m->num_parts; ++i) {
        if (m->parts[i] == 0 || s->paths[i] == 0) return 0;
        if (!store(s->paths[i], m->parts[i])) return 0;
    }
    return 1;
}

template<typename MatrixT>
inline int load_header(const char *filename, sharded<MatrixT> *m) {
    detail::sharded_header_load_result tmp;

    tmp.part_rows = 0;
    tmp.part_nnz = 0;
    tmp.part_aux = 0;
    tmp.shard_offsets = 0;
    if (!detail::load_sharded_header_raw(filename, &tmp)) return 0;

    clear(m);
    init(m);
    m->rows = tmp.h.rows;
    m->cols = tmp.h.cols;
    m->nnz = tmp.h.nnz;
    m->format = tmp.h.format;
    m->num_parts = tmp.num_parts;
    m->part_capacity = tmp.num_parts;
    m->num_shards = tmp.num_shards;
    m->shard_capacity = tmp.num_shards;
    m->parts = 0;
    m->part_offsets = 0;
    m->part_rows = tmp.part_rows;
    m->part_nnz = tmp.part_nnz;
    m->part_aux = tmp.part_aux;
    m->shard_offsets = tmp.shard_offsets;

    if (m->part_capacity != 0) {
        m->parts = (MatrixT **) std::calloc((std::size_t) m->part_capacity, sizeof(MatrixT *));
        m->part_offsets = (Index *) std::calloc((std::size_t) (m->part_capacity + 1), sizeof(Index));
        if (m->parts == 0 || m->part_offsets == 0) {
            clear(m);
            return 0;
        }
    }
    rebuild_part_offsets(m);
    return 1;
}

template<typename MatrixT>
inline int fetch_part(sharded<MatrixT> *m, const shard_storage *s, Index partId) {
    MatrixT *part = 0;
    int ok = 0;

    if (partId >= m->num_parts || s == 0 || partId >= s->capacity || s->paths[partId] == 0) return 0;
    if (m->parts[partId] != 0) destroy(m->parts[partId]);
    m->parts[partId] = 0;

    part = new MatrixT;
    init(part);
    if (!load(s->paths[partId], part)) {
        destroy(part);
        return 0;
    }
    if (part->rows != m->part_rows[partId]) goto fail;
    if (part->nnz != m->part_nnz[partId]) goto fail;
    if (m->cols != 0 && part->cols != m->cols) goto fail;
    if (part->format != m->format) goto fail;
    if (::matrix::part_aux(part) != m->part_aux[partId]) goto fail;
    m->parts[partId] = part;
    ok = 1;

fail:
    if (!ok) destroy(part);
    return ok;
}

template<typename MatrixT>
inline int fetch_all_parts(sharded<MatrixT> *m, const shard_storage *s) {
    Index i = 0;
    for (i = 0; i < m->num_parts; ++i) {
        if (!fetch_part(m, s, i)) return 0;
    }
    if (m->num_shards == 0) return set_shards_to_parts(m);
    return 1;
}

template<typename MatrixT>
inline int fetch_shard(sharded<MatrixT> *m, const shard_storage *s, Index shardId) {
    Index begin = 0;
    Index end = 0;
    Index i = 0;

    if (shardId >= m->num_shards) return 0;
    begin = first_part_in_shard(m, shardId);
    end = last_part_in_shard(m, shardId);
    for (i = begin; i < end; ++i) {
        if (!fetch_part(m, s, i)) return 0;
    }
    return 1;
}

template<typename MatrixT>
inline int drop_part(sharded<MatrixT> *m, Index partId) {
    if (partId >= m->num_parts) return 0;
    destroy(m->parts[partId]);
    m->parts[partId] = 0;
    return 1;
}

template<typename MatrixT>
inline int drop_shard(sharded<MatrixT> *m, Index shardId) {
    Index begin = 0;
    Index end = 0;
    Index i = 0;

    if (shardId >= m->num_shards) return 0;
    begin = first_part_in_shard(m, shardId);
    end = last_part_in_shard(m, shardId);
    for (i = begin; i < end; ++i) {
        if (!drop_part(m, i)) return 0;
    }
    return 1;
}

} // namespace matrix
