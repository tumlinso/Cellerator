#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "matrix.hh"

namespace matrix {

template<typename MatrixT>
struct shard_storage {
    typedef Index index_type;

    Index capacity;
    char **paths;
};

template<typename MatrixT>
inline void init(shard_storage<MatrixT> *s) {
    s->capacity = 0;
    s->paths = 0;
}

template<typename MatrixT>
inline void clear(shard_storage<MatrixT> *s) {
    Index i = 0;
    if (s->paths != 0) {
        for (i = 0; i < s->capacity; ++i) {
            if (s->paths[i] != 0) std::free(s->paths[i]);
        }
    }
    std::free(s->paths);
    s->paths = 0;
    s->capacity = 0;
}

template<typename MatrixT>
inline int reserve(shard_storage<MatrixT> *s, Index capacity) {
    Index i = 0;
    char **paths = 0;

    if (capacity <= s->capacity) return 1;
    paths = (char **) std::malloc((std::size_t) capacity * sizeof(char *));
    if (paths == 0) return 0;
    std::memset(paths, 0, (std::size_t) capacity * sizeof(char *));
    for (i = 0; i < s->capacity; ++i) paths[i] = s->paths[i];
    std::free(s->paths);
    s->paths = paths;
    s->capacity = capacity;
    return 1;
}

template<typename MatrixT>
inline int bind(shard_storage<MatrixT> *s, Index partId, const char *path) {
    std::size_t len = 0;
    char *copy = 0;

    if (partId >= s->capacity) return 0;
    if (s->paths[partId] != 0) std::free(s->paths[partId]);
    s->paths[partId] = 0;
    if (path == 0) return 1;

    len = std::strlen(path);
    copy = (char *) std::malloc(len + 1);
    if (copy == 0) return 0;
    std::memcpy(copy, path, len + 1);
    s->paths[partId] = copy;
    return 1;
}

template<typename MatrixT>
inline int bind_sequential(shard_storage<MatrixT> *s, Index count, const char *prefix) {
    char path[1024];
    Index i = 0;

    if (!reserve(s, count)) return 0;
    for (i = 0; i < count; ++i) {
        if (std::snprintf(path, sizeof(path), "%s.%u", prefix, (unsigned int) i) <= 0) return 0;
        if (!bind(s, i, path)) return 0;
    }
    return 1;
}

inline int write_header(std::FILE *fp, unsigned char format, Index rows, Index cols, Index nnz) {
    if (std::fwrite(&format, sizeof(format), 1, fp) != 1) return 0;
    if (std::fwrite(&rows, sizeof(rows), 1, fp) != 1) return 0;
    if (std::fwrite(&cols, sizeof(cols), 1, fp) != 1) return 0;
    if (std::fwrite(&nnz, sizeof(nnz), 1, fp) != 1) return 0;
    return 1;
}

inline int read_header(std::FILE *fp, header *out) {
    if (std::fread(&out->format, sizeof(out->format), 1, fp) != 1) return 0;
    if (std::fread(&out->rows, sizeof(out->rows), 1, fp) != 1) return 0;
    if (std::fread(&out->cols, sizeof(out->cols), 1, fp) != 1) return 0;
    if (std::fread(&out->nnz, sizeof(out->nnz), 1, fp) != 1) return 0;
    return 1;
}

template<typename ValueT>
inline int store(const char *filename, const dense<ValueT> *m) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) goto done;
    if (!write_header(fp, format_dense, m->rows, m->cols, m->nnz)) goto done;
    if (m->nnz != 0 && std::fwrite(m->val, sizeof(ValueT), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    return ok;
}

template<typename ValueT>
inline int load(const char *filename, dense<ValueT> *m) {
    std::FILE *fp = 0;
    header h;
    int ok = 0;

    fp = std::fopen(filename, "rb");
    if (fp == 0) goto done;
    if (!read_header(fp, &h)) goto done;
    if (!checkformat(format_dense, h.format, "dense matrix")) goto done;
    clear(m);
    init(m, h.rows, h.cols);
    m->nnz = h.nnz;
    if (!allocate(m)) goto done;
    if (m->nnz != 0 && std::fread(m->val, sizeof(ValueT), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    ok = 1;

done:
    if (!ok) clear(m);
    if (fp != 0) std::fclose(fp);
    return ok;
}

template<typename ValueT>
inline int store(const char *filename, const sparse::csr<ValueT> *m) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) goto done;
    if (!write_header(fp, format_csr, m->rows, m->cols, m->nnz)) goto done;
    if (m->nnz != 0 && std::fwrite(m->val, sizeof(ValueT), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    if (m->rows != 0 && std::fwrite(m->rowPtr, sizeof(Index), m->rows + 1, fp) != (std::size_t) (m->rows + 1)) goto done;
    if (m->nnz != 0 && std::fwrite(m->colIdx, sizeof(Index), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    return ok;
}

template<typename ValueT>
inline int load(const char *filename, sparse::csr<ValueT> *m) {
    std::FILE *fp = 0;
    header h;
    int ok = 0;

    fp = std::fopen(filename, "rb");
    if (fp == 0) goto done;
    if (!read_header(fp, &h)) goto done;
    if (!checkformat(format_csr, h.format, "csr matrix")) goto done;
    sparse::clear(m);
    sparse::init(m, h.rows, h.cols, h.nnz);
    if (!sparse::allocate(m)) goto done;
    if (m->nnz != 0 && std::fread(m->val, sizeof(ValueT), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    if (m->rows != 0 && std::fread(m->rowPtr, sizeof(Index), m->rows + 1, fp) != (std::size_t) (m->rows + 1)) goto done;
    if (m->nnz != 0 && std::fread(m->colIdx, sizeof(Index), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    ok = 1;

done:
    if (!ok) sparse::clear(m);
    if (fp != 0) std::fclose(fp);
    return ok;
}

template<typename ValueT>
inline int store(const char *filename, const sparse::coo<ValueT> *m) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) goto done;
    if (!write_header(fp, format_coo, m->rows, m->cols, m->nnz)) goto done;
    if (m->nnz != 0 && std::fwrite(m->val, sizeof(ValueT), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    if (m->nnz != 0 && std::fwrite(m->rowIdx, sizeof(Index), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    if (m->nnz != 0 && std::fwrite(m->colIdx, sizeof(Index), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    return ok;
}

template<typename ValueT>
inline int load(const char *filename, sparse::coo<ValueT> *m) {
    std::FILE *fp = 0;
    header h;
    int ok = 0;

    fp = std::fopen(filename, "rb");
    if (fp == 0) goto done;
    if (!read_header(fp, &h)) goto done;
    if (!checkformat(format_coo, h.format, "coo matrix")) goto done;
    sparse::clear(m);
    sparse::init(m, h.rows, h.cols, h.nnz);
    if (!sparse::allocate(m)) goto done;
    if (m->nnz != 0 && std::fread(m->val, sizeof(ValueT), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    if (m->nnz != 0 && std::fread(m->rowIdx, sizeof(Index), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    if (m->nnz != 0 && std::fread(m->colIdx, sizeof(Index), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    ok = 1;

done:
    if (!ok) sparse::clear(m);
    if (fp != 0) std::fclose(fp);
    return ok;
}

template<typename ValueT>
inline int store(const char *filename, const sparse::dia<ValueT> *m) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) goto done;
    if (!write_header(fp, format_dia, m->rows, m->cols, m->nnz)) goto done;
    if (std::fwrite(&m->num_diagonals, sizeof(Index), 1, fp) != 1) goto done;
    if (m->num_diagonals != 0 && std::fwrite(m->offsets, sizeof(Index), m->num_diagonals, fp) != (std::size_t) m->num_diagonals) goto done;
    if (m->nnz != 0 && std::fwrite(m->val, sizeof(ValueT), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    return ok;
}

template<typename ValueT>
inline int load(const char *filename, sparse::dia<ValueT> *m) {
    std::FILE *fp = 0;
    header h;
    int ok = 0;

    fp = std::fopen(filename, "rb");
    if (fp == 0) goto done;
    if (!read_header(fp, &h)) goto done;
    if (!checkformat(format_dia, h.format, "dia matrix")) goto done;
    sparse::clear(m);
    sparse::init(m, h.rows, h.cols, h.nnz);
    if (std::fread(&m->num_diagonals, sizeof(Index), 1, fp) != 1) goto done;
    if (!sparse::allocate(m)) goto done;
    if (m->num_diagonals != 0 && std::fread(m->offsets, sizeof(Index), m->num_diagonals, fp) != (std::size_t) m->num_diagonals) goto done;
    if (m->nnz != 0 && std::fread(m->val, sizeof(ValueT), m->nnz, fp) != (std::size_t) m->nnz) goto done;
    ok = 1;

done:
    if (!ok) sparse::clear(m);
    if (fp != 0) std::fclose(fp);
    return ok;
}

template<typename MatrixT>
inline int store(const char *filename, const sharded<MatrixT> *m, const shard_storage<MatrixT> *s) {
    std::FILE *fp = 0;
    Index i = 0;
    int ok = 0;

    if (s == 0 || s->capacity < m->num_parts) return 0;
    fp = std::fopen(filename, "wb");
    if (fp == 0) goto done;
    if (!write_header(fp, m->format, m->rows, m->cols, m->nnz)) goto done;
    if (std::fwrite(&m->num_parts, sizeof(Index), 1, fp) != 1) goto done;
    if (std::fwrite(&m->num_shards, sizeof(Index), 1, fp) != 1) goto done;
    if (m->num_parts != 0 && std::fwrite(m->part_rows, sizeof(Index), m->num_parts, fp) != (std::size_t) m->num_parts) goto done;
    if (m->num_parts != 0 && std::fwrite(m->part_nnz, sizeof(Index), m->num_parts, fp) != (std::size_t) m->num_parts) goto done;
    if (m->num_parts != 0 && std::fwrite(m->part_aux, sizeof(Index), m->num_parts, fp) != (std::size_t) m->num_parts) goto done;
    if (m->num_shards != 0 && std::fwrite(m->shard_offsets, sizeof(Index), m->num_shards + 1, fp) != (std::size_t) (m->num_shards + 1)) goto done;
    std::fclose(fp);
    fp = 0;

    for (i = 0; i < m->num_parts; ++i) {
        if (m->parts[i] == 0 || s->paths[i] == 0) goto done;
        if (!store(s->paths[i], m->parts[i])) goto done;
    }
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    return ok;
}

template<typename MatrixT>
inline int load_header(const char *filename, sharded<MatrixT> *m) {
    std::FILE *fp = 0;
    header h;
    Index numParts = 0;
    Index numShards = 0;
    int ok = 0;

    fp = std::fopen(filename, "rb");
    if (fp == 0) goto done;
    if (!read_header(fp, &h)) goto done;
    if (std::fread(&numParts, sizeof(Index), 1, fp) != 1) goto done;
    if (std::fread(&numShards, sizeof(Index), 1, fp) != 1) goto done;

    clear(m);
    init(m);
    m->rows = h.rows;
    m->cols = h.cols;
    m->nnz = h.nnz;
    m->format = h.format;
    if (!reserve_parts(m, numParts)) goto done;
    if (!reserve_shards(m, numShards)) goto done;
    m->num_parts = numParts;
    m->num_shards = numShards;
    if (numParts != 0 && std::fread(m->part_rows, sizeof(Index), numParts, fp) != (std::size_t) numParts) goto done;
    if (numParts != 0 && std::fread(m->part_nnz, sizeof(Index), numParts, fp) != (std::size_t) numParts) goto done;
    if (numParts != 0 && std::fread(m->part_aux, sizeof(Index), numParts, fp) != (std::size_t) numParts) goto done;
    if (numShards != 0 && std::fread(m->shard_offsets, sizeof(Index), numShards + 1, fp) != (std::size_t) (numShards + 1)) goto done;
    rebuild_part_offsets(m);
    ok = 1;

done:
    if (!ok) clear(m);
    if (fp != 0) std::fclose(fp);
    return ok;
}

template<typename MatrixT>
inline int fetch_part(sharded<MatrixT> *m, const shard_storage<MatrixT> *s, Index partId) {
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
inline int fetch_all_parts(sharded<MatrixT> *m, const shard_storage<MatrixT> *s) {
    Index i = 0;
    for (i = 0; i < m->num_parts; ++i) {
        if (!fetch_part(m, s, i)) return 0;
    }
    if (m->num_shards == 0) return set_shards_to_parts(m);
    return 1;
}

template<typename MatrixT>
inline int fetch_shard(sharded<MatrixT> *m, const shard_storage<MatrixT> *s, Index shardId) {
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
