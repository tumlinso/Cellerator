#include "matrix/matrix.cuh"
#include "matrix/matrix_io.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace {

typedef std::uint64_t U64;
typedef matrix::sparse::csr<matrix::Real> CsrMatrix;
typedef matrix::sparse::coo<matrix::Real> CooMatrix;
typedef matrix::sharded<CsrMatrix, long> CsrSharded;
typedef matrix::sharded<CooMatrix, long> CooSharded;

static const U64 U32_LIMIT = 0xffffffffull;
static const std::size_t SOURCE_BUFFER_BYTES = 128u << 20;
static const std::size_t SOURCE_TAIL_BYTES = 1u << 20;

struct gpu_verify_accum {
    unsigned long long samples;
    unsigned long long exact_cast;
    unsigned long long rounded_cast;
    unsigned long long overflow_cast;
    unsigned long long pipeline_match;
    unsigned long long pipeline_mismatch;
    unsigned long long missing_nonzero;
    unsigned long long finite_samples;
    unsigned long long rel_le_001pct;
    unsigned long long rel_le_01pct;
    unsigned long long rel_le_1pct;
    unsigned long long rel_le_5pct;
    unsigned long long rel_gt_5pct;
    double mean_abs_err_sum;
    double mean_rel_err_sum;
    float max_abs_err;
    float max_rel_err;
};

struct mapped_file {
    int fd;
    std::size_t bytes;
    const char *base;
};

struct mm_header {
    U64 rows;
    U64 cols;
    U64 nnz;
    U64 entries_offset;
    const char *entries_begin;
};

struct part_plan {
    matrix::Index count;
    matrix::Index capacity;
    matrix::Index *cell_begin;
    matrix::Index *row_count;
    matrix::Index *exon_nnz;
    matrix::Index *intron_nnz;
};

struct embryo_entry {
    unsigned int embryo_id;
    char *name;
    long global_cell_begin;
    long cells;
    matrix::Index gene_count;
    long part_begin;
    matrix::Index part_count;
    part_plan plan;
    U64 exon_file_bytes;
    U64 intron_file_bytes;
    U64 *exon_counts;
    U64 *intron_counts;
    U64 *exon_offsets;
    U64 *intron_offsets;
};

struct modality_task {
    matrix::Index embryo_i;
    int is_exon;
    U64 work_bytes;
};

struct convert_task {
    matrix::Index embryo_i;
    int is_exon;
    matrix::Index part_begin;
    matrix::Index part_end;
    U64 source_bytes;
    U64 work_bytes;
};

struct verify_request {
    matrix::Index embryo_i;
    int is_exon;
    U64 start_byte;
    U64 end_byte;
    char *data;
    U64 loaded_bytes;
};

struct verify_stats {
    long samples;
    long exact_cast;
    long rounded_cast;
    long overflow_cast;
    long pipeline_match;
    long pipeline_mismatch;
    long missing_nonzero;
    long finite_samples;
    long rel_le_001pct;
    long rel_le_01pct;
    long rel_le_1pct;
    long rel_le_5pct;
    long rel_gt_5pct;
    double mean_abs_err_sum;
    double mean_rel_err_sum;
    double max_abs_err;
    double max_rel_err;
};

struct verify_part_cache {
    long part_id;
    CsrMatrix part;
};

struct gpu_csr_workspace {
    int device;
    cudaStream_t stream;
    matrix::Index row_capacity;
    matrix::Index nnz_capacity;
    std::size_t scan_capacity;
    matrix::Index *d_row_idx;
    matrix::Index *d_col_idx;
    matrix::Real *d_val;
    matrix::Index *d_row_ptr;
    matrix::Index *d_heads;
    matrix::Index *d_out_col;
    matrix::Real *d_out_val;
    void *scan_tmp;
    matrix::Index host_row_capacity;
    matrix::Index host_nnz_capacity;
    matrix::Index *host_row_ptr;
    matrix::Index *host_col;
    matrix::Real *host_val;
};

__device__ __forceinline__ unsigned long long atomic_add_double_bits(double *addr, double value) {
    unsigned long long *addr_i = (unsigned long long *) addr;
    unsigned long long old = *addr_i;
    unsigned long long assumed = old;
    do {
        assumed = old;
        old = atomicCAS(addr_i,
                        assumed,
                        __double_as_longlong(__longlong_as_double((long long) assumed) + value));
    } while (old != assumed);
    return old;
}

__device__ __forceinline__ int atomic_max_float(float *addr, float value) {
    int *addr_i = (int *) addr;
    int old = *addr_i;
    int assumed = old;
    if (__int_as_float(old) >= value) return old;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) break;
        old = atomicCAS(addr_i, assumed, __float_as_int(value));
    } while (old != assumed);
    return old;
}

__global__ void csr_count_rows_kernel(matrix::Index nnz,
                                      const matrix::Index * __restrict__ row_idx,
                                      matrix::Index * __restrict__ row_ptr_shifted) {
    const matrix::Index tid = (matrix::Index) (blockIdx.x * blockDim.x + threadIdx.x);
    const matrix::Index stride = (matrix::Index) (gridDim.x * blockDim.x);
    matrix::Index i = tid;
    while (i < nnz) {
        atomicAdd(row_ptr_shifted + row_idx[i] + 1, 1u);
        i += stride;
    }
}

__global__ void csr_init_heads_kernel(matrix::Index rows,
                                      const matrix::Index * __restrict__ row_ptr,
                                      matrix::Index * __restrict__ heads) {
    const matrix::Index tid = (matrix::Index) (blockIdx.x * blockDim.x + threadIdx.x);
    const matrix::Index stride = (matrix::Index) (gridDim.x * blockDim.x);
    matrix::Index i = tid;
    while (i < rows) {
        heads[i] = row_ptr[i];
        i += stride;
    }
}

__global__ void csr_scatter_kernel(matrix::Index nnz,
                                   const matrix::Index * __restrict__ row_idx,
                                   const matrix::Index * __restrict__ col_idx,
                                   const matrix::Real * __restrict__ val,
                                   matrix::Index * __restrict__ heads,
                                   matrix::Index * __restrict__ out_col,
                                   matrix::Real * __restrict__ out_val) {
    const matrix::Index tid = (matrix::Index) (blockIdx.x * blockDim.x + threadIdx.x);
    const matrix::Index stride = (matrix::Index) (gridDim.x * blockDim.x);
    matrix::Index i = tid;
    while (i < nnz) {
        const matrix::Index dst = atomicAdd(heads + row_idx[i], 1u);
        out_col[dst] = col_idx[i];
        out_val[dst] = val[i];
        i += stride;
    }
}

__global__ void verify_csr_samples_kernel(matrix::Index samples,
                                          const matrix::Index * __restrict__ row_ptr,
                                          const matrix::Index * __restrict__ col_idx,
                                          const matrix::Real * __restrict__ val,
                                          const matrix::Index * __restrict__ sample_rows,
                                          const matrix::Index * __restrict__ sample_cols,
                                          const int * __restrict__ sample_original,
                                          gpu_verify_accum * __restrict__ stats) {
    const matrix::Index tid = (matrix::Index) (blockIdx.x * blockDim.x + threadIdx.x);
    const matrix::Index stride = (matrix::Index) (gridDim.x * blockDim.x);
    matrix::Index i = tid;

    while (i < samples) {
        const matrix::Index row = sample_rows[i];
        const matrix::Index col = sample_cols[i];
        const int original = sample_original[i];
        const float expected_cast = matrix::real_to_float(matrix::real_from_float((float) original));
        const float original_float = (float) original;
        const matrix::Index begin = row_ptr[row];
        const matrix::Index end = row_ptr[row + 1];
        float stored = 0.0f;
        int found = 0;
        matrix::Index j = begin;

        while (j < end) {
            if (col_idx[j] == col) {
                stored = matrix::real_to_float(val[j]);
                found = 1;
                break;
            }
            ++j;
        }

        atomicAdd(&stats->samples, 1ull);
        if (!found) atomicAdd(&stats->missing_nonzero, 1ull);
        if (stored == expected_cast) atomicAdd(&stats->pipeline_match, 1ull);
        else atomicAdd(&stats->pipeline_mismatch, 1ull);

        if (!isfinite(expected_cast)) {
            atomicAdd(&stats->overflow_cast, 1ull);
            i += stride;
            continue;
        }

        if (expected_cast == original_float) atomicAdd(&stats->exact_cast, 1ull);
        else atomicAdd(&stats->rounded_cast, 1ull);

        {
            const float abs_err = fabsf(expected_cast - original_float);
            const float denom = fabsf(original_float) > 1.0f ? fabsf(original_float) : 1.0f;
            const float rel_err = abs_err / denom;

            atomicAdd(&stats->finite_samples, 1ull);
            atomic_add_double_bits(&stats->mean_abs_err_sum, (double) abs_err);
            atomic_add_double_bits(&stats->mean_rel_err_sum, (double) rel_err);
            atomic_max_float(&stats->max_abs_err, abs_err);
            atomic_max_float(&stats->max_rel_err, rel_err);
            if (rel_err <= 0.00001f) atomicAdd(&stats->rel_le_001pct, 1ull);
            else if (rel_err <= 0.0001f) atomicAdd(&stats->rel_le_01pct, 1ull);
            else if (rel_err <= 0.01f) atomicAdd(&stats->rel_le_1pct, 1ull);
            else if (rel_err <= 0.05f) atomicAdd(&stats->rel_le_5pct, 1ull);
            else atomicAdd(&stats->rel_gt_5pct, 1ull);
        }

        i += stride;
    }
}

static void init_part_plan(part_plan *plan);
static void clear_part_plan(part_plan *plan);
static int choose_worker_threads(long work_items, long long bytes_per_worker, int hard_cap);
static int choose_gpu_count();
static int bind_task_gpu(matrix::Index task_id, int gpu_count);
static int cuda_check(cudaError_t err, const char *label);
static int open_readonly_fd(const char *path);
static int stat_file_bytes(const char *path, U64 *bytes_out);

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

static void destroy_csr_part(CsrMatrix *part) {
    if (part == 0) return;
    matrix::sparse::clear(part);
    delete part;
}

static void clear_csr_sharded(CsrSharded *view) {
    long i = 0;

    if (view->parts != 0) {
        for (i = 0; i < view->num_parts; ++i) destroy_csr_part(view->parts[i]);
    }
    std::free(view->parts);
    std::free(view->part_offsets);
    std::free(view->part_rows);
    std::free(view->part_nnz);
    std::free(view->part_aux);
    std::free(view->shard_offsets);
    matrix::init(view);
}

static void mapped_file_init(mapped_file *m) {
    m->fd = -1;
    m->bytes = 0;
    m->base = 0;
}

static void mapped_file_close(mapped_file *m) {
#if defined(POSIX_FADV_DONTNEED)
    if (m->fd >= 0) (void) posix_fadvise(m->fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
#if defined(MADV_DONTNEED)
    if (m->base != 0 && m->bytes != 0) (void) madvise((void *) m->base, m->bytes, MADV_DONTNEED);
#endif
    if (m->base != 0 && m->bytes != 0) munmap((void *) m->base, m->bytes);
    if (m->fd >= 0) close(m->fd);
    mapped_file_init(m);
}

static int mapped_file_open(const char *path, mapped_file *m) {
    struct stat st;

    mapped_file_init(m);
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) {
        std::fprintf(stderr, "Error: open failed for %s: %s\n", path, std::strerror(errno));
        return 0;
    }
    if (fstat(m->fd, &st) != 0) {
        std::fprintf(stderr, "Error: fstat failed for %s: %s\n", path, std::strerror(errno));
        mapped_file_close(m);
        return 0;
    }
    if (st.st_size <= 0) {
        std::fprintf(stderr, "Error: empty file %s\n", path);
        mapped_file_close(m);
        return 0;
    }
#if defined(POSIX_FADV_SEQUENTIAL)
    (void) posix_fadvise(m->fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    m->bytes = (std::size_t) st.st_size;
    m->base = (const char *) mmap(0, m->bytes, PROT_READ, MAP_PRIVATE, m->fd, 0);
    if (m->base == MAP_FAILED) {
        std::fprintf(stderr, "Error: mmap failed for %s: %s\n", path, std::strerror(errno));
        m->base = 0;
        mapped_file_close(m);
        return 0;
    }
#if defined(MADV_SEQUENTIAL)
    (void) madvise((void *) m->base, m->bytes, MADV_SEQUENTIAL);
#endif
#if defined(MADV_WILLNEED)
    (void) madvise((void *) m->base, m->bytes, MADV_WILLNEED);
#endif
    return 1;
}

static int is_digit(char c) {
    return c >= '0' && c <= '9';
}

static void skip_line(const char **p, const char *end) {
    while (*p < end && **p != '\n') ++(*p);
    if (*p < end) ++(*p);
}

static void skip_spaces(const char **p, const char *end) {
    while (*p < end) {
        const char c = **p;
        if (c != ' ' && c != '\t' && c != '\r') break;
        ++(*p);
    }
}

static void skip_empty_and_comment_lines(const char **p, const char *end) {
next_line:
    if (*p >= end) return;
    skip_spaces(p, end);
    if (*p >= end) return;
    if (**p == '%') {
        skip_line(p, end);
        goto next_line;
    }
    if (**p == '\n') {
        ++(*p);
        goto next_line;
    }
}

static int parse_u64_token(const char **p, const char *end, U64 *out) {
    U64 value = 0;

    skip_spaces(p, end);
    if (*p >= end || !is_digit(**p)) return 0;
    while (*p < end && is_digit(**p)) {
        value = value * 10ull + (U64) (**p - '0');
        ++(*p);
    }
    *out = value;
    return 1;
}

static int parse_i32_token(const char **p, const char *end, int *out) {
    long long value = 0;
    int neg = 0;

    skip_spaces(p, end);
    if (*p >= end) return 0;
    if (**p == '-') {
        neg = 1;
        ++(*p);
    } else if (**p == '+') {
        ++(*p);
    }
    if (*p >= end || !is_digit(**p)) return 0;
    while (*p < end && is_digit(**p)) {
        value = value * 10ll + (long long) (**p - '0');
        ++(*p);
    }
    *out = neg ? (int) (-value) : (int) value;
    return 1;
}

static int expect_line_end(const char **p, const char *end) {
    skip_spaces(p, end);
    if (*p >= end) return 1;
    if (**p == '\n') {
        ++(*p);
        return 1;
    }
    return 0;
}

template<typename T>
static int u64_to_value(U64 value, T *out, const char *label, const char *path) {
    if (value > (U64) std::numeric_limits<T>::max()) {
        std::fprintf(stderr,
                     "Error: %s exceeds target integer range in %s: %llu\n",
                     label,
                     path,
                     (unsigned long long) value);
        return 0;
    }
    *out = (T) value;
    return 1;
}

static int parse_matrix_market_header(const char *path, const mapped_file *m, mm_header *out) {
    static const char banner[] = "%%MatrixMarket matrix coordinate integer general";
    const char *p = m->base;
    const char *line = p;
    const char *end = m->base + m->bytes;
    std::size_t line_bytes = 0;

    while (p < end && *p != '\n') ++p;
    line_bytes = (std::size_t) (p - line);
    if (line_bytes > 0 && line[line_bytes - 1] == '\r') --line_bytes;
    if (line_bytes != sizeof(banner) - 1 || std::memcmp(line, banner, sizeof(banner) - 1) != 0) {
        std::fprintf(stderr, "Error: unsupported MatrixMarket banner in %s\n", path);
        return 0;
    }
    if (p < end) ++p;

    skip_empty_and_comment_lines(&p, end);
    if (!parse_u64_token(&p, end, &out->rows)) return 0;
    if (!parse_u64_token(&p, end, &out->cols)) return 0;
    if (!parse_u64_token(&p, end, &out->nnz)) return 0;
    if (!expect_line_end(&p, end)) return 0;
    out->entries_offset = (U64) (p - m->base);
    out->entries_begin = p;
    return 1;
}

static int read_matrix_header_only(const char *path, mm_header *hdr) {
    mapped_file m;
    int ok = 0;

    mapped_file_init(&m);
    if (!mapped_file_open(path, &m)) goto done;
    if (!parse_matrix_market_header(path, &m, hdr)) {
        std::fprintf(stderr, "Error: failed to parse header in %s\n", path);
        goto done;
    }
    ok = 1;

done:
    mapped_file_close(&m);
    return ok;
}

static char *duplicate_cstr(const char *src) {
    std::size_t bytes = 0;
    char *dst = 0;

    bytes = std::strlen(src) + 1;
    dst = (char *) std::malloc(bytes);
    if (dst == 0) return 0;
    std::memcpy(dst, src, bytes);
    return dst;
}

static int build_path3(char *dst, std::size_t capacity, const char *a, const char *b, const char *c) {
    const int written = std::snprintf(dst, capacity, "%s/%s/%s", a, b, c);
    return written > 0 && (std::size_t) written < capacity;
}

static int starts_with_embryo_prefix(const char *name, unsigned int *embryo_id) {
    static const char prefix[] = "embryo_";
    unsigned int value = 0;
    const char *p = name;

    if (std::strncmp(name, prefix, sizeof(prefix) - 1) != 0) return 0;
    p += sizeof(prefix) - 1;
    if (*p == 0) return 0;
    while (*p != 0) {
        if (!is_digit(*p)) return 0;
        value = value * 10u + (unsigned int) (*p - '0');
        ++p;
    }
    *embryo_id = value;
    return 1;
}

static int embryo_compare(const void *lhs, const void *rhs) {
    const embryo_entry *a = (const embryo_entry *) lhs;
    const embryo_entry *b = (const embryo_entry *) rhs;

    if (a->embryo_id < b->embryo_id) return -1;
    if (a->embryo_id > b->embryo_id) return 1;
    return std::strcmp(a->name, b->name);
}

static int modality_task_compare(const void *lhs, const void *rhs) {
    const modality_task *a = (const modality_task *) lhs;
    const modality_task *b = (const modality_task *) rhs;

    if (a->work_bytes > b->work_bytes) return -1;
    if (a->work_bytes < b->work_bytes) return 1;
    if (a->embryo_i < b->embryo_i) return -1;
    if (a->embryo_i > b->embryo_i) return 1;
    return a->is_exon < b->is_exon ? -1 : (a->is_exon > b->is_exon ? 1 : 0);
}

static int convert_task_compare(const void *lhs, const void *rhs) {
    const convert_task *a = (const convert_task *) lhs;
    const convert_task *b = (const convert_task *) rhs;

    if (a->source_bytes > b->source_bytes) return -1;
    if (a->source_bytes < b->source_bytes) return 1;
    if (a->embryo_i < b->embryo_i) return -1;
    if (a->embryo_i > b->embryo_i) return 1;
    if (a->is_exon < b->is_exon) return -1;
    if (a->is_exon > b->is_exon) return 1;
    if (a->part_begin < b->part_begin) return -1;
    if (a->part_begin > b->part_begin) return 1;
    return 0;
}

static int verify_request_compare(const void *lhs, const void *rhs) {
    const verify_request *a = (const verify_request *) lhs;
    const verify_request *b = (const verify_request *) rhs;

    if (a->is_exon < b->is_exon) return -1;
    if (a->is_exon > b->is_exon) return 1;
    if (a->embryo_i < b->embryo_i) return -1;
    if (a->embryo_i > b->embryo_i) return 1;
    if (a->start_byte < b->start_byte) return -1;
    if (a->start_byte > b->start_byte) return 1;
    return 0;
}

static void init_verify_stats(verify_stats *stats) {
    std::memset(stats, 0, sizeof(*stats));
}

static void init_verify_part_cache(verify_part_cache *cache) {
    cache->part_id = -1;
    matrix::sparse::init(&cache->part);
}

static void init_gpu_csr_workspace(gpu_csr_workspace *ws) {
    std::memset(ws, 0, sizeof(*ws));
    ws->device = -1;
}

static void clear_gpu_csr_workspace(gpu_csr_workspace *ws) {
    if (ws->scan_tmp != 0) cudaFree(ws->scan_tmp);
    if (ws->d_out_val != 0) cudaFree(ws->d_out_val);
    if (ws->d_out_col != 0) cudaFree(ws->d_out_col);
    if (ws->d_heads != 0) cudaFree(ws->d_heads);
    if (ws->d_row_ptr != 0) cudaFree(ws->d_row_ptr);
    if (ws->d_val != 0) cudaFree(ws->d_val);
    if (ws->d_col_idx != 0) cudaFree(ws->d_col_idx);
    if (ws->d_row_idx != 0) cudaFree(ws->d_row_idx);
    if (ws->host_val != 0) cudaFreeHost(ws->host_val);
    if (ws->host_col != 0) cudaFreeHost(ws->host_col);
    if (ws->host_row_ptr != 0) cudaFreeHost(ws->host_row_ptr);
    if (ws->stream != 0) cudaStreamDestroy(ws->stream);
    init_gpu_csr_workspace(ws);
}

static int setup_gpu_csr_workspace(gpu_csr_workspace *ws, int device) {
    init_gpu_csr_workspace(ws);
    ws->device = device;
    if (!cuda_check(cudaSetDevice(device), "cudaSetDevice workspace")) return 0;
    return cuda_check(cudaStreamCreateWithFlags(&ws->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
}

static int reserve_gpu_csr_workspace(gpu_csr_workspace *ws, matrix::Index rows, matrix::Index nnz) {
    std::size_t scan_bytes = 0;
    if (rows > ws->row_capacity) {
        if (ws->d_row_ptr != 0) cudaFree(ws->d_row_ptr);
        if (ws->d_heads != 0) cudaFree(ws->d_heads);
        if (ws->host_row_ptr != 0) cudaFreeHost(ws->host_row_ptr);
        ws->d_row_ptr = 0;
        ws->d_heads = 0;
        ws->host_row_ptr = 0;
        if (!cuda_check(cudaMalloc((void **) &ws->d_row_ptr, (std::size_t) (rows + 1) * sizeof(matrix::Index)), "cudaMalloc ws d_row_ptr")) return 0;
        if (rows != 0 && !cuda_check(cudaMalloc((void **) &ws->d_heads, (std::size_t) rows * sizeof(matrix::Index)), "cudaMalloc ws d_heads")) return 0;
        if (!cuda_check(cudaMallocHost((void **) &ws->host_row_ptr, (std::size_t) (rows + 1) * sizeof(matrix::Index)), "cudaMallocHost ws host_row_ptr")) return 0;
        ws->row_capacity = rows;
        ws->host_row_capacity = rows;
    }
    if (nnz > ws->nnz_capacity) {
        if (ws->d_row_idx != 0) cudaFree(ws->d_row_idx);
        if (ws->d_col_idx != 0) cudaFree(ws->d_col_idx);
        if (ws->d_val != 0) cudaFree(ws->d_val);
        if (ws->d_out_col != 0) cudaFree(ws->d_out_col);
        if (ws->d_out_val != 0) cudaFree(ws->d_out_val);
        if (ws->host_col != 0) cudaFreeHost(ws->host_col);
        if (ws->host_val != 0) cudaFreeHost(ws->host_val);
        ws->d_row_idx = 0;
        ws->d_col_idx = 0;
        ws->d_val = 0;
        ws->d_out_col = 0;
        ws->d_out_val = 0;
        ws->host_col = 0;
        ws->host_val = 0;
        if (nnz != 0) {
            if (!cuda_check(cudaMalloc((void **) &ws->d_row_idx, (std::size_t) nnz * sizeof(matrix::Index)), "cudaMalloc ws d_row_idx")) return 0;
            if (!cuda_check(cudaMalloc((void **) &ws->d_col_idx, (std::size_t) nnz * sizeof(matrix::Index)), "cudaMalloc ws d_col_idx")) return 0;
            if (!cuda_check(cudaMalloc((void **) &ws->d_val, (std::size_t) nnz * sizeof(matrix::Real)), "cudaMalloc ws d_val")) return 0;
            if (!cuda_check(cudaMalloc((void **) &ws->d_out_col, (std::size_t) nnz * sizeof(matrix::Index)), "cudaMalloc ws d_out_col")) return 0;
            if (!cuda_check(cudaMalloc((void **) &ws->d_out_val, (std::size_t) nnz * sizeof(matrix::Real)), "cudaMalloc ws d_out_val")) return 0;
            if (!cuda_check(cudaMallocHost((void **) &ws->host_col, (std::size_t) nnz * sizeof(matrix::Index)), "cudaMallocHost ws host_col")) return 0;
            if (!cuda_check(cudaMallocHost((void **) &ws->host_val, (std::size_t) nnz * sizeof(matrix::Real)), "cudaMallocHost ws host_val")) return 0;
        }
        ws->nnz_capacity = nnz;
        ws->host_nnz_capacity = nnz;
    }
    if (!cuda_check(cub::DeviceScan::ExclusiveSum(0, scan_bytes, ws->d_row_ptr, ws->d_row_ptr, rows + 1, ws->stream), "cub dry scan ws")) return 0;
    if (scan_bytes > ws->scan_capacity) {
        if (ws->scan_tmp != 0) cudaFree(ws->scan_tmp);
        ws->scan_tmp = 0;
        if (scan_bytes != 0 && !cuda_check(cudaMalloc((void **) &ws->scan_tmp, scan_bytes), "cudaMalloc ws scan_tmp")) return 0;
        ws->scan_capacity = scan_bytes;
    }
    return 1;
}

static void clear_verify_part_cache(verify_part_cache *cache) {
    matrix::sparse::clear(&cache->part);
    cache->part_id = -1;
}

static std::uint64_t next_rng(std::uint64_t *state) {
    std::uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ull;
}

static void clear_embryos(embryo_entry *entries, matrix::Index count) {
    matrix::Index i = 0;

    if (entries == 0) return;
    for (i = 0; i < count; ++i) {
        clear_part_plan(&entries[i].plan);
        std::free(entries[i].exon_counts);
        std::free(entries[i].intron_counts);
        std::free(entries[i].exon_offsets);
        std::free(entries[i].intron_offsets);
        std::free(entries[i].name);
    }
    std::free(entries);
}

static int list_embryos(const char *base_dir, embryo_entry **out_entries, matrix::Index *out_count) {
    DIR *dir = 0;
    struct dirent *entry = 0;
    embryo_entry *entries = 0;
    embryo_entry *grown = 0;
    matrix::Index count = 0;
    matrix::Index capacity = 0;
    char *name_copy = 0;
    unsigned int embryo_id = 0;

    *out_entries = 0;
    *out_count = 0;
    dir = opendir(base_dir);
    if (dir == 0) {
        std::fprintf(stderr, "Error: opendir failed for %s: %s\n", base_dir, std::strerror(errno));
        return 0;
    }

next_entry:
    entry = readdir(dir);
    if (entry == 0) goto done;
    if (!starts_with_embryo_prefix(entry->d_name, &embryo_id)) goto next_entry;
    if (count == capacity) {
        capacity = capacity == 0 ? 16u : (capacity << 1);
        grown = (embryo_entry *) std::calloc((std::size_t) capacity, sizeof(embryo_entry));
        if (grown == 0) goto fail;
        if (count != 0) std::memcpy(grown, entries, (std::size_t) count * sizeof(embryo_entry));
        std::free(entries);
        entries = grown;
    }
    name_copy = duplicate_cstr(entry->d_name);
    if (name_copy == 0) goto fail;
    entries[count].embryo_id = embryo_id;
    entries[count].name = name_copy;
    entries[count].global_cell_begin = 0;
    entries[count].cells = 0;
    entries[count].gene_count = 0;
    entries[count].part_begin = 0;
    entries[count].part_count = 0;
    entries[count].exon_file_bytes = 0;
    entries[count].intron_file_bytes = 0;
    entries[count].exon_counts = 0;
    entries[count].intron_counts = 0;
    entries[count].exon_offsets = 0;
    entries[count].intron_offsets = 0;
    init_part_plan(&entries[count].plan);
    ++count;
    goto next_entry;

done:
    closedir(dir);
    if (count == 0) {
        clear_embryos(entries, count);
        std::fprintf(stderr, "Error: no embryo_* directories found under %s\n", base_dir);
        return 0;
    }
    qsort(entries, (std::size_t) count, sizeof(embryo_entry), embryo_compare);
    *out_entries = entries;
    *out_count = count;
    return 1;

fail:
    if (dir != 0) closedir(dir);
    clear_embryos(entries, count);
    return 0;
}

static void init_part_plan(part_plan *plan) {
    plan->count = 0;
    plan->capacity = 0;
    plan->cell_begin = 0;
    plan->row_count = 0;
    plan->exon_nnz = 0;
    plan->intron_nnz = 0;
}

static void clear_part_plan(part_plan *plan) {
    std::free(plan->cell_begin);
    std::free(plan->row_count);
    std::free(plan->exon_nnz);
    std::free(plan->intron_nnz);
    init_part_plan(plan);
}

static int reserve_part_plan(part_plan *plan, matrix::Index capacity) {
    matrix::Index *cell_begin = 0;
    matrix::Index *row_count = 0;
    matrix::Index *exon_nnz = 0;
    matrix::Index *intron_nnz = 0;

    if (capacity <= plan->capacity) return 1;

    cell_begin = (matrix::Index *) std::calloc((std::size_t) capacity, sizeof(matrix::Index));
    row_count = (matrix::Index *) std::calloc((std::size_t) capacity, sizeof(matrix::Index));
    exon_nnz = (matrix::Index *) std::calloc((std::size_t) capacity, sizeof(matrix::Index));
    intron_nnz = (matrix::Index *) std::calloc((std::size_t) capacity, sizeof(matrix::Index));
    if (cell_begin == 0 || row_count == 0 || exon_nnz == 0 || intron_nnz == 0) {
        std::free(cell_begin);
        std::free(row_count);
        std::free(exon_nnz);
        std::free(intron_nnz);
        return 0;
    }

    if (plan->count != 0) {
        std::memcpy(cell_begin, plan->cell_begin, (std::size_t) plan->count * sizeof(matrix::Index));
        std::memcpy(row_count, plan->row_count, (std::size_t) plan->count * sizeof(matrix::Index));
        std::memcpy(exon_nnz, plan->exon_nnz, (std::size_t) plan->count * sizeof(matrix::Index));
        std::memcpy(intron_nnz, plan->intron_nnz, (std::size_t) plan->count * sizeof(matrix::Index));
    }

    std::free(plan->cell_begin);
    std::free(plan->row_count);
    std::free(plan->exon_nnz);
    std::free(plan->intron_nnz);
    plan->cell_begin = cell_begin;
    plan->row_count = row_count;
    plan->exon_nnz = exon_nnz;
    plan->intron_nnz = intron_nnz;
    plan->capacity = capacity;
    return 1;
}

static int append_part_plan(part_plan *plan,
                            matrix::Index cell_begin,
                            matrix::Index row_count,
                            matrix::Index exon_nnz,
                            matrix::Index intron_nnz) {
    matrix::Index next = 0;

    if (plan->count == plan->capacity) {
        next = plan->capacity == 0 ? 16u : (plan->capacity << 1);
        if (!reserve_part_plan(plan, next)) return 0;
    }
    plan->cell_begin[plan->count] = cell_begin;
    plan->row_count[plan->count] = row_count;
    plan->exon_nnz[plan->count] = exon_nnz;
    plan->intron_nnz[plan->count] = intron_nnz;
    ++plan->count;
    return 1;
}

static int count_cells_from_matrix(const char *path,
                                   mm_header *hdr,
                                   U64 *cell_counts,
                                   U64 *cell_offsets,
                                   U64 *file_bytes_out) {
    int fd = -1;
    mm_header local_hdr;
    mm_header *parsed_hdr = hdr != 0 ? hdr : &local_hdr;
    char *buf = 0;
    std::size_t carry = 0;
    U64 next_read_offset = 0;
    U64 file_bytes = 0;
    U64 i = 0;
    U64 next_fill_cell = 0;
    U64 entry_bytes = 0;
    int ok = 0;

    if (cell_counts == 0) return 0;
    if (!read_matrix_header_only(path, parsed_hdr)) {
        std::fprintf(stderr, "Error: failed to parse header in %s\n", path);
        goto done;
    }
    fd = open_readonly_fd(path);
    if (fd < 0) goto done;
#if defined(POSIX_FADV_SEQUENTIAL)
    (void) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    if (!stat_file_bytes(path, &file_bytes)) goto done;
    if (file_bytes_out != 0) *file_bytes_out = file_bytes;
    entry_bytes = file_bytes - parsed_hdr->entries_offset;
    next_read_offset = parsed_hdr->entries_offset;
    buf = (char *) std::malloc(SOURCE_BUFFER_BYTES + SOURCE_TAIL_BYTES + 1u);
    if (buf == 0) goto done;

    while (i < parsed_hdr->nnz) {
        const U64 chunk_bytes_u64 = next_read_offset < file_bytes
            ? ((file_bytes - next_read_offset) > (U64) SOURCE_BUFFER_BYTES
                ? (U64) SOURCE_BUFFER_BYTES
                : (file_bytes - next_read_offset))
            : 0;
        const std::size_t chunk_bytes = (std::size_t) chunk_bytes_u64;
        const ssize_t got = chunk_bytes != 0
            ? pread(fd, buf + carry, chunk_bytes, (off_t) next_read_offset)
            : 0;
        const std::size_t total = carry + (got > 0 ? (std::size_t) got : 0u);
        const U64 buffer_file_begin = next_read_offset - (U64) carry;
        std::size_t parse_bytes = total;
        const char *p = buf;
        const char *end = 0;

        if (got < 0) goto done;
        next_read_offset += (got > 0 ? (U64) got : 0ull);
        if (next_read_offset < file_bytes) {
            while (parse_bytes != 0 && buf[parse_bytes - 1] != '\n') --parse_bytes;
            if (parse_bytes == 0) {
                std::fprintf(stderr, "Error: source buffer too small while counting %s\n", path);
                goto done;
            }
        }
        end = buf + parse_bytes;
        while (p < end && i < parsed_hdr->nnz) {
            const char *line = p;
            U64 gene = 0;
            U64 cell = 0;
            int value = 0;

            skip_empty_and_comment_lines(&p, end);
            line = p;
            if (p >= end) break;
            if (!parse_u64_token(&p, end, &gene) ||
                !parse_u64_token(&p, end, &cell) ||
                !parse_i32_token(&p, end, &value) ||
                !expect_line_end(&p, end)) goto done;
            if (cell == 0 || cell > parsed_hdr->cols || gene == 0 || gene > parsed_hdr->rows) goto done;
            if (cell_offsets != 0) {
                while (next_fill_cell < cell) {
                    cell_offsets[next_fill_cell] =
                        (buffer_file_begin + (U64) (line - buf)) - parsed_hdr->entries_offset;
                    ++next_fill_cell;
                }
            }
            ++cell_counts[cell - 1];
            ++i;
        }
        carry = total - parse_bytes;
        if (carry != 0) std::memmove(buf, buf + parse_bytes, carry);
        if (got == 0 && carry == 0) break;
    }
    if (cell_offsets != 0) {
        while (next_fill_cell <= parsed_hdr->cols) {
            cell_offsets[next_fill_cell] = entry_bytes;
            ++next_fill_cell;
        }
    }
    ok = 1;

done:
    if (!ok) std::fprintf(stderr, "Error: failed while counting cells in %s\n", path);
    if (fd >= 0) close(fd);
    std::free(buf);
    return ok;
}

static int build_common_part_plan(const char *label,
                                  matrix::Index cells,
                                  matrix::Index max_rows_per_part,
                                  const U64 *exon_counts,
                                  const U64 *intron_counts,
                                  part_plan *plan) {
    matrix::Index cell_begin = 0;
    matrix::Index row_count = 0;
    U64 exon_acc = 0;
    U64 intron_acc = 0;
    matrix::Index cell = 0;
    U64 exon_next = 0;
    U64 intron_next = 0;

    init_part_plan(plan);
    if (cells == 0) return 1;

    for (cell = 0; cell < cells; ++cell) {
        exon_next = exon_counts[cell];
        intron_next = intron_counts[cell];
        if (row_count != 0 &&
            (row_count == max_rows_per_part ||
             exon_acc + exon_next > U32_LIMIT ||
             intron_acc + intron_next > U32_LIMIT)) {
            if (!append_part_plan(plan,
                                  cell_begin,
                                  row_count,
                                  (matrix::Index) exon_acc,
                                  (matrix::Index) intron_acc)) return 0;
            cell_begin = cell;
            row_count = 0;
            exon_acc = 0;
            intron_acc = 0;
        }
        if (exon_next > U32_LIMIT || intron_next > U32_LIMIT) {
            std::fprintf(stderr,
                         "Error: single cell exceeds uint32 nnz budget in %s at local cell %u\n",
                         label,
                         (unsigned int) cell);
            clear_part_plan(plan);
            return 0;
        }
        exon_acc += exon_next;
        intron_acc += intron_next;
        ++row_count;
    }

    if (row_count != 0) {
        if (!append_part_plan(plan,
                              cell_begin,
                              row_count,
                              (matrix::Index) exon_acc,
                              (matrix::Index) intron_acc)) return 0;
    }
    return 1;
}

static CsrMatrix *allocate_csr_part(matrix::Index rows, matrix::Index cols, matrix::Index nnz) {
    CsrMatrix *part = new CsrMatrix;
    matrix::sparse::init(part, rows, cols, nnz);
    if (!matrix::sparse::allocate(part)) {
        destroy_csr_part(part);
        return 0;
    }
    return part;
}

static int build_part_path(char *dst, std::size_t capacity, const char *prefix, long part_id) {
    const int written = std::snprintf(dst, capacity, "%s.%ld", prefix, part_id);
    return written > 0 && (std::size_t) written < capacity;
}

static int store_part_direct(const char *part_prefix, long part_id, CsrMatrix **part_ptr) {
    CsrMatrix *part = *part_ptr;
    char path[4096];

    if (!build_part_path(path, sizeof(path), part_prefix, part_id)) return 0;
    if (!matrix::store(path, part)) return 0;
    destroy_csr_part(part);
    *part_ptr = 0;
    return 1;
}

static int analyze_embryo(const char *base_dir,
                          embryo_entry *embryo,
                          matrix::Index max_rows_per_part) {
    char exon_matrix_path[4096];
    char intron_matrix_path[4096];
    mm_header exon_hdr;
    mm_header intron_hdr;
    U64 *exon_counts = 0;
    U64 *intron_counts = 0;
    int ok = 0;

    if (!build_path3(exon_matrix_path, sizeof(exon_matrix_path), base_dir, embryo->name, "exon/matrix.mtx")) goto done;
    if (!build_path3(intron_matrix_path, sizeof(intron_matrix_path), base_dir, embryo->name, "intron/matrix.mtx")) goto done;
    if (!read_matrix_header_only(exon_matrix_path, &exon_hdr)) goto done;
    if (!read_matrix_header_only(intron_matrix_path, &intron_hdr)) goto done;
    if (exon_hdr.rows != intron_hdr.rows || exon_hdr.cols != intron_hdr.cols) {
        std::fprintf(stderr,
                     "Error: exon/intron shape mismatch for %s: exon=(%llu,%llu) intron=(%llu,%llu)\n",
                     embryo->name,
                     (unsigned long long) exon_hdr.rows,
                     (unsigned long long) exon_hdr.cols,
                     (unsigned long long) intron_hdr.rows,
                     (unsigned long long) intron_hdr.cols);
        goto done;
    }

    if (!u64_to_value(exon_hdr.cols, &embryo->cells, "cell count", exon_matrix_path)) goto done;
    if (!u64_to_value(exon_hdr.rows, &embryo->gene_count, "gene count", exon_matrix_path)) goto done;
    clear_part_plan(&embryo->plan);
    exon_counts = (U64 *) std::calloc((std::size_t) embryo->cells, sizeof(U64));
    intron_counts = (U64 *) std::calloc((std::size_t) embryo->cells, sizeof(U64));
    if (exon_counts == 0 || intron_counts == 0) goto done;
    if (!count_cells_from_matrix(exon_matrix_path, &exon_hdr, exon_counts, 0, 0)) goto done;
    if (!count_cells_from_matrix(intron_matrix_path, &intron_hdr, intron_counts, 0, 0)) goto done;
    if (!build_common_part_plan(embryo->name,
                                (matrix::Index) embryo->cells,
                                max_rows_per_part,
                                exon_counts,
                                intron_counts,
                                &embryo->plan)) goto done;
    embryo->part_count = embryo->plan.count;
    ok = 1;

done:
    std::free(exon_counts);
    std::free(intron_counts);
    return ok;
}

static int stat_file_bytes(const char *path, U64 *bytes_out) {
    struct stat st;
    if (stat(path, &st) != 0) {
        std::fprintf(stderr, "Error: stat failed for %s: %s\n", path, std::strerror(errno));
        return 0;
    }
    if (st.st_size < 0) {
        std::fprintf(stderr, "Error: negative size for %s\n", path);
        return 0;
    }
    *bytes_out = (U64) st.st_size;
    return 1;
}

static int prepare_verify_metadata(const char *base_dir,
                                   embryo_entry *embryos,
                                   matrix::Index embryo_count,
                                   matrix::Index *gene_count_out) {
    matrix::Index embryo_i = 0;
    matrix::Index gene_count = 0;
    int have_gene_count = 0;
    long global_cell_begin = 0;

    for (embryo_i = 0; embryo_i < embryo_count; ++embryo_i) {
        char exon_matrix_path[4096];
        char intron_matrix_path[4096];
        mm_header exon_hdr;
        mm_header intron_hdr;

        if (!build_path3(exon_matrix_path, sizeof(exon_matrix_path), base_dir, embryos[embryo_i].name, "exon/matrix.mtx")) return 0;
        if (!build_path3(intron_matrix_path, sizeof(intron_matrix_path), base_dir, embryos[embryo_i].name, "intron/matrix.mtx")) return 0;
        if (!read_matrix_header_only(exon_matrix_path, &exon_hdr)) return 0;
        if (!read_matrix_header_only(intron_matrix_path, &intron_hdr)) return 0;
        if (exon_hdr.rows != intron_hdr.rows || exon_hdr.cols != intron_hdr.cols) {
            std::fprintf(stderr,
                         "Error: exon/intron shape mismatch for %s: exon=(%llu,%llu) intron=(%llu,%llu)\n",
                         embryos[embryo_i].name,
                         (unsigned long long) exon_hdr.rows,
                         (unsigned long long) exon_hdr.cols,
                         (unsigned long long) intron_hdr.rows,
                         (unsigned long long) intron_hdr.cols);
            return 0;
        }
        if (!u64_to_value(exon_hdr.cols, &embryos[embryo_i].cells, "cell count", exon_matrix_path)) return 0;
        if (!u64_to_value(exon_hdr.rows, &embryos[embryo_i].gene_count, "gene count", exon_matrix_path)) return 0;
        if (!stat_file_bytes(exon_matrix_path, &embryos[embryo_i].exon_file_bytes)) return 0;
        if (!stat_file_bytes(intron_matrix_path, &embryos[embryo_i].intron_file_bytes)) return 0;
        embryos[embryo_i].global_cell_begin = global_cell_begin;
        embryos[embryo_i].part_begin = 0;
        embryos[embryo_i].part_count = 0;
        global_cell_begin += embryos[embryo_i].cells;

        if (!have_gene_count) {
            gene_count = embryos[embryo_i].gene_count;
            have_gene_count = 1;
        } else if (embryos[embryo_i].gene_count != gene_count) {
            std::fprintf(stderr,
                         "Error: gene count mismatch for %s: expected %u, got %u\n",
                         embryos[embryo_i].name,
                         (unsigned int) gene_count,
                         (unsigned int) embryos[embryo_i].gene_count);
            return 0;
        }
    }

    *gene_count_out = gene_count;
    return 1;
}

static int prepare_embryo_metadata(const char *base_dir,
                                   embryo_entry *embryos,
                                   matrix::Index embryo_count,
                                   matrix::Index *gene_count_out) {
    matrix::Index embryo_i = 0;
    matrix::Index gene_count = 0;
    int have_gene_count = 0;

    for (embryo_i = 0; embryo_i < embryo_count; ++embryo_i) {
        char exon_matrix_path[4096];
        char intron_matrix_path[4096];
        mapped_file m;
        mm_header exon_hdr;
        mm_header intron_hdr;

        if (!build_path3(exon_matrix_path, sizeof(exon_matrix_path), base_dir, embryos[embryo_i].name, "exon/matrix.mtx")) return 0;
        if (!build_path3(intron_matrix_path, sizeof(intron_matrix_path), base_dir, embryos[embryo_i].name, "intron/matrix.mtx")) return 0;
        if (!read_matrix_header_only(exon_matrix_path, &exon_hdr)) return 0;
        if (!read_matrix_header_only(intron_matrix_path, &intron_hdr)) return 0;
        if (exon_hdr.rows != intron_hdr.rows || exon_hdr.cols != intron_hdr.cols) {
            std::fprintf(stderr,
                         "Error: exon/intron shape mismatch for %s: exon=(%llu,%llu) intron=(%llu,%llu)\n",
                         embryos[embryo_i].name,
                         (unsigned long long) exon_hdr.rows,
                         (unsigned long long) exon_hdr.cols,
                         (unsigned long long) intron_hdr.rows,
                         (unsigned long long) intron_hdr.cols);
            return 0;
        }
        if (!u64_to_value(exon_hdr.cols, &embryos[embryo_i].cells, "cell count", exon_matrix_path)) return 0;
        if (!u64_to_value(exon_hdr.rows, &embryos[embryo_i].gene_count, "gene count", exon_matrix_path)) return 0;
        if (!have_gene_count) {
            gene_count = embryos[embryo_i].gene_count;
            have_gene_count = 1;
        } else if (embryos[embryo_i].gene_count != gene_count) {
            std::fprintf(stderr,
                         "Error: gene count mismatch for %s: expected %u, got %u\n",
                         embryos[embryo_i].name,
                         (unsigned int) gene_count,
                         (unsigned int) embryos[embryo_i].gene_count);
            return 0;
        }

        std::free(embryos[embryo_i].exon_counts);
        std::free(embryos[embryo_i].intron_counts);
        std::free(embryos[embryo_i].exon_offsets);
        std::free(embryos[embryo_i].intron_offsets);
        embryos[embryo_i].exon_counts = (U64 *) std::calloc((std::size_t) embryos[embryo_i].cells, sizeof(U64));
        embryos[embryo_i].intron_counts = (U64 *) std::calloc((std::size_t) embryos[embryo_i].cells, sizeof(U64));
        embryos[embryo_i].exon_offsets = (U64 *) std::calloc((std::size_t) embryos[embryo_i].cells + 1, sizeof(U64));
        embryos[embryo_i].intron_offsets = (U64 *) std::calloc((std::size_t) embryos[embryo_i].cells + 1, sizeof(U64));
        if (embryos[embryo_i].exon_counts == 0 || embryos[embryo_i].intron_counts == 0 ||
            embryos[embryo_i].exon_offsets == 0 || embryos[embryo_i].intron_offsets == 0) {
            return 0;
        }

        mapped_file_init(&m);
        if (!mapped_file_open(exon_matrix_path, &m)) return 0;
        embryos[embryo_i].exon_file_bytes = (U64) m.bytes;
        mapped_file_close(&m);
        if (!mapped_file_open(intron_matrix_path, &m)) return 0;
        embryos[embryo_i].intron_file_bytes = (U64) m.bytes;
        mapped_file_close(&m);
    }

    *gene_count_out = gene_count;
    return 1;
}

static int build_modality_tasks(const embryo_entry *embryos,
                                matrix::Index embryo_count,
                                modality_task **out_tasks,
                                matrix::Index *out_task_count) {
    modality_task *tasks = 0;
    matrix::Index task_i = 0;
    matrix::Index embryo_i = 0;

    *out_tasks = 0;
    *out_task_count = 0;
    tasks = (modality_task *) std::calloc((std::size_t) embryo_count * 2u, sizeof(modality_task));
    if (tasks == 0) return 0;
    for (embryo_i = 0; embryo_i < embryo_count; ++embryo_i) {
        tasks[task_i].embryo_i = embryo_i;
        tasks[task_i].is_exon = 1;
        tasks[task_i].work_bytes = embryos[embryo_i].exon_file_bytes;
        ++task_i;
        tasks[task_i].embryo_i = embryo_i;
        tasks[task_i].is_exon = 0;
        tasks[task_i].work_bytes = embryos[embryo_i].intron_file_bytes;
        ++task_i;
    }
    qsort(tasks, (std::size_t) task_i, sizeof(modality_task), modality_task_compare);
    *out_tasks = tasks;
    *out_task_count = task_i;
    return 1;
}

static int build_convert_tasks(const embryo_entry *embryos,
                               matrix::Index embryo_count,
                               U64 target_chunk_bytes,
                               convert_task **out_tasks,
                               matrix::Index *out_task_count) {
    convert_task *tasks = 0;
    matrix::Index capacity = 0;
    matrix::Index count = 0;
    matrix::Index embryo_i = 0;

    *out_tasks = 0;
    *out_task_count = 0;

    for (embryo_i = 0; embryo_i < embryo_count; ++embryo_i) {
        capacity += embryos[embryo_i].part_count * 2u;
    }
    tasks = (convert_task *) std::calloc((std::size_t) capacity, sizeof(convert_task));
    if (tasks == 0) return 0;

    for (embryo_i = 0; embryo_i < embryo_count; ++embryo_i) {
        int modality = 0;
        for (modality = 0; modality < 2; ++modality) {
            const U64 *cell_offsets = modality ? embryos[embryo_i].exon_offsets : embryos[embryo_i].intron_offsets;
            const U64 source_bytes = modality ? embryos[embryo_i].exon_file_bytes : embryos[embryo_i].intron_file_bytes;
            const part_plan *plan = &embryos[embryo_i].plan;
            matrix::Index begin = 0;

            while (begin < plan->count) {
                matrix::Index end = begin;
                U64 start_byte = cell_offsets[plan->cell_begin[begin]];
                U64 end_byte = start_byte;

                do {
                    ++end;
                    if (end < plan->count) end_byte = cell_offsets[plan->cell_begin[end]];
                    else end_byte = cell_offsets[embryos[embryo_i].cells];
                } while (end < plan->count && end_byte - start_byte < target_chunk_bytes);

                tasks[count].embryo_i = embryo_i;
                tasks[count].is_exon = modality;
                tasks[count].part_begin = begin;
                tasks[count].part_end = end;
                tasks[count].source_bytes = source_bytes;
                tasks[count].work_bytes = end_byte - start_byte;
                ++count;
                begin = end;
            }
        }
    }

    qsort(tasks, (std::size_t) count, sizeof(convert_task), convert_task_compare);
    *out_tasks = tasks;
    *out_task_count = count;
    return 1;
}

static void accumulate_verify_stats(verify_stats *dst, const verify_stats *src) {
    dst->samples += src->samples;
    dst->exact_cast += src->exact_cast;
    dst->rounded_cast += src->rounded_cast;
    dst->overflow_cast += src->overflow_cast;
    dst->pipeline_match += src->pipeline_match;
    dst->pipeline_mismatch += src->pipeline_mismatch;
    dst->missing_nonzero += src->missing_nonzero;
    dst->finite_samples += src->finite_samples;
    dst->rel_le_001pct += src->rel_le_001pct;
    dst->rel_le_01pct += src->rel_le_01pct;
    dst->rel_le_1pct += src->rel_le_1pct;
    dst->rel_le_5pct += src->rel_le_5pct;
    dst->rel_gt_5pct += src->rel_gt_5pct;
    dst->mean_abs_err_sum += src->mean_abs_err_sum;
    dst->mean_rel_err_sum += src->mean_rel_err_sum;
    if (src->max_abs_err > dst->max_abs_err) dst->max_abs_err = src->max_abs_err;
    if (src->max_rel_err > dst->max_rel_err) dst->max_rel_err = src->max_rel_err;
}

static int build_verify_tasks(const embryo_entry *embryos,
                              matrix::Index embryo_count,
                              int num_tasks_per_modality,
                              U64 chunk_bytes,
                              convert_task **out_tasks,
                              matrix::Index *out_count) {
    convert_task *all_tasks = 0;
    convert_task *picked = 0;
    matrix::Index all_count = 0;
    matrix::Index pick_count = 0;
    matrix::Index i = 0;
    U64 *prefix_exon = 0;
    U64 *prefix_intron = 0;
    matrix::Index *exon_ids = 0;
    matrix::Index *intron_ids = 0;
    matrix::Index exon_count = 0;
    matrix::Index intron_count = 0;
    U64 exon_total = 0;
    U64 intron_total = 0;
    std::uint64_t rng = 0x1234fedcba987654ull;

    *out_tasks = 0;
    *out_count = 0;
    if (!build_convert_tasks(embryos, embryo_count, chunk_bytes, &all_tasks, &all_count)) return 0;
    prefix_exon = (U64 *) std::calloc((std::size_t) all_count, sizeof(U64));
    prefix_intron = (U64 *) std::calloc((std::size_t) all_count, sizeof(U64));
    exon_ids = (matrix::Index *) std::calloc((std::size_t) all_count, sizeof(matrix::Index));
    intron_ids = (matrix::Index *) std::calloc((std::size_t) all_count, sizeof(matrix::Index));
    picked = (convert_task *) std::calloc((std::size_t) (num_tasks_per_modality * 2), sizeof(convert_task));
    if (prefix_exon == 0 || prefix_intron == 0 || exon_ids == 0 || intron_ids == 0 || picked == 0) goto fail;

    for (i = 0; i < all_count; ++i) {
        if (all_tasks[i].is_exon) {
            exon_total += all_tasks[i].work_bytes;
            exon_ids[exon_count] = i;
            prefix_exon[exon_count++] = exon_total;
        } else {
            intron_total += all_tasks[i].work_bytes;
            intron_ids[intron_count] = i;
            prefix_intron[intron_count++] = intron_total;
        }
    }
    if (exon_total == 0 || intron_total == 0) goto fail;

    for (i = 0; i < (matrix::Index) num_tasks_per_modality; ++i) {
        U64 pick = next_rng(&rng) % exon_total;
        matrix::Index idx = 0;
        while (idx + 1 < exon_count && pick >= prefix_exon[idx]) ++idx;
        picked[pick_count++] = all_tasks[exon_ids[idx]];
    }

    for (i = 0; i < (matrix::Index) num_tasks_per_modality; ++i) {
        U64 pick = next_rng(&rng) % intron_total;
        matrix::Index idx = 0;
        while (idx + 1 < intron_count && pick >= prefix_intron[idx]) ++idx;
        picked[pick_count++] = all_tasks[intron_ids[idx]];
    }

    qsort(picked, (std::size_t) pick_count, sizeof(convert_task), convert_task_compare);
    *out_tasks = picked;
    *out_count = pick_count;
    std::free(prefix_exon);
    std::free(prefix_intron);
    std::free(exon_ids);
    std::free(intron_ids);
    std::free(all_tasks);
    return 1;

fail:
    std::free(prefix_exon);
    std::free(prefix_intron);
    std::free(exon_ids);
    std::free(intron_ids);
    std::free(all_tasks);
    std::free(picked);
    return 0;
}

static int build_view_layout(CsrSharded *view,
                             const embryo_entry *embryos,
                             matrix::Index embryo_count,
                             matrix::Index gene_count,
                             int is_exon) {
    matrix::Index embryo_i = 0;
    matrix::Index local_part = 0;
    long total_parts = 0;

    matrix::init(view);
    view->cols = gene_count;
    view->format = matrix::format_csr;

    for (embryo_i = 0; embryo_i < embryo_count; ++embryo_i) total_parts += embryos[embryo_i].part_count;
    if (!matrix::reserve_parts(view, total_parts)) return 0;
    view->num_parts = total_parts;

    for (embryo_i = 0; embryo_i < embryo_count; ++embryo_i) {
        for (local_part = 0; local_part < embryos[embryo_i].part_count; ++local_part) {
            const long global_part = embryos[embryo_i].part_begin + local_part;
            view->parts[global_part] = 0;
            view->part_rows[global_part] = embryos[embryo_i].plan.row_count[local_part];
            view->part_nnz[global_part] = is_exon ? embryos[embryo_i].plan.exon_nnz[local_part]
                                                  : embryos[embryo_i].plan.intron_nnz[local_part];
            view->part_aux[global_part] = 0;
        }
    }

    matrix::rebuild_part_offsets(view);
    return matrix::set_shards_to_parts(view);
}

static int cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}



static int bind_task_gpu(matrix::Index task_id, int gpu_count) {
    const int device = gpu_count > 1 ? (int) (task_id % (matrix::Index) gpu_count) : 0;
    return cuda_check(cudaSetDevice(device), "cudaSetDevice");
}

static int sum_task_shape(const embryo_entry *embryo,
                          int is_exon,
                          matrix::Index part_begin,
                          matrix::Index part_end,
                          matrix::Index *rows_out,
                          matrix::Index *nnz_out,
                          matrix::Index *cell_begin_out) {
    const part_plan *plan = &embryo->plan;
    const matrix::Index *nnz_per_part = is_exon ? plan->exon_nnz : plan->intron_nnz;
    matrix::Index rows = 0;
    matrix::Index nnz = 0;
    matrix::Index i = 0;

    if (part_begin >= part_end || part_end > plan->count) return 0;
    *cell_begin_out = plan->cell_begin[part_begin];
    for (i = part_begin; i < part_end; ++i) {
        rows += plan->row_count[i];
        nnz += nnz_per_part[i];
    }
    *rows_out = rows;
    *nnz_out = nnz;
    return 1;
}

static int parse_task_to_pinned_coo(const char *path,
                                    matrix::Index cols_out,
                                    const embryo_entry *embryo,
                                    int is_exon,
                                    matrix::Index task_part_begin,
                                    matrix::Index task_part_end,
                                    matrix::Index *rows_out,
                                    matrix::Index *nnz_out,
                                    matrix::Index *cell_begin_out,
                                    matrix::Index **row_idx_out,
                                    matrix::Index **col_idx_out,
                                    matrix::Real **val_out) {
    const part_plan *plan = &embryo->plan;
    const U64 *cell_offsets = is_exon ? embryo->exon_offsets : embryo->intron_offsets;
    int fd = -1;
    mm_header hdr;
    char *buf = 0;
    std::size_t carry = 0;
    U64 scan_begin = 0;
    U64 scan_end = 0;
    U64 next_read_offset = 0;
    matrix::Index rows = 0;
    matrix::Index nnz = 0;
    matrix::Index task_cell_begin = 0;
    matrix::Index write_pos = 0;
    matrix::Index *row_idx = 0;
    matrix::Index *col_idx = 0;
    matrix::Real *val = 0;
    int ok = 0;

    *row_idx_out = 0;
    *col_idx_out = 0;
    *val_out = 0;
    if (!sum_task_shape(embryo, is_exon, task_part_begin, task_part_end, &rows, &nnz, &task_cell_begin)) return 0;
    *rows_out = rows;
    *nnz_out = nnz;
    *cell_begin_out = task_cell_begin;
    if (nnz == 0) return 1;

    if (!cuda_check(cudaMallocHost((void **) &row_idx, (std::size_t) nnz * sizeof(matrix::Index)), "cudaMallocHost row_idx")) goto done;
    if (!cuda_check(cudaMallocHost((void **) &col_idx, (std::size_t) nnz * sizeof(matrix::Index)), "cudaMallocHost col_idx")) goto done;
    if (!cuda_check(cudaMallocHost((void **) &val, (std::size_t) nnz * sizeof(matrix::Real)), "cudaMallocHost val")) goto done;

    if (!read_matrix_header_only(path, &hdr)) goto done;
    if (hdr.rows != cols_out) {
        std::fprintf(stderr, "Error: gene count mismatch in %s: expected %u, got %llu\n",
                     path,
                     (unsigned int) cols_out,
                     (unsigned long long) hdr.rows);
        goto done;
    }
    scan_begin = hdr.entries_offset + cell_offsets[task_cell_begin];
    if (task_part_end < plan->count) scan_end = hdr.entries_offset + cell_offsets[plan->cell_begin[task_part_end]];
    else scan_end = hdr.entries_offset + cell_offsets[embryo->cells];
    fd = open_readonly_fd(path);
    if (fd < 0) goto done;
#if defined(POSIX_FADV_SEQUENTIAL)
    (void) posix_fadvise(fd, (off_t) scan_begin, (off_t) (scan_end - scan_begin), POSIX_FADV_SEQUENTIAL);
#endif
    next_read_offset = scan_begin;
    buf = (char *) std::malloc(SOURCE_BUFFER_BYTES + SOURCE_TAIL_BYTES + 1u);
    if (buf == 0) goto done;

    while (write_pos < nnz) {
        const U64 chunk_bytes_u64 = next_read_offset < scan_end
            ? ((scan_end - next_read_offset) > (U64) SOURCE_BUFFER_BYTES
                ? (U64) SOURCE_BUFFER_BYTES
                : (scan_end - next_read_offset))
            : 0;
        const std::size_t chunk_bytes = (std::size_t) chunk_bytes_u64;
        const ssize_t got = chunk_bytes != 0
            ? pread(fd, buf + carry, chunk_bytes, (off_t) next_read_offset)
            : 0;
        const std::size_t total = carry + (got > 0 ? (std::size_t) got : 0u);
        std::size_t parse_bytes = total;
        const char *p = buf;
        const char *end = 0;

        if (got < 0) goto done;
        next_read_offset += (got > 0 ? (U64) got : 0ull);
        if (next_read_offset < scan_end) {
            while (parse_bytes != 0 && buf[parse_bytes - 1] != '\n') --parse_bytes;
            if (parse_bytes == 0) {
                std::fprintf(stderr, "Error: source buffer too small while parsing %s\n", path);
                goto done;
            }
        }
        end = buf + parse_bytes;
        while (p < end && write_pos < nnz) {
            U64 gene = 0;
            U64 cell = 0;
            int value = 0;
            skip_empty_and_comment_lines(&p, end);
            if (p >= end) break;
            if (!parse_u64_token(&p, end, &gene) ||
                !parse_u64_token(&p, end, &cell) ||
                !parse_i32_token(&p, end, &value) ||
                !expect_line_end(&p, end)) goto done;
            row_idx[write_pos] = (matrix::Index) (cell - 1 - task_cell_begin);
            col_idx[write_pos] = (matrix::Index) (gene - 1);
            val[write_pos] = matrix::real_from_float((float) value);
            ++write_pos;
        }
        carry = total - parse_bytes;
        if (carry != 0) std::memmove(buf, buf + parse_bytes, carry);
        if (got == 0 && carry == 0) break;
    }
    if (write_pos != nnz) {
        std::fprintf(stderr, "Error: parsed nnz mismatch in %s task [%u,%u): expected %u got %u\n",
                     path,
                     (unsigned int) task_part_begin,
                     (unsigned int) task_part_end,
                     (unsigned int) nnz,
                     (unsigned int) write_pos);
        goto done;
    }
    ok = 1;

done:
    if (fd >= 0) close(fd);
    std::free(buf);
    if (!ok) {
        if (row_idx != 0) cudaFreeHost(row_idx);
        if (col_idx != 0) cudaFreeHost(col_idx);
        if (val != 0) cudaFreeHost(val);
        return 0;
    }
    *row_idx_out = row_idx;
    *col_idx_out = col_idx;
    *val_out = val;
    return 1;
}

static int gpu_build_csr_from_coo(gpu_csr_workspace *ws,
                                  matrix::Index rows,
                                  matrix::Index nnz,
                                  const matrix::Index *host_row_idx,
                                  const matrix::Index *host_col_idx,
                                  const matrix::Real *host_val) {
    int blocks_nnz = 0;
    int blocks_rows = 0;
    if (rows == 0 && nnz == 0) return 1;
    if (nnz == 0) {
        if (!reserve_gpu_csr_workspace(ws, rows, 0)) return 0;
        std::memset(ws->host_row_ptr, 0, (std::size_t) (rows + 1) * sizeof(matrix::Index));
        return 1;
    }

    if (!reserve_gpu_csr_workspace(ws, rows, nnz)) return 0;
    blocks_nnz = (int) ((nnz + 255u) / 256u);
    blocks_rows = (int) ((rows + 255u) / 256u);
    if (blocks_nnz < 1) blocks_nnz = 1;
    if (blocks_rows < 1) blocks_rows = 1;
    if (blocks_nnz > 4096) blocks_nnz = 4096;
    if (blocks_rows > 4096) blocks_rows = 4096;

    if (!cuda_check(cudaMemcpyAsync(ws->d_row_idx,
                                    host_row_idx,
                                    (std::size_t) nnz * sizeof(matrix::Index),
                                    cudaMemcpyHostToDevice,
                                    ws->stream),
                    "copy row_idx async")) return 0;
    if (!cuda_check(cudaMemcpyAsync(ws->d_col_idx,
                                    host_col_idx,
                                    (std::size_t) nnz * sizeof(matrix::Index),
                                    cudaMemcpyHostToDevice,
                                    ws->stream),
                    "copy col_idx async")) return 0;
    if (!cuda_check(cudaMemcpyAsync(ws->d_val,
                                    host_val,
                                    (std::size_t) nnz * sizeof(matrix::Real),
                                    cudaMemcpyHostToDevice,
                                    ws->stream),
                    "copy val async")) return 0;
    if (!cuda_check(cudaMemsetAsync(ws->d_row_ptr, 0, (std::size_t) (rows + 1) * sizeof(matrix::Index), ws->stream), "memset row_ptr async")) return 0;
    csr_count_rows_kernel<<<blocks_nnz, 256, 0, ws->stream>>>(nnz, ws->d_row_idx, ws->d_row_ptr);
    if (!cuda_check(cudaGetLastError(), "csr_count_rows_kernel")) return 0;
    if (!cuda_check(cub::DeviceScan::ExclusiveSum(ws->scan_tmp, ws->scan_capacity, ws->d_row_ptr, ws->d_row_ptr, rows + 1, ws->stream), "cub scan")) return 0;
    csr_init_heads_kernel<<<blocks_rows, 256, 0, ws->stream>>>(rows, ws->d_row_ptr, ws->d_heads);
    if (!cuda_check(cudaGetLastError(), "csr_init_heads_kernel")) return 0;
    csr_scatter_kernel<<<blocks_nnz, 256, 0, ws->stream>>>(nnz, ws->d_row_idx, ws->d_col_idx, ws->d_val, ws->d_heads, ws->d_out_col, ws->d_out_val);
    if (!cuda_check(cudaGetLastError(), "csr_scatter_kernel")) return 0;
    if (!cuda_check(cudaMemcpyAsync(ws->host_row_ptr,
                                    ws->d_row_ptr,
                                    (std::size_t) (rows + 1) * sizeof(matrix::Index),
                                    cudaMemcpyDeviceToHost,
                                    ws->stream),
                    "copy row_ptr back async")) return 0;
    if (!cuda_check(cudaMemcpyAsync(ws->host_col,
                                    ws->d_out_col,
                                    (std::size_t) nnz * sizeof(matrix::Index),
                                    cudaMemcpyDeviceToHost,
                                    ws->stream),
                    "copy col back async")) return 0;
    if (!cuda_check(cudaMemcpyAsync(ws->host_val,
                                    ws->d_out_val,
                                    (std::size_t) nnz * sizeof(matrix::Real),
                                    cudaMemcpyDeviceToHost,
                                    ws->stream),
                    "copy val back async")) return 0;
    return cuda_check(cudaStreamSynchronize(ws->stream), "cudaStreamSynchronize csr build");
}

static int store_csr_task_parts_from_combined(const char *part_prefix,
                                              matrix::Index cols_out,
                                              const embryo_entry *embryo,
                                              int is_exon,
                                              matrix::Index task_part_begin,
                                              matrix::Index task_part_end,
                                              matrix::Index task_rows,
                                              const matrix::Index *task_row_ptr,
                                              const matrix::Index *task_col,
                                              const matrix::Real *task_val) {
    const part_plan *plan = &embryo->plan;
    const matrix::Index *nnz_per_part = is_exon ? plan->exon_nnz : plan->intron_nnz;
    matrix::Index local_row_begin = 0;
    matrix::Index part_i = 0;

    (void) task_rows;
    for (part_i = task_part_begin; part_i < task_part_end; ++part_i) {
        const matrix::Index rows = plan->row_count[part_i];
        const matrix::Index nnz = nnz_per_part[part_i];
        const matrix::Index nnz_begin = task_row_ptr[local_row_begin];
        char path[4096];
        matrix::Index *row_ptr = 0;
        matrix::Index r = 0;

        if (!build_part_path(path, sizeof(path), part_prefix, embryo->part_begin + part_i)) return 0;
        row_ptr = (matrix::Index *) std::malloc((std::size_t) (rows + 1) * sizeof(matrix::Index));
        if (row_ptr == 0) return 0;
        for (r = 0; r <= rows; ++r) row_ptr[r] = task_row_ptr[local_row_begin + r] - nnz_begin;
        if (!matrix::detail::store_csr_raw(path,
                                           rows,
                                           cols_out,
                                           nnz,
                                           row_ptr,
                                           task_col + nnz_begin,
                                           task_val + nnz_begin,
                                           sizeof(matrix::Real))) {
            std::free(row_ptr);
            return 0;
        }
        std::free(row_ptr);
        local_row_begin += rows;
    }
    return 1;
}

static int convert_matrix_to_csr_parts(const char *path,
                                       matrix::Index cols_out,
                                       const char *part_prefix,
                                       const embryo_entry *embryo,
                                       int is_exon,
                                       gpu_csr_workspace *ws,
                                       matrix::Index task_part_begin,
                                       matrix::Index task_part_end) {
    matrix::Index rows = 0;
    matrix::Index nnz = 0;
    matrix::Index cell_begin = 0;
    matrix::Index *host_row_idx = 0;
    matrix::Index *host_col_idx = 0;
    matrix::Real *host_in_val = 0;
    int ok = 0;

    if (task_part_begin >= task_part_end) return 1;
    if (!parse_task_to_pinned_coo(path,
                                  cols_out,
                                  embryo,
                                  is_exon,
                                  task_part_begin,
                                  task_part_end,
                                  &rows,
                                  &nnz,
                                  &cell_begin,
                                  &host_row_idx,
                                  &host_col_idx,
                                  &host_in_val)) goto done;
    if (!gpu_build_csr_from_coo(ws,
                                rows,
                                nnz,
                                host_row_idx,
                                host_col_idx,
                                host_in_val)) goto done;
    if (!store_csr_task_parts_from_combined(part_prefix,
                                            cols_out,
                                            embryo,
                                            is_exon,
                                            task_part_begin,
                                            task_part_end,
                                            rows,
                                            ws->host_row_ptr,
                                            ws->host_col,
                                            ws->host_val)) goto done;
    ok = 1;

done:
    if (host_row_idx != 0) cudaFreeHost(host_row_idx);
    if (host_col_idx != 0) cudaFreeHost(host_col_idx);
    if (host_in_val != 0) cudaFreeHost(host_in_val);
    if (!ok) std::fprintf(stderr, "Error: conversion failed for %s\n", path);
    return ok;
}

static int build_shard_offsets_from_parts(const CsrSharded *view,
                                          long target_rows_per_shard,
                                          long **out_offsets,
                                          long *out_count) {
    long *offsets = 0;
    long shard_count = 0;
    long i = 0;
    long used = 0;
    long rows = 0;

    *out_offsets = 0;
    *out_count = 0;
    if (view->num_parts == 0) return 1;

    offsets = (long *) std::calloc((std::size_t) (view->num_parts + 1), sizeof(long));
    if (offsets == 0) return 0;
    offsets[0] = 0;

    if (target_rows_per_shard == 0) {
        for (i = 0; i <= view->num_parts; ++i) offsets[i] = view->part_offsets[i];
        *out_offsets = offsets;
        *out_count = view->num_parts;
        return 1;
    }

    for (i = 0; i < view->num_parts; ++i) {
        rows = view->part_rows[i];
        if (used != 0 && used + rows > target_rows_per_shard) {
            ++shard_count;
            offsets[shard_count] = view->part_offsets[i];
            used = 0;
        }
        used += rows;
    }
    ++shard_count;
    offsets[shard_count] = view->rows;
    *out_offsets = offsets;
    *out_count = shard_count;
    return 1;
}

static int store_sharded_header_only(const char *path, const CsrSharded *view) {
    return matrix::store_header(path, view);
}

static int count_barcodes(const char *path, long expected) {
    mapped_file m;
    const char *p = 0;
    const char *end = 0;
    long count = 0;
    int ok = 0;

    mapped_file_init(&m);
    if (!mapped_file_open(path, &m)) goto done;
    p = m.base;
    end = m.base + m.bytes;
    while (p < end) {
        while (p < end && *p != '\n') ++p;
        ++count;
        if (p < end) ++p;
    }
    if (count != expected) {
        std::fprintf(stderr,
                     "Error: barcode count mismatch for %s: expected %ld, got %ld\n",
                     path,
                     expected,
                     count);
        goto done;
    }
    ok = 1;

done:
    mapped_file_close(&m);
    return ok;
}

static int save_cell_map(const char *path,
                         const char *base_dir,
                         const embryo_entry *embryos,
                         matrix::Index embryo_count,
                         const CsrSharded *view) {
    std::FILE *fp = 0;
    matrix::Index embryo_i = 0;
    char barcodes_path[4096];
    mapped_file m;
    const char *p = 0;
    const char *line = 0;
    const char *end = 0;
    long local_cell = 0;
    long global_cell = 0;
    long part_id = 0;
    long shard_id = 0;
    int ok = 0;

    fp = std::fopen(path, "wb");
    if (fp == 0) {
        std::fprintf(stderr, "Error: fopen failed for %s: %s\n", path, std::strerror(errno));
        return 0;
    }
    if (std::fprintf(fp, "global_cell\tshard_id\trow_in_shard\tpart_id\trow_in_part\tembryo_id\tembryo\tlocal_cell\tbarcode\n") < 0) goto done;

    for (embryo_i = 0; embryo_i < embryo_count; ++embryo_i) {
        if (!build_path3(barcodes_path, sizeof(barcodes_path), base_dir, embryos[embryo_i].name, "exon/barcodes.tsv")) goto done;
        if (!count_barcodes(barcodes_path, embryos[embryo_i].cells)) goto done;

        mapped_file_init(&m);
        if (!mapped_file_open(barcodes_path, &m)) goto done;
        p = m.base;
        end = m.base + m.bytes;
        local_cell = 0;
        while (p < end) {
            line = p;
            while (p < end && *p != '\n') ++p;
            global_cell = embryos[embryo_i].global_cell_begin + local_cell;
            part_id = matrix::find_part(view, global_cell);
            shard_id = matrix::find_shard(view, global_cell);
            if (part_id >= view->num_parts || shard_id >= view->num_shards) {
                mapped_file_close(&m);
                goto done;
            }
            if (p > line && p[-1] == '\r') {
                if (std::fprintf(fp,
                                 "%ld\t%ld\t%ld\t%ld\t%ld\t%u\t%s\t%ld\t%.*s\n",
                                 global_cell,
                                 shard_id,
                                 global_cell - view->shard_offsets[shard_id],
                                 part_id,
                                 global_cell - view->part_offsets[part_id],
                                 embryos[embryo_i].embryo_id,
                                 embryos[embryo_i].name,
                                 local_cell,
                                 (int) (p - line - 1),
                                 line) < 0) {
                    mapped_file_close(&m);
                    goto done;
                }
            } else {
                if (std::fprintf(fp,
                                 "%ld\t%ld\t%ld\t%ld\t%ld\t%u\t%s\t%ld\t%.*s\n",
                                 global_cell,
                                 shard_id,
                                 global_cell - view->shard_offsets[shard_id],
                                 part_id,
                                 global_cell - view->part_offsets[part_id],
                                 embryos[embryo_i].embryo_id,
                                 embryos[embryo_i].name,
                                 local_cell,
                                 (int) (p - line),
                                 line) < 0) {
                    mapped_file_close(&m);
                    goto done;
                }
            }
            ++local_cell;
            if (p < end) ++p;
        }
        mapped_file_close(&m);
    }

    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    return ok;
}

static int load_verify_part(const char *part_prefix,
                            const CsrSharded *view,
                            long part_id,
                            verify_part_cache *cache) {
    char path[4096];

    if (part_id < 0 || part_id >= view->num_parts) return 0;
    if (cache->part_id == part_id) return 1;
    clear_verify_part_cache(cache);
    if (!build_part_path(path, sizeof(path), part_prefix, part_id)) return 0;
    if (!matrix::load(path, &cache->part)) return 0;
    cache->part_id = part_id;
    return 1;
}

static int open_readonly_fd(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) std::fprintf(stderr, "Error: open failed for %s: %s\n", path, std::strerror(errno));
    return fd;
}

static void clear_verify_requests(verify_request *requests, matrix::Index count) {
    matrix::Index i = 0;
    if (requests == 0) return;
    for (i = 0; i < count; ++i) std::free(requests[i].data);
    std::free(requests);
}

static void update_verify_stats(verify_stats *stats, int original_value, int missing, float stored_value) {
    const float expected_cast = matrix::real_to_float(matrix::real_from_float((float) original_value));
    const float original_float = (float) original_value;

    ++stats->samples;
    if (missing) ++stats->missing_nonzero;
    if (stored_value == expected_cast) ++stats->pipeline_match;
    else ++stats->pipeline_mismatch;

    if (!std::isfinite(expected_cast)) {
        ++stats->overflow_cast;
        return;
    }

    if (expected_cast == original_float) ++stats->exact_cast;
    else ++stats->rounded_cast;

    {
        const double abs_err = std::fabs((double) expected_cast - (double) original_float);
        const double rel_err = abs_err / (std::fabs((double) original_float) > 1.0 ? std::fabs((double) original_float) : 1.0);

        ++stats->finite_samples;
        stats->mean_abs_err_sum += abs_err;
        stats->mean_rel_err_sum += rel_err;
        if (abs_err > stats->max_abs_err) stats->max_abs_err = abs_err;
        if (rel_err > stats->max_rel_err) stats->max_rel_err = rel_err;
        if (rel_err <= 0.00001) ++stats->rel_le_001pct;
        else if (rel_err <= 0.0001) ++stats->rel_le_01pct;
        else if (rel_err <= 0.01) ++stats->rel_le_1pct;
        else if (rel_err <= 0.05) ++stats->rel_le_5pct;
        else ++stats->rel_gt_5pct;
    }
}

static int build_verify_requests(const embryo_entry *embryos,
                                 matrix::Index embryo_count,
                                 int is_exon,
                                 int num_requests,
                                 U64 chunk_bytes,
                                 verify_request **out_requests,
                                 matrix::Index *out_count) {
    verify_request *reqs = 0;
    U64 *prefix = 0;
    U64 total_bytes = 0;
    std::uint64_t rng = 0x9e3779b97f4a7c15ull ^ (is_exon ? 0xa5a5a5a5ull : 0x5a5a5a5aull);
    matrix::Index i = 0;

    *out_requests = 0;
    *out_count = 0;
    if (num_requests <= 0) return 1;
    reqs = (verify_request *) std::calloc((std::size_t) num_requests, sizeof(verify_request));
    prefix = (U64 *) std::calloc((std::size_t) embryo_count, sizeof(U64));
    if (reqs == 0 || prefix == 0) {
        std::free(reqs);
        std::free(prefix);
        return 0;
    }

    for (i = 0; i < embryo_count; ++i) {
        total_bytes += is_exon ? embryos[i].exon_file_bytes : embryos[i].intron_file_bytes;
        prefix[i] = total_bytes;
    }
    if (total_bytes == 0) {
        std::free(reqs);
        std::free(prefix);
        return 0;
    }

    for (i = 0; i < (matrix::Index) num_requests; ++i) {
        U64 pick = next_rng(&rng) % total_bytes;
        matrix::Index embryo_i = 0;
        U64 bytes = 0;
        U64 start = 0;

        while (embryo_i + 1 < embryo_count && pick >= prefix[embryo_i]) ++embryo_i;
        bytes = is_exon ? embryos[embryo_i].exon_file_bytes : embryos[embryo_i].intron_file_bytes;
        if (bytes > chunk_bytes) start = next_rng(&rng) % (bytes - chunk_bytes);
        start &= ~((U64) (256u * 1024u) - 1ull);
        reqs[i].embryo_i = embryo_i;
        reqs[i].is_exon = is_exon;
        reqs[i].start_byte = start;
        reqs[i].end_byte = start + chunk_bytes;
        if (reqs[i].end_byte > bytes) reqs[i].end_byte = bytes;
    }

    qsort(reqs, (std::size_t) num_requests, sizeof(verify_request), verify_request_compare);
    *out_requests = reqs;
    *out_count = (matrix::Index) num_requests;
    std::free(prefix);
    return 1;
}

static int verify_requests_for_modality(const char *base_dir,
                                        const char *part_prefix,
                                        const embryo_entry *embryos,
                                        const verify_request *requests,
                                        matrix::Index request_count,
                                        const CsrSharded *view,
                                        verify_stats *stats) {
    char matrix_path[4096];
    mapped_file current_file;
    matrix::Index req_i = 0;
    matrix::Index current_embryo = (matrix::Index) -1;
    verify_part_cache cache;
    int ok = 0;

    mapped_file_init(&current_file);
    init_verify_part_cache(&cache);
    init_verify_stats(stats);

    for (req_i = 0; req_i < request_count; ++req_i) {
        const verify_request *req = requests + req_i;
        const embryo_entry *embryo = embryos + req->embryo_i;
        mm_header hdr;
        const char *p = 0;
        const char *end = 0;
        const char *scan_end = 0;

        if (current_embryo != req->embryo_i) {
            mapped_file_close(&current_file);
            clear_verify_part_cache(&cache);
            if (!build_path3(matrix_path,
                             sizeof(matrix_path),
                             base_dir,
                             embryo->name,
                             req->is_exon ? "exon/matrix.mtx" : "intron/matrix.mtx")) goto done;
            if (!mapped_file_open(matrix_path, &current_file)) goto done;
            if (!parse_matrix_market_header(matrix_path, &current_file, &hdr)) goto done;
            current_embryo = req->embryo_i;
        } else {
            if (!parse_matrix_market_header(matrix_path, &current_file, &hdr)) goto done;
        }

        p = current_file.base + req->start_byte;
        end = current_file.base + current_file.bytes;
        scan_end = current_file.base + req->end_byte;
        if (p < hdr.entries_begin) p = hdr.entries_begin;
        if (p > hdr.entries_begin) skip_line(&p, end);

        while (p < scan_end) {
            const char *line = p;
            U64 gene = 0;
            U64 cell = 0;
            int value = 0;
            long global_cell = 0;
            long part_id = 0;
            float stored_value = 0.0f;
            const matrix::Real *slot = 0;
            int missing = 0;

            skip_empty_and_comment_lines(&p, end);
            if (p >= scan_end) break;
            line = p;
            if (!parse_u64_token(&p, end, &gene)) goto done;
            if (!parse_u64_token(&p, end, &cell)) goto done;
            if (!parse_i32_token(&p, end, &value)) goto done;
            if (!expect_line_end(&p, end)) goto done;
            if (line >= scan_end) break;

            global_cell = embryo->global_cell_begin + (long) cell - 1;
            part_id = matrix::find_part(view, global_cell);
            if (!load_verify_part(part_prefix, view, part_id, &cache)) goto done;
            slot = matrix::sparse::at(&cache.part,
                                      (matrix::Index) (global_cell - view->part_offsets[part_id]),
                                      (matrix::Index) (gene - 1));
            if (slot == 0) {
                missing = 1;
                stored_value = 0.0f;
            } else {
                stored_value = matrix::real_to_float(*slot);
            }
            update_verify_stats(stats, value, missing, stored_value);
        }
    }

    ok = 1;

done:
    clear_verify_part_cache(&cache);
    mapped_file_close(&current_file);
    return ok;
}

static int write_verify_report(const char *path,
                               const verify_stats *exon_stats,
                               const verify_stats *intron_stats) {
    std::FILE *fp = std::fopen(path, "wb");
    if (fp == 0) return 0;
    std::fprintf(fp, "modality\tsamples\tfinite_samples\texact_cast\trounded_cast\toverflow_cast\tpipeline_match\tpipeline_mismatch\tmissing_nonzero\tmean_abs_err\tmean_rel_err_pct\tmax_abs_err\tmax_rel_err_pct\trel_le_0.001pct\trel_le_0.01pct\trel_le_1pct\trel_le_5pct\trel_gt_5pct\n");
    {
        const verify_stats *stats[2] = { exon_stats, intron_stats };
        const char *name[2] = { "exon", "intron" };
        int i = 0;
        for (i = 0; i < 2; ++i) {
            const double mean_abs = stats[i]->finite_samples != 0 ? stats[i]->mean_abs_err_sum / (double) stats[i]->finite_samples : 0.0;
            const double mean_rel_pct = stats[i]->finite_samples != 0 ? 100.0 * stats[i]->mean_rel_err_sum / (double) stats[i]->finite_samples : 0.0;
            std::fprintf(fp,
                         "%s\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%.9g\t%.9g\t%.9g\t%.9g\t%ld\t%ld\t%ld\t%ld\t%ld\n",
                         name[i],
                         stats[i]->samples,
                         stats[i]->finite_samples,
                         stats[i]->exact_cast,
                         stats[i]->rounded_cast,
                         stats[i]->overflow_cast,
                         stats[i]->pipeline_match,
                         stats[i]->pipeline_mismatch,
                         stats[i]->missing_nonzero,
                         mean_abs,
                         mean_rel_pct,
                         stats[i]->max_abs_err,
                         100.0 * stats[i]->max_rel_err,
                         stats[i]->rel_le_001pct,
                         stats[i]->rel_le_01pct,
                         stats[i]->rel_le_1pct,
                         stats[i]->rel_le_5pct,
                         stats[i]->rel_gt_5pct);
        }
    }
    std::fclose(fp);
    return 1;
}

static void accumulate_gpu_verify_accum(verify_stats *dst, const gpu_verify_accum *src) {
    dst->samples += (long) src->samples;
    dst->exact_cast += (long) src->exact_cast;
    dst->rounded_cast += (long) src->rounded_cast;
    dst->overflow_cast += (long) src->overflow_cast;
    dst->pipeline_match += (long) src->pipeline_match;
    dst->pipeline_mismatch += (long) src->pipeline_mismatch;
    dst->missing_nonzero += (long) src->missing_nonzero;
    dst->finite_samples += (long) src->finite_samples;
    dst->rel_le_001pct += (long) src->rel_le_001pct;
    dst->rel_le_01pct += (long) src->rel_le_01pct;
    dst->rel_le_1pct += (long) src->rel_le_1pct;
    dst->rel_le_5pct += (long) src->rel_le_5pct;
    dst->rel_gt_5pct += (long) src->rel_gt_5pct;
    dst->mean_abs_err_sum += src->mean_abs_err_sum;
    dst->mean_rel_err_sum += src->mean_rel_err_sum;
    if ((double) src->max_abs_err > dst->max_abs_err) dst->max_abs_err = src->max_abs_err;
    if ((double) src->max_rel_err > dst->max_rel_err) dst->max_rel_err = src->max_rel_err;
}

static int gpu_compare_loaded_part(const CsrMatrix *part,
                                   matrix::Index samples,
                                   const matrix::Index *host_rows,
                                   const matrix::Index *host_cols,
                                   const int *host_original,
                                   verify_stats *stats) {
    matrix::Index *d_row_ptr = 0;
    matrix::Index *d_col_idx = 0;
    matrix::Real *d_val = 0;
    matrix::Index *d_rows = 0;
    matrix::Index *d_cols = 0;
    int *d_original = 0;
    gpu_verify_accum *d_stats = 0;
    gpu_verify_accum host_stats;
    int blocks = 0;
    int ok = 0;

    if (samples == 0) return 1;
    std::memset(&host_stats, 0, sizeof(host_stats));
    blocks = (int) ((samples + 255u) / 256u);
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;

    if (!cuda_check(cudaMalloc((void **) &d_row_ptr, (std::size_t) (part->rows + 1) * sizeof(matrix::Index)), "cudaMalloc verify d_row_ptr")) goto done;
    if (!cuda_check(cudaMalloc((void **) &d_col_idx, (std::size_t) part->nnz * sizeof(matrix::Index)), "cudaMalloc verify d_col_idx")) goto done;
    if (!cuda_check(cudaMalloc((void **) &d_val, (std::size_t) part->nnz * sizeof(matrix::Real)), "cudaMalloc verify d_val")) goto done;
    if (!cuda_check(cudaMalloc((void **) &d_rows, (std::size_t) samples * sizeof(matrix::Index)), "cudaMalloc verify d_rows")) goto done;
    if (!cuda_check(cudaMalloc((void **) &d_cols, (std::size_t) samples * sizeof(matrix::Index)), "cudaMalloc verify d_cols")) goto done;
    if (!cuda_check(cudaMalloc((void **) &d_original, (std::size_t) samples * sizeof(int)), "cudaMalloc verify d_original")) goto done;
    if (!cuda_check(cudaMalloc((void **) &d_stats, sizeof(gpu_verify_accum)), "cudaMalloc verify d_stats")) goto done;
    if (!cuda_check(cudaMemcpy(d_row_ptr, part->rowPtr, (std::size_t) (part->rows + 1) * sizeof(matrix::Index), cudaMemcpyHostToDevice), "copy verify row_ptr")) goto done;
    if (!cuda_check(cudaMemcpy(d_col_idx, part->colIdx, (std::size_t) part->nnz * sizeof(matrix::Index), cudaMemcpyHostToDevice), "copy verify col_idx")) goto done;
    if (!cuda_check(cudaMemcpy(d_val, part->val, (std::size_t) part->nnz * sizeof(matrix::Real), cudaMemcpyHostToDevice), "copy verify val")) goto done;
    if (!cuda_check(cudaMemcpy(d_rows, host_rows, (std::size_t) samples * sizeof(matrix::Index), cudaMemcpyHostToDevice), "copy verify rows")) goto done;
    if (!cuda_check(cudaMemcpy(d_cols, host_cols, (std::size_t) samples * sizeof(matrix::Index), cudaMemcpyHostToDevice), "copy verify cols")) goto done;
    if (!cuda_check(cudaMemcpy(d_original, host_original, (std::size_t) samples * sizeof(int), cudaMemcpyHostToDevice), "copy verify original")) goto done;
    if (!cuda_check(cudaMemcpy(d_stats, &host_stats, sizeof(host_stats), cudaMemcpyHostToDevice), "zero verify stats")) goto done;
    verify_csr_samples_kernel<<<blocks, 256>>>(samples, d_row_ptr, d_col_idx, d_val, d_rows, d_cols, d_original, d_stats);
    if (!cuda_check(cudaGetLastError(), "verify_csr_samples_kernel")) goto done;
    if (!cuda_check(cudaMemcpy(&host_stats, d_stats, sizeof(host_stats), cudaMemcpyDeviceToHost), "copy verify stats back")) goto done;
    accumulate_gpu_verify_accum(stats, &host_stats);
    ok = 1;

done:
    if (d_stats != 0) cudaFree(d_stats);
    if (d_original != 0) cudaFree(d_original);
    if (d_cols != 0) cudaFree(d_cols);
    if (d_rows != 0) cudaFree(d_rows);
    if (d_val != 0) cudaFree(d_val);
    if (d_col_idx != 0) cudaFree(d_col_idx);
    if (d_row_ptr != 0) cudaFree(d_row_ptr);
    return ok;
}

static int preload_verify_requests(const char *base_dir,
                                   const embryo_entry *embryos,
                                   verify_request *requests,
                                   matrix::Index request_count) {
    static const std::size_t tail_slop = 4096;
    char matrix_path[4096];
    int current_fd = -1;
    matrix::Index current_embryo = (matrix::Index) -1;
    int current_is_exon = -1;
    matrix::Index req_i = 0;

    for (req_i = 0; req_i < request_count; ++req_i) {
        const embryo_entry *embryo = embryos + requests[req_i].embryo_i;
        const U64 request_bytes = requests[req_i].end_byte > requests[req_i].start_byte
            ? requests[req_i].end_byte - requests[req_i].start_byte
            : 0;
        const std::size_t need = (std::size_t) request_bytes + tail_slop + 1u;
        ssize_t got = 0;

        if (request_bytes == 0) continue;
        if (current_embryo != requests[req_i].embryo_i || current_is_exon != requests[req_i].is_exon) {
            if (current_fd >= 0) close(current_fd);
            if (!build_path3(matrix_path,
                             sizeof(matrix_path),
                             base_dir,
                             embryo->name,
                             requests[req_i].is_exon ? "exon/matrix.mtx" : "intron/matrix.mtx")) goto fail;
            current_fd = open_readonly_fd(matrix_path);
            if (current_fd < 0) goto fail;
#if defined(POSIX_FADV_SEQUENTIAL)
            (void) posix_fadvise(current_fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
            current_embryo = requests[req_i].embryo_i;
            current_is_exon = requests[req_i].is_exon;
        }
        requests[req_i].data = (char *) std::malloc(need);
        if (requests[req_i].data == 0) goto fail;
        got = pread(current_fd, requests[req_i].data, need - 1u, (off_t) requests[req_i].start_byte);
        if (got <= 0) goto fail;
        requests[req_i].data[got] = 0;
        requests[req_i].loaded_bytes = (U64) got;
    }

    if (current_fd >= 0) close(current_fd);
    return 1;

fail:
    if (current_fd >= 0) close(current_fd);
    return 0;
}

static int verify_loaded_requests_modality(const char *label,
                                           const char *part_prefix,
                                           const embryo_entry *embryos,
                                           verify_request *requests,
                                           matrix::Index request_count,
                                           const CsrSharded *view,
                                           int verify_threads,
                                           int gpu_count,
                                           verify_stats *stats) {
    long *part_counts = 0;
    long *part_offsets = 0;
    long *thread_part_counts = 0;
    long *thread_part_bases = 0;
    long *active_part_ids = 0;
    matrix::Index *sample_rows = 0;
    matrix::Index *sample_cols = 0;
    int *sample_original = 0;
    long total_samples = 0;
    long active_count = 0;
    matrix::Index req_i = 0;
    long part_i = 0;
    int ok = 0;
    int failed = 0;
    int progress_done = 0;
    long progress_step = 0;
    double stage_t0 = now_seconds();
    double stage_t1 = 0.0;
    int bucket_threads = 1;
#ifdef _OPENMP
    int thread_cap = verify_threads;
    if (thread_cap < 1) thread_cap = 1;
    if (thread_cap > (int) request_count) thread_cap = (int) request_count;
    if (thread_cap < 1) thread_cap = 1;
    bucket_threads = thread_cap;
#else
    bucket_threads = 1;
#endif

    init_verify_stats(stats);
    std::fprintf(stderr, "verify %s: bucketing %u request windows in RAM\n", label, (unsigned int) request_count);
    part_counts = (long *) std::calloc((std::size_t) view->num_parts, sizeof(long));
    part_offsets = (long *) std::calloc((std::size_t) (view->num_parts + 1), sizeof(long));
    thread_part_counts = (long *) std::calloc((std::size_t) bucket_threads * (std::size_t) view->num_parts, sizeof(long));
    thread_part_bases = (long *) std::calloc((std::size_t) bucket_threads * (std::size_t) view->num_parts, sizeof(long));
    if (part_counts == 0 || part_offsets == 0 || thread_part_counts == 0 || thread_part_bases == 0) goto done;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(bucket_threads) reduction(+:total_samples)
#endif
    for (req_i = 0; req_i < request_count; ++req_i) {
        const verify_request *req = requests + req_i;
        const embryo_entry *embryo = embryos + req->embryo_i;
        const char *p = req->data;
        const char *end = req->data + req->loaded_bytes;
        const char *scan_end = req->data + (req->end_byte - req->start_byte);
        int tid = 0;
        long *local_counts = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        local_counts = thread_part_counts + (std::size_t) tid * (std::size_t) view->num_parts;

        if (req->data == 0) {
            failed = 1;
            continue;
        }
        if (req->start_byte != 0) skip_line(&p, end);
        while (p < scan_end && p < end) {
            const char *line = p;
            U64 gene = 0;
            U64 cell = 0;
            int value = 0;
            long global_cell = 0;
            long part_id = 0;

            (void) gene;
            (void) value;
            skip_empty_and_comment_lines(&p, end);
            if (p >= scan_end || p >= end) break;
            line = p;
            if (!parse_u64_token(&p, end, &gene) ||
                !parse_u64_token(&p, end, &cell) ||
                !parse_i32_token(&p, end, &value) ||
                !expect_line_end(&p, end)) {
                failed = 1;
                break;
            }
            if (line >= scan_end) break;
            global_cell = embryo->global_cell_begin + (long) cell - 1;
            part_id = matrix::find_part(view, global_cell);
            if (part_id < 0 || part_id >= view->num_parts) {
                failed = 1;
                break;
            }
            ++local_counts[part_id];
            ++total_samples;
        }
    }
    if (failed) goto done;

    for (part_i = 0; part_i < view->num_parts; ++part_i) {
        int tid = 0;
        for (tid = 0; tid < bucket_threads; ++tid) {
            part_counts[part_i] += thread_part_counts[(std::size_t) tid * (std::size_t) view->num_parts + (std::size_t) part_i];
        }
    }

    for (part_i = 0; part_i < view->num_parts; ++part_i) {
        if (part_counts[part_i] != 0) ++active_count;
        part_offsets[part_i + 1] = part_offsets[part_i] + part_counts[part_i];
    }
    stage_t1 = now_seconds();
    std::fprintf(stderr,
                 "verify %s: bucket count pass complete, samples=%ld active_parts=%ld in %.2fs\n",
                 label,
                 total_samples,
                 active_count,
                 stage_t1 - stage_t0);
    if (total_samples == 0) {
        ok = 1;
        goto done;
    }

    active_part_ids = (long *) std::malloc((std::size_t) active_count * sizeof(long));
    sample_rows = (matrix::Index *) std::malloc((std::size_t) total_samples * sizeof(matrix::Index));
    sample_cols = (matrix::Index *) std::malloc((std::size_t) total_samples * sizeof(matrix::Index));
    sample_original = (int *) std::malloc((std::size_t) total_samples * sizeof(int));
    if (active_part_ids == 0 || sample_rows == 0 || sample_cols == 0 || sample_original == 0) goto done;

    active_count = 0;
    for (part_i = 0; part_i < view->num_parts; ++part_i) {
        if (part_counts[part_i] != 0) active_part_ids[active_count++] = part_i;
    }

    for (part_i = 0; part_i < view->num_parts; ++part_i) {
        long base = part_offsets[part_i];
        int tid = 0;
        for (tid = 0; tid < bucket_threads; ++tid) {
            const std::size_t idx = (std::size_t) tid * (std::size_t) view->num_parts + (std::size_t) part_i;
            thread_part_bases[idx] = base;
            base += thread_part_counts[idx];
        }
    }

    stage_t0 = now_seconds();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(bucket_threads)
#endif
    for (req_i = 0; req_i < request_count; ++req_i) {
        verify_request *req = requests + req_i;
        const embryo_entry *embryo = embryos + req->embryo_i;
        const char *p = req->data;
        const char *end = req->data + req->loaded_bytes;
        const char *scan_end = req->data + (req->end_byte - req->start_byte);
        int tid = 0;
        long *local_write = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        local_write = thread_part_bases + (std::size_t) tid * (std::size_t) view->num_parts;

        if (req->data == 0) {
            failed = 1;
            continue;
        }
        if (req->start_byte != 0) skip_line(&p, end);
        while (p < scan_end && p < end) {
            const char *line = p;
            U64 gene = 0;
            U64 cell = 0;
            int value = 0;
            long global_cell = 0;
            long part_id = 0;
            long write_at = 0;

            skip_empty_and_comment_lines(&p, end);
            if (p >= scan_end || p >= end) break;
            line = p;
            if (!parse_u64_token(&p, end, &gene) ||
                !parse_u64_token(&p, end, &cell) ||
                !parse_i32_token(&p, end, &value) ||
                !expect_line_end(&p, end)) {
                failed = 1;
                break;
            }
            if (line >= scan_end) break;
            global_cell = embryo->global_cell_begin + (long) cell - 1;
            part_id = matrix::find_part(view, global_cell);
            if (part_id < 0 || part_id >= view->num_parts) {
                failed = 1;
                break;
            }
            write_at = local_write[part_id]++;
            sample_rows[write_at] = (matrix::Index) (global_cell - view->part_offsets[part_id]);
            sample_cols[write_at] = (matrix::Index) (gene - 1);
            sample_original[write_at] = value;
        }
    }
    if (failed) goto done;
    for (req_i = 0; req_i < request_count; ++req_i) {
        std::free(requests[req_i].data);
        requests[req_i].data = 0;
        requests[req_i].loaded_bytes = 0;
    }
    stage_t1 = now_seconds();
    progress_step = active_count / 8;
    if (progress_step < 1) progress_step = 1;
    std::fprintf(stderr,
                 "verify %s: bucket fill complete, samples=%ld active_parts=%ld in %.2fs, raw windows released\n",
                 label,
                 total_samples,
                 active_count,
                 stage_t1 - stage_t0);
    stage_t0 = now_seconds();

#ifdef _OPENMP
#pragma omp parallel num_threads(verify_threads)
    {
        verify_stats local_stats;
        verify_part_cache cache;
        long active_i = 0;
        init_verify_stats(&local_stats);
        init_verify_part_cache(&cache);
        if (!bind_task_gpu((matrix::Index) omp_get_thread_num(), gpu_count)) __atomic_store_n(&failed, 1, __ATOMIC_RELAXED);
#pragma omp for schedule(static)
        for (active_i = 0; active_i < active_count; ++active_i) {
            const long local_part_id = active_part_ids[active_i];
            const long begin = part_offsets[local_part_id];
            const long end = part_offsets[local_part_id + 1];
            if (__atomic_load_n(&failed, __ATOMIC_RELAXED)) continue;
            if (end <= begin) continue;
            if (!load_verify_part(part_prefix, view, local_part_id, &cache) ||
                !gpu_compare_loaded_part(&cache.part,
                                         (matrix::Index) (end - begin),
                                         sample_rows + begin,
                                         sample_cols + begin,
                                         sample_original + begin,
                                         &local_stats)) {
                __atomic_store_n(&failed, 1, __ATOMIC_RELAXED);
            } else {
                const int done_now = __atomic_add_fetch(&progress_done, 1, __ATOMIC_RELAXED);
                if (done_now == active_count || (done_now % progress_step) == 0) {
#pragma omp critical
                    std::fprintf(stderr, "verify %s: checked %d/%ld sampled parts\n", label, done_now, active_count);
                }
            }
        }
#pragma omp critical
        accumulate_verify_stats(stats, &local_stats);
        clear_verify_part_cache(&cache);
    }
#else
    {
        verify_part_cache cache;
        long active_i = 0;
        init_verify_part_cache(&cache);
        if (!bind_task_gpu(0, gpu_count)) goto done;
        for (active_i = 0; active_i < active_count; ++active_i) {
            const long local_part_id = active_part_ids[active_i];
            const long begin = part_offsets[local_part_id];
            const long end = part_offsets[local_part_id + 1];
            if (end <= begin) continue;
            if (!load_verify_part(part_prefix, view, local_part_id, &cache) ||
                !gpu_compare_loaded_part(&cache.part,
                                         (matrix::Index) (end - begin),
                                         sample_rows + begin,
                                         sample_cols + begin,
                                         sample_original + begin,
                                         stats)) {
                clear_verify_part_cache(&cache);
                goto done;
            }
            ++progress_done;
            if (progress_done == active_count || (progress_done % progress_step) == 0) {
                std::fprintf(stderr, "verify %s: checked %d/%ld sampled parts\n", label, progress_done, active_count);
            }
        }
        clear_verify_part_cache(&cache);
    }
#endif
    if (!failed) {
        ok = 1;
        std::fprintf(stderr,
                     "verify %s: sampled part compare complete in %.2fs\n",
                     label,
                     now_seconds() - stage_t0);
    }

done:
    std::free(part_counts);
    std::free(part_offsets);
    std::free(thread_part_counts);
    std::free(thread_part_bases);
    std::free(active_part_ids);
    std::free(sample_rows);
    std::free(sample_cols);
    std::free(sample_original);
    return ok;
}

static int verify_saved_conversion(const char *base_dir,
                                   const char *out_prefix,
                                   const embryo_entry *embryos,
                                   matrix::Index embryo_count,
                                   int num_requests,
                                   U64 chunk_bytes) {
    char exon_header_path[4096];
    char intron_header_path[4096];
    char exon_part_prefix[4096];
    char intron_part_prefix[4096];
    char verify_report_path[4096];
    U64 buffered_bytes = 0;
    verify_request *exon_requests = 0;
    verify_request *intron_requests = 0;
    matrix::Index exon_request_count = 0;
    matrix::Index intron_request_count = 0;
    int verify_threads = 1;
    int gpu_count = 1;
    CsrSharded exon_view;
    CsrSharded intron_view;
    verify_stats exon_stats;
    verify_stats intron_stats;
    int ok = 0;

    matrix::init(&exon_view);
    matrix::init(&intron_view);
    init_verify_stats(&exon_stats);
    init_verify_stats(&intron_stats);
    if (std::snprintf(exon_header_path, sizeof(exon_header_path), "%s.exon.header.bin", out_prefix) <= 0) goto done;
    if (std::snprintf(intron_header_path, sizeof(intron_header_path), "%s.intron.header.bin", out_prefix) <= 0) goto done;
    if (std::snprintf(exon_part_prefix, sizeof(exon_part_prefix), "%s.exon.part", out_prefix) <= 0) goto done;
    if (std::snprintf(intron_part_prefix, sizeof(intron_part_prefix), "%s.intron.part", out_prefix) <= 0) goto done;
    if (std::snprintf(verify_report_path, sizeof(verify_report_path), "%s.verify.tsv", out_prefix) <= 0) goto done;
    gpu_count = choose_gpu_count();
    if (!matrix::load_header(exon_header_path, &exon_view)) goto done;
    if (!matrix::load_header(intron_header_path, &intron_view)) goto done;
    if (!build_verify_requests(embryos, embryo_count, 1, num_requests, chunk_bytes, &exon_requests, &exon_request_count)) goto done;
    if (!build_verify_requests(embryos, embryo_count, 0, num_requests, chunk_bytes, &intron_requests, &intron_request_count)) goto done;
    verify_threads = choose_worker_threads((long) (exon_view.num_parts > intron_view.num_parts ? exon_view.num_parts : intron_view.num_parts),
                                           (long long) chunk_bytes + (128ll << 20),
                                           48);
    std::fprintf(stderr,
                 "verify: requests_per_modality=%d chunk_bytes=%llu verify_threads=%d gpu_count=%d\n",
                 num_requests,
                 (unsigned long long) chunk_bytes,
                 verify_threads,
                 gpu_count);
    std::fprintf(stderr, "verify exon: preload begin\n");
    if (!preload_verify_requests(base_dir, embryos, exon_requests, exon_request_count)) goto done;
    buffered_bytes = 0;
    for (matrix::Index i = 0; i < exon_request_count; ++i) buffered_bytes += exon_requests[i].loaded_bytes;
    std::fprintf(stderr,
                 "verify exon: preload complete, buffered_bytes=%llu\n",
                 (unsigned long long) buffered_bytes);
    if (!verify_loaded_requests_modality("exon",
                                         exon_part_prefix,
                                         embryos,
                                         exon_requests,
                                         exon_request_count,
                                         &exon_view,
                                         verify_threads,
                                         gpu_count,
                                         &exon_stats)) goto done;
    clear_verify_requests(exon_requests, exon_request_count);
    exon_requests = 0;
    exon_request_count = 0;
    std::fprintf(stderr, "verify intron: preload begin\n");
    if (!preload_verify_requests(base_dir, embryos, intron_requests, intron_request_count)) goto done;
    buffered_bytes = 0;
    for (matrix::Index i = 0; i < intron_request_count; ++i) buffered_bytes += intron_requests[i].loaded_bytes;
    std::fprintf(stderr,
                 "verify intron: preload complete, buffered_bytes=%llu\n",
                 (unsigned long long) buffered_bytes);
    if (!verify_loaded_requests_modality("intron",
                                         intron_part_prefix,
                                         embryos,
                                         intron_requests,
                                         intron_request_count,
                                         &intron_view,
                                         verify_threads,
                                         gpu_count,
                                         &intron_stats)) goto done;
    clear_verify_requests(intron_requests, intron_request_count);
    intron_requests = 0;
    intron_request_count = 0;
    if (!write_verify_report(verify_report_path, &exon_stats, &intron_stats)) goto done;

    std::printf("verify exon: samples=%ld overflow=%ld pipeline_mismatch=%ld mean_rel_err_pct=%.6f max_rel_err_pct=%.6f\n",
                exon_stats.samples,
                exon_stats.overflow_cast,
                exon_stats.pipeline_mismatch,
                exon_stats.finite_samples != 0 ? 100.0 * exon_stats.mean_rel_err_sum / (double) exon_stats.finite_samples : 0.0,
                100.0 * exon_stats.max_rel_err);
    std::printf("verify intron: samples=%ld overflow=%ld pipeline_mismatch=%ld mean_rel_err_pct=%.6f max_rel_err_pct=%.6f\n",
                intron_stats.samples,
                intron_stats.overflow_cast,
                intron_stats.pipeline_mismatch,
                intron_stats.finite_samples != 0 ? 100.0 * intron_stats.mean_rel_err_sum / (double) intron_stats.finite_samples : 0.0,
                100.0 * intron_stats.max_rel_err);
    std::printf("verify report: %s\n", verify_report_path);
    std::printf("verify_threads: %d\n", verify_threads);
    std::printf("verify_gpu_count: %d\n", gpu_count);
    ok = 1;

done:
    clear_verify_requests(exon_requests, exon_request_count);
    clear_verify_requests(intron_requests, intron_request_count);
    clear_csr_sharded(&exon_view);
    clear_csr_sharded(&intron_view);
    return ok;
}

static int choose_worker_threads(long work_items, long long bytes_per_worker, int hard_cap) {
    int threads = 1;

#ifdef _OPENMP
    long phys_pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    long long phys_bytes = 0;
    int mem_cap = 1;
    int cpu_cap = omp_get_max_threads();

    if (phys_pages > 0 && page_size > 0) phys_bytes = (long long) phys_pages * (long long) page_size;
    if (phys_bytes > 0) {
        phys_bytes = (phys_bytes * 9ll) / 10ll;
        mem_cap = (int) (phys_bytes / bytes_per_worker);
        if (mem_cap < 1) mem_cap = 1;
    }
    threads = cpu_cap;
    if (threads > mem_cap) threads = mem_cap;
    if (threads > hard_cap) threads = hard_cap;
    if (threads > work_items) threads = (int) work_items;
    if (threads < 1) threads = 1;
#else
    (void) work_items;
    (void) bytes_per_worker;
    (void) hard_cap;
#endif
    return threads;
}

static int process_all_embryos(const char *base_dir,
                               const char *out_prefix,
                               matrix::Index max_rows_per_part,
                               matrix::Index target_rows_per_shard) {
    embryo_entry *embryos = 0;
    modality_task *count_tasks = 0;
    convert_task *convert_tasks = 0;
    matrix::Index embryo_count = 0;
    matrix::Index embryo_i = 0;
    matrix::Index count_task_count = 0;
    matrix::Index convert_task_count = 0;
    long total_cells = 0;
    long total_parts = 0;
    matrix::Index gene_count = 0;
    int planner_threads = 1;
    int converter_threads = 1;
    int gpu_count = 1;
    CsrSharded exon_view;
    CsrSharded intron_view;
    long *shard_offsets = 0;
    long shard_count = 0;
    char exon_part_prefix[4096];
    char intron_part_prefix[4096];
    char exon_header_path[4096];
    char intron_header_path[4096];
    char cell_map_path[4096];
    U64 convert_chunk_bytes = 128ull << 20;
    int failed = 0;
    int planned_done = 0;
    int converted_done = 0;
    int ok = 0;

    matrix::init(&exon_view);
    matrix::init(&intron_view);

    if (!list_embryos(base_dir, &embryos, &embryo_count)) goto done;
    gpu_count = choose_gpu_count();
    if (!prepare_embryo_metadata(base_dir, embryos, embryo_count, &gene_count)) goto done;
    if (!build_modality_tasks(embryos, embryo_count, &count_tasks, &count_task_count)) goto done;
    planner_threads = choose_worker_threads((long) count_task_count, 512ll << 20, 96);
    std::fprintf(stderr,
                 "plan: matrix_files=%u planner_threads=%d gpu_count=%d\n",
                 (unsigned int) count_task_count,
                 planner_threads,
                 gpu_count);

    if (std::snprintf(exon_part_prefix, sizeof(exon_part_prefix), "%s.exon.part", out_prefix) <= 0) goto done;
    if (std::snprintf(intron_part_prefix, sizeof(intron_part_prefix), "%s.intron.part", out_prefix) <= 0) goto done;
    if (std::snprintf(exon_header_path, sizeof(exon_header_path), "%s.exon.header.bin", out_prefix) <= 0) goto done;
    if (std::snprintf(intron_header_path, sizeof(intron_header_path), "%s.intron.header.bin", out_prefix) <= 0) goto done;
    if (std::snprintf(cell_map_path, sizeof(cell_map_path), "%s.cells.tsv", out_prefix) <= 0) goto done;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(planner_threads)
#endif
    for (embryo_i = 0; embryo_i < count_task_count; ++embryo_i) {
        const modality_task *task = count_tasks + embryo_i;
        embryo_entry *embryo = embryos + task->embryo_i;
        char matrix_path[4096];
        U64 file_bytes = 0;
        int task_ok = 0;

        if (__atomic_load_n(&failed, __ATOMIC_RELAXED)) continue;
        if (!build_path3(matrix_path,
                         sizeof(matrix_path),
                         base_dir,
                         embryo->name,
                         task->is_exon ? "exon/matrix.mtx" : "intron/matrix.mtx")) {
            __atomic_store_n(&failed, 1, __ATOMIC_RELAXED);
            continue;
        }
        task_ok = count_cells_from_matrix(matrix_path,
                                          0,
                                          task->is_exon ? embryo->exon_counts : embryo->intron_counts,
                                          task->is_exon ? embryo->exon_offsets : embryo->intron_offsets,
                                          &file_bytes);
        if (!task_ok) {
            __atomic_store_n(&failed, 1, __ATOMIC_RELAXED);
        } else {
            const int done_now = __atomic_add_fetch(&planned_done, 1, __ATOMIC_RELAXED);
            if (task->is_exon) embryo->exon_file_bytes = file_bytes;
            else embryo->intron_file_bytes = file_bytes;
            if (done_now == count_task_count || (done_now & 7) == 0) {
#pragma omp critical
                std::fprintf(stderr, "planned %d/%u matrix files\n", done_now, (unsigned int) count_task_count);
            }
        }
    }
    if (failed) goto done;

    for (embryo_i = 0; embryo_i < embryo_count; ++embryo_i) {
        if (!build_common_part_plan(embryos[embryo_i].name,
                                    (matrix::Index) embryos[embryo_i].cells,
                                    max_rows_per_part,
                                    embryos[embryo_i].exon_counts,
                                    embryos[embryo_i].intron_counts,
                                    &embryos[embryo_i].plan)) goto done;
        embryos[embryo_i].part_count = embryos[embryo_i].plan.count;
        embryos[embryo_i].global_cell_begin = total_cells;
        embryos[embryo_i].part_begin = total_parts;
        total_cells += embryos[embryo_i].cells;
        total_parts += embryos[embryo_i].part_count;
        std::free(embryos[embryo_i].exon_counts);
        std::free(embryos[embryo_i].intron_counts);
        embryos[embryo_i].exon_counts = 0;
        embryos[embryo_i].intron_counts = 0;
    }

    if (!build_view_layout(&exon_view, embryos, embryo_count, gene_count, 1)) goto done;
    if (!build_view_layout(&intron_view, embryos, embryo_count, gene_count, 0)) goto done;
    if (!build_convert_tasks(embryos, embryo_count, convert_chunk_bytes, &convert_tasks, &convert_task_count)) goto done;
    converter_threads = choose_worker_threads((long) convert_task_count, 768ll << 20, gpu_count * 12);
    std::fprintf(stderr,
                 "convert: tasks=%u chunk_bytes=%llu converter_threads=%d gpu_count=%d\n",
                 (unsigned int) convert_task_count,
                 (unsigned long long) convert_chunk_bytes,
                 converter_threads,
                 gpu_count);

    failed = 0;
#ifdef _OPENMP
#pragma omp parallel num_threads(converter_threads)
    {
        gpu_csr_workspace ws;
        const int tid = omp_get_thread_num();
        const int device = gpu_count > 1 ? (tid % gpu_count) : 0;
        init_gpu_csr_workspace(&ws);
        if (!setup_gpu_csr_workspace(&ws, device)) __atomic_store_n(&failed, 1, __ATOMIC_RELAXED);
#pragma omp for schedule(dynamic, 1)
        for (embryo_i = 0; embryo_i < convert_task_count; ++embryo_i) {
            const convert_task *task = convert_tasks + embryo_i;
            const embryo_entry *embryo = embryos + task->embryo_i;
            char matrix_path[4096];
            const char *part_prefix = task->is_exon ? exon_part_prefix : intron_part_prefix;

            if (__atomic_load_n(&failed, __ATOMIC_RELAXED)) continue;
            if (!build_path3(matrix_path,
                             sizeof(matrix_path),
                             base_dir,
                             embryo->name,
                             task->is_exon ? "exon/matrix.mtx" : "intron/matrix.mtx") ||
                !convert_matrix_to_csr_parts(matrix_path,
                                             gene_count,
                                             part_prefix,
                                             embryo,
                                             task->is_exon,
                                             &ws,
                                             task->part_begin,
                                             task->part_end)) {
                __atomic_store_n(&failed, 1, __ATOMIC_RELAXED);
            } else {
                const int done_now = __atomic_add_fetch(&converted_done, 1, __ATOMIC_RELAXED);
                if (done_now == convert_task_count || (done_now & 31) == 0) {
#pragma omp critical
                    std::fprintf(stderr, "converted %d/%u conversion tasks\n", done_now, (unsigned int) convert_task_count);
                }
            }
        }
        clear_gpu_csr_workspace(&ws);
    }
#else
    {
        gpu_csr_workspace ws;
        init_gpu_csr_workspace(&ws);
        if (!setup_gpu_csr_workspace(&ws, 0)) goto done;
        for (embryo_i = 0; embryo_i < convert_task_count; ++embryo_i) {
            const convert_task *task = convert_tasks + embryo_i;
            const embryo_entry *embryo = embryos + task->embryo_i;
            char matrix_path[4096];
            const char *part_prefix = task->is_exon ? exon_part_prefix : intron_part_prefix;

            if (!build_path3(matrix_path,
                             sizeof(matrix_path),
                             base_dir,
                             embryo->name,
                             task->is_exon ? "exon/matrix.mtx" : "intron/matrix.mtx") ||
                !convert_matrix_to_csr_parts(matrix_path,
                                             gene_count,
                                             part_prefix,
                                             embryo,
                                             task->is_exon,
                                             &ws,
                                             task->part_begin,
                                             task->part_end)) {
                clear_gpu_csr_workspace(&ws);
                goto done;
            }
            ++converted_done;
            if (converted_done == convert_task_count || (converted_done & 31) == 0) {
                std::fprintf(stderr, "converted %d/%u conversion tasks\n", converted_done, (unsigned int) convert_task_count);
            }
        }
        clear_gpu_csr_workspace(&ws);
    }
#endif
    if (failed) goto done;

    if (exon_view.num_parts != intron_view.num_parts || exon_view.rows != intron_view.rows || exon_view.cols != intron_view.cols) {
        std::fprintf(stderr, "Error: exon/intron final metadata mismatch\n");
        goto done;
    }

    if (!build_shard_offsets_from_parts(&exon_view, target_rows_per_shard, &shard_offsets, &shard_count)) goto done;
    if (!matrix::reshard(&exon_view, shard_count, shard_offsets)) goto done;
    if (!matrix::reshard(&intron_view, shard_count, shard_offsets)) goto done;

    if (!store_sharded_header_only(exon_header_path, &exon_view)) goto done;
    if (!store_sharded_header_only(intron_header_path, &intron_view)) goto done;
    if (!save_cell_map(cell_map_path, base_dir, embryos, embryo_count, &exon_view)) goto done;
    if (!verify_saved_conversion(base_dir, out_prefix, embryos, embryo_count, 12, 512ull << 20)) goto done;

    std::printf("exon: rows=%ld cols=%ld nnz=%ld parts=%ld shards=%ld\n",
                exon_view.rows,
                exon_view.cols,
                exon_view.nnz,
                exon_view.num_parts,
                exon_view.num_shards);
    std::printf("intron: rows=%ld cols=%ld nnz=%ld parts=%ld shards=%ld\n",
                intron_view.rows,
                intron_view.cols,
                intron_view.nnz,
                intron_view.num_parts,
                intron_view.num_shards);
    std::printf("cell_map: %s\n", cell_map_path);
    std::printf("reload exon with header %s and shard prefix %s\n", exon_header_path, exon_part_prefix);
    std::printf("reload intron with header %s and shard prefix %s\n", intron_header_path, intron_part_prefix);
    std::printf("planner_threads: %d\n", planner_threads);
    std::printf("converter_threads: %d\n", converter_threads);
    std::printf("gpu_count: %d\n", gpu_count);
    std::printf("convert_chunk_bytes: %llu\n", (unsigned long long) convert_chunk_bytes);
    ok = 1;

done:
    std::free(count_tasks);
    std::free(convert_tasks);
    std::free(shard_offsets);
    clear_embryos(embryos, embryo_count);
    clear_csr_sharded(&exon_view);
    clear_csr_sharded(&intron_view);
    return ok;
}

static int verify_existing_saved_conversion(const char *base_dir,
                                            const char *out_prefix,
                                            int num_requests,
                                            U64 chunk_bytes) {
    embryo_entry *embryos = 0;
    matrix::Index embryo_count = 0;
    matrix::Index gene_count = 0;
    int ok = 0;

    if (!list_embryos(base_dir, &embryos, &embryo_count)) goto done;
    std::fprintf(stderr, "verify-plan: embryos=%u header-only metadata rebuild\n", (unsigned int) embryo_count);
    if (!prepare_verify_metadata(base_dir, embryos, embryo_count, &gene_count)) goto done;
    (void) gene_count;
    if (!verify_saved_conversion(base_dir, out_prefix, embryos, embryo_count, num_requests, chunk_bytes)) goto done;
    ok = 1;

done:
    clear_embryos(embryos, embryo_count);
    return ok;
}

} // namespace

int main(int argc, char **argv) {
    char *end = 0;
    unsigned long part_rows = 0;
    unsigned long shard_rows = 0;
    unsigned long verify_requests = 12;
    unsigned long verify_chunk_mb = 512;

#ifdef _OPENMP
    omp_set_dynamic(0);
#endif

    if (argc >= 2 && std::strcmp(argv[1], "--verify") == 0) {
        if (argc >= 5) {
            errno = 0;
            verify_requests = std::strtoul(argv[4], &end, 10);
            if (errno != 0 || end == argv[4] || *end != 0 || verify_requests == 0) {
                std::fprintf(stderr, "Error: invalid verify_requests: %s\n", argv[4]);
                return 1;
            }
        }
        if (argc >= 6) {
            errno = 0;
            verify_chunk_mb = std::strtoul(argv[5], &end, 10);
            if (errno != 0 || end == argv[5] || *end != 0 || verify_chunk_mb == 0) {
                std::fprintf(stderr, "Error: invalid verify_chunk_mb: %s\n", argv[5]);
                return 1;
            }
        }
        if (argc < 4 || argc > 6) {
            std::fprintf(stderr,
                         "Usage: %s --verify <embryo_base_dir> <out_prefix> [num_requests] [chunk_mb]\n",
                         argv[0]);
            return 1;
        }
        return verify_existing_saved_conversion(argv[2],
                                                argv[3],
                                                (int) verify_requests,
                                                (U64) verify_chunk_mb << 20)
            ? 0
            : 1;
    }

    if (argc != 5) {
        std::fprintf(stderr,
                     "Usage: %s <embryo_base_dir> <out_prefix> <cells_per_part> <target_cells_per_shard>\n"
                     "       %s --verify <embryo_base_dir> <out_prefix> [num_requests] [chunk_mb]\n"
                     "Example: %s /mnt/block/embryo/embryo_exon_intron_mtx /tmp/embryo 4096 65536\n",
                     argv[0],
                     argv[0],
                     argv[0]);
        return 1;
    }

    errno = 0;
    part_rows = std::strtoul(argv[3], &end, 10);
    if (errno != 0 || end == argv[3] || *end != 0 || part_rows == 0 || part_rows > U32_LIMIT) {
        std::fprintf(stderr, "Error: invalid cells_per_part: %s\n", argv[3]);
        return 1;
    }

    errno = 0;
    shard_rows = std::strtoul(argv[4], &end, 10);
    if (errno != 0 || end == argv[4] || *end != 0 || shard_rows > U32_LIMIT) {
        std::fprintf(stderr, "Error: invalid target_cells_per_shard: %s\n", argv[4]);
        return 1;
    }

    return process_all_embryos(argv[1],
                               argv[2],
                               (matrix::Index) part_rows,
                               (matrix::Index) shard_rows)
        ? 0
        : 1;
}
