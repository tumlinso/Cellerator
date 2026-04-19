#include "../extern/CellShard/include/CellShard/CellShard.hh"

#include <hdf5.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <limits>
#include <sys/stat.h>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

struct owned_text_column {
    std::vector<std::uint32_t> offsets;
    std::vector<char> data;

    cellshard::dataset_text_column_view view() const {
        cellshard::dataset_text_column_view out;
        out.count = offsets.empty() ? 0u : (std::uint32_t) offsets.size() - 1u;
        out.bytes = (std::uint32_t) data.size();
        out.offsets = offsets.empty() ? 0 : offsets.data();
        out.data = data.empty() ? 0 : data.data();
        return out;
    }
};

static owned_text_column make_column(const std::vector<const char *> &values) {
    owned_text_column col;
    std::size_t i = 0;
    std::uint32_t cursor = 0;

    col.offsets.resize(values.size() + 1u, 0u);
    for (i = 0; i < values.size(); ++i) {
        const char *value = values[i] != 0 ? values[i] : "";
        const std::size_t len = std::strlen(value);
        col.offsets[i] = cursor;
        col.data.insert(col.data.end(), value, value + len);
        col.data.push_back(0);
        cursor += (std::uint32_t) len + 1u;
    }
    col.offsets[values.size()] = cursor;
    return col;
}

static bool path_exists(const std::string &path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

static bool path_is_dir(const std::string &path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static std::string find_first_named_subdir(const std::string &root) {
    DIR *dir = ::opendir(root.c_str());
    if (dir == nullptr) return std::string();
    for (;;) {
        struct dirent *entry = ::readdir(dir);
        if (entry == nullptr) break;
        const char *name = entry->d_name;
        if (std::strcmp(name, ".") == 0 || std::strcmp(name, "..") == 0) continue;
        const std::string path = root + "/" + name;
        if (path_is_dir(path)) {
            ::closedir(dir);
            return path;
        }
    }
    ::closedir(dir);
    return std::string();
}

static bool any_named_subdir_has_path(const std::string &root, const std::string &relative_path) {
    DIR *dir = ::opendir(root.c_str());
    if (dir == nullptr) return false;
    for (;;) {
        struct dirent *entry = ::readdir(dir);
        if (entry == nullptr) break;
        const char *name = entry->d_name;
        if (std::strcmp(name, ".") == 0 || std::strcmp(name, "..") == 0) continue;
        const std::string path = root + "/" + name;
        if (path_is_dir(path) && path_exists(path + "/" + relative_path)) {
            ::closedir(dir);
            return true;
        }
    }
    ::closedir(dir);
    return false;
}

template<typename Fn>
static auto with_suppressed_stderr(Fn &&fn) -> decltype(fn()) {
    const int saved_fd = ::dup(STDERR_FILENO);
    std::FILE *devnull = std::fopen("/dev/null", "w");
    if (saved_fd < 0 || devnull == nullptr) {
        if (saved_fd >= 0) ::close(saved_fd);
        if (devnull != nullptr) std::fclose(devnull);
        return fn();
    }
    std::fflush(stderr);
    if (::dup2(::fileno(devnull), STDERR_FILENO) < 0) {
        ::close(saved_fd);
        std::fclose(devnull);
        return fn();
    }
    auto result = fn();
    std::fflush(stderr);
    (void) ::dup2(saved_fd, STDERR_FILENO);
    ::close(saved_fd);
    std::fclose(devnull);
    return result;
}

static int overwrite_u32_attr(const std::string &filename,
                              const char *object_path,
                              const char *attr_name,
                              std::uint32_t value) {
    hid_t file = (hid_t) -1;
    hid_t obj = (hid_t) -1;
    hid_t attr = (hid_t) -1;
    int ok = 0;

    file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    obj = H5Oopen(file, object_path, H5P_DEFAULT);
    if (obj < 0) goto done;
    attr = H5Aopen(obj, attr_name, H5P_DEFAULT);
    if (attr < 0) goto done;
    ok = H5Awrite(attr, H5T_NATIVE_UINT32, &value) >= 0;

done:
    if (attr >= 0) H5Aclose(attr);
    if (obj >= 0) H5Oclose(obj);
    if (file >= 0) H5Fclose(file);
    return ok;
}

static int overwrite_u64_attr(const std::string &filename,
                              const char *object_path,
                              const char *attr_name,
                              std::uint64_t value) {
    hid_t file = (hid_t) -1;
    hid_t obj = (hid_t) -1;
    hid_t attr = (hid_t) -1;
    int ok = 0;

    file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    obj = H5Oopen(file, object_path, H5P_DEFAULT);
    if (obj < 0) goto done;
    attr = H5Aopen(obj, attr_name, H5P_DEFAULT);
    if (attr < 0) goto done;
    ok = H5Awrite(attr, H5T_NATIVE_UINT64, &value) >= 0;

done:
    if (attr >= 0) H5Aclose(attr);
    if (obj >= 0) H5Oclose(obj);
    if (file >= 0) H5Fclose(file);
    return ok;
}

static int replace_u64_dataset(const std::string &filename,
                               const char *dataset_path,
                               const std::vector<std::uint64_t> &values) {
    hid_t file = (hid_t) -1;
    hid_t dset = (hid_t) -1;
    hid_t space = (hid_t) -1;
    int ok = 0;
    hsize_t dims[1] = { (hsize_t) values.size() };

    file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    dset = H5Dopen2(file, dataset_path, H5P_DEFAULT);
    if (dset >= 0) H5Dclose(dset);
    if (H5Ldelete(file, dataset_path, H5P_DEFAULT) < 0) goto done;
    space = H5Screate_simple(1, dims, 0);
    if (space < 0) goto done;
    dset = H5Dcreate2(file, dataset_path, H5T_NATIVE_UINT64, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) goto done;
    ok = values.empty() ? 1 : (H5Dwrite(dset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);

done:
    if (dset >= 0) H5Dclose(dset);
    if (space >= 0) H5Sclose(space);
    if (file >= 0) H5Fclose(file);
    return ok;
}

struct owned_observation_metadata_column {
    std::string name;
    std::uint32_t type = cellshard::dataset_observation_metadata_type_none;
    owned_text_column text_values;
    std::vector<float> float32_values;
    std::vector<std::uint8_t> uint8_values;

    cellshard::dataset_observation_metadata_column_view view() const {
        cellshard::dataset_observation_metadata_column_view out{};
        out.name = name.c_str();
        out.type = type;
        out.text_values = text_values.view();
        out.float32_values = float32_values.empty() ? nullptr : float32_values.data();
        out.uint8_values = uint8_values.empty() ? nullptr : uint8_values.data();
        return out;
    }
};

static int populate_blocked_ell_part(cellshard::sparse::blocked_ell *part,
                                     unsigned int rows,
                                     unsigned int cols,
                                     unsigned int block_size,
                                     unsigned int ell_cols,
                                     const std::vector<unsigned int> &block_idx,
                                     const std::vector<float> &values) {
    std::size_t i = 0;

    cellshard::sparse::init(part,
                            rows,
                            cols,
                            (cellshard::types::nnz_t) values.size(),
                            block_size,
                            ell_cols);
    if (!cellshard::sparse::allocate(part)) return 0;
    std::memcpy(part->blockColIdx, block_idx.data(), block_idx.size() * sizeof(unsigned int));
    for (i = 0; i < values.size(); ++i) part->val[i] = __float2half(values[i]);
    return 1;
}

static int check_blocked_ell_part(const cellshard::sparse::blocked_ell *part,
                                  const std::vector<unsigned int> &block_idx,
                                  const std::vector<float> &values) {
    std::size_t i = 0;
    if (part == 0) return 0;
    for (i = 0; i < block_idx.size(); ++i) {
        if (part->blockColIdx[i] != block_idx[i]) return 0;
    }
    for (i = 0; i < values.size(); ++i) {
        if (__half2float(part->val[i]) != values[i]) return 0;
    }
    return 1;
}

static int populate_sliced_ell_part(cellshard::sparse::sliced_ell *part,
                                    unsigned int rows,
                                    unsigned int cols,
                                    const std::vector<unsigned int> &slice_row_offsets,
                                    const std::vector<unsigned int> &slice_widths,
                                    const std::vector<unsigned int> &col_idx,
                                    const std::vector<float> &values,
                                    unsigned int nnz) {
    std::size_t i = 0;
    cellshard::sparse::init(part, rows, cols, nnz);
    if (!cellshard::sparse::allocate(part,
                                     (cellshard::types::u32) slice_widths.size(),
                                     slice_row_offsets.data(),
                                     slice_widths.data())) {
        return 0;
    }
    if (part->slice_count == 0u) return values.empty() && col_idx.empty();
    std::memcpy(part->col_idx, col_idx.data(), col_idx.size() * sizeof(unsigned int));
    for (i = 0; i < values.size(); ++i) part->val[i] = __float2half(values[i]);
    return 1;
}

static int check_sliced_ell_part(const cellshard::sparse::sliced_ell *part,
                                 const std::vector<unsigned int> &slice_row_offsets,
                                 const std::vector<unsigned int> &slice_widths,
                                 const std::vector<unsigned int> &col_idx,
                                 const std::vector<float> &values) {
    std::size_t i = 0;
    if (part == 0 || part->slice_count != slice_widths.size()) return 0;
    for (i = 0; i < slice_row_offsets.size(); ++i) {
        if (part->slice_row_offsets[i] != slice_row_offsets[i]) return 0;
    }
    for (i = 0; i < slice_widths.size(); ++i) {
        if (part->slice_widths[i] != slice_widths[i]) return 0;
    }
    for (i = 0; i < col_idx.size(); ++i) {
        if (part->col_idx[i] != col_idx[i]) return 0;
    }
    for (i = 0; i < values.size(); ++i) {
        if (__half2float(part->val[i]) != values[i]) return 0;
    }
    return 1;
}

static int populate_quantized_blocked_ell_part(cellshard::sparse::quantized_blocked_ell *part,
                                               unsigned int rows,
                                               unsigned int cols,
                                               unsigned int nnz,
                                               unsigned int block_size,
                                               unsigned int ell_cols,
                                               unsigned int bits,
                                               unsigned int decode_policy,
                                               const std::vector<unsigned int> &block_idx,
                                               const std::vector<unsigned char> &packed_values,
                                               const std::vector<float> &column_scales,
                                               const std::vector<float> &column_offsets,
                                               const std::vector<float> &row_offsets) {
    cellshard::sparse::init(part,
                            rows,
                            cols,
                            nnz,
                            block_size,
                            ell_cols,
                            bits,
                            decode_policy,
                            cellshard::sparse::quantized_blocked_ell_aligned_row_bytes(bits, ell_cols));
    if (!cellshard::sparse::allocate(part)) return 0;
    std::memcpy(part->blockColIdx, block_idx.data(), block_idx.size() * sizeof(unsigned int));
    std::memcpy(part->packed_values, packed_values.data(), packed_values.size() * sizeof(unsigned char));
    std::memcpy(part->column_scales, column_scales.data(), column_scales.size() * sizeof(float));
    std::memcpy(part->column_offsets, column_offsets.data(), column_offsets.size() * sizeof(float));
    std::memcpy(part->row_offsets, row_offsets.data(), row_offsets.size() * sizeof(float));
    return 1;
}

static int close_f32(float lhs, float rhs, float tol = 1.0e-5f) {
    return std::fabs(lhs - rhs) <= tol;
}

static int check_quantized_blocked_ell_part(const cellshard::sparse::quantized_blocked_ell *part,
                                            unsigned int bits,
                                            unsigned int decode_policy,
                                            const std::vector<unsigned int> &block_idx,
                                            const std::vector<unsigned char> &packed_values,
                                            const std::vector<float> &column_scales,
                                            const std::vector<float> &column_offsets,
                                            const std::vector<float> &row_offsets) {
    std::size_t i = 0;
    if (part == 0 || part->bits != bits || part->decode_policy != decode_policy) return 0;
    for (i = 0; i < block_idx.size(); ++i) {
        if (part->blockColIdx[i] != block_idx[i]) return 0;
    }
    for (i = 0; i < packed_values.size(); ++i) {
        if (part->packed_values[i] != packed_values[i]) return 0;
    }
    for (i = 0; i < column_scales.size(); ++i) {
        if (!close_f32(part->column_scales[i], column_scales[i])) return 0;
    }
    for (i = 0; i < column_offsets.size(); ++i) {
        if (!close_f32(part->column_offsets[i], column_offsets[i])) return 0;
    }
    for (i = 0; i < row_offsets.size(); ++i) {
        if (!close_f32(part->row_offsets[i], row_offsets[i])) return 0;
    }
    return 1;
}

static int create_small_blocked_ell_dataset(const std::string &out_path) {
    cellshard::sparse::blocked_ell part;
    cellshard::dataset_codec_descriptor codec{};
    cellshard::dataset_layout_view layout{};
    std::vector<std::uint64_t> partition_rows = { 2u };
    std::vector<std::uint64_t> partition_nnz = { 8u };
    std::vector<std::uint64_t> partition_aux = {
        (std::uint64_t) cellshard::sparse::pack_blocked_ell_aux(2u, 2ul)
    };
    std::vector<std::uint64_t> partition_row_offsets = { 0u, 2u };
    std::vector<std::uint32_t> partition_dataset_ids = { 0u };
    std::vector<std::uint32_t> partition_codec_ids = { 0u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 2u };
    int ok = 0;

    std::remove(out_path.c_str());
    cellshard::sparse::init(&part);
    if (!populate_blocked_ell_part(&part,
                                   2u,
                                   4u,
                                   2u,
                                   4u,
                                   {0u, 1u},
                                   {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f})) {
        goto done;
    }

    codec.codec_id = 0u;
    codec.family = cellshard::dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = 2u;
    layout.cols = 4u;
    layout.nnz = 8u;
    layout.num_partitions = 1u;
    layout.num_shards = 1u;
    layout.partition_rows = partition_rows.data();
    layout.partition_nnz = partition_nnz.data();
    layout.partition_axes = 0;
    layout.partition_aux = partition_aux.data();
    layout.partition_row_offsets = partition_row_offsets.data();
    layout.partition_dataset_ids = partition_dataset_ids.data();
    layout.partition_codec_ids = partition_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    if (!cellshard::create_dataset_blocked_ell_h5(out_path.c_str(), &layout, 0, 0)) goto done;
    if (!cellshard::append_blocked_ell_partition_h5(out_path.c_str(), 0u, &part)) goto done;
    ok = 1;

done:
    cellshard::sparse::clear(&part);
    if (!ok) std::remove(out_path.c_str());
    return ok;
}

static int run_blocked_ell_roundtrip_test() {
    const std::string out_path = "/tmp/cellshard_dataset_blocked_ell_test.csh5";
    const std::string cache_root = "/tmp/cellshard_dataset_blocked_ell_cache";
    cellshard::sparse::blocked_ell part0;
    cellshard::sparse::blocked_ell part1;
    cellshard::sharded<cellshard::sparse::blocked_ell> loaded;
    cellshard::shard_storage storage;
    std::vector<std::uint64_t> partition_rows = { 2u, 2u };
    std::vector<std::uint64_t> partition_nnz = { 8u, 8u };
    std::vector<std::uint64_t> partition_aux = {
        (std::uint64_t) cellshard::sparse::pack_blocked_ell_aux(2u, 2ul),
        (std::uint64_t) cellshard::sparse::pack_blocked_ell_aux(2u, 2ul)
    };
    std::vector<std::uint64_t> partition_row_offsets = { 0u, 2u, 4u };
    std::vector<std::uint32_t> partition_dataset_ids = { 0u, 0u };
    std::vector<std::uint32_t> partition_codec_ids = { 0u, 0u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 4u };
    cellshard::dataset_codec_descriptor codec{};
    cellshard::dataset_layout_view layout{};
    int rc = 1;

    std::remove(out_path.c_str());
    if (cache_root.empty()) {
        std::fprintf(stderr, "failed to create unique runtime service cache root\n");
        return 1;
    }
    cellshard::sparse::init(&part0);
    cellshard::sparse::init(&part1);
    cellshard::init(&loaded);
    cellshard::init(&storage);

    if (!populate_blocked_ell_part(&part0,
                                   2u,
                                   4u,
                                   2u,
                                   4u,
                                   {0u, 1u},
                                   {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f})) goto done;
    if (!populate_blocked_ell_part(&part1,
                                   2u,
                                   4u,
                                   2u,
                                   4u,
                                   {1u, 0u},
                                   {5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 8.5f})) goto done;

    codec.codec_id = 0u;
    codec.family = cellshard::dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = 4u;
    layout.cols = 4u;
    layout.nnz = 16u;
    layout.num_partitions = 2u;
    layout.num_shards = 1u;
    layout.partition_rows = partition_rows.data();
    layout.partition_nnz = partition_nnz.data();
    layout.partition_axes = 0;
    layout.partition_aux = partition_aux.data();
    layout.partition_row_offsets = partition_row_offsets.data();
    layout.partition_dataset_ids = partition_dataset_ids.data();
    layout.partition_codec_ids = partition_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    if (!cellshard::create_dataset_blocked_ell_h5(out_path.c_str(), &layout, 0, 0)) goto done;
    if (!cellshard::append_blocked_ell_partition_h5(out_path.c_str(), 0u, &part0)) goto done;
    if (!cellshard::append_blocked_ell_partition_h5(out_path.c_str(), 1u, &part1)) goto done;
    if (!cellshard::warm_dataset_blocked_ell_h5_cache(out_path.c_str(), cache_root.c_str())) {
        std::fprintf(stderr, "failed to warm blocked ell shard cache\n");
        goto done;
    }

    if (!cellshard::load_header(out_path.c_str(), &loaded, &storage)) {
        std::fprintf(stderr, "failed to load blocked ell dataset header\n");
        goto done;
    }
    if (storage.backend != cellshard::shard_storage_backend_dataset_h5) {
        std::fprintf(stderr, "blocked ell storage backend mismatch\n");
        goto done;
    }
    if (!cellshard::bind_dataset_h5_cache(&storage, cache_root.c_str())) {
        std::fprintf(stderr, "failed to bind blocked ell cache dir\n");
        goto done;
    }
    if (!cellshard::fetch_partition(&loaded, &storage, 0u)) {
        std::fprintf(stderr, "failed to fetch blocked ell part 0\n");
        goto done;
    }
    if (!check_blocked_ell_part(loaded.parts[0], {0u, 1u}, {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f})) {
        std::fprintf(stderr, "blocked ell part 0 mismatch after fetch\n");
        goto done;
    }
    if (!cellshard::drop_partition(&loaded, 0u)) {
        std::fprintf(stderr, "failed to drop blocked ell part 0\n");
        goto done;
    }
    if (!cellshard::fetch_partition(&loaded, &storage, 1u)) {
        std::fprintf(stderr, "failed to fetch blocked ell part 1\n");
        goto done;
    }
    if (!check_blocked_ell_part(loaded.parts[1], {1u, 0u}, {5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 8.5f})) {
        std::fprintf(stderr, "blocked ell part 1 mismatch after fetch\n");
        goto done;
    }
    if (!cellshard::drop_all_partitions(&loaded)) {
        std::fprintf(stderr, "failed to drop blocked ell parts\n");
        goto done;
    }
    if (!cellshard::fetch_shard(&loaded, &storage, 0u)) {
        std::fprintf(stderr, "failed to fetch blocked ell shard 0\n");
        goto done;
    }
    if (!check_blocked_ell_part(loaded.parts[0], {0u, 1u}, {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f})
        || !check_blocked_ell_part(loaded.parts[1], {1u, 0u}, {5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 8.5f})) {
        std::fprintf(stderr, "blocked ell shard payload mismatch after fetch\n");
        goto done;
    }

    rc = 0;

done:
    if (storage.backend == cellshard::shard_storage_backend_dataset_h5
        && storage.role != cellshard::shard_storage_role_executor) {
        cellshard::invalidate_dataset_h5_cache(&storage);
    }
    cellshard::clear(&storage);
    cellshard::clear(&loaded);
    cellshard::sparse::clear(&part0);
    cellshard::sparse::clear(&part1);
    std::remove(out_path.c_str());
    return rc;
}

static int run_executor_role_execution_pack_guard_test() {
    const std::string out_path = "/tmp/cellshard_dataset_executor_guard_test.csh5";
    char cache_root_template[] = "/tmp/cellshard_dataset_executor_guard_cache_XXXXXX";
    const char *cache_root_cstr = ::mkdtemp(cache_root_template);
    const std::string cache_root = cache_root_cstr != nullptr ? cache_root_cstr : "";
    cellshard::sharded<cellshard::sparse::blocked_ell> loaded;
    cellshard::shard_storage storage;
    cellshard::bucketed_blocked_ell_partition exec_part;
    int rc = 1;

    cellshard::init(&loaded);
    cellshard::init(&storage);
    cellshard::init(&exec_part);
    if (cache_root.empty()) {
        std::fprintf(stderr, "failed to create executor guard cache root\n");
        goto done;
    }
    if (!create_small_blocked_ell_dataset(out_path)) {
        std::fprintf(stderr, "failed to create executor guard dataset\n");
        goto done;
    }
    if (!cellshard::warm_dataset_blocked_ell_h5_cache(out_path.c_str(), cache_root.c_str())) {
        std::fprintf(stderr, "failed to warm canonical cache for executor guard test\n");
        goto done;
    }
    if (!cellshard::load_header(out_path.c_str(), &loaded, &storage)
        || storage.backend != cellshard::shard_storage_backend_dataset_h5
        || !cellshard::bind_dataset_h5_cache(&storage, cache_root.c_str())) {
        std::fprintf(stderr, "failed to bind executor guard dataset\n");
        goto done;
    }
    if (!cellshard::adopt_dataset_h5_executor_role(&storage)) {
        std::fprintf(stderr, "failed to adopt executor role\n");
        goto done;
    }
    if (!cellshard::fetch_partition(&loaded, &storage, 0u)
        || !check_blocked_ell_part(loaded.parts[0],
                                   {0u, 1u},
                                   {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f})) {
        std::fprintf(stderr, "executor role canonical pack fetch failed\n");
        goto done;
    }
    if (with_suppressed_stderr([&]() {
            return cellshard::fetch_dataset_blocked_ell_h5_execution_partition(&exec_part, &loaded, &storage, 0u);
        })) {
        std::fprintf(stderr, "executor role unexpectedly rebuilt execution partition without published pack\n");
        goto done;
    }

    rc = 0;

done:
    if (storage.backend == cellshard::shard_storage_backend_dataset_h5
        && storage.role != cellshard::shard_storage_role_executor) {
        cellshard::invalidate_dataset_h5_cache(&storage);
    }
    cellshard::clear(&exec_part);
    cellshard::clear(&storage);
    cellshard::clear(&loaded);
    std::remove(out_path.c_str());
    return rc;
}

static int run_quantized_blocked_ell_roundtrip_test() {
    const std::string out_path = "/tmp/cellshard_dataset_quantized_blocked_ell_test.csh5";
    const std::string cache_root = "/tmp/cellshard_dataset_quantized_blocked_ell_cache";
    cellshard::sparse::quantized_blocked_ell part0;
    cellshard::sharded<cellshard::sparse::quantized_blocked_ell> loaded;
    cellshard::shard_storage storage;
    const std::uint32_t bits = 8u;
    const std::uint32_t block_size = 2u;
    const std::uint32_t ell_cols = 4u;
    const std::uint32_t row_stride_bytes = cellshard::sparse::quantized_blocked_ell_aligned_row_bytes(bits, ell_cols);
    std::vector<std::uint64_t> partition_rows = { 2u };
    std::vector<std::uint64_t> partition_nnz = { 8u };
    std::vector<std::uint64_t> partition_aux = {
        (std::uint64_t) cellshard::sparse::pack_quantized_blocked_ell_aux(bits, block_size, 2ul)
    };
    std::vector<std::uint64_t> partition_row_offsets = { 0u, 2u };
    std::vector<std::uint32_t> partition_dataset_ids = { 0u };
    std::vector<std::uint32_t> partition_codec_ids = { 7u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 2u };
    std::vector<unsigned int> block_idx = { 0u, 1u };
    std::vector<unsigned char> packed_values((std::size_t) row_stride_bytes * 2u, 0u);
    std::vector<float> column_scales = { 1.0f, 1.0f, 1.0f, 1.0f };
    std::vector<float> column_offsets = { 0.0f, 0.0f, 0.0f, 0.0f };
    std::vector<float> row_offsets = { 0.0f, 0.0f };
    cellshard::dataset_codec_descriptor codec{};
    cellshard::dataset_layout_view layout{};
    int rc = 1;

    packed_values[0] = 1u;
    packed_values[1] = 2u;
    packed_values[2] = 3u;
    packed_values[3] = 4u;
    packed_values[(std::size_t) row_stride_bytes + 0u] = 5u;
    packed_values[(std::size_t) row_stride_bytes + 1u] = 6u;
    packed_values[(std::size_t) row_stride_bytes + 2u] = 7u;
    packed_values[(std::size_t) row_stride_bytes + 3u] = 8u;

    std::remove(out_path.c_str());
    cellshard::sparse::init(&part0);
    cellshard::init(&loaded);
    cellshard::init(&storage);

    if (!populate_quantized_blocked_ell_part(&part0,
                                             2u,
                                             4u,
                                             8u,
                                             block_size,
                                             ell_cols,
                                             bits,
                                             cellshard::dataset_quantized_decode_policy_per_gene_affine,
                                             block_idx,
                                             packed_values,
                                             column_scales,
                                             column_offsets,
                                             row_offsets)) {
        goto done;
    }

    codec.codec_id = 7u;
    codec.family = cellshard::dataset_codec_family_quantized_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::value_f32;
    codec.scale_value_code = (std::uint32_t) ::real::code_of< float>::code;
    codec.bits = bits;
    codec.flags = cellshard::dataset_codec_flag_direct_device_delivery
        | cellshard::dataset_codec_flag_live_fused_decode;
    codec.flags = cellshard::set_dataset_codec_quantized_decode_policy(
        codec.flags,
        cellshard::dataset_quantized_decode_policy_per_gene_affine);

    layout.rows = 2u;
    layout.cols = 4u;
    layout.nnz = 8u;
    layout.num_partitions = 1u;
    layout.num_shards = 1u;
    layout.partition_rows = partition_rows.data();
    layout.partition_nnz = partition_nnz.data();
    layout.partition_axes = 0;
    layout.partition_aux = partition_aux.data();
    layout.partition_row_offsets = partition_row_offsets.data();
    layout.partition_dataset_ids = partition_dataset_ids.data();
    layout.partition_codec_ids = partition_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    if (!cellshard::create_dataset_quantized_blocked_ell_h5(out_path.c_str(), &layout, 0, 0)) goto done;
    if (!cellshard::append_quantized_blocked_ell_partition_h5(out_path.c_str(), 0u, &part0)) goto done;
    if (!cellshard::warm_dataset_quantized_blocked_ell_h5_cache(out_path.c_str(), cache_root.c_str())) {
        std::fprintf(stderr, "failed to warm quantized blocked ell shard cache\n");
        goto done;
    }
    if (!cellshard::load_header(out_path.c_str(), &loaded, &storage)) {
        std::fprintf(stderr, "failed to load quantized blocked ell dataset header\n");
        goto done;
    }
    if (storage.backend != cellshard::shard_storage_backend_dataset_h5) {
        std::fprintf(stderr, "quantized blocked ell storage backend mismatch\n");
        goto done;
    }
    if (!cellshard::bind_dataset_h5_cache(&storage, cache_root.c_str())) {
        std::fprintf(stderr, "failed to bind quantized blocked ell cache dir\n");
        goto done;
    }
    if (!cellshard::fetch_partition(&loaded, &storage, 0u)) {
        std::fprintf(stderr, "failed to fetch quantized blocked ell part 0\n");
        goto done;
    }
    if (!check_quantized_blocked_ell_part(loaded.parts[0],
                                          bits,
                                          cellshard::dataset_quantized_decode_policy_per_gene_affine,
                                          block_idx,
                                          packed_values,
                                          column_scales,
                                          column_offsets,
                                          row_offsets)) {
        std::fprintf(stderr, "quantized blocked ell part mismatch after fetch\n");
        goto done;
    }

    rc = 0;

done:
    if (storage.backend == cellshard::shard_storage_backend_dataset_h5) {
        cellshard::invalidate_dataset_h5_cache(&storage);
    }
    cellshard::clear(&storage);
    cellshard::clear(&loaded);
    cellshard::sparse::clear(&part0);
    std::remove(out_path.c_str());
    return rc;
}

static int run_optimized_blocked_ell_roundtrip_test() {
    const std::string out_path = "/tmp/cellshard_dataset_optimized_blocked_ell_test.csh5";
    const std::string cache_root = "/tmp/cellshard_dataset_optimized_blocked_ell_cache";
    cellshard::sparse::blocked_ell part0;
    cellshard::sparse::blocked_ell part1;
    cellshard::bucketed_blocked_ell_partition bucket0;
    cellshard::bucketed_blocked_ell_partition bucket1;
    cellshard::bucketed_blocked_ell_partition exec_part;
    cellshard::bucketed_blocked_ell_shard shard;
    cellshard::sharded<cellshard::sparse::blocked_ell> loaded;
    cellshard::shard_storage storage;
    std::vector<std::uint64_t> partition_rows = { 2u, 2u };
    std::vector<std::uint64_t> partition_nnz = { 8u, 8u };
    std::vector<std::uint64_t> partition_aux = {
        (std::uint64_t) cellshard::sparse::pack_blocked_ell_aux(2u, 2ul),
        (std::uint64_t) cellshard::sparse::pack_blocked_ell_aux(2u, 2ul)
    };
    std::vector<std::uint64_t> partition_row_offsets = { 0u, 2u, 4u };
    std::vector<std::uint32_t> partition_dataset_ids = { 0u, 0u };
    std::vector<std::uint32_t> partition_codec_ids = { 0u, 0u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 4u };
    cellshard::dataset_codec_descriptor codec{};
    cellshard::dataset_layout_view layout{};
    int rc = 1;

    std::remove(out_path.c_str());
    cellshard::sparse::init(&part0);
    cellshard::sparse::init(&part1);
    cellshard::init(&bucket0);
    cellshard::init(&bucket1);
    cellshard::init(&exec_part);
    cellshard::init(&shard);
    cellshard::init(&loaded);
    cellshard::init(&storage);

    if (!populate_blocked_ell_part(&part0,
                                   2u,
                                   4u,
                                   2u,
                                   4u,
                                   {0u, 1u},
                                   {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f})) goto done;
    if (!populate_blocked_ell_part(&part1,
                                   2u,
                                   4u,
                                   2u,
                                   4u,
                                   {1u, 0u},
                                   {5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 8.5f})) goto done;
    if (!cellshard::build_bucketed_blocked_ell_partition(&bucket0, &part0, 2u, nullptr)
        || !cellshard::build_bucketed_blocked_ell_partition(&bucket1, &part1, 2u, nullptr)) {
        goto done;
    }

    shard.rows = 4u;
    shard.cols = 4u;
    shard.nnz = 16u;
    shard.partition_count = 2u;
    shard.partitions = (cellshard::bucketed_blocked_ell_partition *) std::calloc(2u, sizeof(cellshard::bucketed_blocked_ell_partition));
    shard.partition_row_offsets = (std::uint32_t *) std::calloc(3u, sizeof(std::uint32_t));
    shard.exec_to_canonical_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    shard.canonical_to_exec_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
    if (shard.partitions == nullptr
        || shard.partition_row_offsets == nullptr
        || shard.exec_to_canonical_cols == nullptr
        || shard.canonical_to_exec_cols == nullptr) {
        goto done;
    }
    shard.partition_row_offsets[0] = 0u;
    shard.partition_row_offsets[1] = 2u;
    shard.partition_row_offsets[2] = 4u;
    for (std::uint32_t col = 0u; col < 4u; ++col) {
        shard.exec_to_canonical_cols[col] = col;
        shard.canonical_to_exec_cols[col] = col;
    }
    shard.partitions[0] = bucket0;
    shard.partitions[1] = bucket1;
    bucket0 = {};
    bucket1 = {};
    for (std::uint32_t part = 0u; part < shard.partition_count; ++part) {
        shard.partitions[part].exec_to_canonical_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
        shard.partitions[part].canonical_to_exec_cols = (std::uint32_t *) std::calloc(4u, sizeof(std::uint32_t));
        if (shard.partitions[part].exec_to_canonical_cols == nullptr || shard.partitions[part].canonical_to_exec_cols == nullptr) goto done;
        for (std::uint32_t col = 0u; col < 4u; ++col) {
            shard.partitions[part].exec_to_canonical_cols[col] = col;
            shard.partitions[part].canonical_to_exec_cols[col] = col;
        }
    }

    codec.codec_id = 0u;
    codec.family = cellshard::dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = 4u;
    layout.cols = 4u;
    layout.nnz = 16u;
    layout.num_partitions = 2u;
    layout.num_shards = 1u;
    layout.partition_rows = partition_rows.data();
    layout.partition_nnz = partition_nnz.data();
    layout.partition_axes = 0;
    layout.partition_aux = partition_aux.data();
    layout.partition_row_offsets = partition_row_offsets.data();
    layout.partition_dataset_ids = partition_dataset_ids.data();
    layout.partition_codec_ids = partition_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    if (!cellshard::create_dataset_optimized_blocked_ell_h5(out_path.c_str(), &layout, 0, 0)) goto done;
    if (!cellshard::append_bucketed_blocked_ell_shard_h5(out_path.c_str(), 0u, &shard)) goto done;
    {
        hid_t file = H5Fopen(out_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t payload = file >= 0 ? H5Gopen2(file, "/payload", H5P_DEFAULT) : (hid_t) -1;
        const htri_t has_canonical = payload >= 0 ? H5Lexists(payload, "blocked_ell", H5P_DEFAULT) : -1;
        if (payload >= 0) H5Gclose(payload);
        if (file >= 0) H5Fclose(file);
        if (has_canonical > 0) {
            std::fprintf(stderr, "optimized blocked ell file still contains canonical blocked payload\n");
            goto done;
        }
    }
    if (!cellshard::warm_dataset_blocked_ell_h5_cache(out_path.c_str(), cache_root.c_str())
        || !cellshard::warm_dataset_blocked_ell_h5_execution_cache(out_path.c_str(), cache_root.c_str())) {
        std::fprintf(stderr, "failed to warm optimized blocked ell caches\n");
        goto done;
    }
    if (!cellshard::load_header(out_path.c_str(), &loaded, &storage)
        || storage.backend != cellshard::shard_storage_backend_dataset_h5
        || !cellshard::bind_dataset_h5_cache(&storage, cache_root.c_str())) {
        std::fprintf(stderr, "failed to load optimized blocked ell dataset header\n");
        goto done;
    }
    if (!cellshard::fetch_partition(&loaded, &storage, 0u)
        || !check_blocked_ell_part(loaded.parts[0], {0u, 1u}, {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f})) {
        std::fprintf(stderr, "optimized blocked ell canonical fetch mismatch\n");
        goto done;
    }
    if (!cellshard::fetch_dataset_blocked_ell_h5_execution_partition(&exec_part, &loaded, &storage, 0u)) {
        std::fprintf(stderr, "optimized blocked ell execution fetch failed\n");
        goto done;
    }
    if (exec_part.exec_to_canonical_cols == nullptr || exec_part.canonical_to_exec_cols == nullptr) {
        std::fprintf(stderr, "optimized blocked ell execution column maps missing\n");
        goto done;
    }
    for (std::uint32_t col = 0u; col < 4u; ++col) {
        if (exec_part.exec_to_canonical_cols[col] != col || exec_part.canonical_to_exec_cols[col] != col) {
            std::fprintf(stderr, "optimized blocked ell execution column maps mismatch\n");
            goto done;
        }
    }

    rc = 0;

done:
    if (storage.backend == cellshard::shard_storage_backend_dataset_h5) {
        cellshard::invalidate_dataset_h5_cache(&storage);
    }
    cellshard::clear(&exec_part);
    cellshard::clear(&storage);
    cellshard::clear(&loaded);
    cellshard::clear(&shard);
    cellshard::clear(&bucket0);
    cellshard::clear(&bucket1);
    cellshard::sparse::clear(&part0);
    cellshard::sparse::clear(&part1);
    std::remove(out_path.c_str());
    return rc;
}

static int run_sliced_ell_roundtrip_test() {
    const std::string out_path = "/tmp/cellshard_dataset_sliced_ell_test.csh5";
    const std::string cache_root = "/tmp/cellshard_dataset_sliced_ell_cache";
    cellshard::sparse::sliced_ell part0;
    cellshard::sparse::sliced_ell part1;
    cellshard::bucketed_sliced_ell_partition stored0;
    cellshard::bucketed_sliced_ell_partition stored1;
    cellshard::sharded<cellshard::sparse::sliced_ell> loaded;
    cellshard::shard_storage storage;
    std::vector<std::uint64_t> partition_rows = { 2u, 2u };
    std::vector<std::uint64_t> partition_nnz = { 4u, 6u };
    std::vector<std::uint64_t> partition_aux = {
        (std::uint64_t) cellshard::sparse::pack_sliced_ell_aux(2u, 4u),
        (std::uint64_t) cellshard::sparse::pack_sliced_ell_aux(2u, 6u)
    };
    std::vector<std::uint64_t> partition_row_offsets = { 0u, 2u, 4u };
    std::vector<std::uint32_t> partition_dataset_ids = { 0u, 0u };
    std::vector<std::uint32_t> partition_codec_ids = { 0u, 0u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 4u };
    cellshard::dataset_codec_descriptor codec{};
    cellshard::dataset_layout_view layout{};
    int rc = 1;

    std::remove(out_path.c_str());
    cellshard::sparse::init(&part0);
    cellshard::sparse::init(&part1);
    cellshard::init(&stored0);
    cellshard::init(&stored1);
    cellshard::init(&loaded);
    cellshard::init(&storage);

    if (!populate_sliced_ell_part(&part0,
                                  2u,
                                  4u,
                                  {0u, 1u, 2u},
                                  {3u, 1u},
                                  {0u, 1u, 2u, 3u},
                                  {1.0f, 2.0f, 3.0f, 4.0f},
                                  4u)) goto done;
    if (!populate_sliced_ell_part(&part1,
                                  2u,
                                  4u,
                                  {0u, 1u, 2u},
                                  {4u, 2u},
                                  {0u, 1u, 2u, 3u, 0u, 2u},
                                  {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
                                  6u)) goto done;

    codec.codec_id = 0u;
    codec.family = cellshard::dataset_codec_family_sliced_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = 4u;
    layout.cols = 4u;
    layout.nnz = 10u;
    layout.num_partitions = 2u;
    layout.num_shards = 1u;
    layout.partition_rows = partition_rows.data();
    layout.partition_nnz = partition_nnz.data();
    layout.partition_axes = 0;
    layout.partition_aux = partition_aux.data();
    layout.partition_row_offsets = partition_row_offsets.data();
    layout.partition_dataset_ids = partition_dataset_ids.data();
    layout.partition_codec_ids = partition_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    if (!cellshard::create_dataset_sliced_ell_h5(out_path.c_str(), &layout, 0, 0)) goto done;
    if (!cellshard::build_bucketed_sliced_ell_partition(&stored0, &part0, 2u, 0)
        || !cellshard::build_bucketed_sliced_ell_partition(&stored1, &part1, 2u, 0)
        || !cellshard::append_sliced_ell_partition_h5(out_path.c_str(), 0u, &stored0)
        || !cellshard::append_sliced_ell_partition_h5(out_path.c_str(), 1u, &stored1)) {
        goto done;
    }
    if (!cellshard::load_header(out_path.c_str(), &loaded, &storage)) {
        std::fprintf(stderr, "failed to load sliced ell dataset header\n");
        goto done;
    }
    if (!cellshard::bind_dataset_h5_cache(&storage, cache_root.c_str())) {
        std::fprintf(stderr, "failed to bind sliced ell cache dir\n");
        goto done;
    }
    if (!cellshard::fetch_partition(&loaded, &storage, 0u)) {
        std::fprintf(stderr, "failed to fetch sliced ell part 0\n");
        goto done;
    }
    if (!check_sliced_ell_part(loaded.parts[0],
                               {0u, 1u, 2u},
                               {3u, 1u},
                               {0u, 1u, 2u, 3u},
                               {1.0f, 2.0f, 3.0f, 4.0f})) {
        std::fprintf(stderr, "sliced ell part 0 mismatch after fetch\n");
        goto done;
    }
    if (!cellshard::drop_all_partitions(&loaded)) goto done;
    if (!cellshard::fetch_shard(&loaded, &storage, 0u)) {
        std::fprintf(stderr, "failed to fetch sliced ell shard 0\n");
        goto done;
    }
    if (!check_sliced_ell_part(loaded.parts[1],
                               {0u, 1u, 2u},
                               {4u, 2u},
                               {0u, 1u, 2u, 3u, 0u, 2u},
                               {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f})) {
        std::fprintf(stderr, "sliced ell part 1 mismatch after shard fetch\n");
        goto done;
    }

    rc = 0;

done:
    if (storage.backend == cellshard::shard_storage_backend_dataset_h5) {
        cellshard::invalidate_dataset_h5_cache(&storage);
    }
    cellshard::clear(&storage);
    cellshard::clear(&loaded);
    cellshard::clear(&stored0);
    cellshard::clear(&stored1);
    cellshard::sparse::clear(&part0);
    cellshard::sparse::clear(&part1);
    std::remove(out_path.c_str());
    return rc;
}

static int run_optimized_sliced_ell_roundtrip_test() {
    const std::string out_path = "/tmp/cellshard_dataset_optimized_sliced_ell_test.csh5";
    const std::string cache_root = "/tmp/cellshard_dataset_optimized_sliced_ell_cache";
    cellshard::sparse::sliced_ell part0;
    cellshard::sparse::sliced_ell part1;
    cellshard::bucketed_sliced_ell_partition bucket0;
    cellshard::bucketed_sliced_ell_partition bucket1;
    cellshard::bucketed_sliced_ell_partition exec_part;
    cellshard::dataset_sliced_bucketed_device_partition_view staged0;
    cellshard::dataset_sliced_bucketed_device_partition_view staged1;
    cellshard::dataset_sliced_bucketed_device_partition_view staged_after_generation;
    cellshard::sharded<cellshard::sparse::sliced_ell> loaded;
    cellshard::shard_storage storage;
    std::vector<std::uint64_t> partition_rows = { 2u, 2u };
    std::vector<std::uint64_t> partition_nnz = { 4u, 6u };
    std::vector<std::uint64_t> partition_aux = {
        (std::uint64_t) cellshard::sparse::pack_sliced_ell_aux(2u, 4u),
        (std::uint64_t) cellshard::sparse::pack_sliced_ell_aux(2u, 6u)
    };
    std::vector<std::uint64_t> partition_row_offsets = { 0u, 2u, 4u };
    std::vector<std::uint32_t> partition_dataset_ids = { 0u, 0u };
    std::vector<std::uint32_t> partition_codec_ids = { 0u, 0u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 4u };
    cellshard::dataset_codec_descriptor codec{};
    cellshard::dataset_layout_view layout{};
    std::vector<std::uint32_t> part_formats = {
        cellshard::dataset_execution_format_bucketed_sliced_ell,
        cellshard::dataset_execution_format_bucketed_sliced_ell
    };
    std::vector<std::uint32_t> part_block_sizes = { 0u, 0u };
    std::vector<std::uint32_t> part_bucket_counts = { 2u, 2u };
    std::vector<float> part_fill_ratios = { 0.0f, 0.0f };
    std::vector<std::uint64_t> part_execution_bytes(2u, 0u);
    std::vector<std::uint64_t> part_blocked_ell_bytes(2u, 0u);
    std::vector<std::uint64_t> part_bucketed_blocked_ell_bytes(2u, 0u);
    std::vector<std::uint32_t> part_sliced_counts = { 2u, 2u };
    std::vector<std::uint32_t> part_sliced_rows = { 1u, 1u };
    std::vector<std::uint64_t> part_sliced_bytes(2u, 0u);
    std::vector<std::uint64_t> part_bucketed_sliced_bytes(2u, 0u);
    std::vector<std::uint32_t> shard_formats = { cellshard::dataset_execution_format_bucketed_sliced_ell };
    std::vector<std::uint32_t> shard_block_sizes = { 0u };
    std::vector<std::uint32_t> shard_bucketed_partition_counts = { 2u };
    std::vector<std::uint32_t> shard_bucketed_segment_counts = { 4u };
    std::vector<float> shard_fill_ratios = { 0.0f };
    std::vector<std::uint64_t> shard_execution_bytes = { 0u };
    std::vector<std::uint64_t> shard_bucketed_blocked_ell_bytes = { 0u };
    std::vector<std::uint32_t> shard_sliced_counts = { 4u };
    std::vector<std::uint32_t> shard_sliced_rows = { 1u };
    std::vector<std::uint64_t> shard_bucketed_sliced_bytes = { 0u };
    std::vector<std::uint32_t> shard_pair_ids = { 0u };
    std::vector<std::uint32_t> shard_owner_node_ids = { 0u };
    std::vector<std::uint32_t> shard_owner_rank_ids = { 0u };
    cellshard::dataset_execution_view execution{};
    cellshard::dataset_execution_view loaded_execution{};
    cellshard::dataset_runtime_service_view runtime_service{};
    int rc = 1;
    int device_count = 0;

    std::remove(out_path.c_str());
    cellshard::sparse::init(&part0);
    cellshard::sparse::init(&part1);
    cellshard::init(&bucket0);
    cellshard::init(&bucket1);
    cellshard::init(&exec_part);
    cellshard::init(&staged0);
    cellshard::init(&staged1);
    cellshard::init(&staged_after_generation);
    cellshard::init(&loaded);
    cellshard::init(&storage);

    if (!populate_sliced_ell_part(&part0,
                                  2u,
                                  4u,
                                  {0u, 1u, 2u},
                                  {3u, 1u},
                                  {0u, 1u, 2u, 3u},
                                  {1.0f, 2.0f, 3.0f, 4.0f},
                                  4u)) goto done;
    if (!populate_sliced_ell_part(&part1,
                                  2u,
                                  4u,
                                  {0u, 1u, 2u},
                                  {4u, 2u},
                                  {0u, 1u, 2u, 3u, 0u, 2u},
                                  {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
                                  6u)) goto done;
    if (!cellshard::build_bucketed_sliced_ell_partition(&bucket0, &part0, 2u, &part_bucketed_sliced_bytes[0])
        || !cellshard::build_bucketed_sliced_ell_partition(&bucket1, &part1, 2u, &part_bucketed_sliced_bytes[1])) {
        goto done;
    }
    part_sliced_bytes[0] = (std::uint64_t) cellshard::packed_bytes((const cellshard::sparse::sliced_ell *) 0,
                                                                    part0.rows,
                                                                    part0.cols,
                                                                    part0.nnz,
                                                                    cellshard::partition_aux(&part0),
                                                                    sizeof(::real::storage_t));
    part_sliced_bytes[1] = (std::uint64_t) cellshard::packed_bytes((const cellshard::sparse::sliced_ell *) 0,
                                                                    part1.rows,
                                                                    part1.cols,
                                                                    part1.nnz,
                                                                    cellshard::partition_aux(&part1),
                                                                    sizeof(::real::storage_t));
    part_execution_bytes[0] = part_bucketed_sliced_bytes[0];
    part_execution_bytes[1] = part_bucketed_sliced_bytes[1];

    shard_bucketed_sliced_bytes[0] = part_bucketed_sliced_bytes[0] + part_bucketed_sliced_bytes[1];
    shard_execution_bytes[0] = shard_bucketed_sliced_bytes[0];

    codec.codec_id = 0u;
    codec.family = cellshard::dataset_codec_family_sliced_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = 4u;
    layout.cols = 4u;
    layout.nnz = 10u;
    layout.num_partitions = 2u;
    layout.num_shards = 1u;
    layout.partition_rows = partition_rows.data();
    layout.partition_nnz = partition_nnz.data();
    layout.partition_axes = 0;
    layout.partition_aux = partition_aux.data();
    layout.partition_row_offsets = partition_row_offsets.data();
    layout.partition_dataset_ids = partition_dataset_ids.data();
    layout.partition_codec_ids = partition_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    execution.partition_count = (std::uint32_t) part_formats.size();
    execution.partition_execution_formats = part_formats.data();
    execution.partition_blocked_ell_block_sizes = part_block_sizes.data();
    execution.partition_blocked_ell_bucket_counts = part_bucket_counts.data();
    execution.partition_blocked_ell_fill_ratios = part_fill_ratios.data();
    execution.partition_execution_bytes = part_execution_bytes.data();
    execution.partition_blocked_ell_bytes = part_blocked_ell_bytes.data();
    execution.partition_bucketed_blocked_ell_bytes = part_bucketed_blocked_ell_bytes.data();
    execution.partition_sliced_ell_slice_counts = part_sliced_counts.data();
    execution.partition_sliced_ell_slice_rows = part_sliced_rows.data();
    execution.partition_sliced_ell_bytes = part_sliced_bytes.data();
    execution.partition_bucketed_sliced_ell_bytes = part_bucketed_sliced_bytes.data();
    execution.shard_count = (std::uint32_t) shard_formats.size();
    execution.shard_execution_formats = shard_formats.data();
    execution.shard_blocked_ell_block_sizes = shard_block_sizes.data();
    execution.shard_bucketed_partition_counts = shard_bucketed_partition_counts.data();
    execution.shard_bucketed_segment_counts = shard_bucketed_segment_counts.data();
    execution.shard_blocked_ell_fill_ratios = shard_fill_ratios.data();
    execution.shard_execution_bytes = shard_execution_bytes.data();
    execution.shard_bucketed_blocked_ell_bytes = shard_bucketed_blocked_ell_bytes.data();
    execution.shard_sliced_ell_slice_counts = shard_sliced_counts.data();
    execution.shard_sliced_ell_slice_rows = shard_sliced_rows.data();
    execution.shard_bucketed_sliced_ell_bytes = shard_bucketed_sliced_bytes.data();
    execution.shard_preferred_pair_ids = shard_pair_ids.data();
    execution.shard_owner_node_ids = shard_owner_node_ids.data();
    execution.shard_owner_rank_ids = shard_owner_rank_ids.data();
    execution.preferred_base_format = cellshard::dataset_execution_format_bucketed_sliced_ell;
    cellshard::init(&runtime_service);
    runtime_service.service_mode = cellshard::dataset_runtime_service_mode_owner_hosted;
    runtime_service.live_write_mode = cellshard::dataset_live_write_mode_append_only;
    runtime_service.prefer_pack_delivery = 1u;
    runtime_service.single_reader_coordinator = 1u;
    runtime_service.canonical_generation = 1u;
    runtime_service.execution_plan_generation = 1u;
    runtime_service.pack_generation = 1u;
    runtime_service.service_epoch = 1u;
    runtime_service.active_read_generation = 1u;

    if (!cellshard::create_dataset_sliced_ell_h5(out_path.c_str(), &layout, 0, 0)) goto done;
    if (!cellshard::append_sliced_ell_partition_h5(out_path.c_str(), 0u, &bucket0)) goto done;
    if (!cellshard::append_sliced_ell_partition_h5(out_path.c_str(), 1u, &bucket1)) goto done;
    if (!cellshard::append_dataset_execution_h5(out_path.c_str(), &execution)) goto done;
    if (!cellshard::append_dataset_runtime_service_h5(out_path.c_str(), &runtime_service)) goto done;
    if (!cellshard::load_header(out_path.c_str(), &loaded, &storage)) goto done;
    if (!cellshard::bind_dataset_h5_cache(&storage, cache_root.c_str())) goto done;
    if (!cellshard::get_dataset_h5_execution_metadata(&storage, &loaded_execution)) goto done;
    if (loaded_execution.partition_sliced_ell_slice_rows == nullptr
        || loaded_execution.shard_sliced_ell_slice_rows == nullptr
        || loaded_execution.partition_sliced_ell_slice_rows[0] != 1u
        || loaded_execution.partition_sliced_ell_slice_rows[1] != 1u
        || loaded_execution.shard_sliced_ell_slice_rows[0] != 1u) {
        std::fprintf(stderr, "optimized sliced ell slice_rows metadata mismatch\n");
        goto done;
    }
    if (!cellshard::fetch_partition(&loaded, &storage, 0u)
        || !check_sliced_ell_part(loaded.parts[0],
                                  {0u, 1u, 2u},
                                  {3u, 1u},
                                  {0u, 1u, 2u, 3u},
                                  {1.0f, 2.0f, 3.0f, 4.0f})) {
        std::fprintf(stderr, "optimized sliced ell canonical fetch mismatch\n");
        goto done;
    }
    if (!cellshard::fetch_dataset_sliced_ell_h5_bucketed_partition(&exec_part, &loaded, &storage, 0u)) {
        std::fprintf(stderr, "optimized sliced ell execution fetch failed\n");
        goto done;
    }
    if (exec_part.segment_count < 1u
        || exec_part.exec_to_canonical_rows == nullptr
        || exec_part.canonical_to_exec_rows == nullptr
        || exec_part.exec_to_canonical_rows[0] != 1u
        || exec_part.exec_to_canonical_rows[1] != 0u
        || exec_part.canonical_to_exec_rows[0] != 1u
        || exec_part.canonical_to_exec_rows[1] != 0u) {
        std::fprintf(stderr,
                     "optimized sliced ell row reorder mismatch: segments=%u exec=[%u,%u] canon=[%u,%u]\n",
                     exec_part.segment_count,
                     exec_part.exec_to_canonical_rows != nullptr ? exec_part.exec_to_canonical_rows[0] : 999u,
                     exec_part.exec_to_canonical_rows != nullptr ? exec_part.exec_to_canonical_rows[1] : 999u,
                     exec_part.canonical_to_exec_rows != nullptr ? exec_part.canonical_to_exec_rows[0] : 999u,
                     exec_part.canonical_to_exec_rows != nullptr ? exec_part.canonical_to_exec_rows[1] : 999u);
        goto done;
    }
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        if (!cellshard::clear_all_dataset_sliced_ell_h5_bucketed_device_caches()) goto done;
        if (!cellshard::acquire_dataset_sliced_ell_h5_bucketed_partition_device(&staged0,
                                                                                 &loaded,
                                                                                 &storage,
                                                                                 0u,
                                                                                 0,
                                                                                 0u)) {
            std::fprintf(stderr, "optimized sliced ell device cache acquire failed\n");
            goto done;
        }
        if (!cellshard::acquire_dataset_sliced_ell_h5_bucketed_partition_device(&staged1,
                                                                                 &loaded,
                                                                                 &storage,
                                                                                 0u,
                                                                                 0,
                                                                                 0u)) {
            std::fprintf(stderr, "optimized sliced ell device cache reacquire failed\n");
            goto done;
        }
        if (staged0.host_partition == nullptr
            || staged0.device_segments == nullptr
            || staged0.segment_count != exec_part.segment_count
            || staged0.host_partition != staged1.host_partition
            || staged0.device_segments != staged1.device_segments
            || staged0.pack_generation != 1u) {
            std::fprintf(stderr, "optimized sliced ell device cache reuse mismatch\n");
            goto done;
        }
        cellshard::clear(&storage);
        cellshard::clear(&loaded);
        cellshard::init(&loaded);
        cellshard::init(&storage);
        if (!overwrite_u64_attr(out_path, "/runtime_service", "pack_generation", 2u)) {
            std::fprintf(stderr, "optimized sliced ell failed to update pack generation\n");
            goto done;
        }
        if (!cellshard::load_header(out_path.c_str(), &loaded, &storage)
            || !cellshard::bind_dataset_h5_cache(&storage, cache_root.c_str())) {
            std::fprintf(stderr, "optimized sliced ell failed to reload after generation change\n");
            goto done;
        }
        if (!cellshard::acquire_dataset_sliced_ell_h5_bucketed_partition_device(&staged_after_generation,
                                                                                 &loaded,
                                                                                 &storage,
                                                                                 0u,
                                                                                 0,
                                                                                 0u)
            || staged_after_generation.pack_generation != 2u) {
            std::fprintf(stderr, "optimized sliced ell device cache generation invalidation failed\n");
            goto done;
        }
    }

    rc = 0;

done:
    (void) cellshard::release_dataset_sliced_ell_h5_bucketed_partition_device(&staged_after_generation);
    (void) cellshard::release_dataset_sliced_ell_h5_bucketed_partition_device(&staged1);
    (void) cellshard::release_dataset_sliced_ell_h5_bucketed_partition_device(&staged0);
    (void) cellshard::clear_all_dataset_sliced_ell_h5_bucketed_device_caches();
    if (storage.backend == cellshard::shard_storage_backend_dataset_h5) {
        cellshard::invalidate_dataset_h5_cache(&storage);
    }
    cellshard::clear(&exec_part);
    cellshard::clear(&storage);
    cellshard::clear(&loaded);
    cellshard::clear(&bucket0);
    cellshard::clear(&bucket1);
    cellshard::sparse::clear(&part0);
    cellshard::sparse::clear(&part1);
    std::remove(out_path.c_str());
    return rc;
}

static int run_runtime_service_metadata_test() {
    const std::string out_path = "/tmp/cellshard_dataset_runtime_service_test.csh5";
    char cache_root_template[] = "/tmp/cellshard_dataset_runtime_service_cache_XXXXXX";
    const char *cache_root_cstr = ::mkdtemp(cache_root_template);
    const std::string cache_root = cache_root_cstr != nullptr ? cache_root_cstr : "";
    cellshard::sparse::blocked_ell part0;
    cellshard::sparse::blocked_ell part1;
    cellshard::sharded<cellshard::sparse::blocked_ell> loaded;
    cellshard::shard_storage storage;
    std::vector<std::uint64_t> partition_rows = { 2u, 2u };
    std::vector<std::uint64_t> partition_nnz = { 8u, 8u };
    std::vector<std::uint64_t> partition_aux = {
        (std::uint64_t) cellshard::sparse::pack_blocked_ell_aux(2u, 2ul),
        (std::uint64_t) cellshard::sparse::pack_blocked_ell_aux(2u, 2ul)
    };
    std::vector<std::uint64_t> partition_row_offsets = { 0u, 2u, 4u };
    std::vector<std::uint32_t> partition_dataset_ids = { 0u, 0u };
    std::vector<std::uint32_t> partition_codec_ids = { 0u, 0u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 4u };
    cellshard::dataset_codec_descriptor codec{};
    cellshard::dataset_layout_view layout{};
    std::vector<std::uint32_t> part_formats = {
        cellshard::dataset_execution_format_bucketed_blocked_ell,
        cellshard::dataset_execution_format_bucketed_blocked_ell
    };
    std::vector<std::uint32_t> part_block_sizes = { 2u, 2u };
    std::vector<std::uint32_t> part_bucket_counts = { 1u, 1u };
    std::vector<float> part_fill_ratios = { 1.0f, 1.0f };
    std::vector<std::uint64_t> part_execution_bytes = { 64u, 64u };
    std::vector<std::uint64_t> part_blocked_ell_bytes = { 64u, 64u };
    std::vector<std::uint64_t> part_bucketed_blocked_ell_bytes = { 64u, 64u };
    std::vector<std::uint32_t> shard_formats = { cellshard::dataset_execution_format_bucketed_blocked_ell };
    std::vector<std::uint32_t> shard_block_sizes = { 2u };
    std::vector<std::uint32_t> shard_bucketed_partition_counts = { 2u };
    std::vector<std::uint32_t> shard_bucketed_segment_counts = { 2u };
    std::vector<float> shard_fill_ratios = { 1.0f };
    std::vector<std::uint64_t> shard_execution_bytes = { 128u };
    std::vector<std::uint64_t> shard_bucketed_blocked_ell_bytes = { 128u };
    std::vector<std::uint32_t> shard_pair_ids = { 0u };
    std::vector<std::uint32_t> shard_owner_node_ids = { 7u };
    std::vector<std::uint32_t> shard_owner_rank_ids = { 3u };
    cellshard::dataset_execution_view execution{};
    cellshard::dataset_execution_view loaded_execution{};
    cellshard::dataset_runtime_service_view runtime_service{};
    cellshard::dataset_runtime_service_view loaded_runtime_service{};
    int rc = 1;

    std::remove(out_path.c_str());
    cellshard::sparse::init(&part0);
    cellshard::sparse::init(&part1);
    cellshard::init(&loaded);
    cellshard::init(&storage);
    cellshard::init(&runtime_service);
    cellshard::init(&loaded_runtime_service);

    if (!populate_blocked_ell_part(&part0,
                                   2u,
                                   4u,
                                   2u,
                                   4u,
                                   {0u, 1u},
                                   {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f})) goto done;
    if (!populate_blocked_ell_part(&part1,
                                   2u,
                                   4u,
                                   2u,
                                   4u,
                                   {1u, 0u},
                                   {5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 8.5f})) goto done;

    codec.codec_id = 0u;
    codec.family = cellshard::dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = 4u;
    layout.cols = 4u;
    layout.nnz = 16u;
    layout.num_partitions = 2u;
    layout.num_shards = 1u;
    layout.partition_rows = partition_rows.data();
    layout.partition_nnz = partition_nnz.data();
    layout.partition_axes = 0;
    layout.partition_aux = partition_aux.data();
    layout.partition_row_offsets = partition_row_offsets.data();
    layout.partition_dataset_ids = partition_dataset_ids.data();
    layout.partition_codec_ids = partition_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    execution.partition_count = (std::uint32_t) part_formats.size();
    execution.partition_execution_formats = part_formats.data();
    execution.partition_blocked_ell_block_sizes = part_block_sizes.data();
    execution.partition_blocked_ell_bucket_counts = part_bucket_counts.data();
    execution.partition_blocked_ell_fill_ratios = part_fill_ratios.data();
    execution.partition_execution_bytes = part_execution_bytes.data();
    execution.partition_blocked_ell_bytes = part_blocked_ell_bytes.data();
    execution.partition_bucketed_blocked_ell_bytes = part_bucketed_blocked_ell_bytes.data();
    execution.shard_count = (std::uint32_t) shard_formats.size();
    execution.shard_execution_formats = shard_formats.data();
    execution.shard_blocked_ell_block_sizes = shard_block_sizes.data();
    execution.shard_bucketed_partition_counts = shard_bucketed_partition_counts.data();
    execution.shard_bucketed_segment_counts = shard_bucketed_segment_counts.data();
    execution.shard_blocked_ell_fill_ratios = shard_fill_ratios.data();
    execution.shard_execution_bytes = shard_execution_bytes.data();
    execution.shard_bucketed_blocked_ell_bytes = shard_bucketed_blocked_ell_bytes.data();
    execution.shard_preferred_pair_ids = shard_pair_ids.data();
    execution.shard_owner_node_ids = shard_owner_node_ids.data();
    execution.shard_owner_rank_ids = shard_owner_rank_ids.data();
    execution.preferred_base_format = cellshard::dataset_execution_format_bucketed_blocked_ell;

    runtime_service.service_mode = cellshard::dataset_runtime_service_mode_owner_hosted;
    runtime_service.live_write_mode = cellshard::dataset_live_write_mode_append_only;
    runtime_service.prefer_pack_delivery = 1u;
    runtime_service.remote_pack_delivery = 1u;
    runtime_service.single_reader_coordinator = 1u;
    runtime_service.maintenance_lock_blocks_overwrite = 1u;
    runtime_service.canonical_generation = 11u;
    runtime_service.execution_plan_generation = 12u;
    runtime_service.pack_generation = 13u;
    runtime_service.service_epoch = 14u;
    runtime_service.active_read_generation = 15u;
    runtime_service.staged_write_generation = 16u;

    if (!cellshard::create_dataset_blocked_ell_h5(out_path.c_str(), &layout, 0, 0)) goto done;
    if (!cellshard::append_blocked_ell_partition_h5(out_path.c_str(), 0u, &part0)) goto done;
    if (!cellshard::append_blocked_ell_partition_h5(out_path.c_str(), 1u, &part1)) goto done;
    if (!cellshard::append_dataset_execution_h5(out_path.c_str(), &execution)) goto done;
    if (!cellshard::append_dataset_runtime_service_h5(out_path.c_str(), &runtime_service)) goto done;
    if (!cellshard::warm_dataset_blocked_ell_h5_cache(out_path.c_str(), cache_root.c_str())) goto done;
    if (!cellshard::warm_dataset_blocked_ell_h5_execution_cache(out_path.c_str(), cache_root.c_str())) goto done;
    if (!cellshard::load_header(out_path.c_str(), &loaded, &storage)) goto done;
    if (!cellshard::bind_dataset_h5_cache(&storage, cache_root.c_str())) goto done;
    if (!cellshard::get_dataset_h5_execution_metadata(&storage, &loaded_execution)) goto done;
    if (!cellshard::get_dataset_h5_runtime_service(&storage, &loaded_runtime_service)) goto done;
    if (loaded_execution.shard_count != 1u
        || loaded_execution.partition_count != 2u
        || loaded_execution.preferred_base_format != cellshard::dataset_execution_format_bucketed_blocked_ell
        || loaded_execution.shard_owner_node_ids == nullptr
        || loaded_execution.shard_owner_rank_ids == nullptr
        || loaded_execution.shard_owner_node_ids[0] != 7u
        || loaded_execution.shard_owner_rank_ids[0] != 3u) {
        std::fprintf(stderr, "runtime service execution metadata mismatch\n");
        goto done;
    }
    if (loaded_runtime_service.service_mode != cellshard::dataset_runtime_service_mode_owner_hosted
        || loaded_runtime_service.live_write_mode != cellshard::dataset_live_write_mode_append_only
        || loaded_runtime_service.remote_pack_delivery != 1u
        || loaded_runtime_service.canonical_generation != 11u
        || loaded_runtime_service.execution_plan_generation != 12u
        || loaded_runtime_service.pack_generation != 13u
        || loaded_runtime_service.service_epoch != 14u
        || loaded_runtime_service.active_read_generation != 15u
        || loaded_runtime_service.staged_write_generation != 16u) {
        std::fprintf(stderr, "runtime service generation metadata mismatch\n");
        goto done;
    }
    {
        const std::string instances_root = cache_root + "/instances";
        const std::string instance_dir = find_first_named_subdir(instances_root);
        if (!path_is_dir(instances_root) || instance_dir.empty()) {
            std::fprintf(stderr, "runtime service cache instances dir missing\n");
            goto done;
        }
        if (!path_is_dir(instance_dir + "/metadata")
            || !path_exists(instance_dir + "/metadata/manifest.txt")
            || !path_is_dir(instance_dir + "/packs")
            || !path_is_dir(instance_dir + "/packs/canonical")
            || !path_is_dir(instance_dir + "/packs/execution")
            || !path_exists(instance_dir + "/packs/canonical/shard.0.pack")
            || !path_exists(instance_dir + "/packs/execution/plan.12-pack.13-epoch.14/shard.0.exec.pack")) {
            std::fprintf(stderr, "runtime service cache layout mismatch\n");
            goto done;
        }
    }
    runtime_service.execution_plan_generation = 21u;
    runtime_service.pack_generation = 22u;
    runtime_service.service_epoch = 23u;
    runtime_service.active_read_generation = 24u;
    runtime_service.staged_write_generation = 25u;
    if (!overwrite_u64_attr(out_path, "/runtime_service", "execution_plan_generation", runtime_service.execution_plan_generation)) goto done;
    if (!overwrite_u64_attr(out_path, "/runtime_service", "pack_generation", runtime_service.pack_generation)) goto done;
    if (!overwrite_u64_attr(out_path, "/runtime_service", "service_epoch", runtime_service.service_epoch)) goto done;
    if (!overwrite_u64_attr(out_path, "/runtime_service", "active_read_generation", runtime_service.active_read_generation)) goto done;
    if (!overwrite_u64_attr(out_path, "/runtime_service", "staged_write_generation", runtime_service.staged_write_generation)) goto done;
    if (!cellshard::warm_dataset_blocked_ell_h5_execution_cache(out_path.c_str(), cache_root.c_str())) goto done;
    {
        const std::string instances_root = cache_root + "/instances";
        if (!any_named_subdir_has_path(instances_root, "packs/execution/plan.21-pack.22-epoch.23/shard.0.exec.pack")) {
            std::fprintf(stderr, "updated runtime service execution pack generation missing\n");
            goto done;
        }
    }
    if (!cellshard::fetch_partition(&loaded, &storage, 0u)
        || !check_blocked_ell_part(loaded.parts[0], {0u, 1u}, {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f})) {
        std::fprintf(stderr, "runtime service blocked ell fetch mismatch\n");
        goto done;
    }

    rc = 0;

done:
    if (storage.backend == cellshard::shard_storage_backend_dataset_h5) {
        cellshard::invalidate_dataset_h5_cache(&storage);
    }
    cellshard::clear(&storage);
    cellshard::clear(&loaded);
    cellshard::sparse::clear(&part0);
    cellshard::sparse::clear(&part1);
    std::remove(out_path.c_str());
    return rc;
}

static int run_schema_version_rejection_test() {
    const std::string out_path = "/tmp/cellshard_dataset_schema_rejection_test.csh5";
    cellshard::sharded<cellshard::sparse::blocked_ell> loaded;
    int rc = 1;

    cellshard::init(&loaded);
    if (!create_small_blocked_ell_dataset(out_path)) goto done;
    if (!overwrite_u32_attr(out_path, "/", "schema_version", cellshard::dataset_h5_schema_version + 1u)) goto done;
    if (with_suppressed_stderr([&]() {
            return cellshard::load_header(out_path.c_str(), &loaded, nullptr);
        })) {
        std::fprintf(stderr, "schema-version rejection unexpectedly succeeded\n");
        goto done;
    }
    rc = 0;

done:
    cellshard::clear(&loaded);
    std::remove(out_path.c_str());
    return rc;
}

static int run_header_consistency_rejection_test() {
    const std::string out_path = "/tmp/cellshard_dataset_header_consistency_test.csh5";
    cellshard::sharded<cellshard::sparse::blocked_ell> loaded;
    int rc = 1;

    cellshard::init(&loaded);
    if (!create_small_blocked_ell_dataset(out_path)) goto done;
    if (!overwrite_u64_attr(out_path, "/", "rows", 3u)) goto done;
    if (with_suppressed_stderr([&]() {
            return cellshard::load_header(out_path.c_str(), &loaded, nullptr);
        })) {
        std::fprintf(stderr, "header consistency rejection unexpectedly succeeded\n");
        goto done;
    }
    rc = 0;

done:
    cellshard::clear(&loaded);
    std::remove(out_path.c_str());
    return rc;
}

static int run_dataset_extent_rejection_test() {
    const std::string out_path = "/tmp/cellshard_dataset_extent_rejection_test.csh5";
    cellshard::sharded<cellshard::sparse::blocked_ell> loaded;
    int rc = 1;

    cellshard::init(&loaded);
    if (!create_small_blocked_ell_dataset(out_path)) goto done;
    if (!replace_u64_dataset(out_path, "/matrix/partition_row_offsets", {0u, 2u, 2u})) goto done;
    if (with_suppressed_stderr([&]() {
            return cellshard::load_header(out_path.c_str(), &loaded, nullptr);
        })) {
        std::fprintf(stderr, "dataset extent rejection unexpectedly succeeded\n");
        goto done;
    }
    rc = 0;

done:
    cellshard::clear(&loaded);
    std::remove(out_path.c_str());
    return rc;
}

} // namespace

static int run_blocked_ell_side_domain_test() {
    const std::string out_path = "/tmp/cellshard_dataset_blocked_ell_side_domain_test.csh5";
    const std::string cache_root = "/tmp/cellshard_dataset_blocked_ell_side_domain_cache";
    cellshard::sparse::blocked_ell part;
    cellshard::sharded<cellshard::sparse::blocked_ell> loaded;
    cellshard::shard_storage storage;
    owned_text_column dataset_ids = make_column({"embryo_1_counts"});
    owned_text_column matrix_paths = make_column({"/tmp/embryo_1_counts.mtx"});
    owned_text_column feature_paths = make_column({"/tmp/features.tsv"});
    owned_text_column barcode_paths = make_column({"/tmp/barcodes.tsv"});
    owned_text_column metadata_paths = make_column({""});
    owned_text_column global_barcodes = make_column({"bc0", "bc1"});
    owned_text_column feature_ids = make_column({"g0", "g1", "g2"});
    owned_text_column feature_names = make_column({"Gene0", "Gene1", "Gene2"});
    owned_text_column feature_types = make_column({"gene", "gene", "gene"});
    owned_text_column metadata_column_names = make_column({"stage", "batch"});
    owned_text_column metadata_field_values = make_column({"E8", "A", "E9", "A"});
    std::vector<std::uint32_t> dataset_formats = { 2u };
    std::vector<std::uint64_t> dataset_row_begin = { 0u };
    std::vector<std::uint64_t> dataset_row_end = { 2u };
    std::vector<std::uint64_t> dataset_rows = { 2u };
    std::vector<std::uint64_t> dataset_cols = { 3u };
    std::vector<std::uint64_t> dataset_nnz = { 6u };
    std::vector<std::uint32_t> cell_dataset_ids = { 0u, 0u };
    std::vector<std::uint64_t> cell_local_indices = { 0u, 1u };
    std::vector<std::uint32_t> feature_dataset_ids = { 0u, 0u, 0u };
    std::vector<std::uint64_t> feature_local_indices = { 0u, 1u, 2u };
    std::vector<std::uint64_t> dataset_feature_offsets = { 0u, 3u };
    std::vector<std::uint32_t> dataset_feature_to_global = { 0u, 1u, 2u };
    std::vector<std::uint32_t> metadata_row_offsets = { 0u, 2u, 4u };
    std::vector<std::uint32_t> metadata_dataset_indices = { 0u };
    std::vector<std::uint64_t> metadata_global_row_begin = { 0u };
    std::vector<std::uint64_t> metadata_global_row_end = { 2u };
    std::vector<std::uint64_t> partition_rows = { 2u };
    std::vector<std::uint64_t> partition_nnz = { 6u };
    std::vector<std::uint64_t> partition_aux = {
        (std::uint64_t) cellshard::sparse::pack_blocked_ell_aux(1u, 3ul)
    };
    std::vector<std::uint64_t> partition_row_offsets = { 0u, 2u };
    std::vector<std::uint32_t> partition_dataset_ids = { 0u };
    std::vector<std::uint32_t> partition_codec_ids = { 0u };
    std::vector<std::uint64_t> shard_offsets = { 0u, 2u };
    cellshard::dataset_codec_descriptor codec{};
    cellshard::dataset_layout_view layout{};
    cellshard::dataset_dataset_table_view dataset_view{};
    cellshard::dataset_provenance_view provenance_view{};
    cellshard::dataset_metadata_table_view metadata_table_view{};
    cellshard::dataset_embedded_metadata_view embedded_metadata_view{};
    cellshard::dataset_observation_metadata_view observation_metadata_view{};
    cellshard::dataset_feature_metadata_view feature_metadata_view{};
    cellshard::dataset_user_attribute_view dataset_attribute_view{};
    cellshard::dataset_browse_cache_view browse_view{};
    owned_observation_metadata_column obs_day_label;
    owned_observation_metadata_column obs_postnatal;
    owned_observation_metadata_column var_chr;
    owned_observation_metadata_column var_short_name;
    std::vector<cellshard::dataset_observation_metadata_column_view> observation_columns;
    std::vector<cellshard::dataset_observation_metadata_column_view> feature_columns;
    owned_text_column dataset_attribute_keys;
    owned_text_column dataset_attribute_values;
    std::vector<std::uint32_t> browse_feature_indices = { 0u, 2u };
    std::vector<float> browse_gene_sum = { 1.0f, 5.0f };
    std::vector<float> browse_gene_detected = { 1.0f, 2.0f };
    std::vector<float> browse_gene_sq_sum = { 1.0f, 13.0f };
    std::vector<float> browse_dataset_mean = { 0.5f, 0.0f, 2.5f };
    std::vector<float> browse_shard_mean = { 0.5f, 0.0f, 2.5f };
    std::vector<std::uint32_t> browse_part_sample_offsets = { 0u, 2u };
    std::vector<std::uint64_t> browse_part_sample_rows = { 0u, 1u };
    std::vector<float> browse_partition_sample_values = {
        1.0f, 0.0f,
        0.0f, 3.0f
    };
    int rc = 1;

    std::remove(out_path.c_str());
    cellshard::sparse::init(&part);
    cellshard::init(&loaded);
    cellshard::init(&storage);

    if (!populate_blocked_ell_part(&part,
                                   2u,
                                   3u,
                                   1u,
                                   3u,
                                   {0u, 1u, 2u, 0u, 1u, 2u},
                                   {1.0f, 9.0f, 2.0f, 8.0f, 3.0f, 7.0f})) {
        goto done;
    }

    codec.codec_id = 0u;
    codec.family = cellshard::dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    layout.rows = 2u;
    layout.cols = 3u;
    layout.nnz = 6u;
    layout.num_partitions = 1u;
    layout.num_shards = 1u;
    layout.partition_rows = partition_rows.data();
    layout.partition_nnz = partition_nnz.data();
    layout.partition_axes = nullptr;
    layout.partition_aux = partition_aux.data();
    layout.partition_row_offsets = partition_row_offsets.data();
    layout.partition_dataset_ids = partition_dataset_ids.data();
    layout.partition_codec_ids = partition_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    dataset_view.count = 1u;
    dataset_view.dataset_ids = dataset_ids.view();
    dataset_view.matrix_paths = matrix_paths.view();
    dataset_view.feature_paths = feature_paths.view();
    dataset_view.barcode_paths = barcode_paths.view();
    dataset_view.metadata_paths = metadata_paths.view();
    dataset_view.formats = dataset_formats.data();
    dataset_view.row_begin = dataset_row_begin.data();
    dataset_view.row_end = dataset_row_end.data();
    dataset_view.rows = dataset_rows.data();
    dataset_view.cols = dataset_cols.data();
    dataset_view.nnz = dataset_nnz.data();

    provenance_view.global_barcodes = global_barcodes.view();
    provenance_view.cell_dataset_ids = cell_dataset_ids.data();
    provenance_view.cell_local_indices = cell_local_indices.data();
    provenance_view.feature_ids = feature_ids.view();
    provenance_view.feature_names = feature_names.view();
    provenance_view.feature_types = feature_types.view();
    provenance_view.feature_dataset_ids = feature_dataset_ids.data();
    provenance_view.feature_local_indices = feature_local_indices.data();
    provenance_view.dataset_feature_offsets = dataset_feature_offsets.data();
    provenance_view.dataset_feature_to_global = dataset_feature_to_global.data();

    metadata_table_view.rows = 2u;
    metadata_table_view.cols = 2u;
    metadata_table_view.column_names = metadata_column_names.view();
    metadata_table_view.field_values = metadata_field_values.view();
    metadata_table_view.row_offsets = metadata_row_offsets.data();

    embedded_metadata_view.count = 1u;
    embedded_metadata_view.dataset_indices = metadata_dataset_indices.data();
    embedded_metadata_view.global_row_begin = metadata_global_row_begin.data();
    embedded_metadata_view.global_row_end = metadata_global_row_end.data();
    embedded_metadata_view.tables = &metadata_table_view;

    obs_day_label.name = "embryonic_day_label";
    obs_day_label.type = cellshard::dataset_observation_metadata_type_text;
    obs_day_label.text_values = make_column({"E8.5", "P0"});
    obs_postnatal.name = "is_postnatal";
    obs_postnatal.type = cellshard::dataset_observation_metadata_type_uint8;
    obs_postnatal.uint8_values = {0u, 1u};
    observation_columns = {obs_day_label.view(), obs_postnatal.view()};
    observation_metadata_view.rows = 2u;
    observation_metadata_view.cols = (std::uint32_t) observation_columns.size();
    observation_metadata_view.columns = observation_columns.data();

    var_chr.name = "chr";
    var_chr.type = cellshard::dataset_observation_metadata_type_text;
    var_chr.text_values = make_column({"chrM", "chr1", "chr2"});
    var_short_name.name = "gene_short_name";
    var_short_name.type = cellshard::dataset_observation_metadata_type_text;
    var_short_name.text_values = make_column({"MT-CO1", "POU5F1", "SOX2"});
    feature_columns = {var_chr.view(), var_short_name.view()};
    feature_metadata_view.cols = 3u;
    feature_metadata_view.annotation_count = (std::uint32_t) feature_columns.size();
    feature_metadata_view.annotations = feature_columns.data();

    dataset_attribute_keys = make_column({"study", "preprocess.pipeline_scope"});
    dataset_attribute_values = make_column({"demo", "qc_only"});
    dataset_attribute_view.count = 2u;
    dataset_attribute_view.keys = dataset_attribute_keys.view();
    dataset_attribute_view.values = dataset_attribute_values.view();

    browse_view.selected_feature_count = 2u;
    browse_view.selected_feature_indices = browse_feature_indices.data();
    browse_view.gene_sum = browse_gene_sum.data();
    browse_view.gene_detected = browse_gene_detected.data();
    browse_view.gene_sq_sum = browse_gene_sq_sum.data();
    browse_view.dataset_count = 1u;
    browse_view.dataset_feature_mean = browse_dataset_mean.data();
    browse_view.shard_count = 1u;
    browse_view.shard_feature_mean = browse_shard_mean.data();
    browse_view.partition_count = 1u;
    browse_view.sample_rows_per_partition = 2u;
    browse_view.partition_sample_row_offsets = browse_part_sample_offsets.data();
    browse_view.partition_sample_global_rows = browse_part_sample_rows.data();
    browse_view.partition_sample_values = browse_partition_sample_values.data();

    if (!cellshard::create_dataset_blocked_ell_h5(out_path.c_str(), &layout, &dataset_view, &provenance_view)) goto done;
    if (!cellshard::append_blocked_ell_partition_h5(out_path.c_str(), 0u, &part)) goto done;
    if (!cellshard::append_dataset_embedded_metadata_h5(out_path.c_str(), &embedded_metadata_view)
        || !cellshard::append_dataset_observation_metadata_h5(out_path.c_str(), &observation_metadata_view)
        || !cellshard::append_dataset_feature_metadata_h5(out_path.c_str(), &feature_metadata_view)
        || !cellshard::append_dataset_user_attributes_h5(out_path.c_str(), &dataset_attribute_view)
        || !cellshard::append_dataset_browse_cache_h5(out_path.c_str(), &browse_view)) {
        std::fprintf(stderr, "blocked ell side-domain append failed\n");
        goto done;
    }
    if (!cellshard::load_header(out_path.c_str(), &loaded, &storage)) {
        std::fprintf(stderr, "blocked ell side-domain reload failed\n");
        goto done;
    }
    if (loaded.rows != 2u || loaded.cols != 3u || loaded.nnz != 6u) {
        std::fprintf(stderr,
                     "blocked ell side-domain header mismatch rows=%lu cols=%lu nnz=%lu\n",
                     loaded.rows,
                     loaded.cols,
                     loaded.nnz);
        goto done;
    }
    if (!cellshard::bind_dataset_h5_cache(&storage, cache_root.c_str())
        || !cellshard::prefetch_dataset_blocked_ell_h5_shard_cache(&loaded, &storage, 0u)
        || !cellshard::fetch_partition(&loaded, &storage, 0u)) {
        std::fprintf(stderr, "blocked ell side-domain fetch failed\n");
        goto done;
    }
    if (!check_blocked_ell_part(loaded.parts[0],
                                {0u, 1u, 2u, 0u, 1u, 2u},
                                {1.0f, 9.0f, 2.0f, 8.0f, 3.0f, 7.0f})) {
        std::fprintf(stderr, "blocked ell side-domain payload mismatch\n");
        goto done;
    }

    rc = 0;

done:
    if (storage.backend == cellshard::shard_storage_backend_dataset_h5) {
        cellshard::invalidate_dataset_h5_cache(&storage);
    }
    cellshard::clear(&storage);
    cellshard::clear(&loaded);
    cellshard::sparse::clear(&part);
    std::remove(out_path.c_str());
    return rc;
}

int main() {
    if (run_blocked_ell_side_domain_test() != 0) {
        std::fprintf(stderr, "cellShardDatasetH5Test blocked ell side-domain append failed\n");
        return 1;
    }
    if (run_blocked_ell_roundtrip_test() != 0) {
        std::fprintf(stderr, "cellShardDatasetH5Test blocked ell roundtrip failed\n");
        return 1;
    }
    if (run_executor_role_execution_pack_guard_test() != 0) {
        std::fprintf(stderr, "cellShardDatasetH5Test executor role guard failed\n");
        return 1;
    }
    if (run_quantized_blocked_ell_roundtrip_test() != 0) {
        std::fprintf(stderr, "cellShardDatasetH5Test quantized blocked ell roundtrip failed\n");
        return 1;
    }
    if (run_optimized_blocked_ell_roundtrip_test() != 0) {
        std::fprintf(stderr, "cellShardDatasetH5Test optimized blocked ell roundtrip failed\n");
        return 1;
    }
    if (run_sliced_ell_roundtrip_test() != 0) {
        std::fprintf(stderr, "cellShardDatasetH5Test sliced ell roundtrip failed\n");
        return 1;
    }
    if (run_optimized_sliced_ell_roundtrip_test() != 0) {
        std::fprintf(stderr, "cellShardDatasetH5Test optimized sliced ell roundtrip failed\n");
        return 1;
    }
    if (run_runtime_service_metadata_test() != 0) {
        std::fprintf(stderr, "cellShardDatasetH5Test runtime service metadata failed\n");
        return 1;
    }
    if (run_schema_version_rejection_test() != 0) {
        std::fprintf(stderr, "cellShardDatasetH5Test schema-version rejection failed\n");
        return 1;
    }
    if (run_header_consistency_rejection_test() != 0) {
        std::fprintf(stderr, "cellShardDatasetH5Test header consistency rejection failed\n");
        return 1;
    }
    if (run_dataset_extent_rejection_test() != 0) {
        std::fprintf(stderr, "cellShardDatasetH5Test dataset extent rejection failed\n");
        return 1;
    }
    return 0;
}
