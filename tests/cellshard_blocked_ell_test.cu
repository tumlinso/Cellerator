#include "../extern/CellShard/src/CellShard.hh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <unistd.h>

namespace cs = ::cellshard;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void fill_compressed_part(cs::sparse::compressed *part,
                          std::uint32_t rows,
                          std::uint32_t cols,
                          const std::uint32_t *major_ptr,
                          const std::uint32_t *minor_idx,
                          const float *values) {
    cs::sparse::init(part, rows, cols, major_ptr[rows], cs::sparse::compressed_by_row);
    require(cs::sparse::allocate(part) != 0, "compressed allocate failed");
    for (std::uint32_t i = 0; i <= rows; ++i) part->majorPtr[i] = major_ptr[i];
    for (std::uint32_t i = 0; i < part->nnz; ++i) {
        part->minorIdx[i] = minor_idx[i];
        part->val[i] = __float2half(values[i]);
    }
}

bool close_half(__half lhs, float rhs, float tol = 1.0e-3f) {
    return std::fabs(__half2float(lhs) - rhs) <= tol;
}

} // namespace

int main() {
    const std::uint32_t major0[] = { 0u, 2u, 3u };
    const std::uint32_t minor0[] = { 0u, 3u, 1u };
    const float values0[] = { 1.0f, 2.0f, 3.0f };

    const std::uint32_t major1[] = { 0u, 2u, 3u };
    const std::uint32_t minor1[] = { 0u, 2u, 3u };
    const float values1[] = { 4.0f, 5.0f, 6.0f };

    cs::sparse::compressed *part0 = new cs::sparse::compressed();
    cs::sparse::compressed *part1 = new cs::sparse::compressed();
    fill_compressed_part(part0, 2u, 4u, major0, minor0, values0);
    fill_compressed_part(part1, 2u, 4u, major1, minor1, values1);

    cs::sharded<cs::sparse::compressed> src;
    cs::init(&src);
    require(cs::append_part(&src, part0) != 0, "append part0 failed");
    require(cs::append_part(&src, part1) != 0, "append part1 failed");

    const unsigned int candidates[] = { 2u, 4u };
    cs::convert::blocked_ell_tune_result tune = {};
    require(cs::convert::choose_blocked_ell_block_size(src.parts[0], candidates, 2u, &tune) != 0, "choose_blocked_ell_block_size failed");
    require(tune.block_size == 2u || tune.block_size == 4u, "unexpected block size");

    cs::sparse::blocked_ell local;
    cs::sparse::init(&local);
    require(cs::convert::blocked_ell_from_compressed(src.parts[0], 2u, &local) != 0, "blocked_ell_from_compressed failed");
    require(local.block_size == 2u, "blocked ell block size mismatch");
    require(local.ell_cols == 4u, "blocked ell ell_cols mismatch");
    require(local.blockColIdx[0] == 0u && local.blockColIdx[1] == 1u, "row block 0 columns mismatch");
    require(close_half(local.val[0], 1.0f), "blocked ell row0 slot0 col0 mismatch");
    require(close_half(local.val[3], 2.0f), "blocked ell row0 slot1 col1 mismatch");
    require(close_half(local.val[5], 3.0f), "blocked ell row1 slot0 col1 mismatch");

    cs::sharded<cs::sparse::blocked_ell> blocked;
    cs::init(&blocked);
    require(cs::convert::repack_sharded_compressed_to_blocked_ell(&src, 2u, 1ul, &blocked) != 0, "repack_sharded_compressed_to_blocked_ell failed");
    require(blocked.num_parts == 2ul, "blocked ell repack part count mismatch");
    require(blocked.part_aux[0] == cs::sparse::pack_blocked_ell_aux(2u, 2ul), "blocked ell aux mismatch");
    require(cs::device::set_shards_by_device_bytes(&blocked, 128u) != 0, "set_shards_by_device_bytes failed");
    require(blocked.num_shards >= 1ul, "blocked ell shard count mismatch");

    cs::device::part_record<cs::sparse::blocked_ell> record;
    cs::device::zero_record(&record);
    require(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice failed");
    require(cs::device::upload(blocked.parts[0], &record) == cudaSuccess, "blocked ell upload failed");
    require(record.view != 0 && record.a0 != 0 && record.a1 != 0, "blocked ell upload pointers missing");
    require(cs::device::release(&record) == cudaSuccess, "blocked ell release failed");

    {
        char path[] = "/tmp/cellshard_blocked_ell_packXXXXXX";
        const int fd = ::mkstemp(path);
        require(fd >= 0, "mkstemp failed");
        ::close(fd);

        cs::shard_storage storage;
        cs::sharded<cs::sparse::blocked_ell> loaded;
        cs::init(&storage);
        cs::init(&loaded);
        require(cs::store(path, &blocked, &storage) != 0, "blocked ell packfile store failed");
        cs::clear(&storage);
        require(cs::load_header(path, &loaded, &storage) != 0, "blocked ell packfile load_header failed");
        require(loaded.num_parts == blocked.num_parts, "blocked ell loaded part count mismatch");
        require(loaded.part_aux[0] == blocked.part_aux[0], "blocked ell loaded aux mismatch");
        require(cs::fetch_part(&loaded, &storage, 0ul) != 0, "blocked ell fetch_part failed");
        require(loaded.parts[0] != nullptr, "blocked ell loaded part missing");
        require(loaded.parts[0]->block_size == blocked.parts[0]->block_size, "blocked ell loaded block size mismatch");
        require(loaded.parts[0]->ell_cols == blocked.parts[0]->ell_cols, "blocked ell loaded ell_cols mismatch");
        require(loaded.parts[0]->blockColIdx[0] == blocked.parts[0]->blockColIdx[0], "blocked ell loaded block col mismatch");
        require(close_half(loaded.parts[0]->val[0], __half2float(blocked.parts[0]->val[0])), "blocked ell loaded value mismatch");
        cs::clear(&storage);
        cs::clear(&loaded);
        std::remove(path);
    }

    cs::clear(&blocked);
    cs::clear(&src);
    cs::sparse::clear(&local);
    return 0;
}
