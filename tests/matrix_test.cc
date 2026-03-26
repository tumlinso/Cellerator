#include "../src/matrix/matrix.hh"
#include "../src/matrix/matrix_io.hh"

#include <cstdio>

namespace {

template<typename T>
double toDouble(T value) {
    return (double) value;
}

template<>
double toDouble<matrix::Real>(matrix::Real value) {
    return (double) matrix::real_to_float(value);
}

template<typename ActualT, typename ExpectedT>
int checkEqual(const ActualT& actual, const ExpectedT& expected, const char *label) {
    double actualValue = toDouble(actual);
    double expectedValue = toDouble(expected);
    if (actualValue == expectedValue) return 1;
    std::fprintf(stderr, "FAIL: %s expected %g, got %g\n", label, expectedValue, actualValue);
    return 0;
}

int checkTrue(int condition, const char *label) {
    if (condition) return 1;
    std::fprintf(stderr, "FAIL: %s\n", label);
    return 0;
}

int testDenseRoundTrip() {
    const char *filename = "/tmp/cellerator_dense_test.bin";

    matrix::dense<> written;
    matrix::init(&written, 2u, 3u);
    matrix::allocate(&written);
    *matrix::at(&written, 0u, 0u) = matrix::real_from_float(1.0f);
    *matrix::at(&written, 0u, 1u) = matrix::real_from_float(2.0f);
    *matrix::at(&written, 0u, 2u) = matrix::real_from_float(3.0f);
    *matrix::at(&written, 1u, 0u) = matrix::real_from_float(4.0f);
    *matrix::at(&written, 1u, 1u) = matrix::real_from_float(5.0f);
    *matrix::at(&written, 1u, 2u) = matrix::real_from_float(6.0f);
    if (!matrix::store(filename, &written)) return 0;

    matrix::dense<> readBack;
    matrix::init(&readBack);
    if (!matrix::load(filename, &readBack)) return 0;

    int ok = 1;
    ok = checkEqual(readBack.rows, 2u, "dense rows") && ok;
    ok = checkEqual(readBack.cols, 3u, "dense cols") && ok;
    ok = checkEqual(readBack.nnz, 6u, "dense nnz") && ok;
    ok = checkEqual(readBack.format, (unsigned char) matrix::format_dense, "dense format") && ok;
    ok = checkEqual(*matrix::at(&readBack, 0u, 0u), 1.0f, "dense value (0,0)") && ok;
    ok = checkEqual(*matrix::at(&readBack, 1u, 2u), 6.0f, "dense value (1,2)") && ok;

    matrix::clear(&written);
    matrix::clear(&readBack);
    return ok;
}

int testCsrRoundTrip() {
    const char *filename = "/tmp/cellerator_csr_test.bin";

    matrix::sparse::csr<> written;
    matrix::sparse::init(&written, 3u, 4u, 4u);
    matrix::sparse::allocate(&written);
    written.rowPtr[0] = 0;
    written.rowPtr[1] = 2;
    written.rowPtr[2] = 3;
    written.rowPtr[3] = 4;
    written.colIdx[0] = 0;
    written.colIdx[1] = 3;
    written.colIdx[2] = 1;
    written.colIdx[3] = 2;
    written.val[0] = matrix::real_from_float(1.0f);
    written.val[1] = matrix::real_from_float(4.0f);
    written.val[2] = matrix::real_from_float(5.0f);
    written.val[3] = matrix::real_from_float(6.0f);
    if (!matrix::store(filename, &written)) return 0;

    matrix::sparse::csr<> readBack;
    matrix::sparse::init(&readBack);
    if (!matrix::load(filename, &readBack)) return 0;

    int ok = 1;
    ok = checkEqual(readBack.rows, 3u, "csr rows") && ok;
    ok = checkEqual(readBack.cols, 4u, "csr cols") && ok;
    ok = checkTrue(matrix::sparse::at(&readBack, 0u, 3u) != 0, "csr lookup") && ok;
    ok = checkTrue(matrix::sparse::at(&readBack, 1u, 3u) == 0, "csr missing lookup") && ok;
    if (matrix::sparse::at(&readBack, 0u, 3u) != 0) ok = checkEqual(*matrix::sparse::at(&readBack, 0u, 3u), 4.0f, "csr value") && ok;

    matrix::sparse::clear(&written);
    matrix::sparse::clear(&readBack);
    return ok;
}

int testCooConcatenation() {
    matrix::sparse::coo<> top;
    matrix::sparse::coo<> bottom;
    matrix::sparse::coo<> combined;
    matrix::sparse::init(&top, 2u, 3u, 2u);
    matrix::sparse::init(&bottom, 1u, 3u, 2u);
    matrix::sparse::init(&combined);
    matrix::sparse::allocate(&top);
    matrix::sparse::allocate(&bottom);

    top.rowIdx[0] = 0;
    top.colIdx[0] = 1;
    top.val[0] = matrix::real_from_float(1.0f);
    top.rowIdx[1] = 1;
    top.colIdx[1] = 2;
    top.val[1] = matrix::real_from_float(2.0f);

    bottom.rowIdx[0] = 0;
    bottom.colIdx[0] = 0;
    bottom.val[0] = matrix::real_from_float(3.0f);
    bottom.rowIdx[1] = 0;
    bottom.colIdx[1] = 2;
    bottom.val[1] = matrix::real_from_float(4.0f);

    if (!matrix::sparse::concatenate_rows(&combined, &top, &bottom)) return 0;
    if (!matrix::sparse::append_rows(&top, &bottom)) return 0;

    int ok = 1;
    ok = checkEqual(combined.rows, 3u, "coo rows") && ok;
    ok = checkEqual(combined.nnz, 4u, "coo nnz") && ok;
    ok = checkTrue(matrix::sparse::at(&combined, 2u, 0u) != 0, "coo appended row entry") && ok;
    ok = checkTrue(matrix::sparse::at(&top, 2u, 2u) != 0, "coo append in place") && ok;
    if (matrix::sparse::at(&combined, 2u, 0u) != 0) ok = checkEqual(*matrix::sparse::at(&combined, 2u, 0u), 3.0f, "coo combined value") && ok;
    if (matrix::sparse::at(&top, 2u, 2u) != 0) ok = checkEqual(*matrix::sparse::at(&top, 2u, 2u), 4.0f, "coo append value") && ok;

    matrix::sparse::clear(&top);
    matrix::sparse::clear(&bottom);
    matrix::sparse::clear(&combined);
    return ok;
}

int testDiaRoundTrip() {
    const char *filename = "/tmp/cellerator_dia_test.bin";

    matrix::sparse::dia<> written;
    matrix::sparse::init(&written, 3u, 3u, 6u);
    written.num_diagonals = 2;
    matrix::sparse::allocate(&written);
    written.offsets[0] = 0;
    written.offsets[1] = -1;
    written.val[0] = matrix::real_from_float(10.0f);
    written.val[1] = matrix::real_from_float(11.0f);
    written.val[2] = matrix::real_from_float(12.0f);
    written.val[3] = matrix::real_from_float(20.0f);
    written.val[4] = matrix::real_from_float(21.0f);
    written.val[5] = matrix::real_from_float(22.0f);
    if (!matrix::store(filename, &written)) return 0;

    matrix::sparse::dia<> readBack;
    matrix::sparse::init(&readBack);
    if (!matrix::load(filename, &readBack)) return 0;

    int ok = 1;
    ok = checkEqual(readBack.num_diagonals, 2u, "dia diagonals") && ok;
    ok = checkTrue(matrix::sparse::at(&readBack, 1u, 1u) != 0, "dia main diagonal") && ok;
    ok = checkTrue(matrix::sparse::at(&readBack, 2u, 1u) != 0, "dia lower diagonal") && ok;
    ok = checkTrue(matrix::sparse::at(&readBack, 1u, 2u) == 0, "dia missing entry") && ok;
    if (matrix::sparse::at(&readBack, 1u, 1u) != 0) ok = checkEqual(*matrix::sparse::at(&readBack, 1u, 1u), 11.0f, "dia value") && ok;
    if (matrix::sparse::at(&readBack, 2u, 1u) != 0) ok = checkEqual(*matrix::sparse::at(&readBack, 2u, 1u), 22.0f, "dia value 2") && ok;

    matrix::sparse::clear(&written);
    matrix::sparse::clear(&readBack);
    return ok;
}

int testShardedLayout() {
    matrix::dense<> *first = new matrix::dense<>;
    matrix::dense<> *second = new matrix::dense<>;
    matrix::sharded<matrix::dense<> > view;
    matrix::init(first, 1u, 2u);
    matrix::allocate(first);
    *matrix::at(first, 0u, 0u) = matrix::real_from_float(1.0f);
    *matrix::at(first, 0u, 1u) = matrix::real_from_float(2.0f);
    matrix::init(second, 2u, 2u);
    matrix::allocate(second);
    *matrix::at(second, 0u, 0u) = matrix::real_from_float(3.0f);
    *matrix::at(second, 0u, 1u) = matrix::real_from_float(4.0f);
    *matrix::at(second, 1u, 0u) = matrix::real_from_float(5.0f);
    *matrix::at(second, 1u, 1u) = matrix::real_from_float(6.0f);

    matrix::init(&view);
    if (!matrix::append_part(&view, first)) return 0;
    if (!matrix::append_part(&view, second)) return 0;

    matrix::Index offsets[3];
    offsets[0] = 0;
    offsets[1] = 2;
    offsets[2] = 3;
    if (!matrix::reshard(&view, 2u, offsets)) return 0;

    int ok = 1;
    ok = checkEqual(view.num_parts, 2u, "sharded part count") && ok;
    ok = checkEqual(view.part_offsets[0], 0u, "sharded part offset 0") && ok;
    ok = checkEqual(view.part_offsets[1], 1u, "sharded part offset 1") && ok;
    ok = checkEqual(view.part_offsets[2], 3u, "sharded part offset 2") && ok;
    ok = checkEqual(view.shard_offsets[1], 2u, "reshard boundary") && ok;
    ok = checkTrue(matrix::at(&view, 2u, 1u) != 0, "sharded concatenated lookup") && ok;
    ok = checkEqual(matrix::find_shard(&view, 0u), 0u, "find first shard") && ok;
    ok = checkEqual(matrix::find_shard(&view, 2u), 1u, "find second shard") && ok;
    if (matrix::at(&view, 2u, 1u) != 0) ok = checkEqual(*matrix::at(&view, 2u, 1u), 6.0f, "sharded value") && ok;

    matrix::clear(&view);
    return ok;
}

int testShardedDiskRoundTrip() {
    const char *headerFile = "/tmp/cellerator_sharded_header.bin";
    const char *part0 = "/tmp/cellerator_sharded_part0.bin";
    const char *part1 = "/tmp/cellerator_sharded_part1.bin";

    matrix::dense<> *first = new matrix::dense<>;
    matrix::dense<> *second = new matrix::dense<>;
    matrix::sharded<matrix::dense<> > written;
    matrix::sharded<matrix::dense<> > loaded;
    matrix::shard_storage files;
    matrix::shard_storage loadedFiles;

    matrix::init(first, 1u, 2u);
    matrix::allocate(first);
    *matrix::at(first, 0u, 0u) = matrix::real_from_float(1.0f);
    *matrix::at(first, 0u, 1u) = matrix::real_from_float(2.0f);
    matrix::init(second, 1u, 2u);
    matrix::allocate(second);
    *matrix::at(second, 0u, 0u) = matrix::real_from_float(3.0f);
    *matrix::at(second, 0u, 1u) = matrix::real_from_float(4.0f);

    matrix::init(&written);
    matrix::init(&loaded);
    matrix::init(&files);
    matrix::init(&loadedFiles);
    if (!matrix::append_part(&written, first)) return 0;
    if (!matrix::append_part(&written, second)) return 0;
    matrix::reserve(&files, written.num_parts);
    matrix::bind(&files, 0u, part0);
    matrix::bind(&files, 1u, part1);
    if (!matrix::store(headerFile, &written, &files)) return 0;

    if (!matrix::load_header(headerFile, &loaded)) return 0;
    matrix::reserve(&loadedFiles, loaded.num_parts);
    matrix::bind(&loadedFiles, 0u, part0);
    matrix::bind(&loadedFiles, 1u, part1);
    int ok = 1;
    ok = checkEqual(loaded.rows, 2u, "header logical rows") && 1;
    ok = checkEqual(loaded.part_offsets[0], 0u, "header part offset 0") && ok;
    ok = checkEqual(loaded.part_offsets[1], 1u, "header part offset 1") && ok;
    ok = checkEqual(loaded.part_offsets[2], 2u, "header part offset 2") && ok;
    ok = checkEqual(matrix::find_part(&loaded, 1u), 1u, "header logical find part") && ok;
    if (!matrix::fetch_all_parts(&loaded, &loadedFiles)) return 0;

    ok = checkEqual(loaded.num_parts, 2u, "disk part count") && ok;
    ok = checkEqual(loaded.num_shards, 2u, "disk shard count") && ok;
    ok = checkTrue(matrix::at(&loaded, 0u, 1u) != 0, "disk first lookup") && ok;
    ok = checkTrue(matrix::at(&loaded, 1u, 0u) != 0, "disk second lookup") && ok;
    if (matrix::at(&loaded, 0u, 1u) != 0) ok = checkEqual(*matrix::at(&loaded, 0u, 1u), 2.0f, "disk value 1") && ok;
    if (matrix::at(&loaded, 1u, 0u) != 0) ok = checkEqual(*matrix::at(&loaded, 1u, 0u), 3.0f, "disk value 2") && ok;

    matrix::clear(&files);
    matrix::clear(&loadedFiles);
    matrix::clear(&written);
    matrix::clear(&loaded);
    return ok;
}

int testShardedBudgeting() {
    matrix::dense<> *a = new matrix::dense<>;
    matrix::dense<> *b = new matrix::dense<>;
    matrix::dense<> *c = new matrix::dense<>;
    matrix::sharded<matrix::dense<> > view;
    matrix::shard_storage files;

    matrix::init(a, 1u, 2u);
    matrix::allocate(a);
    matrix::init(b, 1u, 2u);
    matrix::allocate(b);
    matrix::init(c, 1u, 2u);
    matrix::allocate(c);
    matrix::init(&view);
    matrix::init(&files);

    if (!matrix::append_part(&view, a)) return 0;
    if (!matrix::append_part(&view, b)) return 0;
    if (!matrix::append_part(&view, c)) return 0;
    if (!matrix::set_shards_by_part_bytes(&view, matrix::bytes(a) + matrix::bytes(b))) return 0;
    if (!matrix::bind_sequential(&files, view.num_parts, "/tmp/cellerator_auto_part")) return 0;

    int ok = 1;
    ok = checkEqual(view.num_shards, 2u, "budget shard count") && ok;
    ok = checkEqual(view.shard_offsets[0], 0u, "budget shard offset 0") && ok;
    ok = checkEqual(view.shard_offsets[1], 2u, "budget shard offset 1") && ok;
    ok = checkEqual(view.shard_offsets[2], 3u, "budget shard offset 2") && ok;
    ok = checkTrue(files.paths[0] != 0, "sequential bind path 0") && ok;
    ok = checkTrue(files.paths[2] != 0, "sequential bind path 2") && ok;

    matrix::clear(&files);
    matrix::clear(&view);
    return ok;
}

int testDropKeepsLogicalMap() {
    const char *headerFile = "/tmp/cellerator_drop_header.bin";
    const char *part0 = "/tmp/cellerator_drop_part0.bin";
    const char *part1 = "/tmp/cellerator_drop_part1.bin";
    matrix::dense<> *first = new matrix::dense<>;
    matrix::dense<> *second = new matrix::dense<>;
    matrix::sharded<matrix::dense<> > written;
    matrix::sharded<matrix::dense<> > loaded;
    matrix::shard_storage files;

    matrix::init(first, 2u, 2u);
    matrix::allocate(first);
    matrix::init(second, 2u, 2u);
    matrix::allocate(second);
    matrix::init(&written);
    matrix::init(&loaded);
    matrix::init(&files);

    if (!matrix::append_part(&written, first)) return 0;
    if (!matrix::append_part(&written, second)) return 0;
    if (!matrix::reserve(&files, 2u)) return 0;
    if (!matrix::bind(&files, 0u, part0)) return 0;
    if (!matrix::bind(&files, 1u, part1)) return 0;
    if (!matrix::store(headerFile, &written, &files)) return 0;
    if (!matrix::load_header(headerFile, &loaded)) return 0;
    if (!matrix::fetch_part(&loaded, &files, 1u)) return 0;
    if (!matrix::drop_part(&loaded, 1u)) return 0;

    int ok = 1;
    ok = checkEqual(loaded.rows, 4u, "drop keeps rows") && ok;
    ok = checkEqual(loaded.part_offsets[0], 0u, "drop part offset 0") && ok;
    ok = checkEqual(loaded.part_offsets[1], 2u, "drop part offset 1") && ok;
    ok = checkEqual(loaded.part_offsets[2], 4u, "drop part offset 2") && ok;
    ok = checkEqual(matrix::find_part(&loaded, 3u), 1u, "drop logical find part") && ok;
    ok = checkTrue(matrix::at(&loaded, 3u, 1u) == 0, "drop unloads host part") && ok;

    matrix::clear(&files);
    matrix::clear(&written);
    matrix::clear(&loaded);
    return ok;
}

} // namespace

int main() {
    int ok = 1;
    ok = testDenseRoundTrip() && ok;
    ok = testCsrRoundTrip() && ok;
    ok = testCooConcatenation() && ok;
    ok = testDiaRoundTrip() && ok;
    ok = testShardedLayout() && ok;
    ok = testShardedDiskRoundTrip() && ok;
    ok = testShardedBudgeting() && ok;
    ok = testDropKeepsLogicalMap() && ok;
    if (!ok) return 1;
    std::printf("matrix tests passed\n");
    return 0;
}
