#include <Cellerator/preprocess/runtime.hh>

#include <cstdlib>
#include <cstdio>

namespace cpre = ::cellerator::preprocess;

namespace {

void require(bool ok, const char *message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

} // namespace

int main() {
    cpre::status status{};
    cpre::preprocess_cellshard_session_result result{};
    cpre::preprocess_cellshard_session_options options{};

    require(!cpre::preprocess_cellshard_session_all_gpus(&options, &result, &status),
            "missing input path should fail");
    require(status.code == cpre::status_invalid_argument, "missing input path should report invalid argument");
    cpre::clear(&result);

    const char *path = std::getenv("CELLERATOR_PREPROCESS_CSH5");
    if (path == nullptr || path[0] == '\0') return 0;

    options = cpre::preprocess_cellshard_session_options{};
    options.input_path = path;
    require(cpre::preprocess_cellshard_session_all_gpus(&options, &result, &status), status.message);
    require(result.rows != 0u, "session should report rows");
    require(result.cols != 0u, "session should report cols");
    require(result.partitions_processed != 0u, "session should process partitions");
    require(result.cell_keep != nullptr, "session should own cell keep mask");
    require(result.gene_keep != nullptr, "session should own gene keep mask");
    require(result.gene_sum != nullptr, "session should own gene metrics");
    cpre::clear(&result);
    return 0;
}
