#include <Cellerator/compiler/discovery/create_temporary_cellshard_compiler_compatibility_adapte_v1.hh>

namespace Cellerator::compiler::discovery {

const cellshard_compatibility_manifest_v1&
cellshard_compiler_compatibility_manifest_v1() noexcept {
    static constexpr cellshard_compatibility_manifest_v1 manifest{
        "cellshard::compiler::compatibility_v1",
        "Cellerator::compiler::discovery",
        "CE-CCP1-E02-018",
    };
    return manifest;
}

}  // namespace Cellerator::compiler::discovery
