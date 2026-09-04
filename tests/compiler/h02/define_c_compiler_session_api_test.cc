#include <Cellerator/sdk/define_cpp_compiler_session_api_v1.hh>

#include <cassert>

namespace ca = cellerator::compiler::api::v1;

int main() {
    ca::compiler_session_v1 session({"sm_70", "cuda-13", "pbmc"});
    session.add_source("embed.cell", "cell x;");
    assert(session.source_manager_entry(0) == "embed.cell");
    assert(session.ast_snapshot() == "ast-v1" && session.sema_snapshot() == "sema-v1");
    assert(session.ceir_builder() == "ceir-builder-v1"
        && session.ceir_reader() == "ceir-reader-v1");
    assert(session.profile() == "pbmc" && session.backend() == "sm_70");
    assert(session.pass_pipeline() == "default-pipeline-v1");
    const auto result = session.compile();
    assert(result.source_count == 1 && !result.object_identity.empty());
}
