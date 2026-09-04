#include <Cellerator/compiler/tooling/implement_clangd_worker_discovery_and_lifecycle_v1.hh>

// Process spawning is supplied by the celleratord executable integration layer;
// this translation unit pins the independently testable worker lifecycle ABI.
static_assert(Cellerator::compiler::tooling::clangd_worker_status_v1::running
              != Cellerator::compiler::tooling::clangd_worker_status_v1::crashed);
