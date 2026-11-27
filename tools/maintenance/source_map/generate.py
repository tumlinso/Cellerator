#!/usr/bin/env python3
"""Generate a normalized, source-bound route map; does not change source files."""
import argparse, csv, json, hashlib, sys
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import snapshot, OUTPUT_NAMES, split0, git

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--repo", required=True, help="Cellerator repository checkout")
PARSER.add_argument("--output", default="docs/development/source_map", help="Generated output directory, relative to --repo")
ARGS = PARSER.parse_args()
REPO = Path(ARGS.repo).resolve()
OUT = Path(ARGS.output)
OUT = (REPO / OUT).resolve() if not OUT.is_absolute() else OUT.resolve()
OUT.mkdir(parents=True, exist_ok=True)

def split_ignored():
    return split0(git(REPO, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"))

# The retired modeling tree is recorded in the active migration closure.
closure_path = REPO / 'planning/active-source-cleanup/source-migrations.json'
closure = json.loads(closure_path.read_text()) if closure_path.is_file() else {}
modeling_migrations = {
    row['old_path']: row['new_path'] for row in closure.get('path_migrations', [])
    if row['old_path'].startswith(('include/Cellerator/modeling/', 'src/modeling/'))
    or row['old_path'] == 'cmake/CelleratorModeling.cmake'
}

# These operation headers and implementations have individual build owners.
native_operation_owners = {
    'reaction': ('host reaction network', 'ops/reaction/network.hh', None, 'reactions_test.cc'),
    'exchange': ('conservative exchange', 'ops/process/exchange.hh', None, 'coupling_test.cc'),
    'inheritance': ('inheritance policies', 'ops/process/inheritance.hh', None, 'inheritance_test.cc'),
    'accessibility': ('host accessibility conditions', 'ops/condition/accessibility.hh', None, 'accessibility_test.cc'),
    'event_memory': ('host event memory', 'ops/sequence/event_memory.hh', None, None),
    'attention_cuda': ('sparse attention', 'ops/attention/sparse_cuda.hh', 'ops/attention/attention.cu', 'attention_smoke.cu'),
    'dispatch_cuda': ('routing gather and combine', 'ops/routing/dispatch_cuda.hh', 'ops/routing/dispatch.cu', 'dispatch_smoke.cu'),
    'conditional_dispatch_cuda': ('conditional dispatch device preparation', 'execution/conditional/dispatch_cuda.hh', 'execution/conditional/dispatch.cu', 'dispatch_smoke.cu'),
    'dependency_frontier_cuda': ('changed-source work frontier', 'execution/dependency/frontier_cuda.hh', 'execution/dependency/frontier.cu', 'frontier_smoke.cu'),
    'grouped_linear_cuda': ('grouped linear', 'ops/matrix/grouped_linear_cuda.hh', 'ops/matrix/grouped_linear.cu', 'grouped_linear_smoke.cu'),
    'event_memory_cuda': ('device event memory', 'ops/sequence/event_memory_cuda.hh', 'ops/sequence/event_memory.cu', 'memory_smoke.cu'),
    'packed_embedding_cuda': ('packed embedding bags', 'ops/sequence/packed_embedding_cuda.hh', 'ops/sequence/packed_embedding.cu', 'packed_embedding_smoke.cu'),
    'framework_parameter_transfer': ('revision-pinned CUDA parameter transfer', 'execution/conditional/parameter_transfer_cuda.hh', 'execution/conditional/parameter_transfer.cu', 'parameter_transfer_smoke.cu'),
    'reaction_cuda': ('device reaction events', 'ops/reaction/apply_cuda.hh', 'ops/reaction/reactions.cu', 'reactions_smoke.cu'),
    'regulatory_gates_cuda': ('packed regulatory conditions', 'ops/condition/regulatory_gates_cuda.hh', 'ops/condition/regulatory_gates.cu', 'regulatory_gates_smoke.cu'),
}
native_operation_paths = {}
for target, (operation, header, source, check) in native_operation_owners.items():
    test_dir = 'cuda' if source else 'native'
    paths = [f'include/Cellerator/{header}']
    if source:
        paths.append(f'src/{source}')
    for path in paths:
        native_operation_paths[path] = (
            operation, f'Cellerator::{target}', f'keep the {operation} contract with its direct build owner',
            'retain_native_operation_owner', 'validation is operation-specific',
            'native callers, installed consumers and relevant adapters',
            f'tests/framework/{test_dir}/{check}' if check else 'host header consumer compilation',
            'preserve numerical, precision and storage contracts through the source move',
        )

# Canonical destinations after the 2026-11-10 compatibility retirement.
COMPATIBILITY_RETIREMENTS = {
    'include/Cellerator/compute/gene_candidate_discovery.hh': 'include/Cellerator/geometry/discovery/gene_candidate_discovery.hh',
    'include/Cellerator/compute/gene_support_bitset.hh': 'include/Cellerator/geometry/discovery/gene_support_bitset.hh',
    'include/Cellerator/compute/math/native_tile_view.hh': 'include/Cellerator/layout/projection/native_tile_view.hh',
    'include/Cellerator/compute/math/native_training_slice.hh': 'include/Cellerator/compute/training/native_training_slice.hh',
    'include/Cellerator/compute/math/operation.hh': 'components/reference/include/Cellerator/reference/operation.hh',
    'include/Cellerator/compute/math/operation_core/builtin_catalog.hh': 'include/Cellerator/compute/operation/builtin_catalog.hh',
    'include/Cellerator/compute/math/operation_core/csr_fallback_candidate.hh': 'include/Cellerator/ops/relation/csr_fallback_candidate.hh',
    'include/Cellerator/compute/math/operation_core/cusparse_csr_candidate.hh': 'include/Cellerator/compute/candidate/cusparse_csr_candidate.hh',
    'include/Cellerator/compute/math/operation_core/feature_major_small_n_candidate.hh': 'include/Cellerator/ops/relation/feature_major_small_n_candidate.hh',
    'include/Cellerator/compute/math/operation_core/operation_core.hh': 'include/Cellerator/compute/operation/operation_core.hh',
    'include/Cellerator/compute/math/operation_core/preparation_factory.hh': 'include/Cellerator/compute/operation/preparation_factory.hh',
    'include/Cellerator/compute/math/operation_core/row_masked_n1_candidate.hh': 'include/Cellerator/ops/relation/row_masked_n1_candidate.hh',
    'include/Cellerator/compute/math/operation_core/transpose_backward_candidate.hh': 'include/Cellerator/compute/candidate/transpose_backward_candidate.hh',
    'include/Cellerator/compute/math/packed_dense_operand.hh': 'include/Cellerator/layout/projection/packed_dense_operand.hh',
    'include/Cellerator/compute/math/physical_bell.hh': 'include/Cellerator/layout/projection/bell.hh',
    'include/Cellerator/compute/math/referee.hh': 'components/reference/include/Cellerator/reference/referee.hh',
    'include/Cellerator/compute/math/tensor_core/v100_dense_fragment_candidate.hh': 'include/Cellerator/compute/candidate/tensor_core/v100_dense_fragment_candidate.hh',
    'include/Cellerator/compute/math/tensor_core/v100_dense_fragment_plan.hh': 'include/Cellerator/compute/candidate/tensor_core/v100_dense_fragment_plan.hh',
    'include/Cellerator/compute/matrix/convert/transpose.cuh': 'include/Cellerator/layout/conversion/compressed.cuh',
    'include/Cellerator/compute/operation/native_numeric/local_arithmetic.hh': 'include/Cellerator/ops/numeric/local_arithmetic.hh',
    'include/Cellerator/compute/operation/sparse_support/training.hh': 'include/Cellerator/ops/sparse_support/training.hh',
    'include/Cellerator/compute/runtime.hh': 'include/Cellerator/runtime/multi_gpu/fleet.cuh',
    'include/Cellerator/compute/sequence/baseplane_integration.cuh': 'include/Cellerator/execution/sequence/regulatory.cuh',
    'include/Cellerator/interop/baseplane/baseplane_integration.cuh': 'include/Cellerator/execution/sequence/regulatory.cuh',
    'include/Cellerator/compute/sparse/ops.hh': 'include/Cellerator/compute/operators/sparse/ops.hh',
    'include/Cellerator/compute/sparse/project.hh': 'include/Cellerator/compute/candidate/sparse/project.hh',
    'include/Cellerator/dist/distributed.cuh': 'include/Cellerator/runtime/multi_gpu/context.cuh',
    'include/Cellerator/dist/nccl_communicator.cuh': 'include/Cellerator/runtime/multi_gpu/nccl_communicator.cuh',
    'include/Cellerator/dist/nccl_select.cuh': 'include/Cellerator/runtime/multi_gpu/nccl_select.cuh',
    'include/Cellerator/distributed/hierarchy.hh': 'include/Cellerator/execution/planning/distributed/hierarchy.hh',
    'include/Cellerator/geometry/optimizer/oracle/optimizer_portfolio_contract.h': 'include/Cellerator/geometry/optimizer/portfolio_v1.hh',
    'include/Cellerator/interop/cellshard.cuh': 'include/Cellerator/interop/cellshard/matrix.cuh',
    'include/Cellerator/interop/cellshard_access.cuh': 'include/Cellerator/interop/cellshard/access.cuh',
    'include/Cellerator/math/matrix/patch.hh': 'include/Cellerator/ops/matrix/patch.hh',
    'include/Cellerator/math/process/ports.hh': 'include/Cellerator/ops/process/ports.hh',
    'include/Cellerator/models/developmental_time.hh': 'examples/models/developmental_time/developmentalTime.hh',
    'include/Cellerator/models/developmental_time_trajectory.hh': 'examples/models/developmental_time_trajectory/developmentalTimeTrajectory.hh',
    'include/Cellerator/models/state_reduce.hh': 'examples/models/state_reduce/stateReduce.hh',
    'include/Cellerator/trajectory/trajectory_tree.cuh': 'examples/trajectory/trajectory_tree.cuh',
    'include/Cellerator/types.cuh': 'include/Cellerator/common/types.cuh',
    'include/Cellerator/runtime/legacy_execution_context.hh': 'include/Cellerator/runtime/multi_gpu/fleet.cuh',
    'src/common/real.cuh': 'include/Cellerator/common/real.cuh',
    'src/common/types.cuh': 'include/Cellerator/common/types.cuh',
    'src/common/cuda_compat.cuh': 'include/Cellerator/common/cuda_compat.cuh',
    'include/Cellerator/math/effects/providers.hh': 'include/Cellerator/ops',
    'include/Cellerator/execution/program/program_v1_adapter.h': 'include/Cellerator/execution/program/stages.hh',
    'src/execution/program/program_v1_adapter.cc': 'src/execution/program/stages.cc',
}

# Order matters: narrow exceptions precede broad tree rules.
def classify(p):
    if p == 'tests/framework/native/CMakeLists.txt':
        return ('native host operation checks', p, 'register checks against their direct operation owners',
                'retain_native_host_operation_checks', 'host CTest and optional Python binding registration',
                'native operation and execution owners', 'focused host CTest suite', 'preserve existing check coverage')
    if p in modeling_migrations:
        destination = modeling_migrations[p]
        return ('retired native modeling source path', destination,
                f'use the canonical path {destination}', 'record_canonical_native_source_move',
                'historical source path', 'native callers and installed consumers',
                'active source migration closure and focused owner checks',
                'preserve historical bytes and qualification in Git and source-bound evidence')
    if p == 'src/compute/tokenization/gene_annotation.cc':
        return ('former annotation reader path', 'components/annotation/src/gene_annotation.cc',
                'route format parsing to Annotation and vocabulary construction to Tokenization',
                'record_canonical_native_source_move', 'historical source path', 'annotation reader and GeneVocabulary',
                'annotation and vocabulary format tests', 'retain source-linked behavior and format evidence')
    moved_component_paths = {
        'include/Cellerator/nn': 'components/nn/include/Cellerator/nn',
        'src/nn/': 'components/nn/src/',
        'tests/nn/': 'components/nn/tests/',
        'examples/native_nn/': 'components/nn/examples/',
        'src/interop/anndata/': 'components/anndata/',
        'tests/interop/anndata/': 'components/anndata/tests/',
        'include/Cellerator/compute/tokenization/': 'components/tokenization/include/Cellerator/compute/tokenization/',
        'src/compute/tokenization/': 'components/tokenization/src/',
    }
    for old_prefix, new_prefix in moved_component_paths.items():
        if p.startswith(old_prefix):
            destination = new_prefix + p[len(old_prefix):]
            return ('former optional component source path', destination, 'use the component source owner',
                    'record_canonical_native_source_move', 'historical ownership path', 'optional native package consumers',
                    'focused component and installed consumer checks', 'preserve historical source and evidence')
    if p in native_operation_paths:
        return native_operation_paths[p]
    if p == 'include/Cellerator/execution/dependency/value_field.hh':
        return ('identity-addressed value storage and write publication', 'Cellerator::dependency_execution',
                'keep the value generation beside the execution state that observes it',
                'retain_native_value_field', 'host ownership and generation contract',
                'native-data and optional host bindings', 'tests/framework/native/sparse_modeling_test.cc',
                'preserve identity-addressed writes and value generation semantics')
    if p.startswith(('include/Cellerator/execution/sequence/', 'src/execution/sequence/')):
        return ('sequence operand binding and prepared regulatory execution', 'internal cellerator_sequence_integration target', 'bind borrowed native sequence planes and run the supported forward predicate and regulatory projection', 'retain_native_sequence_execution', 'bounded forward exact-mask path', 'tests/execution/sequence and the biological operand checks', 'sequence_regulatory CTest and CUDA qualification', 'preserve axis, source generation, readiness, and projection identity')
    if p.startswith('components/sequence/'):
        return ('native genomic sequence component', 'Cellerator::sequence', 'retain exact sequence provenance, encodings, packing, scans, masks, detail queries, and predicate preparation', 'retain_native_sequence_operations', 'host module and optional CUDA operations', 'native sequence callers and regulatory execution', 'tests/sequence, tests/execution/sequence, and bench/sequence', 'preserve coordinates, validity, halo ownership, ordering, and capacity')
    if p.startswith(('include/Cellerator/ops/sequence/', 'src/ops/sequence/')):
        if p.rsplit('/', 1)[-1].startswith('event_memory'):
            return ('model trace event memory', 'Cellerator event_memory host and CUDA targets', 'retain event trace evaluation', 'retain_event_memory', 'host and optional CUDA providers', 'native modeling and framework CUDA', 'tests/framework/cuda/memory_smoke.cu', 'preserve event memory layout and execution semantics')
        if p.rsplit('/', 1)[-1].startswith('packed_embedding'):
            return ('model packed embeddings', 'Cellerator::packed_embedding_cuda', 'retain packed occurrence embedding operations', 'retain_packed_embedding', 'optional CUDA provider', 'framework CUDA', 'tests/framework/cuda/packed_embedding_smoke.cu', 'preserve occurrence identity and packed state-code contracts')
    if p == 'packages/anndatapp':
        return ('pinned AnnData++ source dependency', 'AnnData++ upstream repository', 'bundle the indexed gitlink revision for the optional AnnData adapter', 'retain_pinned_anndatapp', 'standard native build includes H5AD and Zarr when their backends are enabled', 'anndatapp::anndatapp and Cellerator::anndata', 'pinned source archive and installed H5AD/Zarr consumer checks', 'keep upstream module and package names and update the pin explicitly')
    if p.startswith('components/annotation/'):
        return ('native annotation catalog and GTF/GFF3 I/O', 'components/annotation', 'optional host annotation component', 'retain_reference_annotation', 'plain-text annotation input', 'Cellerator::annotation and GeneVocabulary', 'component annotation format checks', 'preserve identifiers, aliases, format selection and diagnostics')
    if p.startswith('components/anndata/'):
        return ('AnnData++ backed reads into native views', 'components/anndata', 'optional native acquisition adapter', 'retain_anndata_adapter', 'AnnData++ backed array reads', 'Cellerator::anndata', 'component read and installed consumer checks', 'preserve native buffer ownership and the AnnData++ package boundary')
    if p in ('src/cellerator.ccm', 'src/tensor/cellerator.tensor-partition.ccm', 'cmake/CelleratorPrimaryModule.cmake'):
        return ('native primary C++ module interface', 'Cellerator::cellerator', 'export canonical tensor declarations through the primary module', 'retain_primary_module', 'host C++20 module frontend', 'import cellerator', 'tests/tensor/modules', 'preserve type identity with native headers')
    for component in ('velocity', 'preprocess', 'neighbors'):
        if p.startswith(f'components/{component}/'):
            return ('native numerical component', f'Cellerator::{component}', 'use the component target and its native interfaces', 'retain_native_method_component', 'host methods or CUDA routed search; imported evidence retains its original source identity', f'Cellerator::{component}', 'component regression and server validation', 'preserve borrowed layouts, ordering, and caller storage')
    if p.startswith('tests/single_cell/'):
        return ('single-cell component composition checks', 'tests/single_cell', 'exercise the direct native method targets together', 'retain_single_cell_composition_checks', 'build-tree, installed, and optional file-backed checks', 'Cellerator::preprocess and Cellerator::velocity', 'server validation', 'preserve independent AnnData discovery')
    if p == '.vscode/settings.json':
        return ('local editor configuration', '.vscode/settings.json', 'retain developer editor preferences without treating paths as package dependency authority', 'retain_editor_configuration', 'editor metadata only', 'local development tools', 'preserve existing preferences; builds use maintained CMake contracts', 'do not rewrite local developer settings as part of component detachment')
    if p.startswith(('components/CellShard/', 'src/preprocess/', 'src/ops/velocity/', 'tests/velocity/', 'include/Cellerator/interop/cellshard/', 'examples/torch/')) or p in {'cmake/CelleratorVelocity.cmake', 'cmake/CelleratorVelocityModule.cmake'}:
        return ('retained inactive data or scientific integration source', 'docs/development/component-ownership.md', 'preserve source in place outside supported Cellerator builds and installations; CCell owns scientific integration', 'retain_detached_component_source', 'inactive source; historical qualification is not current substrate qualification', 'no supported Cellerator target or installed adapter', 'preservation hashes and dependency-free host/SDK consumer checks', 'preserve original source, fixtures, local histories and source data; no algorithm transfer or new acceleration API')
    if p in COMPATIBILITY_RETIREMENTS:
        destination = COMPATIBILITY_RETIREMENTS[p]
        return ('retired source compatibility surface', destination,
                f'use the canonical owner at {destination}; no forwarding facade remains',
                'record_compatibility_retirement',
                'source migration only; qualification belongs to the current owner',
                'direct canonical includes and explicit DynamicalModel consumers',
                'affected source, installed consumers, and modeling checks; consult dated receipts',
                'preserve historical source/evidence and persisted contracts; no numeric or schema change')
    name = Path(p).name.lower()
    suf = Path(p).suffix.lower()
    if p.startswith('docs/history/compat/'):
        return ('source-bound compatibility archive', 'docs/history/compat/manifest.json', 'preserve exact historical sources and former paths; build only the canonical reference and projection owners', 'preserve_compatibility_archive', 'historical evidence only; migration and local checks do not extend old CUDA or timing qualification', 'historical readers and recovery tools; no current target compiles this tree', 'manifest SHA-256 and original Git blob verification', 'preserve original bytes, source revision and path provenance')
    if p.startswith('compat/') or p == 'include/Cellerator/compute/math/execution_plan.hh':
        archive = 'docs/history/compat/source/' + p + '.txt'
        manifest_path = REPO / 'docs/history/compat/manifest.json'
        entries = json.loads(manifest_path.read_text())['files'] if manifest_path.is_file() else []
        entry = next((entry for entry in entries if entry['original_path'] == p), {})
        owner = entry.get('canonical_path', archive)
        return ('retired compatibility source or forwarding header', owner, 'use the canonical owner where retained; recover exact historical bytes from docs/history/compat', 'record_retired_compatibility_source', 'retired path; historical evidence does not qualify the migrated owner', 'canonical numerical reference and projection consumers', 'archive manifest and migrated consumer checks', archive)
    if p.startswith('components/reference/'):
        return ('independent SpMM numerical reference and request validation', 'Cellerator::spmm_reference in components/reference/CMakeLists.txt', 'retain a shared oracle for numerical validation', 'retain_numerical_reference_owner', 'host reference and installed package', 'candidate tests, numerical validation and benchmark reporting', 'cpMathOperationContractTest, cpMathRefereeFoundationTest and the installed reference consumer', 'preserve numerical signatures, precision, transpose and epilogue meanings')
    if p.startswith(('src/layout/conversion/', 'include/Cellerator/layout/conversion/')):
        return ('CUDA compressed conversion and bucket transforms', 'Cellerator::compute_matrix_convert in src/layout/conversion/CMakeLists.txt', 'materialize reordered or converted sparse arrays in supplied storage', 'retain_layout_conversion', 'CUDA implementation and detached component callers', 'native sparse conversion and CellShard callers', 'coreSparseLayoutRuntimeTest', 'preserve index widths, order maps and caller stream semantics')
    if p in {'src/layout/projection/CMakeLists.txt', 'src/layout/projection/csr.cc', 'src/layout/projection/native_tile_view.cc', 'src/layout/projection/bell.cc', 'src/layout/projection/bell_candidates.cc', 'src/layout/projection/packed_dense_operand.cu', 'include/Cellerator/layout/projection/bell.hh', 'include/Cellerator/layout/projection/packed_dense_operand.hh', 'include/Cellerator/layout/projection/dense_layout.hh', 'include/Cellerator/layout/projection/native_tile_view.hh'}:
        return ('physical projection construction or packed operand helper', 'src/layout/projection/CMakeLists.txt; Cellerator::physical_projection and optional Cellerator::packed_dense_operand', 'keep borrowed CSR/native-tile adaptation and BELL construction with layout; CUDA packing depends on the physical owner without pulling numerical reference code into execution', 'retain_physical_projection_owner', 'host construction checks and installed headers; no new CUDA or performance qualification', 'native projection consumers, focused layout tests and optional packing benchmark', 'cpMathNativeTileAdapterTest, cpMathBellLoweringTest and installed projection consumer; CUDA checks deferred', 'preserve canonical recovery, value borrowing, versioned identities and explicit caller storage')
    if p.startswith(('include/Cellerator/layout/factorized_interaction', 'include/Cellerator/ops/mechanism/factorized_interaction', 'src/layout/factorized_interaction', 'src/ops/mechanism/factorized_interaction', 'src/ops/mechanism/prepared_factorized_interaction')):
        return ('factorized conditional joint-interaction layout and native CUDA evaluator', 'native factorized-interaction family; Cellerator::factorized_interaction_cuda with host preparation', 'retain typed physical preparation, explicit forward/VJP, and optional checked logical-pattern binding with the native numerical owner', 'retain_factorized_interaction_component', 'source-bound host, device, and performance qualification are recorded separately; source presence is not a speedup claim', 'native C++ model authors and optional ATen adapters; Torch is not a core dependency', 'tests/factorized_interactions_preparation, tests/factorized_interactions, bench/factorized_interactions and planning/interaction-layout-proposal-2026-11-02', 'preserve ordered/repeated occurrences, parameter identities, caller-owned numerical lifetimes, independent oracle and captured measurement evidence')
    if p in {'include/Cellerator/cells/access.hh', 'src/cell.cc', 'tests/cell/cell_views_bridge_test.cc'}:
        return ('retired native Cell runtime bridge', 'docs/native_nn.qmd',
                'use concrete typed views directly',
                'record_native_cell_runtime_bridge_retirement',
                'removed from the worktree and host package',
                'native C++ and installed header/module consumers',
                'focused host Cell, modeling, and installed consumer checks',
                'historical implementation and evidence remain in Git and the dated record')
    if p in {'include/Cellerator/modeling/model.hh', 'include/Cellerator/modeling/dynamical.hh', 'tests/framework/native/model_contract_test.cc'}:
        return ('retired native dynamical model wrapper or dedicated contract test',
                'tests/framework/native/cell_relation_example.cc',
                'call concrete transitions and observations over caller-owned state',
                'record_native_dynamical_model_retirement', 'historical source and test paths',
                'native C++ callers and installed native-data consumers',
                'direct transition, clock publication, and retained snapshot checks',
                'preserve historical implementation and evidence in Git')
    if p in {
        'include/Cellerator/cell.hh', 'include/Cellerator/modeling/cell.hh',
        'modules/cellerator.ccm', 'modules/cellerator.core.ccm', 'modules/CMakeLists.txt',
        'examples/native_cell/foundation.cc', 'tests/cell/cell_core_test.cc',
        'tests/cell/typed_core_no_rtti_test.cc', 'tests/cell/install_consumer/core.cc',
        'tests/modules/module_contract.cc', 'tests/modules/installed_consumer/CMakeLists.txt',
        'components/nn/tests/native/cell_contract_test.cc', 'components/nn/tests/native/cell_query_integration_test.cc',
    }:
        return ('retired native Cell binding or demonstration', 'docs/native_nn.qmd',
                'call models with concrete data and use typed views directly',
                'record_native_cell_retirement', 'historical source and test paths',
                'native C++ callers and installed NN consumers',
                'native adapter, model, recurrent, query, and installed module checks',
                'preserve historical implementation and evidence in Git')
    if p.startswith(('include/Cellerator/cells/', 'tests/cell/', 'examples/native_cell/')):
        return ('native event and expression selections over caller-owned storage', 'Cellerator::headers; include/Cellerator/cells; tests/cell and examples/native_cell', 'expose concrete event prefixes and expression rows with typed read and edit access', 'retain_native_cell_access', 'host descriptor and backing-storage contracts', 'native C++ callers and the optional NN interface', 'adapter, installed consumer, and native examples', 'preserve view lifetime, logical identity, physical position, and owner mutation contracts')
    if p == 'include/Cellerator/geometry/CP_BP_V1_COMPATIBILITY.md':
        return ('migrated CP-BP compatibility design note', 'docs/development/packed_geometry_images.md', 'the packing image guide records format boundaries and points to current code and tests', 'record_canonical_native_document_move', 'documentation provenance only', 'native geometry contributors', 'review the packed image readers and current consumers', 'preserve the protected cellular geometry definition and keep format compatibility explicit')
    if p == 'src/quantized/README.md':
        return ('retired quantized source-folder guide', 'tensor::quantized and ops::quantized source families', 'the former src/quantized folder no longer owns a monolithic format/operation layer; current encoding views and kernels have separate canonical owners', 'record_retired_quantized_subsystem_guide', 'former documentation path is removed from the worktree; do not infer implementation or performance from its prose', 'maintainers of the split tensor format and operation families', 'review current headers, build targets, exact numerical tests, and operation-specific device receipts', 'preserve the historical design note as provenance without carrying forward stale ownership or universal policy claims')
    if p.startswith('planning/archive/legacy-todo/'):
        return ('legacy Todo archive index or preservation manifest', 'planning/archive/legacy-todo', 'resolve exact historical bytes in the verified durable archive by former path; current work belongs to the supported Project Control authority', 'preserve_archived_legacy_todo', 'historical workflow context, not current execution state or authority', 'archive README and historical readers', 'archive hashes and former-path checks only; do not replay legacy state as current authority', 'preserve original bytes and managed historical references; use the archive index to resolve former paths')
    if p in {'.todo-orchestrator/project.json', '.todo-orchestrator/authority.sqlite3', '.todo-orchestrator/authority.sqlite3-wal', '.todo-orchestrator/authority.sqlite3-shm'}:
        return ('current Project Control project metadata or authority database', 'Project Control / Todo authority', 'keep under supported workflow; never hand-edit metadata or authority state', 'managed_protected', 'current authority state', 'query supported workflow', 'authority/evidence via supported workflow', 'not applicable')
    if p.startswith('.todo-orchestrator/runtime/') or p in {'.todo-orchestrator/state.snapshot.json', '.todo-orchestrator/ce-ptr-plan.json'} or p.startswith('todos/') or p in {'todos.md','todo-status.md'}:
        return ('legacy Todo projection, snapshot, plan input, or runtime artifact', 'planning/archive/legacy-todo', 'preserve as historical material with its former path and content identity; do not treat as current authority', 'archive_after_verified_backup', 'historical workflow material, not current execution state or authority', 'legacy Todo archive index and historical readers', 'archive hashes and former-path checks only; current status comes from supported workflow', 'preserve original bytes and managed historical references; use the archive index to resolve former paths')
    if p.startswith('.todo-orchestrator/'):
        return ('Project Control workflow metadata', 'Project Control / Todo authority', 'keep under supported workflow; inspect the specific file before changing its lifecycle', 'managed_metadata_review', 'workflow metadata; authority status depends on the specific file', 'query supported workflow', 'authority/evidence via supported workflow', 'not applicable')
    if p.startswith('.autobench/'):
        return ('Autobench registry input or ignore metadata', '.autobench registry and its supported Autobench tooling', 'retain benchmark definitions and pause metadata at their source paths; use Autobench for registration and execution', 'retain_autobench_registry_metadata', 'configuration metadata only; active or paused definitions are not measurement or qualification evidence', 'Autobench registry readers and maintainers of the named benchmark definitions', 'validate through Autobench; only source-bound execution receipts establish the result of a run', 'preserve each definition and status marker; do not infer that a paused or registered benchmark ran')
    if p == 'include/CellPack/README.md':
        return ('retired CellPack forwarding-header guide', 'docs/development/packed_geometry_images.md', 'use the canonical Cellerator/geometry headers', 'record_canonical_native_document_move', 'documentation provenance only', 'native geometry contributors and installed header consumers', 'audit include paths and canonical header consumers', 'preserve historical evidence without retaining obsolete forwarding headers')
    if p.startswith('include/CellPack/') and p != 'include/CellPack/semantic_geometry.hh':
        target = p.replace('include/CellPack/', 'include/Cellerator/geometry/', 1)
        return ('retired CellPack forwarding header', target, 'include the canonical geometry header directly; no implementation lived in this wrapper', 'record_canonical_native_source_move', 'source-compatibility include path removed after consumer audit; numerical and payload contracts unchanged', 'native geometry and external C++ consumers', 'canonical header compilation and focused host geometry tests; CUDA qualification remains separate', 'preserve cellpack namespace, CellPack:: targets and versioned payload meanings')
    if p.startswith(('src/','include/Cellerator/','tests/','bench/','examples/')) and Path(p).suffix.lower() in {'.md','.rst','.qmd'}:
        parent = str(Path(p).parent)
        return ('subsystem-local engineering or test documentation', parent, f'retain beside {parent} as subsystem guidance; do not treat prose as an API or capability implementation', 'retain_local_subsystem_documentation', 'documentation presence; capability status comes from cited source and evidence', f'{parent} contributors and maintainers of linked implementations', 'review links and cited source paths; a README is not build or runtime evidence', 'preserve its context when source is moved and update links with the owning code')
    if p == 'include/Cellerator/biology/coordinates.hh':
        return ('migrated canonical occurrence coordinates', 'include/Cellerator/structure/coordinates.hh', 'ordered semantic coordinates and occurrence tokens are independent of biological annotation resolution and physical slot maps', 'record_canonical_native_source_move', 'former tracked path is removed from the worktree; current interface and tests belong to the canonical destination', 'audit includes and installed consumers of structure::OccurrenceCoordinates', 'identity and occurrence consumers by exact contract; this route does not assert that each check ran', 'preserve occurrence order and distinguish semantic identity from physical position')
    if p == 'include/CellPack/semantic_geometry.hh':
        return ('migrated CP-BP semantic/physical geometry contracts', 'include/Cellerator/layout/cell_packing_geometry.hh and include/Cellerator/geometry/strategy/cpbp_v1_compatibility.hh', 'the former combined API was split into physical statistics and explicit adaptation of a validated CellPack payload', 'record_canonical_native_source_move', 'former tracked path is removed from the worktree; exact source and runtime evidence remain destination-specific', 'CellPack payload callers and Cellerator layout/geometry consumers', 'CellPack compatibility and layout-statistics checks by exact contract; no blanket support claim', 'preserve payload version meaning and keep physical summaries separate from semantic relations')
    if p.startswith('include/Cellerator/biology/'):
        return ('native biological identity and alignment contracts', 'include/Cellerator/biology; native NN consumers under components/nn', 'retain caller-defined biological coordinates, feature bindings, aliases, and prepared alignment independently of storage', 'retain_native_biological_boundary_contracts', 'concurrent native NN source; qualification is separate from the Python-facing NN pass', 'native modeling and NN source; inspect current CMake registration and actual consumers', 'components/nn/tests/native and native NN qualification records when available; source presence is not passing evidence', 'preserve typed identity, order, unknown-policy, and ownership semantics; do not infer Python/C++ authoring parity')
    if p.startswith(('components/nn/include/', 'components/nn/src/')):
        return ('native neural composition and differentiable operation source', 'components/nn; inspect top-level CMake registration', 'retain native parameters, layers, interactions, queries, and numerical adapters as an independently qualified modeling surface', 'retain_native_nn_source', 'concurrent native NN implementation; build and runtime qualification remain operation-specific', 'native C++ modeling consumers, optional ATen dependencies, and exact NN tests', 'components/nn/tests/native and source-bound native NN evidence; prior Python tests do not qualify this new source', 'preserve idiomatic native ownership and biological meaning without claiming all components are implemented or qualified')
    if p.startswith(('bindings/nn/', 'python/bindings/nn/')):
        return ('optional Python adapter for shared native neural functions', 'python/bindings/nn with components/nn and python/cellerator/torch', 'retain a thin borrowed-parameter bridge to shared native functions; preserve ordinary Python parameter ownership and explicit optional loading', 'retain_native_nn_python_adapter', 'binding source only; extension import and numerical composition need their own qualification', 'optional _nn_native extension and Python NN facade', 'tests/framework/test_native_nn_bridge.py and isolated native-enabled wheel validation', 'coordinate signatures with native functional owners and Python frontend owners; do not mirror Python registration in native model authoring')
    if p.startswith('components/nn/tests/'):
        return ('native NN contracts and numerical checks', 'components/nn/tests with components/nn/include, include/Cellerator/biology, and components/nn/src', 'retain beside the native NN implementation; execute only with the required optional dependencies and hardware authority', 'retain_native_nn_checks', 'test source only; collection, compilation, and passing execution are separate evidence', 'native NN CMake test registration and operation-specific qualification runners', 'inspect the exact test and its admitted execution receipt; no blanket native or CUDA acceptance', 'preserve independent references, gradient checks, and device/precision contracts')
    if p == 'cmake/CelleratorTorchDependency.cmake':
        return ('shared optional Torch and ATen dependency discovery', 'cmake/CelleratorTorchDependency.cmake and its current CMake include sites', 'retain consistent optional native/Python Torch ABI discovery while allowing an explicit LibTorch prefix', 'retain_optional_torch_dependency_discovery', 'build configuration source; successful configured builds require separate evidence', 'optional native NN, tokenization, and Python binding build registration', 'CMake configure/build acceptance with the selected dependency installation; not numerical qualification', 'preserve optional dependency boundaries and caller-visible package/ABI selection')
    if p.startswith('include/Cellerator/examples/'):
        destination = p.removeprefix('include/Cellerator/')
        return ('former native example header path', destination, f'use {destination} in source builds and retain the installed Cellerator/examples include path', 'record_canonical_native_source_move', 'source moved with existing example contracts', 'model and trajectory tests and installed consumers', 'examples/install_headers.cmake and tests/CMakeLists.txt', 'retain biological assumptions with consumer')
    migrated_native_paths = {
        'include/Cellerator/structure/identity_registry.hh': ('execution::identity_registry', 'include/Cellerator/execution/identity_registry.hh', 'process-local registry handles remain separate from persistent structure identities', 'tests/execution_identity/identity_registry_test.cc'),
        'include/Cellerator/biology/coordinates.hh': ('structure occurrence coordinates', 'include/Cellerator/structure/coordinates.hh', 'canonical occurrence order and domain coordinates are separate from layout-owned physical positions', 'native-data identity and occurrence consumers by exact contract'),
        'include/Cellerator/compute/decomposition/gate_input_v1.hh': ('execution::decomposition::GateInputV1', 'include/Cellerator/execution/decomposition/gate_input.hh', 'runtime operand, replication, and decomposition association stays distinct from semantic condition reads', 'gate-input contract and decomposition consumers by exact contract'),
        'src/compute/decomposition/gate_input_v1.cc': ('execution::decomposition::GateInputV1 validation', 'src/execution/decomposition/gate_input.cc', 'validation moved with the runtime decomposition contract', 'gate-input contract and decomposition consumers by exact contract'),
        'include/Cellerator/compute/gating/condition_program.hh': ('ops::condition::ProgramViewV1', 'include/Cellerator/ops/condition/program.hh', 'condition evaluation is a reusable boolean operation and does not define biological dependency knowledge', 'condition program and execution workset tests by exact contract'),
        'src/gating/condition_program.cc': ('ops::condition condition evaluator', 'src/ops/condition/program.cc', 'validation, bitmap evaluation, and dirty-root updates moved with the condition operation', 'condition program and execution workset tests by exact contract'),
        'include/Cellerator/compute/indexing/reverse_postings.hh': ('containers::PostingIndex and geometry::dependency adapter', 'include/Cellerator/containers/posting_index.hh and include/Cellerator/geometry/dependency/posting_index.hh', 'generic posting storage/query operations and biological compact-support adaptation now have separate owners', 'native-data posting and dependency geometry tests by exact contract'),
        'src/indexing/reverse_postings.cc': ('containers::PostingIndex and geometry::dependency builder', 'src/containers/posting_index.cc and src/geometry/dependency/posting_index.cc', 'generic canonicalization/query and support-to-posting conversion were split by responsibility', 'native-data posting and dependency geometry tests by exact contract'),
        'include/Cellerator/compute/math/physical_csr.hh': ('layout::projection::CsrView', 'include/Cellerator/layout/projection/csr.hh', 'borrowing physical projection adapter only; conventional CSR storage and computation remain external', 'CP Math compatibility and selected CSR consumer checks'),
        'include/Cellerator/compute/math/physical_feature_major.hh': ('layout::projection::FeatureMajor', 'include/Cellerator/layout/projection/feature_major.hh', 'immutable physical feature maps are separate from current values and execution handles', 'feature-major and native training tests by exact contract'),
        'include/Cellerator/compute/math/physical_transpose.hh': ('layout::projection::Transpose', 'include/Cellerator/layout/projection/transpose.hh', 'transpose maps retain canonical recovery while live forward-value association remains execution-owned', 'transpose and native training tests by exact contract'),
        'include/Cellerator/compute/support/logical_support.hh': ('geometry::dependency compact and typed patterns', 'include/Cellerator/geometry/dependency/compact_support.hh and explicit_pattern.hh', 'ordered logical incidences and repeated participants remain semantic; physical projection and mutable execution work are separate', 'dependency geometry and native-data reference tests by exact contract'),
        'src/support/logical_support.cc': ('geometry::dependency validation plus execution support lifecycle', 'src/geometry/dependency/compact_support.cc and src/execution/dependency/support_lifecycle.cc', 'snapshot validation and epoch/tape publication checks now have distinct semantic and execution owners', 'dependency snapshot and lifecycle tests by exact contract'),
        'include/Cellerator/geometry/semantic_geometry.hh': ('layout packing summary and CP-BP compatibility adapter', 'include/Cellerator/layout/cell_packing_geometry.hh and include/Cellerator/geometry/strategy/cpbp_v1_compatibility.hh', 'physical CP-BP statistics are layout metadata, while semantic binding to validated payloads remains a geometry adapter', 'CellPack compatibility and layout statistics checks by exact contract'),
        'include/CellPack/semantic_geometry.hh': ('layout packing summary and CP-BP compatibility adapter', 'include/Cellerator/layout/cell_packing_geometry.hh and include/Cellerator/geometry/strategy/cpbp_v1_compatibility.hh', 'the former combined header was split between physical statistics and an explicit CellPack payload adapter', 'CellPack compatibility and layout statistics checks by exact contract'),
        'src/geometry/semantic_geometry.cc': ('layout packing statistics and CP-BP compatibility adapter', 'src/layout/cell_packing_geometry.cc and src/geometry/planning/cpbp_v1_compatibility.cc', 'statistics evaluation and validated-payload adaptation now have separate layout and geometry owners', 'CellPack compatibility and layout statistics checks by exact contract'),
        'src/layout/physical_csr.cc': ('layout::projection::CsrView adapter', 'src/layout/projection/csr.cc', 'the implementation now belongs to the versioned CP Math compatibility adapter; it does not own a CSR engine', 'CP Math compatibility consumer checks'),
        'src/layout/physical_feature_major.cc': ('layout::projection::FeatureMajor', 'src/layout/projection/feature_major.cc', 'canonical-to-physical feature mapping moved to the projection family', 'feature-major and native training tests by exact contract'),
        'src/layout/physical_mma_hybrid.cc': ('layout::projection::MmaHybrid', 'src/layout/projection/mma_hybrid.cc', 'physical hybrid metadata moved to the projection family', 'MMA cover, roundtrip, and native training tests by exact contract'),
        'src/layout/physical_mma_hybrid_builder.cc': ('layout::projection::MmaHybrid builder', 'src/layout/projection/mma_hybrid_builder.cc', 'the host metadata builder moved with the physical projection implementation', 'MMA residual builder and projection tests by exact contract'),
        'src/layout/physical_transpose.cc': ('layout::projection::Transpose', 'src/layout/projection/transpose.cc', 'transpose mapping moved to the projection family; runtime association remains explicit', 'transpose and native training tests by exact contract'),
        'include/Cellerator/compute/projection/identity.hh': ('layout::projection', 'include/Cellerator/layout/projection/identity.hh', 'value-free maps and execution-owned live handles remain separate', 'projection activation and layout tests by exact contract'),
        'include/Cellerator/compute/projection/physical_csr.hh': ('layout::projection::CsrView', 'include/Cellerator/layout/projection/csr.hh', 'borrowing adapter only; conventional sparse execution remains external', 'CP Math compatibility and selected CSR consumer checks'),
        'include/Cellerator/compute/projection/physical_feature_major.hh': ('layout::projection::FeatureMajor', 'include/Cellerator/layout/projection/feature_major.hh', 'immutable maps are separated from execution-owned current values', 'feature-major and native training tests by exact contract'),
        'include/Cellerator/compute/projection/physical_mma_hybrid.hh': ('layout::projection::MmaHybrid', 'include/Cellerator/layout/projection/mma_hybrid.hh', 'physical projection metadata remains separate from semantic geometry and live values', 'MMA cover, residual builder, roundtrip, and value-pack tests'),
        'include/Cellerator/compute/projection/physical_transpose.hh': ('layout::projection::Transpose', 'include/Cellerator/layout/projection/transpose.hh', 'forward-value association is an execution binding, not a persistent layout ID lookup', 'transpose and native training tests by exact contract'),
        'include/Cellerator/modeling/entity_layout.hh': ('layout::EntityLayout', 'include/Cellerator/layout/entity_layout.hh', 'physical feature order is separate from semantic axes and live value bindings', 'entity-layout and projection tests'),
        'include/Cellerator/modeling/incidence_plan.hh': ('ops::incidence and execution::PreparedIncidence', 'include/Cellerator/ops/incidence.hh plus include/Cellerator/execution/dependency/incidence.hh', 'address preparation and reusable gather/reduce/scatter arithmetic have separate owners', 'ordered-incidence and native-data reference tests'),
        'include/Cellerator/modeling/incremental.hh': ('execution::dependency', 'include/Cellerator/execution/dependency/incremental.hh', 'mutable work and condition invalidation remain execution state over owned semantic postings', 'incremental and condition-index tests'),
        'include/Cellerator/modeling/sparse_relation.hh': ('geometry::dependency', 'include/Cellerator/geometry/dependency/explicit_pattern.hh and pattern_owner.hh', 'immutable logical patterns retain ordered/repeated participants; physical layout and current activity remain separate', 'sparse modeling, geometry, and native-data reference tests'),
        'src/modeling/sparse_relation.cc': ('geometry::dependency plus execution::dependency', 'src/geometry/dependency and src/execution/dependency source families', 'the duplicate compiled relation owner was split across semantic pattern and prepared execution contracts', 'native-data geometry, pattern, incremental, and condition-index tests'),
        'include/Cellerator/modeling/cuda/runtime.hh': ('memory::cuda', 'include/Cellerator/memory/cuda/resources.hh', 'explicit-device DeviceBuffer, PinnedBuffer, Stream, Event, and views move below model operations', 'CUDA resource and migrated consumer compile/runtime checks'),
        'src/modeling/cuda/runtime.cu': ('memory::cuda', 'src/memory/cuda/resources.cu', 'CUDA resource implementation moves to its device/memory owner', 'CUDA resource and migrated consumer compile/runtime checks'),
        'include/Cellerator/modeling/cuda/quantization.hh': ('tensor::state_codes', 'include/Cellerator/tensor/state_codes/view.hh', 'codebook state format is distinct from low-bit sparse matrix values', 'state-code format and packed embedding checks'),
        'src/modeling/cuda/quantization.cu': ('tensor::state_codes', 'src/tensor/state_codes/operations.cu and owner.cu', 'state-code ownership and kernels are split by reusable format/operation responsibility', 'state-code owner and operation checks'),
        'include/Cellerator/runtime/device_buffer.cuh': ('memory::cuda::SharedDeviceBuffer', 'include/Cellerator/memory/cuda/shared_buffer.hh', 'one canonical explicit-device owner replaces the duplicated runtime buffer owner', 'shared buffer and direct consumer compile/runtime checks'),
        'src/quantized/README.md': ('tensor::quantized and ops::quantized', 'include/Cellerator/tensor/quantized and include/Cellerator/ops/quantized', 'format views and encoding/decoding operations now live in their distinct native owners', 'quantized format and operation tests by exact contract'),
    }
    if p in migrated_native_paths:
        owner, destination, meaning, tests = migrated_native_paths[p]
        return (f'migrated native source path for {owner}', owner, f'the canonical implementation now lives at {destination}; {meaning}', 'record_canonical_native_source_move', 'former tracked path is removed from the worktree; current implementation and API evidence belong to the canonical destination', f'audit direct includes, target links, and installed consumers at {destination}', f'{tests}; this route records ownership and does not assert that every listed check ran', 'preserve old-file provenance and explicit caller migration; do not retain a duplicate facade without a real consumer')
    if p.startswith('include/Cellerator/compute/operation/indexed_mechanism/') or p.startswith('src/compute/operation/indexed_mechanism/'):
        return ('migrated indexed mechanism source path', 'ops::mechanism::indexed / Cellerator::indexed_mechanism', 'canonical headers and split implementation units now live under include/Cellerator/ops/mechanism/indexed and src/ops/mechanism/indexed', 'record_canonical_native_source_move', 'former tracked path is removed from the worktree; host and CUDA qualification stays operation-specific', 'native_foundation callers and CMake target Cellerator::indexed_mechanism', 'ordered-incidence, n-ary host composition, evaluator/training tests, and admitted device checks by receipt', 'preserve incidence ordering, tape and parameter ownership semantics through caller migration')
    if p.startswith('include/Cellerator/quantized/'):
        return ('migrated low-bit sparse format or operation header', 'tensor::quantized and ops::quantized', 'canonical value views and encoding/decoding declarations are split between include/Cellerator/tensor/quantized and include/Cellerator/ops/quantized', 'record_canonical_native_source_move', 'former tracked path is removed from the worktree; each width/backend retains its own numerical and device evidence', 'quantized sparse projection and format consumers', 'quantized format/operator tests by exact contract; no general sparse implementation claim', 'preserve bit width, metadata, zero/overflow policies, and explicit legacy payload conversion')
    if p.startswith('include/Cellerator/matrix/'):
        matrix_migrations = {
            'dense.cuh': ('ATen or caller-owned dense storage plus interop::formats::DensePayload', 'include/Cellerator/interop/formats/dense.cuh'),
            'compressed.cuh': ('ATen or caller-owned CSR/CSC storage plus interop::formats::CompressedPayload', 'include/Cellerator/interop/formats/compressed.cuh'),
            'coo.cuh': ('ATen or caller-owned COO storage plus interop::formats::CooPayload', 'include/Cellerator/interop/formats/coo.cuh'),
            'blocked_ell.cuh': ('interop::formats::BlockedEllPayload', 'include/Cellerator/interop/formats/blocked_ell.cuh'),
            'sliced_ell.cuh': ('interop::formats::SlicedEllPayload', 'include/Cellerator/interop/formats/sliced_ell.cuh'),
            'quantized_blocked_ell.cuh': ('tensor::quantized::BlockedEllView and interop::formats::AffineBlockedEllCodec', 'include/Cellerator/tensor/quantized/blocked_ell.hh and include/Cellerator/interop/formats/affine_blocked_ell.hh'),
            'sparse.cuh': ('no conventional sparse owner; direct callers use ATen or an external implementation', 'legacy umbrella retired after direct callers move'),
            'views.cuh': ('format-specific views under tensor or interop::formats', 'legacy forwarding umbrella retired after direct callers move'),
        }
        item = matrix_migrations.get(Path(p).name)
        if item:
            owner, destination = item
            return ('migrated legacy matrix header', owner, f'the canonical interface is {destination}', 'record_canonical_native_source_move', 'former tracked path is removed from the worktree; codec compatibility and runtime support are format-specific', 'audit C++/CUDA and CellShard consumers before retiring wrappers', 'format-specific byte, consumer, or numerical checks; do not infer they all ran', 'do not rebuild conventional sparse execution inside Cellerator; migrate persisted payloads explicitly')
    if p.startswith('include/Cellerator/memory/cuda/') or p == 'src/memory/cuda/resources.cu':
        return ('explicit-device CUDA resource owners and shared buffers', 'Cellerator::memory_cuda; declarations under include/Cellerator/memory/cuda and implementation under src/memory/cuda', 'own streams, device and pinned allocations, events, and shared device-buffer storage with explicit device/stream selection', 'retain_native_cuda_resource_owner', 'CUDA resource sources and consumers are present; runtime behavior remains resource- and operation-specific', 'Cellerator::memory_cuda is linked by optional CUDA owners and direct out-of-line DeviceBuffer consumers', 'NVCC compilation and bounded resource smoke coverage where registered; build evidence does not qualify every allocation or asynchronous lifetime', 'keep memory ownership below model, tensor, and kernel layers; require callers to retain allocations and establish completion')
    if p in {'src/memory/allocation.cc', 'src/memory/cuda/copy.cu', 'src/memory/generation_marks.cc', 'src/memory/image.cc'}:
        return ('memory binding, image, copy and generation primitives', 'src/runtime/sources.cmake supplies Cellerator::runtime and the standalone training build', 'implement caller-visible memory contracts under src/memory', 'retain_native_memory_implementation', 'native host and CUDA implementation', 'memory, runtime, layout and execution callers', 'celleratorMemoryContractTest and celleratorSessionMemoryTest', 'preserve placement checks, byte extents and asynchronous lifetimes')
    if p.startswith('include/Cellerator/memory/'):
        return ('native memory views, placements, generations, and workspace contracts', 'include/Cellerator/memory; CUDA allocation owners live separately in memory/cuda', 'retain caller-visible memory and placement primitives independently of tensor encoding and model semantics', 'retain_native_memory_contracts', 'native headers are present; target, allocation, and runtime evidence depend on the specific contract', 'native data, execution, layout, and runtime consumers; CUDA owners are Cellerator::memory_cuda', 'memory, workspace, copy, and installed-consumer checks where registered; no blanket device-lifetime claim', 'preserve explicit placement, borrowing, and generation semantics')
    if p.startswith('include/Cellerator/interop/formats/'):
        return ('explicit codecs and borrowed conventional-format payloads', 'interop::formats; optional CellShard wrappers adapt selected payloads without owning compute kernels', 'convert recognized persisted formats and expose borrowed payload/device views; leave ordinary CSR/COO computation to established libraries or callers', 'retain_format_interoperability_boundary', 'codec and payload source is present; byte compatibility, package consumers, and device execution are format-specific', 'format payloads, CellPack image adapters, and components/CellShard adapters as individually registered', 'format-specific byte/oracle tests and installed consumers; header presence alone does not validate an archive migration', 'preserve versioned file meaning and ownership boundaries; require explicit migration for changed persisted data')
    if p.startswith('include/Cellerator/matrix/') or p in {'include/Cellerator/matrix.cuh','include/Cellerator/matrix/sparse.cuh'}:
        return ('legacy matrix headers and format facades', 'no canonical matrix-format owner; specialist types live under tensor and interoperability payloads live under interop::formats', 'migrate direct consumers to the canonical specialist, established library, or explicit codec interface; retire umbrella aliases after the last caller moves', 'retire_legacy_matrix_facades_after_consumer_migration', 'legacy paths are covered only if still present; they do not own conventional sparse compute', 'audit direct C++/CUDA includes, CellShard wrappers, and installed-header references', 'build consumer checks for each moved format family; source-map presence is not compatibility evidence', 'do not recreate a universal matrix facade or duplicate conventional sparse implementations')
    if p in {'include/Cellerator/parameters.hh','include/Cellerator/types.cuh'}:
        return ('native scalar parameter or numeric type definitions', 'native top-level data contracts', 'retain shared type definitions while format, operation, and memory ownership remain in their dedicated namespaces', 'retain_native_scalar_and_numeric_types', 'declaration source only; numeric policy and operation support are contract-specific', 'native tensor formats, operations, CUDA adapters, and package consumers', 'type/layout compile checks and exact operation tests; no general dtype support claim', 'preserve explicit storage, accumulation, output, and ABI meanings')
    if p.startswith('include/Cellerator/sdk/'):
        return ('compiler SDK facade and experimental API contract', 'compiler SDK facades in include/Cellerator/sdk; implementations remain owned by src/compiler/<family> and cmake/compiler/CelleratorHostSdk.cmake', 'retain the optional source-facing SDK facade; qualify each compiler API by its actual build and consumer evidence', 'retain_compiler_sdk_facades', 'installed-header checks qualify syntax; the built host SDK contains downstream invocation and tooling descriptors, while broader compiler APIs remain experimental', 'SDK examples, tests/sdk and public runtime facades; Cellerator::libCellerator supports its explicitly linked sources', 'tests/sdk/CMakeLists.txt and source-bound installed C++/C11 consumer checks; source or header presence does not establish complete compiler behavior', 'preserve frozen ABI and exact historical evidence; repair direct header, target and consumer paths when moving source')
    if p.startswith('include/Cellerator/dist/') or p.startswith('include/Cellerator/distributed/'):
        return ('distributed execution API or architecture', 'native distributed subsystem', 'hold behind user-approved scope; preserve any current consumers and mark optional', 'retain_scope_review_required', 'implementation_presence_unqualified', 'unknown; consumers and build feature gates required', 'tests/distributed candidate; qualification unresolved', 'preserve ABI and integration dependencies')
    if p in {'include/Cellerator/Cellerator.hh','include/Cellerator/abi.h','include/Cellerator/bindings/mechanism_handle.hh'}:
        return ('native public umbrella, ABI or binding handle', 'native public API boundary', 'retain direct native API while package-facing route is clarified', 'retain_public_native_boundary', 'public_contract_review_required', 'include consumers and ABI probes required', 'tests/biological_abi, tests/bindings candidate', 'preserve schema/ABI until explicit migration')
    if p in {'src/CMakeLists.txt','src/config.cuh.in'}:
        return ('native source build/configuration', 'native build system', 'retain and update atomically with source ownership moves', 'retain_build_boundary', 'build_infrastructure', 'all native targets and consumers', 'native builds/test registration', 'preserve feature gates')
    if p in {'src/compiler/driver/downstream_compiler_invocation.cc', 'include/Cellerator/compiler/driver/downstream_compiler_invocation_v1.hh'}:
        return ('installed host compiler invocation API', p.rsplit('/', 1)[0], 'retain the narrow downstream compiler invocation contract in the optional host SDK', 'retain_installed_host_compiler_invocation', 'built and installed by Cellerator::libCellerator; it forwards to the configured downstream compiler and is not a biological model compiler', 'cmake/compiler/CelleratorHostSdk.cmake defines libCellerator and installs compiler headers; tools/cellerator/driver_passthrough.cc consumes this API', 'compiler driver acceptance tests exercise bounded passthrough; source build does not qualify all compiler frontends or backends', 'preserve existing subprocess, diagnostics, and error propagation contract')
    if p in {'src/compiler/tooling/celleratord_architecture.cc', 'include/Cellerator/compiler/tooling/celleratord_architecture_v1.hh'}:
        return ('celleratord architecture descriptor and validation', p.rsplit('/', 1)[0], 'retain as the installed host SDK descriptor for supported language-server/tooling architecture', 'retain_installed_celleratord_architecture_contract', 'built and installed by Cellerator::libCellerator; the small descriptor validator does not imply a complete language server', 'cmake/compiler/CelleratorHostSdk.cmake links libCellerator into the celleratord executable; tests/compiler/tooling/celleratord_architecture_test.cc consumes the header', 'target and contract test coverage are bounded to architecture descriptor validation', 'preserve schema version and explicit no-fork/command/source-map invariants')
    if p in {'tools/cellerator/driver_passthrough.cc', 'tools/celleratord/celleratord_architecture.cc', 'cmake/compiler/CelleratorHostSdk.cmake'}:
        if p.startswith('tools/cellerator/'):
            owner, job, consumer = 'cmake/compiler/CelleratorHostSdk.cmake target cellerator', 'provide the installed host compiler passthrough executable', 'links Cellerator::libCellerator; exact driver contract is in src/compiler/driver/downstream_compiler_invocation.cc'
        elif p.startswith('tools/celleratord/'):
            owner, job, consumer = 'cmake/compiler/CelleratorHostSdk.cmake target celleratord', 'provide the installed compiler tooling architecture executable', 'links Cellerator::libCellerator; descriptor contract is implemented in src/compiler/tooling/celleratord_architecture.cc'
        else:
            owner, job, consumer = p, 'build and install the optional host SDK library, command-line tools, public headers, profiles, and standard library resources', 'defines libCellerator, cellerator, celleratord and the CelleratorPartOneTargets export; this bounded package surface is not the entire compiler research tree'
        return ('optional host compiler SDK build and executable surface', owner, job, 'retain_optional_host_compiler_sdk', 'CMake target and install rules are present; end-to-end language compilation is not implied', consumer, 'cmake/compiler/CelleratorPartOneAcceptance.cmake and registered compiler driver/tooling tests where enabled', 'keep the SDK separate from the general Python package and preserve install/export names')
    if p.startswith(('src/compiler/','include/Cellerator/compiler/')):
        parts = Path(p).parts
        base = 'src/compiler' if parts[0] == 'src' else 'include/Cellerator/compiler'
        family = parts[2] if parts[0] == 'src' else (parts[3] if len(parts) > 3 else 'public_root')
        compiler_families = {
            'api': ('compiler API and subsystem boundaries', 'declarations and boundary adapters'),
            'ast': ('compiler syntax tree and source identity', 'syntax ownership, traversal, diagnostics, and provenance'),
            'backend': ('compiler backend planning and emission', 'target-specific planning and object emission'),
            'composition': ('compiler program and component composition', 'compose compiler programs and module-level units'),
            'diagnostics': ('compiler diagnostics and provenance', 'explain validation, planning, and source decisions'),
            'discovery': ('compiler capability and input discovery', 'discover supported inputs and target resources'),
            'driver': ('compiler driver and invocation flow', 'coordinate compilation stages and command behavior'),
            'frontend': ('compiler source frontends', 'parse and lower supported source forms'),
            'ir': ('compiler intermediate representations', 'represent, inspect, and transform compiler programs'),
            'lto': ('compiler link-time composition', 'combine compiler units and link-time state'),
            'pass': ('compiler transformation passes', 'perform bounded transformations over compiler IR'),
            'planning': ('compiler candidate and execution planning', 'derive explainable execution plans from available structure'),
            'profile': ('compiler profile and evidence model', 'represent runtime evidence, alternatives, and profile-guided choices'),
            'reflection': ('compiler reflection and inspection', 'expose inspectable compiler metadata'),
            'sema': ('compiler semantic analysis', 'check source meaning and operation contracts'),
            'tooling': ('compiler developer tools', 'inspect, validate, and diagnose compiler inputs and artifacts'),
            'build': ('compiler public build configuration headers', 'expose generated/build configuration to native consumers'),
            'migration': ('compiler source migration interfaces', 'support bounded native integration/migration surfaces'),
            'program': ('compiler program interfaces', 'expose compiled-program contracts to native callers'),
            'substrate': ('compiler substrate integration interfaces', 'bridge compiler artifacts to native substrate contracts'),
            'public_root': ('compiler public root headers', 'preserve direct native compiler entry points'),
        }
        title, job = compiler_families.get(family, (f'compiler {family} subsystem', 'preserve its current subsystem contract'))
        owner = f'{base}/{family}'
        consumers = 'src/compiler/CMakeLists.txt and cmake/compiler/* define compiler integration; source files are not all proven to belong to a built target' if parts[0] == 'src' else f'include consumers and compiler target include interfaces; verify each header against build/install rules'
        tests = f'purpose-named tests/compiler suites, compiler CMake targets, and source-bound compiler records; inspect exact registration and assertion scope'
        maturity = 'research implementation present; no full source-to-device or live-recompile qualification implied'
        return (title, owner, f'retain at {owner}; {job}', 'retain_in_named_compiler_subsystem', maturity, consumers, tests, 'preserve task IDs only in planning/history; route code by subsystem purpose and preserve native interfaces')
    if p.startswith(('src/common/','include/Cellerator/common/')):
        return ('shared native helpers', 'include/Cellerator/common owns public definitions; src/common retains build configuration helpers', 'keep shared checks/types in common only while concrete source targets include or link them; CUDA-specific helpers stay internal', 'retain_shared_build_helpers', 'build_support_not_model_api', 'src/compute, src/layout, src/indexing, and CUDA targets include shared headers; enumerate exact includes for moves', 'native CMake targets; component checks in tests/math_core, tests/sparse_support, and CUDA-specific consumers where applicable', 'preserve original path and blob identity until all includes/build references move')
    if p == 'include/Cellerator/structure/coordinates.hh':
        return ('persistent occurrence coordinates and logical order tokens', 'include/Cellerator/structure/coordinates.hh', 'retain ordered semantic occurrence identity and coordinates independently of physical positions, buffer offsets, runtime handles, or biological feature alignment', 'retain_persistent_occurrence_coordinates', 'typed header contract is present; callers and exact ABI/test coverage are separate', 'layout::OccurrenceMap projects these coordinates into physical positions; biology::FeatureBinding may resolve identity separately', 'native-data identity and occurrence consumer checks by exact registered contract; no device claim', 'preserve repeated occurrences, canonical order tokens, and valid zero-based IDs')
    if p == 'include/Cellerator/containers/posting_index.hh' or p == 'src/containers/posting_index.cc':
        return ('generic role/entity posting storage and query primitives', 'Cellerator::containers host posting index; geometry-specific construction lives under geometry::dependency', 'canonicalize and query deterministic role/entity-to-logical-item records without imposing semantic meaning on the source relation', 'retain_generic_posting_container', 'header and host implementation are present; target membership and runtime qualification must be checked per consumer', 'geometry::dependency::build_posting_index and selected sparse-support consumers adapt semantic inputs', 'native-data posting/dependency tests by exact registration; CUDA postings have distinct source and runtime evidence', 'preserve stable item order, uniqueness rules, caller workspace, and separation from semantic pattern ownership')
    if p == 'include/Cellerator/containers/value_shape.hh':
        return ('checked value-shape arithmetic helper', 'include/Cellerator/containers/value_shape.hh', 'provide overflow-checked row/feature extent arithmetic without defining a tensor, storage owner, or allocation API', 'retain_native_container_extent_helper', 'header-only helper is present; no general tensor-shape contract is implied', 'value owners and operation consumers that explicitly call the helper', 'compile checks and consumer tests by exact registration', 'preserve overflow rejection and keep shape arithmetic separate from memory ownership')
    if p == 'include/Cellerator/layout/axis_binding.hh':
        return ('physical binding for a canonical logical axis', 'include/Cellerator/layout/axis_binding.hh', 'associate an existing logical axis with geometry and partition identities without redefining its persistent meaning or owning values', 'retain_layout_axis_binding', 'typed header contract is present; consumer and runtime qualification are separate', 'structure::LogicalAxis and layout physical maps', 'identity/layout consumers by exact contract; no GPU or performance claim', 'preserve canonical axis identity and physical partition provenance')
    if p == 'include/Cellerator/layout/occurrence_map.hh':
        return ('physical projection of logical occurrence positions', 'include/Cellerator/layout/occurrence_map.hh', 'map a pinned semantic occurrence order to a selected physical permutation/subset without treating physical slots as biological IDs', 'retain_layout_occurrence_map', 'host mapping implementation is header-present; device use and consumer qualification are separate', 'structure::OccurrenceOrderToken and execution bindings that explicitly borrow the map', 'occurrence, feature-alignment, and projection consumers by exact registered contract', 'preserve occurrence multiplicity, order token, and map-to-axis validation')
    if p == 'include/Cellerator/layout/cell_packing_geometry.hh' or p == 'src/layout/cell_packing_geometry.cc':
        return ('CP-BP physical packing statistics and geometry view', 'Cellerator::layout packing-summary source; declarations under include/Cellerator/layout/cell_packing_geometry.hh', 'describe measured/derived physical packing summaries and borrowed CP-BP metadata; this is not biological dependency knowledge or a live execution binding', 'retain_layout_cell_packing_geometry', 'host statistics implementation is present; exact source target and operation-specific checks are separate', 'CellPack persistent payloads and execution::CellPackingBinding consumers', 'layout statistics and CellPack compatibility checks by exact registration; no runtime or speed claim', 'preserve versioned payload meaning and separate physical summary, semantic structure, and current values')
    if p == 'include/Cellerator/layout/local_index_space.hh':
        return ('component-local to global physical index map', 'include/Cellerator/layout/local_index_space.hh', 'map bounded local positions into global coordinates with an optional distinct identity sidecar; the mapping does not replace the execution-owned hierarchical index space', 'retain_layout_local_index_space', 'POD layout contract and ABI assertions are present; mapping consumers are separately qualified', 'geometry/index and execution joint-compiler consumers use the view explicitly', 'installed/native consumers by exact registration; no device claim from POD compatibility', 'preserve index width encoding, component bounds, and identity-versus-position distinction')
    if p == 'include/Cellerator/execution/cell_packing_binding.hh':
        return ('borrowed binding of CP-BP layout to live execution values', 'include/Cellerator/execution/cell_packing_binding.hh', 'associate a validated existing packing payload and physical layout with current axes, structure, projection, value generation, and placement; the binding owns neither payload bytes nor runtime resources', 'retain_execution_cell_packing_binding', 'validation contract is present; CUDA and multi-device behavior are only established by exact operation evidence', 'layout::CellPackingGeometryView, CellPack payload views, and existing execution handles', 'cell-packing binding and gated-operation checks by exact registration; no blanket execution claim', 'keep payload provenance, identity, value generation, placement, and allocation lifetime explicit')
    if p == 'include/Cellerator/execution/projection_binding.hh':
        return ('live runtime binding for immutable physical projections', 'execution::projection_binding', 'associate current values, gradients, generations, and projections explicitly; do not infer a live forward-value relationship from persistent layout identity', 'retain_execution_projection_bindings', 'binding contracts are present; target registration, lifetime behavior, and numerical use are separate', 'layout::projection maps and native training/execution consumers', 'projection activation and native training tests by exact registered contract', 'preserve physical-map identity separately from runtime handles, value generations, gradients, and device placement')
    if p == 'include/Cellerator/execution/dependency/feature_binding.hh':
        return ('prepared biological feature source-to-model alignment', 'execution::dependency::FeatureSourceBinding', 'carry prepared integer gather/occurrence coordinates after biology::FeatureBinding resolves canonical identity; retain physical source positions only with their occurrence map/order token', 'retain_prepared_feature_source_binding', 'templated host mapping contract is present; model/device consumption is separate', 'include/Cellerator/biology/feature_binding.hh and layout::OccurrenceMap', 'native biological alignment and exact model consumer tests; no device execution claim', 'preserve canonical identity, missing/duplicate policy, occurrence order, and prepared index ownership')
    if p == 'include/Cellerator/execution/dependency/support_lifecycle.hh' or p == 'src/execution/dependency/support_lifecycle.cc':
        return ('host support-epoch publication and forward-tape lifecycle gate', 'Cellerator::dependency_execution; declarations and implementation under execution::dependency', 'gate immutable structure publication on tape retirement and keep structure epoch distinct from mutable state and parameter generations', 'retain_execution_support_lifecycle', 'host lifecycle mechanism is present; it does not itself publish geometry or invalidate every cache', 'geometry::dependency::PatternOwner and operations retaining their own tape contents', 'lifecycle and stale-publication tests by exact registration; not a multi-device readiness contract', 'preserve tape pin/retirement ordering and generation-domain separation')
    if p == 'include/Cellerator/execution/dependency/workset_view.hh' or p == 'src/execution/dependency/workset_view.cc':
        return ('typed candidate and condition-filtered workset views', 'execution::dependency::WorksetView', 'filter bounded per-batch primal/derivative candidates with condition outcomes while keeping semantic candidates and current eligible work distinct', 'retain_execution_dependency_workset_views', 'host filtering implementation is present; CUDA workset paths are separate', 'containers::IndexedWorkItem candidates and ops::condition::ProgramViewV1', 'execution dependency and gated-operation tests by exact registration; no broad adaptive-execution claim', 'preserve row ordering, unique work items, structure epoch, and state generation')
    if p in {'include/Cellerator/execution/dependency/packed_program.cuh', 'src/execution/dependency/packed_program.cu'}:
        return ('bounded multi-device packed gated-product program adapter', 'CMake target Cellerator::dependency_execution_cuda; public contract under execution::dependency', 'prepare the existing program_v2 stages and atom-fragment requirements around the packed gated-product operation; admit all requested devices before dispatch without introducing a general graph, compiler, or scheduler', 'retain_packed_gated_product_program_adapter', 'CUDA implementation and optional target are present; execution, sanitizer, and performance evidence are per operation and admitted device run', 'existing execution::program::prepared_program_v2, atom-fragment and atom-plane contracts, geometry/dependency bindings, tensor::block_masked, and the packed gated-product operation', 'tests/execution_dependency/packed_program_contract_test.cu is registered as an opt-in build target; runtime status requires its exact admitted-run receipt', 'preserve caller-assigned identities, immutable prepared topology, per-launch numerical sources, owner-retained contents through final completion, and partial-dispatch failure semantics')
    if p == 'include/Cellerator/execution/decomposition/gate_input.hh' or p == 'src/execution/decomposition/gate_input.cc':
        return ('operational gate operand and replication binding', 'execution::decomposition::GateInputV1', 'validate how an operation consumes a gate operand across partition-local, replicated, halo, or producer-routed placement; this is not a semantic condition-read declaration', 'retain_execution_gate_input_binding', 'validation source is present; distributed correctness is operation-specific', 'prepared decomposition and program inputs', 'gate-input contract tests and exact execution consumers; no general scheduler claim', 'preserve explicit operand index, indexing mode, replication policy, and read-only status')
    if p == 'include/Cellerator/ops/condition/program.hh' or p == 'src/ops/condition/program.cc':
        return ('boolean condition program and bitmap operations', 'ops::condition; execution workset filtering is a separate owner', 'validate/evaluate caller-supplied predicate programs and update condition bitmaps for edge ordinals; do not interpret this as biological dependency knowledge', 'retain_boolean_condition_program_operations', 'host operation implementation is present; optional CUDA workset consumers are separately qualified', 'execution::dependency::ConditionWorksets and exact condition consumers', 'condition/reference and execution workset checks by exact registration; GPU qualification remains separate', 'preserve predicate ordinals, structure-epoch association, bitmap semantics, and dirty-update locality')
    if p == 'include/Cellerator/geometry/dependency/compact_support.hh' or p == 'src/geometry/dependency/compact_support.cc':
        return ('compact ordered semantic-incidence records', 'geometry::dependency::compact', 'validate stable logical edge records and ordered role/entity participants including repeats; compact encoding does not establish physical positions or numeric activity', 'retain_compact_semantic_support_records', 'host validation and snapshot contracts are present; consumers retain independent semantic checks', 'geometry::dependency typed patterns and posting-index adapter', 'dependency geometry and native-data reference tests by exact registration', 'preserve stable IDs, participant multiplicity/order, snapshot epoch, and explicit domain meaning')
    if p == 'include/Cellerator/geometry/dependency/posting_index.hh' or p == 'src/geometry/dependency/posting_index.cc':
        return ('semantic support to generic posting-index adapter', 'geometry::dependency::build_posting_index', 'lower validated compact support incidences to containers-owned postings while preserving logical IDs and deduplicating only the index representation', 'retain_geometry_posting_adapter', 'host adapter implementation is present; GPU lookup and semantic completeness are not implied', 'geometry::dependency compact snapshots and containers::PostingIndex', 'posting/dependency geometry tests by exact registration; CUDA posting lookup remains separate', 'preserve ordered participant source, snapshot epoch, and distinction between incidence multiplicity and unique posting keys')
    if p.startswith(('src/geometry/dependency/','include/Cellerator/geometry/dependency/')):
        return ('semantic dependency patterns, applicability, conditions, and access contracts', 'geometry::dependency; implementation sources are assembled by Cellerator::dependency_geometry', 'describe admissible explicit or implicit dependencies, declared condition reads, source/order association, scope of knowledge, and operation-specific omission authority', 'retain_semantic_dependency_geometry', 'semantic contracts and host implementations are present; model restrictions, priors, and hints retain distinct meanings', 'Cellerator::dependency_geometry is the host owner; operation-specific binding and execution owners consume validated contracts', 'native_data geometry, binding, Cartesian traversal, and reference tests; each contract remains bounded by its assertions', 'do not turn partial knowledge or learned numeric zero into a global exclusion; keep semantic possibility separate from current work and physical layout')
    if p.startswith(('src/execution/dependency/','include/Cellerator/execution/dependency/')):
        parts = Path(p).parts
        family = Path(p).stem
        owner = 'execution::dependency'
        jobs = {
            'condition_index': 'index declared condition inputs and invalidate affected prepared conditions',
            'distributed': 'bind the bounded operation to explicit devices, ownership, transfers, retained forward contents, reductions, updates, and readiness',
            'incidence': 'prepare addresses from validated logical incidence and physical bindings',
            'incremental': 'track dirty prepared work against source and condition revisions',
            'read_index': 'reuse owned semantic postings for prepared read projections',
            'transfer': 'move only explicitly bound data and validate transfer path/lifetime requirements',
            'workset': 'represent current candidate work independently of immutable semantic structure',
        }
        return (f'prepared dependency execution {family}', owner, f'retain the {family} execution contract at {owner}; {jobs.get(family, "bind validated semantic sources to prepared invocation state")}', 'retain_prepared_dependency_execution', 'implementation source is present; host, CUDA build, device execution, and runtime qualification are separate evidence', 'Cellerator::dependency_execution owns host mechanisms; Cellerator::dependency_execution_cuda adds optional CUDA sources; Cellerator::native_data composes the host surface', 'tests/native_data and tests/execution_dependency by exact registered contract; device execution requires an admitted run', 'keep semantic pattern ownership, mutable value generations, physical placement, and invocation readiness as separate authorities')
    if p.startswith(('include/Cellerator/tensor/block_masked/','src/tensor/block_masked/')):
        return ('specialist block-masked tensor structure and bulk values', 'tensor::block_masked; host structure is Cellerator::block_masked and CUDA owners/kernels are Cellerator::block_masked_cuda', 'store allocated coordinate support, including numerical zeros, in immutable block metadata while keeping bulk mutable values in separate owners and logical identity in separate maps', 'retain_block_masked_tensor_format', 'host format and CUDA implementations are present; GPU correctness and performance are qualified per operation and run', 'Cellerator::tensor exposes the host format surface; Cellerator::native_data_cuda adds optional device owners and operations', 'native_data block-masked host checks and admitted CUDA correctness for the packed exemplar; these do not qualify every format shape or speed', 'preserve logical-to-physical recovery, stored-zero meaning, and retained value-content lifetimes')
    if p.startswith(('include/Cellerator/tensor/state_codes/','src/tensor/state_codes/')):
        return ('specialist codebook and packed state-code tensors', 'tensor::state_codes; header surface in Cellerator::state_codes and optional CUDA implementation in Cellerator::state_codes_cuda', 'keep codebooks, packed codes, exact-value sidecars, ownership, and typed borrowed views together as this encoding', 'retain_state_code_tensor_format', 'host views and CUDA owner/operations are present; device execution is operation-specific', 'optional tensor CUDA aggregate and migrated embedding/quantization consumers', 'state-code format and owner tests plus exact configured CUDA checks; compilation alone does not establish runtime correctness', 'do not conflate codebook quantization with sparse matrix value quantization')
    if p.startswith(('include/Cellerator/tensor/quantized/','src/tensor/quantized/')):
        return ('specialist low-bit sparse value formats', 'tensor::quantized for encoded value views and ops::quantized for encoding/decoding kernels', 'retain CSR/BELL topology reuse and the explicit 1/2/4/8-bit value/metadata contract; keep legacy persisted payload conversion in interop::formats', 'retain_quantized_tensor_formats', 'format views and optional kernels are present; per-format numerical and GPU qualification remain separate', 'quantized native operators, format tests, and explicit legacy codec adapters', 'tests/tensor/quantized_format_test.cu and tests/quantized_matrix_test.cu plus operation-specific CUDA evidence when run; no blanket precision or speed claim', 'preserve code width, metadata, and zero/overflow policies; do not reinterpret legacy archive bytes by type name')
    if p.startswith(('include/Cellerator/tensor/','src/tensor/')):
        return ('native specialist tensor format family', 'Cellerator::tensor host umbrella and optional Cellerator::tensor_cuda aggregate', 'own geometry-specific format owners/views; use established ATen or caller storage for ordinary dense and conventional sparse tensors', 'retain_specialist_tensor_formats', 'named tensor subfamilies are source-present; each target and runtime contract is format-specific', 'tensor::block_masked, tensor::state_codes, and tensor::quantized have distinct encoding and dependency boundaries', 'format-specific host and CUDA tests; no universal tensor facade or complete-operation claim', 'keep dtype, storage encoding, shape/layout, and allocation ownership explicit per format')
    if p == 'src/structure/identity_registry.cc' or p == 'include/Cellerator/execution/identity_registry.hh':
        return ('execution-time registry for persistent identities', 'execution::identity_registry declarations are under include/Cellerator/execution; implementation is currently built from src/structure/identity_registry.cc as Cellerator::identity_registry', 'intern persistent structure identities into process-local generational handles; keep handles, generations, and registries separate from semantic IDs', 'retain_execution_identity_registry', 'host registry implementation and identity contract tests are present; no GPU lookup or biological inference claim', 'src/structure/CMakeLists.txt defines Cellerator::identity_registry; planner and execution callers use the execution registry header', 'tests/execution_identity/identity_registry_test.cc and tests/biological_abi_host_test.cc; exact registrations and runs are separate evidence', 'preserve identity encodings and handle generation semantics while keeping runtime registry ownership explicit')
    if p.startswith(('src/structure/','include/Cellerator/structure/')):
        return ('canonical persistent structure identity and logical axes', 'include/Cellerator/structure/identity.hh; src/structure/CMakeLists.txt also builds the separately declared execution identity registry', 'represent persistent domain/order/geometry/partition/entity/occurrence identities and logical axes independently of allocation, runtime handles, physical layout, and model objects', 'retain_canonical_structure_identity', 'typed identity records and host ABI checks are present; this is an identity contract, not a biological ontology or model layer', 'execution::identity_registry adapts persistent IDs to process-local handles; layout and execution bind logical axes to physical resources', 'tests/biological_abi_host_test.cc plus identity and layout consumers; no GPU or scientific-validity claim', 'preserve persistent encodings; domain-local entity and occurrence zero are valid and physical slots remain separate')
    if p.startswith(('src/support/','include/Cellerator/compute/support/')):
        return ('logical support, support proposals, and eligibility state', 'src/support CMake target Cellerator::support; public headers remain include/Cellerator/compute/support', 'support and eligibility, separate from indexing, gates, mechanism routing, and physical layout', 'retain_native_support_owner', 'host_facility_implemented_with_operation_specific_evidence', 'Cellerator::indexing and Cellerator::gating consume support contracts; ops/sparse_support prepared routing builds logical edges/participants', 'tests/sparse_support/indexing_support_gating_test.cc and physical_image_builder_test.cc; Python fixtures and gates cover adapters, not all native paths', 'preserve support identity/generation behavior and installed compatibility header paths')
    if p.startswith(('src/indexing/','include/Cellerator/compute/indexing/')):
        return ('reverse-posting index, temporal occurrences, and CUDA posting lookup', 'src/indexing target Cellerator::indexing; CUDA postings are compiled into Cellerator::sparse_support_cuda when enabled', 'index construction/lookup remains distinct from structure identity, support meaning, route interpretation, and gates', 'retain_native_indexing_owner', 'host_and_cuda_sources_present; GPU qualification is operation-specific', 'Cellerator::gating uses index contracts; ops/sparse_support/prepared_routing uses reverse postings; sparse-support Torch bindings expose selected adapters', 'tests/sparse_support/indexing_support_gating_test.cc, tests/sparse_support/test_indexing_bindings.py, tests/sparse_support/cuda_worksets_test.cu; CUDA execution requires admitted device qualification', 'preserve ordering, logical IDs, and compatibility headers; no universal index dispatcher')
    if p.startswith(('src/gating/','include/Cellerator/compute/gating/')):
        return ('condition programs, derivative sampling, and conditional worksets', 'src/gating target Cellerator::gating; CUDA worksets are separately built as Cellerator::sparse_support_cuda', 'conditional work eligibility, distinct from support/index construction and physical realization', 'retain_native_gating_owner', 'host_contracts_present; cuda_worksets_have_separate_device_checks', 'Cellerator::sparse_support combines support/indexing/gating/layout; sparse-support Torch consumers and mode operations call selected paths', 'tests/sparse_support/indexing_support_gating_test.cc, derivative_sampling_test.cc, cuda_worksets_test.cu, and tests/sparse_support/test_transition_modes.py; each covers a bounded contract', 'preserve eligibility and derivative semantics; do not infer complete adaptive execution')
    if p == 'src/layout/packing/core/strategy.cc':
        return ('identity-supplied packing strategy owner', 'src/layout/packing/core/CMakeLists.txt; selectable Substrate component packing_strategy', 'retain existing native host packing strategy and its CellPack prerequisite', 'retain_selectable_packing_strategy', 'host component/test evidence is bounded to selected SDK route', 'cmake/substrate/Substrate.cmake; tests/substrate/packing', 'tests/substrate/packing/strategy_test.cc and check.py; not default root package API', 'preserve selector identity and component dependency closure')
    if p == 'src/layout/packing/strategies/placement.cc':
        return ('order-preserving layout placement strategy', 'src/layout/packing/strategies/CMakeLists.txt; selectable Substrate component packing_strategies', 'retain as a distinct placement operation over the existing packing strategy', 'retain_selectable_placement_strategy', 'host component evidence; do not infer GPU execution', 'Cellerator::packing_strategy and tests/substrate/strategies consumer', 'tests/substrate/strategies/native_consumer.cc and check.py; selected SDK scope only', 'preserve input identity/order mapping')
    if p == 'src/layout/projection/feature_major.cc':
        return ('feature-major physical projection implementation', 'src/compute/CMakeLists.txt target Cellerator::feature_major_projection; public value-free map is under include/Cellerator/layout/projection', 'build an immutable canonical-to-physical projection; keep runtime value and gradient handles in execution bindings', 'retain_feature_major_projection_owner', 'host projection source is registered; numerical runtime and speed claims remain operation-specific', 'Cellerator::transpose_projection and selected compute candidates depend on the feature-major target', 'feature-major, projection activation, value-pack, and native training tests by exact contract', 'preserve canonical feature identity and keep map ownership separate from live values, devices, and gradients')
    if p == 'src/layout/projection/transpose.cc':
        return ('transpose physical projection implementation', 'src/compute/CMakeLists.txt target Cellerator::transpose_projection; public map is under include/Cellerator/layout/projection', 'retain the canonical-to-transpose address map as a layout object and bind the current forward values in execution', 'retain_transpose_projection_owner', 'host projection source is registered; training use and runtime performance remain separately qualified', 'Cellerator::transpose_projection links feature-major projection; callers provide the actual forward-value association', 'transpose, projection activation, and native training checks by exact contract', 'preserve explicit runtime association, canonical identity, and value/gradient recovery')
    if p.startswith(('include/Cellerator/layout/','src/layout/')):
        if p.startswith('src/layout/') and p.endswith(('.cc','.cu')):
            layout_target_sources = {
                'src/layout/projection/mma_hybrid.cc',
                'src/layout/projection/mma_hybrid_builder.cc',
                'src/layout/mma_physical_cover_validation.cc',
            }
            if p not in layout_target_sources:
                return ('physical-layout implementation outside the current Cellerator::layout source list', 'src/layout source pending exact build-target ownership', 'retain the physical map or candidate until a real source target and caller are identified; do not infer build integration from its path', 'hold_layout_source_for_target_mapping', 'source present but not in the declared Cellerator::layout source list; runtime/API exposure is not implied', 'inspect the exact source consumers and build lists before assigning a target', 'operation-specific compile or consumer tests only; source-map does not infer registration or execution', 'preserve source identity until its exact target and callers are reconciled')
            return ('MMA hybrid physical projection implementation', 'src/layout/CMakeLists.txt target Cellerator::layout', 'build and validate value-free physical projection metadata; keep live runtime value associations in execution', 'retain_mma_hybrid_layout_owner', 'three listed layout translation units are registered; GPU candidates and performance remain separate evidence', 'provider realization and selected compute projection consumers', 'MMA cover, residual-builder, projection roundtrip, and value-pack tests by exact contract', 'preserve canonical value and gradient maps; a layout does not own semantic geometry or device residency')
        return ('physical layout maps and canonical recovery projections', 'include/Cellerator/layout and Cellerator::layout plus the feature-major/transpose host projection targets', 'represent physical permutations and value-free projections separately from biological dependency knowledge and live runtime bindings', 'retain_native_physical_layout_contracts', 'public headers and selected host targets are present; device qualification is per consumer', 'layout::EntityLayout and layout::projection maps are bound to current values by execution-owned bindings', 'layout, projection activation, MMA cover, feature-major, transpose, and training tests by exact contract', 'preserve canonical order, occurrence identity, padding, and gradient recovery independently of live handles')
    if p == 'include/Cellerator/ops/incidence.hh':
        return ('prepared incidence gather, reduction, and scatter arithmetic', 'ops::incidence; prepared address bindings remain owned by execution', 'provide reusable host reference arithmetic over prepared incidence with caller-selected storage, accumulation, and output types', 'retain_native_incidence_operations', 'header implementation is present; no implicit CUDA dispatch or universal numeric policy is implied', 'Cellerator::native_data host composition and callers that explicitly prepare execution::PreparedIncidence', 'ordered-incidence, native-data reference, and exact installed consumer checks where registered', 'preserve repeated-role semantics and keep arithmetic separate from address preparation')
    if p.startswith(('include/Cellerator/ops/mechanism/indexed/','src/ops/mechanism/indexed/')):
        return ('indexed mechanism programs, tapes, and parameter ownership', 'Cellerator::indexed_mechanism; public contracts under ops::mechanism::indexed and split implementation TUs under src/ops/mechanism/indexed', 'retain the existing indexed operation family with explicit inputs, topology, saved forward state, first-order derivatives, and guarded parameter updates', 'retain_indexed_mechanism_operation_owner', 'host and CUDA sources/tests are present; target compilation and GPU runtime qualification are reported per operation/device', 'CMake target Cellerator::indexed_mechanism and native foundation/model consumers', 'native_foundation n-ary host composition, indexed evaluator/training tests, and source-specific SM70 compilation; do not infer admitted runtime from build', 'preserve logical ordering, repeated incidences, tape contents, exclusive parameter writes, and checkpoint semantics')
    if p.startswith('include/Cellerator/ops/mechanism/') or p.startswith('src/ops/mechanism/'):
        return ('mechanism operation contracts and host or packed realizations', 'ops::mechanism; host exemplar belongs to Cellerator::gated_product and packed CUDA realization to optional Cellerator::block_masked_cuda', 'declare operation-specific reads, effects, omissions, and derivative scope; compose semantic bindings with physical execution without a universal model superclass', 'retain_native_mechanism_operations', 'host contracts and selected CUDA operation sources are present; correctness and performance remain operation-specific', 'Cellerator::dependency_geometry, Cellerator::gated_product, and optional Cellerator::dependency_execution_cuda provide distinct layers', 'native_data binding/reference/host tests and admitted distributed operation tests by exact receipt; no higher-order autodiff claim', 'keep model-defined omission rules explicit and preserve participants, parameter sharing, overlap, and derivative semantics')
    if p.startswith('include/Cellerator/ops/quantized/'):
        return ('low-bit sparse quantization and dequantization kernels', 'ops::quantized execution declarations over tensor::quantized value formats', 'encode or decode explicit low-bit payloads while leaving topology, owners, and persisted legacy codecs with their respective boundaries', 'retain_low_bit_quantized_operations', 'kernel and dispatch source is present; runtime and precision behavior depend on format/backend', 'selected sparse projection consumers and explicit tensor format views', 'quantized format/operator tests and benchmark families; no performance qualification from source presence', 'preserve declared bit widths, overflow and metadata policies, and format-specific numerical comparisons')
    if p.startswith(('src/ops/sparse_support/','include/Cellerator/ops/sparse_support/')):
        if 'prepared_routing' in p:
            return ('prepared entity-to-mechanism routing operation', 'src/ops/sparse_support/prepared_routing.cc/.cu; public contract include/Cellerator/ops/sparse_support/prepared_routing.hh; binding and Python adapter in bindings/sparse_support and python/cellerator/torch', 'opt-in exact route-union query using prepared reverse postings; keep route identity/generation checks and existing route policy; retain Python fallback', 'retain_opt_in_prepared_routing', 'CPU contract and adapter tests pass; CUDA correctness, sanitizer, and performance qualification pending', 'bindings/torch/CMakeLists.txt registers the Torch binding; python/cellerator/torch/__init__.py routes imports; tests/prepared_routing C++/Python files and bench/sparse_support/experiments/prepared_routing are direct consumers/evidence candidates', 'tests/prepared_routing/prepared_routing_test.cc and test_prepared_routing.py; bench KDA draft records GPU qualification path, which remains pending', 'preserve opt-in default and identity dependencies; no GPU performance claim until admitted qualification')
        return ('sparse-support training operation and native numerical worksets', 'src/ops/sparse_support target Cellerator::sparse_support_numeric; CMake gated by CUDA and sparse-support feature', 'reusable operation family under sparse support with Torch adapter as optional consumer', 'retain_native_sparse_support_operations', 'native CPU/CUDA implementations present; numerical and integrated tests are bounded by qualification records', 'bindings/sparse_support, python/cellerator/torch sparse-support modules, and examples/torch; CMake NativeFoundation owns current native link closure', 'tests/sparse_support/numeric_reference.cc, numeric_cuda_test.cu, CUDA training/mode tests and source-bound qualification records; CPU/GPU evidence remains operation-specific', 'preserve parameter ownership, first-order derivatives and feature gates; do not generalize synthetic model slices')
    if p.startswith(('src/ops/relation/', 'include/Cellerator/ops/relation/')):
        return ('native relation operation providers', 'src/ops/relation/CMakeLists.txt', 'provide CSR fallback, row-masked reduction and feature-major multiplication to the execution catalog', 'retain_native_relation_providers', 'CUDA provider sources and registered native callers', 'built-in catalog, prepared relation and execution programs', 'CSR fallback, row-masked and feature-major candidate tests', 'preserve provider identity, layout admission, precision and launch behavior')
    if p.startswith(('src/ops/numeric/response/', 'include/Cellerator/ops/numeric/response/')):
        return ('requested-output scaled tanh and derivatives', 'src/ops/numeric/response and selectable Substrate response target', 'compute requested outputs and derivatives using prepared host arithmetic', 'retain_native_response_operation', 'selectable host operation', 'response consumer and integrated SDK checks', 'tests/substrate/response', 'preserve demand order, numerical and derivative contracts')
    if p.startswith(('src/ops/matrix/frontier/', 'include/Cellerator/ops/matrix/frontier/')):
        return ('host polynomial, solve and multilevel matrix operations', 'src/ops/matrix/frontier and selectable Substrate frontier target', 'evaluate matrix operations and derivatives for the selectable SDK', 'retain_native_frontier_operations', 'selectable host operation family', 'frontier consumer and polynomial program adapter', 'tests/substrate/frontier', 'preserve numeric policy, operation order and derivatives')
    if p.startswith(('src/ops/process/effects/', 'include/Cellerator/ops/process/effects/')):
        return ('affine effect summaries and composition', 'src/ops/process/effects and selectable Substrate effects target', 'apply and compose ordered affine effects', 'retain_native_effect_operations', 'header-only host operations', 'effects and installed Substrate consumers', 'tests/substrate/effects and the installed effects consumer', 'preserve output roles and ordered composition')
    if p.startswith(('src/ops/numeric/','include/Cellerator/ops/numeric/')):
        return ('reusable numeric primitives and precision contracts', 'src/ops/numeric target Cellerator::ops_numeric; native_numeric and Substrate SDK components consume it', 'model-neutral operation family; retain native API and explicit arithmetic contracts', 'retain_native_numeric_family', 'host implementation and consumers verified; individual precision/domain contracts remain operation-specific', 'src/compute/operation/native_numeric, cmake/substrate/Substrate.cmake, tests/substrate/operators, tests/substrate/effects and native SDK consumer checks', 'tests/math_core and substrate SDK/operator/effects/frontier consumer checks; see preserved source-bound math extraction report', 'preserve no-fast-math/contract flags and operation-specific precision semantics')
    if p.startswith(('src/ops/matrix/','include/Cellerator/ops/matrix/')):
        return ('matrix patch operation family', 'src/ops/matrix target Cellerator::matrix_patch; selectable Substrate component matrix_patch', 'reuse the native forward/JVP/VJP contract and keep scientific interpretation with the consumer', 'retain_native_matrix_patch_family', 'host FP32 implementation with a separate stored-half surrogate contract', 'Substrate operators consumer and selectable installed component', 'tests/substrate/operators and the source-bound math extraction report', 'preserve precision and derivative contracts; original proof at the revision in docs/history/baseplane.md')
    if p.startswith(('src/ops/process/','include/Cellerator/ops/process/')):
        return ('private port transport operation family', 'src/ops/process target Cellerator::private_ports; selectable Substrate component private_ports', 'retain the transport primitive with caller-owned local law and edge meaning', 'retain_native_process_ports_family', 'host FP32 implementation with explicit forward/JVP/VJP contracts', 'Substrate operators consumer and selectable installed component', 'tests/substrate/operators and the source-bound math extraction report', 'preserve ragged extents and derivative contracts; original proof at the revision in docs/history/baseplane.md')
    if p.startswith(('src/compute/operation/product2/','include/Cellerator/compute/operation/product2/')):
        return ('prepared Product2 native operation', 'cmake/Product2.cmake builds Cellerator::product2 from src/compute/operation/product2; Python package uses bindings/python and python/cellerator/torch/product2.py', 'retain as a standalone model-neutral product operation; explicit native and Torch composition, not generic indexed mechanism ABI', 'retain_product2_native_and_torch_operation', 'CPU FP32 host operation implemented; CUDA source is optional and has separate runtime qualification', 'cellerator_python_extensions depends on product2; consumers are cellerator.ops, Product2Module, and direct native package callers', 'tests/ops/product2/native_product2_test.cc, tests/bindings/python/test_product2.py, test_product2_torch.py; GPU smoke is not evidence of a general GPU performance regime', 'preserve prepared topology, first-order gradient, and precision contracts')
    if p.startswith(('src/compute/operation/native_numeric/','include/Cellerator/compute/operation/native_numeric/')):
        return ('native relation arithmetic owner', 'src/compute/operation/native_numeric/CMakeLists.txt target Cellerator::native_numeric, linked to Cellerator::ops_numeric', 'retain native relation arithmetic and share the reusable local arithmetic owner with SDK consumers', 'retain_native_numeric_operation_owner', 'host and optional device source present; type/operation claims remain contract-specific', 'Cellerator::native_foundation, Cellerator::local_differential and selectable Substrate native_numeric component', 'tests/native_numeric/device_arithmetic.cc when CUDA/testing is enabled; installed Substrate SDK/operator/effects/frontier checks recorded in source-bound math extraction report', 'preserve relation semantics and shared numeric dependency; do not duplicate local arithmetic')
    if p.startswith(('src/compute/operation/differential/','include/Cellerator/compute/operation/differential/')):
        return ('local operation derivative owner', 'src/compute/operation/differential target Cellerator::local_differential', 'retain derivative actions as a distinct owner consuming the shared native numeric primal', 'retain_shared_local_differential_owner', 'host/device sources present; derivative claims are operation-specific', 'Cellerator::native_numeric and executable program consumers; selectable Substrate native_numeric SDK dependency', 'tests/native_foundation/differential and tests/relation_update_spine_v1; CUDA run evidence remains separately bounded', 'preserve derivative/primal separation and no-fast-math contract')
    if p.startswith('components/tokenization/vocabulary/'):
        return ('vocabulary prototype', 'components/tokenization/vocabulary', 'retain the local vocabulary inputs independently of installed tokenizer targets', 'preserve_tokenizer_vocabulary_prototype', 'prototype source without package build registration', 'inspect explicitly before adoption', 'none recorded by this routing map', 'preserve existing content and provenance')
    if p.startswith('components/tokenization/'):
        return ('optional ATen tokenization and sequence-buffer operations', 'cmake/CelleratorTokenization.cmake target Cellerator::tokenization with optional Python _tokenization binding', 'retain as optional tensor/tokenization operations; biological event and history meaning stay with consumers', 'retain_optional_tokenization_facility', 'CPU and optional CUDA implementations; installed suite evidence remains tied to its record', 'cellerator.tokenization, ATen operation modules, and sequence-buffer callers', 'tests/tokenization/core_test.cc, numerics_test.cc, append_test.cc and source-bound 2026-10-26 qualification; do not generalize to biological model validity', 'preserve caller-defined vocabularies, event interpretation boundary and capacity/overflow contracts')
    if p.startswith(('include/Cellerator/ops/','src/ops/')):
        parts = Path(p).parts
        base = 'src/ops' if parts[0] == 'src' else 'include/Cellerator/ops'
        family = parts[2] if parts[0] == 'src' else (parts[3] if len(parts) > 3 else 'root')
        owner = f'{base}/{family}'
        return (f'native {family} operation family', owner, f'retain the {family} native operation contract with its own source family; expose a Torch/Python composition only where an adapter exists', 'retain_named_native_operation_family', 'source and header presence; build, precision, derivative, and runtime evidence are operation-specific', f'{base} family build files and direct native callers; identify exact targets before claiming package integration', f'tests belonging to {family} operations and native consumers; inspect exact registrations and run receipts', 'preserve numerical, derivative, layout, and source identity contracts')
    if p.startswith(('src/compute/','include/Cellerator/compute/')):
        parts = Path(p).parts
        base = 'src/compute' if parts[0] == 'src' else 'include/Cellerator/compute'
        offset = 2 if parts[0] == 'src' else 3
        family = parts[offset] if len(parts) > offset else 'root'
        operation = parts[offset + 1] if family == 'operation' and len(parts) > offset + 1 else None
        family_names = {
            'architecture': ('architecture providers and target refinement', 'discover device/provider capability and refine candidate targets'),
            'candidate': ('compute candidate portfolio', 'describe and evaluate operation-specific implementation candidates'),
            'dataset': ('compute dataset preparation', 'prepare model data for native compute paths'),
            'decomposition': ('compute decomposition', 'partition work according to operation and structure'),
            'layouts': ('compute-facing layout adapters', 'translate between compute operations and physical layouts'),
            'matrix': ('compute matrix operations', 'provide matrix-format conversion and compute helpers'),
            'ml': ('machine-learning compute operations', 'provide specialized operations used by Torch model workflows'),
            'neighbors': ('neighbor search and scoring operations', 'construct and score neighborhood relations'),
            'operation': (f'{operation or "root"} operation contract and implementation', f'implement the {operation or "shared"} operation contract'),
            'operators': ('compute operator implementations', 'implement reusable sparse and graph operators'),
            'tokenization': ('ATen tokenization and sequence-buffer operations', 'tokenize caller-defined sequences and manage sequence buffers'),
            'training': ('compute training operations', 'provide specialized training-side compute operations'),
            'support': ('logical support and eligibility', 'represent support separately from indices, gates, and physical layout'),
            'indexing': ('sparse and temporal indexing', 'build and query indexed relations'),
            'gating': ('conditional compute and work eligibility', 'select eligible work from prepared structure and values'),
            'candidate_discovery': ('compute candidate discovery', 'discover implementation candidates from structural properties'),
            'projection': ('projection operations', 'apply logical or physical projections'),
            'projection_family': ('projection family operations', 'share contracts across related projection implementations'),
            'sparse': ('sparse compute operations', 'execute sparse numerical operations under their format contracts'),
            'sequence': ('sequence operations', 'operate on caller-defined sequence data'),
            'root': ('compute public root contracts', 'preserve direct native compute entry points'),
        }
        title, job = family_names.get(operation or family, (f'compute {operation or family} subsystem', 'retain its current operation contract'))
        owner = f'{base}/{family}' + (f'/{operation}' if operation else '')
        evidence = f'{base}/CMakeLists.txt and nested family CMake files name the built source set; an existing file alone is not proof of target registration' if parts[0] == 'src' else f'public header interface under {owner}; verify installed/exported headers and actual include callers per API'
        return (title, owner, f'retain at {owner}; {job}', 'retain_in_named_compute_family', 'source/header present; build, runtime, and numerical qualification are operation-specific', evidence, f'tests under the corresponding tests family and native build registration; map does not imply each check ran', 'preserve identity, precision, layout, and derivative contracts; move only with its real callers and build entries')
    if p.startswith(('src/execution/','include/Cellerator/execution/','src/runtime/','include/Cellerator/runtime/')):
        parts = Path(p).parts
        is_runtime = 'runtime' in parts[:3]
        base = ('src/runtime' if parts[0] == 'src' else 'include/Cellerator/runtime') if is_runtime else ('src/execution' if parts[0] == 'src' else 'include/Cellerator/execution')
        offset = 2 if parts[0] == 'src' else 3
        family = parts[offset] if len(parts) > offset else 'root'
        execution_jobs = {
            'atom_fragment': 'represent fragments used by prepared execution', 'atom_plane': 'represent atom/value planes for prepared programs',
            'geometry_acquisition_v2': 'acquire semantic geometry for a prepared execution', 'preparation': 'prepare related operation and execution plans together',
            'native_value_instance': 'bind native value instances to prepared contracts', 'object_binding': 'bind runtime objects to prepared program inputs',
            'prepared_host': 'prepare and validate host execution state', 'program': 'represent prepared executable programs',
            'profiling': 'describe prepared execution and expose static profiling markers',
            'projection_value_plane': 'bind projected values and gradients to execution planes', 'training_program_v2': 'represent prepared training execution',
            'index_space': 'define execution index spaces', 'lowering_resumption': 'resume bounded lowering operations',
            'memory': 'own native runtime allocations and memory resources', 'multi_gpu': 'coordinate multi-device runtime resources', 'root': 'preserve direct native entry points'}
        role = ('runtime resources and device coordination' if is_runtime else f'prepared execution {family} contract')
        owner = f'{base}/{family}'
        consumers = 'src/execution/CMakeLists.txt and its linked CellPack::execution_image_v2 dependency; map preparation and runtime separately' if not is_runtime else 'src/runtime/CMakeLists.txt and concrete native callers; stream/device/resource behavior remains runtime-owned'
        checks = 'tests/execution or tests/runtime family checks when registered; source presence does not establish runtime qualification'
        if family == 'profiling':
            consumers = 'execution preparation links Cellerator::profiling_v1 from src/execution/profiling/CMakeLists.txt'
            checks = 'prepared execution contract checks in tests/jbc and profiling fixtures in tests/profiling'
        return (role, owner, f'retain at {owner}; {execution_jobs.get(family, "retain this execution subsystem contract")}', 'retain_in_named_execution_or_runtime_family', 'implementation present; prepared execution and runtime qualification are operation-specific; ExecutionField live recompilation remains experimental', consumers, checks, 'preserve prepared object identity, runtime lifetime, and ABI; do not merge semantic geometry, physical layout, and runtime state')
    if p.startswith(('src/geometry/discovery/', 'include/Cellerator/geometry/discovery/')):
        return ('sample extraction, gene support bitsets and hash candidate discovery', 'src/geometry/discovery/CMakeLists.txt', 'supply sampled support and nominate candidates for exact geometry scoring', 'retain_geometry_support_discovery', 'host and optional CUDA discovery implementations', 'geometry optimizer, capacity queries and packing consumers', 'sampling, support bitset, candidate discovery and inferred packing tests', 'preserve sample identities, hash ordering, limits and candidate semantics')
    if p.startswith(('src/geometry/','include/Cellerator/geometry/')):
        parts = Path(p).parts
        base = 'src/geometry' if parts[0] == 'src' else 'include/Cellerator/geometry'
        offset = 2 if parts[0] == 'src' else 3
        family = parts[offset] if len(parts) > offset else 'root'
        geometry_jobs = {
            'candidate_discovery': 'discover semantic implementation candidates', 'compiler': 'compile semantic relationships and geometry contracts',
            'optimization': 'optimize semantic partitions using explicit objective contracts', 'optimizer': 'evaluate and refine geometry partitions',
            'persistence': 'serialize or restore geometry-owned state', 'strategy': 'select semantic organization strategies',
            'validation': 'validate semantic geometry and contracts', 'index': 'index semantic geometry',
            'work_layout_v2': 'describe logical work ordering before physical realization', 'work_window_v2': 'describe logical windows of eligible work', 'root': 'preserve geometry public root contracts'}
        owner = f'{base}/{family}'
        evidence = 'src/geometry/CMakeLists.txt and src/geometry/owned_targets.cmake define concrete CellPack and semantic-geometry target families' if parts[0] == 'src' else 'public geometry API; callers include Cellerator::semantic_geometry_v1 and CellPack::execution_image_v2 in bounded targets'
        return (f'semantic geometry {family}', owner, f'retain at {owner}; {geometry_jobs.get(family, "retain its current semantic geometry contract")}', 'retain_in_named_geometry_family', 'geometry implementation present; target and qualification vary by family; physical realization belongs to layout/provider families', evidence, 'tests/geometry contracts, acquisition, projection, providers, catalog, and validation suites by subject; no blanket pass claim', 'preserve structure/order identity and keep semantic discovery separate from physical layout and execution')
    if p.startswith(('src/math/','include/Cellerator/math/','src/packing/','include/Cellerator/packing/','src/quantized/','include/Cellerator/quantized/','include/Cellerator/optimize/','include/Cellerator/models/','include/Cellerator/trajectory/')):
        parts = Path(p).parts
        family = parts[2] if parts[0] == 'src' else (parts[3] if len(parts) > 3 else parts[2])
        owner = '/'.join(parts[:3] if parts[0] == 'src' else parts[:4])
        role = {'math': 'native mathematical operations and numeric helpers', 'packing': 'native packing implementation or compatibility API', 'quantized': 'native quantized operation or storage contract', 'optimize': 'native optimization interface', 'models': 'native model support interface', 'trajectory': 'native trajectory interface'}.get(family, f'native {family} subsystem')
        return (role, owner, f'retain in its existing {family} subsystem until implementation and caller evidence supports a move', 'retain_named_legacy_native_subsystem', 'implementation present; target, consumer, and numerical maturity are operation-specific', 'inspect the family CMake and direct native includes before claiming target or package exposure', 'tests/math_core, tests/native_numeric, model tests, or examples only where they exercise this exact contract', 'preserve formulas, parameters, derivative behavior, and checkpoint/schema semantics')
    if p.startswith(('bindings/torch/', 'src/interop/torch/')):
        return ('native LibTorch or ATen integration', 'src/interop/torch', 'retain native adapters at their C++ build and package boundary', 'retain_native_torch_integration', 'optional native implementation; configure and test each selected component', 'native C++ Torch consumers and optional Python extensions', 'tests/bindings/torch, tests/interop, and the installed Torch consumer', 'preserve installed include paths, target names, and direct native consumers')
    if p.startswith(('src/execution/planning/','include/Cellerator/execution/planning/','src/preprocess/','include/Cellerator/preprocess/','src/state/','include/Cellerator/state/','src/support/','include/Cellerator/support/','src/interop/','include/Cellerator/interop/','src/abi/','include/Cellerator/abi/')):
        parts = Path(p).parts
        family = parts[1] if parts[0] == 'src' else parts[2]
        owner = '/'.join(parts[:3] if parts[0] == 'src' else parts[:4])
        jobs = {'planner': 'create or validate operation plans', 'preprocess': 'prepare caller data for native operations', 'state': 'represent native operation or model state', 'support': 'represent logical support and eligibility', 'interop': 'adapt native contracts to external consumers', 'abi': 'define the stable native ABI'}
        return (f'native {family} subsystem', owner, f'retain under {owner}; {jobs.get(family, "preserve its current native contract")}', 'retain_named_native_shared_subsystem', 'source and interface present; target and consumer qualification remain specific', f'current headers and includes rooted at {owner}; inspect source targets and downstream callers before changing ABI or ownership', f'tests in the matching subsystem and actual native consumer checks; do not infer execution from source presence', 'preserve ABI/schema and real sibling consumers; move related implementation, headers, registrations, and callers together')
    if p in {'bindings/conditional/module.cc', 'python/bindings/conditional/module.cc'}:
        return ('Python binding for host conditional dispatch and parameter storage', 'cmake/CelleratorConditional.cmake; optional _conditional_native module linked to Cellerator::conditional', 'retain as the Python bridge for the generic host conditional-execution library', 'retain_conditional_python_binding', 'binding target is registered when Python support is enabled; source presence does not establish package runtime qualification', 'Cellerator::conditional and Python callers of the native conditional module', 'conditional binding/engine bridge checks are registered under CELLERATOR_BUILD_TESTS; run and package evidence remain separate', 'preserve the native conditional ABI and keep its generic scheduling role distinct from model routing policy')
    if p in {'bindings/ggml/module.cc', 'python/bindings/ggml/module.cc'}:
        return ('optional Python binding for ggml quantization interoperability', 'cmake/CelleratorGGML.cmake; optional _ggml_native module and Cellerator::ggml_quantization', 'retain as a separately gated adapter; do not add a package-wide ggml dependency', 'retain_optional_ggml_binding', 'built only when CELLERATOR_ENABLE_GGML and Python support are enabled; no runtime qualification implied', 'src/interop/ggml/quantization.cc and pinned upstream ggml-base dependency', 'ggml binding and quantization checks where configured; source/build registration is not numerical validation', 'preserve optional dependency boundary and caller-visible storage/quantization contracts')
    if p in {'bindings/modeling/module.cc', 'python/bindings/modeling/module.cc'}:
        return (
            'Python binding for host dependency geometry and value fields',
            'cmake/CelleratorNativeData.cmake; _modeling_native links Cellerator::native_data',
            'adapt current geometry, incidence and value owners',
            'retain_host_modeling_python_binding',
            'available with the Python build; validate the scoped binding',
            'Python host interaction adapters',
            'tests/framework/native/test_native_binding.py',
            'preserve extension import, native type and ownership behavior',
        )
    if p in {'bindings/modeling_cuda/module.cc', 'python/bindings/modeling_cuda/module.cc'}:
        return (
            'optional Python binding for framework CUDA operations',
            'cmake/CelleratorFrameworkCUDA.cmake; _modeling_cuda links the selected CUDA owners',
            'adapt device resources, dispatch, grouped linear and state-code operations',
            'retain_framework_cuda_python_binding',
            'build and device qualification are separate',
            'CUDA owners and Python adapter consumers',
            'tests/framework/cuda/test_runtime_binding.py and operation-specific smoke checks',
            'preserve stream and device contracts and keep optional CUDA dependencies lazy',
        )
    if p in {'bindings/sparse_support/prepared_routing.cc', 'python/bindings/sparse_support/prepared_routing.cc'}:
        return ('Torch binding for opt-in prepared entity-to-mechanism routing', 'python/bindings/sparse_support/prepared_routing.cc registered by python/bindings/sparse_support/module.cc and python/bindings/torch/CMakeLists.txt', 'retain internal binding; preserve opt-in API and Python fallback', 'retain_opt_in_prepared_routing_adapter', 'binding integration present; CUDA runtime qualification pending', 'python/cellerator/torch/prepared_routing.py and tests/prepared_routing/test_prepared_routing.py', 'Torch binding translation unit compiled; CUDA correctness, sanitizers and complete-query timing pending', 'preserve dependency identity/generation arguments and per-stream workspace ownership')
    if p.startswith(('bindings/python/','bindings/sparse_support/','bindings/tokenization/','python/bindings/')):
        family = Path(p).parent.name
        if family == 'python':
            family = 'native'
        owner = f'python/bindings/{family}'
        return (f'{family} operation binding adapter', owner, f'retain as the internal {family} binding layer; expose only APIs imported and tested by the Python package', 'retain_named_binding_adapter', 'adapter source present; import, link, and runtime coverage are API-specific', f'{owner} build registration, Python imports, and linked native targets; verify each adapter path independently', 'tests/bindings and exact operation-family tests where registered; source presence does not establish package exposure', 'preserve direct native consumers and caller-visible semantics')
    if p.startswith('python/cellerator/'):
        if p == 'python/cellerator/biology.py':
            return ('Torch-free biological identity and prediction contracts', 'python/cellerator/biology.py', 'retain optional model-local bindings, prepared alignments, and borrowed prediction/query records; preserve canonical identity, model indices, and occurrence order independently', 'retain_biological_model_boundaries', 'Python-facing semantic contracts; no universal ontology, native authoring parity, or scientific inference claim', 'bound neural components, AnnData adapters, supervision, explicit manifests, and generic Cell queries', 'tests/framework/test_feature_bindings.py, test_biological_queries.py, test_prediction_supervision.py, test_biological_manifest.py, and isolated root import checks', 'keep root imports Torch-free and preserve existing typed native axes, vocabularies, relations, and parameter owners')
        if p == 'python/cellerator/torch/prepared_routing.py':
            return ('Python adapter for prepared entity-to-mechanism routing', 'python/cellerator/torch/prepared_routing.py over the registered sparse-support Torch binding', 'retain opt-in adapter; caller supplies and rechecks structure/route/binding identity generations; retain incumbent fallback', 'retain_opt_in_prepared_routing_adapter', 'CPU wrapper contract passes; CUDA runtime qualification pending', 'bindings/sparse_support/prepared_routing.cc and tests/prepared_routing/test_prepared_routing.py', 'three CPU adapter checks; no GPU proof from wrapper tests', 'do not promote to the default route dispatcher without qualified complete-query evidence')
        return ('Python biological-modeling and optional Torch surface', 'python/cellerator', 'installed biological modeling package: information, identities, interactions, histories, learning, and optional numerical adapters', 'retain_and_grow_from_existing_surface', 'early Python-facing implementation; native C++ authoring and backend qualification remain distinct', 'scan imports and actual model/native/sibling consumers; ordinary neural composition remains valid', 'tests/framework, tests/bindings/python, examples/nn_foundation, and operation-specific native checks', 'preserve optional dependencies and existing native ownership contracts; do not define the framework as a Torch extension')
    if p == 'tests/live/fixture/pbmc3k_quantitative_v1.json':
        return ('checksum-pinned PBMC3K fixture manifest', 'tests/live/fixture', 'track the fixture contract beside its generator and tests', 'retain_ce_live_fixture_manifest', 'fixture metadata; dataset stays under CELLERATOR_DATA_ROOT', 'live fixture tests and CE-LIVE benchmarks', 'tests/live/fixture/test_quantitative_fixture.py', 'preserve axis identity, source checksum, and recorded fixture expectations')
    if p.startswith('tests/'):
        if p.startswith('tests/compiler/history/source_layout_freeze/'):
            return ('historical compiler source-layout freeze check', 'tests/compiler/history/source_layout_freeze', 'preserve as historical evidence of an earlier layout contract; do not register as an active test suite without a renewed contract review', 'preserve_historical_compiler_layout_checks', 'historical assertions retained; not currently registered in compiler CMake targets', 'compiler source tree history and current source map; active compiler suite ownership lives under tests/compiler/<purpose>', 'files are not listed by compiler CMakeLists; no current build/test result implied', 'preserve original assertions and provenance; current ownership decisions supersede obsolete layout expectations')
        if p in {'tests/ops/product2/native_product2_test.cc','tests/ops/product2/cuda_product2_smoke.cu'}:
            return ('Product2 native and optional CUDA smoke tests', 'tests/ops/product2; src/compute/operation/product2 and cmake/Product2.cmake own the operation and targets', 'retain beside the Product2 operation and test its independent native/Torch contracts', 'retain_product2_operation_tests', 'native test is build-registered; CUDA smoke execution still requires admitted-device qualification', 'Cellerator::product2, Python Product2 adapter, and Product2Module', 'tests/ops/product2/* plus tests/bindings/python/test_product2.py and test_product2_torch.py; GPU smoke is not a general performance claim', 'preserve prepared topology, first-order gradient, and operation fixture identities')
        if p in {'tests/bindings/python/test_product2.py','tests/bindings/python/test_product2_torch.py'}:
            return ('Product2 Python/package contract checks', 'tests/bindings/python over native Product2 and Product2Module', 'retain with the standalone Product2 operation', 'retain_product2_python_tests', 'CPU FP32 installed/package behavior only; not a GPU test', 'cellerator.ops and Product2Module', 'exact Python test files; native and GPU coverage are separate', 'preserve operation fixtures and first-order autograd contract')
        if p in {'tests/bindings/native/mechanism_handle_test.cc','tests/bindings/python/test_native_mechanism.py','tests/bindings/python/test_mechanism.py'}:
            return ('optional indexed-mechanism binding lifecycle checks', 'native/Python mechanism adapter checks', 'retain with CUDA indexed-mechanism owner and optional framework adapters', 'retain_mechanism_binding_checks', 'contract coverage present; GPU runtime qualification remains separate', 'Cellerator::indexed_mechanism, bindings/python and bindings/torch', 'exact native and Python test files; do not infer all CUDA mechanism paths are covered', 'preserve owner lifecycle and checkpoint expectations')
        if p.startswith('tests/tokenization/'):
            return ('ATen tokenization CPU/CUDA operation tests', 'tests/tokenization and cmake/CelleratorTokenization.cmake', 'retain with optional tokenization facility', 'retain_tokenization_operation_tests', 'per-operation CPU/CUDA checks; scientific interpretation remains downstream', 'Cellerator::tokenization and optional Python binding', 'core_test.cc, numerics_test.cc, append_test.cc; use the dated tokenization qualification for aggregate installed evidence', 'preserve independent test inputs and reported source identity')
        if p.startswith('tests/substrate/operators/'):
            return ('selectable Substrate native operator consumer checks', 'tests/substrate/operators and cmake/substrate/Substrate.cmake', 'retain as consumer evidence for explicitly selected host operation components', 'retain_substrate_native_operator_checks', 'host SDK checks qualify selected components only', 'Cellerator::ops_numeric, ::matrix_patch, ::private_ports plus sibling effects/frontier consumers', 'native_consumer.cc, check.py and README.md with source-specific math extraction report; not a complete biological application test', 'preserve independent SDK build/install evidence')
        if p.startswith('tests/substrate/packing/'):
            return ('selectable Substrate packing-strategy checks', 'tests/substrate/packing and src/layout/packing/core', 'retain with packing_strategy component', 'retain_selectable_packing_strategy_checks', 'host SDK component only', 'Cellerator::packing_strategy, native_numeric, indexed incidence and CellPack', 'strategy_test.cc and check.py; selected SDK scope only', 'preserve selector inputs and owner closure')
        if p.startswith('tests/substrate/strategies/'):
            return ('order-preserving placement strategy consumer checks', 'tests/substrate/strategies and src/layout/packing/strategies', 'retain with packing_strategies component', 'retain_selectable_placement_strategy_checks', 'host SDK component only', 'Cellerator::packing_strategies and Cellerator::packing_strategy', 'native_consumer.cc, check.py and dependency hash; no GPU claim', 'preserve order/identity fixtures')
        if p.startswith('tests/prepared_routing/'):
            return ('prepared-routing host/adapter checks and GPU runner', 'tests/prepared_routing', 'retain with the opt-in prepared-routing operation', 'retain_prepared_routing_checks_pending_gpu', 'host checks pass; device runner awaits qualification', 'public prepared-routing contract, Python adapter and CUDA Torch binding', 'prepared_routing_test.cc and test_prepared_routing.py pass; run_cuda_correctness.py remains pending admission; C++ check is not currently registered in a tests/prepared_routing CMakeLists', 'preserve fixtures and comparison inputs; no GPU result implied')
        if p == 'tests/sparse_support/test_transition_modes.py':
            return ('transition mode v2 CPU contract and GPU-gated integration checks', 'tests/sparse_support/test_transition_modes.py', 'retain beside sparse-support transition implementation', 'retain_mode_v2_checks', 'five CPU checks pass; fourteen GPU checks await root qualification', 'Python Torch sparse execution mode and native transition owner', 'exact test file; GPU forward/VJP/checkpoint and sanitizer evidence pending', 'a GPU-gated/skip status is not GPU qualification')
        if p == 'tests/sparse_support/test_transition_routing_tail.py':
            return ('transition routing-tail regression check', 'tests/sparse_support/test_transition_routing_tail.py', 'retain beside transition-routing operation', 'retain_tail_routing_check_pending_gpu', 'CPU fixture assertion passes; runtime behavior remains GPU-gated', 'transition routing API in Python Torch sparse execution', 'test source only for runtime route tail; not a GPU result', 'preserve updated-block fixture and device gate')
        if p == 'tests/experimental/test_fields.py':
            return ('experimental ExecutionField installed CPU tests', 'tests/experimental/test_fields.py', 'retain beside Python experimental field interface', 'retain_experimental_field_checks', 'five installed CPU tests pass', 'python/cellerator/experimental/fields.py and native Product2 Torch module', 'exact test file and installed-package run; current field is CPU Product2 composition only', 'preserve Torch oracle, optimizer, and rebind coverage')
        parts = Path(p).parts
        suite = '/'.join(parts[:3]) if len(parts) >= 3 else '/'.join(parts[:2])
        family = parts[1] if len(parts) > 1 else 'root'
        return ('test, fixture, or validation tool', suite, f'retain under tests/{family} beside its subject; move only with implementation and registration', 'retain_in_named_test_suite', 'test source present; execution status and assertion coverage are not established by this map', 'subject is identified by exact suite path; build/CTest/Python registration and assertion scope must be checked separately', f'{p}; registration and run evidence are operation- and environment-specific', 'preserve independent fixtures, source identity, and historical receipts')
    if p.startswith('examples/'):
        return ('example, fixture, or consumer demonstration', 'examples/<family>', 'model builder / consumer example; classical workflows labeled as testbeds', 'retain_and_route_by_api', 'example_not_product_capability', 'unknown; executable/build registration scan pending', 'examples target and related tests; exact relation unresolved', 'preserve biological assumptions as consumer-owned')
    if p.startswith('bench/architecture_evidence/manifests/'):
        return ('architecture benchmark source/workload manifest or smoke trace', 'bench/architecture_evidence', 'track reproducible benchmark inputs with the owning benchmark', 'retain_architecture_evidence_manifests', 'fixture metadata and synthetic traces; local datasets stay under CELLERATOR_DATA_ROOT', 'architecture trace tooling, validators, and campaign recipes', 'bench/architecture_evidence/test_architecture_evidence.py', 'preserve source URLs, checksums, and trace identities')
    if p.startswith('bench/'):
        if p.startswith('bench/sparse_support/experiments/prepared_routing/'):
            return ('prepared-routing CUDA qualification and complete-query benchmark candidate', 'bench/sparse_support/experiments/prepared_routing', 'retain opt-in correctness and performance candidate with implementation; run only through admitted KDA workflow', 'retain_prepared_routing_evidence_pending_gpu', 'source/build/host evidence only; no GPU result', 'prepared routing native operation, Torch binding; Python fallback and dense route baseline', 'KDA draft, run_cuda_correctness.py and source hash records; all device checks/timing pending', 'preserve source/input identities, negative results and qualification receipts')
        return ('benchmark, experiment, source snapshot, or evidence record', 'bench/<family>', 'source-bound experimental evidence by operation/field family', 'preserve_as_evidence_or_reusable_source_after_audit', 'experimental_or_unqualified', 'unknown; check source snapshots and linked implementation', 'co-located bench/evidence; qualification must be checked per record', 'immutable bytes; attach source commit/hash when indexed')
    if p.startswith('docs/history/compiler/source_layout_freeze_tests/'):
        return ('archived compiler source-layout contract tests', 'docs/history/compiler/source_layout_freeze_tests', 'retain as historical source-layout evidence; these files are no longer active tests or build inputs', 'preserve_archived_compiler_layout_tests', 'historical assertion source; not registered or runnable as a current compiler suite', 'docs/history/compiler/source-layout and current source ownership decisions', 'archived C++ test sources are not in a CMake target; consult source-migrations.json for provenance', 'preserve source bytes, archive index, and original test path mapping')
    if p.startswith('docs/history/compiler/'):
        parts=Path(p).parts
        family=parts[2] if len(parts)>2 else 'archive'
        owner='/'.join(parts[:3])
        return (f'historical compiler {family} design or qualification record', owner, f'retain under {owner} as compiler history; current implementation and API claims belong to active docs and source', 'preserve_compiler_history', 'historical or milestone-bound record; not current compiler status', 'compiler contributors consulting migration, source-layout, or performance history', 'use links and source identities recorded in each archived document; content is not current build/test proof', 'preserve original bytes and location; do not revive archived contracts as active package policy')
    if p.startswith('docs/history/programs/'):
        return ('historical program, contract, handoff, or audit record', 'docs/history/programs', 'retain as indexed program history; current contracts live in maintained developer docs and Project Control owns live status', 'preserve_historical_program_record', 'historical source-bound or superseded guidance', 'contributors consulting prior decisions and preservation receipts', 'docs/history/programs/README.md indexes archived records; check referenced source revisions before reuse', 'preserve provenance and link current guidance without rewriting frozen evidence')
    if p.startswith('docs/history/celleratorch/python_source/'):
        return ('preserved historical CelleraTorch Python source snapshot', 'docs/history/celleratorch/python_source', 'retain one provenance-marked copy of the earlier adapter source and tests recovered from generated copies or Git history', 'preserve_historical_torch_source', 'historical source and evidence identity; not part of the installed cellerator API or a newly qualified model operation', 'archive README links the original manifest, source revision and former paths for future review', 'byte hashes and Python AST parsing establish preservation only; archived test snapshots are not collected by pytest', 'preserve original source bytes and distinguish earlier component implementation from the separate paused runtime proposal')
    if p == 'docs/history/celleratorch.qmd' or p.startswith('docs/history/celleratorch/'):
        return ('paused CelleraTorch proposal archive', 'docs/history/celleratorch', 'retain the abandoned proposal and its checkpoint notes as history, separate from the implemented python/cellerator/torch adapters', 'preserve_paused_torch_proposal', 'historical proposal; no separate package/runtime implementation', 'historical readers and contributors reviewing past design decisions', 'docs/history/celleratorch.qmd links the archive; current adapters are covered by docs/development/python-bindings.md', 'preserve original proposal text and provenance; do not treat it as an active direction')
    if p == 'docs/history/current_implementation_2026-10-28.qmd':
        return ('dated implementation snapshot', p, 'retain as a historical source-bound snapshot; read docs/current_implementation.qmd for the current capability boundary', 'preserve_dated_implementation_snapshot', 'dated historical summary, not live task state', 'readers tracing earlier package capability claims', 'source paths and evidence remain tied to the recorded snapshot date', 'preserve snapshot and links unchanged when newer current summaries are written')
    if p.startswith('docs/history/repository-notes/'):
        return ('archived repository note, former guide, or historical assertion', 'docs/history/repository-notes', 'retain byte-identical historical material with its former-path index; maintained guidance lives in the current product and development docs', 'preserve_archived_repository_note', 'historical source context; archived tests are not current regressions', 'docs/history/repository-notes/README.md resolves former root paths and links current guidance', 'archive hashes and original Git blobs establish preservation; do not execute old completion assertions as current package checks', 'preserve original bytes and interpret original relative links in their recorded repository-root context')
    if p.startswith('docs/history/') or p.startswith('docs/archive/'):
        return ('historical or archived documentation', p.rsplit('/',1)[0], 'retain in this archive directory with banner and provenance; exclude from current book except through indexes', 'preserve_archive', 'historical', 'historical readers; current status stays in maintained docs and Project Control', 'source-bound historical records; do not rewrite bytes', 'preserve original path and source identity')
    if p == 'docs/local_pipeline_ops.qmd' or p.startswith('docs/pipeline/'):
        return ('local pipeline operations and scripts', 'docs/pipeline and docs/local_pipeline_ops.qmd; outside installed numerical package', 'retain as local operator/testbed guidance; classical data acquisition and preprocessing remain upstream', 'retain_local_operational_material', 'operational_not_package_api', 'local pipeline operators, not general model builders', 'documented local scripts/manifests; no Cellerator numerical capability claim', 'preserve local configs, scripts and dataset links pending owner review')
    if p == 'docs/development/packed_geometry_images.md':
        return ('packed geometry image guide', p, 'document the versioned image boundaries and native tensor bindings', 'retain_packed_geometry_image_guide', 'maintained documentation', 'native geometry and tensor contributors', 'validate links against the image readers and consumer checks', 'preserve explicit format migration and ownership boundaries')
    if p in {'custom_torch_ops.md', 'docs/development/custom_torch_ops.md'}:
        return ('custom Torch operation authoring guide', 'Cellerator biological-ML and Torch operation documentation', 'retain as an operation-specific extension recipe subordinate to the general biological modeling and neural-network architecture', 'retain_custom_torch_operation_guide', 'developer guidance; operation maturity and qualification remain tied to each linked source and report', 'contributors implementing custom Torch operations and users composing the optional Torch integration', 'the grouped_linear section links its concrete adapter, native operation, tests, and source-bound qualification; no universal operator registry is implied', 'preserve Torch ownership, numerical/lifetime contracts, and the distinction between operation-specific evidence and framework-wide capability')
    if p.startswith('docs/'):
        if p == 'docs/factorized_interactions.qmd':
            return ('native factorized-interaction component guide', 'native factorized-interaction family and documentation spine', 'retain with the native operation, installed component and optional adapter contracts', 'retain_current_component_guide', 'experimental; numerical and performance claims require the linked source-bound record', 'native C++ model authors and optional Torch users', 'native installed consumer, independent oracle, admitted device checks and paired timing', 'preserve the distinction between requested route, selected backend, correctness and measured advantage')
        if p in {'docs/README.md','docs/index.qmd','docs/architecture.qmd','docs/current_implementation.qmd','docs/developer_reference.qmd','docs/scope.qmd','docs/getting_started.qmd','docs/model_building.qmd','docs/operations.qmd','docs/layout_execution.qmd','docs/experiments_and_history.qmd','docs/development/python-bindings.md','docs/performance_validation.qmd','docs/sparse_support_foundation.qmd','docs/tokenization.qmd'}:
            return ('current product, developer, operation, or capability-boundary docs', 'Cellerator documentation spine', 'retain as current Torch-extension docs; update alongside implemented contracts and measured evidence', 'retain_current_spine', 'early_product_with_per_operation_maturity', 'readers, native/Python contributors, and model builders; exact callsites belong in source map', 'Quarto HTML render, local link scan, and claims linked to implementation/tests/evidence', 'preserve frozen evidence bytes; keep source paths and target paths distinct')
        if p == 'docs/development/routing.md':
            return ('agent-facing source ownership and cleanup routing contract', 'documentation/maintenance owner with source owners', 'retain as contributor routing guide and regenerate adjacent normalized source map after source moves', 'retain_agent_routing_contract', 'current routing rules; generated map snapshot may lag until refresh', 'engineers and agents changing source, docs, or artifacts', 'generator/validator lifecycle check plus source map freshness check', 'recovery manifest and per-path archive reference; do not treat map as archive')
        if p == 'docs/development/experimental-fields.md':
            return ('experimental execution-field capability note', 'Python experimental field owner', 'retain beside implementation; revise only with field contract and test changes', 'retain_experimental_capability_doc', 'experimental; CPU Product2 composition only', 'ExecutionField users and contributors', 'tests/experimental/test_fields.py; no JIT or GPU claim', 'preserve source-bound field contract and test identities')
        if p in {'docs/migration_roadmap.qmd','docs/biological_execution_model.qmd','docs/cellpack_cp_bp.qmd','docs/core_execution_cp_math.qmd','docs/baseplane_integration.qmd','docs/storage_distribution_and_interop.qmd','docs/compute_and_models.qmd','docs/quantized_transfer_architecture.qmd','docs/cellshard_behavior.qmd','docs/ingest_and_runtime.qmd'}:
            return ('historical or subsystem design documentation', 'docs subsystem matching path', 'retain accessible under experiments/history with currentness banner; do not use as present package/API claims', 'retain_as_historical_subsystem_doc', 'historical design or mixed implementation', 'historical readers and subsystem contributors', 'verify links and preserve source-specific evidence', 'preserve original path and provenance; archive only after indexed preservation')
        if any(s in p for s in ('result','evidence','qualification','history','record','receipt','audit','status','report')):
            return ('experiment report or capability evidence', 'docs source-bound record family', 'retain as evidence; add navigation wrapper rather than rewriting frozen results', 'preserve_evidence_and_index', 'evidence_scoped_per_record', 'unknown; report links may identify consumers', 'record-defined validation only; do not broaden claim', 'preserve exact bytes and source provenance')
        return ('technical, operational or subsystem documentation', 'docs subsystem matching path', 'route to architecture, operation guide, development guide, or testbed docs', 'retain_and_reconcile_or_archive', 'mixed; inspect currentness', 'unknown; link scan required', 'claims and links require per-doc validation', 'archive superseded records with source identity')
    if p == 'planning/coherence/archive-disposition.md':
        return ('human-readable archival disposition and remaining holds', 'planning/coherence/archive-disposition.md; authored with archive worker', 'retain as preservation evidence/navigation; do not treat as live task status or permission to move held artifacts', 'preserve_archive_disposition_record', 'completed closure plus explicit unresolved holds', 'planning/coherence/README.md, docs/experiments_and_history.qmd, and source-map archive references', 'links to branch closure, audited refs, preservation manifest and workflow repair evidence', 'preserve report bytes and bundle/hash references; source map provides per-path disposition')
    if p in {'planning/coherence/branch-closure.json','planning/coherence/branches.json','planning/coherence/branches.md','planning/coherence/live-branches.tsv'}:
        return ('branch audit, source comparison, or verified closure receipt', 'planning/coherence branch reconciliation evidence', 'retain exact receipts and referenced local archive/bundle; not current branch state', 'preserve_branch_closure_evidence', 'snapshot or completed closure receipt; date and tip bound', 'planning/coherence/archive-disposition.md and branch reconciliation readers', 'JSON/TSV records carry exact ref/tip and closure fields; validate with the receipt, do not rerun ref mutations', 'private preserved bundle path and SHA-256 are in archive disposition; keep tracked evidence bytes unchanged')
    if p in {'planning/coherence/external-payloads.json','planning/coherence/preservation.json','planning/coherence/preservation-validation.json','planning/coherence/baseline-validation.json'}:
        return ('payload-preservation manifest or validation receipt', 'planning/coherence preservation evidence', 'retain manifest and original paths; ignored payload contents remain governed by private recovery archive', 'preserve_payload_recovery_evidence', 'snapshot/validation receipt; not a relocation record', 'archive disposition and separate private recovery manifest', 'recorded paths, hashes and verification scope; external targets not read or moved by source-map tooling', 'do not edit generated manifests by hand; preserve original archive and restore instructions')
    if p.startswith('planning/active-source-cleanup/qualification/'):
        return ('source-cleanup qualification artifact', 'planning/active-source-cleanup/qualification', 'retain as source-bound cleanup build, package, install, or path-closure evidence; later runs require a new receipt', 'preserve_active_cleanup_qualification', 'checkpoint-bound qualification; evidence is scoped to its command and source revision', 'maintainers reviewing final native SDK, test, package, and migration checks', 'source-qualification receipts bind commands, exit states, and relevant paths; no result generalizes beyond its recorded scope', 'preserve exact logs, test sources, JSON receipts, and toolchain details')
    if p == 'planning/active-source-cleanup/retired-payloads.json':
        return ('retired payload recovery manifest', p, 'retain as the current mapping from retired generated/diagnostic payloads to their preserved archive and restore evidence', 'preserve_retired_payload_recovery_manifest', 'payload closure record; not source code or active capability', 'recovery operators and maintainers resolving retired payload paths', 'cross-check archive manifest paths and hashes before any restore or future removal', 'preserve archive identity, hashes, original paths, and restore instructions')
    if p in {'planning/active-source-cleanup/compiler-migration-record.json','planning/active-source-cleanup/compute-migration-record.json','planning/active-source-cleanup/docs-migration-record.json'}:
        family=p.rsplit('/',1)[1].removesuffix('-migration-record.json')
        return (f'{family} source cleanup migration record', p, f'retain as the {family} workstream provenance and review record; current combined path closure is source-migrations.json', 'preserve_cleanup_workstream_migration_record', 'historical workstream evidence; combined destination closure is separately validated', 'cleanup reviewers and maintainers of the named source/document family', 'recorded old/new paths, destination checks, or scope notes are advisory by family; use the composed closure for the repository-wide audit', 'preserve source identities and per-workstream evidence')
    if p == 'planning/active-source-cleanup/README.md':
        return ('active source-cleanup evidence index', p, 'retain as an index to cleanup plans, migration closure, and source-bound qualification evidence', 'retain_cleanup_evidence_index', 'maintenance documentation, not live workflow status', 'maintainers reviewing approved cleanup decisions and preserved evidence', 'links resolve to the active cleanup records and supported Project Control remains authoritative for task state', 'keep the index aligned with the current closure manifest')
    if p in {'planning/active-source-cleanup/plan.json','planning/active-source-cleanup/scope-revision-plan.json'}:
        return ('approved source-cleanup scope and plan snapshot', p, 'preserve the accepted cleanup plan and scope revision as rationale; do not use either as mutable task status', 'preserve_cleanup_plan_snapshot', 'approved intent and scope history, not implementation or live workflow state', 'cleanup reviewers and Project Control planning context', 'source-migrations.json records destination closure; Project Control records task completion', 'preserve exact accepted plan bytes and provenance')
    if p == 'planning/active-source-cleanup/source-migrations.json':
        return ('active source-cleanup path and retirement closure manifest', p, 'retain as the current path-migration and retired-payload closure for this cleanup; create a new revision for later moves rather than editing historical plans', 'preserve_active_source_migration_closure', 'current cleanup closure; not source implementation or live task status', 'source-map migration audit and maintainers resolving old source-bound paths', 'validate with --migration-manifest planning/active-source-cleanup/source-migrations.json; records destination existence and retired-payload mapping', 'preserve original blob and destination identities; do not replace the immutable planning/coherence snapshot')
    if p == 'planning/coherence/source-migrations.json':
        return ('immutable prior source migration history', p, 'retain as the earlier migration record for its recorded baseline; it is not the current cleanup destination closure', 'preserve_historical_source_migration_record', 'historical migration snapshot; current closure is planning/active-source-cleanup/source-migrations.json', 'historical source-bound reports and plans from its original baseline', 'do not run as the current-tree destination audit after later source moves', 'preserve exact original bytes and provenance')
    if p.startswith('planning/coherence/qualification/'):
        return ('source-bound command/build/test qualification receipts and logs', 'planning/coherence/qualification; exact command and scope are recorded by per-campaign receipts', 'retain as immutable qualification evidence with its source checkpoint, command, exit status, log hash, toolchain, and scope; do not present logs as current source status', 'preserve_source_qualification_evidence', 'historical, checkpoint-bound evidence; each receipt states whether device execution occurred', 'archive disposition, source routing map, and maintainers of the specific subsystem; not live task status', 'receipt.json binds command arguments/source paths/exit codes/log hashes; only the listed operation and source checkpoint are evidenced', 'preserve original logs and JSON bytes; later runs belong in a new dated receipt')
    if p.startswith('planning/coherence/workflow-repairs/'):
        return ('source/runtime repair evidence and qualification limits', 'planning/coherence/workflow-repairs; records name specific Todo/Project Control source revisions, candidate environments, wheel hashes, and repair attempts where applicable', 'retain exact repair receipts as historical evidence; do not treat them as proof of current runtime/authority state or an instruction to replay a repair', 'preserve_workflow_repair_evidence', 'repair and package qualification is revision-bound; operation-specific limits are recorded in each JSON receipt', 'workflow repair owner, archive disposition, and supported Project Control workflow', 'per-record source/wheel/manifest identities and recovery checks; no generic claim that the active runtime or authority has changed', 'preserve receipts, logs, hashes, environment references, and provenance')
    if p.startswith('planning/coherence/'):
        return ('approved consolidation plan and deliverables', 'planning/coherence', 'preserve approved plan; live status remains Todo authority', 'preserve_active_plan', 'approved_intent_not_implementation', 'planning context only', 'plan acceptance and registered workflow are separate', 'never duplicate mutable task state')
    if p.startswith('planning/archive/legacy-plans/'):
        return ('archived legacy plan snapshot or archive index', 'planning/archive/legacy-plans', 'retain exact historical plan bytes and former-path mapping; current work belongs to the supported workflow authority', 'preserve_archived_legacy_plan', 'historical intent, not current execution state or capability', 'archive README and historical readers; geometry interlock verifier resolves the archived geometry plan explicitly', 'archive hashes, JSON parsing and verifier path checks only; do not replay historical gates as current authority', 'preserve original plan bytes and managed historical references; use the archive index to resolve their recorded root paths')
    if p.startswith('planning/'):
        return ('historical, active, or proposed research/planning artifact', 'planning/<initiative>', 'preserve until active obligation, unique capability, and provenance are reconciled', 'preserve_plan_or_archive_after_reconciliation', 'plan_or_historical_evidence', 'unknown; identify initiative owner and active task', 'linked evidence/tests per initiative; not a product capability assertion', 'preserve bytes, source refs, receipts, machine state')
    if p.startswith(('compat/','modules/','components/','external/','third_party/')):
        return ('compatibility, component, external dependency or module integration', 'path-owned integration surface', 'retain compatibility/consumer boundary; consolidate only with all callers/build targets', 'retain_and_map_consumers', 'implementation_or_vendor_presence_unqualified', 'must inventory sibling/external consumers', 'compile/link integration tests required per target', 'preserve upstream/submodule identity')
    if p.startswith(('cmake/','scripts/','tools/','profiles/')) or p in {'CMakeLists.txt','pyproject.toml'}:
        return ('build, packaging, automation, or configuration', 'repository build/tooling owner', 'maintain coherent native and installed Torch package routes', 'retain_and_reconcile_with_source_moves', 'infrastructure_presence', 'build target/import references required', 'build/test entrypoints and docs must be updated with moves', 'preserve reproducible experiment configs')
    if p.startswith(('data/','debug_out/')):
        return ('retained data or diagnostic artifact requiring preservation review', p, f'preserve at current path {p} until its producing/consuming workflow and external archive disposition are recorded', 'hold_payload_for_owner_review', 'payload is present; purpose or reproducibility remains to be confirmed', 'inspect repository references and payload provenance before further action', 'source map records file identity only; no test or diagnostic validity inferred', 'do not relocate or delete without exact hash and recovery manifest')
    if p.startswith(('experiments/baseplane_moonshot/', 'experiments/moonshot-parallel-v1/')):
        return ('retired Moonshot source or receipt', 'docs/history/baseplane.md', 'retrieve original sources and receipts at their preserved revision', 'retain_moonshot_history', 'historical research', 'pinned historical replays', 'original source-bound receipts', 'preserve original paths and bytes through the pinned Git tree')
    if p == 'experiments/test-models/transcriptome_autoencoder.py':
        return ('empty model placeholder', 'prior Cellerator Git history', 'remove empty placeholder', 'retire_empty_model_placeholder', 'empty source', 'none', 'tracked empty blob', 'preserved at the absorption review base')
    if p.startswith('experiments/'):
        return ('standalone experiment source, fixture, or receipt', 'experiments/<initiative>', 'preserve and route to owning mechanism or archive; extract reusable source only after consumer audit', 'preserve_experiment_pending_reconciliation', 'experimental_or_historical_per_initiative', 'unknown; identify source snapshots and consumers', 'initiative receipts/tests; claims bounded to each record', 'preserve bytes, manifests, source identities and restore path')
    if p.startswith('stdlib/') or p.startswith('library/cellerator/'):
        return ('compiler language standard library or source artifact', 'compiler research / language artifacts', 'preserve as experimental compiler material; do not imply current installed Python API', 'retain_experimental_language_surface', 'experimental_language_material', 'unknown; compiler tests/build targets need scan', 'tests/compiler and source records; implementation coverage unresolved', 'retain exact source and manifest')
    if p == '.project-context.json':
        return ('Project Control source-guidance declaration', p, 'retain as the repository-facing pointer set for agent source/workflow, architecture, routing, and developer-reference contracts', 'retain_project_context_guidance_declaration', 'tracked guidance metadata; it is not Todo authority, task status, or generated workflow state', 'Project Control/Codex context loader resolves the four declared repository documents', 'validate that each declared target exists and keep target names aligned with maintained docs', 'preserve declarations as source configuration; mutate only with corresponding documentation changes')
    if p == '.ctxpp.toml':
        return ('C++ context compiler source and build configuration', p, 'retain as the context-compaction tool configuration for C++/CUDA source roots, exclusions, budgets, and build/test commands', 'retain_context_compiler_configuration', 'tracked developer-tool config; compilation database path and configured commands are not evidence that they ran', 'cpp-context-compiler reads this file; generated database resolves to build/ctxpp/compile_commands.json', 'config syntax/tool checks and declared compiler/test commands only; actual builds and tests need separate receipts', 'preserve source-write/API/ABI protection settings and explicit source/exclusion patterns')
    if p.startswith('.project-control/') or p in {'.codex','.gitignore','.gitmodules'}:
        return ('repository workflow, tooling, or dependency metadata', 'repository/Project Control owners', 'preserve metadata; mutate only via supported workflow/approved project action', 'managed_or_repository_metadata', 'metadata_not_product_capability', 'not applicable or supported workflow', 'tooling-specific checks', 'do not hand edit generated authority; retain submodule declarations')
    if p == 'cellular_geometry.md':
        return ('protected cellular geometry design authority', 'Cellerator root design guidance', 'govern model-facing operations, Cell design, layout, and execution through the canonical root text and its rendered include', 'retain_protected_cellular_geometry', 'scientific and computational design rationale; not an implementation or performance claim', 'model authors, native and Python developers, layout and execution contributors', 'mathematical review, faithful Quarto include, link/render checks, and source routing validation', 'do not modify, weaken, or replace without an explicit user override authorizing a change to this document')
    if p == 'coding_style.md':
        return ('protected C++ coding style authority', 'Cellerator root contributor guidance', 'govern compact explicit C++ numerical and systems code and the preference for compile-time optimization over recurring hot-path runtime overhead through the canonical root text and its rendered include', 'retain_protected_coding_style', 'user-approved contributor rules; not an implementation or performance claim', 'C++ and CUDA C++ authors and reviewers; AGENTS.md and the implementation style guide', 'conversation coverage review, faithful Quarto include, link/render checks, and source routing validation', 'do not change, rewrite, weaken, replace, move, remove, or weaken protection without explicit user authorization specifically covering that change')
    if p in {'README.md','scope.md','CONTRIBUTING.md','AGENTS.md'}:
        return ('current root product or contributor contract', p, 'retain as current entry guidance; keep product boundary aligned with docs/architecture and current source contracts', 'retain_current_root_guidance', 'current direction; implementation evidence remains dated elsewhere', 'all repository readers and contributors', 'docs render/link checks; source implementation claims are linked to owners', 'preserve managed AGENTS workflow block and all accepted project instructions')
    if p.startswith(('library/','stdlib/')):
        return ('compiler-related language or packaged source', 'compiler research artifact owner', 'preserve as compiler experiment pending exact consumer and install role', 'retain_experimental_language_surface', 'experimental_or_unresolved', 'unknown', 'tests/build references unresolved', 'preserve manifest and exact bytes')
    # A concrete gap, intentionally not a blanket claim.
    return ('unclassified tracked path; route gap', 'unassigned; requires owner decision', 'no destination decided', 'hold_unclassified_review', 'unclassified', 'unknown', 'unknown', 'do not move/delete until classified')

snap=snapshot(REPO, OUT, Path(__file__), Path(__file__).with_name("validate.py"))
generated=snap["generated_paths"]

def add_rule(values):
    rule=dict(zip(("role","current_owner","intended_destination","disposition","maturity","consumers","tests_or_evidence","archive_reference"), values))
    rid=hashlib.sha256(json.dumps(rule,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:12]
    rules[rid]=rule
    return rid

rules={}
entries=[]
source_by_path={row["path"]:row for row in snap["source_rows"]}
tracked_by_path={row["path"]:row for row in snap["tracked"]}

def source_reference(path,row,state,tracked):
    if state.get("worktree_mode")=="missing":
        index_note=f"index blob {row['oid']}" if tracked else "untracked source"
        return f"removed from worktree; prior path {path}; {index_note}; content unavailable here; routing rule records any canonical destination; no archive is claimed"
    content_hash=state.get("content_sha256","missing")
    if tracked:
        return f"not archived; original path retained; index blob {row['oid']}; worktree sha256 {content_hash}; capture HEAD {snap['head']}"
    return f"not archived; original worktree path retained; sha256 {content_hash}"

for path,row in sorted(tracked_by_path.items()):
    if path in generated:
        continue
    values=classify(path)
    rule_id=add_rule(values)
    state=source_by_path.get(path,{})
    content_hash=state.get("content_sha256","missing")
    entries.append({
        "path":path,"source_state":"tracked","git_mode":row["mode"],"blob_oid":row["oid"],
        "worktree_mode":state.get("worktree_mode"),"content_sha256":content_hash,"rule_id":rule_id,
        "archive_reference":source_reference(path,row,state,True),
    })
for path in sorted(snap["untracked_source"]):
    values=classify(path)
    rule_id=add_rule(values)
    state=source_by_path.get(path,{})
    content_hash=state.get("content_sha256","missing")
    entries.append({
        "path":path,"source_state":"untracked","git_mode":"untracked","blob_oid":None,
        "worktree_mode":state.get("worktree_mode"),"content_sha256":content_hash,"rule_id":rule_id,
        "archive_reference":source_reference(path,row,state,False),
    })
generated_rule=add_rule((
    "generated route-map artifact","source-map generator","docs/development/source_map",
    "generated_metadata","metadata_not_product_capability","developer routing and validation",
    "tools/maintenance/source_map/lifecycle_test.py","recreatable from current tracked/worktree inputs; no source payload moved",
))
for path in sorted(generated):
    entries.append({"path":path,"source_state":"generated_metadata","git_mode":"generated","blob_oid":None,
                    "worktree_mode":None,"content_sha256":None,"rule_id":generated_rule,
                    "archive_reference":"generated metadata; regenerate with tools/maintenance/source_map/generate.py"})
entries.sort(key=lambda row:row["path"])

(OUT/"routing_rules.json").write_text(json.dumps(rules,indent=2,sort_keys=True)+"\n")
with (OUT/"path_map.jsonl").open("w") as stream:
    for row in entries:
        stream.write(json.dumps(row,sort_keys=True)+"\n")
with (OUT/"path_map.csv").open("w",newline="") as stream:
    columns=["path","source_state","git_mode","blob_oid","worktree_mode","content_sha256","rule_id","archive_reference"]
    writer=csv.DictWriter(stream,fieldnames=columns,lineterminator="\n"); writer.writeheader(); writer.writerows(entries)

ignored_paths=split_ignored()
def top_counts(paths): return dict(sorted(Counter(path.split("/",1)[0] for path in paths).items()))
untracked=snap["untracked_source"]
ignored=[path for path in ignored_paths if path not in generated]
nontracked={
    "source_identity":snap["source_identity"],
    "untracked_count":len(untracked),"untracked_top_level":top_counts(untracked),
    "ignored_count":len(ignored),"ignored_top_level":top_counts(ignored),
    "scope":"ignored inventory uses path counts by top-level only; ignored file contents were not opened",
    "disposition":"protect all ignored/untracked payloads; verify private recovery manifest and hashes before any cleanup",
    "archive_reference":"external preservation manifest must identify covered paths; this map does not claim ignored files are archived",
}
(OUT/"nontracked_inventory.json").write_text(json.dumps(nontracked,indent=2)+"\n")
counts=Counter(rules[row["rule_id"]]["disposition"] for row in entries)
unclassified=[row["path"] for row in entries if rules[row["rule_id"]]["disposition"]=="hold_unclassified_review"]
validation={
    "capture_head_provenance_only":snap["head"],"source_identity":snap["source_identity"],
    "tracked_source_count":len(snap["tracked_source"]),"untracked_source_count":len(untracked),
    "routed_path_count":len(entries),"rule_count":len(rules),
    "disposition_counts":dict(sorted(counts.items())),"unclassified_paths":unclassified,
    "dirty_tracked_paths":snap["dirty_tracked_paths"],
    "scope":"inventory completeness only; not source semantics, build, test, science, or performance acceptance",
}
(OUT/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")

def relative_tool_path(path):
    try: return str(Path(path).resolve().relative_to(REPO))
    except ValueError: return str(Path(path).resolve())
tool_paths={"generator":relative_tool_path(Path(__file__)),
            "validator":relative_tool_path(Path(__file__).with_name("validate.py")),
            "shared_snapshot":relative_tool_path(Path(__file__).with_name("common.py"))}
rules_hash=hashlib.sha256((OUT/"routing_rules.json").read_bytes()).hexdigest()
generated_hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                  for name in sorted(OUTPUT_NAMES-{"manifest.json"})}
manifest={
    "repository":".","capture_head_provenance_only":snap["head"],
    "source_identity":snap["source_identity"],"source_index_sha256":snap["source_index_sha256"],
    "source_worktree_sha256":snap["source_worktree_sha256"],"toolchain_sha256":snap["toolchain_sha256"],
    "toolchain_paths":tool_paths,"rules_sha256":rules_hash,"generated_output_sha256":generated_hashes,
    "tracked_source_count":len(snap["tracked_source"]),"untracked_source_paths":untracked,
    "generated_paths":sorted(generated),"dirty_tracked_paths":snap["dirty_tracked_paths"],
    "intended_architecture":"High-performance PyTorch extension for cellular model building; Torch supplies general tensor/model facilities; biological structure guides specialized operations and execution; fields/compiler/live recompilation are experimental; classical single-cell analysis is a testbed.",
    "source_identity_contract":"HEAD is provenance only. Identity binds non-generated Git index paths/modes/blob OIDs, current worktree content and mode for tracked and non-ignored untracked files, and generator/validator/snapshot code. Generated outputs are metadata routes and excluded from inputs.",
    "authority":"The approved plan records intent; registered Project Control state records current task status.",
}
manifest["manifest_sha256"]=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(validation,indent=2))
