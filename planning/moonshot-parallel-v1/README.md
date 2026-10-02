# Parallel moonshot bootstrap

Archive SHA256: `dd63f52ec2e25dfcbc3d06e3982a198c0a8b8bbe7575272cfadada545dca6b1a`. Originals in `planning/moonshot-v1` remain byte-identical. These execution plans add isolated prototypes while preserving all original completion contracts and invariant definitions.

## Preservation and ownership

Existing active and scheduled tasks remain unchanged. Use additive integration only; do not supersede, reopen or requeue existing tasks. This execution instruction takes precedence over optional supersession directions in the original package. The active Baseplane companion owns `experiments/baseplane_moonshot/*` and its result directories; this program uses `experiments/moonshot-parallel-v1`.

ADOPT stages immutable seeds and agrees interfaces before child work. CE native integration depends on local NATIVE-READY, whose required live task_state gate checks existing CE-ML2-BIO done. GH consumer depends on RECEIPT, whose required live task_state gate checks existing GH-ML2-ACCEPT done. Earlier isolated experiments proceed independently of these owners.

## Parallel execution and artifact availability

CE: ADOPT → MMA and STATE → MERGE-FOUNDATION → DIFF → MERGE-DIFF → REWRITE and TRAJECTORY → INTEGRATE → gated epic closure.

GH: ADOPT → PLAIN and ACTOR; ACTOR → MERGE-ACTOR → REFACTOR; PLAIN plus MERGE-ACTOR → MODELS aggregate; MODELS plus REFACTOR plus RECEIPT → CONSUMER → gated epic closure.

Each intermediate merge accepts predecessor isolated workspaces into the integrator lineage. Create or reconcile dependent managed workspaces against these exact accepted commits before dispatch; completion alone does not copy artifacts. Intermediate merge scopes cover destination paths. Root retains integration acceptance.

Preflight and isolated prototype tasks use `parallel_safe`; prototype lanes use `isolated_merge` with explicit `parent_lane_id` and the correct intermediate `integration_task_id`. Only final native integration uses `integration_exclusive`. Expected independent experiment concurrency peaks at four across CE/GH, alongside already scheduled work. Seven CE and five GH lanes express lifecycle ownership, rather than twelve simultaneous workers. Respect current total agent and assigned GPU capacity.

## Required CE capability bridge

GH-MOON-RECEIPT has a required command gate invoking `check_receipt.py`. It opens CE authority through the installed read-only SemanticReader and requires CE-MOON-INTEGRATE effectively done. It checks the receipt against exact current CE HEAD, native/framework call paths, shapes, declared precision, forward/input-VJP/parameter-VJP/JVP, and hashed actual native, framework-consumer and derivative evidence. It runs no compute workload. Missing, pending, unsupported or stale inputs fail closed.

Receipt path: `experiments/moonshot-parallel-v1/receipt/ce-native-acceptance.json`. Required fields: `record_kind: moonshot_native_bridge_acceptance`, `schema_version: 1`, `cellerator_source_commit`, `capability` (`native_call`, `framework_call`, `supported_shapes`, `precision`, `capabilities`, `unsupported`), `checks` (`kind`, `project`, `argv`, `exit_code`, `source_commit`, `evidence_path`, `sha256`), `reviewed_by`, `reviewed_at`. Root records actual acceptance evidence; fabricated unit fixtures never satisfy implementation acceptance.

## Exact writable scopes

### cellerator

| Task | Writable paths |
|---|---|
| CE-MOON-0000 |  |
| CE-MOON-ADOPT | `planning/moonshot-parallel-v1`, `experiments/moonshot-parallel-v1/seeds`, `experiments/moonshot-parallel-v1/interfaces` |
| CE-MOON-MMA | `experiments/moonshot-parallel-v1/mma` |
| CE-MOON-STATE | `experiments/moonshot-parallel-v1/state` |
| CE-MOON-DIFF | `experiments/moonshot-parallel-v1/diff` |
| CE-MOON-REWRITE | `experiments/moonshot-parallel-v1/rewrite` |
| CE-MOON-TRAJECTORY | `experiments/moonshot-parallel-v1/trajectory` |
| CE-MOON-INTEGRATE | `experiments/moonshot-parallel-v1/integration`, `include/Cellerator`, `src`, `components/CelleraTorch`, `CMakeLists.txt`, `cmake`, `docs/learning`, `experiments/moonshot-parallel-v1/rewrite`, `experiments/moonshot-parallel-v1/trajectory` |
| CE-MOON-MERGE-FOUNDATION | `experiments/moonshot-parallel-v1/mma`, `experiments/moonshot-parallel-v1/state` |
| CE-MOON-MERGE-DIFF | `experiments/moonshot-parallel-v1/diff` |
| CE-MOON-NATIVE-READY | `experiments/moonshot-parallel-v1/owner-receipt` |

### glasshelix

| Task | Writable paths |
|---|---|
| GH-MOON-0000 |  |
| GH-MOON-ADOPT | `planning/moonshot-parallel-v1`, `experiments/moonshot-parallel-v1/seeds`, `experiments/moonshot-parallel-v1/interfaces` |
| GH-MOON-MODELS | `experiments/moonshot-parallel-v1/model-receipts`, `experiments/moonshot-parallel-v1/plain` |
| GH-MOON-REFACTOR | `experiments/moonshot-parallel-v1/refactor` |
| GH-MOON-CONSUMER | `experiments/moonshot-parallel-v1/integration`, `python/glasshelix`, `include/GlassHelix`, `src/system`, `tests`, `CMakeLists.txt`, `docs/learning`, `experiments/moonshot-parallel-v1/refactor` |
| GH-MOON-PLAIN | `experiments/moonshot-parallel-v1/plain` |
| GH-MOON-ACTOR | `experiments/moonshot-parallel-v1/actor` |
| GH-MOON-RECEIPT | `experiments/moonshot-parallel-v1/receipt` |
| GH-MOON-MERGE-ACTOR | `experiments/moonshot-parallel-v1/actor` |


## Bootstrap acceptance — 2 October 2026

Local run `CE-MOON-RUN-V1` is installed with 11 new tasks and 7 lanes. Native validation and application succeeded; `applied_modify` is empty. Original archive files remain byte-identical. See `native-validation.json`, `native-apply.json`, `preservation-check.json` and `bootstrap-receipt.json`.

The root performed canonical plan administration and final acceptance; configured subagents handled package preparation, live reconciliation and independent review. Existing CE companion work advanced concurrently and was preserved. No new implementation task was claimed, no GPU workload was launched, and no existing task was superseded or requeued. Managed workspace creation and real capability qualification occur during implementation.

For continuation, select the new run and ADOPT task explicitly through Project Control `next_task`; do not use the default active-run selection. Publish the planning inputs to the managed workspace before implementation; the new planning files are currently uncommitted in the primary checkout. Freshly reconcile live claims and provider receipts at ADOPT. Preserve the old campaign's numerical work and reuse qualified providers where appropriate.
