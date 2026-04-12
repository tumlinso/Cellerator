# Current Objective

## Summary
Add on-disk metadata for the native Blocked-ELL layout, fallback compressed annotations, and shard-level replay metadata without changing logical matrix identity.

## Planning Notes
- Disk persistence should treat Blocked-ELL as the primary representation and keep compressed only where a fallback surface still needs it.

## Assumptions
_None recorded yet._

## Suggested Skills
- `cuda-v100` - Guide execution-layout persistence decisions from measured V100 performance.

## Useful Reference Files
- `extern/CellShard/src/disk/matrix.cuh` - Existing disk format codes and packed-byte helpers.
- `extern/CellShard/src/sharded/sharded.cuh` - Shard metadata and part-byte helpers for compressed and Blocked-ELL.

## Plan
- Add execution-layout metadata fields needed by the planner and replay loader.
- Support persisted promoted Blocked-ELL shard layouts when benchmarks justify them.

## Tasks
- [x] Inspect current CellShard disk metadata surfaces for where execution-format annotations belong.
- [x] Implement persistent metadata for promoted block geometry and execution-layout choice.
- [x] Add native Blocked-ELL raw packfile store/load support.
- [x] Add native Blocked-ELL series HDF5 create/load/fetch/prefetch support.

## Blockers
_None recorded yet._

## Progress Notes
- Added native Blocked-ELL packfile roundtrip support in `extern/CellShard/src/disk/matrix.*`.
- Added native Blocked-ELL HDF5 series support in `extern/CellShard/src/sharded/series_h5.*`, including direct payload append plus lazy fetch/prefetch and cache files.
- `src/ingest/series/series_ingest.cuh` now writes Blocked-ELL series payloads directly instead of going through compressed series append helpers.

## Next Actions
- Keep the series metadata readers and tests aligned with the new Blocked-ELL HDF5 payload layout.
- Audit any remaining compressed-only persistence entrypoints before removing legacy assumptions from higher layers.

## Done Criteria
- Planner-selected execution format and geometry round-trip through disk metadata.
