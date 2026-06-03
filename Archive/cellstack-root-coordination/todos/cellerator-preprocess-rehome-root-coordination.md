# Cellerator Preprocess Rehome Root Coordination

Last updated: 2026-05-04

## Quick Start

Root-owned coordination for moving the standalone preprocessing project back
into Cellerator. Implementation details live in the Cellerator and CellShard
submodules.

## Status

- Status: done
- Execution: closed
- Owner: codex

## Completed Work

- Removed the `CellShardPreprocess` and `CellShardNeighbors` submodule mappings
  from `.gitmodules`.
- Removed the root-tracked `CellShardPreprocess` and `CellShardNeighbors`
  gitlinks.
- Updated root docs to describe Cellerator as the preprocessing owner.

## Validation

- `git submodule status` lists only the remaining active submodules relevant to
  this migration.

## Note

The root now treats Cellerator as owner for preprocessing and forward-neighbor
caller policy.

