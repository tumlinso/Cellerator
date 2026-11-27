#pragma once

// Frozen cold-path profiling/readiness contract. This surface provides static
// metadata and optional markers; it does not authorize profiler execution or
// performance promotion.
#include "Cellerator/execution/profiling/hot_path_contract.h"
#include "Cellerator/execution/profiling/mechanism_manifest.h"
#include "Cellerator/execution/profiling/partition_export.h"
#include "Cellerator/execution/profiling/resource_receipt.h"
#include "Cellerator/execution/profiling/static_markers.h"
