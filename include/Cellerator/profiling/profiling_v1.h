#pragma once

// Frozen cold-path profiling/readiness contract. This surface provides static
// metadata and optional markers; it does not authorize profiler execution or
// performance promotion.
#include "Cellerator/profiling/hot_path_contract.h"
#include "Cellerator/profiling/mechanism_manifest.h"
#include "Cellerator/profiling/partition_export.h"
#include "Cellerator/profiling/resource_receipt.h"
#include "Cellerator/profiling/static_markers.h"
