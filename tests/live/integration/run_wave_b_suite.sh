#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
build_root="$repo_root/build-dissolution-smoke"

"$build_root/celleratorProjectionActivationTest"
"$build_root/celleratorBuiltinCatalogTest"
"$build_root/celleratorCusparseCsrCandidateTest"
"$build_root/celleratorPreparationFactoryTest"
"$build_root/celleratorQuantitativeRelationTest"
"$build_root/celleratorLivePlannerFeaturesTest"
"$build_root/celleratorNativeTrainingSliceTest"
"$build_root/celleratorCandidateMeasurementTest"
"$build_root/celleratorExecutableCoreIntegrationTest"
