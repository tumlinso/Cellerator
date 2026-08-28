#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
build_root=${CELLERATOR_BUILD_DIR:-"$repo_root/build"}

"$build_root/celleratorExecutionSessionTest"
"$build_root/cellPackExecutionImageV2DeviceTest"
"$build_root/celleratorRowMaskedN1CandidateTest"
"$build_root/celleratorCsrFallbackCandidateTest"
"$build_root/celleratorFeatureMajorSmallNCandidateTest"
"$build_root/celleratorTransposeBackwardCandidateTest"
"$build_root/celleratorValueGenerationReuseTest"
