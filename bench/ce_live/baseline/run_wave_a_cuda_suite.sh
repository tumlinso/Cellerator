#!/usr/bin/env bash
set -euo pipefail

./build-dissolution-smoke/celleratorExecutionSessionTest
./build-dissolution-smoke/cellPackExecutionImageV2DeviceTest
./build-dissolution-smoke/celleratorRowMaskedN1CandidateTest
./build-dissolution-smoke/celleratorCsrFallbackCandidateTest
./build-dissolution-smoke/celleratorFeatureMajorSmallNCandidateTest
./build-dissolution-smoke/celleratorTransposeBackwardCandidateTest
./build-dissolution-smoke/celleratorValueGenerationReuseTest
