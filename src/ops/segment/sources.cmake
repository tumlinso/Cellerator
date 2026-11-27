# Generic segment primitives remain operation-core candidates. Relation
# programs bind them without transferring ownership to an sm_70 provider.
set(CELLERATOR_RELATION_SEGMENT_SOURCES
    src/ops/segment/cover_v2.cc
    src/ops/segment/mechanism_v2.cc
    src/ops/segment/normalize.cu
    src/ops/segment/normalize_v2.cu
    src/ops/segment/normalize_v2_reference.cc
    src/ops/segment/portfolio_v2.cc
    src/ops/segment/reduce.cu
    src/ops/segment/reduce_v2.cu
    src/ops/segment/reduce_v2_reference.cc
    src/ops/segment/segment_v2.cc
)

set(CELLERATOR_RELATION_SEGMENT_TEST_SOURCES
    src/ops/segment/cover_v2_test.cc
    src/ops/segment/mechanism_v2_test.cc
    src/ops/segment/normalize_v2_test.cc
    src/ops/segment/portfolio_v2_test.cc
    src/ops/segment/reduce_v2_test.cc
    src/ops/segment/segment_v2_test.cc
    src/ops/segment/validation_v2_test.cc
)

set(CELLERATOR_RELATION_SEGMENT_CUDA_TEST_SOURCES
    src/ops/segment/cuda_validation_v2.cu
)
