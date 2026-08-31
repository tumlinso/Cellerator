# Generic segment primitives remain operation-core candidates. Relation
# programs bind them without transferring ownership to an sm_70 provider.
set(CELLERATOR_RELATION_SEGMENT_SOURCES
    src/compute/candidate/segment/cover_v2.cc
    src/compute/candidate/segment/mechanism_v2.cc
    src/compute/candidate/segment/normalize.cu
    src/compute/candidate/segment/normalize_v2.cu
    src/compute/candidate/segment/normalize_v2_reference.cc
    src/compute/candidate/segment/portfolio_v2.cc
    src/compute/candidate/segment/reduce.cu
    src/compute/candidate/segment/reduce_v2.cu
    src/compute/candidate/segment/reduce_v2_reference.cc
    src/compute/candidate/segment/segment_v2.cc
)

set(CELLERATOR_RELATION_SEGMENT_TEST_SOURCES
    src/compute/candidate/segment/cover_v2_test.cc
    src/compute/candidate/segment/mechanism_v2_test.cc
    src/compute/candidate/segment/normalize_v2_test.cc
    src/compute/candidate/segment/portfolio_v2_test.cc
    src/compute/candidate/segment/reduce_v2_test.cc
    src/compute/candidate/segment/segment_v2_test.cc
    src/compute/candidate/segment/validation_v2_test.cc
)

set(CELLERATOR_RELATION_SEGMENT_CUDA_TEST_SOURCES
    src/compute/candidate/segment/cuda_validation_v2.cu
)
