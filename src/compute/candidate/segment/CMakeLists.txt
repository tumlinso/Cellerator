# Generic segment primitives remain operation-core candidates. Relation
# programs bind them without transferring ownership to an sm_70 provider.
set(CELLERATOR_RELATION_SEGMENT_SOURCES
    src/compute/candidate/segment/mechanism_v2.cc
    src/compute/candidate/segment/normalize.cu
    src/compute/candidate/segment/reduce.cu
    src/compute/candidate/segment/segment_v2.cc
)

set(CELLERATOR_RELATION_SEGMENT_TEST_SOURCES
    src/compute/candidate/segment/mechanism_v2_test.cc
    src/compute/candidate/segment/segment_v2_test.cc
)
