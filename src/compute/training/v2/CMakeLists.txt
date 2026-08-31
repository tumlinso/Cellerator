set(CELLERATOR_TRAINING_V2_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/generation_readiness.cc
    ${CMAKE_CURRENT_LIST_DIR}/relation_closure.cc
    ${CMAKE_CURRENT_LIST_DIR}/value_modes.cc
    PARENT_SCOPE)

set(CELLERATOR_TRAINING_V2_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/generation_readiness_test.cc
    ${CMAKE_CURRENT_LIST_DIR}/relation_closure_test.cc
    ${CMAKE_CURRENT_LIST_DIR}/value_modes_test.cc
    PARENT_SCOPE)
