# Repository-local semantic foundation; no installed SDK or runtime ownership.
add_library(cellerator_relation_semantics STATIC
    src/compute/operation/relation_semantics.cc)
add_library(Cellerator::relation_semantics ALIAS cellerator_relation_semantics)
target_include_directories(cellerator_relation_semantics PUBLIC
    "${PROJECT_SOURCE_DIR}/include")
target_compile_features(cellerator_relation_semantics PUBLIC cxx_std_17)

if(CELLERATOR_BUILD_TESTS)
    add_executable(ceSpineCoreTest tests/semantic_spine/core/descriptor_test.cc)
    target_link_libraries(ceSpineCoreTest PRIVATE Cellerator::relation_semantics)
    add_executable(ceSpineMathematicalDualityTest
        tests/semantic_spine/core/mathematical_duality_test.cc)
    target_compile_features(ceSpineMathematicalDualityTest PRIVATE cxx_std_17)
    set_target_properties(ceSpineCoreTest ceSpineMathematicalDualityTest PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
