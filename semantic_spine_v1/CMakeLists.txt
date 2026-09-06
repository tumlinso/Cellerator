# Repository-local post-epic example. I04 supplies Cellerator::prepared_relation_cuda.
# This file is intentionally not added to the current root build by package extraction.
if(NOT TARGET Cellerator::prepared_relation_cuda)
    message(FATAL_ERROR "Semantic Spine v1 implementation target is absent; complete the epic first")
endif()
add_executable(ceSemanticSpineDemo examples/semantic_spine_v1/regulatory_reuse.cc)
target_compile_features(ceSemanticSpineDemo PRIVATE cxx_std_17)
target_link_libraries(ceSemanticSpineDemo PRIVATE Cellerator::prepared_relation_cuda CUDA::cudart)
set_target_properties(ceSemanticSpineDemo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
# Reference fixture is never the required accelerator acceptance target.
add_executable(ceSemanticSpineReferenceFixture examples/semantic_spine_v1/regulatory_reuse.cc)
target_compile_definitions(ceSemanticSpineReferenceFixture PRIVATE CELLERATOR_SPINE_REFERENCE_ONLY=1)
target_compile_features(ceSemanticSpineReferenceFixture PRIVATE cxx_std_17)
set_target_properties(ceSemanticSpineReferenceFixture PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
