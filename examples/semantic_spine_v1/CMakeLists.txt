# Repository-local example included by the opt-in Semantic Spine build.
if(NOT TARGET Cellerator::prepared_relation_cuda)
    message(FATAL_ERROR "Semantic Spine native execution target is required")
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
