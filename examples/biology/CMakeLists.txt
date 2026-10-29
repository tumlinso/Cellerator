add_executable(biological_relations_example
    examples/biology/biological_relations.cc)
target_link_libraries(biological_relations_example PRIVATE
    Cellerator::relation_algebra)
target_compile_features(biological_relations_example PRIVATE cxx_std_17)
set_target_properties(biological_relations_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
