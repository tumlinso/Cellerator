add_executable(ceGeoBiologicalRelationsExample
    examples/ce_geo/biological_relations.cc)
target_link_libraries(ceGeoBiologicalRelationsExample PRIVATE
    Cellerator::relation_algebra)
target_compile_features(ceGeoBiologicalRelationsExample PRIVATE cxx_std_17)
set_target_properties(ceGeoBiologicalRelationsExample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
