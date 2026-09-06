# Internal conformance targets. Device inventory is integrated with its real
# implementations; missing device targets never become passing placeholders.
add_executable(ceRu1CalculusTest tests/relation_update_spine_v1/calculus_test.cc)
target_link_libraries(ceRu1CalculusTest PRIVATE Cellerator::relation_calculus)
target_compile_options(ceRu1CalculusTest PRIVATE -UNDEBUG)
add_test(NAME ru1_calculus COMMAND ceRu1CalculusTest)
