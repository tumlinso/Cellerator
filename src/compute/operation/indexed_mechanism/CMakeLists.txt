add_library(cellerator_indexed_mechanism STATIC incidence.cc)
add_library(Cellerator::indexed_mechanism ALIAS cellerator_indexed_mechanism)
target_include_directories(cellerator_indexed_mechanism PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../../../../include")
target_compile_features(cellerator_indexed_mechanism PUBLIC cxx_std_20)
