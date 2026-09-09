# Integrator first adds local_arithmetic.cc to the sole native_numeric target.
add_library(cellerator_local_differential STATIC local_arithmetic.cc)
add_library(Cellerator::local_differential ALIAS cellerator_local_differential)
target_link_libraries(cellerator_local_differential PUBLIC
  Cellerator::native_numeric Cellerator::operation_schema_v2 Cellerator::prepared_program_v2)
target_compile_features(cellerator_local_differential PUBLIC cxx_std_20)
target_compile_options(cellerator_local_differential PRIVATE -fno-fast-math -ffp-contract=off)
