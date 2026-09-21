# Integrator first adds local_arithmetic.cc to the sole native_numeric target.
# Native numeric has no local-arithmetic translation unit yet; retain it in
# this bounded differential owner so the public primitive callbacks link.
add_library(cellerator_local_differential STATIC
  local_arithmetic.cc local_arithmetic.cu ../native_numeric/local_arithmetic.cc)
add_library(Cellerator::local_differential ALIAS cellerator_local_differential)
target_link_libraries(cellerator_local_differential PUBLIC
  Cellerator::native_numeric Cellerator::executable_program)
target_compile_features(cellerator_local_differential PUBLIC cxx_std_20)
target_compile_options(cellerator_local_differential PRIVATE -fno-fast-math -ffp-contract=off)
set_target_properties(cellerator_local_differential PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
