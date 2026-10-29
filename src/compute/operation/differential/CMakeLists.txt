# Derivative actions have one owner and consume the shared numeric primal target.
add_library(cellerator_local_differential STATIC
  local_arithmetic.cc local_arithmetic.cu)
add_library(Cellerator::local_differential ALIAS cellerator_local_differential)
target_link_libraries(cellerator_local_differential PUBLIC
  Cellerator::native_numeric Cellerator::executable_program)
target_compile_features(cellerator_local_differential PUBLIC cxx_std_20)
target_compile_options(cellerator_local_differential PRIVATE -fno-fast-math -ffp-contract=off)
set_target_properties(cellerator_local_differential PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
