get_filename_component(_ce_effects_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(cellerator_effects INTERFACE)
add_library(Cellerator::effects ALIAS cellerator_effects)
target_include_directories(cellerator_effects INTERFACE
  "${_ce_effects_root}/include"
  "${_ce_effects_root}/experiments/baseplane_moonshot/include"
  "${_ce_effects_root}/experiments/baseplane_moonshot/families/effects/include"
  "${_ce_effects_root}/experiments/baseplane_moonshot/families/mechanisms/include")
target_compile_features(cellerator_effects INTERFACE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cellerator_effects INTERFACE -fno-fast-math -ffp-contract=off)
endif()
unset(_ce_effects_root)
