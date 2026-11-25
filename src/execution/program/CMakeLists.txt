# Both the standalone Product2 shared library and the full native executable
# program package the fixed-stage entry points.
add_library(cellerator_execution_stage_program OBJECT stages.cc)
target_include_directories(cellerator_execution_stage_program PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_compile_features(cellerator_execution_stage_program PRIVATE cxx_std_20)
set_target_properties(cellerator_execution_stage_program PROPERTIES
    EXCLUDE_FROM_ALL ON POSITION_INDEPENDENT_CODE ON)
