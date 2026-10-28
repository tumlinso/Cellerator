get_filename_component(_ce_layout_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_library(cellerator_layout STATIC
    physical_mma_hybrid.cc
    physical_mma_hybrid_builder.cc
    mma_physical_cover_validation.cc)
add_library(Cellerator::layout ALIAS cellerator_layout)
target_include_directories(cellerator_layout PUBLIC
    $<BUILD_INTERFACE:${_ce_layout_root}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(cellerator_layout PUBLIC cxx_std_17)
if(TARGET Cellerator::support)
    target_link_libraries(cellerator_layout PUBLIC Cellerator::support)
endif()
set_target_properties(cellerator_layout PROPERTIES POSITION_INDEPENDENT_CODE ON)
unset(_ce_layout_root)
