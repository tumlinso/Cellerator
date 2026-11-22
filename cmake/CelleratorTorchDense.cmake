include(GNUInstallDirs)

option(CELLERATOR_ENABLE_TORCH_DENSE "Build native C++ LibTorch dense interoperability" OFF)
if(NOT CELLERATOR_ENABLE_TORCH_DENSE)
    return()
endif()

include("${PROJECT_SOURCE_DIR}/cmake/CelleratorTorchDependency.cmake")

add_library(cellerator_torch_dense STATIC
    "${PROJECT_SOURCE_DIR}/src/interop/torch/dense.cc")
add_library(Cellerator::torch_dense ALIAS cellerator_torch_dense)
set_target_properties(cellerator_torch_dense PROPERTIES
    EXPORT_NAME torch_dense
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(cellerator_torch_dense PUBLIC cxx_std_20)
target_include_directories(cellerator_torch_dense PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_torch_dense PUBLIC ${CELLERATOR_TORCH_LINK_TARGETS})
target_compile_options(cellerator_torch_dense PUBLIC ${CELLERATOR_TORCH_CXX_OPTIONS})

if(CELLERATOR_BUILD_TESTS)
    add_executable(cellerator_torch_dense_test
        "${PROJECT_SOURCE_DIR}/tests/interop/torch_dense_test.cc")
    target_link_libraries(cellerator_torch_dense_test PRIVATE Cellerator::torch_dense)
    add_test(NAME cellerator_torch_dense_test COMMAND cellerator_torch_dense_test)
endif()

install(FILES "${PROJECT_SOURCE_DIR}/include/Cellerator/interop/torch/dense.hh"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/interop/torch"
    COMPONENT TorchDense)
install(TARGETS cellerator_torch_dense EXPORT CelleratorTorchDenseTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT TorchDense)
install(EXPORT CelleratorTorchDenseTargets NAMESPACE Cellerator::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator" COMPONENT TorchDense)

function(cellerator_append_torch_dense_package_config)
    if(NOT DEFINED CELLERATOR_HOST_CONFIG OR NOT EXISTS "${CELLERATOR_HOST_CONFIG}")
        return()
    endif()
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "if(\"torch_dense\" IN_LIST Cellerator_FIND_COMPONENTS)\n"
        "  if(NOT TARGET Cellerator::torch_dense)\n"
        "    include(CMakeFindDependencyMacro)\n"
        "    find_dependency(Torch)\n"
        "    include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorTorchDenseTargets.cmake\")\n"
        "  endif()\n"
        "  set(Cellerator_torch_dense_FOUND TRUE)\n"
        "endif()\n")
endfunction()
