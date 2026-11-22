include(GNUInstallDirs)

option(CELLERATOR_ENABLE_TORCH_SPARSE "Build native C++ LibTorch CSR/CSC and COO interoperability" OFF)
if(NOT CELLERATOR_ENABLE_TORCH_SPARSE)
    return()
endif()

include("${PROJECT_SOURCE_DIR}/cmake/CelleratorTorchDependency.cmake")

add_library(cellerator_torch_sparse STATIC
    "${PROJECT_SOURCE_DIR}/src/interop/torch/compressed.cc"
    "${PROJECT_SOURCE_DIR}/src/interop/torch/coo.cc")
add_library(Cellerator::torch_sparse ALIAS cellerator_torch_sparse)
set_target_properties(cellerator_torch_sparse PROPERTIES
    EXPORT_NAME torch_sparse
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(cellerator_torch_sparse PUBLIC cxx_std_20)
target_include_directories(cellerator_torch_sparse PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_torch_sparse PUBLIC ${CELLERATOR_TORCH_LINK_TARGETS})
target_compile_options(cellerator_torch_sparse PUBLIC ${CELLERATOR_TORCH_CXX_OPTIONS})

if(CELLERATOR_BUILD_TESTS)
    add_executable(cellerator_torch_sparse_test
        "${PROJECT_SOURCE_DIR}/tests/interop/torch_compressed_test.cc")
    target_link_libraries(cellerator_torch_sparse_test PRIVATE Cellerator::torch_sparse)
    add_test(NAME cellerator_torch_sparse_test COMMAND cellerator_torch_sparse_test)
    add_executable(cellerator_torch_coo_test
        "${PROJECT_SOURCE_DIR}/tests/interop/torch_coo_test.cc")
    target_link_libraries(cellerator_torch_coo_test PRIVATE Cellerator::torch_sparse)
    add_test(NAME cellerator_torch_coo_test COMMAND cellerator_torch_coo_test)
endif()

install(FILES
    "${PROJECT_SOURCE_DIR}/include/Cellerator/interop/torch/compressed.hh"
    "${PROJECT_SOURCE_DIR}/include/Cellerator/interop/torch/coo.hh"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/interop/torch"
    COMPONENT TorchSparse)
install(TARGETS cellerator_torch_sparse EXPORT CelleratorTorchSparseTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT TorchSparse)
install(EXPORT CelleratorTorchSparseTargets NAMESPACE Cellerator::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator" COMPONENT TorchSparse)

function(cellerator_append_torch_sparse_package_config)
    if(NOT DEFINED CELLERATOR_HOST_CONFIG OR NOT EXISTS "${CELLERATOR_HOST_CONFIG}")
        return()
    endif()
    file(APPEND "${CELLERATOR_HOST_CONFIG}"
        "if(\"torch_sparse\" IN_LIST Cellerator_FIND_COMPONENTS)\n"
        "  if(NOT TARGET Cellerator::torch_sparse)\n"
        "    include(CMakeFindDependencyMacro)\n"
        "    find_dependency(Torch)\n"
        "    include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorTorchSparseTargets.cmake\")\n"
        "  endif()\n"
        "  set(Cellerator_torch_sparse_FOUND TRUE)\n"
        "endif()\n")
endfunction()
