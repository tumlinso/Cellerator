include(GNUInstallDirs)

add_library(cellerator_neighbors STATIC src/exact_search.cu src/forward_neighbors.cu)
add_library(Cellerator::neighbors ALIAS cellerator_neighbors)
set_target_properties(cellerator_neighbors PROPERTIES
    EXPORT_NAME neighbors CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
target_include_directories(cellerator_neighbors PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_neighbors PUBLIC Cellerator::headers CUDA::cudart
    PRIVATE CUDA::cusparse)
cellerator_enable_perf(cellerator_neighbors)

install(TARGETS cellerator_neighbors EXPORT CelleratorNeighborsTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT CelleratorNeighborsTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)
install(DIRECTORY include/Cellerator DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES NeighborsComponent.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator)

add_executable(cellerator_neighbors_exact_search_test ${CELLERATOR_CHECK_EXCLUDE_FROM_ALL}
    tests/exact_search_runtime_test.cu)
target_link_libraries(cellerator_neighbors_exact_search_test PRIVATE Cellerator::neighbors)
set_target_properties(cellerator_neighbors_exact_search_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
cellerator_enable_perf(cellerator_neighbors_exact_search_test)
if(CELLERATOR_BUILD_TESTS)
    add_test(NAME cellerator-neighbors-exact-search COMMAND cellerator_neighbors_exact_search_test)
endif()
