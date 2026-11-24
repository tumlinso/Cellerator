add_library(cellerator_nn_interface STATIC)
set_target_properties(cellerator_nn_interface PROPERTIES
    EXPORT_NAME nn_interface_binary)
target_compile_features(cellerator_nn_interface PUBLIC cxx_std_20)
target_sources(cellerator_nn_interface PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS "${CMAKE_CURRENT_LIST_DIR}"
    FILES "${CMAKE_CURRENT_LIST_DIR}/cellerator.nn.ccm")
target_include_directories(cellerator_nn_interface PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cellerator_nn_interface PUBLIC Cellerator::headers ${CELLERATOR_TORCH_LINK_TARGETS})
if(CELLERATOR_TORCH_CXX_OPTIONS)
    target_compile_options(cellerator_nn_interface PUBLIC ${CELLERATOR_TORCH_CXX_OPTIONS})
endif()
target_link_libraries(cellerator_nn PUBLIC $<BUILD_INTERFACE:cellerator_nn_interface>)

install(TARGETS cellerator_nn_interface EXPORT CelleratorNNModuleTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT NN
    FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/Cellerator/modules COMPONENT NN)
install(EXPORT CelleratorNNModuleTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator
    CXX_MODULES_DIRECTORY nn-module COMPONENT NN)
