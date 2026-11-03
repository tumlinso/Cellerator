include(GNUInstallDirs)

add_library(cellerator_nn_modules STATIC)
add_library(Cellerator::nn_modules ALIAS cellerator_nn_modules)
set_target_properties(cellerator_nn_modules PROPERTIES EXPORT_NAME nn_modules_binary)
target_compile_features(cellerator_nn_modules PUBLIC cxx_std_20)
target_sources(cellerator_nn_modules PUBLIC FILE_SET CXX_MODULES FILES cellerator.nn.ccm)
target_link_libraries(cellerator_nn_modules PUBLIC Cellerator::nn)

install(TARGETS cellerator_nn_modules EXPORT CelleratorNNModulesTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT NN
    FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/Cellerator/modules COMPONENT NN)
install(EXPORT CelleratorNNModulesTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator
    CXX_MODULES_DIRECTORY nn-modules COMPONENT NN)
