add_library(cellerator_module STATIC)
add_library(Cellerator::cellerator ALIAS cellerator_module)
set_target_properties(cellerator_module PROPERTIES EXPORT_NAME module_interface_binary CXX_EXTENSIONS OFF)
target_compile_features(cellerator_module PUBLIC cxx_std_20)
target_link_libraries(cellerator_module PUBLIC Cellerator::tensor)
target_sources(cellerator_module PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS "${PROJECT_SOURCE_DIR}/src"
    FILES "${PROJECT_SOURCE_DIR}/src/cellerator.ccm"
          "${PROJECT_SOURCE_DIR}/src/tensor/cellerator.tensor-partition.ccm")

install(TARGETS cellerator_module EXPORT CelleratorModuleTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT NativeData
    FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/Cellerator/modules COMPONENT NativeData)
install(EXPORT CelleratorModuleTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator
    CXX_MODULES_DIRECTORY core-modules COMPONENT NativeData)
