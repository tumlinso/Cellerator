add_library(cellerator_tensor_interface STATIC)
set_target_properties(cellerator_tensor_interface PROPERTIES EXPORT_NAME tensor_interface_binary)
target_compile_features(cellerator_tensor_interface PUBLIC cxx_std_20)
target_include_directories(cellerator_tensor_interface PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_sources(cellerator_tensor_interface PUBLIC FILE_SET CXX_MODULES
    BASE_DIRS "${CMAKE_CURRENT_LIST_DIR}"
    FILES "${CMAKE_CURRENT_LIST_DIR}/cellerator.tensor.ccm")
target_link_libraries(cellerator_tensor INTERFACE $<BUILD_INTERFACE:cellerator_tensor_interface>)

install(TARGETS cellerator_tensor_interface EXPORT CelleratorTensorInterfaceTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT NativeData
    FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/Cellerator/modules COMPONENT NativeData)
install(EXPORT CelleratorTensorInterfaceTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator
    CXX_MODULES_DIRECTORY tensor-modules COMPONENT NativeData)
