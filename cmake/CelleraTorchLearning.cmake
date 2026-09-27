# Installed learning adapter. Native kernels/session code are linked into this
# shared producer; consumers do not compile private Cellerator source files.
if(TARGET cellera_torch_mechanism)
    set_target_properties(cellera_torch_mechanism PROPERTIES EXPORT_NAME mechanism)
    set_target_properties(cellera_torch PROPERTIES EXPORT_NAME torch)
    install(TARGETS cellera_torch_mechanism cellera_torch
        EXPORT CelleraTorchLearningTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT CelleraTorchLearning
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT CelleraTorchLearning
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT CelleraTorchLearning)
    install(EXPORT CelleraTorchLearningTargets NAMESPACE CelleraTorch::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CelleraTorch
        COMPONENT CelleraTorchLearning)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/components/CelleraTorch/include/CelleraTorch
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT CelleraTorchLearning)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/Cellerator
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT CelleraTorchLearning)
    configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/CelleraTorchConfig.cmake.in
        ${PROJECT_BINARY_DIR}/CelleraTorchConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CelleraTorch)
    install(FILES ${PROJECT_BINARY_DIR}/CelleraTorchConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CelleraTorch
        COMPONENT CelleraTorchLearning)
endif()
