list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(LibR MODULE REQUIRED)

add_library(cellerator_rds STATIC src/matrix.cc)
add_library(Cellerator::rds ALIAS cellerator_rds)
set_target_properties(cellerator_rds PROPERTIES EXPORT_NAME rds)
target_link_libraries(cellerator_rds PUBLIC Cellerator::headers PRIVATE LibR::LibR)
target_sources(cellerator_rds PUBLIC FILE_SET HEADERS BASE_DIRS include
    FILES include/Cellerator/rds/matrix.hh)

install(TARGETS cellerator_rds EXPORT CelleratorRDSTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    FILE_SET HEADERS DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT CelleratorRDSTargets NAMESPACE Cellerator::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")
install(FILES FindLibR.cmake RDSComponent.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")
