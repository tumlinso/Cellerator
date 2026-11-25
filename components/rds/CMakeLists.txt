find_package(ZLIB REQUIRED)
include("${PROJECT_SOURCE_DIR}/cmake/CelleratorModuleInterfaces.cmake")
cellerator_require_module_scanner()

add_library(cellerator_rds STATIC
    src/cellerator.rds.cc
    src/dgcmatrix_rds.cc
    src/gzip_seek.cc
    src/rds_index.cc)
add_library(Cellerator::rds ALIAS cellerator_rds)
set_target_properties(cellerator_rds PROPERTIES EXPORT_NAME rds_binary CXX_EXTENSIONS OFF CXX_SCAN_FOR_MODULES ON)
target_compile_features(cellerator_rds PUBLIC cxx_std_20)
target_link_libraries(cellerator_rds PUBLIC Cellerator::headers PRIVATE ZLIB::ZLIB)
target_sources(cellerator_rds PUBLIC FILE_SET CXX_MODULES FILES cellerator.rds.ccm)

install(TARGETS cellerator_rds EXPORT CelleratorRDSTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    FILE_SET CXX_MODULES DESTINATION "${CMAKE_INSTALL_DATADIR}/Cellerator/modules")
install(EXPORT CelleratorRDSTargets NAMESPACE Cellerator::
    CXX_MODULES_DIRECTORY rds-cxx-modules DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")
install(FILES RDSComponent.cmake "${PROJECT_SOURCE_DIR}/cmake/CelleratorModuleInterfaces.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")

find_package(Threads REQUIRED)
add_executable(cellerator_rds_access_bench EXCLUDE_FROM_ALL bench/access_bench.cc)
target_link_libraries(cellerator_rds_access_bench PRIVATE Cellerator::rds Threads::Threads)
set_target_properties(cellerator_rds_access_bench PROPERTIES CXX_EXTENSIONS OFF CXX_SCAN_FOR_MODULES ON)

if(CELLERATOR_BUILD_TESTS)
    add_executable(cellerator_rds_indexed_test tests/indexed_read.cc)
    target_link_libraries(cellerator_rds_indexed_test PRIVATE Cellerator::rds Threads::Threads)
    set_target_properties(cellerator_rds_indexed_test PROPERTIES CXX_EXTENSIONS OFF CXX_SCAN_FOR_MODULES ON)

    find_program(CELLERATOR_RSCRIPT NAMES Rscript)
    if(CELLERATOR_RSCRIPT)
        execute_process(COMMAND "${CELLERATOR_RSCRIPT}" --vanilla -e
            "quit(status=if(requireNamespace('Matrix', quietly=TRUE)) 0 else 1)"
            RESULT_VARIABLE _rds_fixture_available OUTPUT_QUIET ERROR_QUIET)
        if(_rds_fixture_available EQUAL 0)
            set(_rds_fixture "${CMAKE_CURRENT_BINARY_DIR}/rds-fixture.rds")
            add_test(NAME cellerator-rds-fixture COMMAND "${CELLERATOR_RSCRIPT}" --vanilla
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_fixture.R" "${_rds_fixture}")
            set_tests_properties(cellerator-rds-fixture PROPERTIES FIXTURES_SETUP rds-fixture)
            add_test(NAME cellerator-rds-indexed COMMAND cellerator_rds_indexed_test "${_rds_fixture}")
            set_tests_properties(cellerator-rds-indexed PROPERTIES FIXTURES_REQUIRED rds-fixture)
        endif()
    endif()
endif()
