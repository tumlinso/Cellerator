include(GNUInstallDirs)
get_filename_component(sequence_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_library(cellerator_sequence STATIC
    src/packing.cc src/operations.cc src/motif.cc src/predicate_plan.cc src/source.cc)
add_library(Cellerator::sequence ALIAS cellerator_sequence)
target_compile_features(cellerator_sequence PUBLIC cxx_std_20)
target_include_directories(cellerator_sequence PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${sequence_repo_root}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(cellerator_sequence PROPERTIES
    EXPORT_NAME sequence CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON
    CELLERATOR_CAPABILITIES "host_packed_sequence,predicate_preparation"
    EXPORT_PROPERTIES CELLERATOR_CAPABILITIES)

if(CELLERATOR_HAS_CUDA)
    target_sources(cellerator_sequence PRIVATE src/packing.cu src/operations.cu)
    target_link_libraries(cellerator_sequence PUBLIC CUDA::cudart)
    set_target_properties(cellerator_sequence PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET cellerator_sequence APPEND_STRING PROPERTY CELLERATOR_CAPABILITIES
        ",cuda_packed_sequence")
    install(FILES include/Cellerator/sequence/cuda.hh
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/sequence")
endif()

set(sequence_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator")
if(CELLERATOR_ENABLE_MODULES)
    include("${sequence_repo_root}/cmake/CelleratorModuleInterfaces.cmake")
    cellerator_require_module_scanner()
    add_library(cellerator_sequence_module STATIC)
    target_compile_features(cellerator_sequence_module PUBLIC cxx_std_20)
    target_include_directories(cellerator_sequence_module PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${sequence_repo_root}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    set_target_properties(cellerator_sequence_module PROPERTIES
        CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
    target_sources(cellerator_sequence_module PUBLIC FILE_SET CXX_MODULES
        BASE_DIRS src FILES src/cellerator.sequence.ccm)
    target_link_libraries(cellerator_sequence PUBLIC $<BUILD_INTERFACE:cellerator_sequence_module>)
    install(TARGETS cellerator_sequence_module EXPORT CelleratorSequenceModuleTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        FILE_SET CXX_MODULES DESTINATION "${CMAKE_INSTALL_DATADIR}/Cellerator/modules")
    install(EXPORT CelleratorSequenceModuleTargets DESTINATION "${sequence_package_dir}"
        CXX_MODULES_DIRECTORY sequence-modules)
    install(FILES "${sequence_repo_root}/cmake/CelleratorModuleInterfaces.cmake"
        DESTINATION "${sequence_package_dir}")
endif()

install(TARGETS cellerator_sequence EXPORT CelleratorSequenceTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(EXPORT CelleratorSequenceTargets NAMESPACE Cellerator::
    DESTINATION "${sequence_package_dir}")
install(DIRECTORY include/Cellerator/sequence DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator"
    PATTERN cuda.hh EXCLUDE)
configure_file(SequenceComponent.cmake.in SequenceComponent.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SequenceComponent.cmake" DESTINATION "${sequence_package_dir}")
