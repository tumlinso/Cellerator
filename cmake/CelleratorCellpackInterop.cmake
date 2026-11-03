# Optional zero-copy CellPack-to-block-masked views. This bridge depends on
# CellPack's prepared layout and the native block-masked view vocabulary, but
# neither family becomes the other's implementation owner.
include_guard(GLOBAL)
include(GNUInstallDirs)

if(NOT TARGET cellpack)
    message(FATAL_ERROR "CellPack interop requires the CellPack geometry target")
endif()
if(NOT TARGET Cellerator::block_masked)
    message(FATAL_ERROR "CellPack interop requires the native block-masked target")
endif()
if(NOT TARGET Cellerator::cellpack_geometry)
    message(FATAL_ERROR "CellPack interop requires the host packing geometry target")
endif()

add_library(Cellerator::cellpack ALIAS cellpack)
set_target_properties(cellpack PROPERTIES EXPORT_NAME cellpack)

add_library(cellerator_cellpack_interop INTERFACE)
add_library(Cellerator::cellpack_interop ALIAS cellerator_cellpack_interop)
set_target_properties(cellerator_cellpack_interop PROPERTIES
    EXPORT_NAME cellpack_interop)
target_include_directories(cellerator_cellpack_interop INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(cellerator_cellpack_interop INTERFACE cxx_std_17)
target_link_libraries(cellerator_cellpack_interop INTERFACE
    Cellerator::cellpack Cellerator::cellpack_geometry
    Cellerator::block_masked CUDA::cudart)

install(TARGETS cellerator_cellpack_interop
    EXPORT CelleratorCellpackInteropTargets
    COMPONENT cellpack_interop)
install(EXPORT CelleratorCellpackInteropTargets NAMESPACE Cellerator::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cellerator
    COMPONENT cellpack_interop)

foreach(_header IN ITEMS
        interop/formats/cellpack.hh
        layout/cell_packing_geometry.hh
        execution/cell_packing_binding.hh
        geometry/warp_tiles.hh
        geometry/local_cell_ordering.hh
        geometry/cell_block_records.hh
        geometry/apply_plan.hh
        geometry/packing_plan.hh
        geometry/evaluator.hh
        geometry/planner.hh
        geometry/validate.hh
        geometry/permutation.hh
        geometry/format.hh
        memory/allocation.hh
        memory/image.hh
        memory/status.hh
        memory/domain.hh
        tensor/block_masked/layout.hh
        tensor/block_masked/view.hh
        tensor/block_masked/row_view.hh)
    get_filename_component(_header_parent "${_header}" DIRECTORY)
    install(FILES "${PROJECT_SOURCE_DIR}/include/Cellerator/${_header}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator/${_header_parent}"
        COMPONENT cellpack_interop)
endforeach()

if(TARGET cellPackCellBlockRecordsTest)
    target_link_libraries(cellPackCellBlockRecordsTest PRIVATE
        Cellerator::cellpack_interop)
endif()
if(TARGET cellPackWarpTilesTest)
    target_link_libraries(cellPackWarpTilesTest PRIVATE
        Cellerator::cellpack_interop)
endif()

function(cellerator_append_cellpack_interop_package_config)
    if(DEFINED CELLERATOR_HOST_CONFIG AND EXISTS "${CELLERATOR_HOST_CONFIG}")
        file(APPEND "${CELLERATOR_HOST_CONFIG}"
            "if(\"cellpack_interop\" IN_LIST Cellerator_FIND_COMPONENTS)\n"
            "  if(EXISTS \"\${CMAKE_CURRENT_LIST_DIR}/CelleratorCellpackInteropTargets.cmake\")\n"
            "    include(CMakeFindDependencyMacro)\n"
            "    find_dependency(CUDAToolkit)\n"
            "    include(\"\${CMAKE_CURRENT_LIST_DIR}/CelleratorCellpackInteropTargets.cmake\")\n"
            "    set(Cellerator_cellpack_interop_FOUND TRUE)\n"
            "  else()\n"
            "    set(Cellerator_cellpack_interop_FOUND FALSE)\n"
            "  endif()\n"
            "endif()\n")
    endif()
endfunction()
