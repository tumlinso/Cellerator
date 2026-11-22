function(cellerator_add_ccell)
    set(CMAKE_CXX_SCAN_FOR_MODULES ON)
    if(CELLERATOR_ENABLE_CCELL_ANNDATA)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/anndatapp/CMakeLists.txt")
            message(FATAL_ERROR
                "AnnData++ source is missing. Run git submodule update --init -- extern/anndatapp, "
                "or configure with CELLERATOR_ENABLE_CCELL_ANNDATA=OFF for native methods only.")
        endif()
        add_subdirectory(extern/anndatapp)
    endif()
    add_subdirectory(components/CCell)
endfunction()

cellerator_add_ccell()
