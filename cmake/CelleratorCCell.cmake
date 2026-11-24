function(cellerator_add_ccell)
    set(CMAKE_CXX_SCAN_FOR_MODULES ON)
    add_subdirectory(components/CCell)
endfunction()

cellerator_add_ccell()
