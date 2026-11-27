include_guard(GLOBAL)

function(cellerator_install_native_headers root)
    foreach(dir IN ITEMS bindings biology common execution geometry interop layout memory
            ops runtime state structure tensor)
        install(DIRECTORY "${root}/include/Cellerator/${dir}"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator" ${ARGN}
            PATTERN "cellshard" EXCLUDE)
    endforeach()
    install(FILES "${root}/include/Cellerator/parameters.hh"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Cellerator" ${ARGN})
endfunction()
