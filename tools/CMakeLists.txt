if(CELLERATOR_BUILD_PREPROCESS_WORKBENCH AND PkgConfig_FOUND)
    pkg_check_modules(CELLERATOR_NCURSESW QUIET ncursesw)
    if(CELLERATOR_NCURSESW_FOUND)
        add_executable(celleratorPreprocessWorkbench
            tools/preprocess_workbench/workbench_main.cc
        )
        target_include_directories(celleratorPreprocessWorkbench PRIVATE
            ${CELLERATOR_NCURSESW_INCLUDE_DIRS}
        )
        target_compile_options(celleratorPreprocessWorkbench PRIVATE
            ${CELLERATOR_NCURSESW_CFLAGS_OTHER}
        )
        target_link_directories(celleratorPreprocessWorkbench PRIVATE
            ${CELLERATOR_NCURSESW_LIBRARY_DIRS}
        )
        target_link_libraries(celleratorPreprocessWorkbench PRIVATE
            Cellerator::preprocess
            ${CELLERATOR_NCURSESW_LIBRARIES}
        )
        set_target_properties(celleratorPreprocessWorkbench PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED YES
        )
    endif()
endif()
