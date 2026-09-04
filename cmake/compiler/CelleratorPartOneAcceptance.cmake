function(cellerator_add_part_one_acceptance suffix source)
    set(target "ce_ccp1_j03_${suffix}")
    if(NOT TARGET ${target} AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        add_executable(${target} "${source}")
        target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_compile_features(${target} PRIVATE cxx_std_17)
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES LABELS "ce_ccp1_m90;part_one")
    endif()
endfunction()

foreach(cellerator_j03_suffix IN ITEMS
        001 002 003 004 005 006 007 008 009 010 011 012 013)
    file(GLOB cellerator_j03_source CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/compiler/j03/*_test.cc")
    foreach(cellerator_j03_candidate IN LISTS cellerator_j03_source)
        get_filename_component(cellerator_j03_name "${cellerator_j03_candidate}" NAME)
        if(cellerator_j03_name MATCHES "^${cellerator_j03_suffix}_")
            file(RELATIVE_PATH cellerator_j03_relative
                "${CMAKE_CURRENT_SOURCE_DIR}" "${cellerator_j03_candidate}")
            cellerator_add_part_one_acceptance(
                "${cellerator_j03_suffix}" "${cellerator_j03_relative}")
        endif()
    endforeach()
endforeach()
