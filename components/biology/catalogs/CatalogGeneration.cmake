add_executable(cellerator_generate_catalog EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/generate_catalog.cc"
    $<TARGET_OBJECTS:cellerator_gene_annotation>)
set_target_properties(cellerator_generate_catalog PROPERTIES CXX_EXTENSIONS OFF)
target_compile_features(cellerator_generate_catalog PRIVATE cxx_std_20)
target_include_directories(cellerator_generate_catalog PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../include")

function(cellerator_generate_gene_catalog target directory manifest input)
    include("${manifest}")
    if(NOT input)
        set(data_root "$ENV{CELLERATOR_DATA_ROOT}")
        if(NOT data_root)
            set(data_root "$ENV{HOME}/data/Cellerator")
        endif()
        set(input "${data_root}/reference/biology/${catalog_filename}")
    endif()
    if(NOT EXISTS "${input}")
        message(FATAL_ERROR
            "Missing Biology annotation '${input}'. Stage the pinned file from ${catalog_source} "
            "or set CELLERATOR_BIOLOGY_ANNOTATION to its local path.")
    endif()
    set(output "${directory}/gene_catalog.gen.inc")
    add_custom_command(OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}"
            "-DGENERATOR=$<TARGET_FILE:cellerator_generate_catalog>"
            "-DEMU=$<TARGET_PROPERTY:cellerator_generate_catalog,CROSSCOMPILING_EMULATOR>"
            "-DINPUT=${input}" "-DOUTPUT=${output}" "-DSHA256=${catalog_sha256}"
            "-DFORMAT=${catalog_format}" "-DASSEMBLY=${catalog_assembly}"
            "-DRELEASE=${catalog_release}" "-DSOURCE=${catalog_source}"
            "-DSUBSETS=${catalog_subsets}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/generate_catalog.cmake"
        DEPENDS cellerator_generate_catalog "${manifest}" "${input}" ${catalog_subsets}
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/generate_catalog.cmake"
        VERBATIM)
    target_sources(${target} PRIVATE "${output}")
    target_compile_definitions(${target} PRIVATE CELLERATOR_HAS_BUILTIN_GENE_CATALOG)
    target_include_directories(${target} PRIVATE "${directory}")
endfunction()
