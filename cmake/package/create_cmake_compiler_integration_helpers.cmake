include(CMakeParseArguments)
function(cellerator_compile_cell target)
  cmake_parse_arguments(CELL "LTO" "PROFILE;BACKEND;STDLIB" "SOURCES;OPTIONS" ${ARGN})
  if(NOT CELL_PROFILE)
    message(FATAL_ERROR "cellerator_compile_cell requires PROFILE")
  endif()
  set(_outputs)
  foreach(_source IN LISTS CELL_SOURCES)
    get_filename_component(_stem "${_source}" NAME_WE)
    set(_output "${CMAKE_CURRENT_BINARY_DIR}/${_stem}.o")
    add_custom_command(OUTPUT "${_output}"
      COMMAND Cellerator::Compiler --profile "${CELL_PROFILE}" --backend "${CELL_BACKEND}"
              --stdlib "${CELL_STDLIB}" --depfile "${_output}.d" ${CELL_OPTIONS}
              -c "${_source}" -o "${_output}"
      DEPFILE "${_output}.d" DEPENDS "${_source}" VERBATIM COMMAND_EXPAND_LISTS)
    list(APPEND _outputs "${_output}")
  endforeach()
  target_sources(${target} PRIVATE ${_outputs})
endfunction()
