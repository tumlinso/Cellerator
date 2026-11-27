function(cellerator_target_runtime_sources target)
    target_sources(${target} PRIVATE
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/runtime.cu"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/device_descriptor.cu"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/session_memory.cu"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../ops/matrix/sparse/project/blocked_ell_descriptor.cu"
    )
endfunction()
