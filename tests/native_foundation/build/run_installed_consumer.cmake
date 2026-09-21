foreach(required CE_NF1_CONSUMER_SOURCE CE_NF1_CONSUMER_BINARY CE_NF1_INSTALL_PREFIX CE_NF1_CUDA_COMPILER)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${CE_NF1_CONSUMER_SOURCE}"
    -B "${CE_NF1_CONSUMER_BINARY}" "-DCMAKE_PREFIX_PATH=${CE_NF1_INSTALL_PREFIX}"
    "-DCMAKE_CUDA_COMPILER=${CE_NF1_CUDA_COMPILER}"
    RESULT_VARIABLE configure_result)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "installed consumer configuration failed: ${configure_result}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${CE_NF1_CONSUMER_BINARY}" --parallel 2
    RESULT_VARIABLE build_result)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "installed consumer build failed: ${build_result}")
endif()
execute_process(COMMAND "${CE_NF1_CONSUMER_BINARY}/ce_nf1_installed_width33"
    RESULT_VARIABLE run_result)
if(NOT run_result EQUAL 0)
    message(FATAL_ERROR "installed consumer execution failed: ${run_result}")
endif()
