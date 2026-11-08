# This check builds CUDA consumers only. Run them through Device Scheduler.
function(checked)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  file(APPEND "${VELOCITY_BUILD_DIR}/installed-cuda-test/commands.log" "${ARGV}\n${out}\n${err}\n")
  if(NOT result EQUAL 0)
    string(SUBSTRING "${err}" 0 2500 preview)
    message(FATAL_ERROR "CUDA consumer command failed (${result}): ${ARGV}\n${preview}\nFull output: ${VELOCITY_BUILD_DIR}/installed-cuda-test/commands.log")
  endif()
endfunction()
set(prefix "${VELOCITY_BUILD_DIR}/installed-cuda-test/prefix")
set(consumer "${VELOCITY_BUILD_DIR}/installed-cuda-test/consumer-qualified")
file(MAKE_DIRECTORY "${VELOCITY_BUILD_DIR}/installed-cuda-test")
checked("${CMAKE_COMMAND}" --install "${VELOCITY_BUILD_DIR}" --prefix "${prefix}")
checked("${CMAKE_COMMAND}" -S "${VELOCITY_SOURCE_DIR}/tests/cuda_consumer" -B "${consumer}" -G Ninja
  "-DCMAKE_CXX_COMPILER=${VELOCITY_COMPILER}" "-DCMAKE_BUILD_TYPE=${VELOCITY_CONFIG}"
  "-DCMAKE_CUDA_COMPILER=${VELOCITY_CUDA_COMPILER}"
  "-DCMAKE_CUDA_ARCHITECTURES=70"
  "-DCMAKE_PREFIX_PATH=${prefix}\;${VELOCITY_ANNDATA_PREFIX}")
checked("${CMAKE_COMMAND}" --build "${consumer}" --parallel 2)
