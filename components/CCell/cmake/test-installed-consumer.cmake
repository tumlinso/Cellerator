function(checked)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Consumer command failed (${result}): ${ARGV}\n${out}\n${err}")
  endif()
endfunction()

string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef test_id)
set(test_dir "${CCELL_BUILD_DIR}/installed-test-${test_id}")
set(prefix "${test_dir}/prefix")
set(consumer "${test_dir}/consumer")
checked("${CMAKE_COMMAND}" --install "${CELLERATOR_BUILD_DIR}" --prefix "${prefix}"
  --config "${CCELL_CONFIG}")

foreach(notice IN ITEMS LICENSE THIRD_PARTY_NOTICES.md velocity/THIRD_PARTY_NOTICES.md)
  if(NOT EXISTS "${prefix}/share/licenses/ccell/${notice}")
    message(FATAL_ERROR "Missing installed CCell notice: ${notice}")
  endif()
endforeach()
checked("${CMAKE_COMMAND}" -S "${CCELL_SOURCE_DIR}/tests/consumer" -B "${consumer}" -G Ninja
  "-DCMAKE_CXX_COMPILER=${CCELL_COMPILER}"
  "-DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS=${CCELL_SCAN_DEPS}"
  "-DCMAKE_BUILD_TYPE=${CCELL_CONFIG}"
  "-DCMAKE_PREFIX_PATH=${prefix}"
  "-DCCELL_ANNDATA_INTEROP=${CCELL_ANNDATA_INTEROP}"
  "-DCCELL_TEST_ANNDATA_MODELING=${CCELL_TEST_ANNDATA_MODELING}")
checked("${CMAKE_COMMAND}" --build "${consumer}" --parallel 2)
checked("${CMAKE_CTEST_COMMAND}" --test-dir "${consumer}" --output-on-failure)
file(REMOVE_RECURSE "${test_dir}")
