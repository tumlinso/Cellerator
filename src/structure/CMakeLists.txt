# Stable biological identity ABI and its preparation-time registry are
# structure-owned; the public execution namespace remains ABI-compatible.
add_library(cellerator_biological_abi INTERFACE)
add_library(Cellerator::biological_abi ALIAS cellerator_biological_abi)
target_include_directories(cellerator_biological_abi INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(cellerator_biological_abi INTERFACE cxx_std_17)

add_library(cellerator_identity_registry STATIC identity_registry.cc)
add_library(Cellerator::identity_registry ALIAS cellerator_identity_registry)
target_include_directories(cellerator_identity_registry PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(cellerator_identity_registry PUBLIC Cellerator::biological_abi)
target_compile_features(cellerator_identity_registry PUBLIC cxx_std_17)
