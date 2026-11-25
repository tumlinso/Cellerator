include("${CMAKE_CURRENT_LIST_DIR}/CelleratorPreprocessTargets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CelleratorModuleInterfaces.cmake")
cellerator_import_compiled_module(preprocess Cellerator::preprocess_binary Cellerator::headers)
set(Cellerator_preprocess_FOUND TRUE)
