# CMake compiler integration helpers v1

`cellerator_compile_cell` integrates `.cell` compilation through custom
commands with response-safe argument lists, depfiles, explicit standard library,
profile and backend selection, optional LTO, and generator-independent outputs.
The profile argument is mandatory.
