# Lightweight package metadata v1

Platforms supporting pkg-config receive `cellerator.pc` with prefix-relative
include/library paths and only the public `-lCellerator` link dependency.
Non-CMake Makefile consumers obtain flags through `pkg-config --cflags --libs
cellerator`; compiler resource/profile options remain explicit command flags.
