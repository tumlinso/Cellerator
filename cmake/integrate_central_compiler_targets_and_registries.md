# Central compiler integration receipt

Part One compiler acceptance targets are registered centrally by
`cmake/compiler/CelleratorPartOneAcceptance.cmake`. The registry recognizes the
source, frontend, semantic, profile, three CEIR, reflection, pass, LTO, tooling,
and SDK subsystems exactly once. It does not add another compiler authority.

The module is included only when tests are enabled and discovers the bounded J03
acceptance sources by their task prefix. Production compiler targets remain owned
by their subsystem CMake fragments and exported through the existing compiler
target graph.
