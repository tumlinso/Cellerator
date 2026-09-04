# Source frontend and C bridge benchmark v1

Matched pure-C++ and file-opted-in Cellerator sources isolate preprocessing,
activated-token analysis, shadow generation, Clang Sema, AST construction,
incremental reuse, and source-map memory. Every phase retains eleven raw samples
under the benchmark mutex. The reported Cellerator-field cost is the paired
difference; common Clang work is never attributed to the extension.
