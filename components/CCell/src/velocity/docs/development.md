# Velocity development

Build Velocity through the Cellerator root. The [CCell build guide](../../../README.md)
shows the standard and native-method builds. Native Velocity tests and examples
use caller-provided Cellerator views. Run the CCell tests with `CCELL_BUILD_TESTS`
enabled.

Installed consumers use the CCell package and target:

```cmake
find_package(ccell CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE ccell::velocity)
set_target_properties(my_app PROPERTIES CXX_EXTENSIONS OFF)
```

`import ccell.velocity;` exposes fitting, sensitivity, and selective precision.
Read the [API](api.md) for buffer, lifetime, and preparation contracts and the
[contributor guide](../AGENTS.md) for ownership and style.

The numerical, memory, GPU, and package receipts in [validation](validation.md)
remain tied to their recorded sources. Use the configured Project Control
workflow for current work. Keep generated data, environments, binaries, and
measurement outputs out of the tracked source.
