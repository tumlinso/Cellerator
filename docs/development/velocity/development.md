# Velocity development

Build Velocity through the Cellerator root. The [single-cell build guide](../single-cell-methods.md)
covers the native build and server checks. Enable `CELLERATOR_ENABLE_VELOCITY`
and `CELLERATOR_BUILD_TESTS` for the Velocity checks.

Installed consumers use Cellerator’s Velocity component.

```cmake
find_package(Cellerator CONFIG REQUIRED COMPONENTS velocity)
target_link_libraries(my_app PRIVATE Cellerator::velocity)
set_target_properties(my_app PROPERTIES CXX_EXTENSIONS OFF)
```

`import cellerator.velocity;` exposes fitting, sensitivity, and selective precision.
Read the [API](api.md) for buffer, lifetime, and preparation contracts and the
[contributor guide](../../../components/velocity/AGENTS.md) for ownership and style.

The numerical, memory, GPU, and package receipts in [validation](validation.md)
remain tied to their recorded sources. Use the configured Project Control
workflow for current work. Keep generated data, environments, binaries, and
measurement outputs out of the tracked source.
