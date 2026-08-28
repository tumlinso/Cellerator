# Runtime ownership

The CE-ARCH execution session is Cellerator's runtime authority. Device,
stream, library, scratch, readiness, and multi-GPU collective resources live
here. `legacy_execution_context.hh` retains the older sparse API resource
surface for compatibility; it does not own planning or create a second runtime.
