# Runtime ownership

The CE-ARCH execution session is Cellerator's runtime authority. Device,
stream, library, scratch, readiness, and multi-GPU collective resources live
here. `multi_gpu/fleet.cuh` provides the sparse API resource
surface for compatibility; it does not own planning or create a second runtime.
