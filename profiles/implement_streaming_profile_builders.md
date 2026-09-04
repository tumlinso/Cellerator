# Streaming profile builders v1

Caller-budgeted builders combine count/scan/fill offsets, fixed histograms,
bounded top-L values, a fixed sketch, streaming moments, and an exact-small mode.
Requirements are explicit before initialization and no update allocates. Exact
mode fails at its declared capacity instead of silently approximating.
