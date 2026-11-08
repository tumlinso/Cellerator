"""Independent NumPy oracle for direct binary16 conversions."""
import subprocess
import sys

import numpy as np

rng = np.random.default_rng(98236)
# Adjacent finite binary16 values and exact midpoints with adjacent binary64 values.
h = np.arange(0x7C00, dtype=np.uint16).view(np.float16).astype(np.float64)
midpoints = (h[:-1] + h[1:]) / 2
above, below = np.nextafter(midpoints, np.inf), np.nextafter(midpoints, -np.inf)
boundaries = np.concatenate([
    midpoints, above, below, -midpoints, -above, -below,
    np.array([65520.0, -65520.0, 0.0, -0.0, np.inf, -np.inf, np.nan]),
])
floats = rng.integers(0, 2**32, 100000, dtype=np.uint32).view(np.float32)
doubles = np.concatenate([
    rng.integers(0, 2**64, 100000, dtype=np.uint64).view(np.float64), boundaries
])
with np.errstate(all="ignore"):
    want_f = floats.astype(np.float16).view(np.uint16)
    want_d = doubles.astype(np.float16).view(np.uint16)
want_f[np.isnan(floats)] = 0x7E00
want_d[np.isnan(doubles)] = 0x7E00
inputs = "".join(f"f {int(x):x}\n" for x in floats.view(np.uint32))
inputs += "".join(f"d {int(x):x}\n" for x in doubles.view(np.uint64))
result = subprocess.run(
    [sys.argv[1], "--float16-oracle"], input=inputs, text=True,
    capture_output=True, check=True,
)
got = np.fromiter((int(x, 16) for x in result.stdout.split()), dtype=np.uint16)
want = np.concatenate([want_f, want_d])
np.testing.assert_array_equal(got, want)
print(f"binary16 oracle passed {want.size} direct FP32/FP64 conversions")
