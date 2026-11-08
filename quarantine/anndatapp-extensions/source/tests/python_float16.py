"""Independent binary16 conversion oracle; run with pinned NumPy and test_arrays."""
import subprocess
import sys

import numpy as np

rng = np.random.default_rng(98236)
# All adjacent positive finite binary16 values; test each exact midpoint and
# its neighboring binary64 values with both signs.
h = np.arange(0x7c00, dtype=np.uint16).view(np.float16).astype(np.float64)
m = (h[:-1] + h[1:]) / 2
above, below = np.nextafter(m, np.inf), np.nextafter(m, -np.inf)
boundaries = np.concatenate([m, above, below, -m, -above, -below,
    np.array([65520., -65520., 0., -0., np.inf, -np.inf, np.nan])])
floats = rng.integers(0, 2**32, 100000, dtype=np.uint32).view(np.float32)
doubles = np.concatenate([rng.integers(0, 2**64, 100000, dtype=np.uint64).view(np.float64), boundaries])
with np.errstate(all="ignore"):
    want_f = floats.astype(np.float16).view(np.uint16)
    want_d = doubles.astype(np.float16).view(np.uint16)
want_f[np.isnan(floats)] = 0x7e00
want_d[np.isnan(doubles)] = 0x7e00
inputs = "".join(f"f {int(x):x}\n" for x in floats.view(np.uint32))
inputs += "".join(f"d {int(x):x}\n" for x in doubles.view(np.uint64))
result = subprocess.run([sys.argv[1], "--float16-oracle"], input=inputs,
    text=True, capture_output=True, check=True)
got = np.fromiter((int(x, 16) for x in result.stdout.split()), dtype=np.uint16)
want = np.concatenate([want_f, want_d])
np.testing.assert_array_equal(got, want)
print(f"binary16 oracle passed {want.size} direct FP32/FP64 conversions")
