"""Independent CPU replay of the tiny native training example.

Uses the documented one-mechanism law, FP16 primals/parameters/output, FP32
forward/VJP arithmetic, and round-to-nearest FP16 parameter updates. It does not
call the CUDA library, PyTorch adapter, or validation oracle.
"""

import math
import struct


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f16(value):
    return struct.unpack("<e", struct.pack("<e", float(value)))[0]


def sigmoid_and_slope(logit):
    t = f32(math.exp(-abs(logit)))
    denominator = f32(1.0 + t)
    gate = f32(1.0 / denominator) if logit >= 0 else f32(t / denominator)
    slope = f32(t / f32(denominator * denominator))
    return gate, slope


def replay(iterations):
    x0, x1, z = (f16(value) for value in (0.75, 0.60, 0.50))
    alpha, bias, gamma = (f16(value) for value in (0.80, -0.20, 0.90))
    target = f32(0.25)
    learning_rate = f32(0.05)
    participant_product = f32(float(x0) * float(x1))
    minimum = [alpha, bias, gamma]
    maximum = [alpha, bias, gamma]
    initial_output = None

    for _ in range(iterations):
        # The example has one context term (gamma[0] * z[0]).
        logit = f32(float(bias) + float(gamma) * float(z))
        gate, slope = sigmoid_and_slope(logit)
        output = f16(f32(f32(participant_product * float(alpha)) * gate))
        if initial_output is None:
            initial_output = float(output)
        dy = f32(float(output) - target)

        # One output channel and one mechanism leave one contribution per
        # parameter. The operation order matches the native VJP expressions.
        d_alpha = f32(f32(participant_product * gate) * dy)
        d_bias = f32(
            f32(f32(f32(participant_product * float(alpha)) * dy) * slope)
        )
        d_gamma = f32(d_bias * float(z))

        alpha = f16(f32(float(alpha) - f32(learning_rate * d_alpha)))
        bias = f16(f32(float(bias) - f32(learning_rate * d_bias)))
        gamma = f16(f32(float(gamma) - f32(learning_rate * d_gamma)))

        parameters = (alpha, bias, gamma)
        if not all(math.isfinite(value) and abs(value) <= 1.0 for value in parameters):
            raise RuntimeError(f"parameter left finite [-1,1] domain: {parameters}")
        minimum = [min(lo, value) for lo, value in zip(minimum, parameters)]
        maximum = [max(hi, value) for hi, value in zip(maximum, parameters)]

    logit = f32(float(bias) + float(gamma) * float(z))
    gate, _ = sigmoid_and_slope(logit)
    output = f16(f32(f32(participant_product * float(alpha)) * gate))
    error = abs(float(output) - target)
    return {
        "steps": iterations,
        "initial_output": initial_output,
        "output": float(output),
        "absolute_error": error,
        "alpha": float(alpha),
        "bias": float(bias),
        "gamma": float(gamma),
        "parameter_min": [float(value) for value in minimum],
        "parameter_max": [float(value) for value in maximum],
    }


if __name__ == "__main__":
    for step_count in (200, 10000):
        print(replay(step_count))
