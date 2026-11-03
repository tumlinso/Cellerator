"""Exact Fraction oracle for a gated ordered-multiplicative mechanism batch."""
from fractions import Fraction as F

beta, theta, phi, tau = F(1, 4), F(2, 5), F(1, 2), F(1, 2)
z0, z1 = F(1), F(1, 2)
x = [F(2), F(3), F(1, 2), F(0), F(5, 2)]
loss_seed = [F(1), F(0), F(-1, 2), F(0), F(0)]
groups = [
    {"inputs": [0, 1, 1], "outputs": {0: F(1), 2: F(1, 2)}, "device": 0},
    {"inputs": [1, 2], "outputs": {0: F(2)}, "device": 1},
    {"inputs": [3, 4], "outputs": {0: F(1)}, "device": 1},
    # Repeated zero factors have a finite, zero derivative per occurrence.
    {"inputs": [3, 3], "outputs": {0: F(1, 4)}, "device": 1},
]

def gate(a, b, p, t):
    s = a + p * b - t
    return s, s * s if s > 0 else F(0)

def evaluate(values, a, b, k, p, t):
    _, g = gate(a, b, p, t)
    y = [-beta * value for value in values]
    for group in groups:
        product = F(1)
        for index in group["inputs"]:
            product *= values[index]
        for out, scale in group["outputs"].items():
            y[out] += scale * k * g * product
    return y

def group_vjp(group, values, a, b, k, p, t):
    s, g = gate(a, b, p, t)
    result = {"x": [F(0)] * 5, "theta": F(0), "z0": F(0),
              "z1": F(0), "phi": F(0), "tau": F(0)}
    if s <= 0:
        return result  # Explicit inactive branch: no participant reads/VJPs.
    upstream = sum((scale * loss_seed[out]
                    for out, scale in group["outputs"].items()), F(0))
    product = F(1)
    for index in group["inputs"]:
        product *= values[index]
    for slot, index in enumerate(group["inputs"]):
        other = F(1)
        for j, other_index in enumerate(group["inputs"]):
            if j != slot:
                other *= values[other_index]
        result["x"][index] += upstream * k * g * other
    ds = upstream * k * 2 * s * product
    result.update(theta=upstream * g * product, z0=ds, z1=ds * p,
                  phi=ds * b, tau=-ds)
    return result

def objective(values, a, b, k, p, t):
    return sum((seed * output for seed, output in
                zip(loss_seed, evaluate(values, a, b, k, p, t))), F(0))

def centered_difference(f, value, h=F(1, 10000)):
    return (f(value + h) - f(value - h)) / (2 * h)

# Active exact reference: repeats, overlap, two output routes, shared parameter,
# and a zero factor whose own input derivative remains nonzero.
outputs = evaluate(x, z0, z1, theta, phi, tau)
partials = [group_vjp(g, x, z0, z1, theta, phi, tau) for g in groups]
active = {name: sum((part[name] for part in partials), F(0))
          for name in ("theta", "z0", "z1", "phi", "tau")}
active["x"] = [sum((part["x"][i] for part in partials), F(0))
               + (-beta * loss_seed[i] if i < len(loss_seed) else F(0))
               for i in range(5)]
assert outputs == [F(169, 40), F(-3, 4), F(19, 10), F(0), F(-5, 8)]
assert active["x"] == [F(203, 160), F(9, 4), F(59, 40), F(9, 16), F(0)]
assert tuple(active[k] for k in ("theta", "z0", "z1", "phi", "tau")) == (
    F(297, 32), F(99, 10), F(99, 20), F(99, 20), F(-99, 10))

# All ten active first derivatives agree exactly with centered differences;
# the tested active polynomial makes these centered differences exact.
h = F(1, 10000)
fdx = []
for i in range(len(x)):
    plus, minus = x.copy(), x.copy()
    plus[i] += h
    minus[i] -= h
    fdx.append((objective(plus, z0, z1, theta, phi, tau)
                - objective(minus, z0, z1, theta, phi, tau)) / (2 * h))
fd_params = {
    "z0": centered_difference(lambda a: objective(x, a, z1, theta, phi, tau), z0, h),
    "z1": centered_difference(lambda b: objective(x, z0, b, theta, phi, tau), z1, h),
    "theta": centered_difference(lambda k: objective(x, z0, z1, k, phi, tau), theta, h),
    "phi": centered_difference(lambda p: objective(x, z0, z1, theta, p, tau), phi, h),
    "tau": centered_difference(lambda t: objective(x, z0, z1, theta, phi, t), tau, h),
}
assert fdx == active["x"]
assert all(fd_params[name] == active[name] for name in fd_params)

# Per-device partials must reduce to the single-device result. x1 overlaps
# across devices and theta/condition gradients are shared across the partition.
dev0 = partials[0]
dev1 = {name: sum((p[name] for p in partials[1:]), F(0))
         for name in ("theta", "z0", "z1", "phi", "tau")}
dev1["x"] = [sum((p["x"][i] for p in partials[1:]), F(0)) for i in range(5)]
assert (dev0["theta"], dev1["theta"]) == (F(243, 32), F(27, 16))
assert dev0["x"][1] + dev1["x"][1] == F(9, 4)
assert dev0["z0"] + dev1["z0"] == active["z0"]
# On device 1, the shared participant x1, remote condition z0 and shared
# parameter theta all have active, independently nonzero contributions.
assert dev1["x"][1] == F(9, 40) and dev1["z0"] == F(9, 5)
assert dev1["theta"] == F(27, 16)
# The repeated [zero, zero] group remains finite and has zero per-slot VJP;
# the preceding [zero, nonzero] group separately has nonzero d/dx3.
assert partials[2]["x"][3] == F(9, 16)
assert partials[3]["x"][3] == 0
assert partials[3]["theta"] == partials[3]["z0"] == 0

# One canonical-owner SGD update with eta=1/100, then refresh all replicas.
eta = F(1, 100)
theta_next = theta - eta * active["theta"]
assert theta_next == F(983, 3200)
next_outputs = evaluate(x, z0, z1, theta_next, phi, tau)
assert (next_outputs[0], next_outputs[2]) == (F(160187, 51200), F(73223, 51200))
assert theta - eta * dev0["theta"] != theta_next
assert theta - eta * dev1["theta"] != theta_next

# Strictly inactive gate: exact baseline output/VJP and zero mechanism VJPs.
zi = (F(1, 10), F(1, 2), F(1, 2), F(3, 5))
assert gate(*zi)[0] == F(-1, 4)
inactive_y = evaluate(x, zi[0], zi[1], theta, zi[2], zi[3])
inactive = [group_vjp(g, x, zi[0], zi[1], theta, zi[2], zi[3]) for g in groups]
assert inactive_y == [F(-1, 2), F(-3, 4), F(-1, 8), F(0), F(-5, 8)]
assert all(not any(p["x"]) and all(p[k] == 0 for k in ("theta", "z0", "z1", "phi", "tau"))
           for p in inactive)
boundary_vjp = [group_vjp(g, x, F(0), F(1), F(1), F(1), F(1)) for g in groups]
assert all(not any(p["x"]) and all(p[k] == 0 for k in ("theta", "z0", "z1", "phi", "tau"))
           for p in boundary_vjp)

# A same-size reordered value association changes the reference computation;
# the native binding API must reject the corresponding stale-order descriptor.
wrong_order_x = x.copy()
wrong_order_x[0], wrong_order_x[1] = wrong_order_x[1], wrong_order_x[0]
assert len(wrong_order_x) == len(x)
assert evaluate(wrong_order_x, z0, z1, theta, phi, tau) != outputs

# Boundary s=0 for ReLU(s)^2 has derivative zero; generic gate output zero does
# not certify a safe omission: g(u,v)=u*v has nonzero dg/du when u=0, v=3.
assert gate(F(0), F(1), F(1), F(1)) == (F(0), F(0))
u, v, participant, coeff = F(0), F(3), F(2), F(1, 2)
assert coeff * u * v * participant == 0
assert coeff * v * participant == 3

print("active outputs:", [str(v) for v in outputs])
print("active dx:", [str(v) for v in active["x"]])
print("active dtheta,dz0,dz1,dphi,dtau:",
      [str(active[k]) for k in ("theta", "z0", "z1", "phi", "tau")])
print("device dtheta partials:", str(dev0["theta"]), str(dev1["theta"]))
print("post-update theta and y0,y2:", str(theta_next), str(next_outputs[0]), str(next_outputs[2]))
print("inactive outputs and mechanism VJP: baseline only; 10 active central differences exact")
print("zero-output generic gate counterexample: forward=0, d/du=3")
