"""Finite signal secretion and uptake through one shared extracellular bath.

Run: PYTHONPATH=python python examples/framework/extracellular_coupling.py
This is a conservative numerical example, not a calibrated biological model.
"""
from __future__ import annotations

import numpy as np

from cellerator.modeling import DynamicalModel, StepContext, StepResult
from cellerator.spatial.coupling import ExchangeLink, ExchangePlan, ExchangeState, concentrations


def run(steps: int = 100, dt: float = 0.1):
    identities = ("producer", "receiver-a", "receiver-b")
    plan = ExchangePlan(identities, ("bath",), ("signal", "nutrient"),
                        [ExchangeLink(identity, identity, "bath") for identity in identities])
    cell_volumes = {"producer": 1., "receiver-a": 1., "receiver-b": 2.}
    env_volumes = {"bath": 4.}
    initial = ExchangeState({"producer": [8., 0.], "receiver-a": [0., 0.],
                             "receiver-b": [0., 0.]}, {"bath": [0., 12.]})

    def coupled_step(state, context):
        # Every active law reads this same old extracellular concentration.
        signal, nutrient = concentrations(state.compartments, env_volumes)["bath"]
        rates = {"producer": [1.2, -0.1 * nutrient],
                 "receiver-a": [-0.5 * signal, -0.2 * nutrient],
                 "receiver-b": [-0.8 * signal, -0.3 * nutrient]}
        result = plan.step(state, cell_volumes=cell_volumes,
                           compartment_volumes=env_volumes, dt=context.dt,
                           permeability=[0.05, 0.], active_rates=rates,
                           donor_policy="limit")
        return StepResult(result.state, time=context.time + context.dt,
                          metadata={"limited": result.limited,
                                    "transfers": result.transfers})

    model = DynamicalModel(coupled_step)
    state = initial
    for step in range(steps):
        state = model.step(state, StepContext(time=step * dt, dt=dt)).state
    total = np.sum(list(state.cells.values()) + list(state.compartments.values()), axis=0)
    np.testing.assert_allclose(total, [8., 12.], atol=1e-12)
    return state, total


if __name__ == "__main__":
    state, total = run()
    for identity, amounts in state.cells.items():
        print(f"{identity}: signal={amounts[0]:.5f}, nutrient={amounts[1]:.5f}")
    print("bath:", state.compartments["bath"])
    print("conserved species totals:", total)
