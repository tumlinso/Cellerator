"""A closed two-state stochastic network with a separate noisy observation.

Run from the checkout: PYTHONPATH=python python examples/framework/reaction_network.py
Time is in caller-chosen units; molecule counts are discrete and conserved.
"""
import random

from cellerator.dynamics.reactions import Reaction, ReactionNetwork
from cellerator.modeling import Cell, Model


def main():
    network = ReactionNetwork(["inactive", "active"], [
        Reaction.mass_action("activate", {"inactive": 1}, {"active": 1}, 0.6),
        Reaction.mass_action("deactivate", {"active": 1}, {"inactive": 1}, 0.3),
    ])
    dynamics_rng, observation_rng = random.Random(13), random.Random(31)

    def observation(state, context):
        # A deliberately supplied measurement assumption, not an inferred model.
        return {"fluorescence": state["active"] + observation_rng.gauss(0, 0.5)}

    cell = Cell(Model(network, observation), network.state({"inactive": 20}), id="cell-1")
    print("Conservation:", network.conservation_residuals({"inactive": 1, "active": 1}))
    for _ in range(5):
        step = cell.advance(0.5, rng=dynamics_rng)
        assert sum(cell.state.values()) == 20
        print({"time": cell.time, "counts": dict(cell.state), "events": len(step.events),
               "hazard_evaluations": step.metadata["propensity_evaluations"],
               "observation": cell.observe()})


if __name__ == "__main__":
    main()
