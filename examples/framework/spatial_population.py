"""Route updates across a sparse spatial neighborhood and conserve field mass."""
from cellerator.interactions import InteractionResult
from cellerator.spatial import SpatialIndex, diffuse


def main():
    positions = SpatialIndex({
        "cell-a": (0.0, 0.0),
        "cell-b": (0.2, 0.1),
        "cell-c": (3.0, 0.0),
    }, search_radius=0.5)
    expression = {"cell-a": 1.0, "cell-b": 3.0, "cell-c": 7.0}
    graph = positions.interaction_graph(
        lambda source, target, distance, context: context["coupling"] * (source - target),
        active_if=lambda state, _: state.get("regulatory_active", False),
    )
    step = graph.evaluate({**expression, "regulatory_active": True},
                          context={"coupling": 0.1}, reducer="sum")
    print("nearby directed relations:", len(graph))
    print("local regulatory contributions:", dict(step.writes))

    concentrations = {"cell-a": 1.0, "cell-b": 0.0, "cell-c": 4.0}
    next_concentrations = diffuse(
        positions, concentrations, dt=0.1, diffusivity=0.2)
    print("concentration before:", concentrations, "mass=", sum(concentrations.values()))
    print("concentration after: ", next_concentrations,
          "mass=", sum(next_concentrations.values()))

    positions.move("cell-b", (2.9, 0.0))
    print("nearby pairs after movement:", positions.pairs(directed=False))


if __name__ == "__main__":
    main()
