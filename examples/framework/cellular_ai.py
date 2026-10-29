"""A small, end-to-end cellular model built from Cellerator primitives.

This invented example has sparse gene-addressed state, two state-gated
interactions, shared dynamics and parameters, an observation map, and a signal
routed between cells. Its values and rules are toy inputs, not biological
claims or validation.
"""
from cellerator.events import Event
from cellerator.execution import Population, Send
from cellerator.interactions import Interaction, InteractionGraph
from cellerator.modeling import Cell, Model, StepContext, StepResult
from cellerator.state import SparseState


def build_model() -> Model:
    relations = InteractionGraph(
        (
            Interaction.hyperedge(
                "interferon-response",
                ("STAT1", "IFNAR"),
                ("IFIT1",),
                lambda values, context: max(0.0, values[0] * values[1] * context["gain"]),
                active_if=lambda state, _: state["IFNAR"] >= 0.5,
            ),
            Interaction.hyperedge(
                "inflammatory-response",
                ("NFKB", "TNFR"),
                ("IL1B",),
                lambda values, context: max(0.0, values[0] * values[1] * context["gain"]),
                active_if=lambda state, _: state["TNFR"] >= 0.5,
            ),
        )
    )
    parameters = {"gain": 0.8, "turnover": 0.25, "event_threshold": 0.25}

    def advance(state: SparseState, context: StepContext) -> StepResult:
        before = dict(state)
        values = dict(before)
        incoming = getattr(context.inputs, "incoming", ())
        for record in incoming:
            message = record.value
            if message.get("kind") == "cytokine-pulse":
                target = message["target_quantity"]
                values[target] = values.get(target, 0.0) + float(message["dose"])

        # Candidate lookup visits only relations incident to these quantities.
        changed = set(values) if incoming else {"STAT1", "IFNAR", "NFKB", "TNFR"}
        interaction_result = relations.evaluate(
            values,
            changed=changed,
            context={"gain": context.parameters["gain"]},
        )

        end_time = context.time + context.dt
        turnover = float(context.parameters["turnover"])
        next_values = dict(values)
        activated = []
        for target, production in interaction_result.writes.items():
            previous = float(values.get(target, 0.0))
            current = (previous + context.dt * float(production)) / (1.0 + context.dt * turnover)
            next_values[target] = current
            if previous < context.parameters["event_threshold"] <= current:
                activated.append(Event(
                    "gene-activation", time=end_time,
                    payload={"feature": target, "value": current},
                    source=context.metadata.get("population_id"),
                ))

        effects = ()
        source_id = context.metadata.get("population_id")
        if source_id == "interferon-cell" and next_values["IFIT1"] > 0.25:
            effects = (Send(
                "neighbor-cell",
                {"kind": "cytokine-pulse", "target_quantity": "NFKB", "dose": 0.7},
            ),)
        return StepResult(
            state=SparseState(next_values), events=tuple(activated),
            effects=effects, time=end_time,
        )

    def observe(state: SparseState, _: StepContext) -> dict[str, float]:
        """Return selected readouts, separately from the modeled state."""
        return {name: float(state[name]) for name in ("IFIT1", "IL1B")}

    return Model(advance, observation=observe, parameters=parameters)


def main() -> None:
    model = build_model()  # Shared rule and parameters; cells own distinct state.
    first = Cell(
        model,
        SparseState({"STAT1": 1.2, "IFNAR": 1.0, "IFIT1": 0.0,
                     "NFKB": 0.0, "TNFR": 1.0, "IL1B": 0.0}),
        id="interferon-cell",
    )
    neighbor = Cell(
        model,
        SparseState({"STAT1": 0.0, "IFNAR": 0.0, "IFIT1": 0.0,
                     "NFKB": 0.0, "TNFR": 1.0, "IL1B": 0.0}),
        id="neighbor-cell",
    )
    population = Population((first, neighbor), context_capacity=8)

    initial_observation = first.observe()
    first_update = population.advance(dt=1.0)
    assert first_update.results[first.id].events
    assert population.pending(neighbor.id)
    first_readout = first.observe()

    population.advance(dt=1.0)
    neighbor_readout = neighbor.observe()

    assert initial_observation["IFIT1"] == 0.0
    assert first_readout["IFIT1"] > 0.0
    assert neighbor_readout["IL1B"] > 0.0
    assert first.model is neighbor.model is model
    print("shared rule; distinct state:", first.model is neighbor.model, first.state is not neighbor.state)
    print("first cell observation:", first_readout)
    print("neighbor after routed signal:", neighbor_readout)
    print("neighbor state identities:", tuple(neighbor.state))


if __name__ == "__main__":
    main()
