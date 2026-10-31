"""Route a signal between persistent cells; suspend and resume its reader.

Signal delivery here illustrates a computational contract, not a model of a
particular ligand, receptor, or biological timescale.
"""
from cellerator.execution import Population, Send
from cellerator.modeling import DynamicalCell, DynamicalModel, StepResult


def main() -> None:
    def signal_rule(state, context):
        if state["emitted"]:
            return state
        return StepResult(
            {**state, "emitted": True},
            effects=(Send("receiver", {"signal": "pulse", "dose": 1.0}),),
        )

    def receive_rule(state, context):
        incoming = getattr(context.inputs, "incoming", ())
        signals = tuple(record.value for record in incoming)
        return {**state, "received": state["received"] + signals}

    sender = DynamicalCell(DynamicalModel(signal_rule), {"emitted": False}, id="sender")
    receiver = DynamicalCell(DynamicalModel(receive_rule), {"received": ()}, id="receiver")
    population = Population((sender, receiver), context_capacity=4)

    first = population.advance(dt=1.0)
    assert first.results["sender"].effects and population.pending("receiver")
    population.suspend("receiver", reason="wait for scheduled wakeup")
    population.advance(dt=1.0)
    assert receiver.state["received"] == ()
    assert len(population.pending("receiver")) == 1

    population.resume("receiver")
    population.advance(dt=1.0)
    received = receiver.state["received"]
    assert received == ({"signal": "pulse", "dose": 1.0},)
    print("routed signals:", received)
    print("cell times:", sender.time, receiver.time)


if __name__ == "__main__":
    main()
