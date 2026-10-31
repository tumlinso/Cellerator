"""Simulate a stochastic two-channel process and record its event trajectory.

Rates and events are invented for the example; they do not encode a biological
reaction mechanism.
"""
from random import Random

from cellerator.dynamics import JumpChannel, gillespie
from cellerator.events import Event, EventHistory, Trajectory
from cellerator.modeling import DynamicalCell, DynamicalModel


def main() -> None:
    channels = (
        JumpChannel(
            "transcript_birth",
            rate=lambda state, _: state["substrate"] * 0.9,
            update=lambda state, _: {
                **state,
                "substrate": state["substrate"] - 1,
                "transcripts": state["transcripts"] + 1,
            },
        ),
        JumpChannel(
            "transcript_decay",
            rate=lambda state, _: state["transcripts"] * 0.2,
            update=lambda state, _: {
                **state,
                "transcripts": state["transcripts"] - 1,
            },
        ),
    )
    cell = DynamicalCell(
        DynamicalModel(gillespie(channels)),
        {"substrate": 10, "transcripts": 0},
        id="toy-cell",
    )
    history = EventHistory()
    trajectory = Trajectory(clone_policy="deepcopy")
    trajectory.record(cell.time, cell.state)

    result = cell.advance(dt=2.0, rng=Random(7))
    for occurrence in result.events:
        event = Event(
            occurrence.channel,
            time=occurrence.time,
            target=cell.id,
            source="toy jump process",
        )
        history.append(event)

    assert result.time == 2.0
    assert len(history) == result.metadata["jump_count"]
    # This integrator returns event times but not per-event state snapshots.
    # Record only states actually observed rather than interpolating them.
    trajectory.record(result.time, result.state, tuple(history))
    assert trajectory.final.state == cell.state
    print("final state:", cell.state)
    print("firing events:", len(history))
    print("first events:", tuple(event.kind for event in list(history)[:5]))


if __name__ == "__main__":
    main()
