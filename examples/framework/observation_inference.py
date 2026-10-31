"""Compare hidden dynamics using a caller-specified observation likelihood.

The state, observation and score are invented examples. Returned weights are
compatibility weights under this score, not biological evidence.
"""
from cellerator.inference import Observation, Particle, particle_filter
from cellerator.modeling import DynamicalModel


def main() -> None:
    # Rate is an unknown candidate parameter encoded in the supplied state.
    # The same update law propagates all candidate realizations.
    model = DynamicalModel(
        dynamics=lambda state, context: {
            "measurement": state["measurement"] + state["rate"] * context.dt,
            "rate": state["rate"],
        },
        observation=lambda state, _: state["measurement"],
    )
    candidates = (
        Particle({"measurement": 0.0, "rate": rate})
        for rate in (0.1, 0.9, 1.8)
    )
    observed = (Observation(time=1.0, value=0.95, provenance="toy-readout"),)
    posterior = particle_filter(
        model,
        candidates,
        observed,
        log_likelihood=lambda predicted, value: -0.5 * ((predicted - value) / 0.2) ** 2,
        copy_state=lambda state: dict(state),
        resample_below=None,
    )

    best = max(zip(posterior.particles, posterior.weights), key=lambda pair: pair[1])
    assert best[0].state["rate"] == 0.9
    assert best[1] > 0.9
    print("best candidate rate:", best[0].state["rate"])
    print("candidate weights:", tuple(round(weight, 5) for weight in posterior.weights))


if __name__ == "__main__":
    main()
