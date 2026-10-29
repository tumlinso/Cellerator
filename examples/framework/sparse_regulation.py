"""Execute one state-gated sparse regulatory update.

This invented two-gene system is a computational usage example, not a
biological claim. Only the changed regulator selects candidate interactions.
"""
from cellerator.interactions import Interaction, InteractionGraph


def main() -> None:
    graph = InteractionGraph(
        [
            Interaction.directed(
                "STAT1-to-IFIT1",
                "STAT1",
                "IFIT1_activity",
                lambda values, context: values[0] * context["gain"],
                active_if=lambda state, _: state["competent"],
            ),
            Interaction.directed(
                "NFkB-to-IL1B",
                "NFKB",
                "IL1B_activity",
                lambda values, context: values[0],
                active_if=lambda state, _: state["competent"],
            ),
        ]
    )
    state = {
        "STAT1": 0.8,
        "NFKB": 0.3,
        "IFIT1_activity": 0.0,
        "IL1B_activity": 0.0,
        "competent": True,
    }
    result = graph.evaluate(state, changed=("STAT1",), context={"gain": 2.0})
    assert result.evaluated == ("STAT1-to-IFIT1",)
    assert result.state["IFIT1_activity"] == 1.6
    assert result.state["IL1B_activity"] == 0.0
    print("evaluated:", result.evaluated)
    print("IFIT1 activity:", result.state["IFIT1_activity"])


if __name__ == "__main__":
    main()
