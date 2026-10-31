"""Run one cellular model through input changes and observe its trajectory."""
from cellerator.events import Event
from cellerator.modeling import DynamicalCell, Experiment, DynamicalModel, StepResult


def update(state, context):
    expression = state["expression"] + (
        context.inputs["production"]
        - context.parameters["degradation"] * state["expression"]
    ) * context.dt
    event = Event("transcription", time=context.time + context.dt,
                  target="gene-a", payload=expression)
    return StepResult({"expression": expression}, events=(event,))


model = DynamicalModel(
    update,
    observation=lambda state, context: {"RNA": state["expression"]},
    parameters={"degradation": 0.02},
)
cell = DynamicalCell(model, {"expression": 0.0}, id="founder")
result = Experiment(
    cell,
    times=(0.5, 1.0, 2.0),
    input_schedule={0.0: {"production": 1.0}, 1.0: {"production": 0.25}},
    max_step=0.25,
).run()

print("observations:", result.observations)
print("events:", len(result.events))
print("final time:", result.final_time)
