# Cellerator modeling core

`cellerator.modeling` supplies a small Python interface for stateful models.
It defines the relationship among a model, a persistent instance of that model,
and the context used for each state transition. State values remain opaque:
the model may operate on mappings, sparse/ragged structures, arrays, tensors,
domain objects, or externally owned state references.

```python
from cellerator.modeling import Cell, Model, StepResult


def update(state, context):
    next_state = state.advance(context.inputs, context.parameters, context.dt)
    return StepResult(next_state, events=state.events_since(next_state))


model = Model(update, observation=lambda state, context: state.readout(),
              parameters=shared_parameters)
cell = Cell(model, initial_state, id="founder")
result = cell.advance(dt=0.1, inputs=signal, rng=random_generator)
measurement = cell.observe()
```

Use an `Experiment` to advance a cell through requested observation times and
retain its state and event history:

```python
from cellerator.modeling import Experiment

result = Experiment(
    cell,
    times=(1.0, 2.0, 4.0),
    input_schedule={0.0: baseline, 2.0: perturbation},
    max_step=0.1,
).run()
```

Scheduled inputs use zero order hold and each scheduled time clips the update.
A callable schedule receives `(current_time, current_state)` before each
update. `result.trajectory` stores cloned snapshots, `result.observations` aligns
with requested times, `result.events` preserves emitted event objects, and
`result.final_state` references the final live cell state. The experiment
advances the supplied `Cell` in place.

`Model.parameters` are shared by cells using that model; each `Cell` retains its
own state and time. Cells do not clone, densify, or interpret their initial
state. Supply distinct state objects for independent cells. For state held by
another owner, `Cell.from_reference(model, reference)` reads and writes through
the reference's `get()` and `set()` methods.

`Model.context()` supplies shared model parameters unless the caller overrides
them; pass `parameters=None` to deliberately clear them. `Cell.advance(...,
until=t)` checks a model-supplied time against the interval from the current cell
time through `t` before adopting its result. Unbounded advances may follow a
model-defined time convention, including reverse-time updates.

Dynamics are ordinary callables receiving `(state, StepContext)`. They may
return a new state directly or a `StepResult` carrying emitted events, effects,
time, and metadata. `sequential_updates` applies component updates in declared
order to one input state using the same context; callers choose that ordering
and its numerical interpretation. With no observation rule, `Cell.observe()`
returns the modeled state.

The core does not prescribe a transition law or measurement ontology. It is a
starting interface for combining learned, discrete-event, continuous, stochastic,
and structured numerical components; those components remain responsible for
their own assumptions and numerical validity.
